import torch
import torch.nn.functional as F
from torch import nn

from prism_infer.analysis.performance_profile import profile_region
from prism_infer.ops.paged_decode import (
    HAS_TRITON as HAS_PAGED_DECODE_TRITON,
    paged_decode_attention,
)
from prism_infer.analysis.kv_trace import is_trace_enabled, record_attention_layer
from prism_infer.engine.compression import (
    ensure_supported_compression_metadata,
)
from prism_infer.utils.context import get_context

# flash_attn 和 triton 是可选依赖: 有 GPU 时手动安装
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _float8_cache_dtypes() -> tuple[torch.dtype, ...]:
    """当前运行时支持的 FP8 cache dtype 集合。"""

    dtypes = []
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    return tuple(dtypes)


FP8_CACHE_DTYPES = _float8_cache_dtypes()


def _is_fp8_cache_tensor(tensor: torch.Tensor) -> bool:
    """判断 tensor 是否为 FP8 KV cache。"""

    return tensor.dtype in FP8_CACHE_DTYPES


# ============================================================
# store_kvcache — 将当前 K/V 写入 KV Cache
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _store_kvcache_triton(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def _store_kvcache_eager(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """PyTorch fallback: 按 flat slot 写入 canonical paged KV cache。

    key/value: [num_tokens, num_kv_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: [num_tokens]，slot = block_id * block_size + block_offset。
    """

    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache/v_cache shape mismatch: {list(k_cache.shape)} vs {list(v_cache.shape)}"
        )
    if k_cache.ndim not in (3, 4):
        raise ValueError(
            "k_cache/v_cache must be [slots, heads, dim] or "
            f"[num_blocks, block_size, heads, dim], got {list(k_cache.shape)}"
        )

    for i in range(key.shape[0]):
        slot = int(slot_mapping[i].item())
        if slot == -1:
            continue
        if k_cache.ndim == 4:
            block_size = k_cache.shape[1]
            block_id = slot // block_size
            block_offset = slot % block_size
            k_cache[block_id, block_offset] = key[i].to(k_cache.dtype)
            v_cache[block_id, block_offset] = value[i].to(v_cache.dtype)
        else:
            # Legacy flat-cache fallback kept for small unit tests.
            k_cache[slot] = key[i].to(k_cache.dtype)
            v_cache[slot] = value[i].to(v_cache.dtype)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """将当前 K/V 写入 KV Cache (GPU→Triton, CPU→PyTorch fallback)。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    use_triton = HAS_TRITON and key.is_cuda
    if use_triton and (
        not value.is_cuda
        or not k_cache.is_cuda
        or not v_cache.is_cuda
        or not slot_mapping.is_cuda
    ):
        raise RuntimeError("Triton KV store requires all tensors on CUDA")
    if use_triton and len(
        {key.device, value.device, k_cache.device, v_cache.device, slot_mapping.device}
    ) != 1:
        raise RuntimeError("Triton KV store requires all tensors on the same device")
    if use_triton and (
        key.stride(-1) != 1
        or value.stride(-1) != 1
        or key.stride(-2) != head_dim
        or value.stride(-2) != head_dim
    ):
        raise RuntimeError("Triton KV store requires contiguous head/dim inputs")
    if use_triton and (not k_cache.is_contiguous() or not v_cache.is_contiguous()):
        raise RuntimeError("Triton KV store requires contiguous KV caches")

    if use_triton and _is_fp8_cache_tensor(k_cache):
        region_name = "attention.kv_store.triton_fp8"
    elif use_triton:
        region_name = "attention.kv_store.triton"
    else:
        region_name = "attention.kv_store.eager"
    with profile_region(
        region_name,
        metadata={"cache_dtype": str(k_cache.dtype), "tokens": N},
    ):
        if use_triton:
            # tl.store 根据 destination pointer dtype 执行 BF16/FP32 -> FP8 转换。
            _store_kvcache_triton[(N,)](
                key, key.stride(0), value, value.stride(0),
                k_cache, v_cache, slot_mapping, D)
        else:
            _store_kvcache_eager(key, value, k_cache, v_cache, slot_mapping)


# ============================================================
# Attention — 注意力层 (FlashAttention / PyTorch fallback)
# ============================================================
class Attention(nn.Module):

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_idx: int | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: [N, num_heads, head_dim]
        # k: [N, num_kv_heads, head_dim]
        # v: [N, num_kv_heads, head_dim]
        context = get_context()
        compression_metadata = context.compression_metadata
        ensure_supported_compression_metadata(compression_metadata)
        k_cache, v_cache = self.k_cache, self.v_cache

        # 写入 KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                o = self._forward_prefill_paged(
                    q,
                    k_cache,
                    v_cache,
                    context,
                )
            elif HAS_FLASH_ATTN and q.is_cuda:
                with profile_region("attention.prefill.flash_attn_varlen"):
                    o = flash_attn_varlen_func(
                        q, k, v,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        cu_seqlens_k=context.cu_seqlens_k,
                        softmax_scale=self.scale, causal=True,
                        deterministic=True,
                    )
            else:
                with profile_region("attention.prefill.sdpa"):
                    o = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, scale=self.scale)
        else:
            if (
                compression_metadata is not None
                and compression_metadata.visual_pruning_effective
            ):
                o = self._forward_decode_visual_prune_eager(
                    q,
                    k_cache,
                    v_cache,
                    context,
                )
            elif (
                compression_metadata is not None
                and compression_metadata.fp8_kv_active
            ):
                if context.block_tables is None:
                    raise RuntimeError("fp8_kv decode requires paged block_tables")
                if HAS_PAGED_DECODE_TRITON and q.is_cuda:
                    with profile_region("attention.decode.fp8_paged_triton"):
                        o = paged_decode_attention(
                            q,
                            k_cache,
                            v_cache,
                            context.block_tables,
                            context.context_lens,
                            self.scale,
                        )
                else:
                    # CPU/no-Triton path is the explicit correctness reference.
                    o = self._forward_decode_eager(
                        q,
                        k_cache,
                        v_cache,
                        context,
                        profile_prefix="attention.decode.fp8_reference",
                    )
            elif HAS_FLASH_ATTN and q.is_cuda and context.block_tables is None:
                with profile_region("attention.decode.flash_attn_kvcache"):
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(1), k_cache, v_cache,
                        cache_seqlens=context.context_lens,
                        softmax_scale=self.scale, causal=True)
            elif context.block_tables is not None:
                if HAS_PAGED_DECODE_TRITON and q.is_cuda:
                    with profile_region("attention.decode.paged_triton"):
                        o = paged_decode_attention(
                            q,
                            k_cache,
                            v_cache,
                            context.block_tables,
                            context.context_lens,
                            self.scale,
                        )
                else:
                    o = self._forward_decode_eager(
                        q,
                        k_cache,
                        v_cache,
                        context,
                        profile_prefix="attention.decode.bf16_eager",
                    )
            else:
                with profile_region("attention.decode.sdpa"):
                    o = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, scale=self.scale)
        visual_pruning_scorer = context.visual_pruning_scorer
        if context.is_prefill and visual_pruning_scorer is not None:
            if self.layer_idx is None:
                raise RuntimeError("runtime visual scorer requires attention layer_idx")
            visual_pruning_scorer.observe(
                layer_id=self.layer_idx,
                q=q,
                k=k,
                scale=self.scale,
            )
        if is_trace_enabled():
            record_attention_layer(
                layer_id=self.layer_idx,
                q=q,
                k=k,
                v=v,
                output=o,
                k_cache=k_cache,
                v_cache=v_cache,
                context=context,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                scale=self.scale,
            )
        return o

    def _forward_prefill_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Correctness path for chunked/prefix-hit paged prefill.

        Newly computed K/V have already been stored in the paged cache.  Each
        query chunk attends to its full cached history with an explicit
        bottom-right causal mask; ``is_causal=True`` would use an upper-left
        mask when Q and K lengths differ and is therefore incorrect here.
        """

        if (
            context.block_tables is None
            or context.context_lens is None
            or context.cu_seqlens_q is None
        ):
            raise RuntimeError(
                "paged prefill requires block_tables/context_lens/cu_seqlens_q"
            )
        outputs: list[torch.Tensor] = []
        for seq_index in range(context.context_lens.numel()):
            query_start = int(context.cu_seqlens_q[seq_index].item())
            query_end = int(context.cu_seqlens_q[seq_index + 1].item())
            query_len = query_end - query_start
            context_len = int(context.context_lens[seq_index].item())
            if query_len <= 0 or context_len < query_len:
                raise RuntimeError(
                    "invalid paged prefill lengths: "
                    f"query={query_len} context={context_len}"
                )
            prefix_len = context_len - query_len
            with profile_region(
                "attention.prefill.paged_gather",
                metadata={
                    "query_len": query_len,
                    "context_len": context_len,
                },
            ):
                keys, values = self._gather_paged_kv_for_sequence(
                    k_cache,
                    v_cache,
                    context.block_tables[seq_index],
                    context_len,
                )
                keys = self._dequantize_cache_for_attention(keys, q.dtype)
                values = self._dequantize_cache_for_attention(values, q.dtype)
                keys, values = self._expand_gqa_kv(keys, values)

            queries = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
            keys = keys.transpose(0, 1).unsqueeze(0)
            values = values.transpose(0, 1).unsqueeze(0)
            query_positions = torch.arange(
                prefix_len,
                context_len,
                device=q.device,
            )
            key_positions = torch.arange(context_len, device=q.device)
            causal_mask = (
                key_positions.unsqueeze(0)
                <= query_positions.unsqueeze(1)
            ).unsqueeze(0).unsqueeze(0)
            with profile_region("attention.prefill.paged_sdpa"):
                output = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=causal_mask,
                    is_causal=False,
                    scale=self.scale,
                )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def _gather_paged_kv_for_sequence(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: torch.Tensor,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect one sequence's contiguous logical history from paged KV.

        k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_ids: [num_blocks_for_sequence]
        返回: keys/values [context_len, num_kv_heads, head_dim]
        """

        pieces_k = []
        pieces_v = []
        remaining = context_len
        block_size = k_cache.shape[1]
        for block_id in block_ids.tolist():
            if remaining <= 0:
                break
            if block_id < 0:
                break
            take = min(block_size, remaining)
            pieces_k.append(k_cache[block_id, :take])
            pieces_v.append(v_cache[block_id, :take])
            remaining -= take
        if remaining != 0 or not pieces_k:
            raise RuntimeError(
                "invalid decode block table for paged KV fallback: "
                f"context_len={context_len}, remaining={remaining}"
            )
        return torch.cat(pieces_k, dim=0), torch.cat(pieces_v, dim=0)

    def _expand_gqa_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand GQA KV heads to query-head count when needed."""

        if self.num_heads == self.num_kv_heads:
            return keys, values
        groups = self.num_heads // self.num_kv_heads
        return (
            keys.repeat_interleave(groups, dim=1),
            values.repeat_interleave(groups, dim=1),
        )

    @staticmethod
    def _dequantize_cache_for_attention(
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """把低精度 KV cache 转成 attention 可计算 dtype。"""

        if _is_fp8_cache_tensor(tensor):
            return tensor.to(target_dtype)
        return tensor

    def _gather_paged_kv_slots_for_sequence(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        retained_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """通过 flat physical slots 一次收集一个序列的 retained KV。

        retained_slots: [retained_len]，由 ModelRunner 每个 decode step 构造一次。
        返回: keys/values [retained_len, num_kv_heads, head_dim]
        """

        if retained_slots.ndim != 1 or retained_slots.numel() == 0:
            raise RuntimeError(
                "retained slots must be a non-empty 1D tensor, "
                f"got shape={list(retained_slots.shape)}"
            )
        if retained_slots.dtype != torch.long:
            raise RuntimeError(
                f"retained slots must use torch.long, got {retained_slots.dtype}"
            )
        if retained_slots.device != k_cache.device:
            raise RuntimeError(
                "retained slots and KV cache must share a device: "
                f"slots={retained_slots.device}, cache={k_cache.device}"
            )

        # flat cache: [num_blocks * block_size, num_kv_heads, head_dim]
        flat_k = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
        return (
            flat_k.index_select(0, retained_slots),
            flat_v.index_select(0, retained_slots),
        )

    def _forward_decode_eager(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
        *,
        profile_prefix: str,
    ) -> torch.Tensor:
        """Decode fallback: 从 paged KV cache 收集历史 token 后做单步 SDPA。

        q: [batch, num_heads, head_dim]
        k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        返回: [batch, num_heads, head_dim]
        """

        if context.block_tables is None or context.context_lens is None:
            with profile_region(f"{profile_prefix}.sdpa_no_cache"):
                return F.scaled_dot_product_attention(
                    q,
                    q.new_empty(0),
                    q.new_empty(0),
                    is_causal=True,
                    scale=self.scale,
                )

        outputs = []
        for seq_idx in range(q.shape[0]):
            with profile_region(f"{profile_prefix}.context_len", cuda=False):
                context_len = int(context.context_lens[seq_idx].item())
            block_ids = context.block_tables[seq_idx]
            with profile_region(
                f"{profile_prefix}.gather",
                metadata={"context_len": context_len},
            ):
                keys, values = self._gather_paged_kv_for_sequence(
                    k_cache,
                    v_cache,
                    block_ids,
                    context_len,
                )
            with profile_region(f"{profile_prefix}.dequant"):
                keys = self._dequantize_cache_for_attention(keys, q.dtype)
                values = self._dequantize_cache_for_attention(values, q.dtype)
            with profile_region(f"{profile_prefix}.expand_gqa"):
                keys, values = self._expand_gqa_kv(keys, values)

            # q_i: [1, heads, 1, dim], keys/values: [1, heads, context_len, dim]
            q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
            k_i = keys.transpose(0, 1).unsqueeze(0)
            v_i = values.transpose(0, 1).unsqueeze(0)
            with profile_region(f"{profile_prefix}.sdpa"):
                out_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    is_causal=False,
                    scale=self.scale,
                )
            outputs.append(out_i.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)

    def _forward_decode_visual_prune_eager(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Decode with logical visual-token pruning over a compact KV view.

        q: [batch, num_heads, head_dim]
        k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        返回: [batch, num_heads, head_dim]
        """

        if context.block_tables is None or context.context_lens is None:
            raise RuntimeError(
                "visual_prune decode requires paged block_tables and context_lens"
            )
        if not k_cache.numel() or not v_cache.numel():
            raise RuntimeError("visual_prune decode requires populated KV cache")
        slot_mappings = context.visual_pruning_slot_mappings
        if len(slot_mappings) != q.shape[0]:
            raise RuntimeError(
                "visual_prune decode requires one retained slot mapping per batch row: "
                f"mappings={len(slot_mappings)}, batch={q.shape[0]}"
            )

        outputs = []
        for seq_idx in range(q.shape[0]):
            retained_slots = slot_mappings[seq_idx]
            with profile_region(
                "attention.decode.visual_prune.gather",
                metadata={
                    "retained_len": retained_slots.numel(),
                },
            ):
                keys, values = self._gather_paged_kv_slots_for_sequence(
                    k_cache,
                    v_cache,
                    retained_slots,
                )
            with profile_region("attention.decode.visual_prune.dequant"):
                keys = self._dequantize_cache_for_attention(keys, q.dtype)
                values = self._dequantize_cache_for_attention(values, q.dtype)
            with profile_region("attention.decode.visual_prune.expand_gqa"):
                keys, values = self._expand_gqa_kv(keys, values)

            # q_i: [1, heads, 1, dim], compact KV: [1, heads, retained_len, dim]
            q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
            k_i = keys.transpose(0, 1).unsqueeze(0)
            v_i = values.transpose(0, 1).unsqueeze(0)
            with profile_region("attention.decode.visual_prune.sdpa"):
                out_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    is_causal=False,
                    scale=self.scale,
                )
            outputs.append(out_i.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)
