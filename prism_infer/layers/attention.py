import torch
import torch.nn.functional as F
from torch import nn

from prism_infer.ops.paged_decode import (
    HAS_TRITON as HAS_PAGED_DECODE_TRITON,
    paged_decode_attention,
)
from prism_infer.analysis.kv_trace import is_trace_enabled, record_attention_layer
from prism_infer.engine.compression import (
    CompressionMetadata,
    ensure_supported_compression_metadata,
    get_visual_pruning_record_for_batch,
)
from prism_infer.engine.visual_pruning import build_retained_context_indices
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

    if HAS_TRITON and key.is_cuda and not _is_fp8_cache_tensor(k_cache):
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
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
                raise RuntimeError(
                    "paged prefix-cache prefill is not supported by the local "
                    "flash_attn_varlen_func signature"
                )
            if HAS_FLASH_ATTN and q.is_cuda:
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
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale)
        else:
            if (
                compression_metadata is not None
                and compression_metadata.visual_pruning_active
            ):
                o = self._forward_decode_visual_prune_eager(
                    q,
                    k_cache,
                    v_cache,
                    context,
                    compression_metadata,
                )
            elif (
                compression_metadata is not None
                and compression_metadata.fp8_kv_active
            ):
                if context.block_tables is None:
                    raise RuntimeError("fp8_kv decode requires paged block_tables")
                o = self._forward_decode_eager(q, k_cache, v_cache, context)
            elif HAS_FLASH_ATTN and q.is_cuda and context.block_tables is None:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    softmax_scale=self.scale, causal=True)
            elif context.block_tables is not None:
                if HAS_PAGED_DECODE_TRITON and q.is_cuda:
                    o = paged_decode_attention(
                        q,
                        k_cache,
                        v_cache,
                        context.block_tables,
                        context.context_lens,
                        self.scale,
                    )
                else:
                    o = self._forward_decode_eager(q, k_cache, v_cache, context)
            else:
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale)
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

    def _gather_paged_kv_indices_for_sequence(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: torch.Tensor,
        retained_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect only retained logical positions from one paged KV sequence.

        retained_indices: sorted logical token positions in the decode context.
        返回: keys/values [retained_len, num_kv_heads, head_dim]
        """

        if not retained_indices:
            raise RuntimeError("cannot gather zero retained KV positions")

        pieces_k = []
        pieces_v = []
        block_size = k_cache.shape[1]
        cursor = 0
        while cursor < len(retained_indices):
            token_index = retained_indices[cursor]
            if token_index < 0:
                raise RuntimeError(f"negative retained token index: {token_index}")
            block_ordinal = token_index // block_size
            if block_ordinal >= block_ids.numel():
                raise RuntimeError(
                    "retained token index outside block table: "
                    f"token_index={token_index}, block_table_len={block_ids.numel()}"
                )
            block_id = int(block_ids[block_ordinal].item())
            if block_id < 0:
                raise RuntimeError(
                    "retained token index maps to padded block: "
                    f"token_index={token_index}, block_ordinal={block_ordinal}"
                )
            start_offset = token_index % block_size
            end_offset = start_offset + 1
            cursor += 1

            while cursor < len(retained_indices):
                next_index = retained_indices[cursor]
                next_block_ordinal = next_index // block_size
                next_offset = next_index % block_size
                if (
                    next_block_ordinal != block_ordinal
                    or next_offset != end_offset
                ):
                    break
                end_offset += 1
                cursor += 1

            pieces_k.append(k_cache[block_id, start_offset:end_offset])
            pieces_v.append(v_cache[block_id, start_offset:end_offset])

        return torch.cat(pieces_k, dim=0), torch.cat(pieces_v, dim=0)

    def _forward_decode_eager(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Decode fallback: 从 paged KV cache 收集历史 token 后做单步 SDPA。

        q: [batch, num_heads, head_dim]
        k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        返回: [batch, num_heads, head_dim]
        """

        if context.block_tables is None or context.context_lens is None:
            return F.scaled_dot_product_attention(
                q, q.new_empty(0), q.new_empty(0), is_causal=True, scale=self.scale)

        outputs = []
        for seq_idx in range(q.shape[0]):
            context_len = int(context.context_lens[seq_idx].item())
            block_ids = context.block_tables[seq_idx]
            keys, values = self._gather_paged_kv_for_sequence(
                k_cache,
                v_cache,
                block_ids,
                context_len,
            )
            keys = self._dequantize_cache_for_attention(keys, q.dtype)
            values = self._dequantize_cache_for_attention(values, q.dtype)
            keys, values = self._expand_gqa_kv(keys, values)

            # q_i: [1, heads, 1, dim], keys/values: [1, heads, context_len, dim]
            q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
            k_i = keys.transpose(0, 1).unsqueeze(0)
            v_i = values.transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=False, scale=self.scale)
            outputs.append(out_i.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)

    def _forward_decode_visual_prune_eager(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
        compression_metadata: CompressionMetadata,
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

        outputs = []
        for seq_idx in range(q.shape[0]):
            context_len = int(context.context_lens[seq_idx].item())
            record = get_visual_pruning_record_for_batch(
                compression_metadata,
                seq_idx,
            )
            retained_indices = build_retained_context_indices(record, context_len)
            if not retained_indices:
                raise RuntimeError(
                    "visual_prune retained zero tokens for decode attention"
                )
            # Directly gather retained context positions from paged KV.
            keys, values = self._gather_paged_kv_indices_for_sequence(
                k_cache,
                v_cache,
                context.block_tables[seq_idx],
                retained_indices,
            )
            keys = self._dequantize_cache_for_attention(keys, q.dtype)
            values = self._dequantize_cache_for_attention(values, q.dtype)
            keys, values = self._expand_gqa_kv(keys, values)

            # q_i: [1, heads, 1, dim], compact KV: [1, heads, retained_len, dim]
            q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
            k_i = keys.transpose(0, 1).unsqueeze(0)
            v_i = values.transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i,
                k_i,
                v_i,
                is_causal=False,
                scale=self.scale,
            )
            outputs.append(out_i.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)
