import torch
import torch.nn.functional as F
from torch import nn

from prism_infer.ops.paged_decode import (
    HAS_TRITON as HAS_PAGED_DECODE_TRITON,
    paged_decode_attention,
)
from prism_infer.analysis.kv_trace import record_attention_layer
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

    if HAS_TRITON and key.is_cuda:
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
            if HAS_FLASH_ATTN:
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True,
                )
            else:
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale)
        else:
            if HAS_FLASH_ATTN and q.is_cuda and context.block_tables is None:
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
        block_size = k_cache.shape[1]
        for seq_idx in range(q.shape[0]):
            context_len = int(context.context_lens[seq_idx].item())
            block_ids = context.block_tables[seq_idx]
            pieces_k = []
            pieces_v = []
            remaining = context_len
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

            keys = torch.cat(pieces_k, dim=0)
            values = torch.cat(pieces_v, dim=0)
            if self.num_heads != self.num_kv_heads:
                groups = self.num_heads // self.num_kv_heads
                keys = keys.repeat_interleave(groups, dim=1)
                values = values.repeat_interleave(groups, dim=1)

            # q_i: [1, heads, 1, dim], keys/values: [1, heads, context_len, dim]
            q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
            k_i = keys.transpose(0, 1).unsqueeze(0)
            v_i = values.transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=False, scale=self.scale)
            outputs.append(out_i.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)
