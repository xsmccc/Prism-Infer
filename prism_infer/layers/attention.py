import torch
import torch.nn.functional as F
from torch import nn

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


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """将当前 K/V 写入 KV Cache (GPU→Triton, CPU→PyTorch fallback)"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    if HAS_TRITON and key.is_cuda:
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        _store_kvcache_triton[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D)
    else:
        for i in range(N):
            slot = slot_mapping[i].item()
            if slot == -1:
                continue
            k_cache[slot] = key[i].to(k_cache.dtype)
            v_cache[slot] = value[i].to(v_cache.dtype)


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
                k, v = k_cache, v_cache
            if HAS_FLASH_ATTN:
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True,
                    block_table=context.block_tables)
            else:
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale)
        else:
            if HAS_FLASH_ATTN and q.is_cuda:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale, causal=True)
            else:
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale)
        return o
