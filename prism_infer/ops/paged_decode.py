"""Paged decode attention Triton kernel.

本模块只实现 decode 阶段单 query attention:

- q: [batch, num_heads, head_dim]
- k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
- block_tables: [batch, max_blocks_per_seq]
- context_lens: [batch]

eager fallback 仍保留在 `Attention._forward_decode_eager`，作为 correctness
reference。这里不做 unsupported shape 的静默降级；调用方如果要 fallback，
必须显式选择 fallback 路径。
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on CPU-only envs
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_attention_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        out_ptr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_block: tl.constexpr,
        k_stride_token: tl.constexpr,
        k_stride_head: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_block: tl.constexpr,
        v_stride_token: tl.constexpr,
        v_stride_head: tl.constexpr,
        v_stride_d: tl.constexpr,
        block_tables_stride_b: tl.constexpr,
        block_tables_stride_n: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        scale: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PAGE_BLOCK_SIZE: tl.constexpr,
        KV_GROUPS: tl.constexpr,
        MAX_CONTEXT_LEN: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // KV_GROUPS

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr + seq_idx * q_stride_b + q_head * q_stride_h + offs_d * q_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        context_len = tl.load(context_lens_ptr + seq_idx)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for start in tl.range(0, MAX_CONTEXT_LEN, BLOCK_N):
            token_idx = start + offs_n
            valid_n = token_idx < context_len
            page_idx = token_idx // PAGE_BLOCK_SIZE
            page_offset = token_idx - page_idx * PAGE_BLOCK_SIZE
            block_id = tl.load(
                block_tables_ptr
                + seq_idx * block_tables_stride_b
                + page_idx * block_tables_stride_n,
                mask=valid_n,
                other=-1,
            )
            valid = valid_n & (block_id >= 0)

            k = tl.load(
                k_cache_ptr
                + block_id[:, None] * k_stride_block
                + page_offset[:, None] * k_stride_token
                + kv_head * k_stride_head
                + offs_d[None, :] * k_stride_d,
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))

            block_m = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_m)
            m_new_for_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_new_for_exp))
            p = tl.exp(scores - m_new_for_exp)
            p = tl.where(valid, p, 0.0)

            v = tl.load(
                v_cache_ptr
                + block_id[:, None] * v_stride_block
                + page_offset[:, None] * v_stride_token
                + kv_head * v_stride_head
                + offs_d[None, :] * v_stride_d,
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        denom = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / denom
        out = tl.where(l_i > 0.0, out, 0.0)
        tl.store(
            out_ptr + seq_idx * out_stride_b + q_head * out_stride_h + offs_d * out_stride_d,
            out,
            mask=d_mask,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    *,
    block_n: int = 32,
) -> torch.Tensor:
    """执行自实现 Triton paged decode attention。

    返回 shape 与 `q` 一致: `[batch, num_heads, head_dim]`。
    """

    if not HAS_TRITON:
        raise RuntimeError("Triton is required for paged_decode_attention")
    if not q.is_cuda:
        raise RuntimeError("paged_decode_attention requires CUDA tensors")
    if q.ndim != 3:
        raise ValueError(f"q must be [batch, num_heads, head_dim], got {list(q.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache/v_cache must be [num_blocks, block_size, num_kv_heads, head_dim], "
            f"got {list(k_cache.shape)} and {list(v_cache.shape)}"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"k_cache and v_cache shapes differ: {list(k_cache.shape)} vs {list(v_cache.shape)}")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be [batch, max_blocks], got {list(block_tables.shape)}")
    if context_lens.ndim != 1:
        raise ValueError(f"context_lens must be [batch], got {list(context_lens.shape)}")

    batch, num_heads, head_dim = q.shape
    _, page_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim}, cache={cache_head_dim}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads must be divisible by num_kv_heads, got {num_heads} and {num_kv_heads}")
    if block_tables.shape[0] < batch or context_lens.shape[0] < batch:
        raise ValueError(
            "block_tables/context_lens batch must cover q batch, "
            f"got q={batch}, block_tables={block_tables.shape[0]}, context_lens={context_lens.shape[0]}"
        )

    block_d = _next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError(f"paged_decode_attention supports head_dim <= 256, got {head_dim}")

    output = torch.empty_like(q)
    max_context_len = block_tables.shape[1] * page_block_size
    kv_groups = num_heads // num_kv_heads
    grid = (batch, num_heads)
    _paged_decode_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        float(scale),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        PAGE_BLOCK_SIZE=page_block_size,
        KV_GROUPS=kv_groups,
        MAX_CONTEXT_LEN=max_context_len,
        num_warps=4,
    )
    return output
