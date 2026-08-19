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


QUERY_TENSOR_RANK = 3
PAGED_KV_CACHE_RANK = 4
BLOCK_TABLE_TENSOR_RANK = 2
CONTEXT_LENGTH_VECTOR_RANK = 1
DEFAULT_PAGED_DECODE_BLOCK_N = 32
TRITON_MAX_PAGED_DECODE_HEAD_DIM = 256
TRITON_PAGED_DECODE_NUM_WARPS = 4


if HAS_TRITON:

    @triton.jit
    def _paged_decode_attention_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_cache_ptr,
        v_scale_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        max_context_len_ptr,
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
        k_scale_stride_block: tl.constexpr,
        k_scale_stride_token: tl.constexpr,
        k_scale_stride_head: tl.constexpr,
        v_scale_stride_block: tl.constexpr,
        v_scale_stride_token: tl.constexpr,
        v_scale_stride_head: tl.constexpr,
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
        MAX_CONTEXT_CAPACITY: tl.constexpr,
        SCALED_KV: tl.constexpr,
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
        # Runtime upper bound keeps CUDA Graph replay independent of the model's
        # configured maximum length.  The constexpr capacity remains a hard
        # guard against malformed metadata indexing beyond the block table.
        max_context_len = tl.load(max_context_len_ptr)
        max_context_len = tl.minimum(max_context_len, MAX_CONTEXT_CAPACITY)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for start in tl.range(0, max_context_len, BLOCK_N):
            token_idx = start + offs_n
            valid_n = (token_idx < context_len) & (token_idx < MAX_CONTEXT_CAPACITY)
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
            if SCALED_KV:
                k_scale = tl.load(
                    k_scale_cache_ptr
                    + block_id * k_scale_stride_block
                    + page_offset * k_scale_stride_token
                    + kv_head * k_scale_stride_head,
                    mask=valid,
                    other=1.0,
                ).to(tl.float32)
                k *= k_scale[:, None]
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
            if SCALED_KV:
                v_scale = tl.load(
                    v_scale_cache_ptr
                    + block_id * v_scale_stride_block
                    + page_offset * v_scale_stride_token
                    + kv_head * v_scale_stride_head,
                    mask=valid,
                    other=1.0,
                ).to(tl.float32)
                v *= v_scale[:, None]
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


def _validate_paged_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    block_n: int = DEFAULT_PAGED_DECODE_BLOCK_N,
) -> bool:
    """Validate the public paged-decode contract outside compiler regions."""

    _validate_paged_decode_runtime(q)
    scaled_kv = _validate_paged_scale_pair(k_scale_cache, v_scale_cache)
    _validate_paged_cuda_tensors(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        k_scale_cache,
        v_scale_cache,
    )
    _validate_paged_query_cache_contract(q, k_cache, v_cache)
    _validate_paged_cache_dtypes(q, k_cache)
    if scaled_kv:
        _validate_paged_scale_cache(k_cache, k_scale_cache, v_scale_cache)
    _validate_paged_metadata(block_tables, context_lens, max_context_len)
    _validate_paged_geometry(q, k_cache, block_tables, context_lens, block_n)
    return scaled_kv


def _validate_paged_decode_runtime(q: torch.Tensor) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for paged_decode_attention")
    if not q.is_cuda:
        raise RuntimeError("paged_decode_attention requires CUDA tensors")


def _validate_paged_scale_pair(
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> bool:
    if (k_scale_cache is None) != (v_scale_cache is None):
        raise ValueError("K/V scale caches must be provided together")
    return k_scale_cache is not None


def _validate_paged_cuda_tensors(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: torch.Tensor,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> None:
    tensors = (
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        *(tensor for tensor in (k_scale_cache, v_scale_cache) if tensor is not None),
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("paged_decode_attention requires all tensors on CUDA")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise RuntimeError("paged_decode_attention requires all tensors on one device")


def _validate_paged_query_cache_contract(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    if q.ndim != QUERY_TENSOR_RANK:
        raise ValueError(f"q must be [batch, num_heads, head_dim], got {list(q.shape)}")
    if k_cache.ndim != PAGED_KV_CACHE_RANK or v_cache.ndim != PAGED_KV_CACHE_RANK:
        raise ValueError(
            "k_cache/v_cache must be [num_blocks, block_size, num_kv_heads, head_dim], "
            f"got {list(k_cache.shape)} and {list(v_cache.shape)}"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache shapes differ: {list(k_cache.shape)} vs {list(v_cache.shape)}"
        )
    if k_cache.dtype != v_cache.dtype:
        raise ValueError(f"k_cache/v_cache dtype mismatch: {k_cache.dtype} vs {v_cache.dtype}")


def _validate_paged_cache_dtypes(q: torch.Tensor, k_cache: torch.Tensor) -> None:
    supported_query_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if q.dtype not in supported_query_dtypes:
        raise ValueError(f"unsupported paged decode query dtype: {q.dtype}")
    supported_cache_dtypes = set(supported_query_dtypes)
    if hasattr(torch, "float8_e4m3fn"):
        supported_cache_dtypes.add(torch.float8_e4m3fn)
    if k_cache.dtype not in supported_cache_dtypes:
        raise ValueError(f"unsupported paged decode cache dtype: {k_cache.dtype}")


def _validate_paged_scale_cache(
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or k_cache.dtype != fp8_dtype:
        raise ValueError("token-head scales require FP8 E4M3FN K/V payload caches")
    expected_scale_shape = tuple(k_cache.shape[:-1])
    if (
        k_scale_cache.shape != v_scale_cache.shape
        or tuple(k_scale_cache.shape) != expected_scale_shape
    ):
        raise ValueError(
            "K/V scale cache shape must equal payload shape without head_dim: "
            f"expected={list(expected_scale_shape)}"
        )
    if k_scale_cache.dtype != torch.float32 or v_scale_cache.dtype != torch.float32:
        raise ValueError("K/V scale caches must use torch.float32")


def _validate_paged_metadata(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: torch.Tensor,
) -> None:
    if block_tables.ndim != BLOCK_TABLE_TENSOR_RANK:
        raise ValueError(
            f"block_tables must be [batch, max_blocks], got {list(block_tables.shape)}"
        )
    if context_lens.ndim != CONTEXT_LENGTH_VECTOR_RANK:
        raise ValueError(f"context_lens must be [batch], got {list(context_lens.shape)}")
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise ValueError(
            "block_tables/context_lens must use torch.int32, got "
            f"{block_tables.dtype}/{context_lens.dtype}"
        )
    if max_context_len.numel() != 1 or max_context_len.dtype != torch.int32:
        raise ValueError(
            "max_context_len must be one torch.int32 scalar tensor, got "
            f"shape={list(max_context_len.shape)} dtype={max_context_len.dtype}"
        )


def _validate_paged_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_n: int,
) -> None:
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a positive power of two, got {block_n}")

    batch, num_heads, head_dim = q.shape
    _, page_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if page_block_size <= 0:
        raise ValueError("paged decode cache block size must be positive")
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim}, cache={cache_head_dim}")
    if num_kv_heads <= 0:
        raise ValueError("paged decode cache must contain at least one KV head")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads, got {num_heads} and {num_kv_heads}"
        )
    if block_tables.shape[0] < batch or context_lens.shape[0] < batch:
        raise ValueError(
            "block_tables/context_lens batch must cover q batch, "
            f"got q={batch}, block_tables={block_tables.shape[0]}, "
            f"context_lens={context_lens.shape[0]}"
        )

    block_d = _next_power_of_2(head_dim)
    if block_d > TRITON_MAX_PAGED_DECODE_HEAD_DIM:
        raise ValueError(
            "paged_decode_attention supports head_dim <= "
            f"{TRITON_MAX_PAGED_DECODE_HEAD_DIM}, got {head_dim}"
        )




if HAS_TRITON:

    @triton.jit
    def _paged_decode_attention_split_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_cache_ptr,
        v_scale_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        max_context_len_ptr,
        partials_ptr,
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
        k_scale_stride_block: tl.constexpr,
        k_scale_stride_token: tl.constexpr,
        k_scale_stride_head: tl.constexpr,
        v_scale_stride_block: tl.constexpr,
        v_scale_stride_token: tl.constexpr,
        v_scale_stride_head: tl.constexpr,
        block_tables_stride_b: tl.constexpr,
        block_tables_stride_n: tl.constexpr,
        partials_stride_b: tl.constexpr,
        partials_stride_h: tl.constexpr,
        partials_stride_s: tl.constexpr,
        scale: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PAGE_BLOCK_SIZE: tl.constexpr,
        KV_GROUPS: tl.constexpr,
        MAX_CONTEXT_CAPACITY: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_CHUNK: tl.constexpr,
        SCALED_KV: tl.constexpr,
    ):
        """One context split: partial online-softmax (acc, l, m) for one head."""

        seq_idx = tl.program_id(0)
        q_head = tl.program_id(1)
        split = tl.program_id(2)
        kv_head = q_head // KV_GROUPS

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr + seq_idx * q_stride_b + q_head * q_stride_h + offs_d * q_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        context_len = tl.load(context_lens_ptr + seq_idx)
        max_context_len = tl.load(max_context_len_ptr)
        max_context_len = tl.minimum(max_context_len, MAX_CONTEXT_CAPACITY)
        split_start = split * SPLIT_CHUNK
        split_end = tl.minimum(split_start + SPLIT_CHUNK, max_context_len)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for start in tl.range(split_start, split_end, BLOCK_N):
            token_idx = start + offs_n
            valid_n = (token_idx < context_len) & (token_idx < split_end) & (token_idx < MAX_CONTEXT_CAPACITY)
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
            if SCALED_KV:
                k_scale = tl.load(
                    k_scale_cache_ptr
                    + block_id * k_scale_stride_block
                    + page_offset * k_scale_stride_token
                    + kv_head * k_scale_stride_head,
                    mask=valid,
                    other=1.0,
                ).to(tl.float32)
                k *= k_scale[:, None]
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
            if SCALED_KV:
                v_scale = tl.load(
                    v_scale_cache_ptr
                    + block_id * v_scale_stride_block
                    + page_offset * v_scale_stride_token
                    + kv_head * v_scale_stride_head,
                    mask=valid,
                    other=1.0,
                ).to(tl.float32)
                v *= v_scale[:, None]
            acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        partials_base = (
            partials_ptr
            + seq_idx * partials_stride_b
            + q_head * partials_stride_h
            + split * partials_stride_s
        )
        tl.store(partials_base + offs_d, acc, mask=d_mask)
        tl.store(partials_base + BLOCK_D, l_i)
        tl.store(partials_base + BLOCK_D + 1, m_i)


    @triton.jit
    def _paged_decode_attention_reduce_kernel(
        partials_ptr,
        out_ptr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        partials_stride_b: tl.constexpr,
        partials_stride_h: tl.constexpr,
        partials_stride_s: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Reduce split partials into the final attention output."""

        seq_idx = tl.program_id(0)
        q_head = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        m_global = tl.full((), -float("inf"), tl.float32)
        for split in tl.static_range(NUM_SPLITS):
            l_s = tl.load(
                partials_ptr
                + seq_idx * partials_stride_b
                + q_head * partials_stride_h
                + split * partials_stride_s
                + BLOCK_D
            )
            m_s = tl.load(
                partials_ptr
                + seq_idx * partials_stride_b
                + q_head * partials_stride_h
                + split * partials_stride_s
                + BLOCK_D
                + 1
            )
            m_global = tl.where((l_s > 0.0) & (m_s > m_global), m_s, m_global)

        acc = tl.zeros((BLOCK_D,), tl.float32)
        l_total = tl.zeros((), tl.float32)
        for split in tl.static_range(NUM_SPLITS):
            base = (
                partials_ptr
                + seq_idx * partials_stride_b
                + q_head * partials_stride_h
                + split * partials_stride_s
            )
            m_s = tl.load(base + BLOCK_D + 1)
            l_s = tl.load(base + BLOCK_D)
            w = tl.where(l_s > 0.0, tl.exp(m_s - m_global), 0.0)
            acc_s = tl.load(base + offs_d, mask=d_mask, other=0.0)
            acc += acc_s * w
            l_total += l_s * w

        denom = tl.where(l_total > 0.0, l_total, 1.0)
        out = acc / denom
        out = tl.where(l_total > 0.0, out, 0.0)
        tl.store(
            out_ptr
            + seq_idx * out_stride_b
            + q_head * out_stride_h
            + offs_d * out_stride_d,
            out,
            mask=d_mask,
        )


def _launch_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    max_context_len: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
    scaled_kv: bool,
    block_n: int,
    num_splits: int = 1,
) -> torch.Tensor:
    """Launch the Triton kernel after the caller has validated all inputs.

    ``num_splits > 1`` 走 flash-decoding split-K: 每个 split 独立在线 softmax
    出 (acc, l, m) 部分和, 第二个 kernel 归约。split 范围由 constexpr
    SPLIT_CHUNK 确定 (仅依赖 MAX_CONTEXT_CAPACITY), grid 形状固定,
    CUDA Graph 捕获安全; 上下文较短的 split 空转 (l=0) 由归约跳过。
    """

    batch, num_heads, head_dim = q.shape
    _, page_block_size, num_kv_heads, _ = k_cache.shape
    block_d = _next_power_of_2(head_dim)
    output = torch.empty_like(q)
    max_context_capacity = block_tables.shape[1] * page_block_size
    kv_groups = num_heads // num_kv_heads
    if num_splits <= 1:
        grid = (batch, num_heads)
        _paged_decode_attention_kernel[grid](
            q,
            k_cache,
            v_cache,
            k_scale_cache if scaled_kv else k_cache,
            v_scale_cache if scaled_kv else v_cache,
            block_tables,
            context_lens,
            max_context_len,
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
            k_scale_cache.stride(0) if scaled_kv else 0,
            k_scale_cache.stride(1) if scaled_kv else 0,
            k_scale_cache.stride(2) if scaled_kv else 0,
            v_scale_cache.stride(0) if scaled_kv else 0,
            v_scale_cache.stride(1) if scaled_kv else 0,
            v_scale_cache.stride(2) if scaled_kv else 0,
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
            MAX_CONTEXT_CAPACITY=max_context_capacity,
            SCALED_KV=scaled_kv,
            num_warps=TRITON_PAGED_DECODE_NUM_WARPS,
        )
        return output

    partials_row = block_d + 2
    # 分配顺序固定 (partials 在 output 前), 依赖缓存分配器确定性以兼容 graph replay
    partials = torch.empty(
        (batch, num_heads, num_splits, partials_row),
        dtype=torch.float32,
        device=q.device,
    )
    split_chunk = (max_context_capacity + num_splits - 1) // num_splits
    _paged_decode_attention_split_kernel[(batch, num_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        k_scale_cache if scaled_kv else k_cache,
        v_scale_cache if scaled_kv else v_cache,
        block_tables,
        context_lens,
        max_context_len,
        partials,
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
        k_scale_cache.stride(0) if scaled_kv else 0,
        k_scale_cache.stride(1) if scaled_kv else 0,
        k_scale_cache.stride(2) if scaled_kv else 0,
        v_scale_cache.stride(0) if scaled_kv else 0,
        v_scale_cache.stride(1) if scaled_kv else 0,
        v_scale_cache.stride(2) if scaled_kv else 0,
        block_tables.stride(0),
        block_tables.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        float(scale),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        PAGE_BLOCK_SIZE=page_block_size,
        KV_GROUPS=kv_groups,
        MAX_CONTEXT_CAPACITY=max_context_capacity,
        NUM_SPLITS=num_splits,
        SPLIT_CHUNK=split_chunk,
        SCALED_KV=scaled_kv,
        num_warps=TRITON_PAGED_DECODE_NUM_WARPS,
    )
    _paged_decode_attention_reduce_kernel[(batch, num_heads)](
        partials,
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        NUM_SPLITS=num_splits,
        num_warps=TRITON_PAGED_DECODE_NUM_WARPS,
    )
    return output

def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    *,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    max_context_len: torch.Tensor | None = None,
    block_n: int = DEFAULT_PAGED_DECODE_BLOCK_N,
    num_splits: int = 1,
) -> torch.Tensor:
    """Execute Prism's validated Triton paged-decode attention kernel.

    ``num_splits > 1`` 启用 flash-decoding split-K (grid 形状固定,
    CUDA Graph 捕获安全)。"""

    if isinstance(num_splits, bool) or not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError(f"num_splits must be a positive integer, got {num_splits!r}")
    if max_context_len is None:
        # Standalone callers do not own ModelRunner's graph-stable scalar.
        # Derive one on device without synchronizing through Tensor.item().
        max_context_len = context_lens.max()
    scaled_kv = _validate_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        block_n=block_n,
    )
    return _launch_paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        max_context_len,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        scaled_kv=scaled_kv,
        block_n=block_n,
        num_splits=num_splits,
    )


__all__ = [
    "DEFAULT_PAGED_DECODE_BLOCK_N",
    "HAS_TRITON",
    "paged_decode_attention",
]
