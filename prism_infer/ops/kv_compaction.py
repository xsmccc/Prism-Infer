"""Paged KV slot compaction primitives.

该模块只负责物理 slot copy，不决定保留哪些 token，也不修改页表。CUDA FP8
不能使用 PyTorch ``index_select/index_copy``，因此使用两阶段 Triton copy：
先 gather 到独立 temporary，再 scatter 到 destination，避免重叠区间覆盖 source。
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environment
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _gather_kv_slots_kernel(
        cache_ptr,
        source_slots_ptr,
        temporary_ptr,
        cache_stride_row: tl.constexpr,
        cache_stride_slot: tl.constexpr,
        temporary_stride_row: tl.constexpr,
        temporary_stride_token: tl.constexpr,
        VECTOR_SIZE: tl.constexpr,
        BLOCK_VECTOR: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_VECTOR)
        mask = offsets < VECTOR_SIZE
        source_slot = tl.load(source_slots_ptr + token)
        values = tl.load(
            cache_ptr
            + row * cache_stride_row
            + source_slot * cache_stride_slot
            + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            temporary_ptr
            + row * temporary_stride_row
            + token * temporary_stride_token
            + offsets,
            values,
            mask=mask,
        )

    @triton.jit
    def _scatter_kv_slots_kernel(
        temporary_ptr,
        destination_slots_ptr,
        cache_ptr,
        temporary_stride_row: tl.constexpr,
        temporary_stride_token: tl.constexpr,
        cache_stride_row: tl.constexpr,
        cache_stride_slot: tl.constexpr,
        VECTOR_SIZE: tl.constexpr,
        BLOCK_VECTOR: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_VECTOR)
        mask = offsets < VECTOR_SIZE
        destination_slot = tl.load(destination_slots_ptr + token)
        values = tl.load(
            temporary_ptr
            + row * temporary_stride_row
            + token * temporary_stride_token
            + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            cache_ptr
            + row * cache_stride_row
            + destination_slot * cache_stride_slot
            + offsets,
            values,
            mask=mask,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def compact_kv_slots(
    cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """把 paged cache retained slots 安全移动到 destination slots。

    cache: payload [kv, layers, slots, kv_heads, head_dim] or token-head
        scales [kv, layers, slots, kv_heads]
    source_slots/destination_slots: [retained_tokens]
    """

    if cache.ndim not in (4, 5):
        raise ValueError(
            "cache must be [kv, layers, slots, kv_heads] or "
            "[kv, layers, slots, kv_heads, head_dim], "
            f"got {list(cache.shape)}"
        )
    if source_slots.ndim != 1 or destination_slots.ndim != 1:
        raise ValueError("source/destination slots must be 1D")
    if source_slots.numel() == 0:
        raise ValueError("KV compaction requires at least one retained slot")
    if source_slots.shape != destination_slots.shape:
        raise ValueError(
            "source/destination slot shapes must match: "
            f"{list(source_slots.shape)} vs {list(destination_slots.shape)}"
        )
    if source_slots.dtype != torch.long or destination_slots.dtype != torch.long:
        raise ValueError("source/destination slots must use torch.long")
    if source_slots.device != cache.device or destination_slots.device != cache.device:
        raise RuntimeError("cache and slot tensors must share one device")

    if cache.is_cuda and cache.dtype == getattr(torch, "float8_e4m3fn", None):
        if not HAS_TRITON:
            raise RuntimeError("CUDA FP8 KV compaction requires Triton")
        if not cache.is_contiguous():
            raise RuntimeError("CUDA FP8 KV compaction requires contiguous cache")
        rows = cache.shape[0] * cache.shape[1]
        slots = cache.shape[2]
        vector_size = math.prod(cache.shape[3:])
        flat_cache = cache.view(rows, slots, vector_size)
        temporary = torch.empty(
            rows,
            source_slots.numel(),
            vector_size,
            device=cache.device,
            dtype=cache.dtype,
        )
        block_vector = _next_power_of_2(vector_size)
        if block_vector > 65536:
            raise ValueError(
                f"FP8 KV compaction vector is too large: {vector_size}"
            )
        grid = (rows, source_slots.numel())
        _gather_kv_slots_kernel[grid](
            flat_cache,
            source_slots,
            temporary,
            flat_cache.stride(0),
            flat_cache.stride(1),
            temporary.stride(0),
            temporary.stride(1),
            VECTOR_SIZE=vector_size,
            BLOCK_VECTOR=block_vector,
            num_warps=8,
        )
        _scatter_kv_slots_kernel[grid](
            temporary,
            destination_slots,
            flat_cache,
            temporary.stride(0),
            temporary.stride(1),
            flat_cache.stride(0),
            flat_cache.stride(1),
            VECTOR_SIZE=vector_size,
            BLOCK_VECTOR=block_vector,
            num_warps=8,
        )
        return

    # PyTorch path covers CPU reference and CUDA BF16/FP16/FP32 caches.
    retained = cache.index_select(2, source_slots).clone()
    cache.index_copy_(2, destination_slots, retained)
