"""Specialized two-stage BF16 top-k for selective greedy decoding."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_SELECTIVE_TOPK_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    HAS_SELECTIVE_TOPK_TRITON = False


SELECTIVE_TOPK_TILE_SIZE = 1024
SELECTIVE_TOPK_NUM_WARPS = 8
MAX_SELECTIVE_TOPK_BATCH = 4


if HAS_SELECTIVE_TOPK_TRITON:

    @triton.jit
    def _selective_topk_tiles_kernel(
        logits_ptr,
        candidate_values_ptr,
        candidate_ids_ptr,
        VOCAB_SIZE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        TOP_K: tl.constexpr,
        TILE_COUNT: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        local_ids = tl.arange(0, TILE_SIZE)
        global_ids = tile_id * TILE_SIZE + local_ids
        logits_offsets = batch_id * VOCAB_SIZE + global_ids
        values = tl.load(
            logits_ptr + logits_offsets,
            mask=global_ids < VOCAB_SIZE,
            other=-float("inf"),
        )

        # BF16's sign/magnitude bit pattern becomes monotonically ordered after
        # this standard float-key transform.  Packing a local id into the low
        # bits makes every key unique, so Triton's bitonic top-k also returns
        # the corresponding vocabulary ids without a second search.
        bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
        sign = bits & 0x8000
        ordered = tl.where(sign != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
        keys = (ordered << 16) | local_ids.to(tl.uint32)
        top_keys = tl.topk(keys, TOP_K)
        selected_local_ids = top_keys & 0xFFFF
        selected_global_ids = tile_id * TILE_SIZE + selected_local_ids

        output_ids = (
            batch_id * TILE_COUNT * TOP_K
            + tile_id * TOP_K
            + tl.arange(0, TOP_K)
        )
        selected_values = tl.load(
            logits_ptr + batch_id * VOCAB_SIZE + selected_global_ids,
            mask=selected_global_ids < VOCAB_SIZE,
            other=-float("inf"),
        )
        tl.store(candidate_values_ptr + output_ids, selected_values)
        tl.store(candidate_ids_ptr + output_ids, selected_global_ids)

    @triton.jit
    def _selective_topk_merge_kernel(
        candidate_values_ptr,
        candidate_ids_ptr,
        output_ids_ptr,
        CANDIDATE_COUNT: tl.constexpr,
        MERGE_SIZE: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        local_ids = tl.arange(0, MERGE_SIZE)
        candidate_offsets = batch_id * CANDIDATE_COUNT + local_ids
        values = tl.load(
            candidate_values_ptr + candidate_offsets,
            mask=local_ids < CANDIDATE_COUNT,
            other=-float("inf"),
        )
        bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
        sign = bits & 0x8000
        ordered = tl.where(sign != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
        keys = (ordered << 16) | local_ids.to(tl.uint32)
        top_keys = tl.topk(keys, TOP_K)
        selected_candidate_ids = top_keys & 0xFFFF
        selected_global_ids = tl.load(
            candidate_ids_ptr
            + batch_id * CANDIDATE_COUNT
            + selected_candidate_ids
        )
        tl.store(
            output_ids_ptr + batch_id * TOP_K + tl.arange(0, TOP_K),
            selected_global_ids.to(tl.int64),
        )


def selective_topk_indices(logits: torch.Tensor, *, k: int) -> torch.Tensor:
    """Return the top-k BF16 vocabulary ids using two CUDA Graph-safe kernels."""

    if not HAS_SELECTIVE_TOPK_TRITON:
        raise RuntimeError("selective top-k requires Triton")
    if logits.ndim != 2:
        raise ValueError("selective top-k expects rank-2 logits")
    if not logits.is_cuda or logits.dtype != torch.bfloat16:
        raise ValueError("selective top-k requires CUDA BF16 logits")
    if not logits.is_contiguous():
        raise ValueError("selective top-k requires contiguous logits")
    if not 1 <= logits.shape[0] <= MAX_SELECTIVE_TOPK_BATCH:
        raise ValueError(
            "selective top-k supports batch sizes 1 through "
            f"{MAX_SELECTIVE_TOPK_BATCH}, got {logits.shape[0]}"
        )
    if k <= 0 or k > SELECTIVE_TOPK_TILE_SIZE or k & (k - 1):
        raise ValueError(f"k must be a power of two in [1, 1024], got {k}")

    batch_size, vocab_size = logits.shape
    tile_count = triton.cdiv(vocab_size, SELECTIVE_TOPK_TILE_SIZE)
    candidate_count = tile_count * k
    merge_size = triton.next_power_of_2(candidate_count)
    if merge_size > 65536:
        raise ValueError("selective top-k candidate merge exceeds the packed id range")

    candidate_values = torch.empty(
        (batch_size, candidate_count),
        dtype=logits.dtype,
        device=logits.device,
    )
    candidate_ids = torch.empty(
        (batch_size, candidate_count),
        dtype=torch.int32,
        device=logits.device,
    )
    output_ids = torch.empty(
        (batch_size, k),
        dtype=torch.int64,
        device=logits.device,
    )
    _selective_topk_tiles_kernel[(batch_size, tile_count)](
        logits,
        candidate_values,
        candidate_ids,
        VOCAB_SIZE=vocab_size,
        TILE_SIZE=SELECTIVE_TOPK_TILE_SIZE,
        TOP_K=k,
        TILE_COUNT=tile_count,
        num_warps=SELECTIVE_TOPK_NUM_WARPS,
    )
    _selective_topk_merge_kernel[(batch_size,)](
        candidate_values,
        candidate_ids,
        output_ids,
        CANDIDATE_COUNT=candidate_count,
        MERGE_SIZE=merge_size,
        TOP_K=k,
        num_warps=SELECTIVE_TOPK_NUM_WARPS,
    )
    return output_ids


__all__ = ["HAS_SELECTIVE_TOPK_TRITON", "selective_topk_indices"]
