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
SELECTIVE_RERANK_NUM_WARPS = 8


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

    @triton.jit
    def _selective_fp32_scores_kernel(
        candidate_ids_ptr,
        hidden_ptr,
        weight_ptr,
        scores_ptr,
        HIDDEN_SIZE: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        candidate_offset = tl.program_id(1)
        hidden_offsets = tl.arange(0, BLOCK_SIZE)
        candidate_id = tl.load(
            candidate_ids_ptr + batch_id * TOP_K + candidate_offset
        )
        hidden = tl.load(
            hidden_ptr + batch_id * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_offsets < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + candidate_id * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_offsets < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            scores_ptr + batch_id * TOP_K + candidate_offset,
            tl.sum(hidden * weight, axis=0),
        )

    @triton.jit
    def _selective_fp32_winner_kernel(
        candidate_ids_ptr,
        scores_ptr,
        output_ptr,
        TOP_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        offsets = tl.arange(0, TOP_K)
        scores = tl.load(scores_ptr + batch_id * TOP_K + offsets)
        candidate_ids = tl.load(candidate_ids_ptr + batch_id * TOP_K + offsets)
        winning_score = tl.max(scores, axis=0)
        # torch.argmax over the full vocabulary resolves exact ties toward the
        # lowest token id, independent of the unsorted top-k candidate order.
        tied_ids = tl.where(scores == winning_score, candidate_ids, 0x7FFFFFFF)
        token_id = tl.min(tied_ids, axis=0)
        tl.store(output_ptr + batch_id, token_id)


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


def rerank_greedy_candidates(
    candidate_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Select one token with FP32 dots without materializing candidate embeddings."""

    if not HAS_SELECTIVE_TOPK_TRITON:
        raise RuntimeError("selective FP32 reranking requires Triton")
    if candidate_ids.ndim != 2 or hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("selective FP32 reranking expects rank-2 tensors")
    if candidate_ids.dtype != torch.int64:
        raise ValueError("selective FP32 reranking requires int64 candidate ids")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("selective FP32 reranking requires BF16 hidden states and weight")
    if (
        not candidate_ids.is_cuda
        or candidate_ids.device != hidden_states.device
        or hidden_states.device != weight.device
    ):
        raise ValueError("selective FP32 reranking requires one CUDA device")
    if not candidate_ids.is_contiguous() or not hidden_states.is_contiguous():
        raise ValueError("candidate ids and hidden states must be contiguous")
    if not weight.is_contiguous():
        raise ValueError("selective FP32 reranking requires contiguous model weight")
    batch_size, top_k = candidate_ids.shape
    if hidden_states.shape[0] != batch_size:
        raise ValueError("candidate ids and hidden states must have the same batch size")
    if weight.shape[1] != hidden_states.shape[1]:
        raise ValueError("LM-head weight and hidden size must match")
    if not 1 <= batch_size <= MAX_SELECTIVE_TOPK_BATCH:
        raise ValueError("selective FP32 reranking supports batch sizes 1 through 4")
    if top_k <= 0 or top_k & (top_k - 1):
        raise ValueError("selective FP32 reranking candidate count must be a power of two")
    if top_k > SELECTIVE_TOPK_TILE_SIZE:
        raise ValueError(
            f"selective FP32 reranking supports at most {SELECTIVE_TOPK_TILE_SIZE} candidates"
        )

    scores = torch.empty(
        (batch_size, top_k),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    output_ids = torch.empty(
        (batch_size,),
        dtype=torch.int64,
        device=hidden_states.device,
    )
    _selective_fp32_scores_kernel[(batch_size, top_k)](
        candidate_ids,
        hidden_states,
        weight,
        scores,
        HIDDEN_SIZE=hidden_states.shape[1],
        TOP_K=top_k,
        BLOCK_SIZE=triton.next_power_of_2(hidden_states.shape[1]),
        num_warps=SELECTIVE_RERANK_NUM_WARPS,
    )
    _selective_fp32_winner_kernel[(batch_size,)](
        candidate_ids,
        scores,
        output_ids,
        TOP_K=top_k,
        num_warps=1,
    )
    return output_ids


__all__ = [
    "HAS_SELECTIVE_TOPK_TRITON",
    "rerank_greedy_candidates",
    "selective_topk_indices",
]
