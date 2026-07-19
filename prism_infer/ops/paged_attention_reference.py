"""Explicit PyTorch correctness paths for paged attention.

These routines intentionally favor auditable semantics over speed. Production
CUDA decode uses :mod:`prism_infer.ops.paged_decode`; CPU execution and
unsupported optimized paths use this module as their reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism_infer.engine.kv_quantization import KV_SCALE_DTYPE
from prism_infer.observability import profile_region
from prism_infer.ops.kv_cache_store import is_fp8_cache_tensor
from prism_infer.utils.context import Context


MINIMUM_PAGED_CACHE_RANK = 3
INDEX_VECTOR_RANK = 1


def gather_paged_kv_for_sequence(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_ids: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect one sequence's contiguous logical history from paged storage."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("K/V cache shapes must match for paged gather")
    if k_cache.ndim < MINIMUM_PAGED_CACHE_RANK:
        raise ValueError(
            "paged K/V caches must contain block and page dimensions, "
            f"got shape={list(k_cache.shape)}"
        )
    if block_ids.ndim != INDEX_VECTOR_RANK:
        raise ValueError(f"block_ids must be one-dimensional, got {list(block_ids.shape)}")
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")

    key_pieces: list[torch.Tensor] = []
    value_pieces: list[torch.Tensor] = []
    remaining = context_len
    block_size = k_cache.shape[1]
    for block_id in block_ids.tolist():
        if remaining <= 0 or block_id < 0:
            break
        if block_id >= k_cache.shape[0]:
            raise RuntimeError(
                "paged KV block id exceeds cache capacity: "
                f"block_id={block_id}, num_blocks={k_cache.shape[0]}"
            )
        take = min(block_size, remaining)
        key_pieces.append(k_cache[block_id, :take])
        value_pieces.append(v_cache[block_id, :take])
        remaining -= take
    if remaining != 0 or not key_pieces:
        raise RuntimeError(
            "invalid block table for paged KV reference: "
            f"context_len={context_len}, remaining={remaining}"
        )
    return torch.cat(key_pieces, dim=0), torch.cat(value_pieces, dim=0)


def gather_paged_kv_slots_for_sequence(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    retained_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect retained KV entries from flat physical slots."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("K/V cache shapes must match for retained-slot gather")
    if k_cache.ndim < MINIMUM_PAGED_CACHE_RANK:
        raise ValueError(
            "paged K/V caches must contain block and page dimensions, "
            f"got shape={list(k_cache.shape)}"
        )
    if retained_slots.ndim != INDEX_VECTOR_RANK or retained_slots.numel() == 0:
        raise RuntimeError(
            f"retained slots must be a non-empty 1D tensor, got shape={list(retained_slots.shape)}"
        )
    if retained_slots.dtype != torch.long:
        raise RuntimeError(f"retained slots must use torch.long, got {retained_slots.dtype}")
    if retained_slots.device != k_cache.device:
        raise RuntimeError(
            "retained slots and KV cache must share a device: "
            f"slots={retained_slots.device}, cache={k_cache.device}"
        )

    # Flatten only physical block/page dimensions. Payload becomes
    # [slots, kv_heads, head_dim], scales become [slots, kv_heads].
    flat_k = k_cache.reshape(-1, *k_cache.shape[2:])
    flat_v = v_cache.reshape(-1, *v_cache.shape[2:])
    return (
        flat_k.index_select(0, retained_slots),
        flat_v.index_select(0, retained_slots),
    )


def dequantize_cache_for_attention(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a low-precision payload and optional scales to compute dtype."""

    if scales is not None:
        if not is_fp8_cache_tensor(tensor):
            raise ValueError("token-head scales require an FP8 payload tensor")
        if tuple(scales.shape) != tuple(tensor.shape[:-1]):
            raise ValueError(
                "scale shape must equal payload shape without head_dim: "
                f"payload={list(tensor.shape)}, scales={list(scales.shape)}"
            )
        if scales.dtype != KV_SCALE_DTYPE:
            raise ValueError(f"token-head scales must use {KV_SCALE_DTYPE}, got {scales.dtype}")
    result = tensor.to(target_dtype) if is_fp8_cache_tensor(tensor) else tensor
    if scales is not None:
        result = result * scales.to(target_dtype).unsqueeze(-1)
    return result


def expand_gqa_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand grouped-query KV heads to the query-head count."""

    if num_heads == num_kv_heads:
        return keys, values
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(
            "num_heads must be divisible by num_kv_heads, "
            f"got num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )
    groups = num_heads // num_kv_heads
    return (
        keys.repeat_interleave(groups, dim=1),
        values.repeat_interleave(groups, dim=1),
    )


def _gather_and_prepare(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_ids: torch.Tensor,
    context_len: int,
    *,
    target_dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys, values = gather_paged_kv_for_sequence(
        k_cache,
        v_cache,
        block_ids,
        context_len,
    )
    if k_scale_cache is None:
        k_scales = v_scales = None
    else:
        k_scales, v_scales = gather_paged_kv_for_sequence(
            k_scale_cache,
            v_scale_cache,
            block_ids,
            context_len,
        )
    keys = dequantize_cache_for_attention(keys, target_dtype, k_scales)
    values = dequantize_cache_for_attention(values, target_dtype, v_scales)
    return expand_gqa_kv(
        keys,
        values,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def paged_prefill_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context: Context,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run chunked/prefix-hit prefill against already-written paged KV."""

    if context.block_tables is None or context.context_lens is None or context.cu_seqlens_q is None:
        raise RuntimeError("paged prefill requires block_tables/context_lens/cu_seqlens_q")
    outputs: list[torch.Tensor] = []
    for sequence_index in range(context.context_lens.numel()):
        query_start = int(context.cu_seqlens_q[sequence_index].item())
        query_end = int(context.cu_seqlens_q[sequence_index + 1].item())
        query_len = query_end - query_start
        context_len = int(context.context_lens[sequence_index].item())
        if query_len <= 0 or context_len < query_len:
            raise RuntimeError(
                f"invalid paged prefill lengths: query={query_len} context={context_len}"
            )
        prefix_len = context_len - query_len
        with profile_region(
            "attention.prefill.paged_gather",
            metadata={"query_len": query_len, "context_len": context_len},
        ):
            keys, values = _gather_and_prepare(
                k_cache,
                v_cache,
                context.block_tables[sequence_index],
                context_len,
                target_dtype=q.dtype,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
            )

        queries = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
        keys = keys.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)
        query_positions = torch.arange(prefix_len, context_len, device=q.device)
        key_positions = torch.arange(context_len, device=q.device)
        causal_mask = (
            (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        )
        with profile_region("attention.prefill.paged_sdpa"):
            output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=causal_mask,
                is_causal=False,
                scale=scale,
            )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def paged_decode_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context: Context,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    profile_prefix: str,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one decode step using gathered paged KV and PyTorch SDPA."""

    if context.block_tables is None or context.context_lens is None:
        with profile_region(f"{profile_prefix}.sdpa_no_cache"):
            return F.scaled_dot_product_attention(
                q,
                q.new_empty(0),
                q.new_empty(0),
                is_causal=True,
                scale=scale,
            )

    outputs: list[torch.Tensor] = []
    for sequence_index in range(q.shape[0]):
        with profile_region(f"{profile_prefix}.context_len", cuda=False):
            context_len = int(context.context_lens[sequence_index].item())
        with profile_region(
            f"{profile_prefix}.gather",
            metadata={"context_len": context_len},
        ):
            keys, values = gather_paged_kv_for_sequence(
                k_cache,
                v_cache,
                context.block_tables[sequence_index],
                context_len,
            )
            if k_scale_cache is None:
                k_scales = v_scales = None
            else:
                k_scales, v_scales = gather_paged_kv_for_sequence(
                    k_scale_cache,
                    v_scale_cache,
                    context.block_tables[sequence_index],
                    context_len,
                )
        with profile_region(f"{profile_prefix}.dequant"):
            keys = dequantize_cache_for_attention(keys, q.dtype, k_scales)
            values = dequantize_cache_for_attention(values, q.dtype, v_scales)
        with profile_region(f"{profile_prefix}.expand_gqa"):
            keys, values = expand_gqa_kv(
                keys,
                values,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        query = q[sequence_index].unsqueeze(0).unsqueeze(2)
        keys = keys.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)
        with profile_region(f"{profile_prefix}.sdpa"):
            output = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                is_causal=False,
                scale=scale,
            )
        outputs.append(output.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


def visual_pruned_decode_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context: Context,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode against the retained physical-slot view for visual pruning."""

    if context.block_tables is None or context.context_lens is None:
        raise RuntimeError("visual_prune decode requires paged block_tables and context_lens")
    if not k_cache.numel() or not v_cache.numel():
        raise RuntimeError("visual_prune decode requires populated KV cache")
    slot_mappings = context.visual_pruning_slot_mappings
    if len(slot_mappings) != q.shape[0]:
        raise RuntimeError(
            "visual_prune decode requires one retained slot mapping per batch row: "
            f"mappings={len(slot_mappings)}, batch={q.shape[0]}"
        )

    outputs: list[torch.Tensor] = []
    for sequence_index, retained_slots in enumerate(slot_mappings):
        with profile_region(
            "attention.decode.visual_prune.gather",
            metadata={"retained_len": retained_slots.numel()},
        ):
            keys, values = gather_paged_kv_slots_for_sequence(
                k_cache,
                v_cache,
                retained_slots,
            )
            if k_scale_cache is None:
                k_scales = v_scales = None
            else:
                k_scales, v_scales = gather_paged_kv_slots_for_sequence(
                    k_scale_cache,
                    v_scale_cache,
                    retained_slots,
                )
        with profile_region("attention.decode.visual_prune.dequant"):
            keys = dequantize_cache_for_attention(keys, q.dtype, k_scales)
            values = dequantize_cache_for_attention(values, q.dtype, v_scales)
        with profile_region("attention.decode.visual_prune.expand_gqa"):
            keys, values = expand_gqa_kv(
                keys,
                values,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        query = q[sequence_index].unsqueeze(0).unsqueeze(2)
        keys = keys.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)
        with profile_region("attention.decode.visual_prune.sdpa"):
            output = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                is_causal=False,
                scale=scale,
            )
        outputs.append(output.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


__all__ = [
    "dequantize_cache_for_attention",
    "expand_gqa_kv",
    "gather_paged_kv_for_sequence",
    "gather_paged_kv_slots_for_sequence",
    "paged_decode_attention_reference",
    "paged_prefill_attention_reference",
    "visual_pruned_decode_attention_reference",
]
