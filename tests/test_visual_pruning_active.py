"""P5.2 active logical visual-pruning decode tests."""

import pytest
import torch
import torch.nn.functional as F

from prism_infer.engine.compression import CompressionMetadata
from prism_infer.engine.visual_pruning import build_retained_context_indices
from prism_infer.layers.attention import Attention
from prism_infer.utils.context import reset_context, set_context


def _visual_pruning_record(
    *,
    prompt_token_count: int = 6,
    kept_token_indices: list[int] | None = None,
    dropped_token_indices: list[int] | None = None,
) -> dict[str, object]:
    kept = [1, 5] if kept_token_indices is None else kept_token_indices
    dropped = [2, 3] if dropped_token_indices is None else dropped_token_indices
    return {
        "seq_id": 0,
        "batch_index": 0,
        "prompt_token_count": prompt_token_count,
        "total_visual_tokens": len(kept) + len(dropped),
        "kept_visual_tokens": len(kept),
        "dropped_visual_tokens": len(dropped),
        "keep_ratio_target": len(kept) / max(1, len(kept) + len(dropped)),
        "keep_ratio_actual": len(kept) / max(1, len(kept) + len(dropped)),
        "strategy": "uniform",
        "physical_compaction": False,
        "visual_token_spans": [
            {"modality": "image", "start": 1, "end": 4, "index": 0, "token_count": 3},
            {"modality": "image", "start": 5, "end": 6, "index": 1, "token_count": 1},
        ],
        "kept_token_indices": kept,
        "dropped_token_indices": dropped,
    }


def _make_decode_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260709)
    dtype = torch.float32
    # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.randn(2, 4, 2, 8, dtype=dtype)
    v_cache = torch.randn(2, 4, 2, 8, dtype=dtype)
    # q: [batch, num_heads, head_dim], current k/v: [batch, num_kv_heads, head_dim]
    q = torch.randn(1, 4, 8, dtype=dtype)
    current_k = torch.randn(1, 2, 8, dtype=dtype)
    current_v = torch.randn(1, 2, 8, dtype=dtype)
    return q, current_k, current_v, k_cache, v_cache


def _run_decode_attention(
    metadata: CompressionMetadata,
    q: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    attn = Attention(num_heads=4, num_kv_heads=2, head_dim=8, scale=8 ** -0.5)
    attn.k_cache = k_cache.clone()
    attn.v_cache = v_cache.clone()
    try:
        set_context(
            False,
            slot_mapping=torch.tensor([6], dtype=torch.int32),
            context_lens=torch.tensor([7], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
            compression_metadata=metadata,
        )
        with torch.inference_mode():
            return attn(q, current_k, current_v)
    finally:
        reset_context()


def _reference_decode_output(
    record: dict[str, object] | None,
    q: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    ref_k_cache = k_cache.clone()
    ref_v_cache = v_cache.clone()
    # slot_mapping=[6] -> block_id=1, block_offset=2 when block_size=4.
    ref_k_cache[1, 2] = current_k[0]
    ref_v_cache[1, 2] = current_v[0]

    keys = torch.cat([ref_k_cache[0, :4], ref_k_cache[1, :3]], dim=0)
    values = torch.cat([ref_v_cache[0, :4], ref_v_cache[1, :3]], dim=0)
    retained_indices = build_retained_context_indices(record, context_len=7)
    retained_index_tensor = torch.tensor(retained_indices, dtype=torch.long)
    keys = keys.index_select(0, retained_index_tensor)
    values = values.index_select(0, retained_index_tensor)
    keys = keys.repeat_interleave(2, dim=1)
    values = values.repeat_interleave(2, dim=1)

    # q_i: [1, heads, 1, dim], compact KV: [1, heads, retained_len, dim]
    q_i = q[0].unsqueeze(0).unsqueeze(2)
    k_i = keys.transpose(0, 1).unsqueeze(0)
    v_i = values.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(
        q_i,
        k_i,
        v_i,
        is_causal=False,
        scale=8 ** -0.5,
    ).squeeze(0).squeeze(1).unsqueeze(0)


def _visual_prune_metadata(record: dict[str, object] | None) -> CompressionMetadata:
    records = () if record is None else (record,)
    records_by_batch = () if record is None else (record,)
    return CompressionMetadata(
        mode="visual_prune",
        is_prefill=False,
        num_sequences=1,
        total_prompt_tokens=6,
        total_image_tokens=0 if record is None else 4,
        total_video_tokens=0,
        block_size=4,
        visual_pruning_config={
            "keep_ratio": 0.5,
            "min_keep_tokens": 1,
            "strategy": "uniform",
        },
        visual_pruning_decision_records=records,
        visual_pruning_records_by_batch=records_by_batch,
    )


def test_active_visual_prune_decode_matches_compact_reference():
    """Active decode output must match an independently compacted KV reference."""

    q, current_k, current_v, k_cache, v_cache = _make_decode_tensors()
    record = _visual_pruning_record()
    metadata = _visual_prune_metadata(record)

    output = _run_decode_attention(metadata, q, current_k, current_v, k_cache, v_cache)
    reference = _reference_decode_output(record, q, current_k, current_v, k_cache, v_cache)
    diff = (output - reference).abs()

    print(f"active prune q shape: {list(q.shape)}")
    print(f"active prune k_cache shape: {list(k_cache.shape)}")
    print(f"active prune output shape: {list(output.shape)}")
    print(f"active prune reference shape: {list(reference.shape)}")
    print(f"active prune output mean/std: {output.mean().item():.6e}/{output.std().item():.6e}")
    print(
        "active prune reference mean/std: "
        f"{reference.mean().item():.6e}/{reference.std().item():.6e}"
    )
    print(f"active prune max diff: {diff.max().item():.6e}")

    assert output.shape == reference.shape == (1, 4, 8)
    assert diff.max().item() < 1e-5
    print("active visual prune compact reference: PASS")


def test_active_visual_prune_keep_all_matches_off_decode():
    """keep_ratio=1.0 active path should be exactly equal to off decode."""

    q, current_k, current_v, k_cache, v_cache = _make_decode_tensors()
    keep_all_record = _visual_pruning_record(
        kept_token_indices=[1, 2, 3, 5],
        dropped_token_indices=[],
    )
    active_metadata = _visual_prune_metadata(keep_all_record)
    off_metadata = CompressionMetadata(
        mode="off",
        is_prefill=False,
        num_sequences=1,
        total_prompt_tokens=6,
        total_image_tokens=4,
        total_video_tokens=0,
        block_size=4,
    )

    active = _run_decode_attention(active_metadata, q, current_k, current_v, k_cache, v_cache)
    off = _run_decode_attention(off_metadata, q, current_k, current_v, k_cache, v_cache)
    diff = (active - off).abs()

    print(f"keep-all active output shape: {list(active.shape)}")
    print(f"keep-all off output shape: {list(off.shape)}")
    print(f"keep-all max diff: {diff.max().item():.6e}")

    assert torch.equal(active, off)
    print("active visual prune keep-all off equivalence: PASS")


def test_active_visual_prune_decode_requires_prefill_record():
    """Active decode must not silently fall back when VL pruning records are missing."""

    q, current_k, current_v, k_cache, v_cache = _make_decode_tensors()
    metadata = CompressionMetadata(
        mode="visual_prune",
        is_prefill=False,
        num_sequences=1,
        total_prompt_tokens=6,
        total_image_tokens=4,
        total_video_tokens=0,
        block_size=4,
    )

    with pytest.raises(RuntimeError, match="batch-aligned pruning records"):
        _run_decode_attention(metadata, q, current_k, current_v, k_cache, v_cache)
    print("active visual prune missing-record guard: PASS")
