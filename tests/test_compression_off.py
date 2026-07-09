"""P5 compression metadata and compression-off baseline tests."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.compression import (
    COMPRESSION_FP8_KV,
    COMPRESSION_VISUAL_PRUNE,
    CompressionMetadata,
    build_compression_metadata,
    ensure_compression_off,
    ensure_supported_compression_metadata,
    normalize_compression_mode,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.layers.attention import Attention
from prism_infer.sampling_params import SamplingParams
from prism_infer.utils.context import reset_context, set_context


def test_compression_mode_validation():
    """P5 accepts the off baseline and the first active visual-prune mode."""

    assert normalize_compression_mode(None) == "off"
    assert normalize_compression_mode("OFF") == "off"
    assert normalize_compression_mode("visual_prune") == COMPRESSION_VISUAL_PRUNE
    assert normalize_compression_mode("fp8_kv") == COMPRESSION_FP8_KV
    with pytest.raises(ValueError, match="supported compression_mode"):
        normalize_compression_mode("int4_kv")
    print("compression mode off validation: PASS")


def test_compression_metadata_counts_visual_tokens():
    """Off metadata should expose visual-token counts without changing behavior."""

    config = SimpleNamespace(compression_mode="off", kvcache_block_size=256)
    image_seq = Sequence(
        [1, 99, 99, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=2,
    )
    video_seq = Sequence(
        [1, 98, 98, 98, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        video_token_id=98,
        video_token_count=3,
    )

    metadata = build_compression_metadata(
        config,
        [image_seq, video_seq],
        is_prefill=True,
    )

    print(f"compression metadata mode: {metadata.mode}")
    print(f"compression metadata visual tokens: {metadata.total_visual_tokens}")
    print(f"compression metadata prompt tokens: {metadata.total_prompt_tokens}")

    assert metadata.mode == "off"
    assert not metadata.enabled
    assert metadata.is_prefill
    assert metadata.num_sequences == 2
    assert metadata.total_image_tokens == 2
    assert metadata.total_video_tokens == 3
    assert metadata.total_visual_tokens == 5
    assert metadata.total_prompt_tokens == 9
    assert metadata.block_size == 256
    print("compression metadata visual-token counts: PASS")


def test_visual_pruning_shadow_metadata_records_prefill_decisions():
    """P5.2 shadow decisions must be auditable without enabling compression."""

    config = SimpleNamespace(
        compression_mode="off",
        kvcache_block_size=256,
        enable_visual_pruning_shadow=True,
        visual_pruning_keep_ratio=0.5,
        visual_pruning_min_keep_tokens=1,
        visual_pruning_strategy="uniform",
    )
    seq = Sequence(
        [1, 99, 99, 2, 99],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=3,
    )

    metadata = build_compression_metadata(config, [seq], is_prefill=True)
    record = metadata.visual_pruning_decision_records[0]

    print(f"shadow metadata enabled: {metadata.visual_pruning_shadow_enabled}")
    print(f"shadow metadata compression enabled: {metadata.enabled}")
    print(f"shadow metadata decision records: {metadata.visual_pruning_decision_records}")

    assert metadata.mode == "off"
    assert not metadata.enabled
    assert metadata.visual_pruning_shadow_enabled
    assert metadata.visual_pruning_config == {
        "keep_ratio": 0.5,
        "min_keep_tokens": 1,
        "strategy": "uniform",
    }
    assert len(metadata.visual_pruning_decision_records) == 1
    assert record["total_visual_tokens"] == 3
    assert record["kept_visual_tokens"] == 2
    assert record["dropped_visual_tokens"] == 1
    assert record["kept_token_indices"] == [1, 4]
    assert record["dropped_token_indices"] == [2]
    assert record["physical_compaction"] is False
    print("visual pruning shadow prefill decision metadata: PASS")


def test_visual_pruning_shadow_metadata_is_prefill_only():
    """Decode metadata should not recompute pruning decisions from partial state."""

    config = SimpleNamespace(
        compression_mode="off",
        kvcache_block_size=256,
        enable_visual_pruning_shadow=True,
        visual_pruning_keep_ratio=0.5,
        visual_pruning_min_keep_tokens=1,
        visual_pruning_strategy="uniform",
    )
    seq = Sequence(
        [1, 99, 99, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=2,
    )

    metadata = build_compression_metadata(config, [seq], is_prefill=False)

    print(f"shadow decode decision records: {metadata.visual_pruning_decision_records}")
    assert metadata.visual_pruning_shadow_enabled
    assert metadata.visual_pruning_decision_records == ()
    assert not metadata.enabled
    print("visual pruning shadow decode metadata: PASS")


def test_visual_pruning_shadow_score_mode_fails_without_runtime_scores():
    """Score-based shadow mode must not silently fall back to uniform."""

    config = SimpleNamespace(
        compression_mode="off",
        kvcache_block_size=256,
        enable_visual_pruning_shadow=True,
        visual_pruning_keep_ratio=0.5,
        visual_pruning_min_keep_tokens=1,
        visual_pruning_strategy="score",
    )
    seq = Sequence(
        [1, 99, 99, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=2,
    )

    with pytest.raises(ValueError, match="requires token_scores"):
        build_compression_metadata(config, [seq], is_prefill=True)
    print("visual pruning shadow score-mode guard: PASS")


def test_visual_prune_active_metadata_persists_prefill_to_decode():
    """Active visual pruning must store prefill decisions for decode reuse."""

    config = SimpleNamespace(
        compression_mode="visual_prune",
        kvcache_block_size=256,
        enable_visual_pruning_shadow=False,
        visual_pruning_keep_ratio=0.5,
        visual_pruning_min_keep_tokens=1,
        visual_pruning_strategy="uniform",
    )
    seq = Sequence(
        [1, 99, 99, 2, 99],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=3,
    )

    prefill_metadata = build_compression_metadata(config, [seq], is_prefill=True)
    seq.append_token(7)
    decode_metadata = build_compression_metadata(config, [seq], is_prefill=False)

    print(f"active prefill metadata enabled: {prefill_metadata.enabled}")
    print(f"active prefill records: {prefill_metadata.visual_pruning_decision_records}")
    print(f"active decode records: {decode_metadata.visual_pruning_records_by_batch}")

    assert prefill_metadata.mode == "visual_prune"
    assert prefill_metadata.enabled
    assert seq.visual_pruning_decision_record is not None
    assert prefill_metadata.visual_pruning_decision_records[0]["prompt_token_count"] == 5
    assert prefill_metadata.visual_pruning_decision_records[0]["batch_index"] == 0
    assert decode_metadata.visual_pruning_records_by_batch[0]["kept_visual_tokens"] == 2
    assert decode_metadata.visual_pruning_records_by_batch[0]["dropped_visual_tokens"] == 1
    print("visual pruning active metadata prefill->decode: PASS")


def test_visual_prune_decision_survives_decode_serialization():
    """Decode pickle payload must keep pruning state even without full token_ids."""

    seq = Sequence(
        [1, 99, 99, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=2,
    )
    seq.visual_pruning_decision_record = {
        "seq_id": seq.seq_id,
        "prompt_token_count": 4,
        "total_visual_tokens": 2,
        "kept_visual_tokens": 1,
        "dropped_visual_tokens": 1,
        "keep_ratio_target": 0.5,
        "keep_ratio_actual": 0.5,
        "strategy": "uniform",
        "physical_compaction": False,
        "visual_token_spans": [
            {"modality": "image", "start": 1, "end": 3, "index": 0, "token_count": 2}
        ],
        "kept_token_indices": [1],
        "dropped_token_indices": [2],
    }
    seq.append_token(7)

    restored = pickle.loads(pickle.dumps(seq))

    print(f"serialized pruning record: {restored.visual_pruning_decision_record}")
    assert restored.num_completion_tokens == 1
    assert restored.visual_pruning_decision_record is not None
    assert restored.visual_pruning_decision_record["dropped_token_indices"] == [2]
    print("visual pruning active decode serialization: PASS")


def _run_prefill_attention(compression_metadata: CompressionMetadata | None) -> torch.Tensor:
    torch.manual_seed(20260705)
    dtype = torch.float32
    seqlen = 5
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    q = torch.randn(seqlen, num_heads, head_dim, dtype=dtype)
    k = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype)
    v = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype)
    attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
    )
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32)

    try:
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            slot_mapping=torch.arange(seqlen, dtype=torch.int32),
            compression_metadata=compression_metadata,
        )
        with torch.inference_mode():
            return attn(q, k, v)
    finally:
        reset_context()


def test_compression_off_attention_output_identical():
    """Carrying off metadata must not change attention outputs."""

    metadata = CompressionMetadata(
        mode="off",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=2,
        total_video_tokens=0,
        block_size=256,
    )

    baseline = _run_prefill_attention(None)
    with_metadata = _run_prefill_attention(metadata)
    diff = (baseline - with_metadata).abs()

    print(f"compression off attention output shape: {list(with_metadata.shape)}")
    print(f"compression off attention max diff: {diff.max().item():.6e}")

    assert torch.equal(baseline, with_metadata)
    print("compression off attention equivalence: PASS")


def test_visual_pruning_shadow_metadata_does_not_change_attention_output():
    """Shadow pruning records are observability metadata, not active compression."""

    metadata = CompressionMetadata(
        mode="off",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=3,
        total_video_tokens=0,
        block_size=256,
        visual_pruning_shadow_enabled=True,
        visual_pruning_config={
            "keep_ratio": 0.5,
            "min_keep_tokens": 1,
            "strategy": "uniform",
        },
        visual_pruning_decision_records=(
            {
                "seq_id": 0,
                "total_visual_tokens": 3,
                "kept_visual_tokens": 2,
                "dropped_visual_tokens": 1,
                "keep_ratio_target": 0.5,
                "keep_ratio_actual": 2 / 3,
                "strategy": "uniform",
                "physical_compaction": False,
                "visual_token_spans": [
                    {"modality": "image", "start": 1, "end": 4, "index": 0, "token_count": 3}
                ],
                "kept_token_indices": [1, 3],
                "dropped_token_indices": [2],
            },
        ),
    )

    baseline = _run_prefill_attention(None)
    with_shadow = _run_prefill_attention(metadata)
    diff = (baseline - with_shadow).abs()

    print(f"shadow attention output shape: {list(with_shadow.shape)}")
    print(f"shadow attention max diff: {diff.max().item():.6e}")
    assert torch.equal(baseline, with_shadow)
    print("visual pruning shadow attention no-op: PASS")


def test_off_only_guard_rejects_active_compression_mode():
    """Off-only paths must still reject active compression metadata."""

    metadata = CompressionMetadata(
        mode="visual_prune",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=2,
        total_video_tokens=0,
        block_size=256,
    )
    with pytest.raises(NotImplementedError, match="off-only path"):
        ensure_compression_off(metadata)
    print("compression off-only guard: PASS")


def test_attention_runtime_rejects_unsupported_compression_mode():
    """Attention.forward must fail loudly for modes with no implementation."""

    metadata = CompressionMetadata(
        mode="int4_kv",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=2,
        total_video_tokens=0,
        block_size=256,
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        _run_prefill_attention(metadata)
    with pytest.raises(NotImplementedError, match="not implemented"):
        ensure_supported_compression_metadata(metadata)
    print("compression unsupported attention runtime guard: PASS")
