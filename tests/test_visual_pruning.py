"""P5.2 visual-token pruning decision helper tests."""

import pytest
import torch

from prism_infer.engine.sequence import Sequence
from prism_infer.engine.visual_pruning import (
    VisualPruningConfig,
    apply_pruning_to_slot_mapping,
    compute_pruning_decision,
    find_visual_token_spans,
)


def _mixed_visual_sequence() -> Sequence:
    return Sequence(
        [1, 99, 99, 2, 98, 98, 3, 99],
        image_token_id=99,
        image_token_count=3,
        video_token_id=98,
        video_token_count=2,
    )


def test_find_visual_token_spans_handles_multiple_modal_spans():
    """Span scanning must not assume one contiguous visual-token block."""

    spans = find_visual_token_spans(_mixed_visual_sequence())
    records = [span.to_record() for span in spans]
    print(f"visual pruning spans: {records}")

    assert records == [
        {"modality": "image", "start": 1, "end": 3, "index": 0, "token_count": 2},
        {"modality": "video", "start": 4, "end": 6, "index": 0, "token_count": 2},
        {"modality": "image", "start": 7, "end": 8, "index": 1, "token_count": 1},
    ]
    print("visual pruning multi-span scan: PASS")


def test_uniform_pruning_decision_records_keep_and_drop_counts():
    """Uniform decision should be deterministic and auditable."""

    config = VisualPruningConfig(keep_ratio=0.4, min_keep_tokens=1)
    decision = compute_pruning_decision(_mixed_visual_sequence(), config)

    assert decision is not None
    record = decision.to_record()
    print(f"visual pruning decision record: {record}")

    assert decision.total_visual_tokens == 5
    assert decision.kept_visual_tokens == 2
    assert decision.dropped_visual_tokens == 3
    assert decision.keep_ratio_actual == pytest.approx(0.4)
    assert decision.kept_token_indices == (1, 7)
    assert decision.dropped_token_indices == (2, 4, 5)
    assert not decision.physical_compaction
    print("visual pruning uniform decision: PASS")


def test_score_pruning_requires_scores_for_all_visual_tokens():
    """Score strategy must fail loudly instead of falling back to uniform."""

    config = VisualPruningConfig(keep_ratio=0.4, min_keep_tokens=1, strategy="score")
    with pytest.raises(ValueError, match="missing score"):
        compute_pruning_decision(
            _mixed_visual_sequence(),
            config,
            token_scores={1: 0.1, 2: 0.9},
        )
    print("visual pruning missing score guard: PASS")


def test_score_pruning_selects_highest_scored_visual_tokens():
    """Score strategy keeps the highest-scored visual token indices."""

    config = VisualPruningConfig(keep_ratio=0.4, min_keep_tokens=1, strategy="score")
    decision = compute_pruning_decision(
        _mixed_visual_sequence(),
        config,
        token_scores={1: 0.1, 2: 0.9, 4: 0.2, 5: 1.0, 7: 0.8},
    )

    assert decision is not None
    print(f"visual pruning score kept indices: {decision.kept_token_indices}")
    assert decision.kept_token_indices == (2, 5)
    assert decision.dropped_token_indices == (1, 4, 7)
    print("visual pruning score decision: PASS")


def test_pruning_slot_mapping_masks_only_dropped_visual_tokens():
    """Slot masking is a prefill helper and must be explicit about dropped tokens."""

    config = VisualPruningConfig(keep_ratio=0.4, min_keep_tokens=1)
    decision = compute_pruning_decision(_mixed_visual_sequence(), config)
    slot_mapping = torch.arange(10, dtype=torch.int32)
    masked = apply_pruning_to_slot_mapping(slot_mapping, [decision], [0])

    print(f"visual pruning slot_mapping input shape: {list(slot_mapping.shape)}")
    print(f"visual pruning slot_mapping output: {masked.tolist()}")
    assert masked.tolist() == [0, 1, -1, 3, -1, -1, 6, 7, 8, 9]
    assert slot_mapping.tolist() == list(range(10))
    print("visual pruning slot mask helper: PASS")


def test_visual_pruning_rejects_unsupported_strategy_and_bad_metadata():
    """Unsupported strategies and inconsistent token metadata must fail loudly."""

    with pytest.raises(ValueError, match="unsupported strategy"):
        VisualPruningConfig(strategy="importance")

    bad_seq = Sequence([1, 99, 99, 2], image_token_id=99, image_token_count=3)
    with pytest.raises(ValueError, match="image token count mismatch"):
        find_visual_token_spans(bad_seq)
    print("visual pruning metadata guards: PASS")
