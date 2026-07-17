"""P5.2 visual-token pruning decision helper tests."""

import pytest
import torch

from prism_infer.engine.sequence import Sequence
from prism_infer.engine.visual_pruning import (
    RuntimeVisualTokenScorer,
    VisualPruningConfig,
    apply_pruning_to_slot_mapping,
    build_runtime_visual_token_scorer,
    compute_pruning_decision,
    finalize_attention_pruning_decisions,
    find_visual_token_spans,
)


def _mixed_visual_sequence() -> Sequence:
    return Sequence(
        [1, 99, 99, 2, 98, 98, 3, 99],
        block_size=256,
        request_id=0,
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
    assert record["kept_visual_tokens_by_span"] == [
        {"modality": "image", "span_index": 0, "kept_tokens": 1},
        {"modality": "video", "span_index": 0, "kept_tokens": 0},
        {"modality": "image", "span_index": 1, "kept_tokens": 1},
    ]
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

    bad_seq = Sequence(
        [1, 99, 99, 2],
        block_size=256,
        request_id=1,
        image_token_id=99,
        image_token_count=3,
    )
    with pytest.raises(ValueError, match="image token count mismatch"):
        find_visual_token_spans(bad_seq)
    print("visual pruning metadata guards: PASS")


def test_runtime_attention_scorer_matches_independent_reference():
    """Runtime scorer must match a direct last-query GQA attention reference."""

    torch.manual_seed(20260714)
    seq = _mixed_visual_sequence()
    scorer = RuntimeVisualTokenScorer([seq], layer_ids=(1, 2))
    scale = 2 ** -0.5
    reference_layers = []
    for layer_id in (1, 2):
        # q: [tokens=8, q_heads=4, dim=2], k: [tokens=8, kv_heads=2, dim=2]
        q = torch.randn(8, 4, 2)
        k = torch.randn(8, 2, 2)
        scorer.observe(layer_id=layer_id, q=q, k=k, scale=scale)

        # Independent reference: [q_heads=4, tokens=8]
        expanded_k = k.repeat_interleave(2, dim=1)
        logits = torch.einsum("hd,thd->ht", q[-1].float(), expanded_k.float()) * scale
        reference_layers.append(torch.softmax(logits, dim=-1).mean(dim=0))

    actual_map = scorer.finalize()[seq.seq_id]
    visual_indices = [1, 2, 4, 5, 7]
    actual = torch.tensor([actual_map[index] for index in visual_indices])
    reference = torch.stack(reference_layers).mean(dim=0)[visual_indices]
    max_diff = float((actual - reference).abs().max().item())

    print(f"runtime attention q shape: {list(q.shape)}")
    print(f"runtime attention k shape: {list(k.shape)}")
    print(f"runtime attention score shape: {list(actual.shape)}")
    print(f"runtime attention actual mean/std: {actual.mean().item():.6e}/{actual.std(unbiased=False).item():.6e}")
    print(f"runtime attention ref mean/std: {reference.mean().item():.6e}/{reference.std(unbiased=False).item():.6e}")
    print(f"runtime attention max diff: {max_diff:.6e}")
    assert max_diff < 1e-5
    print("P6.12 runtime attention score reference: PASS")


def test_runtime_attention_quality_default_selects_final_decoder_layer():
    """The quality-qualified default must observe only the final decoder layer."""

    config = VisualPruningConfig(strategy="attention")
    scorer = build_runtime_visual_token_scorer(
        [_mixed_visual_sequence()],
        num_hidden_layers=36,
        attention_last_n_layers=config.attention_last_n_layers,
    )

    assert config.attention_last_n_layers == 1
    assert scorer.layer_ids == (35,)
    print("P6.12-C attention default final-layer selection: PASS")


def test_runtime_attention_finalize_persists_auditable_decision():
    """Finalization must persist selected layers, score stats, and top tokens."""

    torch.manual_seed(17)
    seq = _mixed_visual_sequence()
    scorer = build_runtime_visual_token_scorer(
        [seq],
        num_hidden_layers=4,
        attention_last_n_layers=2,
    )
    for layer_id in (2, 3):
        scorer.observe(
            layer_id=layer_id,
            q=torch.randn(8, 4, 2),
            k=torch.randn(8, 2, 2),
            scale=2 ** -0.5,
        )
    config = VisualPruningConfig(
        keep_ratio=0.4,
        min_keep_tokens=1,
        strategy="attention",
        attention_last_n_layers=2,
    )
    records = finalize_attention_pruning_decisions([seq], config, scorer)
    record = records[0]

    assert record is not None
    assert seq.visual_pruning_decision_record == record
    assert record["strategy"] == "attention"
    assert record["score_source"] == "prefill_last_query_attention"
    assert record["score_layers"] == [2, 3]
    assert record["kept_visual_tokens"] == 2
    assert record["dropped_visual_tokens"] == 3
    kept_by_span = record["kept_visual_tokens_by_span"]
    assert [
        (str(item["modality"]), int(item["span_index"]))
        for item in kept_by_span
    ] == [("image", 0), ("video", 0), ("image", 1)]
    assert sum(
        int(item["kept_tokens"])
        for item in kept_by_span
    ) == 2
    assert float(record["score_min"]) <= float(record["score_mean"])
    assert float(record["score_mean"]) <= float(record["score_max"])
    print(f"runtime attention decision: {record}")
    print("P6.12 runtime attention decision audit: PASS")


def test_runtime_attention_scorer_rejects_missing_layer():
    """Incomplete layer observations must fail instead of using partial scores."""

    seq = _mixed_visual_sequence()
    scorer = RuntimeVisualTokenScorer([seq], layer_ids=(1, 2))
    scorer.observe(
        layer_id=1,
        q=torch.randn(8, 4, 2),
        k=torch.randn(8, 2, 2),
        scale=2 ** -0.5,
    )
    with pytest.raises(RuntimeError, match=r"missing layers: \[2\]"):
        scorer.finalize()
    print("P6.12 runtime attention missing-layer guard: PASS")
