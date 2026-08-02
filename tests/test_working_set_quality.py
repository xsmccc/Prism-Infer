"""Focused checks for repeated-context quality comparisons."""

import copy
from types import SimpleNamespace

import pytest

from prism_infer.analysis.working_set_quality import (
    QUALITY_STAGE_SPECS,
    build_muirbench_media_first_prompt,
    media_group_id,
    paired_deleted_subset,
    select_multi_question_groups,
    summarize_stage_samples,
)
from prism_infer.engine.compression import build_compression_metadata
from prism_infer.engine.input_preparation import ModelInputPreparer
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.visual_pruning import (
    VisualPruningConfig,
    compute_pruning_decision,
)


def _record(sample_id: str, media_digest: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "media": [{"sha256": media_digest}],
    }


def _docvqa_sample(
    sample_id: str,
    *,
    anls: float,
    dropped_visual_tokens: int = 0,
) -> dict[str, object]:
    decision = None
    if dropped_visual_tokens:
        decision = {"dropped_visual_tokens": dropped_visual_tokens}
    return {
        "sample_id": sample_id,
        "question_index": 0,
        "score": {"anls": anls},
        "compression_decision": decision,
    }


def test_multi_question_groups_are_content_sorted_and_keep_source_question_order():
    records = [
        _record("b-first", "b" * 64),
        _record("single", "c" * 64),
        _record("a-first", "a" * 64),
        _record("b-second", "b" * 64),
        _record("a-second", "a" * 64),
    ]

    grouped = select_multi_question_groups(records)

    group_ids = list(dict.fromkeys(sample.media_group_id for sample in grouped))
    expected_group_ids = sorted((media_group_id(records[0]), media_group_id(records[2])))
    assert group_ids == expected_group_ids
    samples_by_group = {
        group_id: [
            sample.record["sample_id"] for sample in grouped if sample.media_group_id == group_id
        ]
        for group_id in group_ids
    }
    assert samples_by_group[media_group_id(records[0])] == ["b-first", "b-second"]
    assert samples_by_group[media_group_id(records[2])] == ["a-first", "a-second"]
    assert [sample.question_index for sample in grouped] == [0, 1, 0, 1]
    assert all(sample.group_size == 2 for sample in grouped)


def test_media_first_muirbench_prompt_preserves_ordered_image_references():
    prompt = build_muirbench_media_first_prompt(
        "Compare <image> with <image> and choose.",
        ["same", "<image>"],
    )

    assert "<image>" not in prompt
    assert "Image 1 through Image 3" in prompt
    assert "Compare Image 1 with Image 2 and choose." in prompt
    assert "(A) same" in prompt
    assert "(B) Image 3" in prompt


def test_deleted_subset_does_not_use_uncompressed_samples_to_dilute_quality():
    dense = [
        _docvqa_sample("kept", anls=1.0),
        _docvqa_sample("deleted", anls=0.8),
    ]
    compact = [
        _docvqa_sample("kept", anls=1.0),
        _docvqa_sample("deleted", anls=0.4, dropped_visual_tokens=200),
    ]

    stage = summarize_stage_samples("docvqa_validation", compact)
    comparison = paired_deleted_subset(
        dataset_id="docvqa_validation",
        reference_samples=dense,
        candidate_samples=compact,
    )

    assert stage["all_samples"]["mean_anls"] == 0.7
    assert stage["actual_token_deletion"]["quality"]["mean_anls"] == 0.4
    assert comparison["actual_token_deletion_samples"] == 1
    assert comparison["reference_quality"]["mean_anls"] == 0.8
    assert comparison["candidate_quality"]["mean_anls"] == 0.4
    assert comparison["dropped_visual_tokens"] == 200


def test_first_question_attention_stage_uses_real_prefix_cache_reuse():
    spec = QUALITY_STAGE_SPECS["muir_attention_first_reuse"]

    assert spec.pruning_strategy == "attention"
    assert spec.enable_prefix_caching
    assert spec.prompt_layout == "media_first"
    assert spec.uses_compaction
    assert spec.attention_selection_scope == "first_question"
    assert spec.reuse_scope == "media_group"


def test_dense_stages_keep_all_tokens_and_only_media_first_requires_prefix_boundary():
    dense_stages = [spec for stage, spec in QUALITY_STAGE_SPECS.items() if "dense" in stage]
    official = QUALITY_STAGE_SPECS["muir_dense_official"]

    assert dense_stages
    assert all(spec.uses_compaction for spec in dense_stages)
    assert all(spec.enable_prefix_caching for spec in dense_stages)
    assert all(spec.effective_keep_ratio(0.6) == 1.0 for spec in dense_stages)
    assert not official.requires_physical_prefix_kv
    assert all(
        spec.requires_physical_prefix_kv for spec in dense_stages if spec is not official
    )


def test_mvbench_group_identity_includes_temporal_bound_and_sampling_contract():
    records = [
        {**_record("first", "a" * 64), "temporal_bound": {"start": 0, "end": 5}},
        {**_record("second", "a" * 64), "temporal_bound": {"start": 0, "end": 5}},
        {**_record("other-time", "a" * 64), "temporal_bound": {"start": 5, "end": 10}},
    ]
    sampling = {"algorithm": "segment_center", "frames": 16}

    grouped = select_multi_question_groups(records, sampling_identity=sampling)

    assert [sample.record["sample_id"] for sample in grouped] == ["first", "second"]
    assert media_group_id(records[0], sampling_identity=sampling) != media_group_id(
        records[0],
        sampling_identity={"algorithm": "segment_center", "frames": 8},
    )


def _attention_replay_fixture():
    seq = Sequence(
        [1, 9, 9, 9, 9, 2],
        block_size=4,
        request_id=17,
        image_token_id=9,
        image_token_count=4,
    )
    pruning = VisualPruningConfig(
        keep_ratio=0.5,
        min_keep_tokens=1,
        strategy="attention",
    )
    decision = compute_pruning_decision(
        seq,
        pruning,
        token_scores={1: 0.1, 2: 0.8, 3: 0.2, 4: 0.9},
    )
    assert decision is not None
    config = SimpleNamespace(
        compression_mode="visual_compact_scaled_fp8",
        enable_visual_pruning_shadow=False,
        visual_pruning_keep_ratio=0.5,
        visual_pruning_min_keep_tokens=1,
        visual_pruning_video_min_keep_tokens=1,
        visual_pruning_strategy="attention",
        visual_pruning_attention_last_n_layers=1,
        kvcache_block_size=4,
    )
    return seq, config, decision.to_record()


def test_locked_attention_selection_is_validated_and_serialized():
    seq, config, replay = _attention_replay_fixture()
    replay["selection_source_sample_id"] = "sample-17"
    seq.visual_pruning_replay_record = replay

    metadata = build_compression_metadata(config, [seq], is_prefill=True)
    record = metadata.visual_pruning_records_by_batch[0]
    scorer = ModelInputPreparer._visual_pruning_scorer(
        SimpleNamespace(config=None),
        [seq],
        (SimpleNamespace(sequence_id=seq.seq_id, token_start=0, token_end=3),),
        has_visual_payload=True,
        compression_metadata=metadata,
    )
    state = seq.__getstate__()
    restored = Sequence.__new__(Sequence)
    restored.__setstate__(state)

    assert record is not None
    assert record["selection_replay_locked"]
    assert record["selection_source_sample_id"] == "sample-17"
    assert tuple(record["kept_token_indices"]) == (2, 4)
    assert scorer is None
    assert restored.visual_pruning_replay_record == replay


@pytest.mark.parametrize("tamper", ["span", "partition"])
def test_locked_attention_selection_rejects_changed_visual_context(tamper):
    seq, config, replay = _attention_replay_fixture()
    replay = copy.deepcopy(replay)
    if tamper == "span":
        replay["visual_token_spans"][0]["end"] = 4
        replay["visual_token_spans"][0]["token_count"] = 3
    else:
        replay["dropped_token_indices"] = replay["dropped_token_indices"][:-1]
    seq.visual_pruning_replay_record = replay

    with pytest.raises(ValueError):
        build_compression_metadata(config, [seq], is_prefill=True)
