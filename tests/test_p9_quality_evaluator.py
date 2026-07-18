"""P9 quality evaluator 的纯 CPU artifact 与计分契约。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from benchmarks.bench_p9_quality import (
    _prepare_dataset_records,
    aggregate_predictions,
    score_prediction,
)
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_metrics import MUIRBENCH_RANDOM_FALLBACK_SEED


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"


def test_evaluator_protocol_is_frozen_and_references_quality_protocol() -> None:
    evaluator = json.loads(EVALUATOR.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert canonical_json_sha256(evaluator) == (
        "aa00962fd516c08d7a9fb42df33f20929e360b17b97c369757ce4bd46999d91b"
    )
    assert evaluator["quality_protocol_sha256"] == canonical_json_sha256(protocol)
    assert evaluator["runtime"]["enable_chunked_prefill"] is False
    assert evaluator["runtime"]["image_max_pixels"] == 602112
    assert evaluator["runtime"]["video_frames"] == 16
    assert evaluator["datasets"]["mvbench_test"]["video_sampling"][
        "video_file_decoder"
    ] == {
        "distribution": "opencv-python-headless",
        "distribution_version": "4.10.0.84",
        "api_version": "4.10.0",
        "backend": "FFMPEG",
        "color_conversion": "BGR_to_RGB",
        "frame_access_policy": "random_seek_then_sequential_count_and_decode",
    }
    assert evaluator["artifact_contract"]["aggregate_only_is_invalid"] is True
    assert evaluator["artifact_contract"]["decoded_with_special_tokens"] is True


def test_dataset_scorers_keep_raw_parser_guardrails() -> None:
    doc = score_prediction(
        "docvqa_validation",
        {"answers": ["invoice"]},
        "invoice",
    )
    muir = score_prediction(
        "muirbench_test",
        {"options": ["red", "blue"], "answer": "B"},
        "(B)",
        muirbench_random=random.Random(MUIRBENCH_RANDOM_FALLBACK_SEED),
    )
    mv = score_prediction(
        "mvbench_test",
        {"candidates": ["run", "sit"], "answer_index": 0},
        "A. run",
    )

    assert doc["anls"] == 1.0
    assert muir["strict_score"] == muir["official_score"] == 1
    assert mv["score"] == 1
    assert mv["answered"] is True


def test_aggregates_are_recomputed_from_per_sample_scores() -> None:
    muir = aggregate_predictions(
        "muirbench_test",
        [
            {
                "score": {
                    "strict_score": 0,
                    "official_score": 1,
                    "official_parse_method": "official_random_fallback",
                }
            },
            {
                "score": {
                    "strict_score": 1,
                    "official_score": 1,
                    "official_parse_method": "bracketed_label",
                }
            },
        ],
    )
    mv = aggregate_predictions(
        "mvbench_test",
        [
            {"task": "a", "score": {"score": 1, "answered": True}},
            {"task": "a", "score": {"score": 0, "answered": False}},
        ],
    )

    assert muir["strict_accuracy"] == 0.5
    assert muir["official_compatible_accuracy"] == 1.0
    assert muir["official_random_fallback_samples"] == 1
    assert mv["selected_denominator_accuracy"] == 0.5
    assert mv["lmms_answered_denominator_accuracy"] == 1.0


def test_subset_selection_preserves_frozen_order_and_explicit_exclusion(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.jsonl"
    records = [
        {"sample_id": "a", "media": [{"sha256": "a" * 64}]},
        {
            "sample_id": "b",
            "media": [
                {
                    "sha256": None,
                    "materialization_status": "excluded_fixture",
                }
            ],
        },
        {"sample_id": "c", "media": [{"sha256": "c" * 64}]},
    ]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    artifact = {
        "selected_records": {"path": records_path.name},
        "selection": {
            "development": {"sample_ids": ["a", "b", "c"]},
            "final": {"sample_ids": ["a", "b", "c"]},
        },
    }

    eligible, exclusions, contract_ids = _prepare_dataset_records(
        artifact=artifact,
        materialized_root=tmp_path,
        subset="development",
        max_samples=1,
    )

    assert [record["sample_id"] for record in eligible] == ["a"]
    assert exclusions == [{"sample_id": "b", "reason": "excluded_fixture"}]
    assert contract_ids == ["a", "b", "c"]
