"""P9 paired quality artifact validation and non-inferiority tests."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_comparison import (
    _validate_paired_samples,
    compare_quality_artifacts,
    paired_bootstrap_non_inferiority,
    validate_quality_artifact,
)
from prism_infer.analysis.p9_quality_materialization import selected_ids_sha256
from prism_infer.analysis.p9_quality_metrics import aggregate_quality_predictions


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = json.loads(
    (REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json").read_text(
        encoding="utf-8"
    )
)
PROTOCOL = json.loads(
    (REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json").read_text(
        encoding="utf-8"
    )
)


def _cache(mode: str) -> dict[str, object]:
    payload_shape = [2, 1, 40, 256, 1, 128]
    payload_elements = math.prod(payload_shape)
    if mode == "off":
        return {
            "payload_dtype": "torch.bfloat16",
            "payload_shape": payload_shape,
            "scale_dtype": "none",
            "scale_shape": [],
            "payload_bytes": payload_elements * 2,
            "scale_bytes": 0,
            "total_bytes": payload_elements * 2,
        }
    scale_shape = payload_shape[:-1]
    scale_bytes = math.prod(scale_shape) * 4
    return {
        "payload_dtype": "torch.float8_e4m3fn",
        "payload_shape": payload_shape,
        "scale_dtype": "torch.float32",
        "scale_shape": scale_shape,
        "payload_bytes": payload_elements,
        "scale_bytes": scale_bytes,
        "total_bytes": payload_elements + scale_bytes,
    }


def _artifact(mode: str, *, score: float = 1.0) -> dict[str, object]:
    sample_id = "doc-1"
    sample = {
        "sample_id": sample_id,
        "input": {
            "source_prompt_sha256": "1" * 64,
            "chat_prompt_sha256": "2" * 64,
            "prompt_token_count": 5,
            "prompt_token_ids_sha256": "3" * 64,
            "media_sha256": ["4" * 64],
            "modality": "image",
            "image_grid_thw": [[1, 2, 2]],
            "visual_placeholder_tokens": 1,
        },
        "raw_prediction": "answer",
        "decoded_with_special_tokens": "answer<|im_end|>",
        "output_token_ids": [42, 151645],
        "score": {"target": ["answer"], "anls": score},
    }
    eligible_sha256 = selected_ids_sha256([sample_id])
    manifest_sha256 = "5" * 64
    contract = {
        "dataset": "docvqa_validation",
        "subset": "development",
        "scope": "smoke_not_quality_gate",
        "mode": mode,
        "model": "/model",
        "model_revision": EVALUATOR["model"]["revision"],
        "git": {"commit": "a" * 40, "dirty": True},
        "evaluator_sha256": canonical_json_sha256(EVALUATOR),
        "materialization_manifest_sha256": manifest_sha256,
        "eligible_sample_ids_sha256": eligible_sha256,
        "runtime": EVALUATOR["runtime"],
        "dataset_evaluator": EVALUATOR["datasets"]["docvqa_validation"],
    }
    samples = [sample]
    return {
        "schema_version": 1,
        "record_type": "p9_quality_predictions",
        "status": "complete",
        "headline_eligible": False,
        "run_identity_sha256": canonical_json_sha256(contract),
        "run_contract": contract,
        "selection": {
            "selected_contract_samples": 1,
            "selected_contract_ids_sha256": eligible_sha256,
            "eligible_run_samples": 1,
            "eligible_run_ids_sha256": eligible_sha256,
            "protocol_exclusions": [],
        },
        "materialization_verification": {
            "status": "PASS",
            "manifest_sha256": manifest_sha256,
        },
        "environment": {
            "gpu": "NVIDIA GeForce RTX 5090",
            "torch": "2.6.0",
            "cuda": "12.8",
        },
        "kv_cache": _cache(mode),
        "samples": samples,
        "completed_samples": 1,
        "aggregate": aggregate_quality_predictions(
            "docvqa_validation",
            samples,
        ),
    }


def _reference_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "doc-1",
            "answers": ["answer"],
            "media": [{"sha256": "4" * 64}],
        }
    ]


def _formal_artifact(mode: str) -> dict[str, object]:
    artifact = _artifact(mode)
    artifact["headline_eligible"] = True
    contract = artifact["run_contract"]
    contract["scope"] = "formal_development"
    contract["git"]["dirty"] = False
    artifact["run_identity_sha256"] = canonical_json_sha256(contract)
    return artifact


def test_paired_smoke_comparison_validates_identity_bytes_and_scores() -> None:
    result = compare_quality_artifacts(
        _artifact("off"),
        _artifact("scaled_fp8_kv"),
        evaluator=EVALUATOR,
        protocol=PROTOCOL,
        reference_records=_reference_records(),
    )

    assert result["decision"] == "SMOKE_ONLY"
    assert result["all_required_metrics_pass"] is True
    assert result["headline_eligible"] is False
    assert result["reference_scores_recomputed"] is True
    assert result["metrics"]["anls"]["confidence_interval_95"] == {
        "lower": 0.0,
        "upper": 0.0,
    }
    assert result["kv_cache"]["candidate_to_baseline_total_ratio"] == 0.515625


def test_quality_artifact_rejects_tampered_aggregate() -> None:
    artifact = _artifact("off")
    artifact["aggregate"]["mean_anls"] = 0.0

    with pytest.raises(ValueError, match="aggregate"):
        validate_quality_artifact(artifact, evaluator=EVALUATOR)


def test_quality_artifact_rejects_tampered_sample_score_against_reference() -> None:
    artifact = _artifact("off", score=0.0)

    with pytest.raises(ValueError, match="independently recomputed score"):
        validate_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            reference_records=_reference_records(),
        )


def test_quality_pair_rejects_unpaired_input_identity() -> None:
    candidate = _artifact("scaled_fp8_kv")
    candidate["samples"][0]["input"]["media_sha256"] = ["6" * 64]

    with pytest.raises(ValueError, match="input identity"):
        compare_quality_artifacts(
            _artifact("off"),
            candidate,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
        )


def test_quality_pair_rejects_different_decoded_video_identity() -> None:
    baseline = {
        "sample_id": "mv-1",
        "input": {"media_sha256": ["1" * 64]},
        "score": {"target": "A"},
        "task": "action",
        "video_sampling": {"sampled_rgb_identity_sha256": "2" * 64},
    }
    candidate = deepcopy(baseline)
    candidate["video_sampling"]["sampled_rgb_identity_sha256"] = "3" * 64

    with pytest.raises(ValueError, match="decoded video identity"):
        _validate_paired_samples("mvbench_test", [baseline], [candidate])


def test_paired_bootstrap_fails_when_lower_bound_exceeds_margin() -> None:
    failed = paired_bootstrap_non_inferiority(
        [1.0] * 8,
        [0.0] * 8,
        margin=0.01,
        seed=20260717,
        resamples=100,
    )
    passed = paired_bootstrap_non_inferiority(
        [0.0, 1.0] * 4,
        [0.0, 1.0] * 4,
        margin=0.01,
        seed=20260717,
        resamples=100,
    )

    assert failed["confidence_interval_95"]["lower"] == -1.0
    assert failed["pass"] is False
    assert passed["pass"] is True


def test_headline_requirement_rejects_smoke_pair() -> None:
    with pytest.raises(ValueError, match="headline"):
        compare_quality_artifacts(
            deepcopy(_artifact("off")),
            deepcopy(_artifact("scaled_fp8_kv")),
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            require_headline=True,
        )


def test_formal_comparison_requires_reference_records() -> None:
    with pytest.raises(ValueError, match="require reference records"):
        compare_quality_artifacts(
            _formal_artifact("off"),
            _formal_artifact("scaled_fp8_kv"),
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
        )
