"""P9 runtime/quality protocol freeze tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_protocol import (
    load_p9_quality_protocol,
    load_p9_runtime_manifest,
    validate_p9_quality_protocol,
    validate_p9_runtime_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MANIFEST = REPO_ROOT / "benchmarks/workloads/p9_headline.json"
QUALITY_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"


def test_p9_runtime_manifest_freezes_headline_shapes_and_traffic() -> None:
    """H1/H2/H3 的核心形状、流量与物理显存口径应可机器校验。"""

    manifest = load_p9_runtime_manifest(RUNTIME_MANIFEST)
    cases = {case["id"]: case for case in manifest["cases"]}
    h1 = cases["h1_eight_image_448"]["requests"][0]
    h2 = cases["h2_video_16x448"]["requests"][0]
    h3 = manifest["p9_protocol"]["headline"]["H3"]

    assert len(h1["images"]) == 8
    assert len(h2["frames"]) == 16
    assert h3["completed_requests_per_run"] == 600
    assert h3["arrival_seeds"] == [20260717, 20260718, 20260719]
    assert manifest["p9_protocol"]["memory_contract"]["kv_pool_bytes"] == 2**32


def test_p9_quality_protocol_freezes_sources_revisions_and_margins() -> None:
    """标准质量来源不能在看到压缩结果后静默更换。"""

    protocol = load_p9_quality_protocol(QUALITY_PROTOCOL)
    datasets = {dataset["id"]: dataset for dataset in protocol["datasets"]}

    assert set(datasets) == {
        "docvqa_validation",
        "muirbench_test",
        "mvbench_test",
    }
    assert datasets["docvqa_validation"]["revision"] == (
        "539088ef8a8ada01ac8e2e6d4e372586748a265e"
    )
    assert protocol["non_inferiority"][
        "bounded_accuracy_margin_percentage_points"
    ] == 1.0
    assert protocol["existing_preflight_only"]["headline_eligible"] is False


def test_p9_protocol_canonical_hashes_are_stable_and_distinct() -> None:
    """两个 versioned contract 应有稳定且不同的内容身份。"""

    runtime = load_p9_runtime_manifest(RUNTIME_MANIFEST)
    quality = load_p9_quality_protocol(QUALITY_PROTOCOL)
    runtime_hash = canonical_json_sha256(runtime)
    quality_hash = canonical_json_sha256(quality)

    assert runtime_hash == (
        "42d1387320b1b30c3b0afa0bf3113f0dd905a38b38bc583cfe6c6eb3ef4f8656"
    )
    assert quality_hash == (
        "85adb4b246ab3fc55bc70e02ad75d97c5aa903e89387e499fc3aea1ac2edb25d"
    )


def test_p9_runtime_manifest_rejects_unknown_case_or_invalid_weight_sum() -> None:
    """H3 class typo 和比例漂移必须 fail closed。"""

    manifest = load_p9_runtime_manifest(RUNTIME_MANIFEST)
    unknown_case = deepcopy(manifest)
    unknown_case["p9_protocol"]["headline"]["H3"]["primary_classes"][0][
        "case_id"
    ] = "missing"
    with pytest.raises(ValueError, match="unknown case"):
        validate_p9_runtime_manifest(unknown_case)

    invalid_sum = deepcopy(manifest)
    invalid_sum["p9_protocol"]["headline"]["H3"]["primary_classes"][0][
        "weight"
    ] = 0.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_p9_runtime_manifest(invalid_sum)


def test_p9_quality_protocol_rejects_post_result_selection_or_bad_revision() -> None:
    """选样时序和 source revision 是最终质量 claim 的硬门禁。"""

    protocol = load_p9_quality_protocol(QUALITY_PROTOCOL)
    post_result = deepcopy(protocol)
    post_result["selection"][
        "selection_occurs_before_any_compression_candidate_result"
    ] = False
    # False is syntactically valid but violates the frozen selection policy.
    with pytest.raises(ValueError, match="must be true"):
        validate_p9_quality_protocol(post_result)

    bad_revision = deepcopy(protocol)
    bad_revision["datasets"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="full lowercase Git revision"):
        validate_p9_quality_protocol(bad_revision)

    bad_interval = deepcopy(protocol)
    bad_interval["non_inferiority"]["confidence_interval"] = "point_estimate"
    with pytest.raises(ValueError, match="paired bootstrap"):
        validate_p9_quality_protocol(bad_interval)
