"""P9 headline runtime 与标准质量协议校验。

本模块只校验 versioned JSON contract，不执行模型、下载数据或计算指标。把协议
校验放在独立模块中，可防止 benchmark runner 对拼错字段或不完整公平性配置静默
继续运行。
"""

from __future__ import annotations

import json
import math

from prism_infer.analysis.schema_constants import PERCENTAGE_POINTS_SCALE
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prism_infer.analysis.benchmark_schema import validate_workload_manifest


P9_PROTOCOL_SCHEMA_VERSION = 1
GIT_REVISION_HEX_LENGTH = 40
WEIGHT_SUM_ABS_TOLERANCE = 1e-9
QUALITY_MATERIALIZATION_STATUSES = {
    "pending",
    "materialized",
    "conditional_manual_media",
    "excluded",
}


def _mapping(container: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _list(container: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty list")
    return value


def _string(container: Mapping[str, Any], key: str, path: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _bool(container: Mapping[str, Any], key: str, path: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a bool")
    return value


def _positive_int(container: Mapping[str, Any], key: str, path: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive int")
    return value


def _positive_number(container: Mapping[str, Any], key: str, path: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{path}.{key} must be finite and positive")
    return numeric


def _git_revision(container: Mapping[str, Any], key: str, path: str) -> str:
    revision = _string(container, key, path)
    if len(revision) != GIT_REVISION_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{path}.{key} must be a full lowercase Git revision")
    return revision


def _positive_number_list(container: Mapping[str, Any], key: str, path: str) -> tuple[float, ...]:
    values = _list(container, key, path)
    numeric = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}.{key}[{index}] must be numeric")
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{path}.{key}[{index}] must be finite and positive")
        numeric.append(converted)
    if len(set(numeric)) != len(numeric):
        raise ValueError(f"{path}.{key} values must be unique")
    return tuple(numeric)


def _validate_weighted_classes(
    classes: list[Any],
    *,
    case_ids: set[str],
    path: str,
) -> None:
    seen = set()
    total = 0.0
    for index, item in enumerate(classes):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_path} must be an object")
        case_id = _string(item, "case_id", item_path)
        if case_id not in case_ids:
            raise ValueError(f"{item_path}.case_id references unknown case {case_id!r}")
        if case_id in seen:
            raise ValueError(f"{path} repeats case {case_id!r}")
        seen.add(case_id)
        total += _positive_number(item, "weight", item_path)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_ABS_TOLERANCE):
        raise ValueError(f"{path} weights must sum to 1.0, got {total}")


def validate_p9_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    """校验 H1/H2/H3 runtime workload 与公平测量 contract。"""

    validate_workload_manifest(manifest)
    protocol = _mapping(manifest, "p9_protocol", "manifest")
    if protocol.get("schema_version") != P9_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported manifest.p9_protocol.schema_version")

    case_ids = {str(case["id"]) for case in manifest["cases"]}
    _validate_runtime_model(protocol)
    _validate_runtime_sampling(protocol)
    _validate_runtime_memory_contract(protocol)
    _validate_runtime_headlines(protocol, case_ids)
    _validate_runtime_measurement(protocol)


def _validate_runtime_model(protocol: Mapping[str, Any]) -> None:
    model = _mapping(protocol, "model", "manifest.p9_protocol")
    _string(model, "name", "manifest.p9_protocol.model")
    _git_revision(model, "revision", "manifest.p9_protocol.model")
    _string(model, "weight_dtype", "manifest.p9_protocol.model")
    _positive_int(model, "tensor_parallel_size", "manifest.p9_protocol.model")


def _validate_runtime_sampling(protocol: Mapping[str, Any]) -> None:
    sampling = _mapping(protocol, "sampling", "manifest.p9_protocol")
    temperature = sampling.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("manifest.p9_protocol.sampling.temperature must be numeric")
    if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError("manifest.p9_protocol.sampling.temperature must be non-negative")
    _bool(sampling, "ignore_eos", "manifest.p9_protocol.sampling")
    _positive_int(sampling, "seed", "manifest.p9_protocol.sampling")


def _validate_runtime_memory_contract(protocol: Mapping[str, Any]) -> None:
    memory = _mapping(protocol, "memory_contract", "manifest.p9_protocol")
    _string(memory, "gpu_class", "manifest.p9_protocol.memory_contract")
    _positive_int(
        memory,
        "observed_total_memory_bytes",
        "manifest.p9_protocol.memory_contract",
    )
    _positive_int(memory, "kv_pool_bytes", "manifest.p9_protocol.memory_contract")
    _bool(
        memory,
        "include_scale_and_metadata_bytes",
        "manifest.p9_protocol.memory_contract",
    )


def _validate_runtime_headlines(
    protocol: Mapping[str, Any],
    case_ids: set[str],
) -> None:
    headline = _mapping(protocol, "headline", "manifest.p9_protocol")
    for headline_id in ("H1", "H2"):
        _validate_latency_headline(headline, headline_id, case_ids)
    _validate_online_headline(headline, case_ids)


def _validate_latency_headline(
    headline: Mapping[str, Any],
    headline_id: str,
    case_ids: set[str],
) -> None:
    spec = _mapping(headline, headline_id, "manifest.p9_protocol.headline")
    path = f"manifest.p9_protocol.headline.{headline_id}"
    case_id = _string(spec, "case_id", path)
    if case_id not in case_ids:
        raise ValueError(f"headline {headline_id} references unknown case {case_id!r}")
    for key in ("max_tokens", "max_model_len", "fresh_process_repeats"):
        _positive_int(spec, key, path)
    _bool(spec, "required_cross_framework_prompt_token_identity", path)


def _validate_online_headline(headline: Mapping[str, Any], case_ids: set[str]) -> None:
    h3 = _mapping(headline, "H3", "manifest.p9_protocol.headline")
    _validate_weighted_classes(
        _list(h3, "primary_classes", "manifest.p9_protocol.headline.H3"),
        case_ids=case_ids,
        path="manifest.p9_protocol.headline.H3.primary_classes",
    )
    _validate_weighted_classes(
        _list(h3, "conditional_video_classes", "manifest.p9_protocol.headline.H3"),
        case_ids=case_ids,
        path="manifest.p9_protocol.headline.H3.conditional_video_classes",
    )
    _positive_number_list(h3, "request_rates_per_second", "manifest.p9_protocol.headline.H3")
    _positive_int(h3, "completed_requests_per_run", "manifest.p9_protocol.headline.H3")
    arrival_seeds = _list(h3, "arrival_seeds", "manifest.p9_protocol.headline.H3")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in arrival_seeds
    ):
        raise ValueError("manifest.p9_protocol.headline.H3.arrival_seeds must be positive ints")
    if len(set(arrival_seeds)) != len(arrival_seeds):
        raise ValueError("manifest.p9_protocol.headline.H3.arrival_seeds must be unique")
    _string(h3, "ttft_slo_formula", "manifest.p9_protocol.headline.H3")
    _string(h3, "tpot_slo_formula", "manifest.p9_protocol.headline.H3")


def _validate_runtime_measurement(protocol: Mapping[str, Any]) -> None:
    measurement = _mapping(protocol, "measurement", "manifest.p9_protocol")
    improvement = _positive_number(
        measurement,
        "minimum_practical_improvement_fraction",
        "manifest.p9_protocol.measurement",
    )
    if improvement >= 1.0:
        raise ValueError("minimum practical improvement fraction must be < 1")
    _string(
        measurement,
        "confidence_interval",
        "manifest.p9_protocol.measurement",
    )
    _string(measurement, "run_order", "manifest.p9_protocol.measurement")


def validate_p9_quality_protocol(protocol: Mapping[str, Any]) -> None:
    """校验数据 revision、确定性选样与 non-inferiority contract。"""

    if protocol.get("schema_version") != P9_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported quality protocol schema_version")
    _string(protocol, "name", "protocol")
    model = _mapping(protocol, "model", "protocol")
    _git_revision(model, "revision", "protocol.model")
    _git_revision(model, "processor_revision", "protocol.model")
    _validate_quality_non_inferiority(protocol)
    _validate_quality_selection(protocol)
    _validate_quality_datasets(protocol)
    _validate_quality_artifact_contract(protocol)


def _validate_quality_non_inferiority(protocol: Mapping[str, Any]) -> None:
    non_inferiority = _mapping(protocol, "non_inferiority", "protocol")
    accuracy_margin = _positive_number(
        non_inferiority,
        "bounded_accuracy_margin_percentage_points",
        "protocol.non_inferiority",
    )
    if accuracy_margin >= PERCENTAGE_POINTS_SCALE:
        raise ValueError("bounded accuracy margin must be < 100 percentage points")
    normalized_margin = _positive_number(
        non_inferiority,
        "normalized_generation_metric_margin",
        "protocol.non_inferiority",
    )
    if normalized_margin >= 1.0:
        raise ValueError("normalized generation margin must be < 1")
    _positive_int(
        non_inferiority,
        "bootstrap_resamples",
        "protocol.non_inferiority",
    )
    _positive_int(
        non_inferiority,
        "bootstrap_seed",
        "protocol.non_inferiority",
    )
    if (
        _string(
            non_inferiority,
            "confidence_interval",
            "protocol.non_inferiority",
        )
        != "paired_bootstrap_95_percent"
    ):
        raise ValueError("quality confidence interval must be paired bootstrap 95%")
    if (
        _string(
            non_inferiority,
            "eligibility",
            "protocol.non_inferiority",
        )
        != "confidence_interval_lower_bound_within_margin"
    ):
        raise ValueError("quality eligibility rule is unsupported")


def _validate_quality_selection(protocol: Mapping[str, Any]) -> None:
    selection = _mapping(protocol, "selection", "protocol")
    _string(selection, "algorithm", "protocol.selection")
    _positive_int(selection, "seed", "protocol.selection")
    selection_before_results = _bool(
        selection,
        "selection_occurs_before_any_compression_candidate_result",
        "protocol.selection",
    )
    if not selection_before_results:
        raise ValueError(
            "protocol.selection."
            "selection_occurs_before_any_compression_candidate_result must be true"
        )
    _bool(selection, "materialization_requires_media_sha256", "protocol.selection")


def _validate_quality_datasets(protocol: Mapping[str, Any]) -> None:
    dataset_ids = set()
    categories = set()
    for index, dataset in enumerate(_list(protocol, "datasets", "protocol")):
        dataset_id, category = _validate_quality_dataset(dataset, index)
        if dataset_id in dataset_ids:
            raise ValueError(f"duplicate quality dataset id {dataset_id!r}")
        dataset_ids.add(dataset_id)
        categories.add(category)
    required_categories = {
        "single_image_document_ocr",
        "multi_image_reasoning",
        "video_temporal_reasoning",
    }
    if categories != required_categories:
        raise ValueError(
            f"quality datasets must cover exactly the frozen categories, got {sorted(categories)}"
        )


def _validate_quality_dataset(dataset: object, index: int) -> tuple[str, str]:
    path = f"protocol.datasets[{index}]"
    if not isinstance(dataset, Mapping):
        raise ValueError(f"{path} must be an object")
    dataset_id = _string(dataset, "id", path)
    category = _string(dataset, "category", path)
    for key in ("source", "repository", "split", "sample_id_field", "metric"):
        _string(dataset, key, path)
    _git_revision(dataset, "revision", path)
    development_samples = _positive_int(dataset, "development_samples", path)
    final_samples = _positive_int(dataset, "final_samples", path)
    if development_samples > final_samples:
        raise ValueError(f"{path}.development_samples exceeds final_samples")
    status = _string(dataset, "materialization_status", path)
    if status not in QUALITY_MATERIALIZATION_STATUSES:
        raise ValueError(f"{path}.materialization_status is unsupported: {status!r}")
    return dataset_id, category


def _validate_quality_artifact_contract(protocol: Mapping[str, Any]) -> None:
    artifact = _mapping(protocol, "artifact_contract", "protocol")
    for key in (
        "selected_sample_ids_sha256",
        "media_sha256_per_sample",
        "evaluator_commit",
        "raw_predictions",
        "per_sample_scores",
    ):
        _string(artifact, key, "protocol.artifact_contract")
    _bool(artifact, "aggregate_only_is_invalid", "protocol.artifact_contract")


def load_p9_runtime_manifest(path: str | Path) -> dict[str, Any]:
    """读取并校验 P9 runtime manifest。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_p9_runtime_manifest(manifest)
    return manifest


def load_p9_quality_protocol(path: str | Path) -> dict[str, Any]:
    """读取并校验 P9 quality protocol。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_p9_quality_protocol(protocol)
    return protocol
