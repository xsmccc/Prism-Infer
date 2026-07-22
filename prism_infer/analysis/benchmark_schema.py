"""P6 系统 benchmark 结果与 workload schema 工具。

本模块不包含模型执行逻辑。它校验带版本且可 JSON 序列化的 benchmark
记录，防止性能脚本静默遗漏环境、输入、正确性、计时、显存或 KV cache 证据。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_infer.analysis.reference_quality import normalize_reference_text
from prism_infer.analysis.schema_constants import (
    DECODE_COMPILE_KV_BOUNDARY,
    DECODE_COMPILE_SUBGRAPH,
    LOWERCASE_HEX_DIGITS,
    RGB_CHANNEL_COUNT,
    SHA256_HEX_LENGTH,
    UINT8_CHANNEL_MAX,
)
from prism_infer.models.qwen3_vl_architecture import VISION_GRID_DIMENSIONS

BENCHMARK_SCHEMA_VERSION = 9
SUPPORTED_BENCHMARK_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
WORKLOAD_SCHEMA_VERSION = 1
SCHEMA_REPRODUCIBLE_WORKLOAD_VERSION = 2
SCHEMA_COMPILE_EVIDENCE_VERSION = 3
SCHEMA_PHYSICAL_KV_LAYOUT_VERSION = 4
SCHEMA_MATERIALIZED_OUTPUT_VERSION = 5
SCHEMA_PACKED_MLP_VERSION = 6
SCHEMA_SCALED_KV_VERSION = 7
SCHEMA_VISION_BACKEND_VERSION = 8
SCHEMA_DYNAMIC_DECODE_TRAJECTORY_VERSION = 9
STAT_KEYS = ("count", "median", "p90", "p99", "min", "max")
TORCH_DTYPE_ELEMENT_BYTES = {
    "torch.bool": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1,
    "torch.float8_e5m2fnuz": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
}


def percentile(values: Sequence[float], fraction: float) -> float:
    """计算非空数值序列的 nearest-rank 分位数。"""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_values(values: Sequence[float]) -> dict[str, int | float]:
    """生成要求的 median/p90/p99/min/max 汇总。"""

    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise ValueError(f"statistics require finite non-negative values, got {numeric}")
    return {
        "count": len(numeric),
        "median": statistics.median(numeric),
        "p90": percentile(numeric, 0.90),
        "p99": percentile(numeric, 0.99),
        "min": min(numeric),
        "max": max(numeric),
    }


def canonical_json_sha256(value: object) -> str:
    """通过稳定 JSON 序列化计算结构化数据的 SHA256。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes_from_metadata(
    *,
    dtype: str,
    shape: Sequence[int],
    path: str,
) -> int:
    """Compute tensor bytes from auditable dtype/shape metadata."""

    element_bytes = TORCH_DTYPE_ELEMENT_BYTES.get(dtype)
    if element_bytes is None:
        raise ValueError(f"{path}.dtype has unsupported element size: {dtype!r}")
    return math.prod(shape) * element_bytes


def load_workload_manifest(path: str | Path) -> dict[str, Any]:
    """加载并校验 deterministic P6 workload manifest。"""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_workload_manifest(manifest)
    return manifest


def _require_mapping(container: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _require_list(container: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}.{key} must be a list")
    return value


def _require_string(container: Mapping[str, Any], key: str, path: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _require_bool(container: Mapping[str, Any], key: str, path: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a bool")
    return value


def _require_int(
    container: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: int = 0,
) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}.{key} must be an int >= {minimum}")
    return value


def _require_number(
    container: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: float = 0.0,
) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{path}.{key} must be finite and >= {minimum}")
    return numeric


def _require_sha256(container: Mapping[str, Any], key: str, path: str) -> str:
    """要求字段为一个小写十六进制 SHA256 摘要。"""

    value = _require_string(container, key, path)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in LOWERCASE_HEX_DIGITS for character in value
    ):
        raise ValueError(f"{path}.{key} must be a lowercase SHA256 digest")
    return value


def _validate_color_image(spec: Mapping[str, Any], path: str) -> None:
    _require_int(spec, "width", path, minimum=1)
    _require_int(spec, "height", path, minimum=1)
    color = _require_list(spec, "color", path)
    if len(color) != RGB_CHANNEL_COUNT:
        raise ValueError(f"{path}.color must have exactly three channels")
    for index, channel in enumerate(color):
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= UINT8_CHANNEL_MAX
        ):
            raise ValueError(f"{path}.color[{index}] must be an int in [0, 255]")


def _validate_file_image(spec: Mapping[str, Any], path: str) -> None:
    """校验固定真实图片的来源与内容身份元数据。"""

    _require_string(spec, "path", path)
    _require_string(spec, "source_url", path)
    _require_sha256(spec, "sha256", path)
    _require_int(spec, "width", path, minimum=1)
    _require_int(spec, "height", path, minimum=1)


def _validate_reference_sources(manifest: Mapping[str, Any]) -> set[str]:
    """校验可复现的任务 reference 来源，并返回 source id 集合。"""

    raw_sources = manifest.get("reference_sources", {})
    if not isinstance(raw_sources, Mapping):
        raise ValueError("manifest.reference_sources must be an object")
    source_ids: set[str] = set()
    for source_id, source in raw_sources.items():
        path = f"manifest.reference_sources.{source_id}"
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("manifest reference source ids must be non-empty strings")
        if not isinstance(source, Mapping):
            raise ValueError(f"{path} must be an object")
        for key in (
            "dataset",
            "split",
            "task",
            "source_url",
            "annotation_file",
            "mirror_url",
            "mirror_revision",
        ):
            _require_string(source, key, path)
        _require_sha256(source, "content_sha256", path)
        source_ids.add(source_id)
    return source_ids


def _validate_task_reference(
    evaluation: Mapping[str, Any],
    path: str,
    *,
    source_ids: set[str] | None = None,
    source_tasks: Mapping[str, str] | None = None,
) -> None:
    """校验一条 caption/free-text 多参考任务定义。"""

    task = _require_string(evaluation, "task", path)
    if task not in ("caption", "free_text_qa"):
        raise ValueError(f"{path}.task must be 'caption' or 'free_text_qa'")
    source_id = _require_string(evaluation, "reference_source", path)
    if source_ids is not None and source_id not in source_ids:
        raise ValueError(f"{path}.reference_source references unknown source {source_id!r}")
    if source_tasks is not None and source_tasks.get(source_id) != task:
        raise ValueError(f"{path}.task does not match reference source {source_id!r}")
    _require_int(evaluation, "image_id", path, minimum=1)
    references = _require_list(evaluation, "references", path)
    if not references:
        raise ValueError(f"{path}.references must not be empty")
    annotation_ids: set[int] = set()
    for reference_index, reference in enumerate(references):
        reference_path = f"{path}.references[{reference_index}]"
        if not isinstance(reference, Mapping):
            raise ValueError(f"{reference_path} must be an object")
        annotation_id = _require_int(
            reference,
            "annotation_id",
            reference_path,
            minimum=1,
        )
        if annotation_id in annotation_ids:
            raise ValueError(f"{path} has duplicate annotation_id {annotation_id}")
        annotation_ids.add(annotation_id)
        text = _require_string(reference, "text", reference_path)
        if not normalize_reference_text(text):
            raise ValueError(f"{reference_path}.text has no normalized tokens")


def validate_workload_manifest(manifest: Mapping[str, Any]) -> None:
    """校验 deterministic synthetic/固定真实 workload manifest contract。"""

    if manifest.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise ValueError(f"unsupported workload schema_version: {manifest.get('schema_version')!r}")
    _require_string(manifest, "name", "manifest")
    reference_source_ids = _validate_reference_sources(manifest)
    reference_source_tasks = {
        source_id: str(source["task"])
        for source_id, source in manifest.get("reference_sources", {}).items()
    }
    cases = _require_list(manifest, "cases", "manifest")
    if not cases:
        raise ValueError("manifest.cases must not be empty")
    _validate_workload_cases(
        cases,
        reference_source_ids=reference_source_ids,
        reference_source_tasks=reference_source_tasks,
    )


def _validate_workload_cases(
    cases: list[Any],
    *,
    reference_source_ids: set[str],
    reference_source_tasks: Mapping[str, str],
) -> None:
    case_ids: set[str] = set()
    for case_index, case in enumerate(cases):
        path = f"manifest.cases[{case_index}]"
        if not isinstance(case, Mapping):
            raise ValueError(f"{path} must be an object")
        case_id = _require_string(case, "id", path)
        if case_id in case_ids:
            raise ValueError(f"duplicate workload case id: {case_id!r}")
        case_ids.add(case_id)
        _validate_workload_case_requests(
            case,
            path,
            reference_source_ids=reference_source_ids,
            reference_source_tasks=reference_source_tasks,
        )


def _validate_workload_case_requests(
    case: Mapping[str, Any],
    path: str,
    *,
    reference_source_ids: set[str],
    reference_source_tasks: Mapping[str, str],
) -> None:
    requests = _require_list(case, "requests", path)
    if not requests:
        raise ValueError(f"{path}.requests must not be empty")
    for request_index, request in enumerate(requests):
        request_path = f"{path}.requests[{request_index}]"
        if not isinstance(request, Mapping):
            raise ValueError(f"{request_path} must be an object")
        _validate_workload_request(
            request,
            request_path,
            reference_source_ids=reference_source_ids,
            reference_source_tasks=reference_source_tasks,
        )


def _validate_workload_request(
    request: Mapping[str, Any],
    path: str,
    *,
    reference_source_ids: set[str],
    reference_source_tasks: Mapping[str, str],
) -> None:
    request_type = _require_string(request, "type", path)
    _require_string(request, "prompt", path)
    _validate_optional_request_evaluation(
        request.get("evaluation"),
        path,
        reference_source_ids=reference_source_ids,
        reference_source_tasks=reference_source_tasks,
    )
    _validate_workload_request_payload(request, request_type, path)


def _validate_optional_request_evaluation(
    evaluation: object,
    request_path: str,
    *,
    reference_source_ids: set[str],
    reference_source_tasks: Mapping[str, str],
) -> None:
    if evaluation is None:
        return
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{request_path}.evaluation must be an object")
    _validate_task_reference(
        evaluation,
        f"{request_path}.evaluation",
        source_ids=reference_source_ids,
        source_tasks=reference_source_tasks,
    )


def _validate_workload_request_payload(
    request: Mapping[str, Any],
    request_type: str,
    path: str,
) -> None:
    if request_type == "text":
        return
    if request_type in ("image", "image_file"):
        image = _require_mapping(request, "image", path)
        validator = _validate_color_image if request_type == "image" else _validate_file_image
        validator(image, f"{path}.image")
        return
    if request_type == "images":
        _validate_color_image_list(request, "images", path)
        return
    if request_type == "video":
        _validate_color_image_list(request, "frames", path)
        return
    raise ValueError(f"unsupported request type: {request_type!r}")


def _validate_color_image_list(
    request: Mapping[str, Any],
    key: str,
    path: str,
) -> None:
    images = _require_list(request, key, path)
    if not images:
        raise ValueError(f"{path}.{key} must not be empty")
    for index, image in enumerate(images):
        image_path = f"{path}.{key}[{index}]"
        if not isinstance(image, Mapping):
            raise ValueError(f"{image_path} must be an object")
        _validate_color_image(image, image_path)


def _validate_stats(stats: Mapping[str, Any], path: str) -> None:
    values: dict[str, float] = {}
    for key in STAT_KEYS:
        if key == "count":
            _require_int(stats, key, path, minimum=1)
        else:
            values[key] = _require_number(stats, key, path)
    ordered = [
        values["min"],
        values["median"],
        values["p90"],
        values["p99"],
        values["max"],
    ]
    if ordered != sorted(ordered):
        raise ValueError(f"{path} must satisfy min <= median <= p90 <= p99 <= max")


def _validate_input_shapes(shapes: list[Any], expected_requests: int) -> None:
    """按 [height, width, channels] 校验每个请求的视觉输入 shape。"""

    if len(shapes) != expected_requests:
        raise ValueError("record.workload.input_shapes length must match workload.num_requests")
    for request_index, request_shape in enumerate(shapes):
        path = f"record.workload.input_shapes[{request_index}]"
        if not isinstance(request_shape, Mapping):
            raise ValueError(f"{path} must be an object")
        _require_string(request_shape, "type", path)
        visual_shapes = _require_list(request_shape, "visual_shapes", path)
        for visual_index, visual_shape in enumerate(visual_shapes):
            visual_path = f"{path}.visual_shapes[{visual_index}]"
            if (
                not isinstance(visual_shape, list)
                or len(visual_shape) != VISION_GRID_DIMENSIONS
                or not all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension >= 1
                    for dimension in visual_shape
                )
            ):
                raise ValueError(f"{visual_path} must be [height, width, channels] positive ints")


def validate_benchmark_record(record: Mapping[str, Any]) -> None:
    """校验一条 P6 系统 benchmark JSONL 记录。"""

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_BENCHMARK_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported benchmark schema_version: {schema_version!r}")
    if record.get("record_type") != "system_benchmark":
        raise ValueError("record_type must be 'system_benchmark'")
    _require_string(record, "timestamp_utc", "record")

    _validate_benchmark_environment(record)
    _validate_benchmark_model(record, schema_version)
    _validate_benchmark_mode(record)

    num_requests = _validate_benchmark_workload(record, schema_version)

    batch_size = _validate_benchmark_traffic(record, num_requests)
    if schema_version >= SCHEMA_REPRODUCIBLE_WORKLOAD_VERSION:
        _validate_execution_backend(record, schema_version, batch_size)

    _validate_benchmark_measurement(record)
    _validate_benchmark_correctness(record, schema_version, num_requests)
    _validate_benchmark_stat_groups(record)

    _validate_benchmark_kv_cache(record, schema_version, num_requests)


def _validate_benchmark_environment(record: Mapping[str, Any]) -> None:
    environment = _require_mapping(record, "environment", "record")
    for key in ("git_commit", "python", "torch", "transformers", "gpu"):
        _require_string(environment, key, "record.environment")
    _require_bool(environment, "git_dirty", "record.environment")
    if "cuda" not in environment:
        raise ValueError("record.environment.cuda is required")


def _validate_benchmark_model(record: Mapping[str, Any], schema_version: int) -> None:
    model = _require_mapping(record, "model", "record")
    for key in ("path", "dtype"):
        _require_string(model, key, "record.model")
    for key in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kvcache_block_size",
        "num_kvcache_blocks",
    ):
        _require_int(model, key, "record.model", minimum=1)
    _require_number(model, "gpu_memory_utilization", "record.model")
    if "prefix_caching_enabled" in model:
        _require_bool(model, "prefix_caching_enabled", "record.model")
    if schema_version >= SCHEMA_PACKED_MLP_VERSION:
        _validate_mlp_projection_mode(model)
    if "logits_precision" in model:
        logits_precision = _require_string(model, "logits_precision", "record.model")
        if logits_precision not in ("fp32", "model", "selective_fp32"):
            raise ValueError("record.model.logits_precision is unsupported")
    if "paged_decode_block_n" in model:
        block_n = _require_int(model, "paged_decode_block_n", "record.model", minimum=1)
        if block_n not in (16, 32, 64, 128, 256):
            raise ValueError("record.model.paged_decode_block_n is unsupported")


def _validate_mlp_projection_mode(model: Mapping[str, Any]) -> None:
    projection_mode = _require_string(model, "mlp_projection_mode", "record.model")
    if projection_mode not in ("legacy", "packed"):
        raise ValueError("record.model.mlp_projection_mode must be 'legacy' or 'packed'")


def _validate_benchmark_mode(record: Mapping[str, Any]) -> None:
    mode = _require_mapping(record, "mode", "record")
    for key in (
        "name",
        "execution",
        "attention",
        "compression",
        "visual_pruning_strategy",
    ):
        _require_string(mode, key, "record.mode")
    _require_number(mode, "visual_pruning_keep_ratio", "record.mode")
    _require_int(mode, "visual_pruning_min_keep_tokens", "record.mode")
    if "logits_precision" in mode:
        _require_string(mode, "logits_precision", "record.mode")
    if "paged_decode_block_n" in mode:
        _require_int(mode, "paged_decode_block_n", "record.mode", minimum=1)


def _validate_benchmark_workload(record: Mapping[str, Any], schema_version: int) -> int:
    workload = _require_mapping(record, "workload", "record")
    for key in ("manifest_name", "case_id"):
        _require_string(workload, key, "record.workload")
    _require_sha256(workload, "manifest_sha256", "record.workload")
    request_types = _require_list(workload, "request_types", "record.workload")
    _validate_request_types(request_types)
    for key in (
        "num_requests",
        "prompt_tokens",
        "image_tokens",
        "video_tokens",
        "image_count",
        "video_count",
        "video_frame_count",
        "max_tokens",
    ):
        _require_int(workload, key, "record.workload", minimum=0)
    num_requests = _require_int(workload, "num_requests", "record.workload", minimum=1)
    if schema_version >= SCHEMA_REPRODUCIBLE_WORKLOAD_VERSION:
        _validate_workload_replication(workload, num_requests)
    if len(request_types) != num_requests:
        raise ValueError("record.workload.request_types length must match workload.num_requests")
    _validate_input_shapes(
        _require_list(workload, "input_shapes", "record.workload"),
        num_requests,
    )
    _require_bool(workload, "preprocessing_included_in_e2e", "record.workload")
    if schema_version >= SCHEMA_MATERIALIZED_OUTPUT_VERSION:
        _validate_materialized_workload(workload, num_requests)
    return num_requests


def _validate_request_types(request_types: list[Any]) -> None:
    if not request_types or not all(
        isinstance(request_type, str) and request_type for request_type in request_types
    ):
        raise ValueError("record.workload.request_types must contain non-empty strings")


def _validate_workload_replication(
    workload: Mapping[str, Any],
    num_requests: int,
) -> None:
    source_num_requests = _require_int(
        workload,
        "source_num_requests",
        "record.workload",
        minimum=1,
    )
    replication_factor = _require_int(
        workload,
        "request_replication_factor",
        "record.workload",
        minimum=1,
    )
    if num_requests != source_num_requests * replication_factor:
        raise ValueError(
            "record.workload.num_requests must equal source_num_requests * "
            "request_replication_factor"
        )


def _validate_materialized_workload(
    workload: Mapping[str, Any],
    num_requests: int,
) -> None:
    if _require_bool(workload, "output_decoding_included_in_e2e", "record.workload"):
        raise ValueError("schema-v5 benchmark output decoding must remain outside E2E timing")
    reference_sources = _require_mapping(workload, "reference_sources", "record.workload")
    reference_source_tasks = _validate_benchmark_reference_sources(reference_sources)
    task_references = _require_list(workload, "task_references", "record.workload")
    if len(task_references) != num_requests:
        raise ValueError("record.workload.task_references length must match workload.num_requests")
    _validate_benchmark_task_references(
        task_references,
        reference_source_tasks=reference_source_tasks,
    )


def _validate_benchmark_reference_sources(
    reference_sources: Mapping[str, Any],
) -> dict[str, str]:
    source_tasks: dict[str, str] = {}
    for source_id, source in reference_sources.items():
        path = f"record.workload.reference_sources.{source_id}"
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("record.workload reference source ids must be non-empty strings")
        if not isinstance(source, Mapping):
            raise ValueError(f"{path} must be an object")
        for key in (
            "dataset",
            "split",
            "task",
            "source_url",
            "annotation_file",
            "mirror_url",
            "mirror_revision",
        ):
            _require_string(source, key, path)
        _require_sha256(source, "content_sha256", path)
        source_tasks[source_id] = str(source["task"])
    return source_tasks


def _validate_benchmark_task_references(
    task_references: list[Any],
    *,
    reference_source_tasks: Mapping[str, str],
) -> None:
    source_ids = set(reference_source_tasks)
    for request_index, task_reference in enumerate(task_references):
        if task_reference is None:
            continue
        path = f"record.workload.task_references[{request_index}]"
        if not isinstance(task_reference, Mapping):
            raise ValueError(f"{path} must be an object or null")
        _validate_task_reference(
            task_reference,
            path,
            source_ids=source_ids,
            source_tasks=reference_source_tasks,
        )


def _validate_benchmark_traffic(record: Mapping[str, Any], num_requests: int) -> int:
    traffic = _require_mapping(record, "traffic", "record")
    if _require_string(traffic, "kind", "record.traffic") != "offline_closed_loop":
        raise ValueError("record.traffic.kind must be 'offline_closed_loop'")
    batch_size = _require_int(traffic, "batch_size", "record.traffic", minimum=1)
    concurrency = _require_int(traffic, "concurrency", "record.traffic", minimum=1)
    if batch_size != num_requests or concurrency != num_requests:
        raise ValueError("offline traffic batch_size/concurrency must match workload.num_requests")
    if traffic.get("request_rate_per_s") is not None:
        _require_number(traffic, "request_rate_per_s", "record.traffic")
    return batch_size


def _validate_execution_backend(
    record: Mapping[str, Any],
    schema_version: int,
    batch_size: int,
) -> None:
    execution = _require_mapping(record, "execution_backend", "record")
    path = "record.execution_backend"
    if schema_version >= SCHEMA_VISION_BACKEND_VERSION:
        vision_backend = _require_string(execution, "vision_attention_backend", path)
        if vision_backend not in ("sdpa", "flash_attn"):
            raise ValueError(
                "record.execution_backend.vision_attention_backend must be 'sdpa' or 'flash_attn'"
            )
    prefill_backend = _require_string(execution, "prefill_backend", path)
    decode_backend = _require_string(execution, "decode_backend", path)
    if "paged_decode_block_n" in execution:
        _require_int(execution, "paged_decode_block_n", path, minimum=1)
    graph_enabled = _require_bool(execution, "cuda_graph_enabled", path)
    capture_scope = _require_string(execution, "cuda_graph_capture_scope", path)
    capture_ms = _require_number(execution, "cuda_graph_capture_ms", path)
    graph_batch_sizes = _require_list(execution, "cuda_graph_batch_sizes", path)
    _validate_graph_batch_sizes(graph_batch_sizes)
    requested_batch_size, selected_batch_size, batch_padding = _read_decode_batch_selection(
        execution
    )
    _validate_decode_batch_selection(
        prefill_backend=prefill_backend,
        traffic_batch_size=batch_size,
        requested_batch_size=requested_batch_size,
        selected_batch_size=selected_batch_size,
        batch_padding=batch_padding,
    )
    _validate_cuda_graph_state(
        schema_version=schema_version,
        decode_backend=decode_backend,
        graph_enabled=graph_enabled,
        capture_scope=capture_scope,
        capture_ms=capture_ms,
        graph_batch_sizes=graph_batch_sizes,
        selected_batch_size=selected_batch_size,
        batch_padding=batch_padding,
    )
    if schema_version >= SCHEMA_DYNAMIC_DECODE_TRAJECTORY_VERSION:
        _validate_dynamic_decode_trajectory(
            record,
            execution,
            graph_enabled=graph_enabled,
            graph_batch_sizes=graph_batch_sizes,
            requested_batch_size=requested_batch_size,
        )
    if schema_version >= SCHEMA_COMPILE_EVIDENCE_VERSION:
        _validate_compile_state(execution, decode_backend, graph_enabled)


def _validate_graph_batch_sizes(graph_batch_sizes: list[Any]) -> None:
    if not all(
        isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size >= 1
        for batch_size in graph_batch_sizes
    ):
        raise ValueError(
            "record.execution_backend.cuda_graph_batch_sizes must contain positive ints"
        )
    if graph_batch_sizes != sorted(set(graph_batch_sizes)):
        raise ValueError(
            "record.execution_backend.cuda_graph_batch_sizes must be sorted and unique"
        )


def _read_decode_batch_selection(execution: Mapping[str, Any]) -> tuple[int, int, int]:
    path = "record.execution_backend"
    return (
        _require_int(execution, "requested_decode_batch_size", path, minimum=1),
        _require_int(execution, "selected_decode_batch_size", path, minimum=1),
        _require_int(execution, "decode_batch_padding", path, minimum=0),
    )


def _validate_decode_batch_selection(
    *,
    prefill_backend: str,
    traffic_batch_size: int,
    requested_batch_size: int,
    selected_batch_size: int,
    batch_padding: int,
) -> None:
    if prefill_backend != "eager":
        raise ValueError("record.execution_backend.prefill_backend must be 'eager'")
    if requested_batch_size != traffic_batch_size:
        raise ValueError(
            "record.execution_backend.requested_decode_batch_size must match traffic.batch_size"
        )
    if selected_batch_size < requested_batch_size:
        raise ValueError(
            "record.execution_backend.selected_decode_batch_size must be >= "
            "requested_decode_batch_size"
        )
    if batch_padding != selected_batch_size - requested_batch_size:
        raise ValueError(
            "record.execution_backend.decode_batch_padding does not match "
            "selected-requested batch size"
        )


def _validate_cuda_graph_state(
    *,
    schema_version: int,
    decode_backend: str,
    graph_enabled: bool,
    capture_scope: str,
    capture_ms: float,
    graph_batch_sizes: list[Any],
    selected_batch_size: int,
    batch_padding: int,
) -> None:
    if graph_enabled:
        _validate_enabled_cuda_graph(
            decode_backend,
            capture_scope,
            graph_batch_sizes,
            selected_batch_size,
        )
        return
    allowed_backends = (
        ("eager", "torch_compile_attention")
        if schema_version >= SCHEMA_COMPILE_EVIDENCE_VERSION
        else ("eager",)
    )
    if (
        decode_backend not in allowed_backends
        or capture_scope != "none"
        or capture_ms != 0.0
        or graph_batch_sizes
        or batch_padding != 0
    ):
        raise ValueError("eager execution must not report CUDA Graph capture state")


def _validate_enabled_cuda_graph(
    decode_backend: str,
    capture_scope: str,
    graph_batch_sizes: list[Any],
    selected_batch_size: int,
) -> None:
    if decode_backend != "cuda_graph":
        raise ValueError("CUDA Graph execution requires decode_backend='cuda_graph'")
    if capture_scope != "decode_model_forward":
        raise ValueError("CUDA Graph execution requires capture scope 'decode_model_forward'")
    if selected_batch_size not in graph_batch_sizes:
        raise ValueError("selected CUDA Graph batch size is absent from captured sizes")


def _validate_dynamic_decode_trajectory(
    record: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    graph_enabled: bool,
    graph_batch_sizes: list[Any],
    requested_batch_size: int,
) -> None:
    path = "record.execution_backend"
    decode_counts = _read_decode_batch_counts(
        execution,
        requested_batch_size=requested_batch_size,
    )
    timing = _require_mapping(record, "timing_ms", "record")
    decode_timing = _require_mapping(timing, "decode_step", "record.timing_ms")
    expected_steps = _require_int(
        decode_timing,
        "count",
        "record.timing_ms.decode_step",
        minimum=1,
    )
    if sum(decode_counts.values()) != expected_steps:
        raise ValueError(f"{path}.decode_batch_size_counts must cover every measured decode step")
    _validate_graph_replay_counts(
        execution,
        graph_enabled=graph_enabled,
        graph_batch_sizes=graph_batch_sizes,
        decode_counts=decode_counts,
    )


def _read_decode_batch_counts(
    execution: Mapping[str, Any],
    *,
    requested_batch_size: int,
) -> dict[int, int]:
    path = "record.execution_backend"
    decode_entries = _require_list(execution, "decode_batch_size_counts", path)
    decode_counts: dict[int, int] = {}
    for index, entry in enumerate(decode_entries):
        entry_path = f"{path}.decode_batch_size_counts[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{entry_path} must be an object")
        actual = _require_int(entry, "actual_batch_size", entry_path, minimum=1)
        count = _require_int(entry, "count", entry_path, minimum=1)
        if actual > requested_batch_size:
            raise ValueError("observed decode batch exceeds the requested traffic batch")
        if actual in decode_counts:
            raise ValueError(f"{path}.decode_batch_size_counts contains duplicate batches")
        decode_counts[actual] = count
    if not decode_counts:
        raise ValueError(f"{path}.decode_batch_size_counts must not be empty")
    return decode_counts


def _validate_graph_replay_counts(
    execution: Mapping[str, Any],
    *,
    graph_enabled: bool,
    graph_batch_sizes: list[Any],
    decode_counts: Mapping[int, int],
) -> None:
    path = "record.execution_backend"
    replay_entries = _require_list(execution, "cuda_graph_replay_counts", path)
    if not graph_enabled:
        if replay_entries:
            raise ValueError("non-Graph execution must report empty CUDA Graph replay counts")
        return
    replay_counts: dict[tuple[int, int], int] = {}
    projected_counts: dict[int, int] = {}
    for index, entry in enumerate(replay_entries):
        entry_path = f"{path}.cuda_graph_replay_counts[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{entry_path} must be an object")
        actual = _require_int(entry, "actual_batch_size", entry_path, minimum=1)
        captured = _require_int(entry, "captured_batch_size", entry_path, minimum=1)
        count = _require_int(entry, "count", entry_path, minimum=1)
        pair = (actual, captured)
        if pair in replay_counts:
            raise ValueError(f"{path}.cuda_graph_replay_counts contains duplicate pairs")
        if captured < actual or captured not in graph_batch_sizes:
            raise ValueError("CUDA Graph replay count reports an invalid captured batch size")
        replay_counts[pair] = count
        projected_counts[actual] = projected_counts.get(actual, 0) + count
    if projected_counts != decode_counts:
        raise ValueError("CUDA Graph replay counts must match observed decode batch counts")


def _validate_compile_state(
    execution: Mapping[str, Any],
    decode_backend: str,
    graph_enabled: bool,
) -> None:
    path = "record.execution_backend"
    compile_enabled = _require_bool(execution, "torch_compile_enabled", path)
    compile_region = _require_string(execution, "torch_compile_region", path)
    compile_subgraph, compile_kv_boundary = _read_compile_boundaries(execution)
    compile_backend = _require_string(execution, "torch_compile_backend", path)
    compile_mode = _require_string(execution, "torch_compile_mode", path)
    emulate_precision_casts = _require_bool(
        execution, "torch_compile_emulate_precision_casts", path
    )
    force_same_precision = _require_bool(execution, "torch_compile_force_same_precision", path)
    first_call_ms = _require_number(execution, "torch_compile_first_call_ms", path)
    if compile_enabled:
        _validate_enabled_compile_state(
            graph_enabled=graph_enabled,
            decode_backend=decode_backend,
            compile_region=compile_region,
            compile_subgraph=compile_subgraph,
            compile_kv_boundary=compile_kv_boundary,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            first_call_ms=first_call_ms,
        )
        return
    if (
        compile_region != "none"
        or compile_backend != "none"
        or compile_mode != "none"
        or emulate_precision_casts
        or force_same_precision
        or first_call_ms != 0.0
        or (
            compile_subgraph is not None
            and (compile_subgraph != "none" or compile_kv_boundary != "none")
        )
    ):
        raise ValueError("disabled torch.compile execution must report empty state")


def _read_compile_boundaries(
    execution: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    compile_subgraph = execution.get("torch_compile_subgraph")
    compile_kv_boundary = execution.get("torch_compile_kv_cache_boundary")
    if (compile_subgraph is None) != (compile_kv_boundary is None):
        raise ValueError(
            "record.execution_backend must report both torch.compile "
            "subgraph and KV-cache boundary metadata"
        )
    if compile_subgraph is not None and (
        not isinstance(compile_subgraph, str) or not isinstance(compile_kv_boundary, str)
    ):
        raise TypeError("torch.compile boundary metadata must use strings")
    return compile_subgraph, compile_kv_boundary


def _validate_enabled_compile_state(
    *,
    graph_enabled: bool,
    decode_backend: str,
    compile_region: str,
    compile_subgraph: str | None,
    compile_kv_boundary: str | None,
    compile_backend: str,
    compile_mode: str,
    first_call_ms: float,
) -> None:
    if graph_enabled or decode_backend != "torch_compile_attention":
        raise ValueError(
            "torch.compile execution must be graph-disabled and use "
            "decode_backend='torch_compile_attention'"
        )
    if (
        compile_region != "decode_attention"
        or compile_backend != "inductor"
        or compile_mode not in ("default", "reduce-overhead")
        or first_call_ms <= 0.0
    ):
        raise ValueError("torch.compile execution metadata is invalid")
    if compile_subgraph is not None and (
        compile_subgraph != DECODE_COMPILE_SUBGRAPH
        or compile_kv_boundary != DECODE_COMPILE_KV_BOUNDARY
    ):
        raise ValueError("torch.compile subgraph boundary metadata is invalid")


def _validate_benchmark_measurement(record: Mapping[str, Any]) -> None:
    measurement = _require_mapping(record, "measurement", "record")
    _require_int(measurement, "warmup", "record.measurement", minimum=0)
    _require_int(measurement, "repeat", "record.measurement", minimum=1)
    _require_bool(measurement, "cuda_synchronize_timing", "record.measurement")


def _validate_benchmark_correctness(
    record: Mapping[str, Any],
    schema_version: int,
    num_requests: int,
) -> None:
    correctness = _require_mapping(record, "correctness", "record")
    _require_bool(
        correctness,
        "outputs_identical_across_repeats",
        "record.correctness",
    )
    token_ids = _require_list(correctness, "token_ids", "record.correctness")
    _validate_output_token_ids(token_ids, num_requests)
    output_tokens = _require_int(
        correctness,
        "output_tokens",
        "record.correctness",
        minimum=1,
    )
    if output_tokens != sum(len(request_tokens) for request_tokens in token_ids):
        raise ValueError("record.correctness.output_tokens must equal the token_ids length sum")
    output_sha256 = _require_sha256(correctness, "output_sha256", "record.correctness")
    if output_sha256 != canonical_json_sha256(token_ids):
        raise ValueError("record.correctness.output_sha256 does not match token_ids")
    if schema_version >= SCHEMA_MATERIALIZED_OUTPUT_VERSION:
        _validate_decoded_texts(correctness, num_requests)


def _validate_output_token_ids(token_ids: list[Any], num_requests: int) -> None:
    valid = token_ids and all(
        isinstance(request_tokens, list)
        and all(
            isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
            for token_id in request_tokens
        )
        for request_tokens in token_ids
    )
    if not valid:
        raise ValueError("record.correctness.token_ids must contain non-negative integer lists")
    if len(token_ids) != num_requests:
        raise ValueError("record.correctness.token_ids length must match workload.num_requests")


def _validate_decoded_texts(correctness: Mapping[str, Any], num_requests: int) -> None:
    decoded_texts = _require_list(correctness, "decoded_texts", "record.correctness")
    if len(decoded_texts) != num_requests or not all(
        isinstance(text, str) for text in decoded_texts
    ):
        raise ValueError("record.correctness.decoded_texts must contain one string per request")
    decoded_hash = _require_sha256(
        correctness,
        "decoded_texts_sha256",
        "record.correctness",
    )
    if decoded_hash != canonical_json_sha256(decoded_texts):
        raise ValueError("record.correctness.decoded_texts_sha256 does not match decoded_texts")


def _validate_benchmark_stat_groups(record: Mapping[str, Any]) -> None:
    _validate_named_stats(
        record,
        "timing_ms",
        (
            "preprocessing",
            "engine_ttft",
            "end_to_end_ttft",
            "prefill",
            "decode_step",
            "end_to_end",
        ),
    )
    _validate_named_stats(
        record,
        "throughput",
        (
            "engine_output_tokens_per_s",
            "e2e_output_tokens_per_s",
            "decode_tokens_per_s",
            "engine_requests_per_s",
            "e2e_requests_per_s",
        ),
    )
    _validate_named_stats(record, "memory_mb", ("allocated", "reserved", "peak_allocated"))


def _validate_named_stats(
    record: Mapping[str, Any],
    section_name: str,
    keys: tuple[str, ...],
) -> None:
    section = _require_mapping(record, section_name, "record")
    section_path = f"record.{section_name}"
    for key in keys:
        _validate_stats(_require_mapping(section, key, section_path), f"{section_path}.{key}")


@dataclass(frozen=True, slots=True)
class _PromptKVAccounting:
    dense_blocks: int
    active_blocks: int
    dense_bytes: int
    active_bytes: int


def _validate_benchmark_kv_cache(
    record: Mapping[str, Any],
    schema_version: int,
    num_requests: int,
) -> None:
    kv_cache = _require_mapping(record, "kv_cache", "record")
    kv_dtype = _require_string(kv_cache, "dtype", "record.kv_cache")
    shape = _require_list(kv_cache, "shape", "record.kv_cache")
    if not shape or not all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0
        for dimension in shape
    ):
        raise ValueError("record.kv_cache.shape must contain non-negative ints")
    for key in ("bytes", "blocks", "block_size", "capacity_tokens"):
        _require_int(kv_cache, key, "record.kv_cache", minimum=1)
    if kv_cache["capacity_tokens"] != kv_cache["blocks"] * kv_cache["block_size"]:
        raise ValueError("record.kv_cache.capacity_tokens must equal blocks * block_size")
    if schema_version >= SCHEMA_PHYSICAL_KV_LAYOUT_VERSION:
        _validate_physical_kv_cache(
            kv_cache,
            schema_version=schema_version,
            num_requests=num_requests,
            kv_dtype=kv_dtype,
            shape=shape,
        )


def _validate_physical_kv_cache(
    kv_cache: Mapping[str, Any],
    *,
    schema_version: int,
    num_requests: int,
    kv_dtype: str,
    shape: list[Any],
) -> None:
    accounting = _read_prompt_kv_accounting(kv_cache)
    if schema_version >= SCHEMA_SCALED_KV_VERSION:
        _validate_scaled_kv_metadata(
            kv_cache,
            kv_dtype=kv_dtype,
            shape=shape,
            accounting=accounting,
        )
    _validate_kv_layout_records(kv_cache, num_requests)


def _read_prompt_kv_accounting(kv_cache: Mapping[str, Any]) -> _PromptKVAccounting:
    path = "record.kv_cache"
    logical_tokens = _require_int(kv_cache, "logical_prompt_tokens", path, minimum=1)
    physical_tokens = _require_int(kv_cache, "physical_prompt_tokens", path, minimum=1)
    dense_blocks = _require_int(kv_cache, "dense_prompt_blocks", path, minimum=1)
    active_blocks = _require_int(kv_cache, "active_prompt_blocks", path, minimum=1)
    released_blocks = _require_int(kv_cache, "released_prompt_blocks", path, minimum=0)
    dense_bytes = _require_int(kv_cache, "dense_prompt_bytes", path, minimum=1)
    active_bytes = _require_int(kv_cache, "active_prompt_bytes", path, minimum=1)
    if physical_tokens > logical_tokens:
        raise ValueError("physical prompt KV tokens cannot exceed logical prompt tokens")
    if active_blocks > dense_blocks:
        raise ValueError("active prompt blocks cannot exceed dense prompt blocks")
    if released_blocks != dense_blocks - active_blocks:
        raise ValueError("released prompt blocks must equal dense-active blocks")
    _validate_active_bytes_from_dense(
        dense_bytes,
        active_bytes,
        dense_blocks,
        active_blocks,
        component="prompt",
    )
    return _PromptKVAccounting(dense_blocks, active_blocks, dense_bytes, active_bytes)


def _validate_active_bytes_from_dense(
    dense_bytes: int,
    active_bytes: int,
    dense_blocks: int,
    active_blocks: int,
    *,
    component: str,
) -> None:
    if dense_bytes % dense_blocks != 0:
        raise ValueError(f"dense {component} bytes must be divisible by dense blocks")
    if active_bytes != active_blocks * (dense_bytes // dense_blocks):
        raise ValueError(f"active {component} bytes do not match active blocks")


def _validate_scaled_kv_metadata(
    kv_cache: Mapping[str, Any],
    *,
    kv_dtype: str,
    shape: list[Any],
    accounting: _PromptKVAccounting,
) -> None:
    path = "record.kv_cache"
    scale_dtype = _require_string(kv_cache, "scale_dtype", path)
    scale_shape = _require_list(kv_cache, "scale_shape", path)
    _validate_non_negative_shape(scale_shape, "record.kv_cache.scale_shape")
    payload_bytes = _require_int(kv_cache, "payload_bytes", path, minimum=1)
    scale_bytes = _require_int(kv_cache, "scale_bytes", path, minimum=0)
    if kv_cache["bytes"] != payload_bytes + scale_bytes:
        raise ValueError("record.kv_cache.bytes must equal payload_bytes + scale_bytes")
    expected_payload_bytes = _tensor_bytes_from_metadata(
        dtype=kv_dtype,
        shape=shape,
        path=path,
    )
    if payload_bytes != expected_payload_bytes:
        raise ValueError("record.kv_cache.payload_bytes does not match dtype and shape")
    _validate_scale_allocation(
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
        scale_bytes=scale_bytes,
        payload_shape=shape,
    )
    _validate_prompt_component_accounting(kv_cache, accounting)


def _validate_non_negative_shape(shape: list[Any], path: str) -> None:
    if not all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0
        for dimension in shape
    ):
        raise ValueError(f"{path} must contain non-negative ints")


def _validate_scale_allocation(
    *,
    scale_dtype: str,
    scale_shape: list[Any],
    scale_bytes: int,
    payload_shape: list[Any],
) -> None:
    if scale_bytes == 0:
        if scale_dtype != "none" or scale_shape:
            raise ValueError("zero scale bytes require scale_dtype='none' and empty scale_shape")
        return
    if scale_dtype == "none" or not scale_shape:
        raise ValueError("non-zero scale bytes require a scale dtype and shape")
    if scale_shape != payload_shape[:-1]:
        raise ValueError(
            "record.kv_cache.scale_shape must equal payload shape without the head dimension"
        )
    expected_scale_bytes = _tensor_bytes_from_metadata(
        dtype=scale_dtype,
        shape=scale_shape,
        path="record.kv_cache.scale",
    )
    if scale_bytes != expected_scale_bytes:
        raise ValueError("record.kv_cache.scale_bytes does not match dtype and shape")


def _validate_prompt_component_accounting(
    kv_cache: Mapping[str, Any],
    accounting: _PromptKVAccounting,
) -> None:
    path = "record.kv_cache"
    dense_payload = _require_int(kv_cache, "dense_prompt_payload_bytes", path, minimum=1)
    dense_scale = _require_int(kv_cache, "dense_prompt_scale_bytes", path, minimum=0)
    active_payload = _require_int(kv_cache, "active_prompt_payload_bytes", path, minimum=1)
    active_scale = _require_int(kv_cache, "active_prompt_scale_bytes", path, minimum=0)
    if accounting.dense_bytes != dense_payload + dense_scale:
        raise ValueError("dense prompt bytes must equal payload + scale bytes")
    if accounting.active_bytes != active_payload + active_scale:
        raise ValueError("active prompt bytes must equal payload + scale bytes")
    _validate_active_bytes_from_dense(
        dense_payload,
        active_payload,
        accounting.dense_blocks,
        accounting.active_blocks,
        component="prompt payload",
    )
    _validate_active_bytes_from_dense(
        dense_scale,
        active_scale,
        accounting.dense_blocks,
        accounting.active_blocks,
        component="prompt scale",
    )


def _validate_kv_layout_records(kv_cache: Mapping[str, Any], num_requests: int) -> None:
    layouts = _require_list(kv_cache, "layouts", "record.kv_cache")
    if len(layouts) != num_requests:
        raise ValueError("KV layout record count must match workload requests")
    for layout in layouts:
        _validate_kv_layout_record(layout)


def _validate_kv_layout_record(layout: object) -> None:
    path = "record.kv_cache.layouts[]"
    if not isinstance(layout, Mapping):
        raise ValueError("KV layout record must be an object")
    for key in ("mode", "kv_dtype"):
        _require_string(layout, key, path)
    for key in (
        "logical_context_len",
        "physical_kv_len",
        "prompt_logical_len",
        "compressed_prompt_kv_len",
    ):
        _require_int(layout, key, path, minimum=1)
    block_table = _require_list(layout, "block_table", path)
    if not block_table or not all(
        isinstance(block_id, int) and not isinstance(block_id, bool) and block_id >= 0
        for block_id in block_table
    ):
        raise ValueError("KV layout block table must contain non-negative ints")
