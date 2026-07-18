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
from pathlib import Path
from typing import Any

from prism_infer.analysis.reference_quality import normalize_reference_text


BENCHMARK_SCHEMA_VERSION = 7
SUPPORTED_BENCHMARK_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7)
WORKLOAD_SCHEMA_VERSION = 1
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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{path}.{key} must be a lowercase SHA256 digest")
    return value


def _validate_color_image(spec: Mapping[str, Any], path: str) -> None:
    _require_int(spec, "width", path, minimum=1)
    _require_int(spec, "height", path, minimum=1)
    color = _require_list(spec, "color", path)
    if len(color) != 3:
        raise ValueError(f"{path}.color must have exactly three channels")
    for index, channel in enumerate(color):
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255:
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
        raise ValueError(
            f"{path}.reference_source references unknown source {source_id!r}"
        )
    if source_tasks is not None and source_tasks.get(source_id) != task:
        raise ValueError(
            f"{path}.task does not match reference source {source_id!r}"
        )
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
        raise ValueError(
            "unsupported workload schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    _require_string(manifest, "name", "manifest")
    reference_source_ids = _validate_reference_sources(manifest)
    reference_source_tasks = {
        source_id: str(source["task"])
        for source_id, source in manifest.get("reference_sources", {}).items()
    }
    cases = _require_list(manifest, "cases", "manifest")
    if not cases:
        raise ValueError("manifest.cases must not be empty")

    case_ids: set[str] = set()
    for case_index, case in enumerate(cases):
        path = f"manifest.cases[{case_index}]"
        if not isinstance(case, Mapping):
            raise ValueError(f"{path} must be an object")
        case_id = _require_string(case, "id", path)
        if case_id in case_ids:
            raise ValueError(f"duplicate workload case id: {case_id!r}")
        case_ids.add(case_id)
        requests = _require_list(case, "requests", path)
        if not requests:
            raise ValueError(f"{path}.requests must not be empty")

        for request_index, request in enumerate(requests):
            request_path = f"{path}.requests[{request_index}]"
            if not isinstance(request, Mapping):
                raise ValueError(f"{request_path} must be an object")
            request_type = _require_string(request, "type", request_path)
            _require_string(request, "prompt", request_path)
            evaluation = request.get("evaluation")
            if evaluation is not None:
                if not isinstance(evaluation, Mapping):
                    raise ValueError(f"{request_path}.evaluation must be an object")
                _validate_task_reference(
                    evaluation,
                    f"{request_path}.evaluation",
                    source_ids=reference_source_ids,
                    source_tasks=reference_source_tasks,
                )
            if request_type == "text":
                continue
            if request_type == "image":
                image = _require_mapping(request, "image", request_path)
                _validate_color_image(image, f"{request_path}.image")
                continue
            if request_type == "image_file":
                image = _require_mapping(request, "image", request_path)
                _validate_file_image(image, f"{request_path}.image")
                continue
            if request_type == "images":
                images = _require_list(request, "images", request_path)
                if not images:
                    raise ValueError(f"{request_path}.images must not be empty")
                for image_index, image in enumerate(images):
                    if not isinstance(image, Mapping):
                        raise ValueError(
                            f"{request_path}.images[{image_index}] must be an object"
                        )
                    _validate_color_image(
                        image,
                        f"{request_path}.images[{image_index}]",
                    )
                continue
            if request_type == "video":
                frames = _require_list(request, "frames", request_path)
                if not frames:
                    raise ValueError(f"{request_path}.frames must not be empty")
                for frame_index, frame in enumerate(frames):
                    if not isinstance(frame, Mapping):
                        raise ValueError(
                            f"{request_path}.frames[{frame_index}] must be an object"
                        )
                    _validate_color_image(
                        frame,
                        f"{request_path}.frames[{frame_index}]",
                    )
                continue
            raise ValueError(f"unsupported request type: {request_type!r}")


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
        raise ValueError(
            f"{path} must satisfy min <= median <= p90 <= p99 <= max"
        )


def _validate_input_shapes(shapes: list[Any], expected_requests: int) -> None:
    """按 [height, width, channels] 校验每个请求的视觉输入 shape。"""

    if len(shapes) != expected_requests:
        raise ValueError(
            "record.workload.input_shapes length must match workload.num_requests"
        )
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
                or len(visual_shape) != 3
                or not all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension >= 1
                    for dimension in visual_shape
                )
            ):
                raise ValueError(
                    f"{visual_path} must be [height, width, channels] positive ints"
                )


def validate_benchmark_record(record: Mapping[str, Any]) -> None:
    """校验一条 P6 系统 benchmark JSONL 记录。"""

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_BENCHMARK_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported benchmark schema_version: "
            f"{schema_version!r}"
        )
    if record.get("record_type") != "system_benchmark":
        raise ValueError("record_type must be 'system_benchmark'")
    _require_string(record, "timestamp_utc", "record")

    environment = _require_mapping(record, "environment", "record")
    for key in ("git_commit", "python", "torch", "transformers", "gpu"):
        _require_string(environment, key, "record.environment")
    _require_bool(environment, "git_dirty", "record.environment")
    if "cuda" not in environment:
        raise ValueError("record.environment.cuda is required")

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
    if schema_version >= 6:
        projection_mode = _require_string(
            model,
            "mlp_projection_mode",
            "record.model",
        )
        if projection_mode not in ("legacy", "packed"):
            raise ValueError(
                "record.model.mlp_projection_mode must be 'legacy' or 'packed'"
            )

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

    workload = _require_mapping(record, "workload", "record")
    for key in ("manifest_name", "case_id"):
        _require_string(workload, key, "record.workload")
    _require_sha256(workload, "manifest_sha256", "record.workload")
    request_types = _require_list(workload, "request_types", "record.workload")
    if not request_types or not all(
        isinstance(request_type, str) and request_type for request_type in request_types
    ):
        raise ValueError("record.workload.request_types must contain non-empty strings")
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
    num_requests = _require_int(
        workload,
        "num_requests",
        "record.workload",
        minimum=1,
    )
    if schema_version >= 2:
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
    if len(request_types) != num_requests:
        raise ValueError(
            "record.workload.request_types length must match workload.num_requests"
        )
    _validate_input_shapes(
        _require_list(workload, "input_shapes", "record.workload"),
        num_requests,
    )
    _require_bool(
        workload,
        "preprocessing_included_in_e2e",
        "record.workload",
    )
    if schema_version >= 5:
        if _require_bool(
            workload,
            "output_decoding_included_in_e2e",
            "record.workload",
        ):
            raise ValueError(
                "schema-v5 benchmark output decoding must remain outside E2E timing"
            )
        reference_sources = _require_mapping(
            workload,
            "reference_sources",
            "record.workload",
        )
        reference_source_ids = set(reference_sources)
        for source_id, source in reference_sources.items():
            path = f"record.workload.reference_sources.{source_id}"
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(
                    "record.workload reference source ids must be non-empty strings"
                )
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
        reference_source_tasks = {
            source_id: str(source["task"])
            for source_id, source in reference_sources.items()
        }
        task_references = _require_list(
            workload,
            "task_references",
            "record.workload",
        )
        if len(task_references) != num_requests:
            raise ValueError(
                "record.workload.task_references length must match "
                "workload.num_requests"
            )
        for request_index, task_reference in enumerate(task_references):
            if task_reference is None:
                continue
            path = f"record.workload.task_references[{request_index}]"
            if not isinstance(task_reference, Mapping):
                raise ValueError(f"{path} must be an object or null")
            _validate_task_reference(
                task_reference,
                path,
                source_ids=reference_source_ids,
                source_tasks=reference_source_tasks,
            )

    traffic = _require_mapping(record, "traffic", "record")
    if _require_string(traffic, "kind", "record.traffic") != "offline_closed_loop":
        raise ValueError("record.traffic.kind must be 'offline_closed_loop'")
    batch_size = _require_int(traffic, "batch_size", "record.traffic", minimum=1)
    concurrency = _require_int(
        traffic,
        "concurrency",
        "record.traffic",
        minimum=1,
    )
    if batch_size != num_requests or concurrency != num_requests:
        raise ValueError(
            "offline traffic batch_size/concurrency must match workload.num_requests"
        )
    request_rate = traffic.get("request_rate_per_s")
    if request_rate is not None:
        _require_number(traffic, "request_rate_per_s", "record.traffic")

    if schema_version >= 2:
        execution = _require_mapping(record, "execution_backend", "record")
        prefill_backend = _require_string(
            execution,
            "prefill_backend",
            "record.execution_backend",
        )
        decode_backend = _require_string(
            execution,
            "decode_backend",
            "record.execution_backend",
        )
        graph_enabled = _require_bool(
            execution,
            "cuda_graph_enabled",
            "record.execution_backend",
        )
        capture_scope = _require_string(
            execution,
            "cuda_graph_capture_scope",
            "record.execution_backend",
        )
        capture_ms = _require_number(
            execution,
            "cuda_graph_capture_ms",
            "record.execution_backend",
        )
        graph_batch_sizes = _require_list(
            execution,
            "cuda_graph_batch_sizes",
            "record.execution_backend",
        )
        if not all(
            isinstance(graph_batch_size, int)
            and not isinstance(graph_batch_size, bool)
            and graph_batch_size >= 1
            for graph_batch_size in graph_batch_sizes
        ):
            raise ValueError(
                "record.execution_backend.cuda_graph_batch_sizes must contain "
                "positive ints"
            )
        if graph_batch_sizes != sorted(set(graph_batch_sizes)):
            raise ValueError(
                "record.execution_backend.cuda_graph_batch_sizes must be sorted "
                "and unique"
            )
        requested_batch_size = _require_int(
            execution,
            "requested_decode_batch_size",
            "record.execution_backend",
            minimum=1,
        )
        selected_batch_size = _require_int(
            execution,
            "selected_decode_batch_size",
            "record.execution_backend",
            minimum=1,
        )
        batch_padding = _require_int(
            execution,
            "decode_batch_padding",
            "record.execution_backend",
            minimum=0,
        )
        if prefill_backend != "eager":
            raise ValueError("record.execution_backend.prefill_backend must be 'eager'")
        if requested_batch_size != batch_size:
            raise ValueError(
                "record.execution_backend.requested_decode_batch_size must match "
                "traffic.batch_size"
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
        if graph_enabled:
            if decode_backend != "cuda_graph":
                raise ValueError(
                    "CUDA Graph execution requires decode_backend='cuda_graph'"
                )
            if capture_scope != "decode_model_forward":
                raise ValueError(
                    "CUDA Graph execution requires capture scope "
                    "'decode_model_forward'"
                )
            if selected_batch_size not in graph_batch_sizes:
                raise ValueError(
                    "selected CUDA Graph batch size is absent from captured sizes"
                )
        elif (
            decode_backend
            not in (
                ("eager", "torch_compile_attention")
                if schema_version >= 3
                else ("eager",)
            )
            or capture_scope != "none"
            or capture_ms != 0.0
            or graph_batch_sizes
            or batch_padding != 0
        ):
            raise ValueError("eager execution must not report CUDA Graph capture state")

        if schema_version >= 3:
            compile_enabled = _require_bool(
                execution,
                "torch_compile_enabled",
                "record.execution_backend",
            )
            compile_region = _require_string(
                execution,
                "torch_compile_region",
                "record.execution_backend",
            )
            compile_backend = _require_string(
                execution,
                "torch_compile_backend",
                "record.execution_backend",
            )
            compile_mode = _require_string(
                execution,
                "torch_compile_mode",
                "record.execution_backend",
            )
            emulate_precision_casts = _require_bool(
                execution,
                "torch_compile_emulate_precision_casts",
                "record.execution_backend",
            )
            force_same_precision = _require_bool(
                execution,
                "torch_compile_force_same_precision",
                "record.execution_backend",
            )
            compile_first_call_ms = _require_number(
                execution,
                "torch_compile_first_call_ms",
                "record.execution_backend",
            )
            if compile_enabled:
                if graph_enabled or decode_backend != "torch_compile_attention":
                    raise ValueError(
                        "torch.compile execution must be graph-disabled and use "
                        "decode_backend='torch_compile_attention'"
                    )
                if (
                    compile_region != "decode_attention"
                    or compile_backend != "inductor"
                    or compile_mode not in ("default", "reduce-overhead")
                    or compile_first_call_ms <= 0.0
                ):
                    raise ValueError("torch.compile execution metadata is invalid")
            elif (
                compile_region != "none"
                or compile_backend != "none"
                or compile_mode != "none"
                or emulate_precision_casts
                or force_same_precision
                or compile_first_call_ms != 0.0
            ):
                raise ValueError(
                    "disabled torch.compile execution must report empty state"
                )

    measurement = _require_mapping(record, "measurement", "record")
    _require_int(measurement, "warmup", "record.measurement", minimum=0)
    _require_int(measurement, "repeat", "record.measurement", minimum=1)
    _require_bool(
        measurement,
        "cuda_synchronize_timing",
        "record.measurement",
    )

    correctness = _require_mapping(record, "correctness", "record")
    _require_bool(
        correctness,
        "outputs_identical_across_repeats",
        "record.correctness",
    )
    token_ids = _require_list(correctness, "token_ids", "record.correctness")
    if not token_ids or not all(
        isinstance(request_tokens, list)
        and all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in request_tokens
        )
        for request_tokens in token_ids
    ):
        raise ValueError(
            "record.correctness.token_ids must contain non-negative integer lists"
        )
    if len(token_ids) != num_requests:
        raise ValueError(
            "record.correctness.token_ids length must match workload.num_requests"
        )
    output_tokens = _require_int(
        correctness,
        "output_tokens",
        "record.correctness",
        minimum=1,
    )
    if output_tokens != sum(len(request_tokens) for request_tokens in token_ids):
        raise ValueError(
            "record.correctness.output_tokens must equal the token_ids length sum"
        )
    output_sha256 = _require_sha256(
        correctness,
        "output_sha256",
        "record.correctness",
    )
    if output_sha256 != canonical_json_sha256(token_ids):
        raise ValueError("record.correctness.output_sha256 does not match token_ids")
    if schema_version >= 5:
        decoded_texts = _require_list(
            correctness,
            "decoded_texts",
            "record.correctness",
        )
        if len(decoded_texts) != num_requests or not all(
            isinstance(text, str) for text in decoded_texts
        ):
            raise ValueError(
                "record.correctness.decoded_texts must contain one string per request"
            )
        decoded_texts_sha256 = _require_sha256(
            correctness,
            "decoded_texts_sha256",
            "record.correctness",
        )
        if decoded_texts_sha256 != canonical_json_sha256(decoded_texts):
            raise ValueError(
                "record.correctness.decoded_texts_sha256 does not match decoded_texts"
            )

    timing = _require_mapping(record, "timing_ms", "record")
    for key in (
        "preprocessing",
        "engine_ttft",
        "end_to_end_ttft",
        "prefill",
        "decode_step",
        "end_to_end",
    ):
        _validate_stats(
            _require_mapping(timing, key, "record.timing_ms"),
            f"record.timing_ms.{key}",
        )

    throughput = _require_mapping(record, "throughput", "record")
    for key in (
        "engine_output_tokens_per_s",
        "e2e_output_tokens_per_s",
        "decode_tokens_per_s",
        "engine_requests_per_s",
        "e2e_requests_per_s",
    ):
        _validate_stats(
            _require_mapping(throughput, key, "record.throughput"),
            f"record.throughput.{key}",
        )

    memory = _require_mapping(record, "memory_mb", "record")
    for key in ("allocated", "reserved", "peak_allocated"):
        _validate_stats(
            _require_mapping(memory, key, "record.memory_mb"),
            f"record.memory_mb.{key}",
        )

    kv_cache = _require_mapping(record, "kv_cache", "record")
    kv_dtype = _require_string(kv_cache, "dtype", "record.kv_cache")
    shape = _require_list(kv_cache, "shape", "record.kv_cache")
    if not shape or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError("record.kv_cache.shape must contain non-negative ints")
    for key in ("bytes", "blocks", "block_size", "capacity_tokens"):
        _require_int(kv_cache, key, "record.kv_cache", minimum=1)
    if kv_cache["capacity_tokens"] != kv_cache["blocks"] * kv_cache["block_size"]:
        raise ValueError(
            "record.kv_cache.capacity_tokens must equal blocks * block_size"
        )
    if schema_version >= 4:
        logical_prompt_tokens = _require_int(
            kv_cache,
            "logical_prompt_tokens",
            "record.kv_cache",
            minimum=1,
        )
        physical_prompt_tokens = _require_int(
            kv_cache,
            "physical_prompt_tokens",
            "record.kv_cache",
            minimum=1,
        )
        dense_prompt_blocks = _require_int(
            kv_cache,
            "dense_prompt_blocks",
            "record.kv_cache",
            minimum=1,
        )
        active_prompt_blocks = _require_int(
            kv_cache,
            "active_prompt_blocks",
            "record.kv_cache",
            minimum=1,
        )
        released_prompt_blocks = _require_int(
            kv_cache,
            "released_prompt_blocks",
            "record.kv_cache",
            minimum=0,
        )
        dense_prompt_bytes = _require_int(
            kv_cache,
            "dense_prompt_bytes",
            "record.kv_cache",
            minimum=1,
        )
        active_prompt_bytes = _require_int(
            kv_cache,
            "active_prompt_bytes",
            "record.kv_cache",
            minimum=1,
        )
        if physical_prompt_tokens > logical_prompt_tokens:
            raise ValueError("physical prompt KV tokens cannot exceed logical prompt tokens")
        if active_prompt_blocks > dense_prompt_blocks:
            raise ValueError("active prompt blocks cannot exceed dense prompt blocks")
        if released_prompt_blocks != dense_prompt_blocks - active_prompt_blocks:
            raise ValueError("released prompt blocks must equal dense-active blocks")
        if dense_prompt_bytes % dense_prompt_blocks != 0:
            raise ValueError("dense prompt bytes must be divisible by dense blocks")
        bytes_per_block = dense_prompt_bytes // dense_prompt_blocks
        if active_prompt_bytes != active_prompt_blocks * bytes_per_block:
            raise ValueError("active prompt bytes do not match active blocks")
        if schema_version >= 7:
            scale_dtype = _require_string(
                kv_cache,
                "scale_dtype",
                "record.kv_cache",
            )
            scale_shape = _require_list(
                kv_cache,
                "scale_shape",
                "record.kv_cache",
            )
            if not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension >= 0
                for dimension in scale_shape
            ):
                raise ValueError(
                    "record.kv_cache.scale_shape must contain non-negative ints"
                )
            payload_bytes = _require_int(
                kv_cache,
                "payload_bytes",
                "record.kv_cache",
                minimum=1,
            )
            scale_bytes = _require_int(
                kv_cache,
                "scale_bytes",
                "record.kv_cache",
                minimum=0,
            )
            if kv_cache["bytes"] != payload_bytes + scale_bytes:
                raise ValueError(
                    "record.kv_cache.bytes must equal payload_bytes + scale_bytes"
                )
            expected_payload_bytes = _tensor_bytes_from_metadata(
                dtype=kv_dtype,
                shape=shape,
                path="record.kv_cache",
            )
            if payload_bytes != expected_payload_bytes:
                raise ValueError(
                    "record.kv_cache.payload_bytes does not match dtype and shape"
                )
            if scale_bytes == 0:
                if scale_dtype != "none" or scale_shape:
                    raise ValueError(
                        "zero scale bytes require scale_dtype='none' and empty scale_shape"
                    )
            elif scale_dtype == "none" or not scale_shape:
                raise ValueError(
                    "non-zero scale bytes require a scale dtype and shape"
                )
            else:
                if scale_shape != shape[:-1]:
                    raise ValueError(
                        "record.kv_cache.scale_shape must equal payload shape without "
                        "the head dimension"
                    )
                expected_scale_bytes = _tensor_bytes_from_metadata(
                    dtype=scale_dtype,
                    shape=scale_shape,
                    path="record.kv_cache.scale",
                )
                if scale_bytes != expected_scale_bytes:
                    raise ValueError(
                        "record.kv_cache.scale_bytes does not match dtype and shape"
                    )

            dense_payload_bytes = _require_int(
                kv_cache,
                "dense_prompt_payload_bytes",
                "record.kv_cache",
                minimum=1,
            )
            dense_scale_bytes = _require_int(
                kv_cache,
                "dense_prompt_scale_bytes",
                "record.kv_cache",
                minimum=0,
            )
            active_payload_bytes = _require_int(
                kv_cache,
                "active_prompt_payload_bytes",
                "record.kv_cache",
                minimum=1,
            )
            active_scale_bytes = _require_int(
                kv_cache,
                "active_prompt_scale_bytes",
                "record.kv_cache",
                minimum=0,
            )
            if dense_prompt_bytes != dense_payload_bytes + dense_scale_bytes:
                raise ValueError(
                    "dense prompt bytes must equal payload + scale bytes"
                )
            if active_prompt_bytes != active_payload_bytes + active_scale_bytes:
                raise ValueError(
                    "active prompt bytes must equal payload + scale bytes"
                )
            if dense_payload_bytes % dense_prompt_blocks != 0:
                raise ValueError(
                    "dense prompt payload bytes must be divisible by dense blocks"
                )
            payload_bytes_per_block = dense_payload_bytes // dense_prompt_blocks
            if active_payload_bytes != active_prompt_blocks * payload_bytes_per_block:
                raise ValueError(
                    "active prompt payload bytes do not match active blocks"
                )
            if dense_scale_bytes % dense_prompt_blocks != 0:
                raise ValueError(
                    "dense prompt scale bytes must be divisible by dense blocks"
                )
            scale_bytes_per_block = dense_scale_bytes // dense_prompt_blocks
            if active_scale_bytes != active_prompt_blocks * scale_bytes_per_block:
                raise ValueError(
                    "active prompt scale bytes do not match active blocks"
                )
        layouts = _require_list(kv_cache, "layouts", "record.kv_cache")
        if len(layouts) != num_requests:
            raise ValueError("KV layout record count must match workload requests")
        for layout in layouts:
            if not isinstance(layout, Mapping):
                raise ValueError("KV layout record must be an object")
            for key in ("mode", "kv_dtype"):
                _require_string(layout, key, "record.kv_cache.layouts[]")
            for key in (
                "logical_context_len",
                "physical_kv_len",
                "prompt_logical_len",
                "compressed_prompt_kv_len",
            ):
                _require_int(
                    layout,
                    key,
                    "record.kv_cache.layouts[]",
                    minimum=1,
                )
            layout_blocks = _require_list(
                layout,
                "block_table",
                "record.kv_cache.layouts[]",
            )
            if not layout_blocks or not all(
                isinstance(block_id, int)
                and not isinstance(block_id, bool)
                and block_id >= 0
                for block_id in layout_blocks
            ):
                raise ValueError("KV layout block table must contain non-negative ints")
