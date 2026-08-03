"""Schema and latency aggregation for online-serving runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

ONLINE_SUMMARY_SCHEMA_VERSION = 2
ONLINE_BENCHMARK_SCHEMA_VERSION = 2
SUPPORTED_ONLINE_BENCHMARK_SCHEMA_VERSIONS = (1, 2)
ONLINE_PROJECTION_MODE_SCHEMA_VERSION = 2


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
        }
    floats = [float(value) for value in values]
    return {
        "count": len(floats),
        "min": min(floats),
        "max": max(floats),
        "mean": sum(floats) / len(floats),
        "p50": percentile(floats, 0.50),
        "p90": percentile(floats, 0.90),
        "p99": percentile(floats, 0.99),
    }


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def summarize_online_run(
    run: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate request counts, latency, throughput, and scheduler metrics."""

    duration_s = float(run.get("duration_s", 0.0))
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("online run duration_s must be positive and finite")
    engine_metrics = _require_mapping(run.get("engine_metrics"), "engine_metrics")
    request_records_raw = engine_metrics.get("requests")
    if not isinstance(request_records_raw, list) or not request_records_raw:
        raise ValueError("engine_metrics.requests must be a non-empty list")
    request_records = [
        _require_mapping(record, f"engine_metrics.requests[{index}]")
        for index, record in enumerate(request_records_raw)
    ]

    completed, rejected, cancelled = _partition_terminal_requests(request_records)
    if len(completed) + len(rejected) + len(cancelled) != len(request_records):
        raise ValueError("every online request must have a terminal finish_reason")
    output_tokens = sum(int(record.get("output_tokens", 0)) for record in completed)
    scheduler_metrics = _require_mapping(run.get("scheduler_metrics", {}), "scheduler_metrics")
    return {
        "schema_version": ONLINE_SUMMARY_SCHEMA_VERSION,
        "record_type": "prism_online_summary",
        "counts": {
            "submitted": len(request_records),
            "completed": len(completed),
            "rejected": len(rejected),
            "cancelled": len(cancelled),
        },
        "latency_ms": {
            "queue": summarize_distribution(_completed_metric_values(completed, "queue_ms")),
            "ttft": summarize_distribution(_completed_metric_values(completed, "ttft_ms")),
            "tpot": summarize_distribution(
                _completed_metric_values(completed, "tpot_ms", allow_none=True)
            ),
            "request": summarize_distribution(_completed_metric_values(completed, "latency_ms")),
        },
        "throughput": {
            "requests_per_s": len(completed) / duration_s,
            "output_tokens_per_s": output_tokens / duration_s,
        },
        "scheduler": dict(scheduler_metrics),
    }


def _partition_terminal_requests(
    request_records: list[Mapping[str, object]],
) -> tuple[
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
]:
    completed = [
        record for record in request_records if record.get("finish_reason") in {"eos", "length"}
    ]
    rejected = [record for record in request_records if record.get("finish_reason") == "rejected"]
    cancelled = [record for record in request_records if record.get("finish_reason") == "cancelled"]
    return completed, rejected, cancelled


def _completed_metric_values(
    completed: list[Mapping[str, object]],
    name: str,
    *,
    allow_none: bool = False,
) -> list[float]:
    values: list[float] = []
    for record in completed:
        value = record.get(name)
        if value is None and allow_none:
            continue
        if value is None:
            raise ValueError(f"completed request missing {name}")
        number = float(value)
        if not isfinite(number) or number < 0:
            raise ValueError(f"invalid request metric {name}={value!r}")
        values.append(number)
    return values


def validate_online_benchmark_record(record: Mapping[str, object]) -> None:
    """Validate the structure and internal consistency of an online run record."""

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_ONLINE_BENCHMARK_SCHEMA_VERSIONS:
        raise ValueError("unsupported online benchmark schema_version")
    if record.get("record_type") != "prism_online_run":
        raise ValueError("online benchmark record_type must be prism_online_run")
    if not isinstance(record.get("git_commit"), str) or not record["git_commit"]:
        raise ValueError("online benchmark requires git_commit")
    if not isinstance(record.get("git_dirty"), bool):
        raise ValueError("online benchmark requires boolean git_dirty")
    request_count = _validate_online_workload(record)
    _validate_online_arrival(record, request_count)
    _validate_online_hardware_and_engine(record, schema_version)
    run = _validate_online_run(record, request_count)
    _validate_online_summary(record, run)


def _validate_online_workload(record: Mapping[str, object]) -> int:
    workload = _require_mapping(record.get("workload"), "workload")
    _require_keys(workload, ("manifest", "case", "max_tokens"), "workload")
    request_count = int(workload.get("requests", 0))
    if request_count <= 0:
        raise ValueError("workload.requests must be positive")
    return request_count


def _validate_online_arrival(record: Mapping[str, object], request_count: int) -> None:
    arrival = _require_mapping(record.get("arrival"), "arrival")
    _require_keys(arrival, ("process", "request_rate_per_s", "seed"), "arrival")
    offsets = arrival.get("offsets_s")
    if not isinstance(offsets, list) or len(offsets) != request_count:
        raise ValueError("arrival offsets must match workload request count")
    numeric_offsets = [float(offset) for offset in offsets]
    if any(
        not isfinite(offset) or offset < 0 for offset in numeric_offsets
    ) or numeric_offsets != sorted(numeric_offsets):
        raise ValueError("arrival offsets must be finite, non-negative and sorted")


def _validate_online_hardware_and_engine(
    record: Mapping[str, object],
    schema_version: object,
) -> None:
    hardware = _require_mapping(record.get("hardware"), "hardware")
    _require_keys(hardware, ("gpu", "gpu_uuid", "total_memory_bytes"), "hardware")
    engine = _require_mapping(record.get("engine"), "engine")
    _require_keys(
        engine,
        (
            "mode",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_chunk_size",
            "num_kvcache_blocks",
            "kvcache_block_size",
            "enable_prefix_caching",
        ),
        "engine",
    )
    if int(schema_version) >= ONLINE_PROJECTION_MODE_SCHEMA_VERSION:
        projection_mode = engine.get("mlp_projection_mode")
        if projection_mode not in ("legacy", "packed"):
            raise ValueError("engine.mlp_projection_mode must be 'legacy' or 'packed'")


def _require_keys(
    mapping: Mapping[str, object],
    keys: tuple[str, ...],
    label: str,
) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"{label} missing {key}")


def _validate_online_run(
    record: Mapping[str, object],
    request_count: int,
) -> Mapping[str, object]:
    run = _require_mapping(record.get("run"), "run")
    results = run.get("requests")
    if not isinstance(results, list) or len(results) != request_count:
        raise ValueError("run.requests must match workload request count")
    request_ids = [
        int(_require_mapping(result, "run request").get("request_id", -1)) for result in results
    ]
    if any(request_id < 0 for request_id in request_ids) or len(set(request_ids)) != len(
        request_ids
    ):
        raise ValueError("run request ids must be unique non-negative integers")
    return run


def _validate_online_summary(
    record: Mapping[str, object],
    run: Mapping[str, object],
) -> None:
    summary = _require_mapping(record.get("summary"), "summary")
    expected = summarize_online_run(run)
    if dict(summary) != expected:
        raise ValueError("online summary does not match recomputed run metrics")
