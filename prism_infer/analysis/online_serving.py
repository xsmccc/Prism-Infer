"""Schema and SLO aggregation for P7.3 online-serving runs."""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence


ONLINE_SUMMARY_SCHEMA_VERSION = 1
ONLINE_BENCHMARK_SCHEMA_VERSION = 1


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
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, object]:
    """Aggregate one online run without converting throughput into goodput."""

    if ttft_slo_ms <= 0 or tpot_slo_ms <= 0:
        raise ValueError("SLO thresholds must be positive")
    duration_s = float(run.get("duration_s", 0.0))
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("online run duration_s must be positive and finite")
    engine_metrics = _require_mapping(
        run.get("engine_metrics"), "engine_metrics"
    )
    request_records_raw = engine_metrics.get("requests")
    if not isinstance(request_records_raw, list) or not request_records_raw:
        raise ValueError("engine_metrics.requests must be a non-empty list")
    request_records = [
        _require_mapping(record, f"engine_metrics.requests[{index}]")
        for index, record in enumerate(request_records_raw)
    ]

    completed = [
        record
        for record in request_records
        if record.get("finish_reason") in {"eos", "length"}
    ]
    rejected = [
        record for record in request_records if record.get("finish_reason") == "rejected"
    ]
    cancelled = [
        record for record in request_records if record.get("finish_reason") == "cancelled"
    ]
    if len(completed) + len(rejected) + len(cancelled) != len(request_records):
        raise ValueError("every online request must have a terminal finish_reason")

    def metric_values(name: str, *, allow_none: bool = False) -> list[float]:
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

    good_requests = 0
    for record in completed:
        ttft = record.get("ttft_ms")
        if ttft is None:
            raise ValueError("completed request missing ttft_ms")
        output_tokens = int(record.get("output_tokens", 0))
        tpot = record.get("tpot_ms")
        effective_tpot = 0.0 if output_tokens <= 1 and tpot is None else tpot
        if effective_tpot is None:
            raise ValueError("multi-token completed request missing tpot_ms")
        if float(ttft) <= ttft_slo_ms and float(effective_tpot) <= tpot_slo_ms:
            good_requests += 1

    output_tokens = sum(int(record.get("output_tokens", 0)) for record in completed)
    scheduler_metrics = _require_mapping(
        run.get("scheduler_metrics", {}), "scheduler_metrics"
    )
    return {
        "schema_version": ONLINE_SUMMARY_SCHEMA_VERSION,
        "record_type": "prism_online_summary",
        "slo": {
            "ttft_ms": float(ttft_slo_ms),
            "tpot_ms": float(tpot_slo_ms),
        },
        "counts": {
            "submitted": len(request_records),
            "completed": len(completed),
            "rejected": len(rejected),
            "cancelled": len(cancelled),
            "good": good_requests,
        },
        "latency_ms": {
            "queue": summarize_distribution(metric_values("queue_ms")),
            "ttft": summarize_distribution(metric_values("ttft_ms")),
            "tpot": summarize_distribution(
                metric_values("tpot_ms", allow_none=True)
            ),
            "request": summarize_distribution(metric_values("latency_ms")),
        },
        "throughput": {
            "requests_per_s": len(completed) / duration_s,
            "output_tokens_per_s": output_tokens / duration_s,
        },
        "goodput": {
            "requests_per_s": good_requests / duration_s,
            "fraction_of_completed": (
                0.0 if not completed else good_requests / len(completed)
            ),
        },
        "scheduler": dict(scheduler_metrics),
    }


def validate_online_benchmark_record(record: Mapping[str, object]) -> None:
    """Fail closed on malformed/tampered formal online benchmark records."""

    if record.get("schema_version") != ONLINE_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported online benchmark schema_version")
    if record.get("record_type") != "prism_online_run":
        raise ValueError("online benchmark record_type must be prism_online_run")
    if not isinstance(record.get("git_commit"), str) or not record["git_commit"]:
        raise ValueError("online benchmark requires git_commit")
    if not isinstance(record.get("git_dirty"), bool):
        raise ValueError("online benchmark requires boolean git_dirty")

    workload = _require_mapping(record.get("workload"), "workload")
    for key in ("manifest", "case", "max_tokens"):
        if key not in workload:
            raise ValueError(f"workload missing {key}")
    request_count = int(workload.get("requests", 0))
    if request_count <= 0:
        raise ValueError("workload.requests must be positive")
    arrival = _require_mapping(record.get("arrival"), "arrival")
    for key in ("process", "request_rate_per_s", "seed"):
        if key not in arrival:
            raise ValueError(f"arrival missing {key}")
    offsets = arrival.get("offsets_s")
    if not isinstance(offsets, list) or len(offsets) != request_count:
        raise ValueError("arrival offsets must match workload request count")
    numeric_offsets = [float(offset) for offset in offsets]
    if any(
        not isfinite(offset) or offset < 0 for offset in numeric_offsets
    ) or numeric_offsets != sorted(numeric_offsets):
        raise ValueError("arrival offsets must be finite, non-negative and sorted")

    hardware = _require_mapping(record.get("hardware"), "hardware")
    for key in ("gpu", "gpu_uuid", "total_memory_bytes"):
        if key not in hardware:
            raise ValueError(f"hardware missing {key}")
    engine = _require_mapping(record.get("engine"), "engine")
    for key in (
        "mode",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_chunk_size",
        "num_kvcache_blocks",
        "kvcache_block_size",
        "enable_prefix_caching",
    ):
        if key not in engine:
            raise ValueError(f"engine missing {key}")
    run = _require_mapping(record.get("run"), "run")
    results = run.get("requests")
    if not isinstance(results, list) or len(results) != request_count:
        raise ValueError("run.requests must match workload request count")
    request_ids = [
        int(_require_mapping(result, "run request").get("request_id", -1))
        for result in results
    ]
    if any(request_id < 0 for request_id in request_ids) or len(
        set(request_ids)
    ) != len(request_ids):
        raise ValueError("run request ids must be unique non-negative integers")

    summary = _require_mapping(record.get("summary"), "summary")
    slo = _require_mapping(summary.get("slo"), "summary.slo")
    expected = summarize_online_run(
        run,
        ttft_slo_ms=float(slo.get("ttft_ms", 0.0)),
        tpot_slo_ms=float(slo.get("tpot_ms", 0.0)),
    )
    if dict(summary) != expected:
        raise ValueError("online summary does not match recomputed run metrics")
