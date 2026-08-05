"""Summarize labeled multimodal working-set benchmark records.

The tool accepts raw Prism, vLLM, and SGLang JSON records produced from the
same working-set plan.  It fails closed when plan, model, KV-budget, or
per-workset request-trace/prompt-token identities disagree, when population is
incomplete, or when the complete 15-cell matrix is missing. Missing backend
metrics are reported as ``unavailable`` instead of being estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from prism_infer.analysis.identity import sha256_bytes

UNAVAILABLE = "unavailable"
WORKSET_ORDER = ("fit", "knee", "pressure")
PRISM_VARIANT_ORDER = ("vision_only", "dense_prefix", "compact_prefix")
ENGINE_ORDER = ("prism", "vllm", "sglang")
_COMPLETED_REASONS = frozenset({"eos", "length", "stop"})
_POPULATION_POLICY = "one_request_per_group_closed_loop"
_REQUIRED_CELLS = frozenset(
    {
        ("prism", variant, workset_id)
        for variant in PRISM_VARIANT_ORDER
        for workset_id in WORKSET_ORDER
    }
    | {
        (engine, "engine_default", workset_id)
        for engine in ("vllm", "sglang")
        for workset_id in WORKSET_ORDER
    }
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _path(value: object, *keys: str) -> object | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return int(value) if isinstance(value, int) else numeric


def _optional_metric(mapping: Mapping[str, Any], key: str) -> int | float | str:
    value = mapping.get(key)
    if value is None:
        return UNAVAILABLE
    return _nonnegative_number(value, key)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(requests: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = [float(_nonnegative_number(request[field], field)) for request in requests]
    return {"p50": _percentile(values, 0.50), "p99": _percentile(values, 0.99)}


def _framework(record: Mapping[str, Any]) -> str:
    record_type = record.get("record_type")
    if record_type == "prism_online_run":
        name = _path(record, "framework", "name")
        if name != "prism-infer":
            raise ValueError(f"unexpected Prism framework label: {name!r}")
        return "prism"
    if record_type == "external_online_run":
        name = _path(record, "environment", "framework")
        if name in ("vllm", "sglang"):
            return str(name)
        raise ValueError(f"unsupported external framework label: {name!r}")
    raise ValueError(f"unsupported working-set record_type: {record_type!r}")


def _requests(record: Mapping[str, Any], engine: str) -> list[Mapping[str, Any]]:
    raw = (
        _path(record, "run", "engine_metrics", "requests")
        if engine == "prism"
        else _path(record, "run", "requests")
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{engine} run must contain request records")
    requests = [_mapping(item, f"{engine} request[{index}]") for index, item in enumerate(raw)]
    completed = [
        request for request in requests if request.get("finish_reason") in _COMPLETED_REASONS
    ]
    if not completed:
        raise ValueError(f"{engine} run has no completed requests")
    for request in completed:
        for field in ("ttft_ms", "latency_ms", "output_tokens"):
            if request.get(field) is None:
                raise ValueError(f"completed {engine} request is missing {field}")
            _nonnegative_number(request[field], field)
    return completed


def _population_summary(
    record: Mapping[str, Any],
    engine: str,
    expected_requests: int,
) -> dict[str, int | str | bool]:
    raw_population = (
        _path(record, "population", "run") if engine == "prism" else record.get("population")
    )
    population = _mapping(raw_population, f"{engine} population")
    policy = _nonempty_string(population.get("policy"), f"{engine} population policy")
    if policy != _POPULATION_POLICY:
        raise ValueError(f"unsupported {engine} population policy: {policy!r}")
    raw_runs = population.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError(f"{engine} population.runs must be a list")
    if len(raw_runs) != expected_requests:
        raise ValueError(
            f"{engine} population has {len(raw_runs)} runs for {expected_requests} planned requests"
        )

    completed_requests = 0
    previous_prism_request_ids: list[object] = []
    for run_index, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"{engine} population run[{run_index}]")
        raw_requests = (
            _path(run, "engine_metrics", "requests") if engine == "prism" else run.get("requests")
        )
        if not isinstance(raw_requests, list):
            raise ValueError(f"{engine} population run[{run_index}] has no request list")
        if engine == "prism":
            expected_snapshot_requests = run_index + 1
            if len(raw_requests) != expected_snapshot_requests:
                raise ValueError(
                    f"prism population run[{run_index}] must contain "
                    f"{expected_snapshot_requests} cumulative requests; found {len(raw_requests)}"
                )
            request_ids = [
                _mapping(item, f"prism population request[{run_index}][{index}]").get("request_id")
                for index, item in enumerate(raw_requests)
            ]
            if request_ids[:-1] != previous_prism_request_ids:
                raise ValueError(
                    f"prism population run[{run_index}] changed prior cumulative requests"
                )
            if request_ids[-1] is None or request_ids[-1] in previous_prism_request_ids:
                raise ValueError(
                    f"prism population run[{run_index}] did not add one unique request"
                )
            previous_prism_request_ids = request_ids
            raw_request = raw_requests[-1]
        elif len(raw_requests) == 1:
            raw_request = raw_requests[0]
        else:
            count = len(raw_requests) if isinstance(raw_requests, list) else 0
            raise ValueError(
                f"{engine} population run[{run_index}] must contain exactly one request; "
                f"found {count}"
            )
        request = _mapping(raw_request, f"{engine} population request[{run_index}]")
        if request.get("finish_reason") not in _COMPLETED_REASONS:
            raise ValueError(
                f"{engine} population request[{run_index}] did not complete: "
                f"finish_reason={request.get('finish_reason')!r}"
            )
        completed_requests += 1

    return {
        "policy": policy,
        "expected_requests": expected_requests,
        "runs": len(raw_runs),
        "completed_requests": completed_requests,
        "complete": completed_requests == expected_requests,
        "request_record_scope": (
            "cumulative_snapshots_one_new_request_per_run"
            if engine == "prism"
            else "one_request_per_run"
        ),
    }


def _kv_budget_bytes(record: Mapping[str, Any], engine: str) -> int:
    audit_budget = _path(record, "workload", "working_set_plan", "kv_budget_bytes")
    if audit_budget is not None:
        return int(_nonnegative_number(audit_budget, "working-set KV budget"))
    candidates: tuple[tuple[str, ...], ...]
    if engine == "prism":
        candidates = (("memory", "kv_cache", "total_bytes"),)
    elif engine == "vllm":
        candidates = (
            ("backend", "kv_cache_memory_bytes_requested"),
            ("backend", "kv_cache_memory_bytes_effective"),
        )
    else:
        candidates = (
            ("backend", "kv_cache_budget_bytes"),
            ("backend", "kv_cache_memory_bytes_theoretical"),
        )
    for candidate in candidates:
        value = _path(record, *candidate)
        if value is not None:
            budget = int(_nonnegative_number(value, ".".join(candidate)))
            if budget > 0:
                return budget
    raise ValueError(f"{engine} record does not expose the working-set KV budget")


def _explicit_recomputed_prompt_tokens(
    record: Mapping[str, Any],
    prefix_cache: Mapping[str, Any],
) -> tuple[int | float | str, str]:
    for location in (
        _path(record, "run", "recomputed_prompt_tokens"),
        _path(record, "summary", "recomputed_prompt_tokens"),
        prefix_cache.get("recomputed_prompt_tokens"),
    ):
        if location is not None:
            return (
                _nonnegative_number(location, "recomputed_prompt_tokens"),
                "explicit",
            )
    return UNAVAILABLE, UNAVAILABLE


def _cached_token_signal(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int | str], int | str, str]:
    values = [request.get("cached_tokens") for request in requests]
    if any(value is None for value in values):
        unavailable = {
            "requests_with_cached_tokens": UNAVAILABLE,
            "requests_without_cached_tokens": UNAVAILABLE,
            "cached_tokens_total": UNAVAILABLE,
        }
        return unavailable, UNAVAILABLE, UNAVAILABLE
    cached = [int(_nonnegative_number(value, "cached_tokens")) for value in values]
    signal: dict[str, int | str] = {
        "requests_with_cached_tokens": sum(value > 0 for value in cached),
        "requests_without_cached_tokens": sum(value == 0 for value in cached),
        "cached_tokens_total": sum(cached),
    }
    prompt_values = [request.get("prompt_tokens") for request in requests]
    if any(value is None for value in prompt_values):
        return signal, UNAVAILABLE, UNAVAILABLE
    prompt = [int(_nonnegative_number(value, "prompt_tokens")) for value in prompt_values]
    recomputed = sum(
        max(0, prompt_tokens - cached_tokens)
        for prompt_tokens, cached_tokens in zip(
            prompt,
            cached,
            strict=True,
        )
    )
    return signal, recomputed, "derived_from_prompt_tokens_minus_cached_tokens"


def _cache_metrics(
    record: Mapping[str, Any],
    engine: str,
    variant: str,
    completed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prefix_raw = _path(record, "engine", "multimodal_prefix_cache")
    vision_raw = _path(record, "engine", "visual_embedding_cache")
    prefix = prefix_raw if isinstance(prefix_raw, Mapping) else {}
    vision = vision_raw if isinstance(vision_raw, Mapping) else {}
    prefix_metrics = {
        key: _optional_metric(prefix, key)
        for key in (
            "pre_admission_hits",
            "visual_hydration_skips",
            "stale_probe_fallbacks",
            "hits",
            "misses",
            "evictions",
            "entries",
            "resident_blocks",
            "tail_clone_hits",
            "tail_clone_admissions",
            "tail_clone_evictions",
            "tail_clone_reused_rows",
            "resident_tail_clone_blocks",
        )
    }
    vision_metrics = {
        key: _optional_metric(vision, key) for key in ("hits", "misses", "evictions", "entries")
    }
    cached_signal, derived_recomputed, derived_source = _cached_token_signal(completed)
    recomputed, recomputed_source = _explicit_recomputed_prompt_tokens(record, prefix)
    if recomputed == UNAVAILABLE and derived_recomputed != UNAVAILABLE:
        recomputed = derived_recomputed
        recomputed_source = derived_source

    if engine != "prism":
        resident_media: int | float | str = UNAVAILABLE
        resident_source = UNAVAILABLE
    elif variant == "vision_only":
        resident_media = vision_metrics["entries"]
        resident_source = "vision_embedding_cache.entries"
    else:
        resident_media = prefix_metrics["entries"]
        resident_source = "multimodal_prefix_cache.entries"
    return {
        "prefix_cache": prefix_metrics,
        "vision_cache": vision_metrics,
        "cached_token_signal": cached_signal,
        "resident_media_entries": resident_media,
        "resident_media_source": resident_source,
        "recomputed_prompt_tokens": recomputed,
        "recomputed_prompt_tokens_source": recomputed_source,
    }


def _compaction_metrics(record: Mapping[str, Any], engine: str) -> dict[str, Any]:
    raw = record.get("visual_compaction") if engine == "prism" else None
    compaction = raw if isinstance(raw, Mapping) else {}
    actual_compact_pages = _optional_metric(compaction, "physical_prompt_blocks")
    return {
        "decisions": _optional_metric(compaction, "decisions"),
        "effective_reclaims": _optional_metric(compaction, "effective_reclaims"),
        "dense_prompt_pages": _optional_metric(compaction, "dense_prompt_blocks"),
        "actual_compact_pages": actual_compact_pages,
        "actual_compact_pages_source": (
            "visual_compaction.physical_prompt_blocks"
            if actual_compact_pages != UNAVAILABLE
            else UNAVAILABLE
        ),
        "released_pages": _optional_metric(compaction, "released_blocks"),
    }


def _process_memory_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = _path(record, "memory", "process_device")
    process = raw if isinstance(raw, Mapping) else {}
    measurement = process.get("measurement")
    return {
        "peak_serving_mib": _optional_metric(process, "peak_serving_mib"),
        "after_llm_init_mib": _optional_metric(process, "after_llm_init_mib"),
        "after_benchmark_mib": _optional_metric(process, "after_benchmark_mib"),
        "measurement": measurement if isinstance(measurement, str) else UNAVAILABLE,
    }


def _summarize_record(
    record: Mapping[str, Any],
    *,
    source: str,
    source_sha256: str,
) -> dict[str, Any]:
    engine = _framework(record)
    audit = _mapping(
        _path(record, "workload", "working_set_plan"),
        "workload.working_set_plan",
    )
    workset_id = _nonempty_string(audit.get("workset_id"), "working-set id")
    if workset_id not in WORKSET_ORDER:
        raise ValueError(f"unsupported working-set id: {workset_id!r}")
    variant = (
        _nonempty_string(audit.get("variant"), "Prism working-set variant")
        if engine == "prism"
        else "engine_default"
    )
    if engine == "prism" and variant not in PRISM_VARIANT_ORDER:
        raise ValueError(f"unsupported Prism working-set variant: {variant!r}")
    group_ids = audit.get("group_ids")
    if (
        not isinstance(group_ids, list)
        or not group_ids
        or not all(isinstance(group_id, str) and group_id for group_id in group_ids)
    ):
        raise ValueError("working-set group_ids must be a non-empty string list")
    completed = _requests(record, engine)
    duration_s = float(_nonnegative_number(_path(record, "run", "duration_s"), "duration_s"))
    if duration_s <= 0.0:
        raise ValueError("run.duration_s must be positive")
    output_tokens = sum(int(request["output_tokens"]) for request in completed)
    measured_requests = int(
        _nonnegative_number(audit.get("measured_requests"), "measured requests")
    )
    if len(completed) != measured_requests:
        raise ValueError(
            f"{engine} completed {len(completed)} of {measured_requests} planned requests"
        )
    population_requests = int(
        _nonnegative_number(audit.get("population_requests"), "population requests")
    )
    population = _population_summary(record, engine, population_requests)
    cache = _cache_metrics(record, engine, variant, completed)
    return {
        "source": source,
        "source_sha256": source_sha256,
        "engine": engine,
        "variant": variant,
        "workset_id": workset_id,
        "plan_sha256": _nonempty_string(audit.get("plan_sha256"), "plan SHA256"),
        "model_config_sha256": _nonempty_string(
            _path(record, "model", "config_sha256"),
            "model config SHA256",
        ),
        "kv_budget_bytes": _kv_budget_bytes(record, engine),
        "request_trace_sha256": _nonempty_string(
            _path(record, "arrival", "trace_sha256"),
            "request trace SHA256",
        ),
        "prompt_token_ids_sha256": _nonempty_string(
            _path(record, "workload", "prompt_token_ids_sha256"),
            "prompt token IDs SHA256",
        ),
        "group_ids": list(group_ids),
        "dense_prefix_pages": int(
            _nonnegative_number(audit.get("dense_prefix_pages"), "dense prefix pages")
        ),
        "media_groups": len(group_ids),
        "questions": {
            "available": int(
                _nonnegative_number(audit.get("available_questions"), "available questions")
            ),
            "observed": int(
                _nonnegative_number(audit.get("observed_questions"), "observed questions")
            ),
            "measured_switches": int(
                _nonnegative_number(
                    audit.get("measured_question_switches"),
                    "measured question switches",
                )
            ),
        },
        "population": population,
        "measured_requests": measured_requests,
        "completed_requests": len(completed),
        "latency_ms": {
            "ttft": _distribution(completed, "ttft_ms"),
            "e2e": _distribution(completed, "latency_ms"),
        },
        "throughput": {
            "requests_per_s": len(completed) / duration_s,
            "output_tokens_per_s": output_tokens / duration_s,
        },
        "process_memory": _process_memory_metrics(record),
        "compaction": _compaction_metrics(record, engine),
        **cache,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str] | None = None,
    source_sha256: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate and aggregate compatible raw working-set records."""

    if not records:
        raise ValueError("at least one working-set record is required")
    names = list(sources or (f"record-{index}" for index in range(len(records))))
    digests = list(source_sha256 or (UNAVAILABLE for _ in records))
    if len(names) != len(records) or len(digests) != len(records):
        raise ValueError("record provenance counts must match input records")
    cells = [
        _summarize_record(record, source=name, source_sha256=digest)
        for record, name, digest in zip(records, names, digests, strict=True)
    ]

    plan_sha256 = {cell["plan_sha256"] for cell in cells}
    model_sha256 = {cell["model_config_sha256"] for cell in cells}
    budgets = {cell["kv_budget_bytes"] for cell in cells}
    if len(plan_sha256) != 1:
        raise ValueError("refusing to mix different working-set plan SHA256 values")
    if len(model_sha256) != 1:
        raise ValueError("refusing to mix different model identities")
    if len(budgets) != 1:
        raise ValueError("refusing to mix different KV budgets")

    seen: set[tuple[str, str, str]] = set()
    workset_identity: dict[str, dict[str, Any]] = {}
    for cell in cells:
        identity = {
            "request_trace_sha256": cell["request_trace_sha256"],
            "prompt_token_ids_sha256": cell["prompt_token_ids_sha256"],
            "group_ids": list(cell["group_ids"]),
            "dense_prefix_pages": cell["dense_prefix_pages"],
            "media_groups": cell["media_groups"],
            "questions": dict(cell["questions"]),
            "population_requests": cell["population"]["expected_requests"],
            "measured_requests": cell["measured_requests"],
        }
        previous = workset_identity.setdefault(cell["workset_id"], identity)
        if previous != identity:
            raise ValueError(
                "refusing to mix different request traces, prompt token identities, "
                f"or workload contracts for {cell['workset_id']}"
            )
        key = (cell["engine"], cell["variant"], cell["workset_id"])
        if key in seen:
            raise ValueError(f"duplicate working-set result cell: {key}")
        seen.add(key)

    missing_cells = _REQUIRED_CELLS - seen
    if missing_cells and not allow_partial:
        missing = ", ".join("/".join(cell) for cell in sorted(missing_cells))
        raise ValueError(f"working-set comparison is missing: {missing}")

    cells.sort(
        key=lambda cell: (
            WORKSET_ORDER.index(cell["workset_id"]),
            ENGINE_ORDER.index(cell["engine"]),
            (PRISM_VARIANT_ORDER.index(cell["variant"]) if cell["engine"] == "prism" else 0),
        )
    )
    return {
        "schema_version": 3,
        "record_type": "multimodal_working_set_summary",
        "matrix": {
            "required_cells": len(_REQUIRED_CELLS),
            "observed_cells": len(seen),
            "complete": not missing_cells,
            "partial_allowed": allow_partial,
            "missing_cells": [list(cell) for cell in sorted(missing_cells)],
        },
        "identity": {
            "plan_sha256": next(iter(plan_sha256)),
            "model_config_sha256": next(iter(model_sha256)),
            "kv_budget_bytes": next(iter(budgets)),
            "worksets": {
                workset_id: identity
                for workset_id, identity in sorted(
                    workset_identity.items(),
                    key=lambda item: WORKSET_ORDER.index(item[0]),
                )
            },
        },
        "cells": cells,
    }


_TABLE_COLUMNS = (
    ("Workset", "workset_id"),
    ("Label", "label"),
    ("Dense pages", "dense_prefix_pages"),
    ("Media groups", "media_groups"),
    ("Resident media", "resident_media_entries"),
    ("TTFT p50 ms", "ttft_p50_ms"),
    ("TTFT p99 ms", "ttft_p99_ms"),
    ("E2E p50 ms", "e2e_p50_ms"),
    ("E2E p99 ms", "e2e_p99_ms"),
    ("Output tok/s", "output_tokens_per_s"),
    ("Process peak MiB", "process_peak_mib"),
    ("Pre-admission hits", "prefix_pre_admission_hits"),
    ("Visual hydration skips", "visual_hydration_skips"),
    ("Stale-probe fallbacks", "stale_probe_fallbacks"),
    ("Prefix hits", "prefix_hits"),
    ("Prefix misses", "prefix_misses"),
    ("Prefix evictions", "prefix_evictions"),
    ("Prefix entries", "prefix_entries"),
    ("Prefix resident blocks", "prefix_resident_blocks"),
    ("Tail-clone hits", "tail_clone_hits"),
    ("Tail-clone admissions", "tail_clone_admissions"),
    ("Tail-clone evictions", "tail_clone_evictions"),
    ("Tail-clone resident blocks", "resident_tail_clone_blocks"),
    ("Dense prompt pages", "dense_prompt_pages"),
    ("Actual compact pages", "actual_compact_pages"),
    ("Released pages", "released_pages"),
    ("Vision hits", "vision_hits"),
    ("Vision misses", "vision_misses"),
    ("Requests with cached-token signal", "requests_with_cached_tokens"),
    ("Cached-token signal total", "cached_tokens_total"),
    ("Recomputed prompt tokens", "recomputed_prompt_tokens"),
)


def _table_row(
    cell: Mapping[str, Any] | None, workset: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if cell is None:
        return {
            "workset_id": workset["workset_id"],
            "label": label,
            "dense_prefix_pages": workset["dense_prefix_pages"],
            "media_groups": workset["media_groups"],
            **{key: UNAVAILABLE for _, key in _TABLE_COLUMNS[4:]},
        }
    return {
        "workset_id": cell["workset_id"],
        "label": label,
        "dense_prefix_pages": cell["dense_prefix_pages"],
        "media_groups": cell["media_groups"],
        "resident_media_entries": cell["resident_media_entries"],
        "ttft_p50_ms": cell["latency_ms"]["ttft"]["p50"],
        "ttft_p99_ms": cell["latency_ms"]["ttft"]["p99"],
        "e2e_p50_ms": cell["latency_ms"]["e2e"]["p50"],
        "e2e_p99_ms": cell["latency_ms"]["e2e"]["p99"],
        "output_tokens_per_s": cell["throughput"]["output_tokens_per_s"],
        "process_peak_mib": cell["process_memory"]["peak_serving_mib"],
        "prefix_pre_admission_hits": cell["prefix_cache"]["pre_admission_hits"],
        "visual_hydration_skips": cell["prefix_cache"]["visual_hydration_skips"],
        "stale_probe_fallbacks": cell["prefix_cache"]["stale_probe_fallbacks"],
        "prefix_hits": cell["prefix_cache"]["hits"],
        "prefix_misses": cell["prefix_cache"]["misses"],
        "prefix_evictions": cell["prefix_cache"]["evictions"],
        "prefix_entries": cell["prefix_cache"]["entries"],
        "prefix_resident_blocks": cell["prefix_cache"]["resident_blocks"],
        "tail_clone_hits": cell["prefix_cache"]["tail_clone_hits"],
        "tail_clone_admissions": cell["prefix_cache"]["tail_clone_admissions"],
        "tail_clone_evictions": cell["prefix_cache"]["tail_clone_evictions"],
        "resident_tail_clone_blocks": cell["prefix_cache"]["resident_tail_clone_blocks"],
        "dense_prompt_pages": cell["compaction"]["dense_prompt_pages"],
        "actual_compact_pages": cell["compaction"]["actual_compact_pages"],
        "released_pages": cell["compaction"]["released_pages"],
        "vision_hits": cell["vision_cache"]["hits"],
        "vision_misses": cell["vision_cache"]["misses"],
        "requests_with_cached_tokens": cell["cached_token_signal"]["requests_with_cached_tokens"],
        "cached_tokens_total": cell["cached_token_signal"]["cached_tokens_total"],
        "recomputed_prompt_tokens": cell["recomputed_prompt_tokens"],
    }


def build_tables(summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build complete Prism-ablation and three-engine comparison rows."""

    cells = {
        (cell["engine"], cell["variant"], cell["workset_id"]): cell for cell in summary["cells"]
    }
    worksets = summary["identity"]["worksets"]
    prism_rows = []
    engine_rows = []
    for workset_id in WORKSET_ORDER:
        if workset_id not in worksets:
            continue
        workset = {"workset_id": workset_id, **worksets[workset_id]}
        for variant in PRISM_VARIANT_ORDER:
            prism_rows.append(
                _table_row(cells.get(("prism", variant, workset_id)), workset, variant)
            )
        for engine in ENGINE_ORDER:
            variant = "compact_prefix" if engine == "prism" else "engine_default"
            engine_rows.append(
                _table_row(cells.get((engine, variant, workset_id)), workset, engine)
            )
    return prism_rows, engine_rows


def _formatted(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for _, key in _TABLE_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _formatted(row[key]) for _, key in _TABLE_COLUMNS})


def _write_markdown(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [header for header, _ in _TABLE_COLUMNS]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [_formatted(row[key]).replace("|", "\\|") for _, key in _TABLE_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _series_label(cell: Mapping[str, Any]) -> str:
    if cell["engine"] == "prism":
        return {
            "vision_only": "Prism Vision only",
            "dense_prefix": "Prism Dense Prefix",
            "compact_prefix": "Prism Compact Prefix",
        }[str(cell["variant"])]
    return {"vllm": "vLLM", "sglang": "SGLang"}[str(cell["engine"])]


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    y_label: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    colors: Mapping[str, str],
    dashed: frozenset[str] = frozenset(),
) -> None:
    left, top, right, bottom = box
    title_font = _font(20, bold=True)
    axis_font = _font(14)
    draw.text((left, top - 34), title, fill="#202124", font=title_font)
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        draw.rectangle(box, outline="#9aa0a6", width=2)
        draw.text((left + 20, top + 20), "unavailable", fill="#5f6368", font=axis_font)
        return
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, max(y_values) * 1.10
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_max <= 0.0:
        y_max = 1.0

    def xy(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = left + int((x - x_min) / (x_max - x_min) * (right - left))
        py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
        return px, py

    for tick in range(6):
        y_value = y_max * tick / 5
        py = bottom - int((bottom - top) * tick / 5)
        draw.line((left, py, right, py), fill="#e8eaed", width=1)
        draw.text((left - 62, py - 8), f"{y_value:.1f}", fill="#5f6368", font=axis_font)
    for x_value in sorted(set(x_values)):
        px, _ = xy((x_value, 0.0))
        draw.line((px, top, px, bottom), fill="#f1f3f4", width=1)
        draw.text((px - 18, bottom + 8), f"{x_value:.0f}", fill="#5f6368", font=axis_font)
    draw.line((left, bottom, right, bottom), fill="#5f6368", width=2)
    draw.line((left, top, left, bottom), fill="#5f6368", width=2)
    draw.text(
        (left + (right - left) // 2 - 55, bottom + 35),
        "Dense prefix pages",
        fill="#3c4043",
        font=axis_font,
    )
    if y_label:
        draw.text((left - 70, top - 3), y_label, fill="#3c4043", font=axis_font)

    for label, points in series.items():
        ordered = sorted(points)
        coordinates = [xy(point) for point in ordered]
        color = colors[label]
        width = 4 if label == "Prism Compact Prefix" else 2
        if len(coordinates) > 1:
            if label in dashed:
                for start, end in zip(coordinates, coordinates[1:], strict=False):
                    segments = 12
                    for segment in range(0, segments, 2):
                        ratio_a = segment / segments
                        ratio_b = min(1.0, (segment + 1) / segments)
                        a = (
                            int(start[0] + (end[0] - start[0]) * ratio_a),
                            int(start[1] + (end[1] - start[1]) * ratio_a),
                        )
                        b = (
                            int(start[0] + (end[0] - start[0]) * ratio_b),
                            int(start[1] + (end[1] - start[1]) * ratio_b),
                        )
                        draw.line((*a, *b), fill=color, width=width)
            else:
                draw.line(coordinates, fill=color, width=width)
        for px, py in coordinates:
            radius = 5 if width == 4 else 4
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)


def _render_plot(path: Path, summary: Mapping[str, Any]) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 28),
        "Repeated visual-context working set: capacity, residency, and TTFT",
        fill="#202124",
        font=_font(27, bold=True),
    )
    cells = list(summary["cells"])
    worksets = summary["identity"]["worksets"]
    annotations = [
        f"{name}: {details['dense_prefix_pages']} dense pages / "
        f"{details['media_groups']} media groups"
        for name, details in worksets.items()
    ]
    draw.text((72, 72), " | ".join(annotations), fill="#5f6368", font=_font(15))

    palette = {
        "Prism Vision only": "#4285f4",
        "Prism Dense Prefix": "#f9ab00",
        "Prism Compact Prefix": "#7b1fa2",
        "vLLM": "#0f9d58",
        "SGLang": "#d93025",
    }
    resident_series: dict[str, list[tuple[float, float]]] = {}
    ttft_series: dict[str, list[tuple[float, float]]] = {}
    dashed: set[str] = set()
    colors: dict[str, str] = {}
    for cell in cells:
        label = _series_label(cell)
        color = palette[label]
        colors[label] = color
        x_value = float(cell["dense_prefix_pages"])
        resident = _numeric(cell["resident_media_entries"])
        if resident is not None:
            resident_series.setdefault(label, []).append((x_value, resident))
        ttft_series.setdefault(label, []).append(
            (x_value, float(cell["latency_ms"]["ttft"]["p50"]))
        )
    _draw_panel(
        draw,
        (105, 160, 750, 690),
        title="Resident Prefix entries reported by Prism",
        y_label="",
        series=resident_series,
        colors=colors,
    )
    _draw_panel(
        draw,
        (900, 160, 1535, 690),
        title="TTFT p50 across engines (ms)",
        y_label="",
        series=ttft_series,
        colors=colors,
        dashed=frozenset(dashed),
    )
    legend_y = 770
    x = 85
    for label in palette:
        if not any(_series_label(cell) == label for cell in cells):
            continue
        draw.line((x, legend_y, x + 28, legend_y), fill=palette[label], width=4)
        draw.text((x + 36, legend_y - 9), label, fill="#3c4043", font=_font(14))
        x += 260
    draw.text(
        (85, 830),
        "vLLM and SGLang do not expose a directly comparable resident-entry counter.",
        fill="#5f6368",
        font=_font(14),
    )
    image.save(path, format="PNG")


def write_outputs(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write the machine-readable summary, plot, and comparison tables."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": directory / "working_set_summary.json",
        "main_png": directory / "working_set_summary.png",
        "prism_csv": directory / "working_set_prism_ablation.csv",
        "prism_markdown": directory / "working_set_prism_ablation.md",
        "engines_csv": directory / "working_set_engine_comparison.csv",
        "engines_markdown": directory / "working_set_engine_comparison.md",
    }
    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prism_rows, engine_rows = build_tables(summary)
    _write_csv(paths["prism_csv"], prism_rows)
    _write_csv(paths["engines_csv"], engine_rows)
    _write_markdown(paths["prism_markdown"], "Prism working-set ablation", prism_rows)
    _write_markdown(paths["engines_markdown"], "Three-engine working-set comparison", engine_rows)
    _render_plot(paths["main_png"], summary)
    return {name: str(path) for name, path in paths.items()}


def _load_inputs(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records = []
    sources = []
    digests = []
    for path in paths:
        payload = path.read_bytes()
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")
        records.append(value)
        sources.append(path.name)
        digests.append(sha256_bytes(payload))
    return records, sources, digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw labeled working-set JSON files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="summarize the available engine and workset combinations",
    )
    args = parser.parse_args()
    records, sources, digests = _load_inputs(args.inputs)
    summary = summarize_records(
        records,
        sources=sources,
        source_sha256=digests,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(write_outputs(summary, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
