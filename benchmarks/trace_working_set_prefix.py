#!/usr/bin/env python3
"""Capture one cold-to-prefix-hit Prism working-set trace with Nsight Systems.

Run this entry point under ``nsys profile --capture-range=cudaProfilerApi``.
Model construction and warmup stay outside the capture range.  The captured
range contains exactly one cold request followed by one same-media,
different-question multimodal prefix hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACE_SCHEMA_VERSION = 1
TRACE_RECORD_TYPE = "working_set_prefix_trace"
WORKSET_ID = "knee"
VARIANT = "compact_prefix"
MODE = "visual_compact_scaled_fp8_compile_graph"
_COMPLETED_REASONS = frozenset({"eos", "length", "stop"})
_PREFIX_COUNTERS = (
    "pre_admission_hits",
    "visual_hydration_skips",
    "stale_probe_fallbacks",
    "hits",
    "misses",
    "admissions",
    "evictions",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _objects_by_id(
    values: object,
    *,
    id_key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        item = _mapping(value, f"{label}[{index}]")
        item_id = _nonempty_string(item.get(id_key), f"{label}[{index}].{id_key}")
        if item_id in indexed:
            raise ValueError(f"duplicate {label} id: {item_id!r}")
        indexed[item_id] = item
    return indexed


def _distinct_question_pair(
    group: Mapping[str, Any],
    *,
    measured_sample_ids: set[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    raw_samples = group.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        return None
    samples = [_mapping(sample, "group sample") for sample in raw_samples]
    first = samples[0]
    first_id = _nonempty_string(first.get("sample_id"), "sample.sample_id")
    first_prompt = _nonempty_string(first.get("source_prompt"), "sample.source_prompt")
    for second in samples[1:]:
        second_id = _nonempty_string(second.get("sample_id"), "sample.sample_id")
        second_prompt = _nonempty_string(second.get("source_prompt"), "sample.source_prompt")
        if (
            second_id in measured_sample_ids
            and first_id != second_id
            and first_prompt != second_prompt
        ):
            return first, second
    return None


def select_knee_trace_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Select a deterministic multi-question knee group and another warmup group."""

    groups = _objects_by_id(plan.get("groups"), id_key="group_id", label="plan.groups")
    worksets = _objects_by_id(plan.get("worksets"), id_key="id", label="plan.worksets")
    knee = worksets.get(WORKSET_ID)
    if knee is None:
        raise ValueError("working-set plan has no knee workset")
    raw_group_ids = knee.get("group_ids")
    if (
        not isinstance(raw_group_ids, list)
        or len(raw_group_ids) < 2
        or not all(isinstance(group_id, str) and group_id for group_id in raw_group_ids)
    ):
        raise ValueError("knee trace requires at least two media groups")
    unknown = [group_id for group_id in raw_group_ids if group_id not in groups]
    if unknown:
        raise ValueError(f"knee workset references unknown groups: {unknown}")

    raw_measured = knee.get("measured_requests")
    if not isinstance(raw_measured, list) or not raw_measured:
        raise ValueError("knee trace requires a non-empty measured request schedule")
    measured_sample_ids: dict[str, set[str]] = {}
    for index, raw_request in enumerate(raw_measured):
        request = _mapping(raw_request, f"knee measured_requests[{index}]")
        group_id = _nonempty_string(request.get("group_id"), "measured request group_id")
        sample_id = _nonempty_string(request.get("sample_id"), "measured request sample_id")
        if group_id not in raw_group_ids:
            raise ValueError(f"measured request references non-knee group: {group_id!r}")
        measured_sample_ids.setdefault(group_id, set()).add(sample_id)

    target_group = None
    question_pair = None
    for group_id in raw_group_ids:
        candidate = groups[group_id]
        pair = _distinct_question_pair(
            candidate,
            measured_sample_ids=measured_sample_ids.get(group_id, set()),
        )
        if pair is not None:
            target_group = candidate
            question_pair = pair
            break
    if target_group is None or question_pair is None:
        raise ValueError("knee workset has no media group with two distinct questions")

    target_group_id = str(target_group["group_id"])
    warmup_group_id = next(group_id for group_id in raw_group_ids if group_id != target_group_id)
    warmup_group = groups[warmup_group_id]
    raw_warmup_samples = warmup_group.get("samples")
    if not isinstance(raw_warmup_samples, list) or not raw_warmup_samples:
        raise ValueError("warmup media group has no questions")
    warmup_sample = _mapping(raw_warmup_samples[0], "warmup sample")
    cold_sample, hit_sample = question_pair

    def sample_identity(sample: Mapping[str, Any]) -> dict[str, str]:
        prompt = _nonempty_string(sample.get("source_prompt"), "sample.source_prompt")
        return {
            "sample_id": _nonempty_string(sample.get("sample_id"), "sample.sample_id"),
            "source_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    return {
        "workset_id": WORKSET_ID,
        "variant": VARIANT,
        "target": {
            "group_id": target_group_id,
            "ordered_media_sha256": list(target_group.get("ordered_media_sha256", [])),
            "dense_prefix_pages": _nonnegative_int(
                target_group.get("dense_prefix_pages"),
                "target dense_prefix_pages",
            ),
            "cold": sample_identity(cold_sample),
            "prefix_hit": sample_identity(hit_sample),
        },
        "warmup": {
            "group_id": warmup_group_id,
            **sample_identity(warmup_sample),
        },
    }


def compact_prism_config(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed Prism Compact configuration bound to the plan."""

    model = _mapping(plan.get("model"), "plan.model")
    budget = _mapping(plan.get("kv_budget"), "plan.kv_budget")
    processor = _mapping(plan.get("processor"), "plan.processor")
    serving = _mapping(plan.get("serving"), "plan.serving")
    return {
        "mode": MODE,
        "tensor_parallel_size": 1,
        "max_model_len": _nonnegative_int(model.get("max_model_len"), "max_model_len"),
        "max_num_batched_tokens": _nonnegative_int(
            model.get("max_model_len"),
            "max_num_batched_tokens",
        ),
        "max_num_seqs": _nonnegative_int(serving.get("max_num_seqs"), "max_num_seqs"),
        "gpu_memory_utilization": 0.9,
        "num_kvcache_blocks": _nonnegative_int(budget.get("pages"), "kv_budget.pages"),
        "kvcache_block_size": _nonnegative_int(
            budget.get("page_size_tokens"),
            "kv_budget.page_size_tokens",
        ),
        "max_chunk_size": _nonnegative_int(
            serving.get("max_chunk_size"),
            "max_chunk_size",
        ),
        "enable_prefix_caching": True,
        "max_queue_size": None,
        "scheduler_policy": "fcfs",
        "max_consecutive_prefill_batches": 1,
        "heavy_prefill_vision_patch_threshold": 4096,
        "min_decode_batches_between_heavy_prefills": 32,
        "max_light_prefill_bypasses_per_heavy": 2,
        "visual_pruning_keep_ratio": 0.6,
        "visual_pruning_min_keep_tokens": 768,
        "visual_pruning_video_min_keep_tokens": 256,
        "visual_pruning_strategy": "uniform",
        "visual_pruning_attention_last_n_layers": 1,
        "logits_precision": "model",
        "mlp_projection_mode": "packed",
        "enable_cooperative_prefill": False,
        "cooperative_prefill_layer_quantum": 1,
        "cooperative_prefill_vision_block_quantum": None,
        "enable_vision_tensor_cudagraph": False,
        "enable_visual_embedding_cache": True,
        "image_max_pixels": _nonnegative_int(
            processor.get("image_max_pixels"),
            "processor.image_max_pixels",
        ),
    }


def _counter_deltas(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, int]:
    deltas = {}
    for key in _PREFIX_COUNTERS:
        before_value = _nonnegative_int(before.get(key), f"{phase}.before.{key}")
        after_value = _nonnegative_int(after.get(key), f"{phase}.after.{key}")
        if after_value < before_value:
            raise ValueError(f"{phase} prefix counter decreased: {key}")
        deltas[key] = after_value - before_value
    return deltas


def validate_prefix_hit_evidence(
    before: Mapping[str, Any],
    after_cold: Mapping[str, Any],
    after_hit: Mapping[str, Any],
    cold_request_metric: Mapping[str, Any],
    hit_request_metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the second request is an observed, non-stale prefix hit."""

    cold = _counter_deltas(before, after_cold, phase="cold_request")
    hit = _counter_deltas(after_cold, after_hit, phase="prefix_hit_request")
    if cold["pre_admission_hits"] != 0 or cold["hits"] != 0:
        raise ValueError("cold request unexpectedly reused a multimodal prefix")
    if cold["misses"] != 1 or cold["admissions"] != 1:
        raise ValueError("cold request did not miss and admit exactly one multimodal prefix")
    cold_finish_reason = cold_request_metric.get("finish_reason")
    if cold_finish_reason not in _COMPLETED_REASONS:
        raise ValueError(f"cold request did not complete: {cold_finish_reason!r}")
    cold_cached_tokens = _nonnegative_int(
        cold_request_metric.get("cached_tokens"),
        "cold-request cached_tokens",
    )
    if cold_cached_tokens != 0:
        raise ValueError("cold request unexpectedly reported cached tokens")
    if hit["pre_admission_hits"] != 1:
        raise ValueError("second request did not produce exactly one pre-admission hit")
    if hit["visual_hydration_skips"] != 1:
        raise ValueError("second request did not skip exactly one visual hydration")
    if hit["hits"] != 1 or hit["misses"] != 0:
        raise ValueError("second request was not exactly one realized prefix-cache hit")
    if hit["stale_probe_fallbacks"] != 0:
        raise ValueError("second request fell back after a stale prefix probe")

    finish_reason = hit_request_metric.get("finish_reason")
    if finish_reason not in _COMPLETED_REASONS:
        raise ValueError(f"prefix-hit request did not complete: {finish_reason!r}")
    cached_tokens = _nonnegative_int(
        hit_request_metric.get("cached_tokens"),
        "prefix-hit cached_tokens",
    )
    prompt_tokens = _nonnegative_int(
        hit_request_metric.get("prompt_tokens"),
        "prefix-hit prompt_tokens",
    )
    if cached_tokens <= 0:
        raise ValueError("prefix-hit request reported no actual cached tokens")
    if cached_tokens > prompt_tokens:
        raise ValueError("prefix-hit cached tokens exceed prompt tokens")
    return {
        "passed": True,
        "cold_request_counter_deltas": cold,
        "cold_request_actual_cached_tokens": cold_cached_tokens,
        "prefix_hit_request_counter_deltas": hit,
        "actual_cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_fraction": cached_tokens / prompt_tokens,
        "stale_probe_fallback": False,
    }


def _payload_for(
    payloads: Sequence[dict[str, Any]],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    group_id: str,
    sample_id: str,
    phase: str,
) -> dict[str, Any]:
    if not (len(payloads) == len(group_ids) == len(sample_ids)):
        raise ValueError(f"{phase} materialized request arrays have different lengths")
    for payload, current_group_id, current_sample_id in zip(
        payloads,
        group_ids,
        sample_ids,
        strict=True,
    ):
        if current_group_id == group_id and current_sample_id == sample_id:
            return payload
    raise ValueError(
        f"{phase} schedule has no request for group={group_id!r}, sample={sample_id!r}"
    )


def _one_request_metric(run_record: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_requests = run_record.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 1:
        raise ValueError("trace run must contain exactly one request result")
    result = _mapping(raw_requests[0], "trace request result")
    request_id = _nonnegative_int(result.get("request_id"), "trace request_id")
    metrics = _mapping(run_record.get("engine_metrics"), "trace engine_metrics")
    raw_metric_requests = metrics.get("requests")
    if not isinstance(raw_metric_requests, list):
        raise ValueError("trace engine metrics must contain requests")
    matches = [
        _mapping(metric, "trace request metric")
        for metric in raw_metric_requests
        if isinstance(metric, Mapping) and metric.get("request_id") == request_id
    ]
    if len(matches) != 1:
        raise ValueError(f"trace run has {len(matches)} metrics for request {request_id}")
    return matches[0]


def _online_request(request_key: str, payload: dict[str, Any], sampling: Any) -> Any:
    from prism_infer.engine.online import OnlineRequest

    return OnlineRequest(
        request_key=request_key,
        arrival_offset_s=0.0,
        payload=payload,
        sampling_params=sampling,
    )


def _run_trace(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from benchmarks.bench_online import _build_engine
    from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
    from benchmarks.working_set_workload import (
        materialize_working_set,
        verify_working_set_model,
        verify_working_set_processor,
    )
    from prism_infer import SamplingParams
    from prism_infer.analysis.performance_profile import (
        performance_profile,
        profile_region,
        validate_performance_profile_record,
    )
    from prism_infer.engine.online import OnlineServingSession

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the working-set prefix trace")
    torch.set_num_threads(args.online_cpu_intraop_threads)
    plan_path = Path(args.working_set_plan).resolve()
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    working_set = materialize_working_set(
        plan_path,
        workset_id=WORKSET_ID,
        materialized_root=args.materialized_root,
    )
    llm = None
    try:
        selection = select_knee_trace_contract(working_set.plan)
        fixed_config = compact_prism_config(working_set.plan)
        engine_args = SimpleNamespace(model=args.model, **fixed_config)
        model_verification = verify_working_set_model(working_set.plan, args.model)
        llm = _build_engine(engine_args)
        processor_verification = verify_working_set_processor(
            llm.vl_processor,
            working_set.plan,
        )
        session = OnlineServingSession(llm)
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(working_set.plan["traffic"]["max_new_tokens"]),
            ignore_eos=True,
        )

        warmup = selection["warmup"]
        target = selection["target"]
        cold = target["cold"]
        prefix_hit = target["prefix_hit"]
        warmup_payload = _payload_for(
            working_set.population_payloads,
            working_set.population_group_ids,
            working_set.population_sample_ids,
            group_id=warmup["group_id"],
            sample_id=warmup["sample_id"],
            phase="population",
        )
        cold_payload = _payload_for(
            working_set.population_payloads,
            working_set.population_group_ids,
            working_set.population_sample_ids,
            group_id=target["group_id"],
            sample_id=cold["sample_id"],
            phase="population",
        )
        hit_payload = _payload_for(
            working_set.measured_payloads,
            working_set.measured_group_ids,
            working_set.measured_sample_ids,
            group_id=target["group_id"],
            sample_id=prefix_hit["sample_id"],
            phase="measured",
        )

        warmup_run = session.run(
            (_online_request("trace-warmup", warmup_payload, sampling),)
        ).to_record()
        warmup_metric = _one_request_metric(warmup_run)
        if warmup_metric.get("finish_reason") not in _COMPLETED_REASONS:
            raise RuntimeError("model warmup did not complete")
        torch.cuda.synchronize()
        llm.reset_metrics()
        session.reset_metrics()

        prefix_before = llm.multimodal_prefix_cache_metadata()
        profile_metadata = {
            "record_type": TRACE_RECORD_TYPE,
            "workset_id": WORKSET_ID,
            "variant": VARIANT,
            "target_group_id": target["group_id"],
        }
        with performance_profile(metadata=profile_metadata) as profile_session:
            profiler_started = False
            try:
                torch.cuda.cudart().cudaProfilerStart()
                profiler_started = True
                with profile_region(
                    "cold_request",
                    metadata={"sample_id": cold["sample_id"]},
                ):
                    cold_run = session.run(
                        (_online_request("trace-cold", cold_payload, sampling),)
                    ).to_record()
                prefix_after_cold = llm.multimodal_prefix_cache_metadata()
                with profile_region(
                    "prefix_hit_request",
                    metadata={"sample_id": prefix_hit["sample_id"]},
                ):
                    hit_run = session.run(
                        (_online_request("trace-prefix-hit", hit_payload, sampling),)
                    ).to_record()
                prefix_after_hit = llm.multimodal_prefix_cache_metadata()
            finally:
                if profiler_started:
                    try:
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.cudart().cudaProfilerStop()

        profile_record = profile_session.to_record()
        validate_performance_profile_record(profile_record)
        cold_metric = _one_request_metric(cold_run)
        hit_metric = _one_request_metric(hit_run)
        evidence = validate_prefix_hit_evidence(
            prefix_before,
            prefix_after_cold,
            prefix_after_hit,
            cold_metric,
            hit_metric,
        )
        git = collect_git_metadata(REPO_ROOT)
        gpu = collect_gpu_metadata(0)
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "record_type": TRACE_RECORD_TYPE,
            "identity": {
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha256,
                "model": model_verification,
                "processor": processor_verification,
                "git": git.as_dict(),
                "gpu": gpu.detailed_dict(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "selection": selection,
            "engine": {
                "variant": VARIANT,
                "fixed_config": fixed_config,
                "max_new_tokens": int(working_set.plan["traffic"]["max_new_tokens"]),
            },
            "warmup": {
                "outside_capture": True,
                "group_id": warmup["group_id"],
                "sample_id": warmup["sample_id"],
                "run": warmup_run,
            },
            "capture": {
                "range": "cudaProfilerApi",
                "nvtx_ranges": [
                    "prism::cold_request",
                    "prism::prefix_hit_request",
                ],
                "warmup_group_is_distinct": warmup["group_id"] != target["group_id"],
                "prefix_cache_before": prefix_before,
                "prefix_cache_after_cold": prefix_after_cold,
                "prefix_cache_after_hit": prefix_after_hit,
                "cold_run": cold_run,
                "prefix_hit_run": hit_run,
            },
            "evidence": evidence,
            "semantic_profile": profile_record,
        }
    finally:
        if llm is not None:
            llm.exit()
        working_set.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen3-VL model snapshot")
    parser.add_argument("--working-set-plan", required=True, help="final working-set plan JSON")
    parser.add_argument(
        "--materialized-root",
        required=True,
        help="root containing the plan's materialized MuirBench media",
    )
    parser.add_argument("--output", required=True, type=Path, help="trace evidence JSON")
    parser.add_argument("--online-cpu-intraop-threads", type=int, default=8)
    args = parser.parse_args()
    if args.online_cpu_intraop_threads <= 0:
        raise SystemExit("--online-cpu-intraop-threads must be positive")
    record = _run_trace(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "evidence": record["evidence"]}))


if __name__ == "__main__":
    main()
