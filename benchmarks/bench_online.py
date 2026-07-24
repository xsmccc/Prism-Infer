"""P7.3 single-node online arrival/continuous-batching benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_system import (
    DEFAULT_MANIFEST,
    MODE_SPECS,
)
from benchmarks.harness import (
    collect_git_metadata,
    collect_gpu_metadata,
    find_workload_case,
    materialize_requests,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.analysis.online_serving import (
    ONLINE_BENCHMARK_SCHEMA_VERSION,
    summarize_distribution,
    summarize_online_run,
    validate_online_benchmark_record,
)
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes
from prism_infer.engine.online import OnlineRequest, OnlineServingSession


P12_ONLINE_MODES = (
    "off_eager",
    "off_graph",
    "visual_compact_graph",
    "bf16_compile_graph",
    "scaled_fp8_kv_compile_graph",
    "visual_compact_scaled_fp8_compile_graph",
)
H3_PROFILES = ("primary", "conditional_video")


def _arrival_offsets(
    count: int,
    *,
    process: str,
    request_rate: float,
    seed: int,
) -> list[float]:
    if count <= 0:
        raise ValueError("online request count must be positive")
    if process == "burst":
        return [0.0] * count
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if process == "constant":
        return [index / request_rate for index in range(count)]
    if process != "poisson":
        raise ValueError(f"unsupported arrival process: {process!r}")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, count):
        offsets.append(offsets[-1] + rng.expovariate(request_rate))
    return offsets


def _online_requests(
    payloads: list[dict],
    *,
    count: int,
    process: str,
    request_rate: float,
    seed: int,
    sampling: SamplingParams,
    key_prefix: str,
) -> tuple[OnlineRequest, ...]:
    offsets = _arrival_offsets(
        count,
        process=process,
        request_rate=request_rate,
        seed=seed,
    )
    return tuple(
        OnlineRequest(
            request_key=f"{key_prefix}-{index:05d}",
            arrival_offset_s=offset,
            payload=payloads[index % len(payloads)],
            sampling_params=sampling,
        )
        for index, offset in enumerate(offsets)
    )


def _smooth_weighted_period(
    classes: list[dict[str, object]],
) -> tuple[str, ...]:
    """Build the frozen period-10 smooth weighted round-robin schedule."""

    case_ids = [str(item["case_id"]) for item in classes]
    counts = [int(round(float(item["weight"]) * 10)) for item in classes]
    if sum(counts) != 10 or any(count <= 0 for count in counts):
        raise ValueError(f"H3 class weights must form a positive period 10: {counts}")
    current = [0] * len(counts)
    period: list[str] = []
    for _ in range(sum(counts)):
        for index, count in enumerate(counts):
            current[index] += count
        selected = max(
            range(len(counts)),
            key=lambda index: (current[index], -index),
        )
        period.append(case_ids[selected])
        current[selected] -= sum(counts)
    if {case_id: period.count(case_id) for case_id in case_ids} != dict(
        zip(case_ids, counts, strict=True)
    ):
        raise RuntimeError("smooth weighted H3 schedule changed class counts")
    return tuple(period)


def _h3_payload_schedule(
    manifest: dict[str, object],
    *,
    profile: str,
    count: int,
) -> tuple[list[dict], list[str], dict[str, object]]:
    """Materialize the frozen H3 class mix without changing arrival timing."""

    headline = manifest.get("p9_protocol", {})
    if not isinstance(headline, dict):
        raise ValueError("H3 profile requires manifest.p9_protocol")
    h3 = headline.get("headline", {}).get("H3")
    if not isinstance(h3, dict):
        raise ValueError("H3 profile requires p9_protocol.headline.H3")
    field = "primary_classes" if profile == "primary" else "conditional_video_classes"
    classes = h3.get(field)
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"H3 profile is missing {field}")
    period = _smooth_weighted_period(classes)
    payload_by_case: dict[str, dict] = {}
    for item in classes:
        case_id = str(item["case_id"])
        case = find_workload_case(manifest, case_id)
        payloads = materialize_requests(case, repo_root=REPO_ROOT)
        if len(payloads) != 1:
            raise ValueError(f"H3 class must materialize one request: {case_id}")
        payload_by_case[case_id] = payloads[0]
    request_classes = [period[index % len(period)] for index in range(count)]
    payloads = [payload_by_case[case_id] for case_id in request_classes]
    return payloads, request_classes, {
        "profile": profile,
        "class_field": field,
        "class_schedule": h3.get("class_schedule"),
        "materialized_schedule_algorithm": (
            "smooth_weighted_round_robin_integer_counts"
        ),
        "period": list(period),
        "classes": classes,
        "arrival_process": h3.get("arrival_process"),
        "request_rates_per_second": h3.get("request_rates_per_second"),
        "completed_requests_per_run": h3.get("completed_requests_per_run"),
        "arrival_seeds": h3.get("arrival_seeds"),
        "max_tokens": h3.get("max_tokens"),
        "max_model_len": h3.get("max_model_len"),
        "ttft_slo_formula": h3.get("ttft_slo_formula"),
        "tpot_slo_formula": h3.get("tpot_slo_formula"),
        "goodput_unit": h3.get("goodput_unit"),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_audit(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    prompt_ids = []
    prompt_counts = []
    by_class: dict[str, list[list[int]]] = {}
    for result, case_id in zip(run.requests, request_classes, strict=True):
        seq = llm.scheduler.requests[result.request_id]
        ids = list(seq.token_ids[: seq.num_prompt_tokens])
        prompt_ids.append(ids)
        prompt_counts.append(len(ids))
        by_class.setdefault(case_id, []).append(ids)
    return {
        "prompt_tokens": sum(prompt_counts),
        "prompt_tokens_per_request": prompt_counts,
        "prompt_token_ids_sha256": _canonical_sha256(prompt_ids),
        "prompt_token_ids_sha256_by_class": {
            case_id: _canonical_sha256(rows)
            for case_id, rows in sorted(by_class.items())
        },
    }


def _visual_compaction_summary(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    class_by_id = {
        result.request_id: case_id
        for result, case_id in zip(run.requests, request_classes, strict=True)
    }
    decisions: list[tuple[str, dict[str, object]]] = []
    for result in run.requests:
        request_id = result.request_id
        seq = llm.scheduler.requests[request_id]
        record = seq.visual_pruning_decision_record
        if record and record.get("physical_compaction"):
            decisions.append((class_by_id[request_id], record))

    def aggregate(rows: list[dict[str, object]]) -> dict[str, int]:
        return {
            "decisions": len(rows),
            "effective_reclaims": sum(
                bool(row.get("released_block_ids")) for row in rows
            ),
            "logical_prompt_tokens": sum(
                int(row["logical_prompt_tokens"]) for row in rows
            ),
            "physical_prompt_tokens": sum(
                int(row["physical_prompt_kv_tokens"]) for row in rows
            ),
            "released_blocks": sum(
                len(row.get("released_block_ids", [])) for row in rows
            ),
            "dense_prompt_blocks": sum(
                len(row.get("old_block_table", [])) for row in rows
            ),
            "physical_prompt_blocks": sum(
                len(row.get("new_block_table", [])) for row in rows
            ),
            "dropped_visual_tokens": sum(
                int(row["dropped_visual_tokens"]) for row in rows
            ),
        }

    by_class: dict[str, list[dict[str, object]]] = {}
    for case_id, record in decisions:
        by_class.setdefault(case_id, []).append(record)
    return {
        **aggregate([record for _, record in decisions]),
        "by_class": {
            case_id: aggregate(rows)
            for case_id, rows in sorted(by_class.items())
        },
    }


def _execution_evidence(llm, run_record: dict[str, object]) -> dict[str, object]:
    decode_batch_counts = Counter(
        int(batch["batch_size"])
        for batch in run_record["engine_metrics"]["batches"]
        if batch["phase"] == "decode"
    )
    max_decode_batch = max(decode_batch_counts, default=1)
    cuda_graph = llm.model_runner.cudagraph_metadata(max_decode_batch)
    capture_sizes = list(cuda_graph["batch_sizes"])
    replay_counts = []
    if cuda_graph["enabled"]:
        for actual_batch_size, count in sorted(decode_batch_counts.items()):
            captured_batch_size = next(
                size for size in capture_sizes if size >= actual_batch_size
            )
            replay_counts.append(
                {
                    "actual_batch_size": actual_batch_size,
                    "captured_batch_size": captured_batch_size,
                    "count": count,
                }
            )
    return {
        "observed_decode_batch_counts": [
            {"batch_size": batch_size, "count": count}
            for batch_size, count in sorted(decode_batch_counts.items())
        ],
        "cuda_graph": {
            **cuda_graph,
            "replay_counts": replay_counts,
            "total_observed_replays": (
                sum(decode_batch_counts.values()) if cuda_graph["enabled"] else 0
            ),
        },
        "torch_compile": llm.model_runner.compile_metadata(),
        "vision_tensor_cudagraph": (
            llm.model_runner.vision_tensor_cudagraph_metadata()
        ),
    }


def _terminal_failure_summary(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    failures = []
    reasons: dict[str, int] = {}
    for result, case_id in zip(run.requests, request_classes, strict=True):
        if result.finish_reason not in {"rejected", "cancelled"}:
            continue
        seq = llm.scheduler.requests[result.request_id]
        transition = (
            None if not seq.lifecycle.transitions else seq.lifecycle.transitions[-1]
        )
        detail = None if transition is None else transition.reason
        reason_key = detail or result.finish_reason
        reasons[reason_key] = reasons.get(reason_key, 0) + 1
        failures.append(
            {
                "request_id": result.request_id,
                "request_class": case_id,
                "finish_reason": result.finish_reason,
                "detail": detail,
                "prompt_tokens": seq.num_prompt_tokens,
                "dense_kv_blocks": seq.num_blocks,
            }
        )
    return {
        "count": len(failures),
        "by_reason": reasons,
        "requests": failures,
    }


def _annotate_request_classes(
    run_record: dict[str, object],
    request_classes: list[str],
) -> None:
    results = run_record["requests"]
    metrics = run_record["engine_metrics"]["requests"]
    metrics_by_id = {int(record["request_id"]): record for record in metrics}
    for result, case_id in zip(results, request_classes, strict=True):
        result["request_class"] = case_id
        metrics_by_id[int(result["request_id"])]["request_class"] = case_id


def _load_class_slos(
    path: str | None,
    *,
    request_classes: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, object] | None]:
    if path is None:
        return {}, None
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = source.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("class SLO file requires a classes object")
    missing = sorted(set(request_classes) - set(classes))
    if missing:
        raise ValueError(f"class SLO file is missing request classes: {missing}")
    parsed: dict[str, dict[str, float]] = {}
    for case_id in sorted(set(request_classes)):
        row = classes[case_id]
        if not isinstance(row, dict):
            raise ValueError(f"class SLO entry must be an object: {case_id}")
        ttft_ms = float(row["ttft_ms"])
        tpot_ms = float(row["tpot_ms"])
        if ttft_ms <= 0 or tpot_ms <= 0:
            raise ValueError(f"class SLO thresholds must be positive: {case_id}")
        parsed[case_id] = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms}
    return parsed, {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "record_type": source.get("record_type"),
        "source": source.get("source"),
    }


def _summarize_by_class(
    run_record: dict[str, object],
    request_classes: list[str],
    class_slos: dict[str, dict[str, float]],
) -> dict[str, object]:
    duration_s = float(run_record["duration_s"])
    metrics = run_record["engine_metrics"]["requests"]
    metrics_by_id = {int(record["request_id"]): record for record in metrics}
    grouped: dict[str, list[dict[str, object]]] = {}
    for result, case_id in zip(
        run_record["requests"],
        request_classes,
        strict=True,
    ):
        grouped.setdefault(case_id, []).append(
            metrics_by_id[int(result["request_id"])]
        )

    total_good_requests = 0
    total_good_output_tokens = 0
    rows: dict[str, object] = {}
    for case_id, records in sorted(grouped.items()):
        completed = [
            record
            for record in records
            if record.get("finish_reason") in {"eos", "length"}
        ]
        slo = class_slos.get(case_id)
        good: list[dict[str, object]] = []
        if slo is not None:
            good = [
                record
                for record in completed
                if float(record["ttft_ms"]) <= slo["ttft_ms"]
                and float(record["tpot_ms"]) <= slo["tpot_ms"]
            ]
        total_good_requests += len(good)
        total_good_output_tokens += sum(
            int(record["output_tokens"]) for record in good
        )
        rows[case_id] = {
            "slo": slo,
            "counts": {
                "submitted": len(records),
                "completed": len(completed),
                "rejected": sum(
                    record.get("finish_reason") == "rejected"
                    for record in records
                ),
                "cancelled": sum(
                    record.get("finish_reason") == "cancelled"
                    for record in records
                ),
                "good": None if slo is None else len(good),
            },
            "latency_ms": {
                "queue": summarize_distribution(
                    [float(record["queue_ms"]) for record in completed]
                ),
                "ttft": summarize_distribution(
                    [float(record["ttft_ms"]) for record in completed]
                ),
                "tpot": summarize_distribution(
                    [float(record["tpot_ms"]) for record in completed]
                ),
                "request": summarize_distribution(
                    [float(record["latency_ms"]) for record in completed]
                ),
            },
            "throughput": {
                "requests_per_s": len(completed) / duration_s,
                "output_tokens_per_s": (
                    sum(int(record["output_tokens"]) for record in completed)
                    / duration_s
                ),
            },
            "goodput": (
                None
                if slo is None
                else {
                    "requests_per_s": len(good) / duration_s,
                    "output_tokens_per_s": (
                        sum(int(record["output_tokens"]) for record in good)
                        / duration_s
                    ),
                    "fraction_of_completed": (
                        0.0 if not completed else len(good) / len(completed)
                    ),
                }
            ),
        }
    return {
        "headline_unit": "output_tokens_per_second_meeting_both_slos",
        "slo_available": bool(class_slos),
        "by_class": rows,
        "aggregate_goodput": (
            None
            if not class_slos
            else {
                "requests_per_s": total_good_requests / duration_s,
                "output_tokens_per_s": total_good_output_tokens / duration_s,
            }
        ),
    }


def _h3_conformance(
    h3_contract: dict[str, object] | None,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if h3_contract is None:
        return None
    checks = {
        "arrival_process": args.arrival_process == h3_contract["arrival_process"],
        "request_rate": args.request_rate
        in h3_contract["request_rates_per_second"],
        "requests": args.requests == h3_contract["completed_requests_per_run"],
        "seed": args.seed in h3_contract["arrival_seeds"],
        "warmup_requests": args.warmup_requests == 10,
        "max_tokens": args.max_tokens == h3_contract["max_tokens"],
        "max_model_len": args.max_model_len == h3_contract["max_model_len"],
    }
    return {
        "full_frozen_h3": all(checks.values()),
        "checks": checks,
        "deviations": [name for name, passed in checks.items() if not passed],
    }


def _build_engine(args: argparse.Namespace):
    mode = MODE_SPECS[args.mode]
    return LLM(
        args.model,
        enforce_eager=mode.enforce_eager,
        execution_backend=mode.execution,
        decode_compile_region=mode.decode_compile_region,
        decode_compile_mode="max-autotune-no-cudagraphs",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
        allow_unsafe_decode_compile=(mode.decode_compile_region != "none"),
        compression_mode=mode.compression,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        enable_chunked_prefill=True,
        max_chunk_size=args.max_chunk_size,
        enable_prefix_caching=args.enable_prefix_caching,
        max_queue_size=args.max_queue_size,
        max_consecutive_prefill_batches=(args.max_consecutive_prefill_batches),
        visual_pruning_keep_ratio=args.visual_pruning_keep_ratio,
        visual_pruning_min_keep_tokens=args.visual_pruning_min_keep_tokens,
        visual_pruning_video_min_keep_tokens=(
            args.visual_pruning_video_min_keep_tokens
        ),
        visual_pruning_strategy=args.visual_pruning_strategy,
        visual_pruning_attention_last_n_layers=(args.visual_pruning_attention_last_n_layers),
        logits_precision=mode.logits_precision or args.logits_precision,
        mlp_projection_mode=args.mlp_projection_mode,
        paged_decode_block_n=mode.paged_decode_block_n or 32,
        enable_fused_qk_rmsnorm=mode.fused_qk_rmsnorm,
        enable_fused_qk_mrope=mode.fused_qk_mrope,
        enable_fused_add_rmsnorm=mode.fused_add_rmsnorm,
        enable_packed_kv_projection=mode.packed_kv_projection,
        enable_vision_tensor_cudagraph=args.enable_vision_tensor_cudagraph,
        vision_attention_backend="sdpa",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case", default="single_image_448")
    parser.add_argument(
        "--mode",
        choices=P12_ONLINE_MODES,
        default="off_graph",
    )
    parser.add_argument(
        "--h3-profile",
        choices=H3_PROFILES,
        help="run the frozen P9 H3 weighted class schedule instead of one case",
    )
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument(
        "--arrival-process",
        choices=("constant", "poisson", "burst"),
        default="constant",
    )
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-chunk-size", type=int, default=512)
    parser.add_argument("--max-queue-size", type=int)
    parser.add_argument("--max-consecutive-prefill-batches", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="enable text-only full-block online prefix reuse; VL hashes stay disabled",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="explicitly keep online prefix reuse disabled (default)",
    )
    parser.set_defaults(enable_prefix_caching=False)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=50.0)
    parser.add_argument("--visual-pruning-keep-ratio", type=float, default=0.5)
    parser.add_argument("--visual-pruning-min-keep-tokens", type=int, default=32)
    parser.add_argument("--visual-pruning-video-min-keep-tokens", type=int)
    parser.add_argument(
        "--visual-pruning-strategy",
        choices=("uniform", "attention"),
        default="attention",
    )
    parser.add_argument(
        "--visual-pruning-attention-last-n-layers",
        type=int,
        default=1,
    )
    parser.add_argument("--logits-precision", choices=("model", "fp32"), default="model")
    parser.add_argument(
        "--mlp-projection-mode",
        choices=("legacy", "packed"),
        default="packed",
    )
    parser.add_argument("--enable-vision-tensor-cudagraph", action="store_true")
    parser.add_argument(
        "--class-slo-file",
        help="JSON file with per-class TTFT/TPOT SLOs derived from vLLM low load",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the online benchmark")
    if args.requests <= 0 or args.warmup_requests < 0:
        raise SystemExit("--requests must be positive and warmup must be >= 0")
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be >= 2 for TPOT/goodput")
    if args.request_rate <= 0 and args.arrival_process != "burst":
        raise SystemExit("--request-rate must be positive")

    manifest = load_workload_manifest(args.manifest)
    if args.h3_profile is None:
        case = find_workload_case(manifest, args.case)
        payloads = materialize_requests(case, repo_root=REPO_ROOT)
        request_classes = [args.case] * args.requests
        h3_contract = None
        workload_case = args.case
    else:
        payloads, request_classes, h3_contract = _h3_payload_schedule(
            manifest,
            profile=args.h3_profile,
            count=args.requests,
        )
        workload_case = f"h3_{args.h3_profile}"
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    mode = MODE_SPECS[args.mode]
    llm = _build_engine(args)
    try:
        if args.warmup_requests:
            warmup_payloads = payloads[: args.warmup_requests]
            warmup = _online_requests(
                warmup_payloads,
                count=args.warmup_requests,
                process="burst",
                request_rate=args.request_rate,
                seed=args.seed,
                sampling=sampling,
                key_prefix="warmup",
            )
            OnlineServingSession(llm).run(warmup)
            llm.reset_metrics()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        requests = _online_requests(
            payloads,
            count=args.requests,
            process=args.arrival_process,
            request_rate=args.request_rate,
            seed=args.seed,
            sampling=sampling,
            key_prefix="formal",
        )
        run = OnlineServingSession(llm).run(requests)
        torch.cuda.synchronize()
        run_record = run.to_record()
        _annotate_request_classes(run_record, request_classes)
        class_slos, class_slo_source = _load_class_slos(
            args.class_slo_file,
            request_classes=request_classes,
        )
        prompt_audit = _prompt_audit(llm, run, request_classes)
        compaction_summary = _visual_compaction_summary(
            llm,
            run,
            request_classes,
        )
        summary = summarize_online_run(
            run_record,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        git = collect_git_metadata(REPO_ROOT, strict=True)
        config_path = Path(args.model) / "config.json"
        kv_storage = kv_cache_storage_bytes(
            llm.model_runner.kv_cache,
            llm.model_runner.kv_scale_cache,
        )
        record = {
            "schema_version": ONLINE_BENCHMARK_SCHEMA_VERSION,
            "record_type": "prism_online_run",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "framework": {
                "name": "prism-infer",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "hardware": collect_gpu_metadata().environment_dict(),
            "model": {
                "path": str(Path(args.model).resolve()),
                "config_sha256": (
                    hashlib.sha256(config_path.read_bytes()).hexdigest()
                    if config_path.is_file()
                    else None
                ),
            },
            "workload": {
                "manifest": manifest["name"],
                "case": workload_case,
                "source_request_types": [payload["type"] for payload in payloads],
                "request_classes": request_classes,
                "requests": args.requests,
                "max_tokens": args.max_tokens,
                "h3_contract": h3_contract,
                "h3_conformance": _h3_conformance(h3_contract, args),
                **prompt_audit,
            },
            "arrival": {
                "process": args.arrival_process,
                "request_rate_per_s": args.request_rate,
                "seed": args.seed,
                "offsets_s": [request.arrival_offset_s for request in requests],
                "trace_sha256": _canonical_sha256(
                    {
                        "classes": request_classes,
                        "offsets_s": [
                            request.arrival_offset_s for request in requests
                        ],
                    }
                ),
            },
            "engine": {
                "mode": args.mode,
                "execution_backend": mode.execution,
                "decode_compile_region": mode.decode_compile_region,
                "compression_mode": mode.compression,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "max_chunk_size": args.max_chunk_size,
                "max_queue_size": args.max_queue_size,
                "max_consecutive_prefill_batches": (args.max_consecutive_prefill_batches),
                "num_kvcache_blocks": args.num_kvcache_blocks,
                "kvcache_block_size": args.kvcache_block_size,
                "enable_prefix_caching": args.enable_prefix_caching,
                "logits_precision": mode.logits_precision or args.logits_precision,
                "mlp_projection_mode": args.mlp_projection_mode,
                "visual_pruning_keep_ratio": args.visual_pruning_keep_ratio,
                "visual_pruning_min_keep_tokens": (
                    args.visual_pruning_min_keep_tokens
                ),
                "visual_pruning_video_min_keep_tokens": (
                    args.visual_pruning_video_min_keep_tokens
                ),
                "visual_pruning_strategy": args.visual_pruning_strategy,
                "vision_tensor_cudagraph": (
                    args.enable_vision_tensor_cudagraph
                ),
            },
            "memory": {
                "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mib": torch.cuda.memory_reserved() / (1024**2),
                "peak_allocated_mib": (torch.cuda.max_memory_allocated() / (1024**2)),
                "kv_cache": {
                    "payload_bytes": kv_storage.payload,
                    "scale_bytes": kv_storage.scales,
                    "total_bytes": kv_storage.total,
                },
            },
            "visual_compaction": compaction_summary,
            "execution_evidence": _execution_evidence(llm, run_record),
            "terminal_failures": _terminal_failure_summary(
                llm,
                run,
                request_classes,
            ),
            "class_slo_source": class_slo_source,
            "class_aware_summary": _summarize_by_class(
                run_record,
                request_classes,
                class_slos,
            ),
            "run": run_record,
            "summary": summary,
        }
        validate_online_benchmark_record(record)
    finally:
        llm.exit()

    rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote online record to {output}", file=sys.stderr)
    print(rendered)


if __name__ == "__main__":
    main()
