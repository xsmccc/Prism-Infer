#!/usr/bin/env python3
"""Run vLLM on the frozen P9 H3 online mixed-multimodal protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    import pynvml
except ImportError:  # pragma: no cover - formal environment provides NVML
    pynvml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_external_vllm import (
    _build_vllm_prompts,
    _effective_backend,
    _materialize_case,
)
from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
from prism_infer.analysis.online_serving import summarize_distribution


H3_PROFILES = ("primary", "conditional_video")
TERMINAL_FINISH_REASONS = {"eos", "length", "stop"}


class _DeviceProcessMemorySampler:
    """Sample all compute-process memory on the benchmark's dedicated GPU."""

    def __init__(self, *, device_index: int = 0, interval_ms: float = 10.0) -> None:
        if pynvml is None:
            raise RuntimeError("NVML process-memory sampling requires nvidia-ml-py")
        self.device_index = device_index
        self.interval_ms = interval_ms
        self._stop = Event()
        self._thread: Thread | None = None
        self._handle = None
        self._samples = 0
        self._initial_bytes = 0
        self._peak_bytes = 0
        self._final_bytes = 0
        self._pids: set[int] = set()
        self._failure: BaseException | None = None

    def _sample(self) -> int:
        used_bytes = 0
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
            value = process.usedGpuMemory
            if isinstance(value, int) and 0 <= value < (1 << 63):
                used_bytes += value
                self._pids.add(int(process.pid))
        self._samples += 1
        self._peak_bytes = max(self._peak_bytes, used_bytes)
        return used_bytes

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            try:
                self._sample()
            except BaseException as exc:
                self._failure = exc
                self._stop.set()

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._initial_bytes = self._sample()
        self._thread = Thread(
            target=self._run,
            name="vllm-device-process-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("NVML process-memory sampler was not started")
        self._stop.set()
        self._thread.join()
        try:
            self._final_bytes = self._sample()
        finally:
            pynvml.nvmlShutdown()
            self._thread = None
        if self._failure is not None:
            raise RuntimeError("NVML process-memory sampling failed") from self._failure
        mib = 1024 * 1024
        return {
            "measurement": "NVML total compute-process usedGpuMemory",
            "scope": "post-LLM-init through warmup and measured generation",
            "dedicated_gpu_required": True,
            "device_index": self.device_index,
            "sampling_interval_ms": self.interval_ms,
            "samples": self._samples,
            "observed_pids": sorted(self._pids),
            "after_llm_init_mib": self._initial_bytes / mib,
            "peak_serving_mib": self._peak_bytes / mib,
            "after_benchmark_mib": self._final_bytes / mib,
        }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _smooth_weighted_period(
    classes: list[dict[str, object]],
) -> tuple[str, ...]:
    case_ids = [str(item["case_id"]) for item in classes]
    counts = [int(round(float(item["weight"]) * 10)) for item in classes]
    if sum(counts) != 10 or any(count <= 0 for count in counts):
        raise ValueError(f"H3 class weights must form a positive period 10: {counts}")
    current = [0] * len(counts)
    period: list[str] = []
    for _ in range(10):
        for index, count in enumerate(counts):
            current[index] += count
        selected = max(
            range(len(counts)),
            key=lambda index: (current[index], -index),
        )
        period.append(case_ids[selected])
        current[selected] -= 10
    return tuple(period)


def _h3_schedule(
    manifest: dict[str, Any],
    *,
    profile: str,
    count: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    h3 = manifest["p9_protocol"]["headline"]["H3"]
    field = "primary_classes" if profile == "primary" else "conditional_video_classes"
    classes = h3[field]
    period = _smooth_weighted_period(classes)
    case_by_id = {case["id"]: case for case in manifest["cases"]}
    request_by_class = {}
    for item in classes:
        case_id = item["case_id"]
        requests = _materialize_case(case_by_id[case_id])
        if len(requests) != 1:
            raise ValueError(f"H3 class must materialize one request: {case_id}")
        request_by_class[case_id] = requests[0]
    request_classes = [period[index % len(period)] for index in range(count)]
    requests = [request_by_class[case_id] for case_id in request_classes]
    return requests, request_classes, {
        "profile": profile,
        "class_field": field,
        "class_schedule": h3["class_schedule"],
        "materialized_schedule_algorithm": (
            "smooth_weighted_round_robin_integer_counts"
        ),
        "period": list(period),
        "classes": classes,
        "arrival_process": h3["arrival_process"],
        "request_rates_per_second": h3["request_rates_per_second"],
        "completed_requests_per_run": h3["completed_requests_per_run"],
        "arrival_seeds": h3["arrival_seeds"],
        "max_tokens": h3["max_tokens"],
        "max_model_len": h3["max_model_len"],
        "ttft_slo_formula": h3["ttft_slo_formula"],
        "tpot_slo_formula": h3["tpot_slo_formula"],
        "goodput_unit": h3["goodput_unit"],
    }


def _protocol_conformance(
    contract: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, object]:
    checks = {
        "arrival_process": args.arrival_process == contract["arrival_process"],
        "request_rate": args.request_rate in contract["request_rates_per_second"],
        "requests": args.requests == contract["completed_requests_per_run"],
        "seed": args.seed in contract["arrival_seeds"],
        "warmup_requests": args.warmup_requests == 10,
        "max_tokens": args.max_tokens == contract["max_tokens"],
        "max_model_len": args.max_model_len == contract["max_model_len"],
    }
    return {
        "full_frozen_h3": all(checks.values()),
        "checks": checks,
        "deviations": [name for name, passed in checks.items() if not passed],
    }


def _finish_reason(output: Any) -> str:
    reason = output.outputs[0].finish_reason
    value = str(getattr(reason, "value", reason))
    return "eos" if value == "stop" else value


def _run_arrivals(
    engine: Any,
    processor: Any,
    requests: list[dict[str, Any]],
    request_classes: list[str],
    offsets_s: list[float],
    sampling: SamplingParams,
    *,
    key_prefix: str,
) -> dict[str, Any]:
    if not (len(requests) == len(request_classes) == len(offsets_s)):
        raise ValueError("request, class and arrival schedules must have equal lengths")
    pending = deque(range(len(requests)))
    started = time.perf_counter()
    observed_submit: dict[str, float] = {}
    first_token: dict[str, float] = {}
    finished: dict[str, float] = {}
    latest_outputs: dict[str, Any] = {}
    peak_inflight = 0

    while pending or engine.has_unfinished_requests():
        now = time.perf_counter()
        while pending and offsets_s[pending[0]] <= now - started:
            index = pending.popleft()
            request_id = f"{key_prefix}-{index:05d}"
            prompt = _build_vllm_prompts(processor, [requests[index]])[0]
            intended_wall_arrival = time.time() - (time.perf_counter() - started)
            intended_wall_arrival += offsets_s[index]
            engine.add_request(
                request_id,
                prompt,
                sampling,
                arrival_time=intended_wall_arrival,
            )
            observed_submit[request_id] = time.perf_counter()
        peak_inflight = max(
            peak_inflight,
            len(observed_submit) - len(finished),
        )
        if engine.has_unfinished_requests():
            outputs = engine.step()
            observed = time.perf_counter()
            for output in outputs:
                request_id = output.request_id
                latest_outputs[request_id] = output
                if output.outputs and output.outputs[0].token_ids:
                    first_token.setdefault(request_id, observed)
                if output.finished:
                    finished[request_id] = observed
            continue
        if pending:
            wait_s = max(0.0, started + offsets_s[pending[0]] - time.perf_counter())
            if wait_s:
                time.sleep(wait_s)

    completed = time.perf_counter()
    records = []
    prompt_ids = []
    output_ids = []
    for index, (case_id, offset_s) in enumerate(
        zip(request_classes, offsets_s, strict=True)
    ):
        request_id = f"{key_prefix}-{index:05d}"
        output = latest_outputs[request_id]
        generated = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids or [])
        prompt_ids.append(prompt_token_ids)
        output_ids.append(generated)
        arrival = started + offset_s
        first = first_token[request_id]
        end = finished[request_id]
        metrics = output.metrics
        queue_ms = None
        if metrics is not None and metrics.scheduled_ts and metrics.queued_ts:
            queue_ms = max(0.0, (metrics.scheduled_ts - metrics.queued_ts) * 1000.0)
        records.append(
            {
                "request_id": request_id,
                "request_class": case_id,
                "finish_reason": _finish_reason(output),
                "prompt_tokens": len(prompt_token_ids),
                "output_tokens": len(generated),
                "token_ids": generated,
                "arrival_offset_s": offset_s,
                "controller_submit_delay_ms": (
                    observed_submit[request_id] - arrival
                )
                * 1000.0,
                "queue_ms": queue_ms,
                "ttft_ms": (first - arrival) * 1000.0,
                "tpot_ms": (
                    0.0
                    if len(generated) <= 1
                    else (end - first) * 1000.0 / (len(generated) - 1)
                ),
                "latency_ms": (end - arrival) * 1000.0,
            }
        )
    return {
        "started_perf_counter_s": started,
        "finished_perf_counter_s": completed,
        "duration_s": completed - started,
        "requests": records,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
        "controller": {
            "peak_inflight": peak_inflight,
            "submitted": len(observed_submit),
            "completed": len(finished),
        },
    }


def _summarize_by_class(run: dict[str, Any]) -> dict[str, object]:
    duration_s = float(run["duration_s"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in run["requests"]:
        grouped.setdefault(record["request_class"], []).append(record)
    rows = {}
    for case_id, records in sorted(grouped.items()):
        completed = [
            record
            for record in records
            if record["finish_reason"] in TERMINAL_FINISH_REASONS
        ]
        rows[case_id] = {
            "counts": {
                "submitted": len(records),
                "completed": len(completed),
            },
            "latency_ms": {
                name: summarize_distribution(
                    [
                        float(record[name])
                        for record in completed
                        if record[name] is not None
                    ]
                )
                for name in ("queue_ms", "ttft_ms", "tpot_ms", "latency_ms")
            },
            "throughput": {
                "requests_per_s": len(completed) / duration_s,
                "output_tokens_per_s": (
                    sum(int(record["output_tokens"]) for record in completed)
                    / duration_s
                ),
            },
        }
    output_tokens = sum(
        int(record["output_tokens"]) for record in run["requests"]
    )
    return {
        "by_class": rows,
        "throughput": {
            "requests_per_s": len(run["requests"]) / duration_s,
            "output_tokens_per_s": output_tokens / duration_s,
        },
    }


def _prompt_audit(
    prompt_ids: list[list[int]],
    request_classes: list[str],
) -> dict[str, object]:
    by_class: dict[str, list[list[int]]] = {}
    for token_ids, case_id in zip(prompt_ids, request_classes, strict=True):
        by_class.setdefault(case_id, []).append(token_ids)
    return {
        "prompt_tokens": sum(map(len, prompt_ids)),
        "prompt_tokens_per_request": list(map(len, prompt_ids)),
        "prompt_token_ids_sha256": _canonical_sha256(prompt_ids),
        "prompt_token_ids_sha256_by_class": {
            case_id: _canonical_sha256(rows)
            for case_id, rows in sorted(by_class.items())
        },
    }


def _write_slo_record(
    path: Path,
    *,
    record: dict[str, Any],
    output_path: Path,
) -> None:
    classes = {}
    for case_id, summary in record["class_summary"]["by_class"].items():
        classes[case_id] = {
            "ttft_ms": 5.0 * float(summary["latency_ms"]["ttft_ms"]["p50"]),
            "tpot_ms": 2.0 * float(summary["latency_ms"]["tpot_ms"]["p50"]),
        }
    source_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    slo_record = {
        "schema_version": 1,
        "record_type": "p12_class_slo",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "framework": "vllm",
            "framework_version": record["environment"]["framework_version"],
            "online_record": str(output_path.resolve()),
            "online_record_sha256": source_sha256,
            "request_rate_per_s": record["arrival"]["request_rate_per_s"],
            "profile": record["workload"]["h3_contract"]["profile"],
            "ttft_formula": "5 * vllm_best_stable_low_load_p50_by_class",
            "tpot_formula": "2 * vllm_best_stable_low_load_p50_by_class",
        },
        "classes": classes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(slo_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--h3-profile", choices=H3_PROFILES, required=True)
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument(
        "--arrival-process",
        choices=("constant", "poisson", "burst"),
        default="poisson",
    )
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=4294967296)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--slo-output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.warmup_requests < 0 or args.max_tokens < 2:
        raise SystemExit(
            "--requests must be positive, warmup non-negative and max-tokens >= 2"
        )

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests, request_classes, h3_contract = _h3_schedule(
        manifest,
        profile=args.h3_profile,
        count=args.requests,
    )
    offsets_s = _arrival_offsets(
        args.requests,
        process=args.arrival_process,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    conformance = _protocol_conformance(h3_contract, args)
    git = collect_git_metadata(REPO_ROOT, strict=True)
    if args.formal and (git.dirty or not conformance["full_frozen_h3"]):
        raise SystemExit(
            "--formal requires a clean harness and the complete frozen H3 contract"
        )
    if args.slo_output and (
        not args.formal
        or args.request_rate != min(h3_contract["request_rates_per_second"])
    ):
        raise SystemExit("--slo-output requires a formal lowest-rate H3 run")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    image_limit = max(
        (len(request.get("images", [])) for request in requests),
        default=0,
    )
    video_limit = max(
        (1 if "video" in request else 0 for request in requests),
        default=0,
    )
    limit_mm_per_prompt = {
        modality: limit
        for modality, limit in (("image", image_limit), ("video", video_limit))
        if limit > 0
    }
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        block_size=args.block_size,
        enforce_eager=False,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        attention_config={"backend": args.attention_backend},
        enable_chunked_prefill=True,
        async_scheduling=True,
        disable_log_stats=False,
        seed=0,
    )
    engine = llm.llm_engine
    process_memory_sampler = _DeviceProcessMemorySampler()
    process_memory_sampler.start()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    warmup_requests = [
        requests[index % len(requests)]
        for index in range(args.warmup_requests)
    ]
    warmup_classes = [
        request_classes[index % len(request_classes)]
        for index in range(args.warmup_requests)
    ]
    if args.warmup_requests:
        _run_arrivals(
            engine,
            processor,
            warmup_requests,
            warmup_classes,
            [0.0] * args.warmup_requests,
            sampling,
            key_prefix="warmup",
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run = _run_arrivals(
        engine,
        processor,
        requests,
        request_classes,
        offsets_s,
        sampling,
        key_prefix="formal",
    )
    torch.cuda.synchronize()
    process_device_memory = process_memory_sampler.stop()
    if any(
        record["finish_reason"] not in TERMINAL_FINISH_REASONS
        for record in run["requests"]
    ):
        raise RuntimeError("vLLM online run contains a non-completed request")

    effective_backend = _effective_backend(llm, enforce_eager=False)
    vllm_config = engine.vllm_config
    model_config_path = Path(args.model) / "config.json"
    prompt_audit = _prompt_audit(run.pop("prompt_token_ids"), request_classes)
    output_token_ids = run.pop("output_token_ids")
    class_summary = _summarize_by_class(run)
    record = {
        "schema_version": 1,
        "record_type": "external_online_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "p12_external_online_h3_v1",
            "formal": args.formal,
            "command": [sys.executable, *sys.argv],
            "harness_git_commit": git.commit,
            "harness_git_dirty": git.dirty,
        },
        "environment": {
            "framework": "vllm",
            "framework_version": importlib.metadata.version("vllm"),
            "framework_source": "installed_wheel",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **collect_gpu_metadata().environment_dict(),
        },
        "model": {
            "path": str(Path(args.model).resolve()),
            "config_sha256": (
                hashlib.sha256(model_config_path.read_bytes()).hexdigest()
                if model_config_path.is_file()
                else None
            ),
            "dtype": "torch.bfloat16",
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
        },
        "workload": {
            "manifest": manifest["name"],
            "manifest_sha256": _canonical_sha256(manifest),
            "case": f"h3_{args.h3_profile}",
            "requests": args.requests,
            "request_classes": request_classes,
            "max_tokens": args.max_tokens,
            "h3_contract": h3_contract,
            "h3_conformance": conformance,
            **prompt_audit,
        },
        "arrival": {
            "process": args.arrival_process,
            "request_rate_per_s": args.request_rate,
            "seed": args.seed,
            "offsets_s": offsets_s,
            "trace_sha256": _canonical_sha256(
                {"classes": request_classes, "offsets_s": offsets_s}
            ),
        },
        "backend": {
            **effective_backend,
            "attention": args.attention_backend,
            "block_size": args.block_size,
            "prefix_caching": False,
            "mm_processor_cache_gb": 0,
            "kv_cache_memory_bytes_requested": args.kv_cache_memory_bytes,
            "kv_cache_memory_bytes_effective": (
                vllm_config.cache_config.kv_cache_memory_bytes
            ),
        },
        "memory": {
            "process_device": process_device_memory,
            "torch_parent_process_allocator_is_headline_eligible": False,
            "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mib": torch.cuda.memory_reserved() / (1024**2),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        },
        "correctness": {
            "output_token_ids": output_token_ids,
            "output_token_ids_sha256": _canonical_sha256(output_token_ids),
        },
        "run": run,
        "class_summary": class_summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.slo_output:
        _write_slo_record(
            Path(args.slo_output),
            record=record,
            output_path=output_path,
        )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "git_commit": git.commit,
                "git_dirty": git.dirty,
                "h3_conformance": conformance,
                "class_summary": class_summary,
                "prompt_token_ids_sha256": prompt_audit[
                    "prompt_token_ids_sha256"
                ],
                "output_token_ids_sha256": record["correctness"][
                    "output_token_ids_sha256"
                ],
                "controller": run["controller"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
