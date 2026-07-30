#!/usr/bin/env python3
"""Run SGLang on the frozen P9 H3 online mixed-multimodal protocol."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import GenerateReqInput
from transformers import AutoProcessor

try:
    import pynvml
except ImportError:  # pragma: no cover - formal environment provides NVML
    pynvml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_external_sglang import (
    DEFAULT_VIDEO_FPS,
    _image,
    _prompts,
    _stage_lossless_video,
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
            name="sglang-device-process-memory-sampler",
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


def _materialize_case(
    case: dict[str, Any],
    *,
    video_staging_dir: Path,
) -> list[dict[str, Any]]:
    requests = []
    for index, request in enumerate(case["requests"]):
        request_type = request["type"]
        if request_type == "text":
            requests.append(
                {
                    "type": request_type,
                    "prompt": request["prompt"],
                    "images": [],
                }
            )
            continue
        if request_type in ("image", "image_file"):
            images = [_image(request["image"])]
        elif request_type == "images":
            images = [_image(spec) for spec in request["images"]]
        elif request_type == "video":
            frames = [_image(spec) for spec in request["frames"]]
            fps = float(request.get("fps", DEFAULT_VIDEO_FPS))
            staging_path = video_staging_dir / f"{case['id']}-{index}.mkv"
            staging = _stage_lossless_video(
                frames,
                path=staging_path,
                fps=fps,
            )
            requests.append(
                {
                    "type": request_type,
                    "prompt": request["prompt"],
                    "video": str(staging_path),
                    "fps": fps,
                    "video_staging": staging,
                }
            )
            continue
        else:
            raise ValueError(f"unsupported SGLang H3 request type: {request_type!r}")
        requests.append(
            {
                "type": request_type,
                "prompt": request["prompt"],
                "images": images,
            }
        )
    return requests


def _h3_schedule(
    manifest: dict[str, Any],
    *,
    profile: str,
    count: int,
    video_staging_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    h3 = manifest["p9_protocol"]["headline"]["H3"]
    field = "primary_classes" if profile == "primary" else "conditional_video_classes"
    classes = h3[field]
    period = _smooth_weighted_period(classes)
    case_by_id = {case["id"]: case for case in manifest["cases"]}
    request_by_class = {}
    for item in classes:
        case_id = item["case_id"]
        requests = _materialize_case(
            case_by_id[case_id],
            video_staging_dir=video_staging_dir,
        )
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


def _finish_reason(output: dict[str, Any]) -> str:
    reason = output["meta_info"].get("finish_reason")
    if isinstance(reason, dict):
        value = str(reason.get("type", ""))
    else:
        value = str(getattr(reason, "value", reason or ""))
    return "eos" if value == "stop" else value


async def _run_arrivals_async(
    engine: Engine,
    processor: Any,
    requests: list[dict[str, Any]],
    request_classes: list[str],
    offsets_s: list[float],
    sampling: dict[str, Any],
    *,
    key_prefix: str,
) -> dict[str, Any]:
    if not (len(requests) == len(request_classes) == len(offsets_s)):
        raise ValueError("request, class and arrival schedules must have equal lengths")
    started = time.perf_counter()
    active = 0
    peak_inflight = 0

    async def run_one(index: int) -> tuple[dict[str, Any], list[int], list[int]]:
        nonlocal active, peak_inflight
        request = requests[index]
        case_id = request_classes[index]
        offset_s = offsets_s[index]
        arrival = started + offset_s
        wait_s = max(0.0, arrival - time.perf_counter())
        if wait_s:
            await asyncio.sleep(wait_s)
        request_id = f"{key_prefix}-{index:05d}"
        prompt = (
            request["prompt"]
            if request["type"] == "text"
            else _prompts(processor, [request])[0]
        )
        image_data = request.get("images")
        if image_data:
            image_data = image_data[0] if len(image_data) == 1 else image_data
        else:
            image_data = None
        observed_submit = time.perf_counter()
        active += 1
        peak_inflight = max(peak_inflight, active)
        obj = GenerateReqInput(
            text=prompt,
            image_data=image_data,
            video_data=request.get("video"),
            sampling_params=sampling,
            stream=True,
            return_prompt_token_ids=True,
            rid=request_id,
        )
        generator = engine.tokenizer_manager.generate_request(obj, None)
        output: dict[str, Any] | None = None
        token_arrivals: list[float] = []
        observed_tokens = 0
        try:
            async for chunk in generator:
                observed = time.perf_counter()
                output = chunk
                current_tokens = len(chunk["output_ids"])
                if current_tokens < observed_tokens:
                    raise RuntimeError(
                        f"SGLang output token count regressed for {request_id}"
                    )
                token_arrivals.extend(
                    [observed] * (current_tokens - observed_tokens)
                )
                observed_tokens = current_tokens
        finally:
            active -= 1
        finished = time.perf_counter()
        if output is None or not token_arrivals:
            raise RuntimeError(f"SGLang returned no output tokens for {request_id}")
        generated = list(output["output_ids"])
        prompt_token_ids = list(output.get("prompt_token_ids") or [])
        if len(prompt_token_ids) != int(output["meta_info"]["prompt_tokens"]):
            raise RuntimeError(
                f"SGLang prompt token audit disagrees for {request_id}"
            )
        first = token_arrivals[0]
        queue_s = output["meta_info"].get("queue_time")
        record = {
                "request_id": request_id,
                "request_class": case_id,
                "finish_reason": _finish_reason(output),
                "prompt_tokens": len(prompt_token_ids),
                "output_tokens": len(generated),
                "token_ids": generated,
                "arrival_offset_s": offset_s,
                "controller_submit_delay_ms": (observed_submit - arrival) * 1000.0,
                "queue_ms": (
                    None if queue_s is None else max(0.0, float(queue_s) * 1000.0)
                ),
                "ttft_ms": (first - arrival) * 1000.0,
                "tpot_ms": (
                    0.0
                    if len(generated) <= 1
                    else (finished - first) * 1000.0 / (len(generated) - 1)
                ),
                "latency_ms": (finished - arrival) * 1000.0,
            }
        return record, prompt_token_ids, generated

    results = await asyncio.gather(
        *(run_one(index) for index in range(len(requests)))
    )
    completed = time.perf_counter()
    records = [record for record, _, _ in results]
    prompt_ids = [prompt_ids for _, prompt_ids, _ in results]
    output_ids = [output_ids for _, _, output_ids in results]
    return {
        "started_perf_counter_s": started,
        "finished_perf_counter_s": completed,
        "duration_s": completed - started,
        "requests": records,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
        "controller": {
            "peak_inflight": peak_inflight,
            "submitted": len(records),
            "completed": len(records),
        },
    }


def _run_arrivals(
    engine: Engine,
    processor: Any,
    requests: list[dict[str, Any]],
    request_classes: list[str],
    offsets_s: list[float],
    sampling: dict[str, Any],
    *,
    key_prefix: str,
) -> dict[str, Any]:
    return engine.loop.run_until_complete(
        _run_arrivals_async(
            engine,
            processor,
            requests,
            request_classes,
            offsets_s,
            sampling,
            key_prefix=key_prefix,
        )
    )


def _summarize_by_class(
    run: dict[str, Any],
    *,
    class_slos: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
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
        row = {
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
        if class_slos is not None:
            thresholds = class_slos[case_id]
            meeting = [
                record
                for record in completed
                if float(record["ttft_ms"]) <= thresholds["ttft_ms"]
                and float(record["tpot_ms"]) <= thresholds["tpot_ms"]
            ]
            meeting_tokens = sum(
                int(record["output_tokens"]) for record in meeting
            )
            row["slo"] = {
                "thresholds_ms": thresholds,
                "requests_meeting_both": len(meeting),
                "output_tokens_meeting_both": meeting_tokens,
                "request_attainment": (
                    len(meeting) / len(completed) if completed else 0.0
                ),
                "goodput_output_tokens_per_s": meeting_tokens / duration_s,
            }
        rows[case_id] = row
    output_tokens = sum(
        int(record["output_tokens"]) for record in run["requests"]
    )
    summary = {
        "by_class": rows,
        "throughput": {
            "requests_per_s": len(run["requests"]) / duration_s,
            "output_tokens_per_s": output_tokens / duration_s,
        },
    }
    if class_slos is not None:
        meeting_requests = sum(
            int(row["slo"]["requests_meeting_both"]) for row in rows.values()
        )
        meeting_tokens = sum(
            int(row["slo"]["output_tokens_meeting_both"]) for row in rows.values()
        )
        summary["slo_goodput"] = {
            "requests_meeting_both": meeting_requests,
            "output_tokens_meeting_both": meeting_tokens,
            "request_attainment": (
                meeting_requests / len(run["requests"]) if run["requests"] else 0.0
            ),
            "output_tokens_per_s": meeting_tokens / duration_s,
        }
    return summary


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


def _load_class_slos(
    path: Path,
    *,
    expected_classes: set[str],
    expected_profile: str,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("record_type") != "p12_class_slo":
        raise ValueError(f"not a P12 class-SLO record: {path}")
    if payload.get("source", {}).get("framework") != "vllm":
        raise ValueError("P12 class SLO must be frozen from vLLM")
    if payload.get("source", {}).get("profile") != expected_profile:
        raise ValueError("P12 class SLO profile does not match the run")
    classes = payload.get("classes", {})
    if set(classes) != expected_classes:
        raise ValueError(
            f"P12 class SLO classes differ: {set(classes)} != {expected_classes}"
        )
    normalized = {}
    for case_id, thresholds in classes.items():
        ttft_ms = float(thresholds["ttft_ms"])
        tpot_ms = float(thresholds["tpot_ms"])
        if ttft_ms <= 0 or tpot_ms <= 0:
            raise ValueError(f"invalid P12 class SLO for {case_id}")
        normalized[case_id] = {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
        }
    return normalized, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": payload["source"],
    }


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
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=28928)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--chunked-prefill-size", type=int, default=-1)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mm-attention-backend", default="triton_attn")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-mm-global-cache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--class-slo-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.warmup_requests < 0 or args.max_tokens < 2:
        raise SystemExit(
            "--requests must be positive, warmup non-negative and max-tokens >= 2"
        )

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = Path(args.output)
    requests, request_classes, h3_contract = _h3_schedule(
        manifest,
        profile=args.h3_profile,
        count=args.requests,
        video_staging_dir=output_path.parent / "sglang_staging",
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
    class_slos = None
    class_slo_audit = None
    if args.class_slo_file:
        class_slos, class_slo_audit = _load_class_slos(
            Path(args.class_slo_file),
            expected_classes=set(request_classes),
            expected_profile=args.h3_profile,
        )
    elif args.formal:
        raise SystemExit("formal SGLang H3 runs require --class-slo-file")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    video_fps = {
        float(request["fps"]) for request in requests if "video" in request
    }
    if len(video_fps) > 1:
        raise ValueError(f"SGLang H3 requires one global video fps: {video_fps}")
    mm_process_config = (
        {"video": {"fps": next(iter(video_fps))}}
        if video_fps
        else {}
    )
    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=args.max_model_len,
        max_total_tokens=args.max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_num_seqs,
        chunked_prefill_size=args.chunked_prefill_size,
        disable_cuda_graph=args.enforce_eager,
        cuda_graph_max_bs_decode=args.max_num_seqs,
        disable_radix_cache=not args.enable_prefix_caching,
        attention_backend=args.attention_backend,
        mm_attention_backend=args.mm_attention_backend,
        enable_mm_global_cache=args.enable_mm_global_cache,
        enable_request_time_stats_logging=True,
        stream_interval=1,
        random_seed=0,
        mm_process_config=mm_process_config,
    )
    process_memory_sampler = _DeviceProcessMemorySampler()
    process_memory_sampler.start()
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": True,
    }
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
        raise RuntimeError("SGLang online run contains a non-completed request")

    model_config_path = Path(args.model) / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    text_config = model_config.get("text_config", model_config)
    head_dim = int(
        text_config.get(
            "head_dim",
            int(text_config["hidden_size"]) // int(text_config["num_attention_heads"]),
        )
    )
    kv_bytes_per_token = (
        int(text_config["num_hidden_layers"])
        * 2
        * int(text_config["num_key_value_heads"])
        * head_dim
        * 2
    )
    prompt_audit = _prompt_audit(run.pop("prompt_token_ids"), request_classes)
    output_token_ids = run.pop("output_token_ids")
    class_summary = _summarize_by_class(run, class_slos=class_slos)
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
            "framework": "sglang",
            "framework_version": importlib.metadata.version("sglang"),
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
            "max_num_seqs": args.max_num_seqs,
            "max_total_tokens": args.max_total_tokens,
            "mem_fraction_static": args.mem_fraction_static,
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
            "class_slo": class_slo_audit,
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
            "execution": "eager" if args.enforce_eager else "cuda_graph",
            "attention": args.attention_backend,
            "mm_attention": args.mm_attention_backend,
            "chunked_prefill": args.chunked_prefill_size > 0,
            "chunked_prefill_size": args.chunked_prefill_size,
            "prefix_caching": args.enable_prefix_caching,
            "mm_global_cache": args.enable_mm_global_cache,
            "cuda_graph_max_bs_decode": args.max_num_seqs,
            "kv_cache_capacity_tokens": args.max_total_tokens,
            "kv_cache_bytes_per_token_theoretical": kv_bytes_per_token,
            "kv_cache_memory_bytes_theoretical": (
                args.max_total_tokens * kv_bytes_per_token
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    engine.shutdown()


if __name__ == "__main__":
    main()
