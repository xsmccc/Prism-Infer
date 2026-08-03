#!/usr/bin/env python3
"""Run SGLang on a shared repeated-visual-context working-set plan."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
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
except ImportError:  # pragma: no cover - optional process-memory telemetry
    pynvml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_external_sglang import (
    _prompts,
)
from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
from benchmarks.working_set_workload import (
    MaterializedWorkingSet,
    materialize_working_set,
    source_prompt_schedule_sha256,
    verify_working_set_model,
    verify_working_set_processor,
    working_set_processor_kwargs,
)
from prism_infer.analysis.quality_materialization import write_json_atomic
from prism_infer.analysis.working_set_plan import DEFAULT_MAX_NUM_SEQS

TERMINAL_FINISH_REASONS = {"eos", "length", "stop"}
WORKING_SET_KV_CACHE_DTYPE = "fp8_e4m3"


def _kv_bytes_per_token(model_path: str | Path, *, kv_cache_dtype: str) -> int:
    """Derive decoder KV payload bytes per token from the model config."""

    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    layers = int(text_config["num_hidden_layers"])
    kv_heads = int(text_config["num_key_value_heads"])
    attention_heads = int(text_config["num_attention_heads"])
    head_dim = int(
        text_config.get(
            "head_dim",
            int(text_config["hidden_size"]) // attention_heads,
        )
    )
    element_bytes = 1 if kv_cache_dtype.startswith("fp8") else 2
    return 2 * layers * kv_heads * head_dim * element_bytes


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
        self._nvml_initialized = False

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
        self._nvml_initialized = True
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._initial_bytes = self._sample()
            self._thread = Thread(
                target=self._run,
                name="sglang-device-process-memory-sampler",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("NVML process-memory sampler was not started")
        self._stop.set()
        self._thread.join()
        try:
            self._final_bytes = self._sample()
        finally:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
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

    def close(self) -> None:
        """Stop sampling and release NVML after partial or complete startup."""

        if self._thread is not None:
            self.stop()
        elif self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False


class _RunResources:
    """Own external-engine resources so every exit path releases them."""

    def __init__(self) -> None:
        self.working_set: MaterializedWorkingSet | None = None
        self.engine: Engine | None = None
        self.memory_sampler: _DeviceProcessMemorySampler | None = None

    def close(self, *, suppress_errors: bool) -> None:
        failures: list[BaseException] = []
        callbacks = (
            (
                "memory sampler",
                None if self.memory_sampler is None else self.memory_sampler.close,
            ),
            ("SGLang engine", None if self.engine is None else self.engine.shutdown),
            (
                "working-set images",
                None if self.working_set is None else self.working_set.close,
            ),
        )
        for name, callback in callbacks:
            if callback is None:
                continue
            try:
                callback()
            except BaseException as exc:  # Preserve an active benchmark failure.
                failures.append(RuntimeError(f"failed to close {name}"))
                failures[-1].__cause__ = exc
        self.memory_sampler = None
        self.engine = None
        self.working_set = None
        if failures and not suppress_errors:
            raise failures[0]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_new_output(path: Path) -> None:
    """Reject an existing artifact instead of silently overwriting evidence."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


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
    request_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not (len(requests) == len(request_classes) == len(offsets_s)):
        raise ValueError("request, class and arrival schedules must have equal lengths")
    if request_ids is not None and len(request_ids) != len(requests):
        raise ValueError("request id count must match the request schedule")
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
        request_id = f"{key_prefix}-{index:05d}" if request_ids is None else request_ids[index]
        prompt = (
            request["prompt"] if request["type"] == "text" else _prompts(processor, [request])[0]
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
                    raise RuntimeError(f"SGLang output token count regressed for {request_id}")
                token_arrivals.extend([observed] * (current_tokens - observed_tokens))
                observed_tokens = current_tokens
        finally:
            active -= 1
        finished = time.perf_counter()
        if output is None or not token_arrivals:
            raise RuntimeError(f"SGLang returned no output tokens for {request_id}")
        generated = list(output["output_ids"])
        prompt_token_ids = list(output.get("prompt_token_ids") or [])
        if len(prompt_token_ids) != int(output["meta_info"]["prompt_tokens"]):
            raise RuntimeError(f"SGLang prompt token audit disagrees for {request_id}")
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
            "queue_ms": (None if queue_s is None else max(0.0, float(queue_s) * 1000.0)),
            "ttft_ms": (first - arrival) * 1000.0,
            "tpot_ms": (
                0.0 if len(generated) <= 1 else (finished - first) * 1000.0 / (len(generated) - 1)
            ),
            "latency_ms": (finished - arrival) * 1000.0,
            "cached_tokens": output["meta_info"].get(
                "cached_tokens",
                output["meta_info"].get("prefix_matched_len"),
            ),
        }
        return record, prompt_token_ids, generated

    results = await asyncio.gather(*(run_one(index) for index in range(len(requests))))
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
    request_ids: list[str] | None = None,
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
            request_ids=request_ids,
        )
    )


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
            case_id: _canonical_sha256(rows) for case_id, rows in sorted(by_class.items())
        },
    }


def _resolve_working_set_kv_cache_dtype(requested: str) -> str:
    """Resolve the one supported working-set KV format, failing closed."""

    if requested == "auto":
        return WORKING_SET_KV_CACHE_DTYPE
    if requested != WORKING_SET_KV_CACHE_DTYPE:
        raise ValueError(
            "working-set SGLang requires --kv-cache-dtype="
            f"{WORKING_SET_KV_CACHE_DTYPE}, got {requested!r}"
        )
    return requested


def _verify_fixed_working_set_plan(audit: dict[str, object]) -> None:
    """Fail if the authoritative plan changed while the engine was running."""

    plan_path = Path(str(audit["plan_path"]))
    actual_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if actual_sha256 != audit["plan_sha256"]:
        raise RuntimeError("working-set plan changed during the benchmark run")
    audit["fixed_plan_verified"] = True


def _verify_sglang_working_set_runtime(
    engine: Engine,
    working_set: MaterializedWorkingSet,
    *,
    kv_bytes_per_token: int,
    processor_verification: dict[str, Any],
    mm_process_config: dict[str, Any],
) -> dict[str, Any]:
    server_args = engine.server_args
    budget = working_set.plan["kv_budget"]
    expected_max_num_seqs = int(working_set.plan["serving"]["max_num_seqs"])
    expected_chunked_prefill_size = int(working_set.plan["serving"]["max_chunk_size"])
    capacity_bytes = int(server_args.max_total_tokens) * kv_bytes_per_token
    unused_budget_bytes = int(budget["bytes"]) - capacity_bytes
    one_page_bytes = int(budget["page_size_tokens"]) * kv_bytes_per_token
    actual_cache_dtype = str(getattr(server_args, "kv_cache_dtype", "unknown"))
    checks = {
        "kv_budget_not_exceeded": 0 <= unused_budget_bytes < one_page_bytes,
        "kv_cache_dtype": actual_cache_dtype == WORKING_SET_KV_CACHE_DTYPE,
        "page_size_tokens": int(server_args.page_size) == int(budget["page_size_tokens"]),
        "max_num_seqs": int(server_args.max_running_requests) == expected_max_num_seqs,
        "chunked_prefill_size": int(server_args.chunked_prefill_size)
        == expected_chunked_prefill_size,
        "processor_config": server_args.mm_process_config == mm_process_config,
        "local_processor": processor_verification["image_size"]
        == {
            "shortest_edge": working_set_processor_kwargs(working_set.plan)["min_pixels"],
            "longest_edge": working_set_processor_kwargs(working_set.plan)["max_pixels"],
        },
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"SGLang working-set runtime contract mismatch: {failed}")
    return {
        "checks": checks,
        "kv_budget_bytes": int(budget["bytes"]),
        "kv_capacity_bytes": capacity_bytes,
        "unused_budget_bytes": unused_budget_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_cache_dtype_effective": actual_cache_dtype,
        "page_size_tokens": int(server_args.page_size),
        "max_total_tokens": int(server_args.max_total_tokens),
        "max_num_seqs": int(server_args.max_running_requests),
        "chunked_prefill_size": int(server_args.chunked_prefill_size),
        "mm_process_config": server_args.mm_process_config,
        "processor": processor_verification,
    }


def _run(resources: _RunResources) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--working-set-plan", required=True)
    parser.add_argument("--working-set-id", choices=("fit", "knee", "pressure"), required=True)
    parser.add_argument("--materialized-root", required=True)
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
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--chunked-prefill-size", type=int, default=-1)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mm-attention-backend", default="triton_attn")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-mm-global-cache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    _require_new_output(output_path)
    kv_cache_dtype_requested_cli = args.kv_cache_dtype
    if args.requests <= 0 or args.warmup_requests < 0 or args.max_tokens < 2:
        raise SystemExit("--requests must be positive, warmup non-negative and max-tokens >= 2")

    working_set = materialize_working_set(
        args.working_set_plan,
        workset_id=args.working_set_id,
        materialized_root=args.materialized_root,
    )
    resources.working_set = working_set
    model_verification = verify_working_set_model(working_set.plan, args.model)
    traffic = working_set.plan["traffic"]
    kv_budget = working_set.plan["kv_budget"]
    model_contract = working_set.plan["model"]
    processor_contract = working_set.plan["processor"]
    serving_contract = working_set.plan["serving"]
    args.requests = len(working_set.measured_payloads)
    args.warmup_requests = 0
    args.max_tokens = int(traffic["max_new_tokens"])
    args.request_rate = float(traffic["request_rate_per_s"])
    args.arrival_process = str(traffic["arrival_process"])
    args.seed = int(traffic["seed"])
    args.max_model_len = int(model_contract["max_model_len"])
    args.max_num_seqs = int(serving_contract["max_num_seqs"])
    args.chunked_prefill_size = int(serving_contract["max_chunk_size"])
    if args.max_num_seqs != DEFAULT_MAX_NUM_SEQS:
        raise ValueError("working-set SGLang requires max_num_seqs=8")
    args.page_size = int(kv_budget["page_size_tokens"])
    args.enable_prefix_caching = True
    args.enable_mm_global_cache = True
    args.kv_cache_dtype = _resolve_working_set_kv_cache_dtype(args.kv_cache_dtype)
    kv_bytes_per_token = _kv_bytes_per_token(
        args.model,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    raw_capacity_tokens = int(kv_budget["bytes"]) // kv_bytes_per_token
    args.max_total_tokens = (raw_capacity_tokens // args.page_size) * args.page_size
    if args.max_total_tokens <= 0:
        raise ValueError("fixed KV budget cannot hold one SGLang page")
    requests = working_set.measured_payloads
    request_classes = working_set.measured_sample_ids
    offsets_s = working_set.measured_offsets_s
    manifest = {"name": "muirbench_media_first_working_set"}
    plan_path = Path(args.working_set_plan)
    working_set_audit = {
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "workset_id": args.working_set_id,
        "group_ids": list(working_set.workset["group_ids"]),
        "available_questions": int(working_set.workset["available_questions"]),
        "observed_questions": int(working_set.workset["observed_questions"]),
        "measured_question_switches": int(working_set.workset["measured_question_switches"]),
        "dense_prefix_pages": int(working_set.workset["dense_prefix_pages"]),
        "kv_budget_bytes": int(kv_budget["bytes"]),
        "kv_budget_pages": int(kv_budget["pages"]),
        "page_size_tokens": int(kv_budget["page_size_tokens"]),
        "model_revision": str(model_contract["revision"]),
        "model_verification": model_verification,
        "image_min_pixels": int(processor_contract["image_min_pixels"]),
        "image_max_pixels": int(processor_contract["image_max_pixels"]),
        "max_num_seqs": int(serving_contract["max_num_seqs"]),
        "chunked_prefill_size": int(serving_contract["max_chunk_size"]),
        "source_prompt_sha256": source_prompt_schedule_sha256(working_set.measured_payloads),
        "population_source_prompt_sha256": source_prompt_schedule_sha256(
            working_set.population_payloads
        ),
        "prompt_token_hash_contract": "exact_token_ids_sha256_across_engines",
        "population_requests": len(working_set.population_payloads),
        "measured_requests": len(working_set.measured_payloads),
        "kv_cache_dtype_requested_cli": kv_cache_dtype_requested_cli,
        "kv_cache_dtype_enforced": args.kv_cache_dtype,
    }
    git = collect_git_metadata(REPO_ROOT, strict=True)
    processor_kwargs = working_set_processor_kwargs(working_set.plan)
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        **processor_kwargs,
    )
    processor_verification = verify_working_set_processor(processor, working_set.plan)
    mm_process_config = {"image": dict(processor_kwargs)}
    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=args.max_model_len,
        max_total_tokens=args.max_total_tokens,
        page_size=args.page_size,
        kv_cache_dtype=args.kv_cache_dtype,
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
    resources.engine = engine
    working_set_audit["runtime_verification"] = _verify_sglang_working_set_runtime(
        engine,
        working_set,
        kv_bytes_per_token=kv_bytes_per_token,
        processor_verification=processor_verification,
        mm_process_config=mm_process_config,
    )
    process_memory_sampler = _DeviceProcessMemorySampler()
    resources.memory_sampler = process_memory_sampler
    process_memory_sampler.start()
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": True,
    }
    population_runs = []
    for payload, group_id, sample_id, request_id in zip(
        working_set.population_payloads,
        working_set.population_group_ids,
        working_set.population_sample_ids,
        working_set.population_request_ids,
        strict=True,
    ):
        population_result = _run_arrivals(
            engine,
            processor,
            [payload],
            [sample_id],
            [0.0],
            sampling,
            key_prefix="population",
            request_ids=[request_id],
        )
        population_result["group_id"] = group_id
        population_result["source_prompt_sha256"] = source_prompt_schedule_sha256([payload])
        population_result["prompt_token_ids_sha256"] = _canonical_sha256(
            population_result["prompt_token_ids"]
        )
        population_runs.append(population_result)
    population_run = {
        "policy": "one_request_per_group_closed_loop",
        "runs": population_runs,
    }

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run = _run_arrivals(
        engine,
        processor,
        requests,
        request_classes,
        offsets_s,
        sampling,
        key_prefix="measured",
        request_ids=working_set.measured_request_ids,
    )
    torch.cuda.synchronize()
    process_device_memory = process_memory_sampler.stop()
    if any(record["finish_reason"] not in TERMINAL_FINISH_REASONS for record in run["requests"]):
        raise RuntimeError("SGLang online run contains a non-completed request")
    _verify_fixed_working_set_plan(working_set_audit)

    model_config_path = Path(args.model) / "config.json"
    kv_bytes_per_token = _kv_bytes_per_token(
        args.model,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    prompt_audit = _prompt_audit(run.pop("prompt_token_ids"), request_classes)
    output_token_ids = run.pop("output_token_ids")
    record = {
        "schema_version": 1,
        "record_type": "external_online_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "muirbench_media_first_working_set_v1",
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
            "case": f"muirbench_{args.working_set_id}",
            "requests": args.requests,
            "request_classes": request_classes,
            "max_tokens": args.max_tokens,
            "working_set_plan": working_set_audit,
            **prompt_audit,
        },
        "arrival": {
            "process": args.arrival_process,
            "request_rate_per_s": args.request_rate,
            "seed": args.seed,
            "offsets_s": offsets_s,
            "trace_sha256": _canonical_sha256({"classes": request_classes, "offsets_s": offsets_s}),
        },
        "backend": {
            "execution": "eager" if args.enforce_eager else "cuda_graph",
            "attention": args.attention_backend,
            "mm_attention": args.mm_attention_backend,
            "chunked_prefill": args.chunked_prefill_size > 0,
            "chunked_prefill_size": args.chunked_prefill_size,
            "prefix_caching": args.enable_prefix_caching,
            "mm_global_cache": args.enable_mm_global_cache,
            "mm_process_config": mm_process_config,
            "kv_cache_dtype_requested": kv_cache_dtype_requested_cli,
            "kv_cache_dtype_enforced": args.kv_cache_dtype,
            "kv_cache_dtype_effective": str(
                getattr(engine.server_args, "kv_cache_dtype", "unknown")
            ),
            "cuda_graph_max_bs_decode": args.max_num_seqs,
            "kv_cache_capacity_tokens": args.max_total_tokens,
            "page_size": args.page_size,
            "kv_cache_bytes_per_token_theoretical": kv_bytes_per_token,
            "kv_cache_memory_bytes_theoretical": (args.max_total_tokens * kv_bytes_per_token),
            "kv_cache_budget_bytes": int(working_set.plan["kv_budget"]["bytes"]),
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
        "population": population_run,
        "run": run,
    }
    write_json_atomic(output_path, record)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "git_commit": git.commit,
                "git_dirty": git.dirty,
                "prompt_token_ids_sha256": prompt_audit["prompt_token_ids_sha256"],
                "output_token_ids_sha256": record["correctness"]["output_token_ids_sha256"],
                "controller": run["controller"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    resources = _RunResources()
    failed = False
    try:
        _run(resources)
    except BaseException:
        failed = True
        raise
    finally:
        resources.close(suppress_errors=failed)


if __name__ == "__main__":
    main()
