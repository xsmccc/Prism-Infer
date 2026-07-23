#!/usr/bin/env python3
"""Measure Prism with the same first-to-last token-arrival TPOT as external engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any

import torch
import transformers

try:
    import pynvml
except ImportError:
    pynvml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import (
    collect_git_metadata,
    collect_gpu_metadata,
    find_workload_case,
    materialize_requests,
)
from prism_infer import LLM, SamplingParams
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _stats(values: list[float]) -> dict[str, int | float]:
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"statistics require finite non-negative values: {values}")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _add_request(llm: LLM, request: dict[str, Any], sampling: SamplingParams) -> int:
    request_type = request["type"]
    if request_type == "text":
        return llm.add_request(request["prompt"], sampling)
    if request_type == "image":
        return llm.add_vl_request(request["prompt"], request["image"], sampling)
    if request_type == "images":
        return llm.add_images_request(request["prompt"], request["images"], sampling)
    if request_type == "video":
        return llm.add_video_request(request["prompt"], request["video"], sampling)
    raise ValueError(f"unsupported request type: {request_type!r}")


class _ProcessDeviceMemorySampler:
    """Sample driver-accounted GPU memory for this fresh benchmark process."""

    def __init__(self, *, device_index: int = 0, interval_ms: float = 10.0) -> None:
        if pynvml is None:
            raise RuntimeError(
                "--sample-process-memory requires the nvidia-ml-py/pynvml package"
            )
        self.device_index = device_index
        self.interval_ms = interval_ms
        self._stop = Event()
        self._thread: Thread | None = None
        self._handle = None
        self._samples = 0
        self._initial_bytes = 0
        self._peak_bytes = 0
        self._final_bytes = 0
        self._failure: BaseException | None = None
        self._record: dict[str, Any] | None = None

    def _read_process_bytes(self) -> int:
        used_bytes = 0
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
            if process.pid != os.getpid():
                continue
            value = process.usedGpuMemory
            if isinstance(value, int) and 0 <= value < (1 << 63):
                used_bytes += value
        return used_bytes

    def _sample(self) -> int:
        value = self._read_process_bytes()
        self._samples += 1
        self._peak_bytes = max(self._peak_bytes, value)
        return value

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            try:
                self._sample()
            except BaseException as exc:  # surfaced synchronously by stop()
                self._failure = exc
                self._stop.set()

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._initial_bytes = self._sample()
        self._thread = Thread(
            target=self._run,
            name="prism-nvml-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            if self._record is None:
                raise RuntimeError("process device memory sampler was not started")
            return self._record
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
        self._record = {
            "measurement": "NVML compute-process usedGpuMemory",
            "scope": "post-LLM-init through warmup and measured generation",
            "device_index": self.device_index,
            "sampling_interval_ms": self.interval_ms,
            "samples": self._samples,
            "after_llm_init_mib": self._initial_bytes / mib,
            "peak_serving_mib": self._peak_bytes / mib,
            "after_benchmark_mib": self._final_bytes / mib,
        }
        return self._record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--num-kvcache-blocks", type=int, default=113)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--execution-backend",
        choices=("cuda_graph", "compile_graph"),
        default="cuda_graph",
    )
    parser.add_argument(
        "--compression-mode",
        choices=("off", "scaled_fp8_kv"),
        default="off",
        help="physical KV storage profile; scaled FP8 includes FP32 token-head scales",
    )
    parser.add_argument(
        "--compile-region",
        choices=("stateless",),
        default="stateless",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "wrap measured repeats in cudaProfilerStart/Stop so Nsight "
            "captures generation without model load and warmup"
        ),
    )
    parser.add_argument(
        "--enable-decode-block4-gate-up",
        action="store_true",
        help=(
            "retain an SM120 block-4 FP8-weight copy and use the fused "
            "batch-one decode gate-up/SwiGLU kernel"
        ),
    )
    parser.add_argument(
        "--sample-process-memory",
        action="store_true",
        help=(
            "sample driver-accounted memory for this process with NVML; "
            "use dedicated memory artifacts rather than latency headlines"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeat < 1 or args.max_tokens < 2:
        raise SystemExit("warmup >= 0, repeat >= 1 and max-tokens >= 2 are required")
    if args.enable_decode_block4_gate_up and args.compression_mode != "off":
        raise SystemExit(
            "the memory profile forbids the duplicate block4 decode weight "
            "when compression-mode is not off"
        )

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = find_workload_case(manifest, args.case)
    if len(case["requests"]) != 1:
        raise ValueError("token-arrival comparison currently requires exactly one request")

    llm = LLM(
        args.model,
        enforce_eager=False,
        execution_backend=args.execution_backend,
        decode_compile_region=(
            args.compile_region if args.execution_backend == "compile_graph" else "none"
        ),
        decode_compile_mode="max-autotune-no-cudagraphs",
        compression_mode=args.compression_mode,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        logits_precision="selective_fp32",
        mlp_projection_mode="packed",
        paged_decode_block_n=256,
        enable_fused_qk_rmsnorm=True,
        enable_fused_qk_mrope=True,
        enable_fused_add_rmsnorm=True,
        enable_packed_kv_projection=True,
        enable_decode_block4_gate_up=args.enable_decode_block4_gate_up,
        vision_attention_backend="sdpa",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    kv_cache = llm.model_runner.kv_cache
    kv_scale_cache = llm.model_runner.kv_scale_cache
    kv_storage = kv_cache_storage_bytes(kv_cache, kv_scale_cache)
    attention_backend = (
        "vllm_flash_attn_paged_bf16"
        if args.compression_mode == "off"
        else "prism_triton_paged_scaled_fp8"
    )
    process_memory_sampler = (
        _ProcessDeviceMemorySampler() if args.sample_process_memory else None
    )
    if process_memory_sampler is not None:
        process_memory_sampler.start()

    def run_once() -> tuple[list[list[int]], list[int], float, float, float]:
        requests = materialize_requests(case, repo_root=REPO_ROOT)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        request_id = _add_request(llm, requests[0], sampling)
        prompt_token_ids = list(llm.scheduler.waiting[-1].prompt_token_ids)
        preprocessing_finished = perf_counter()
        arrivals: list[float] = []
        final_tokens: list[int] | None = None
        while not llm.is_finished():
            step = llm.step_result()
            arrived = perf_counter()
            emitted = [token for token in step.execution.token_ids if token is not None]
            if len(emitted) != 1:
                raise RuntimeError(f"expected one emitted token per step, got {emitted}")
            arrivals.append(arrived)
            for output in step.outputs:
                if output.request_id == request_id:
                    final_tokens = list(output.token_ids)
        finished = perf_counter()
        if final_tokens is None:
            raise RuntimeError("Prism request completed without a final output")
        if len(arrivals) != len(final_tokens) or len(arrivals) < 2:
            raise RuntimeError(
                f"token arrivals/output mismatch: arrivals={len(arrivals)} "
                f"tokens={len(final_tokens)}"
            )
        ttft_ms = (arrivals[0] - preprocessing_finished) * 1000.0
        tpot_ms = (arrivals[-1] - arrivals[0]) * 1000.0 / (len(arrivals) - 1)
        return (
            [final_tokens],
            prompt_token_ids,
            (finished - started) * 1000.0,
            ttft_ms,
            tpot_ms,
        )

    try:
        for _ in range(args.warmup):
            run_once()
        token_runs: list[list[list[int]]] = []
        prompt_token_runs: list[list[int]] = []
        e2e_ms: list[float] = []
        ttft_ms: list[float] = []
        tpot_ms: list[float] = []
        throughput: list[float] = []
        allocated_mb: list[float] = []
        reserved_mb: list[float] = []
        peak_allocated_mb: list[float] = []
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(args.repeat):
                tokens, prompt_token_ids, elapsed, ttft, tpot = run_once()
                token_runs.append(tokens)
                prompt_token_runs.append(prompt_token_ids)
                e2e_ms.append(elapsed)
                ttft_ms.append(ttft)
                tpot_ms.append(tpot)
                throughput.append(sum(len(row) for row in tokens) / (elapsed / 1000.0))
                allocated_mb.append(torch.cuda.memory_allocated() / 1024 / 1024)
                reserved_mb.append(torch.cuda.memory_reserved() / 1024 / 1024)
                peak_allocated_mb.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        finally:
            if args.cuda_profiler_range:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
        if any(tokens != token_runs[0] for tokens in token_runs[1:]):
            raise RuntimeError("Prism greedy token ids changed across measured repeats")
        if any(prompt_ids != prompt_token_runs[0] for prompt_ids in prompt_token_runs[1:]):
            raise RuntimeError("Prism prompt token ids changed across measured repeats")
        audited_prompt_ids = prompt_token_runs[0]
        process_device_memory = (
            None if process_memory_sampler is None else process_memory_sampler.stop()
        )

        git = collect_git_metadata(REPO_ROOT)
        gpu = collect_gpu_metadata().environment_dict()
        graph = llm.model_runner.cudagraph_metadata(1)
        compile_metadata = llm.model_runner.compile_metadata()
        block4_gate_up = llm.model_runner.block4_gate_up_metadata()
        record = {
            "schema_version": 1,
            "record_type": "external_system_benchmark",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "name": "p9_external_prism_token_arrival_v1",
                "process_scope": "fresh_process_per_artifact",
                "command": [sys.executable, *sys.argv],
            },
            "environment": {
                "framework": "prism-infer",
                "framework_source_commit": git.commit,
                "framework_source_dirty": git.dirty,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                **gpu,
            },
            "model": {
                "path": str(Path(args.model).resolve()),
                "dtype": str(llm.model_runner.model_dtype),
                "tensor_parallel_size": 1,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "kv_cache_capacity_tokens": (
                    llm.model_runner.num_kvcache_blocks * args.kvcache_block_size
                ),
            },
            "backend": {
                "execution": args.execution_backend,
                "attention": attention_backend,
                "compression": args.compression_mode,
                "prefix_caching": False,
                "chunked_prefill": False,
                "logits_precision": "selective_fp32",
                "mlp_projection": "packed",
                "fused_qk_rmsnorm": True,
                "fused_qk_mrope": True,
                "fused_add_rmsnorm": True,
                "packed_kv_projection": True,
                "decode_block4_gate_up": block4_gate_up,
                "cuda_graph": graph,
                "torch_compile": compile_metadata,
            },
            "kv_cache": {
                "compression": args.compression_mode,
                "block_size": args.kvcache_block_size,
                "blocks": llm.model_runner.num_kvcache_blocks,
                "capacity_tokens": (
                    llm.model_runner.num_kvcache_blocks * args.kvcache_block_size
                ),
                "payload_dtype": str(kv_cache.dtype),
                "scale_dtype": (
                    "none" if kv_scale_cache is None else str(kv_scale_cache.dtype)
                ),
                "payload_bytes": kv_storage.payload,
                "scale_bytes": kv_storage.scales,
                "total_bytes": kv_storage.total,
            },
            "workload": {
                "manifest_name": manifest["name"],
                "manifest_sha256": _sha256(manifest),
                "case_id": case["id"],
                "request_types": [request["type"] for request in case["requests"]],
                "num_requests": 1,
                "prompt_tokens": len(audited_prompt_ids),
                "prompt_token_ids_sha256": _sha256([audited_prompt_ids]),
                "max_tokens": args.max_tokens,
                "traffic": "offline_closed_loop",
            },
            "measurement": {
                "warmup": args.warmup,
                "repeat": args.repeat,
                "cuda_profiler_range": args.cuda_profiler_range,
                "nvml_process_sampling_enabled": args.sample_process_memory,
                "latency_headline_eligible": not args.sample_process_memory,
                "explicit_per_step_cuda_synchronize": False,
                "token_arrival_boundary": "step_result_return_after_sampled_token_d2h",
                "decode_tpot_scope": "first_to_last_token_divided_by_output_intervals",
            },
            "correctness": {
                "outputs_identical_across_repeats": True,
                "token_ids": token_runs[0],
                "output_sha256": _sha256(token_runs[0]),
            },
            "timing_ms": {
                "end_to_end": _stats(e2e_ms),
                "engine_ttft": _stats(ttft_ms),
                "decode_tpot": _stats(tpot_ms),
            },
            "throughput": {"e2e_output_tokens_per_s": _stats(throughput)},
            "memory_mb": {
                "measurement": "torch_cuda_allocator",
                "allocated": _stats(allocated_mb),
                "reserved": _stats(reserved_mb),
                "peak_allocated": _stats(peak_allocated_mb),
                "process_device": process_device_memory,
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2, sort_keys=True))
    finally:
        if process_memory_sampler is not None:
            process_memory_sampler.stop()
        llm.exit()


if __name__ == "__main__":
    main()
