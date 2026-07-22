#!/usr/bin/env python3
"""Measure Prism with the same first-to-last token-arrival TPOT as external engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers


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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeat < 1 or args.max_tokens < 2:
        raise SystemExit("warmup >= 0, repeat >= 1 and max-tokens >= 2 are required")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = find_workload_case(manifest, args.case)
    if len(case["requests"]) != 1:
        raise ValueError("token-arrival comparison currently requires exactly one request")

    llm = LLM(
        args.model,
        enforce_eager=False,
        compression_mode="off",
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
        vision_attention_backend="sdpa",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    def run_once() -> tuple[list[list[int]], int, float, float, float]:
        requests = materialize_requests(case, repo_root=REPO_ROOT)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        request_id = _add_request(llm, requests[0], sampling)
        prompt_count = llm.scheduler.waiting[-1].num_prompt_tokens
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
            prompt_count,
            (finished - started) * 1000.0,
            ttft_ms,
            tpot_ms,
        )

    try:
        for _ in range(args.warmup):
            run_once()
        token_runs: list[list[list[int]]] = []
        prompt_tokens: list[int] = []
        e2e_ms: list[float] = []
        ttft_ms: list[float] = []
        tpot_ms: list[float] = []
        throughput: list[float] = []
        allocated_mb: list[float] = []
        reserved_mb: list[float] = []
        peak_allocated_mb: list[float] = []
        for _ in range(args.repeat):
            tokens, prompt_count, elapsed, ttft, tpot = run_once()
            token_runs.append(tokens)
            prompt_tokens.append(prompt_count)
            e2e_ms.append(elapsed)
            ttft_ms.append(ttft)
            tpot_ms.append(tpot)
            throughput.append(sum(len(row) for row in tokens) / (elapsed / 1000.0))
            allocated_mb.append(torch.cuda.memory_allocated() / 1024 / 1024)
            reserved_mb.append(torch.cuda.memory_reserved() / 1024 / 1024)
            peak_allocated_mb.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if any(tokens != token_runs[0] for tokens in token_runs[1:]):
            raise RuntimeError("Prism greedy token ids changed across measured repeats")

        git = collect_git_metadata(REPO_ROOT)
        gpu = collect_gpu_metadata().environment_dict()
        graph = llm.model_runner.cudagraph_metadata(1)
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
                "kv_cache_capacity_tokens": args.num_kvcache_blocks * args.kvcache_block_size,
            },
            "backend": {
                "execution": "cuda_graph",
                "attention": "vllm_flash_attn_paged_bf16",
                "prefix_caching": False,
                "chunked_prefill": False,
                "logits_precision": "selective_fp32",
                "mlp_projection": "packed",
                "fused_qk_rmsnorm": True,
                "fused_qk_mrope": True,
                "fused_add_rmsnorm": True,
                "packed_kv_projection": True,
                "cuda_graph": graph,
            },
            "workload": {
                "manifest_name": manifest["name"],
                "manifest_sha256": _sha256(manifest),
                "case_id": case["id"],
                "request_types": [request["type"] for request in case["requests"]],
                "num_requests": 1,
                "prompt_tokens": prompt_tokens[-1],
                "max_tokens": args.max_tokens,
                "traffic": "offline_closed_loop",
            },
            "measurement": {
                "warmup": args.warmup,
                "repeat": args.repeat,
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
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2, sort_keys=True))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
