"""P6.6 fixed-GPU-memory KV concurrency capacity benchmark.

每次进程只运行一个 mode，避免多个 near-capacity KV pool 在同一 CUDA context
连续创建造成资源碎片。该 benchmark 记录实际峰值 GPU-resident sequences，而不是
把 submitted requests 误写成同时并发。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_system import (
    DEFAULT_MANIFEST,
    MODE_SPECS,
    _add_requests,
    _build_llm,
)
from benchmarks.harness import (
    collect_git_metadata,
    expand_case_batch,
    find_workload_case,
    materialize_requests,
)
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.sampling_params import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case", default="multi_image_2x448")
    parser.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--visual-pruning-keep-ratio", type=float, default=0.5)
    parser.add_argument("--visual-pruning-min-keep-tokens", type=int, default=1)
    parser.set_defaults(enable_prefix_caching=False)
    parser.add_argument("--decode-compile-mode", default="default")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.requests <= 0 or args.max_tokens < 2:
        raise SystemExit("--requests must be positive and --max-tokens must be >= 2")

    manifest = load_workload_manifest(args.manifest)
    source_case = find_workload_case(manifest, args.case)
    case, source_requests, replication = expand_case_batch(
        source_case,
        args.requests,
    )
    mode = MODE_SPECS[args.mode]
    llm = _build_llm(args, mode, max_num_seqs=args.requests)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    try:
        requests = materialize_requests(case, repo_root=REPO_ROOT)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        seq_ids, token_counts = _add_requests(llm, requests, sampling)

        completed: set[int] = set()
        peak_gpu_running = 0
        peak_swapped = 0
        peak_total_active = 0
        peak_used_blocks = 0
        prefill_steps = 0
        decode_steps = 0
        while not llm.is_finished():
            finished, num_tokens = llm.step()
            completed.update(seq_id for seq_id, _ in finished)
            running = len(llm.scheduler.running)
            swapped = len(llm.scheduler.swapped)
            used_blocks = len(llm.scheduler.block_manager.used_block_ids)
            peak_gpu_running = max(peak_gpu_running, running)
            peak_swapped = max(peak_swapped, swapped)
            peak_total_active = max(peak_total_active, running + swapped)
            peak_used_blocks = max(peak_used_blocks, used_blocks)
            if num_tokens > 0:
                prefill_steps += 1
            else:
                decode_steps += 1

        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0
        if len(completed) != len(seq_ids):
            raise RuntimeError(f"capacity run completed {len(completed)}/{len(seq_ids)} requests")

        git = collect_git_metadata(REPO_ROOT, strict=True)
        cache = llm.model_runner.kv_cache
        config = llm.config
        record = {
            "record_type": "kv_capacity_benchmark",
            "schema_version": 1,
            "environment": {
                "git_commit": git.commit,
                "git_dirty": git.dirty,
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
            },
            "mode": {
                "name": mode.name,
                "compression": mode.compression,
                "attention": mode.attention,
                "visual_pruning_keep_ratio": args.visual_pruning_keep_ratio,
            },
            "workload": {
                "case_id": source_case["id"],
                "submitted_requests": len(seq_ids),
                "source_requests": source_requests,
                "request_replication_factor": replication,
                "prompt_tokens": token_counts[0],
                "image_tokens": token_counts[1],
                "video_tokens": token_counts[2],
                "max_tokens": args.max_tokens,
                "enable_prefix_caching": args.enable_prefix_caching,
            },
            "kv_cache": {
                "dtype": str(cache.dtype),
                "shape": list(cache.shape),
                "blocks": config.num_kvcache_blocks,
                "block_size": config.kvcache_block_size,
                "bytes": cache.numel() * cache.element_size(),
                "peak_used_blocks": peak_used_blocks,
            },
            "capacity": {
                "completed_requests": len(completed),
                "peak_gpu_running_sequences": peak_gpu_running,
                "peak_swapped_sequences": peak_swapped,
                "peak_total_active_sequences": peak_total_active,
                "prefill_steps": prefill_steps,
                "decode_steps": decode_steps,
            },
            "timing_ms": {
                "end_to_end": elapsed_ms,
                "cuda_synchronize_timing": True,
            },
            "memory_mb": {
                "allocated": torch.cuda.memory_allocated() / 2**20,
                "reserved": torch.cuda.memory_reserved() / 2**20,
                "peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
