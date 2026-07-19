"""P7.3 single-node online arrival/continuous-batching benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
    summarize_online_run,
    validate_online_benchmark_record,
)
from prism_infer.engine.online import OnlineRequest, OnlineServingSession


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


def _build_engine(args: argparse.Namespace):
    mode = MODE_SPECS[args.mode]
    return LLM(
        args.model,
        enforce_eager=mode.enforce_eager,
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
        visual_pruning_strategy=args.visual_pruning_strategy,
        visual_pruning_attention_last_n_layers=(args.visual_pruning_attention_last_n_layers),
        logits_precision=args.logits_precision,
        mlp_projection_mode=args.mlp_projection_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case", default="single_image_448")
    parser.add_argument(
        "--mode",
        choices=("off_eager", "off_graph", "visual_compact_graph"),
        default="off_graph",
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
    case = find_workload_case(manifest, args.case)
    payloads = materialize_requests(case, repo_root=REPO_ROOT)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    llm = _build_engine(args)
    try:
        if args.warmup_requests:
            warmup = _online_requests(
                payloads,
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
        summary = summarize_online_run(
            run_record,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        git = collect_git_metadata(REPO_ROOT, strict=True)
        config_path = Path(args.model) / "config.json"
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
                "case": args.case,
                "source_request_types": [request["type"] for request in case["requests"]],
                "requests": args.requests,
                "max_tokens": args.max_tokens,
            },
            "arrival": {
                "process": args.arrival_process,
                "request_rate_per_s": args.request_rate,
                "seed": args.seed,
                "offsets_s": [request.arrival_offset_s for request in requests],
            },
            "engine": {
                "mode": args.mode,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "max_chunk_size": args.max_chunk_size,
                "max_queue_size": args.max_queue_size,
                "max_consecutive_prefill_batches": (args.max_consecutive_prefill_batches),
                "num_kvcache_blocks": args.num_kvcache_blocks,
                "kvcache_block_size": args.kvcache_block_size,
                "enable_prefix_caching": args.enable_prefix_caching,
                "logits_precision": args.logits_precision,
                "mlp_projection_mode": args.mlp_projection_mode,
            },
            "memory": {
                "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mib": torch.cuda.memory_reserved() / (1024**2),
                "peak_allocated_mib": (torch.cuda.max_memory_allocated() / (1024**2)),
            },
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
