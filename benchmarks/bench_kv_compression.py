"""P5 KV compression benchmark and quality smoke.

The benchmark uses deterministic local synthetic text/image/video inputs and
does not download data.  It is intended for P5 compression reports, not for
general serving throughput claims.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image

from prism_infer import LLM, SamplingParams
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes


def _kv_cache_record(llm: LLM) -> dict[str, Any]:
    """Build auditable payload/scale/total physical KV storage metadata."""

    payload_cache = llm.model_runner.kv_cache
    scale_cache = llm.model_runner.kv_scale_cache
    storage = kv_cache_storage_bytes(payload_cache, scale_cache)
    return {
        "dtype": str(payload_cache.dtype),
        "shape": list(payload_cache.shape),
        "scale_dtype": "none" if scale_cache is None else str(scale_cache.dtype),
        "scale_shape": [] if scale_cache is None else list(scale_cache.shape),
        "payload_bytes": storage.payload,
        "scale_bytes": storage.scales,
        "bytes": storage.total,
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
    return ordered[idx]


def _mb(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def _make_common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_kvcache_blocks": args.num_kvcache_blocks,
    }


def _make_images() -> tuple[Image.Image, Image.Image, list[Image.Image]]:
    image_a = Image.new("RGB", (448, 448), color=(100, 150, 200))
    image_b = Image.new("RGB", (448, 448), color=(200, 120, 80))
    frames = [
        Image.new("RGB", (448, 448), color=(80 + i * 30, 120, 180))
        for i in range(4)
    ]
    return image_a, image_b, frames


def _run_single_image_once(llm: LLM, params: SamplingParams) -> list[int]:
    image, _, _ = _make_images()
    return llm.generate_vl(
        "Describe this image.",
        image,
        params,
        use_tqdm=False,
    )["token_ids"]


def _run_timed_case(
    *,
    args: argparse.Namespace,
    mode: str,
    run_once: Callable[[LLM, SamplingParams], list[int]],
) -> dict[str, Any]:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    llm = LLM(
        args.model,
        compression_mode=mode,
        **_make_common_kwargs(args),
    )
    try:
        kv_cache_record = _kv_cache_record(llm)
        for _ in range(args.warmup):
            run_once(llm, params)

        latencies: list[float] = []
        token_s: list[float] = []
        outputs: list[list[int]] = []
        allocated: list[int] = []
        reserved: list[int] = []
        peak: list[int] = []
        for _ in range(args.repeat):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            tokens = run_once(llm, params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            latencies.append(elapsed)
            token_s.append(len(tokens) / elapsed)
            outputs.append(tokens)
            allocated.append(torch.cuda.memory_allocated())
            reserved.append(torch.cuda.memory_reserved())
            peak.append(torch.cuda.max_memory_allocated())

        return {
            "mode": mode,
            "gpu": torch.cuda.get_device_name(0),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "input": {
                "case": args.case,
                "max_tokens": args.max_tokens,
                "num_kvcache_blocks": args.num_kvcache_blocks,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "kv_cache": {
                **kv_cache_record,
            },
            "tokens": outputs[0],
            "outputs_identical": all(tokens == outputs[0] for tokens in outputs),
            "latency_s": {
                "median": statistics.median(latencies),
                "p90": _percentile(latencies, 0.9),
                "min": min(latencies),
                "max": max(latencies),
            },
            "output_token_s": {
                "median": statistics.median(token_s),
                "p90": _percentile(token_s, 0.9),
                "min": min(token_s),
                "max": max(token_s),
            },
            "memory_mb": {
                "allocated_median": statistics.median([_mb(x) for x in allocated]),
                "reserved_median": statistics.median([_mb(x) for x in reserved]),
                "peak_allocated_median": statistics.median([_mb(x) for x in peak]),
            },
        }
    finally:
        llm.exit()
        torch.cuda.empty_cache()


def _run_quality_matrix(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    image_a, image_b, frames = _make_images()
    llm = LLM(
        args.model,
        compression_mode=mode,
        **_make_common_kwargs(args),
    )
    try:
        return {
            "mode": mode,
            "kv_cache": _kv_cache_record(llm),
            "outputs": {
                "text": llm.generate(
                    [[151644, 872, 198, 77091, 198]],
                    params,
                    use_tqdm=False,
                )[0]["token_ids"],
                "single_image": llm.generate_vl(
                    "Describe this image.",
                    image_a,
                    params,
                    use_tqdm=False,
                )["token_ids"],
                "multi_image": llm.generate_images(
                    "Compare these images.",
                    [image_a, image_b],
                    params,
                    use_tqdm=False,
                )["token_ids"],
                "video": llm.generate_video(
                    "Describe this video.",
                    frames,
                    params,
                    use_tqdm=False,
                )["token_ids"],
            },
        }
    finally:
        llm.exit()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", choices=["single_image", "quality_matrix"], required=True)
    parser.add_argument("--modes", default="off,fp8_kv")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if args.case == "single_image":
        records = [
            _run_timed_case(args=args, mode=mode, run_once=_run_single_image_once)
            for mode in modes
        ]
    else:
        records = [_run_quality_matrix(args, mode) for mode in modes]
        if len(records) >= 2:
            baseline = records[0]["outputs"]
            for record in records[1:]:
                total = 0
                matched = 0
                per_case: dict[str, dict[str, Any]] = {}
                for key, baseline_tokens in baseline.items():
                    current_tokens = record["outputs"][key]
                    row_total = min(len(baseline_tokens), len(current_tokens))
                    row_matched = sum(
                        a == b for a, b in zip(baseline_tokens, current_tokens)
                    )
                    total += row_total
                    matched += row_matched
                    per_case[key] = {
                        "matched": row_matched,
                        "total": row_total,
                        "exact": baseline_tokens == current_tokens,
                    }
                record["comparison_to_first_mode"] = {
                    "first_mode": records[0]["mode"],
                    "matched": matched,
                    "total": total,
                    "exact": matched == total,
                    "per_case": per_case,
                    "kv_byte_ratio": (
                        record["kv_cache"]["bytes"] / records[0]["kv_cache"]["bytes"]
                    ),
                }

    for record in records:
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
