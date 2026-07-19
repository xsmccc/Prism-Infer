#!/usr/bin/env python3
"""在固定 Prism image workload 上运行 SGLang offline eager baseline。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pynvml
import torch
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.entrypoints.engine import Engine


REPO_ROOT = Path(__file__).resolve().parents[1]


def _stats(values: list[float]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(value) and value >= 0 for value in ordered):
        raise ValueError(f"invalid statistics values: {values}")

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _image(spec: dict[str, Any]) -> Image.Image:
    if "color" in spec:
        return Image.new(
            "RGB",
            (int(spec["width"]), int(spec["height"])),
            tuple(int(channel) for channel in spec["color"]),
        )
    configured = Path(spec["path"])
    path = configured if configured.is_absolute() else REPO_ROOT / configured
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError(f"image SHA256 mismatch: expected {spec['sha256']}, got {actual}")
    with Image.open(path) as source:
        loaded = source.convert("RGB")
    if loaded.size != (int(spec["width"]), int(spec["height"])):
        raise ValueError(f"image size mismatch: {loaded.size}")
    return loaded


def _materialize(case: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for request in case["requests"]:
        request_type = request["type"]
        if request_type in ("image", "image_file"):
            images = [_image(request["image"])]
        elif request_type == "images":
            images = [_image(spec) for spec in request["images"]]
        else:
            raise ValueError(
                "SGLang P6 adapter currently supports image/image_file/images only; "
                f"got {request_type!r}"
            )
        requests.append({"type": request_type, "prompt": request["prompt"], "images": images})
    return requests


def _prompts(processor: Any, requests: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for request in requests:
        content = [{"type": "image", "image": image} for image in request["images"]]
        content.append({"type": "text", "text": request["prompt"]})
        result.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return result


def _nvml_compute_memory_mib() -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return sum(process.usedGpuMemory for process in processes) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    case = next((item for item in manifest["cases"] if item["id"] == args.case), None)
    if case is None:
        raise ValueError(f"case not found: {args.case}")
    requests = _materialize(case)
    if len(requests) != 1:
        raise ValueError("SGLang streaming P6 adapter currently requires one request")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    prompts = _prompts(processor, requests)
    image_data = [request["images"] for request in requests]

    pynvml.nvmlInit()
    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=1280,
        max_total_tokens=4096,
        mem_fraction_static=0.60,
        max_running_requests=len(requests),
        chunked_prefill_size=-1,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        attention_backend="triton",
        mm_attention_backend="triton_attn",
        enable_request_time_stats_logging=True,
        stream_interval=1,
        random_seed=0,
    )
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": True,
    }

    def run_once() -> tuple[list[list[int]], list[int], float, list[float], list[float]]:
        torch.cuda.synchronize()
        started = perf_counter()
        stream = engine.generate(
            prompt=prompts[0],
            image_data=(image_data[0][0] if len(image_data[0]) == 1 else image_data[0]),
            sampling_params=sampling,
            stream=True,
        )
        final_output: dict[str, Any] | None = None
        arrival_times: list[float] = []
        observed_tokens = 0
        for chunk in stream:
            torch.cuda.synchronize()
            final_output = chunk
            current_tokens = len(chunk["output_ids"])
            now = perf_counter()
            arrival_times.extend([now] * (current_tokens - observed_tokens))
            observed_tokens = current_tokens
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        if final_output is None or not arrival_times:
            raise RuntimeError("SGLang stream returned no output tokens")
        token_ids = [list(final_output["output_ids"])]
        prompt_tokens = [int(final_output["meta_info"]["prompt_tokens"])]
        ttft_ms = [(arrival_times[0] - started) * 1000.0]
        tpot_ms = [
            (current - previous) * 1000.0
            for previous, current in zip(arrival_times, arrival_times[1:])
        ]
        return token_ids, prompt_tokens, elapsed_ms, ttft_ms, tpot_ms

    try:
        for _ in range(args.warmup):
            run_once()
        token_runs: list[list[list[int]]] = []
        e2e_ms: list[float] = []
        ttft_ms: list[float] = []
        tpot_ms: list[float] = []
        throughput: list[float] = []
        process_memory: list[float] = []
        prompt_tokens: list[int] = []
        for _ in range(args.repeat):
            tokens, prompt_tokens, elapsed, ttft, tpot = run_once()
            token_runs.append(tokens)
            e2e_ms.append(elapsed)
            ttft_ms.extend(ttft)
            tpot_ms.extend(tpot)
            throughput.append(sum(len(row) for row in tokens) / (elapsed / 1000.0))
            process_memory.append(_nvml_compute_memory_mib())
        if any(tokens != token_runs[0] for tokens in token_runs[1:]):
            raise RuntimeError("SGLang greedy outputs changed across repeats")

        record = {
            "schema_version": 1,
            "record_type": "external_system_benchmark",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "framework": "sglang",
                "framework_version": args.framework_version,
                "framework_source_commit": args.source_commit,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "model": {
                "path": args.model,
                "dtype": "torch.bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": 1280,
                "max_num_seqs": len(requests),
                "kv_cache_capacity_tokens": 4096,
                "mem_fraction_static": 0.60,
            },
            "backend": {
                "execution": "eager",
                "attention": "triton",
                "mm_attention": "triton_attn",
                "prefix_caching": False,
                "chunked_prefill": False,
            },
            "workload": {
                "manifest_name": manifest["name"],
                "manifest_sha256": _sha256(manifest),
                "case_id": case["id"],
                "request_types": [request["type"] for request in case["requests"]],
                "num_requests": len(requests),
                "prompt_tokens": sum(prompt_tokens),
                "prompt_tokens_per_request": prompt_tokens,
                "max_tokens": args.max_tokens,
                "preprocessing_included_in_e2e": True,
                "traffic": "offline_closed_loop",
            },
            "measurement": {
                "warmup": args.warmup,
                "repeat": args.repeat,
                "cuda_synchronize_timing": True,
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
                "measurement": "nvml_compute_process_used",
                "process_used": _stats(process_memory),
            },
        }
        Path(args.output).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2, sort_keys=True))
    finally:
        engine.shutdown()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
