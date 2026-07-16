#!/usr/bin/env python3
"""在固定 Prism workload 上运行 vLLM offline Qwen3-VL baseline。

该脚本必须使用安装了 vLLM 的独立环境执行。它不导入 Prism engine，也不把
offline closed-loop 结果表述为 online serving 吞吐。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SCHEMA_VERSION = 2
COMPARISON_PROFILES = ("diagnostic_matched", "best_stable")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_metadata() -> tuple[str, bool]:
    """Return the benchmark harness commit and dirty state.

    The external framework has its own source commit.  Keeping the harness
    commit separately prevents a later script edit from being confused with a
    framework change.
    """

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def _gpu_metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    metadata: dict[str, Any] = {
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        uuid, driver = (part.strip() for part in raw.split(",", maxsplit=1))
        metadata["gpu_uuid"] = uuid
        metadata["driver"] = driver
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        metadata["gpu_uuid"] = "unknown"
        metadata["driver"] = "unknown"
    return metadata


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _effective_backend(llm: LLM, *, enforce_eager: bool) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    scheduler = config.scheduler_config
    capture_sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    return {
        "execution": "eager" if enforce_eager else "cuda_graph",
        "compilation_mode": _enum_name(getattr(compilation, "mode", "unknown")),
        "compilation_backend": str(
            getattr(compilation, "backend", "unknown")
        ),
        "cudagraph_mode": _enum_name(
            getattr(compilation, "cudagraph_mode", "unknown")
        ),
        "cudagraph_capture_sizes": (
            list(capture_sizes) if capture_sizes is not None else []
        ),
        "chunked_prefill": bool(
            getattr(scheduler, "enable_chunked_prefill", False)
        ),
        "async_scheduling": bool(
            getattr(scheduler, "async_scheduling", False)
        ),
    }


def _load_manifest(path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        if case["id"] == case_id:
            return manifest, case
    raise ValueError(f"case {case_id!r} not found in {path}")


def _synthetic_image(spec: dict[str, Any]) -> Image.Image:
    return Image.new(
        "RGB",
        (int(spec["width"]), int(spec["height"])),
        tuple(int(channel) for channel in spec["color"]),
    )


def _file_image(spec: dict[str, Any]) -> Image.Image:
    configured = Path(spec["path"])
    path = configured if configured.is_absolute() else REPO_ROOT / configured
    if not path.is_file():
        raise FileNotFoundError(f"benchmark image is missing: {path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != spec["sha256"]:
        raise ValueError(
            f"benchmark image SHA256 mismatch: expected {spec['sha256']}, "
            f"got {actual_sha256}"
        )
    with Image.open(path) as source:
        image = source.convert("RGB")
    expected_size = (int(spec["width"]), int(spec["height"]))
    if image.size != expected_size:
        raise ValueError(
            f"benchmark image size mismatch: expected {expected_size}, got {image.size}"
        )
    return image


def _materialize_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for request in case["requests"]:
        request_type = request["type"]
        item: dict[str, Any] = {"type": request_type, "prompt": request["prompt"]}
        if request_type == "image":
            item["images"] = [_synthetic_image(request["image"])]
        elif request_type == "image_file":
            item["images"] = [_file_image(request["image"])]
        elif request_type == "images":
            item["images"] = [_synthetic_image(spec) for spec in request["images"]]
        elif request_type == "video":
            item["video"] = [_synthetic_image(spec) for spec in request["frames"]]
        elif request_type != "text":
            raise ValueError(f"unsupported request type: {request_type!r}")
        requests.append(item)
    return requests


def _build_vllm_prompts(
    processor: Any,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request in requests:
        if request["type"] == "text":
            prompts.append({"prompt": request["prompt"]})
            continue
        content: list[dict[str, Any]] = []
        multi_modal_data: dict[str, Any] = {}
        if "images" in request:
            images = request["images"]
            content.extend({"type": "image", "image": image} for image in images)
            multi_modal_data["image"] = images[0] if len(images) == 1 else images
        elif "video" in request:
            content.append({"type": "video", "video": request["video"]})
            frames = request["video"]
            frame_count = len(frames)
            fps = 2.0
            # vLLM 0.24.0 的 Qwen3-VL parser 要求内存视频携带 HF metadata。
            multi_modal_data["video"] = (
                np.stack([np.asarray(frame) for frame in frames]),
                {
                    "total_num_frames": frame_count,
                    "fps": fps,
                    "duration": frame_count / fps,
                    "video_backend": "prism_synthetic_frames",
                    "frames_indices": list(range(frame_count)),
                },
            )
        content.append({"type": "text", "text": request["prompt"]})
        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt: dict[str, Any] = {"prompt": prompt_text}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument(
        "--comparison-profile",
        choices=COMPARISON_PROFILES,
        help=(
            "diagnostic_matched requires eager; best_stable requires the "
            "framework's CUDA Graph path. Defaults from --enforce-eager."
        ),
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "wrap measured repeats in cudaProfilerStart/Stop so Nsight "
            "captures generation without model load and warmup"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeat < 1 or args.max_tokens < 2:
        raise SystemExit("warmup >= 0, repeat >= 1 and max-tokens >= 2 are required")
    comparison_profile = args.comparison_profile or (
        "diagnostic_matched" if args.enforce_eager else "best_stable"
    )
    if comparison_profile == "diagnostic_matched" and not args.enforce_eager:
        raise SystemExit("diagnostic_matched requires --enforce-eager")
    if comparison_profile == "best_stable" and args.enforce_eager:
        raise SystemExit("best_stable must not use --enforce-eager")
    manifest_path = Path(args.manifest)
    manifest, case = _load_manifest(manifest_path, args.case)
    requests = _materialize_case(case)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    image_limit = max(
        (len(request.get("images", [])) for request in requests),
        default=0,
    )
    video_limit = max((1 if "video" in request else 0 for request in requests), default=0)
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
        max_num_seqs=len(requests),
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        attention_config={"backend": args.attention_backend},
        enable_chunked_prefill=(comparison_profile == "best_stable"),
        async_scheduling=(comparison_profile == "best_stable"),
        disable_log_stats=False,
        seed=0,
    )
    effective_backend = _effective_backend(llm, enforce_eager=args.enforce_eager)
    if args.enforce_eager and effective_backend["cudagraph_mode"] != "NONE":
        raise RuntimeError(
            "vLLM eager baseline unexpectedly retained CUDA Graph mode: "
            f"{effective_backend['cudagraph_mode']}"
        )
    if (
        not args.enforce_eager
        and effective_backend["cudagraph_mode"] in ("NONE", "unknown")
    ):
        raise RuntimeError(
            "vLLM best-stable baseline did not enable an effective CUDA Graph mode"
        )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    def run_once() -> tuple[
        list[list[int]],
        float,
        list[float],
        list[float],
        list[int],
    ]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = perf_counter()
        prompts = _build_vllm_prompts(processor, requests)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        ttft_ms: list[float] = []
        tpot_ms: list[float] = []
        prompt_token_counts: list[int] = []
        for output, generated in zip(outputs, token_ids, strict=True):
            if output.metrics is None:
                raise RuntimeError("vLLM RequestOutput.metrics is unavailable")
            ttft_ms.append(float(output.metrics.first_token_latency) * 1000.0)
            if len(generated) > 1:
                decode_seconds = max(
                    0.0,
                    float(output.metrics.last_token_ts - output.metrics.first_token_ts),
                )
                tpot_ms.append(decode_seconds * 1000.0 / (len(generated) - 1))
            prompt_token_counts.append(len(output.prompt_token_ids or []))
        return token_ids, elapsed_ms, ttft_ms, tpot_ms, prompt_token_counts

    for _ in range(args.warmup):
        run_once()

    token_runs: list[list[list[int]]] = []
    e2e_ms: list[float] = []
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    throughput: list[float] = []
    allocated_mb: list[float] = []
    reserved_mb: list[float] = []
    peak_allocated_mb: list[float] = []
    prompt_token_counts: list[int] = []
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(args.repeat):
            tokens, elapsed, request_ttft, request_tpot, prompt_token_counts = run_once()
            token_runs.append(tokens)
            e2e_ms.append(elapsed)
            ttft_ms.extend(request_ttft)
            tpot_ms.extend(request_tpot)
            throughput.append(sum(len(row) for row in tokens) / (elapsed / 1000.0))
            allocated_mb.append(torch.cuda.memory_allocated() / 1024 / 1024)
            reserved_mb.append(torch.cuda.memory_reserved() / 1024 / 1024)
            peak_allocated_mb.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
    finally:
        if args.cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
    if any(tokens != token_runs[0] for tokens in token_runs[1:]):
        raise RuntimeError("vLLM greedy token ids changed across measured repeats")

    harness_commit, harness_dirty = _git_metadata()
    gpu_metadata = _gpu_metadata()
    model_config_path = Path(args.model) / "config.json"
    record = {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "record_type": "external_system_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "p7.1_external_offline_v2",
            "comparison_profile": comparison_profile,
            "harness_git_commit": harness_commit,
            "harness_git_dirty": harness_dirty,
            "framework_source_dirty": args.source_dirty,
            "process_scope": "fresh_process_per_case_and_backend",
            "command": [sys.executable, *sys.argv],
        },
        "environment": {
            "framework": "vllm",
            "framework_version": importlib.metadata.version("vllm"),
            "framework_source_commit": args.source_commit,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            **gpu_metadata,
        },
        "model": {
            "path": str(Path(args.model).resolve()),
            "config_sha256": (
                hashlib.sha256(model_config_path.read_bytes()).hexdigest()
                if model_config_path.is_file()
                else "unknown"
            ),
            "dtype": "torch.bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": len(requests),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        },
        "backend": {
            **effective_backend,
            "attention": args.attention_backend,
            "block_size": args.block_size,
            "prefix_caching": False,
            "mm_processor_cache_gb": 0,
            "sampler": (
                "pytorch_native"
                if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
                else "vllm_default"
            ),
            "engine_multiprocessing": (
                os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0"
            ),
        },
        "workload": {
            "manifest_name": manifest["name"],
            "manifest_sha256": _canonical_sha256(manifest),
            "case_id": case["id"],
            "request_types": [request["type"] for request in case["requests"]],
            "num_requests": len(requests),
            "prompt_tokens": sum(prompt_token_counts),
            "prompt_tokens_per_request": prompt_token_counts,
            "max_tokens": args.max_tokens,
            "preprocessing_included_in_e2e": True,
            "traffic": "offline_closed_loop",
            "output_decoding_included_in_e2e": False,
        },
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": args.max_tokens,
        },
        "measurement": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cuda_synchronize_timing": True,
            "cuda_profiler_range": args.cuda_profiler_range,
            "engine_ttft_scope": "vllm_request_metrics_first_token_latency",
            "decode_tpot_scope": "first_to_last_token_divided_by_output_intervals",
            "end_to_end_scope": "prompt_build_plus_llm_generate",
        },
        "correctness": {
            "outputs_identical_across_repeats": True,
            "token_ids": token_runs[0],
            "output_sha256": _canonical_sha256(token_runs[0]),
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
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
