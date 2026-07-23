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
from typing import Any, Iterator

import cv2
import numpy as np
import pynvml
import torch
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import collect_git_metadata, collect_gpu_metadata


EXTERNAL_SCHEMA_VERSION = 2
DEFAULT_VIDEO_FPS = 24.0


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


def _stage_lossless_video(
    frames: list[Image.Image],
    *,
    path: Path,
    fps: float,
) -> dict[str, Any]:
    arrays = np.stack([np.asarray(frame) for frame in frames])
    height, width = arrays.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the lossless FFV1 video")
    try:
        for frame in arrays:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    with VideoDecoderWrapper(str(path), device="cpu") as decoder:
        decoded = decoder.get_frames_at(list(range(len(decoder))))
        decoded_fps = decoder.avg_fps
    if decoded.shape != arrays.shape or not np.array_equal(decoded, arrays):
        raise RuntimeError("SGLang video staging changed the frozen RGB frames")
    if decoded_fps != fps:
        raise RuntimeError(f"SGLang video staging changed fps: {decoded_fps} != {fps}")
    return {
        "codec": "ffv1",
        "container": "matroska",
        "frames": len(frames),
        "fps": fps,
        "height": height,
        "width": width,
        "decoded_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_exact": True,
    }


def _materialize(
    case: dict[str, Any],
    *,
    video_staging_path: Path,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for request in case["requests"]:
        request_type = request["type"]
        if request_type in ("image", "image_file"):
            images = [_image(request["image"])]
        elif request_type == "images":
            images = [_image(spec) for spec in request["images"]]
        elif request_type == "video":
            frames = [_image(spec) for spec in request["frames"]]
            fps = float(request.get("fps", DEFAULT_VIDEO_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"video fps must be finite and positive: {fps}")
            staging = _stage_lossless_video(
                frames,
                path=video_staging_path,
                fps=fps,
            )
            requests.append(
                {
                    "type": request_type,
                    "prompt": request["prompt"],
                    "video": str(video_staging_path),
                    "fps": fps,
                    "video_staging": staging,
                }
            )
            continue
        else:
            raise ValueError(
                "SGLang P6 adapter supports image/image_file/images/video only; "
                f"got {request_type!r}"
            )
        requests.append({"type": request_type, "prompt": request["prompt"], "images": images})
    return requests


def _prompts(processor: Any, requests: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for request in requests:
        if "video" in request:
            content = [{"type": "video", "video": request["video"]}]
        else:
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


def _generate_stream_with_prompt_ids(
    engine: Engine,
    *,
    prompt: str,
    image_data: Any,
    video_data: Any,
    sampling_params: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Run the public Engine request path while auditing post-tokenization IDs."""

    request = GenerateReqInput(
        text=prompt,
        image_data=image_data,
        video_data=video_data,
        sampling_params=sampling_params,
        stream=True,
        return_prompt_token_ids=True,
    )
    generator = engine.tokenizer_manager.generate_request(request, None)
    while True:
        try:
            yield engine.loop.run_until_complete(generator.__anext__())
        except StopAsyncIteration:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.60)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mm-attention-backend", default="triton_attn")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    case = next((item for item in manifest["cases"] if item["id"] == args.case), None)
    if case is None:
        raise ValueError(f"case not found: {args.case}")
    output_path = Path(args.output)
    requests = _materialize(
        case,
        video_staging_path=output_path.with_suffix(".video.mkv"),
    )
    if len(requests) != 1:
        raise ValueError("SGLang streaming P6 adapter currently requires one request")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    prompts = _prompts(processor, requests)
    image_data = [request.get("images") for request in requests]
    video_data = [request.get("video") for request in requests]
    mm_process_config = (
        {
            "video": {
                "fps": requests[0]["fps"],
            }
        }
        if video_data[0] is not None
        else {}
    )

    pynvml.nvmlInit()
    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=args.max_model_len,
        max_total_tokens=args.max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=len(requests),
        chunked_prefill_size=-1,
        disable_cuda_graph=args.enforce_eager,
        cuda_graph_max_bs_decode=len(requests),
        disable_radix_cache=True,
        attention_backend=args.attention_backend,
        mm_attention_backend=args.mm_attention_backend,
        enable_request_time_stats_logging=True,
        stream_interval=1,
        random_seed=0,
        mm_process_config=mm_process_config,
    )
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": True,
    }

    def run_once() -> tuple[
        list[list[int]],
        list[int],
        list[list[int]],
        float,
        list[float],
        list[float],
    ]:
        torch.cuda.synchronize()
        started = perf_counter()
        stream = _generate_stream_with_prompt_ids(
            engine,
            prompt=prompts[0],
            image_data=(
                None
                if image_data[0] is None
                else (image_data[0][0] if len(image_data[0]) == 1 else image_data[0])
            ),
            video_data=video_data[0],
            sampling_params=sampling,
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
        prompt_token_ids = [list(final_output["prompt_token_ids"])]
        ttft_ms = [(arrival_times[0] - started) * 1000.0]
        if len(arrival_times) < 2:
            raise RuntimeError("SGLang TPOT measurement requires at least two output tokens")
        tpot_ms = [
            (arrival_times[-1] - arrival_times[0])
            * 1000.0
            / (len(arrival_times) - 1)
        ]
        return token_ids, prompt_tokens, prompt_token_ids, elapsed_ms, ttft_ms, tpot_ms

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
        prompt_token_runs: list[list[list[int]]] = []
        for _ in range(args.repeat):
            tokens, prompt_tokens, prompt_token_ids, elapsed, ttft, tpot = run_once()
            token_runs.append(tokens)
            prompt_token_runs.append(prompt_token_ids)
            e2e_ms.append(elapsed)
            ttft_ms.extend(ttft)
            tpot_ms.extend(tpot)
            throughput.append(sum(len(row) for row in tokens) / (elapsed / 1000.0))
            process_memory.append(_nvml_compute_memory_mib())
        if any(tokens != token_runs[0] for tokens in token_runs[1:]):
            raise RuntimeError("SGLang greedy outputs changed across repeats")
        if any(ids != prompt_token_runs[0] for ids in prompt_token_runs[1:]):
            raise RuntimeError("SGLang prompt token IDs changed across repeats")
        audited_prompt_ids = prompt_token_runs[0]
        if [len(ids) for ids in audited_prompt_ids] != prompt_tokens:
            raise RuntimeError("SGLang prompt token count disagrees with audited token IDs")

        git = collect_git_metadata(REPO_ROOT)
        gpu_metadata = collect_gpu_metadata().environment_dict()
        record = {
            "schema_version": EXTERNAL_SCHEMA_VERSION,
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
                **gpu_metadata,
            },
            "model": {
                "path": args.model,
                "dtype": "torch.bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": args.max_model_len,
                "max_num_seqs": len(requests),
                "kv_cache_capacity_tokens": args.max_total_tokens,
                "mem_fraction_static": args.mem_fraction_static,
            },
            "backend": {
                "execution": "eager" if args.enforce_eager else "cuda_graph",
                "attention": args.attention_backend,
                "mm_attention": args.mm_attention_backend,
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
                "prompt_token_ids_sha256": _sha256(audited_prompt_ids),
                "media_identity": [
                    request["video_staging"]
                    for request in requests
                    if "video_staging" in request
                ],
                "max_tokens": args.max_tokens,
                "preprocessing_included_in_e2e": True,
                "traffic": "offline_closed_loop",
            },
            "measurement": {
                "warmup": args.warmup,
                "repeat": args.repeat,
                "cuda_synchronize_timing": True,
                "decode_tpot_scope": "first_to_last_token_divided_by_output_intervals",
            },
            "protocol": {
                "name": "p9_external_sglang_offline_v2",
                "harness_git_commit": git.commit,
                "harness_git_dirty": git.dirty,
                "framework_source_dirty": False,
                "command": [sys.executable, *sys.argv],
                "process_scope": "fresh_process_per_case_and_backend",
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
        output_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2, sort_keys=True))
    finally:
        engine.shutdown()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
