"""VL CUDA Graph decode benchmark.

该脚本在同一输入集合上比较 `enforce_eager=True` 和
`enforce_eager=False`。计时按 engine step 拆分 prefill/decode，并只把
`num_tokens < 0` 的步骤计入 decode latency。
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import statistics
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import collect_git_metadata
from prism_infer import LLM
from prism_infer.sampling_params import SamplingParams


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[index]


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _video_frames() -> list[Image.Image]:
    return [Image.new("RGB", (448, 448), color=(80 + i * 30, 120, 180)) for i in range(4)]


def _make_requests(case: str, suffix: str) -> list[dict]:
    if case == "single-image":
        return [
            {
                "type": "image",
                "prompt": f"Describe this image. {suffix}",
                "image": _image((100, 150, 200)),
            }
        ]
    if case == "multi-image":
        return [
            {
                "type": "images",
                "prompt": f"Compare these images. {suffix}",
                "images": [_image((100, 150, 200)), _image((200, 120, 80))],
            }
        ]
    if case == "video":
        return [
            {"type": "video", "prompt": f"Describe this video. {suffix}", "video": _video_frames()}
        ]
    if case == "mixed":
        return [
            {"type": "text", "prompt": f"Hello {suffix}"},
            {
                "type": "images",
                "prompt": f"Compare these images. {suffix}",
                "images": [_image((100, 150, 200)), _image((200, 120, 80))],
            },
            {"type": "video", "prompt": f"Describe this video. {suffix}", "video": _video_frames()},
        ]
    raise ValueError(f"unsupported case: {case}")


def _add_requests(llm: LLM, requests: list[dict], sampling: SamplingParams) -> list[int]:
    seq_ids = []
    for request in requests:
        kind = request["type"]
        if kind == "text":
            seq_ids.append(llm.add_request(request["prompt"], sampling))
        elif kind == "image":
            seq_ids.append(llm.add_vl_request(request["prompt"], request["image"], sampling))
        elif kind == "images":
            seq_ids.append(llm.add_images_request(request["prompt"], request["images"], sampling))
        elif kind == "video":
            seq_ids.append(llm.add_video_request(request["prompt"], request["video"], sampling))
        else:
            raise ValueError(kind)
    return seq_ids


def _run_once(
    llm: LLM,
    requests: list[dict],
    sampling: SamplingParams,
) -> tuple[list[list[int]], list[float], int, list[float]]:
    seq_ids = _add_requests(llm, requests, sampling)
    outputs: dict[int, list[int]] = {}
    decode_times_ms = []
    prefill_times_ms = []
    decode_tokens = 0
    while not llm.is_finished():
        torch.cuda.synchronize()
        start = perf_counter()
        finished, num_tokens = llm.step()
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0
        if num_tokens < 0:
            decode_times_ms.append(elapsed_ms)
            decode_tokens += -num_tokens
        else:
            prefill_times_ms.append(elapsed_ms)
        for seq_id, token_ids in finished:
            outputs[seq_id] = token_ids
    return [outputs[seq_id] for seq_id in seq_ids], decode_times_ms, decode_tokens, prefill_times_ms


def _new_llm(args, *, enforce_eager: bool) -> LLM:
    max_num_seqs = 3 if args.case == "mixed" else 1
    return LLM(
        args.model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=False,
        kvcache_block_size=args.kvcache_block_size,
    )


def _summarize(
    label: str, decode_times: list[float], decode_tokens: int, prefill_times: list[float]
) -> None:
    total_decode_s = sum(decode_times) / 1000.0
    token_s = decode_tokens / total_decode_s if total_decode_s > 0 else 0.0
    print(
        f"{label} decode: median={statistics.median(decode_times):.4f}ms "
        f"p90={_p90(decode_times):.4f}ms min={min(decode_times):.4f}ms "
        f"max={max(decode_times):.4f}ms token/s={token_s:.2f} "
        f"decode_steps={len(decode_times)} decode_tokens={decode_tokens}"
    )
    print(
        f"{label} prefill: median={statistics.median(prefill_times):.4f}ms "
        f"p90={_p90(prefill_times):.4f}ms min={min(prefill_times):.4f}ms "
        f"max={max(prefill_times):.4f}ms steps={len(prefill_times)}"
    )


def _bench_mode(
    args, *, enforce_eager: bool
) -> tuple[list[list[int]], list[float], int, list[float]]:
    llm = _new_llm(args, enforce_eager=enforce_eager)
    try:
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        for idx in range(args.warmup):
            _run_once(llm, _make_requests(args.case, f"warmup-{idx}"), sampling)
        decode_times = []
        prefill_times = []
        decode_tokens = 0
        last_tokens = []
        torch.cuda.reset_peak_memory_stats()
        for idx in range(args.repeat):
            tokens, step_times, step_tokens, step_prefill = _run_once(
                llm,
                _make_requests(args.case, f"repeat-{idx}"),
                sampling,
            )
            last_tokens = tokens
            decode_times.extend(step_times)
            decode_tokens += step_tokens
            prefill_times.extend(step_prefill)
        return last_tokens, decode_times, decode_tokens, prefill_times
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--case", choices=["single-image", "multi-image", "video", "mixed"], default="mixed"
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kvcache-block-size", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    git = collect_git_metadata(REPO_ROOT)
    print(f"commit: {git.commit[:12]} dirty={git.dirty}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(
        f"case={args.case}, max_tokens={args.max_tokens}, warmup={args.warmup}, "
        f"repeat={args.repeat}, kvcache_block_size={args.kvcache_block_size}"
    )
    eager_tokens, eager_decode, eager_decode_tokens, eager_prefill = _bench_mode(
        args, enforce_eager=True
    )
    graph_tokens, graph_decode, graph_decode_tokens, graph_prefill = _bench_mode(
        args, enforce_eager=False
    )

    print(f"last eager token_ids: {eager_tokens}")
    print(f"last graph token_ids: {graph_tokens}")
    print("correctness: PASS" if eager_tokens == graph_tokens else "correctness: FAIL")
    _summarize("eager", eager_decode, eager_decode_tokens, eager_prefill)
    _summarize("graph", graph_decode, graph_decode_tokens, graph_prefill)
    print(
        "memory: "
        f"allocated={torch.cuda.memory_allocated() / 2**20:.2f}MiB "
        f"reserved={torch.cuda.memory_reserved() / 2**20:.2f}MiB "
        f"peak={torch.cuda.max_memory_allocated() / 2**20:.2f}MiB"
    )


if __name__ == "__main__":
    main()
