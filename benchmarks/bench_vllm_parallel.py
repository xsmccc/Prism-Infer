"""Short vLLM 0.25.1 TP/PP comparison through its AsyncLLM token client.

This compares parallel strategies inside vLLM, not absolute Prism/vLLM ranking.
Images and question text match bench_vision_parallel.py; chat templating and
model startup are outside timing. Timing begins immediately before generate().
AsyncLLM may coalesce several tokens into one yield: keep those raw chunks and
assign their observed timestamp to each token, without inventing token timings.
For two independent TP1 replicas, use separate CUDA_VISIBLE_DEVICES processes
and --wait-for-start; send START after each prints VLLM_PARALLEL_READY.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_vision_parallel import DEFAULT_HOT_PROMPT, DEFAULT_PROMPT, _load_images


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--hot-prompt", default=DEFAULT_HOT_PROMPT)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--pp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--encoder-mode", choices=("weights", "data"), default="weights")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--kv-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--replica-id", default="0")
    parser.add_argument("--wait-for-start", action="store_true")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "finished"]
    if not complete:
        return {"completed_requests": 0}
    start = min(row["client_start_ns"] for row in complete)
    end = max(row["client_finish_ns"] for row in complete)
    duration_s = (end - start) / 1e9
    tokens = sum(len(row["token_ids"]) for row in complete)
    return {
        "completed_requests": len(complete),
        "client_start_ns": start,
        "client_finish_ns": end,
        "duration_s": duration_s,
        "output_tokens": tokens,
        "output_tokens_per_s": tokens / duration_s if duration_s > 0 else None,
        "ttft_median_ms": statistics.median(row["ttft_ms"] for row in complete),
        "tpot_median_ms": statistics.median(row["tpot_ms"] for row in complete),
        "e2e_median_ms": statistics.median(row["e2e_ms"] for row in complete),
    }


async def _consume(
    engine: Any,
    sampling: Any,
    images: list[Any],
    formatted_prompt: str,
    row: dict[str, Any],
) -> None:
    prompt = {"prompt": formatted_prompt, "multi_modal_data": {"image": images}}
    row["client_start_ns"] = perf_counter_ns()
    row["status"] = "running"
    async for output in engine.generate(prompt, sampling, request_id=row["request_id"]):
        arrived_ns = perf_counter_ns()
        if output.prompt_token_ids is not None:
            row["prompt_token_ids"] = list(output.prompt_token_ids)
        if output.num_cached_tokens is not None:
            row["num_cached_tokens"] = output.num_cached_tokens
        for part in output.outputs:
            if part.index != 0:
                raise RuntimeError("this benchmark expects one completion per request")
            new_ids = list(part.token_ids)  # RequestOutputKind.DELTA
            if new_ids:
                row["chunks"].append({"arrival_ns": arrived_ns, "token_ids": new_ids})
                row["token_ids"].extend(new_ids)
                row["token_arrival_ns"].extend([arrived_ns] * len(new_ids))
            if part.finish_reason is not None:
                row["finish_reason"] = part.finish_reason
        if output.finished:
            row["client_finish_ns"] = arrived_ns
    if "client_finish_ns" not in row or len(row["token_ids"]) < 2:
        raise RuntimeError("request ended without a final output and at least two tokens")
    arrivals = row["token_arrival_ns"]
    row["ttft_ms"] = (arrivals[0] - row["client_start_ns"]) / 1e6
    row["itl_ms"] = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:], strict=False)]
    row["tpot_ms"] = statistics.mean(row["itl_ms"])
    row["e2e_ms"] = (row["client_finish_ns"] - row["client_start_ns"]) / 1e6
    row["output_tokens"] = len(row["token_ids"])
    row["coalesced_chunk_count"] = sum(len(chunk["token_ids"]) > 1 for chunk in row["chunks"])
    row["status"] = "finished"


async def _reset_caches(engine: Any) -> None:
    # Only called after the preceding requests have fully drained.
    if not await engine.reset_prefix_cache():
        raise RuntimeError("vLLM did not clear its prefix cache before a cold phase")
    await engine.reset_encoder_cache()
    await engine.reset_mm_cache()


async def _run(args: argparse.Namespace, report: dict[str, Any], images: list[Any]) -> None:
    import torch
    from transformers import AutoConfig, AutoProcessor
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from benchmarks.harness import collect_git_metadata, collect_gpu_metadata

    report["vllm_version"] = importlib.metadata.version("vllm")
    report["torch_version"] = torch.__version__
    report["git"] = collect_git_metadata(REPO_ROOT).as_dict()
    report["gpus"] = [asdict(collect_gpu_metadata(i)) for i in range(args.tp * args.pp)]
    options = {
        "model": args.model,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "distributed_executor_backend": "mp",
        "mm_encoder_tp_mode": args.encoder_mode,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "block_size": args.block_size,
        "num_gpu_blocks_override": args.kv_blocks,
        "gpu_memory_utilization": 0.85,
        "limit_mm_per_prompt": {"image": len(images), "video": 0},
        "mm_processor_kwargs": {"max_pixels": 448 * 448},
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": False,
        "disable_log_stats": False,
    }
    if args.pp > 1:
        # v0.25.1 needs this label when validating Qwen3-VL's language sub-config.
        options["hf_overrides"] = {"text_config": {"architectures": ["Qwen3ForCausalLM"]}}
    report["engine_options"] = options
    hf_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    layers = text_config.num_hidden_layers
    kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    report["configured_kv_capacity"] = {
        "logical_token_slots_per_engine": args.kv_blocks * args.block_size,
        "bf16_payload_bytes_all_ranks": (
            2 * layers * kv_heads * head_dim * 2 * args.kv_blocks * args.block_size
        ),
        "note": "TP2 and PP2 shard the same logical KV; two TP1 engines duplicate this capacity",
    }
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    formatted = {}
    for text in (args.prompt, args.hot_prompt):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": text}],
            }
        ]
        formatted[text] = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    report["formatted_prompts"] = formatted
    engine = None
    try:
        load_start = perf_counter_ns()
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        report["model_load_ms"] = (perf_counter_ns() - load_start) / 1e6
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )

        async def batch(phase: str, question: str, count: int) -> None:
            rows = []
            for index in range(count):
                row = {
                    "request_id": f"{args.replica_id}-{len(report['requests'])}",
                    "replica_id": args.replica_id,
                    "phase": phase,
                    "phase_request_index": index,
                    "question": question,
                    "token_ids": [],
                    "token_arrival_ns": [],
                    "chunks": [],
                    "status": "created",
                }
                report["requests"].append(row)
                rows.append(row)
            await asyncio.gather(
                *(_consume(engine, sampling, images, formatted[question], row) for row in rows)
            )
            if any(row["output_tokens"] != args.max_tokens for row in rows):
                raise RuntimeError("fixed-length greedy request returned an unexpected token count")
            print(
                "VLLM_PARALLEL_PHASE " + json.dumps({"phase": phase, **_summary(rows)}),
                flush=True,
            )

        for _ in range(args.warmup):
            await batch("warmup", args.prompt, 1)
        for _ in range(args.repeat):
            await _reset_caches(engine)
            await batch("sequential_cold", args.prompt, 1)
            await batch("sequential_hot_new_question", args.hot_prompt, 1)
        if args.wait_for_start:
            print("VLLM_PARALLEL_READY " + json.dumps({"replica_id": args.replica_id}), flush=True)
            if sys.stdin.readline().strip() != "START":
                raise RuntimeError("expected START on stdin before concurrent phases")
        await _reset_caches(engine)
        await batch("concurrent_cold_start", args.prompt, args.concurrent_requests)
        await batch("concurrent_hot", args.hot_prompt, args.concurrent_requests)
        cold_ids = next(
            row["token_ids"] for row in report["requests"] if row["phase"] == "sequential_cold"
        )
        hot_ids = next(
            row["token_ids"]
            for row in report["requests"]
            if row["phase"] == "sequential_hot_new_question"
        )
        for row in report["requests"]:
            reference = cold_ids if row["question"] == args.prompt else hot_ids
            row["matches_same_question_sequential_ids"] = row["token_ids"] == reference
    finally:
        if engine is not None:
            engine.shutdown()


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.tp * args.pp > 2:
        parser.error("this short comparison uses at most two GPUs per engine")
    if args.repeat < 1 or args.warmup < 0 or args.max_tokens < 2:
        parser.error("repeat >= 1, warmup >= 0 and max-tokens >= 2 are required")
    if min(args.concurrent_requests, args.image_size, args.kv_blocks, args.block_size) < 1:
        parser.error("request count, image size and KV sizes must be positive")
    images, input_record = _load_images(args)
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "replica_id": args.replica_id,
        "model_path": str(Path(args.model).resolve()),
        "model_revision": args.model_revision,
        "input": input_record,
        "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens, "ignore_eos": True},
        "protocol": {
            "scope": "vLLM internal TP/PP direction comparison, not an absolute Prism ranking",
            "client": "AsyncLLM.generate with DELTA RequestOutput",
            "ttft": "immediately before generate to first nonempty token yield",
            "itl": "observed client token intervals; coalesced tokens share one timestamp",
            "e2e": "immediately before generate to finished RequestOutput",
            "image_loading_and_chat_template_in_timing": False,
            "vllm_processing_and_admission_in_timing": True,
            "cold": "drain then reset prefix, encoder and multimodal processor caches",
            "hot": "same images, changed question, resident prefix and encoder caches",
            "concurrent_cold": "cold at burst start; reuse within the burst is allowed",
            "concurrency": args.concurrent_requests,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "throughput_scope": "finite burst including prefill and drain, not saturation",
            "pp_encoder": (
                "only first PP stage runs vision; data mode splits TP ranks, not PP stages"
            ),
            "async_scheduling": False,
        },
        "requests": [],
    }
    try:
        asyncio.run(_run(args, report, images))
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["phase_summaries"] = {
            phase: _summary([row for row in report["requests"] if row["phase"] == phase])
            for phase in dict.fromkeys(row["phase"] for row in report["requests"])
        }
        for image in images:
            image.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
