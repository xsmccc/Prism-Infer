"""Compare replicated/data-parallel vision through one in-process token client.

Each process loads one LLM once. TTFT starts before add_images_request (including
Processor and admission); token arrivals are observed after step_result returns.
This is not HTTP latency or synchronized kernel/engine-step timing. For two TP1
replicas, start two processes on separate CUDA_VISIBLE_DEVICES and optionally use
--wait-for-start: cold and hot phases each emit VISION_PARALLEL_READY with a phase
name; write START on each process stdin at both phase barriers.
"""

from __future__ import annotations

import argparse
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

FIXTURE_COLORS = (
    (220, 50, 50),
    (50, 180, 80),
    (60, 100, 220),
    (220, 180, 40),
    (170, 70, 200),
    (40, 190, 190),
    (230, 120, 40),
    (120, 120, 120),
)
DEFAULT_PROMPT = (
    "Compare all eight images carefully. Describe the important details in each "
    "image, then identify similarities, differences, and any cross-image pattern."
)
DEFAULT_HOT_PROMPT = "Compare the first and last images. Describe their most important differences."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", help="Explicit snapshot label, if not encoded in path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, help="Real image paths in model order")
    parser.add_argument("--image-size", type=int, default=448, help="Eight-color fixture size only")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--hot-prompt", default=DEFAULT_HOT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--vision-mode", choices=("replicated", "data"), default="replicated")
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "cuda_graph"),
        default="cuda_graph",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--kvcache-block-size", type=int, choices=(16, 256), default=256)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3, help="Sequential cold/hot repetitions")
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--replica-id", default="0")
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="Wait for START on stdin separately before cold and hot concurrent phases",
    )
    return parser


def _load_images(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    from PIL import Image

    if args.images:
        images = []
        for path in args.images:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        source: dict[str, Any] = {"kind": "files", "paths": [str(p.resolve()) for p in args.images]}
    else:
        images = [Image.new("RGB", (args.image_size, args.image_size), c) for c in FIXTURE_COLORS]
        source = {"kind": "h1_eight_color_fixture", "rgb_colors": list(FIXTURE_COLORS)}
    source["image_sizes_wh"] = [list(image.size) for image in images]
    source["image_count"] = len(images)
    return images, source


def _finish_timings(row: dict[str, Any]) -> None:
    arrivals = row["token_arrival_ns"]
    row["itl_ms"] = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:], strict=False)]
    row["ttft_ms"] = (arrivals[0] - row["client_start_ns"]) / 1e6 if arrivals else None
    row["tpot_ms"] = statistics.mean(row["itl_ms"]) if row["itl_ms"] else None
    row["e2e_ms"] = (row["client_finish_ns"] - row["client_start_ns"]) / 1e6
    row["output_tokens"] = len(row["token_ids"])
    row["observed_ids_match_final_ids"] = row["observed_token_ids"] == row["token_ids"]


def _run_requests(
    llm: Any,
    images: list[Any],
    prompts: list[str],
    sampling: Any,
    *,
    phase: str,
    prefix_enabled: bool,
    output_rows: list[dict[str, Any]],
    replica_id: str,
    interleaved: bool = False,
) -> list[dict[str, Any]]:
    """Submit all requests, then consume the same client-visible StepResult API."""
    if not llm.is_finished():
        raise RuntimeError("a benchmark phase must start with no active requests")
    llm.scheduler.block_manager.enable_prefix_caching = prefix_enabled
    rows: dict[int, dict[str, Any]] = {}
    for index, prompt in enumerate(prompts):
        row: dict[str, Any] = {
            "phase": phase,
            "phase_request_index": index,
            "replica_id": replica_id,
            "prefix_enabled": prefix_enabled,
            "prompt": prompt,
            "submission_api": (
                "add_interleaved_images_request" if interleaved else "add_images_request"
            ),
            "image_sizes_wh": [list(image.size) for image in images],
            "token_arrival_ns": [],
            "observed_token_ids": [],
            "token_ids": [],
            "status": "submitting",
            "prefill_slices": [],
        }
        output_rows.append(row)
        row["client_start_ns"] = perf_counter_ns()
        add_request = llm.add_interleaved_images_request if interleaved else llm.add_images_request
        request_id = add_request(prompt, images, sampling)
        row["add_request_return_ns"] = perf_counter_ns()
        row["request_id"] = request_id
        sequence = next(seq for seq in llm.scheduler.waiting if seq.seq_id == request_id)
        row["prompt_token_ids"] = list(sequence.prompt_token_ids)
        row["prompt_tokens"] = len(row["prompt_token_ids"])
        row["image_grid_thw"] = sequence.image_grid_thw.tolist()
        spans = sequence.image_token_spans()
        row["image_token_spans"] = None if spans is None else [list(span) for span in spans]
        row["prefix_candidate_tokens_at_submit"] = sequence.prefix_cache_candidate_tokens
        row["status"] = "running"
        rows[request_id] = row

    while not llm.is_finished():
        step = llm.step_result()
        arrived_ns = perf_counter_ns()
        for index, (seq, token_id) in enumerate(
            zip(step.plan.sequences, step.execution.token_ids, strict=True)
        ):
            row = rows[seq.seq_id]
            if "cached_tokens_at_first_step" not in row:
                row["cached_tokens_at_first_step"] = seq.num_cached_tokens
                row["prefix_boundary"] = seq.multimodal_prefix_boundary
                row["prefix_entry_hit"] = seq.multimodal_prefix_cache_hit
                boundary = seq.multimodal_prefix_boundary
                row["full_visual_prefix_reused"] = (
                    boundary is not None and seq.num_cached_tokens >= boundary
                )
            if step.plan.is_prefill:
                part = step.plan.prefill_slices[index]
                row["prefill_slices"].append(
                    {
                        "token_start": part.token_start,
                        "token_end": part.token_end,
                        "vision_patch_rows": seq.vision_patch_count_for_prefill_range(
                            part.token_start, part.token_end
                        ),
                    }
                )
            # Chunked/boundary prefill may compute KV without emitting a token.
            if token_id is not None:
                row["observed_token_ids"].append(int(token_id))
                row["token_arrival_ns"].append(arrived_ns)
        for result in step.outputs:
            row = rows[result.request_id]
            row["token_ids"] = list(result.token_ids)
            row["finish_reason"] = result.finish_reason
            row["client_finish_ns"] = arrived_ns
            row["status"] = "finished"
            _finish_timings(row)

    if any(row["status"] != "finished" for row in rows.values()):
        raise RuntimeError("engine stopped without returning every final request output")
    return list(rows.values())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [row for row in rows if row["status"] == "finished"]
    if not finished:
        return {"completed_requests": 0}
    start = min(row["client_start_ns"] for row in finished)
    end = max(row["client_finish_ns"] for row in finished)
    duration_s = (end - start) / 1e9
    output_tokens = sum(row["output_tokens"] for row in finished)
    return {
        "completed_requests": len(finished),
        "client_start_ns": start,
        "client_finish_ns": end,
        "duration_s": duration_s,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / duration_s if duration_s > 0 else None,
        "requests_per_s": len(finished) / duration_s if duration_s > 0 else None,
        "ttft_median_ms": statistics.median(row["ttft_ms"] for row in finished),
        "tpot_median_ms": statistics.median(row["tpot_ms"] for row in finished),
        "e2e_median_ms": statistics.median(row["e2e_ms"] for row in finished),
    }


def _partial_prefix_case(
    llm: Any,
    sampling: Any,
    report: dict[str, Any],
    replica_id: str,
) -> None:
    """Exercise A+B -> A+C through the real interleaved image submission API."""
    from PIL import Image

    images = [Image.new("RGB", (448, 448), color) for color in FIXTURE_COLORS[:3]]
    description = (
        "The first reference stays unchanged and must be compared with the next reference. "
    )
    filler = description * 32
    while len(llm.tokenizer.encode(filler, add_special_tokens=False)) < 256:
        filler += description
    prompt = (
        "Reference A: <image>\n"
        + filler
        + "\nReference B: <image>\nWhat are the main visual differences between A and B?"
    )

    def run(phase: str, selected: list[Any], prefix: bool) -> dict[str, Any]:
        return _run_requests(
            llm,
            selected,
            [prompt],
            sampling,
            phase=phase,
            prefix_enabled=prefix,
            output_rows=report["requests"],
            replica_id=replica_id,
            interleaved=True,
        )[0]

    try:
        reference = run("partial_ac_reference_prefix_disabled", [images[0], images[2]], False)
        prime = run("partial_ab_prime", [images[0], images[1]], True)
        partial = run("partial_ac_hit", [images[0], images[2]], True)
        spans = partial["image_token_spans"]
        cached = partial["cached_tokens_at_first_step"]
        report["partial_prefix_validation"] = {
            "input": "448x448 fixture colors A=red, B=green, C=blue",
            "fixed_text_tokens_between_images": len(
                llm.tokenizer.encode(filler, add_special_tokens=False)
            ),
            "second_image_grid_equal": prime["image_grid_thw"][1] == partial["image_grid_thw"][1],
            "first_image_fully_reused": bool(spans and cached >= spans[0][1]),
            "changed_second_image_not_reused": bool(spans and cached <= spans[1][0]),
            "cached_tokens": cached,
            "matches_apc_off_reference_ids": partial["token_ids"] == reference["token_ids"],
            "reference_token_ids": reference["token_ids"],
            "partial_token_ids": partial["token_ids"],
        }
    finally:
        for image in images:
            image.close()


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.max_tokens < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("max-tokens >= 2, repeat >= 1 and warmup >= 0 are required")
    if args.concurrent_requests < 1 or args.image_size < 1:
        parser.error("concurrent-requests and image-size must be positive")
    images, input_record = _load_images(args)
    options = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "vision_encoder_parallel_mode": args.vision_mode,
        "execution_backend": args.execution_backend,
        "enforce_eager": args.execution_backend == "eager",
        "decode_compile_region": "none",
        "decode_compile_mode": "default",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "kvcache_block_size": args.kvcache_block_size,
        "compression_mode": "scaled_fp8_kv",
        "logits_precision": "model",
        "enable_prefix_caching": True,
        "enable_visual_embedding_cache": False,
        "enable_chunked_prefill": False,
        "mlp_projection_mode": "packed",
        "paged_decode_block_n": 256,
        "enable_fused_qk_rmsnorm": True,
        "enable_fused_qk_mrope": True,
        "enable_fused_add_rmsnorm": True,
        "enable_packed_kv_projection": True,
        "vision_attention_backend": "sdpa",
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "replica_id": args.replica_id,
        "model_path": str(Path(args.model).resolve()),
        "model_revision": args.model_revision,
        "engine_options": options,
        "input": input_record,
        "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens, "ignore_eos": True},
        "protocol": {
            "client": "in_process_add_images_request_then_step_result",
            "ttft": "client_start_before_add_request_to_first_returned_token",
            "tpot": "mean_client_token_intervals_equals_last_minus_first_over_N_minus_1",
            "e2e": "client_start_before_add_request_to_final_output_return",
            "image_file_loading_in_timing": False,
            "processor_and_admission_in_timing": True,
            "explicit_per_step_cuda_synchronize": False,
            "prefix_disabled_for_cold_repeats": True,
            "prefix_prime_is_cache_enabled_cold_first_visit": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "concurrent_requests": args.concurrent_requests,
            "concurrent_policy": "submit_all_before_driving_engine",
            "throughput_scope": "finite_batch_including_prefill_and_drain_not_saturation",
        },
        "requests": [],
        "phase_summaries": {},
    }
    llm = None
    try:
        import torch

        from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
        from prism_infer import LLM, SamplingParams
        from prism_infer.engine.kv_quantization import kv_cache_storage_bytes

        report["git"] = collect_git_metadata(REPO_ROOT).as_dict()
        report["torch"] = torch.__version__
        report["gpus"] = [asdict(collect_gpu_metadata(i)) for i in range(args.tensor_parallel_size)]
        load_start = perf_counter_ns()
        llm = LLM(args.model, **options)
        report["model_load_ms"] = (perf_counter_ns() - load_start) / 1e6
        storage = kv_cache_storage_bytes(llm.model_runner.kv_cache, llm.model_runner.kv_scale_cache)
        report["kv_cache"] = {
            "rank0_payload_shape": list(llm.model_runner.kv_cache.shape),
            "payload_dtype": str(llm.model_runner.kv_cache.dtype),
            "rank0_payload_bytes": storage.payload,
            "rank0_scale_bytes": storage.scales,
            "bytes_per_rank": storage.total,
            "total_bytes_equal_shards": storage.total * args.tensor_parallel_size,
            "capacity_tokens": llm.model_runner.num_kvcache_blocks * args.kvcache_block_size,
        }
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

        def run(phase: str, prompts: list[str], prefix: bool) -> list[dict[str, Any]]:
            return _run_requests(
                llm,
                images,
                prompts,
                sampling,
                phase=phase,
                prefix_enabled=prefix,
                output_rows=report["requests"],
                replica_id=args.replica_id,
            )

        for _ in range(args.warmup):
            run("warmup", [args.prompt], False)
        cold = []
        for _ in range(args.repeat):
            cold.extend(run("cold_prefix_disabled", [args.prompt], False))
        run("prefix_prime", [args.prompt], True)
        for _ in range(args.repeat):
            run("hot_same_prompt", [args.prompt], True)
        for _ in range(args.repeat):
            run("hot_new_question", [args.hot_prompt], True)
        _partial_prefix_case(llm, sampling, report, args.replica_id)

        for phase, prefix_enabled in (
            ("concurrent_cold_prefix_disabled", False),
            ("concurrent_hot", True),
        ):
            if args.wait_for_start:
                print(
                    "VISION_PARALLEL_READY "
                    + json.dumps(
                        {
                            "replica_id": args.replica_id,
                            "phase": phase,
                        }
                    ),
                    flush=True,
                )
                if sys.stdin.readline().strip() != "START":
                    raise RuntimeError(f"expected START on stdin before {phase}")
            run(phase, [args.prompt] * args.concurrent_requests, prefix_enabled)

        reference_ids = cold[0]["token_ids"]
        for row in report["requests"]:
            if row["prompt"] == args.prompt:
                row["matches_sequential_cold_ids"] = row["token_ids"] == reference_ids
        for phase in dict.fromkeys(row["phase"] for row in report["requests"]):
            phase_rows = [row for row in report["requests"] if row["phase"] == phase]
            report["phase_summaries"][phase] = _summary(phase_rows)
        report["prefix_cache_after"] = llm.multimodal_prefix_cache_metadata()
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            if llm is not None:
                llm.exit()
        finally:
            for image in images:
                image.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
