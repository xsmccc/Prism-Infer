"""P6 统一 Prism-Infer internal system benchmark。

runner 在同一个 deterministic workload manifest 上比较正交 internal mode，
测量 preprocessing、engine TTFT、prefill step、decode-step ITL、端到端延迟、
吞吐、GPU 显存和物理 KV cache bytes；不实现外部框架 adapter 或 online serving claim。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
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
    describe_case_inputs,
    expand_case_batch,
    find_workload_case,
    materialize_requests,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import (
    BENCHMARK_SCHEMA_VERSION,
    canonical_json_sha256,
    load_workload_manifest,
    summarize_values,
    validate_benchmark_record,
)
from prism_infer.analysis.performance_profile import (
    performance_profile,
    validate_performance_profile_record,
)
from prism_infer.engine.kv_quantization import (
    kv_cache_storage_bytes,
    tensor_storage_bytes,
)


DEFAULT_MANIFEST = Path(__file__).with_name("workloads") / "p6_internal_smoke.json"


@dataclass(frozen=True)
class ModeSpec:
    """一个受支持的 internal execution/compression 组合。"""

    name: str
    execution: str
    attention: str
    compression: str
    enforce_eager: bool
    decode_compile_region: str = "none"


MODE_SPECS = {
    "off_eager": ModeSpec(
        name="off_eager",
        execution="eager",
        attention="prefill_sdpa_decode_paged",
        compression="off",
        enforce_eager=True,
    ),
    "off_graph": ModeSpec(
        name="off_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_paged",
        compression="off",
        enforce_eager=False,
    ),
    "off_compile_attention": ModeSpec(
        name="off_compile_attention",
        execution="torch_compile_attention",
        attention="prefill_sdpa_decode_compiled_qkv_paged",
        compression="off",
        enforce_eager=True,
        decode_compile_region="attention",
    ),
    "visual_prune": ModeSpec(
        name="visual_prune",
        execution="eager",
        attention="prefill_sdpa_decode_retained_sdpa",
        compression="visual_prune",
        enforce_eager=True,
    ),
    "visual_compact": ModeSpec(
        name="visual_compact",
        execution="eager",
        attention="prefill_sdpa_decode_compact_paged",
        compression="visual_compact",
        enforce_eager=True,
    ),
    "visual_compact_graph": ModeSpec(
        name="visual_compact_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_compact_paged",
        compression="visual_compact",
        enforce_eager=False,
    ),
    "fp8_kv": ModeSpec(
        name="fp8_kv",
        execution="eager",
        attention="prefill_sdpa_decode_fp8_paged_triton",
        compression="fp8_kv",
        enforce_eager=True,
    ),
    "fp8_kv_graph": ModeSpec(
        name="fp8_kv_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_fp8_paged_triton",
        compression="fp8_kv",
        enforce_eager=False,
    ),
    "scaled_fp8_kv": ModeSpec(
        name="scaled_fp8_kv",
        execution="eager",
        attention="prefill_sdpa_decode_scaled_fp8_paged_triton",
        compression="scaled_fp8_kv",
        enforce_eager=True,
    ),
    "scaled_fp8_kv_graph": ModeSpec(
        name="scaled_fp8_kv_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_scaled_fp8_paged_triton",
        compression="scaled_fp8_kv",
        enforce_eager=False,
    ),
    "visual_compact_fp8": ModeSpec(
        name="visual_compact_fp8",
        execution="eager",
        attention="prefill_sdpa_decode_compact_fp8_paged_triton",
        compression="visual_compact_fp8",
        enforce_eager=True,
    ),
    "visual_compact_fp8_graph": ModeSpec(
        name="visual_compact_fp8_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_compact_fp8_paged_triton",
        compression="visual_compact_fp8",
        enforce_eager=False,
    ),
    "visual_compact_scaled_fp8": ModeSpec(
        name="visual_compact_scaled_fp8",
        execution="eager",
        attention="prefill_sdpa_decode_compact_scaled_fp8_paged_triton",
        compression="visual_compact_scaled_fp8",
        enforce_eager=True,
    ),
    "visual_compact_scaled_fp8_graph": ModeSpec(
        name="visual_compact_scaled_fp8_graph",
        execution="cuda_graph",
        attention="prefill_sdpa_decode_compact_scaled_fp8_paged_triton",
        compression="visual_compact_scaled_fp8",
        enforce_eager=False,
    ),
}


@dataclass
class IterationResult:
    """一次 deterministic workload 执行的原始测量。"""

    token_ids: list[list[int]]
    decoded_texts: list[str]
    preprocessing_ms: float
    ttft_ms: float
    prefill_step_ms: list[float]
    decode_step_ms: list[float]
    end_to_end_ms: float
    engine_output_tokens_per_s: float
    e2e_output_tokens_per_s: float
    decode_tokens_per_s: float
    engine_requests_per_s: float
    e2e_requests_per_s: float
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    prompt_tokens: int
    image_tokens: int
    video_tokens: int
    logical_prompt_kv_tokens: int
    physical_prompt_kv_tokens: int
    dense_prompt_blocks: int
    active_prompt_blocks: int
    released_prompt_blocks: int
    dense_prompt_payload_bytes: int
    dense_prompt_scale_bytes: int
    dense_prompt_bytes: int
    active_prompt_payload_bytes: int
    active_prompt_scale_bytes: int
    active_prompt_bytes: int
    kv_layouts: list[dict[str, Any]]


def _mb(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def _add_requests(
    llm: LLM,
    requests: list[dict[str, Any]],
    sampling: SamplingParams,
) -> tuple[list[int], tuple[int, int, int]]:
    """添加公开请求，并返回 ID 与 prompt/image/video token 计数。"""

    seq_ids: list[int] = []
    prompt_tokens = 0
    image_tokens = 0
    video_tokens = 0
    for request in requests:
        request_type = request["type"]
        if request_type == "text":
            seq_id = llm.add_request(request["prompt"], sampling)
        elif request_type == "image":
            seq_id = llm.add_vl_request(
                request["prompt"],
                request["image"],
                sampling,
            )
        elif request_type == "images":
            seq_id = llm.add_images_request(
                request["prompt"],
                request["images"],
                sampling,
            )
        elif request_type == "video":
            seq_id = llm.add_video_request(
                request["prompt"],
                request["video"],
                sampling,
            )
        else:
            raise ValueError(f"unsupported request type: {request_type!r}")

        seq = llm.scheduler.waiting[-1]
        if seq.seq_id != seq_id:
            raise RuntimeError("scheduler waiting order changed while adding benchmark requests")
        seq_ids.append(seq_id)
        prompt_tokens += seq.num_prompt_tokens
        image_tokens += seq.image_token_count
        video_tokens += seq.video_token_count
    return seq_ids, (prompt_tokens, image_tokens, video_tokens)


def _run_iteration(
    llm: LLM,
    case: dict[str, Any],
    sampling: SamplingParams,
) -> IterationResult:
    """运行一次 workload 并测量公开 engine step，不改变推理语义。"""

    requests = materialize_requests(case, repo_root=REPO_ROOT)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    preprocessing_start = perf_counter()
    seq_ids, token_counts = _add_requests(llm, requests, sampling)
    preprocessing_ms = (perf_counter() - preprocessing_start) * 1000.0

    outputs: dict[int, list[int]] = {}
    prefill_step_ms: list[float] = []
    decode_step_ms: list[float] = []
    decode_tokens = 0
    prompt_kv_snapshot: dict[str, Any] | None = None
    while not llm.is_finished():
        torch.cuda.synchronize()
        step_start = perf_counter()
        finished, num_tokens = llm.step()
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - step_start) * 1000.0
        if num_tokens > 0:
            prefill_step_ms.append(elapsed_ms)
            active_sequences = {
                seq.seq_id: seq for seq in llm.scheduler.running if seq.seq_id in seq_ids
            }
            if set(active_sequences) != set(seq_ids):
                raise RuntimeError("benchmark could not capture all post-prefill KV layouts")
            ordered_sequences = [active_sequences[seq_id] for seq_id in seq_ids]
            logical_prompt_kv_tokens = sum(seq.num_prompt_tokens for seq in ordered_sequences)
            physical_prompt_kv_tokens = sum(
                (
                    seq.num_prompt_tokens
                    if seq.kv_layout is None
                    else seq.kv_layout.compressed_prompt_kv_len
                )
                for seq in ordered_sequences
            )
            dense_prompt_blocks = sum(
                (seq.num_prompt_tokens + seq.block_size - 1) // seq.block_size
                for seq in ordered_sequences
            )
            active_prompt_blocks = sum(len(seq.block_table) for seq in ordered_sequences)
            payload_block_bytes = tensor_storage_bytes(llm.model_runner.kv_cache[:, :, 0])
            scale_cache = llm.model_runner.kv_scale_cache
            scale_block_bytes = (
                0 if scale_cache is None else tensor_storage_bytes(scale_cache[:, :, 0])
            )
            block_bytes = payload_block_bytes + scale_block_bytes
            layouts = []
            for seq in ordered_sequences:
                if seq.kv_layout is None:
                    layouts.append(
                        {
                            "schema_version": 1,
                            "mode": "dense",
                            "logical_context_len": seq.num_prompt_tokens,
                            "physical_kv_len": seq.num_prompt_tokens,
                            "prompt_logical_len": seq.num_prompt_tokens,
                            "compressed_prompt_kv_len": seq.num_prompt_tokens,
                            "retained_original_positions": [],
                            "block_table": list(seq.block_table),
                            "kv_dtype": str(llm.model_runner.kv_cache_dtype),
                            "compression_record": {},
                        }
                    )
                else:
                    layouts.append(seq.kv_layout.to_record(block_table=seq.block_table))
            prompt_kv_snapshot = {
                "logical_prompt_kv_tokens": logical_prompt_kv_tokens,
                "physical_prompt_kv_tokens": physical_prompt_kv_tokens,
                "dense_prompt_blocks": dense_prompt_blocks,
                "active_prompt_blocks": active_prompt_blocks,
                "released_prompt_blocks": (dense_prompt_blocks - active_prompt_blocks),
                "dense_prompt_payload_bytes": (dense_prompt_blocks * payload_block_bytes),
                "dense_prompt_scale_bytes": (dense_prompt_blocks * scale_block_bytes),
                "dense_prompt_bytes": dense_prompt_blocks * block_bytes,
                "active_prompt_payload_bytes": (active_prompt_blocks * payload_block_bytes),
                "active_prompt_scale_bytes": (active_prompt_blocks * scale_block_bytes),
                "active_prompt_bytes": active_prompt_blocks * block_bytes,
                "kv_layouts": layouts,
            }
        else:
            decode_step_ms.append(elapsed_ms)
            decode_tokens += -num_tokens
        for seq_id, token_ids in finished:
            outputs[seq_id] = list(token_ids)

    if not prefill_step_ms:
        raise RuntimeError("benchmark workload completed without a prefill step")
    if prompt_kv_snapshot is None:
        raise RuntimeError("benchmark did not capture a prompt KV snapshot")
    if not decode_step_ms or decode_tokens <= 0:
        raise RuntimeError("benchmark requires max_tokens >= 2 so decode TPOT can be measured")
    if set(outputs) != set(seq_ids):
        raise RuntimeError(
            f"finished outputs mismatch: expected={seq_ids}, actual={sorted(outputs)}"
        )

    ordered_outputs = [outputs[seq_id] for seq_id in seq_ids]
    # 解码不计入 engine/E2E timing，只为 schema-v5+ reference task 评估留证。
    decoded_texts = [
        llm.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_ids in ordered_outputs
    ]
    engine_ms = sum(prefill_step_ms) + sum(decode_step_ms)
    end_to_end_ms = preprocessing_ms + engine_ms
    output_tokens = sum(len(tokens) for tokens in ordered_outputs)
    num_requests = len(ordered_outputs)
    decode_ms = sum(decode_step_ms)
    prompt_tokens, image_tokens, video_tokens = token_counts
    return IterationResult(
        token_ids=ordered_outputs,
        decoded_texts=decoded_texts,
        preprocessing_ms=preprocessing_ms,
        ttft_ms=sum(prefill_step_ms),
        prefill_step_ms=prefill_step_ms,
        decode_step_ms=decode_step_ms,
        end_to_end_ms=end_to_end_ms,
        engine_output_tokens_per_s=output_tokens / (engine_ms / 1000.0),
        e2e_output_tokens_per_s=output_tokens / (end_to_end_ms / 1000.0),
        decode_tokens_per_s=decode_tokens / (decode_ms / 1000.0),
        engine_requests_per_s=num_requests / (engine_ms / 1000.0),
        e2e_requests_per_s=num_requests / (end_to_end_ms / 1000.0),
        allocated_mb=_mb(torch.cuda.memory_allocated()),
        reserved_mb=_mb(torch.cuda.memory_reserved()),
        peak_allocated_mb=_mb(torch.cuda.max_memory_allocated()),
        prompt_tokens=prompt_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
        **prompt_kv_snapshot,
    )


def _build_llm(
    args: argparse.Namespace,
    mode: ModeSpec,
    *,
    max_num_seqs: int,
) -> LLM:
    """构造一个 internal mode，并保持所有非 mode 设置不变。"""

    return LLM(
        args.model,
        enforce_eager=mode.enforce_eager,
        decode_compile_region=mode.decode_compile_region,
        decode_compile_mode=args.decode_compile_mode,
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
        allow_unsafe_decode_compile=(mode.decode_compile_region != "none"),
        compression_mode=mode.compression,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        enable_chunked_prefill=False,
        enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
        visual_pruning_keep_ratio=args.visual_pruning_keep_ratio,
        visual_pruning_min_keep_tokens=args.visual_pruning_min_keep_tokens,
        visual_pruning_strategy=args.visual_pruning_strategy,
        visual_pruning_attention_last_n_layers=(args.visual_pruning_attention_last_n_layers),
        logits_precision=args.logits_precision,
        mlp_projection_mode=args.mlp_projection_mode,
    )


def _build_record(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case: dict[str, Any],
    mode: ModeSpec,
    llm: LLM,
    results: list[IterationResult],
    source_num_requests: int,
    request_replication_factor: int,
) -> dict[str, Any]:
    """把多次测量汇总为一条通过校验的 JSONL 记录。"""

    first = results[0]
    outputs_identical = all(result.token_ids == first.token_ids for result in results)
    decoded_texts_identical = all(result.decoded_texts == first.decoded_texts for result in results)
    if not outputs_identical or not decoded_texts_identical:
        raise RuntimeError(f"deterministic greedy outputs changed across {len(results)} repeats")
    kv_metric_names = (
        "logical_prompt_kv_tokens",
        "physical_prompt_kv_tokens",
        "dense_prompt_blocks",
        "active_prompt_blocks",
        "released_prompt_blocks",
        "dense_prompt_payload_bytes",
        "dense_prompt_scale_bytes",
        "dense_prompt_bytes",
        "active_prompt_payload_bytes",
        "active_prompt_scale_bytes",
        "active_prompt_bytes",
    )
    for metric_name in kv_metric_names:
        if any(
            getattr(result, metric_name) != getattr(first, metric_name) for result in results[1:]
        ):
            raise RuntimeError(f"prompt KV metric changed across repeats: {metric_name}")
    git = collect_git_metadata(REPO_ROOT)
    gpu_metadata = collect_gpu_metadata().environment_dict()
    kv_cache = llm.model_runner.kv_cache
    kv_scale_cache = llm.model_runner.kv_scale_cache
    storage_bytes = kv_cache_storage_bytes(kv_cache, kv_scale_cache)
    config = llm.config
    model_config_path = Path(args.model) / "config.json"
    request_types = [request["type"] for request in case["requests"]]
    input_shapes, image_count, video_count, video_frame_count = describe_case_inputs(case)
    output_tokens = sum(len(tokens) for tokens in first.token_ids)
    graph_metadata = llm.model_runner.cudagraph_metadata(len(case["requests"]))
    compile_metadata = llm.model_runner.compile_metadata()
    if graph_metadata["enabled"]:
        if (
            llm.model_runner.last_cudagraph_actual_batch_size
            != graph_metadata["requested_batch_size"]
            or llm.model_runner.last_cudagraph_replay_batch_size
            != graph_metadata["selected_batch_size"]
        ):
            raise RuntimeError("observed CUDA Graph replay batch does not match recorded metadata")
    record: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "record_type": "system_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "p7.1_prism_offline_v2",
            "process_scope": "fresh_model_per_case_and_mode",
        },
        "environment": {
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
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
            "dtype": str(llm.model_runner.model_dtype),
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_num_seqs": config.max_num_seqs,
            "kvcache_block_size": config.kvcache_block_size,
            "num_kvcache_blocks": config.num_kvcache_blocks,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "prefix_caching_enabled": config.enable_prefix_caching,
            "chunked_prefill_enabled": config.enable_chunked_prefill,
            "logits_precision": config.logits_precision,
            "mlp_projection_mode": config.mlp_projection_mode,
        },
        "mode": {
            "name": mode.name,
            "execution": mode.execution,
            "attention": mode.attention,
            "compression": mode.compression,
            "visual_pruning_keep_ratio": args.visual_pruning_keep_ratio,
            "visual_pruning_min_keep_tokens": args.visual_pruning_min_keep_tokens,
            "visual_pruning_strategy": args.visual_pruning_strategy,
            "visual_pruning_attention_last_n_layers": (args.visual_pruning_attention_last_n_layers),
        },
        "workload": {
            "manifest_name": manifest["name"],
            "manifest_sha256": canonical_json_sha256(manifest),
            "case_id": case["id"],
            "request_types": request_types,
            "input_shapes": input_shapes,
            "num_requests": len(case["requests"]),
            "source_num_requests": source_num_requests,
            "request_replication_factor": request_replication_factor,
            "prompt_tokens": first.prompt_tokens,
            "image_tokens": first.image_tokens,
            "video_tokens": first.video_tokens,
            "image_count": image_count,
            "video_count": video_count,
            "video_frame_count": video_frame_count,
            "max_tokens": args.max_tokens,
            "preprocessing_included_in_e2e": True,
            "output_decoding_included_in_e2e": False,
            "reference_sources": deepcopy(manifest.get("reference_sources", {})),
            "task_references": [
                deepcopy(request.get("evaluation")) for request in case["requests"]
            ],
        },
        "traffic": {
            "kind": "offline_closed_loop",
            "batch_size": len(case["requests"]),
            "concurrency": len(case["requests"]),
            "request_rate_per_s": None,
        },
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": args.max_tokens,
        },
        "execution_backend": {
            "prefill_backend": "eager",
            "decode_backend": mode.execution,
            "cuda_graph_enabled": graph_metadata["enabled"],
            "cuda_graph_capture_scope": graph_metadata["capture_scope"],
            "cuda_graph_capture_ms": graph_metadata["capture_ms"],
            "cuda_graph_batch_sizes": graph_metadata["batch_sizes"],
            "requested_decode_batch_size": graph_metadata["requested_batch_size"],
            "selected_decode_batch_size": graph_metadata["selected_batch_size"],
            "decode_batch_padding": graph_metadata["batch_padding"],
            "torch_compile_enabled": compile_metadata["enabled"],
            "torch_compile_region": compile_metadata["region"],
            "torch_compile_subgraph": compile_metadata["subgraph"],
            "torch_compile_kv_cache_boundary": compile_metadata["kv_cache_boundary"],
            "torch_compile_backend": compile_metadata["backend"],
            "torch_compile_mode": compile_metadata["mode"],
            "torch_compile_emulate_precision_casts": compile_metadata["emulate_precision_casts"],
            "torch_compile_force_same_precision": compile_metadata["force_same_precision"],
            "torch_compile_first_call_ms": compile_metadata["first_call_ms"],
        },
        "measurement": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cuda_synchronize_timing": True,
            "engine_ttft_scope": "sum_of_synchronized_prefill_steps",
            "decode_tpot_scope": "synchronized_engine_decode_step",
            "end_to_end_scope": "request_preprocessing_plus_engine_steps",
        },
        "correctness": {
            "outputs_identical_across_repeats": outputs_identical,
            "token_ids": first.token_ids,
            "output_tokens": output_tokens,
            "output_sha256": canonical_json_sha256(first.token_ids),
            "decoded_texts": first.decoded_texts,
            "decoded_texts_sha256": canonical_json_sha256(first.decoded_texts),
        },
        "timing_ms": {
            "preprocessing": summarize_values([result.preprocessing_ms for result in results]),
            "engine_ttft": summarize_values([result.ttft_ms for result in results]),
            "end_to_end_ttft": summarize_values(
                [result.preprocessing_ms + result.ttft_ms for result in results]
            ),
            "prefill": summarize_values(
                [step_ms for result in results for step_ms in result.prefill_step_ms]
            ),
            "decode_step": summarize_values(
                [step_ms for result in results for step_ms in result.decode_step_ms]
            ),
            "end_to_end": summarize_values([result.end_to_end_ms for result in results]),
        },
        "throughput": {
            "engine_output_tokens_per_s": summarize_values(
                [result.engine_output_tokens_per_s for result in results]
            ),
            "e2e_output_tokens_per_s": summarize_values(
                [result.e2e_output_tokens_per_s for result in results]
            ),
            "decode_tokens_per_s": summarize_values(
                [result.decode_tokens_per_s for result in results]
            ),
            "engine_requests_per_s": summarize_values(
                [result.engine_requests_per_s for result in results]
            ),
            "e2e_requests_per_s": summarize_values(
                [result.e2e_requests_per_s for result in results]
            ),
        },
        "memory_mb": {
            "measurement": "torch_cuda_allocator",
            "allocated": summarize_values([result.allocated_mb for result in results]),
            "reserved": summarize_values([result.reserved_mb for result in results]),
            "peak_allocated": summarize_values([result.peak_allocated_mb for result in results]),
        },
        "kv_cache": {
            "dtype": str(kv_cache.dtype),
            "shape": list(kv_cache.shape),
            "scale_dtype": ("none" if kv_scale_cache is None else str(kv_scale_cache.dtype)),
            "scale_shape": ([] if kv_scale_cache is None else list(kv_scale_cache.shape)),
            "payload_bytes": storage_bytes.payload,
            "scale_bytes": storage_bytes.scales,
            "bytes": storage_bytes.total,
            "blocks": kv_cache.shape[2],
            "block_size": config.kvcache_block_size,
            "capacity_tokens": kv_cache.shape[2] * config.kvcache_block_size,
            "logical_prompt_tokens": first.logical_prompt_kv_tokens,
            "physical_prompt_tokens": first.physical_prompt_kv_tokens,
            "dense_prompt_blocks": first.dense_prompt_blocks,
            "active_prompt_blocks": first.active_prompt_blocks,
            "released_prompt_blocks": first.released_prompt_blocks,
            "dense_prompt_payload_bytes": first.dense_prompt_payload_bytes,
            "dense_prompt_scale_bytes": first.dense_prompt_scale_bytes,
            "dense_prompt_bytes": first.dense_prompt_bytes,
            "active_prompt_payload_bytes": first.active_prompt_payload_bytes,
            "active_prompt_scale_bytes": first.active_prompt_scale_bytes,
            "active_prompt_bytes": first.active_prompt_bytes,
            "layouts": first.kv_layouts,
        },
    }
    validate_benchmark_record(record)
    return record


def _bench_mode(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case: dict[str, Any],
    mode: ModeSpec,
    source_num_requests: int,
    request_replication_factor: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """运行基准和可选的独立 profiling iterations。"""

    llm = _build_llm(
        args,
        mode,
        max_num_seqs=_resolve_engine_max_num_seqs(
            getattr(args, "max_num_seqs", None),
            len(case["requests"]),
        ),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    try:
        for _ in range(args.warmup):
            _run_iteration(llm, case, sampling)
        results = [_run_iteration(llm, case, sampling) for _ in range(args.repeat)]
        benchmark_record = _build_record(
            args=args,
            manifest=manifest,
            case=case,
            mode=mode,
            llm=llm,
            results=results,
            source_num_requests=source_num_requests,
            request_replication_factor=request_replication_factor,
        )
        profile_record = None
        if args.profile_output is not None:
            git = collect_git_metadata(REPO_ROOT)
            with performance_profile(
                metadata={
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": git.commit,
                    "git_dirty": git.dirty,
                    "gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda,
                    "torch": torch.__version__,
                    "model": str(Path(args.model).resolve()),
                    "mode": mode.name,
                    "execution": mode.execution,
                    "attention": mode.attention,
                    "compression": mode.compression,
                    "mlp_projection_mode": args.mlp_projection_mode,
                    "case_id": case["id"],
                    "profile_repeat": args.profile_repeat,
                    "warmup_before_profile": args.warmup,
                    "max_tokens": args.max_tokens,
                    "batch_size": len(case["requests"]),
                    "source_num_requests": source_num_requests,
                    "request_replication_factor": request_replication_factor,
                    "execution_backend": llm.model_runner.cudagraph_metadata(len(case["requests"])),
                    "max_model_len": args.max_model_len,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "num_kvcache_blocks": args.num_kvcache_blocks,
                    "kvcache_block_size": args.kvcache_block_size,
                },
                cuda_timing=True,
            ) as profile_session:
                if args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStart()
                try:
                    profile_results = [
                        _run_iteration(llm, case, sampling) for _ in range(args.profile_repeat)
                    ]
                finally:
                    if args.cuda_profiler_range:
                        torch.cuda.cudart().cudaProfilerStop()
            baseline_tokens = results[0].token_ids
            if not all(result.token_ids == baseline_tokens for result in profile_results):
                raise RuntimeError(
                    f"profiled {mode.name} output differs from its benchmark baseline"
                )
            profile_session.metadata["correctness"] = {
                "token_exact_to_unprofiled": True,
                "output_sha256": canonical_json_sha256(baseline_tokens),
            }
            profile_record = profile_session.to_record()
            validate_performance_profile_record(profile_record)
        return benchmark_record, profile_record
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()


def _annotate_comparisons(records: list[dict[str, Any]]) -> None:
    """在每个 workload/batch/output/keep-ratio cell 内增加首 mode 对比。"""

    if not records:
        return
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for record in records:
        workload = record["workload"]
        key = (
            workload["manifest_sha256"],
            workload["case_id"],
            workload["num_requests"],
            workload["max_tokens"],
            record["mode"]["visual_pruning_keep_ratio"],
        )
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        baseline = group[0]
        baseline_tokens = baseline["correctness"]["token_ids"]
        baseline_kv_bytes = baseline["kv_cache"].get(
            "active_prompt_bytes",
            baseline["kv_cache"]["bytes"],
        )
        for record in group:
            tokens = record["correctness"]["token_ids"]
            record["comparison_to_first_mode"] = {
                "first_mode": baseline["mode"]["name"],
                "token_exact": tokens == baseline_tokens,
                "kv_byte_ratio": record["kv_cache"].get(
                    "active_prompt_bytes",
                    record["kv_cache"]["bytes"],
                )
                / baseline_kv_bytes,
            }


def _parse_modes(value: str) -> list[ModeSpec]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("at least one benchmark mode is required")
    unknown = [name for name in names if name not in MODE_SPECS]
    if unknown:
        raise ValueError(f"unsupported modes {unknown}; supported modes: {sorted(MODE_SPECS)}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate modes are not allowed: {names}")
    return [MODE_SPECS[name] for name in names]


def _parse_positive_ints(value: str, *, label: str, minimum: int = 1) -> list[int]:
    """解析有序、去重的逗号分隔正整数 matrix 轴。"""

    raw_values = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_values:
        raise ValueError(f"{label} requires at least one value")
    try:
        values = [int(item) for item in raw_values]
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers: {raw_values}") from exc
    if any(value < minimum for value in values):
        raise ValueError(f"{label} values must be >= {minimum}: {values}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} duplicate values are not allowed: {values}")
    return values


def _parse_keep_ratios(value: str) -> list[float]:
    """解析有序、去重且位于 (0, 1] 的 visual keep-ratio matrix。"""

    raw_values = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_values:
        raise ValueError("--visual-pruning-keep-ratios requires at least one value")
    try:
        values = [float(item) for item in raw_values]
    except ValueError as exc:
        raise ValueError(
            f"--visual-pruning-keep-ratios must contain numbers: {raw_values}"
        ) from exc
    if any(not 0.0 < ratio <= 1.0 for ratio in values):
        raise ValueError(f"--visual-pruning-keep-ratios values must be in (0, 1]: {values}")
    if len(values) != len(set(values)):
        raise ValueError(f"--visual-pruning-keep-ratios duplicate values are not allowed: {values}")
    return values


def _resolve_engine_max_num_seqs(
    configured: int | None,
    actual_batch_size: int,
) -> int:
    """解耦 workload batch 与预录制 Graph 的 engine batch 上限。"""

    if actual_batch_size <= 0:
        raise ValueError("actual batch size must be positive")
    if configured is None:
        return actual_batch_size
    if configured < actual_batch_size:
        raise ValueError(
            "--max-num-seqs must cover every requested batch: "
            f"configured={configured}, actual={actual_batch_size}"
        )
    return configured


def _write_records(records: list[dict[str, Any]], output: str | None) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} records to {output_path}", file=sys.stderr)
    for line in lines:
        print(line)


def _write_profile_records(records: list[dict[str, Any]], output: str) -> None:
    """校验并写出 profiling JSONL，不把大体积 raw regions 打到 stdout。"""

    for record in records:
        validate_performance_profile_record(record)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} profile records to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case", default="single_image_448")
    parser.add_argument(
        "--modes",
        default="off_eager,off_graph,visual_prune,fp8_kv",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--output-lengths",
        help="comma-separated max-token matrix; overrides --max-tokens",
    )
    parser.add_argument(
        "--batch-sizes",
        help="comma-separated offline batch matrix; defaults to source case size",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help=("engine/CUDA Graph capture batch ceiling; defaults to each cell's actual batch size"),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="disable full-block prefix reuse for mixed-VL fidelity batches",
    )
    parser.set_defaults(enable_prefix_caching=True)
    parser.add_argument(
        "--decode-compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--visual-pruning-keep-ratio", type=float, default=0.5)
    parser.add_argument(
        "--visual-pruning-keep-ratios",
        help="comma-separated keep-ratio matrix; overrides --visual-pruning-keep-ratio",
    )
    parser.add_argument("--visual-pruning-min-keep-tokens", type=int, default=32)
    parser.add_argument(
        "--visual-pruning-strategy",
        choices=("uniform", "attention"),
        default="uniform",
    )
    parser.add_argument(
        "--visual-pruning-attention-last-n-layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--logits-precision",
        choices=("fp32", "model"),
        default="model",
        help="lm_head projection precision; model uses the loaded model dtype",
    )
    parser.add_argument(
        "--mlp-projection-mode",
        choices=("legacy", "packed"),
        default="packed",
        help="execute gate/up as two legacy projections or one packed projection",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--profile-output",
        help="write separate semantic CPU/CUDA profile JSONL",
    )
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="mark profiled iterations with cudaProfilerStart/Stop for Nsight",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the P6 system benchmark")
    if args.warmup < 0 or args.repeat < 1:
        raise SystemExit("--warmup must be >= 0 and --repeat must be >= 1")
    if args.profile_repeat < 1:
        raise SystemExit("--profile-repeat must be >= 1")
    if args.cuda_profiler_range and args.profile_output is None:
        raise SystemExit("--cuda-profiler-range requires --profile-output")

    manifest = load_workload_manifest(args.manifest)
    source_case = find_workload_case(manifest, args.case)
    modes = _parse_modes(args.modes)
    output_lengths = (
        _parse_positive_ints(
            args.output_lengths,
            label="--output-lengths",
            minimum=2,
        )
        if args.output_lengths is not None
        else [args.max_tokens]
    )
    if any(output_length < 2 for output_length in output_lengths):
        raise SystemExit("output lengths must be >= 2 to measure decode TPOT")
    source_batch_size = len(source_case["requests"])
    batch_sizes = (
        _parse_positive_ints(args.batch_sizes, label="--batch-sizes")
        if args.batch_sizes is not None
        else [source_batch_size]
    )
    if args.max_num_seqs is not None and args.max_num_seqs < max(batch_sizes):
        raise SystemExit("--max-num-seqs must be >= the largest requested --batch-sizes value")
    keep_ratios = (
        _parse_keep_ratios(args.visual_pruning_keep_ratios)
        if args.visual_pruning_keep_ratios is not None
        else [args.visual_pruning_keep_ratio]
    )
    records = []
    profile_records = []
    for batch_size in batch_sizes:
        case, source_num_requests, replication_factor = expand_case_batch(
            source_case,
            batch_size,
        )
        for output_length in output_lengths:
            for keep_ratio in keep_ratios:
                cell_args = argparse.Namespace(**vars(args))
                cell_args.max_tokens = output_length
                cell_args.visual_pruning_keep_ratio = keep_ratio
                for mode in modes:
                    print(
                        f"running case={args.case} batch={batch_size} "
                        f"max_tokens={output_length} keep_ratio={keep_ratio} "
                        f"mode={mode.name} "
                        f"warmup={args.warmup} repeat={args.repeat}",
                        file=sys.stderr,
                    )
                    benchmark_record, profile_record = _bench_mode(
                        args=cell_args,
                        manifest=manifest,
                        case=case,
                        mode=mode,
                        source_num_requests=source_num_requests,
                        request_replication_factor=replication_factor,
                    )
                    records.append(benchmark_record)
                    if profile_record is not None:
                        profile_records.append(profile_record)
    _annotate_comparisons(records)
    _write_records(records, args.output)
    if args.profile_output is not None:
        _write_profile_records(profile_records, args.profile_output)


if __name__ == "__main__":
    main()
