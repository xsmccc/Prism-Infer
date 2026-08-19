"""Measure single-node online arrival and continuous batching."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread

import torch
import transformers

try:
    import pynvml
except ImportError:  # pragma: no cover - optional process-memory telemetry
    pynvml = None


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
from benchmarks.multimodal_cache_workload import (
    build_multimodal_cache_workload,
)
from benchmarks.working_set_workload import (
    MaterializedWorkingSet,
    materialize_working_set,
    source_prompt_schedule_sha256,
    verify_working_set_model,
    verify_working_set_processor,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.analysis.identity import canonical_json_sha256, sha256_file
from prism_infer.analysis.online_serving import (
    ONLINE_BENCHMARK_SCHEMA_VERSION,
    summarize_online_run,
    validate_online_benchmark_record,
)
from prism_infer.analysis.performance_profile import (
    performance_profile,
    validate_performance_profile_record,
)
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes
from prism_infer.engine.online import OnlineRequest, OnlineServingSession

ONLINE_MODES = (
    "off_eager",
    "off_graph",
    "visual_compact_graph",
    "bf16_compile_graph",
    "scaled_fp8_kv_compile_graph",
    "visual_compact_scaled_fp8_compile_graph",
)


def _write_text_atomic(path: Path, text: str) -> None:
    """Create one evidence file atomically without replacing an earlier run."""

    if path.exists():
        raise FileExistsError(f"refusing to replace existing benchmark output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _DeviceProcessMemorySampler:
    """Sample total compute-process memory on the benchmark's dedicated GPU."""

    def __init__(self, *, device_index: int = 0, interval_ms: float = 10.0) -> None:
        if pynvml is None:
            raise RuntimeError("NVML process-memory sampling requires nvidia-ml-py")
        self.device_index = device_index
        self.interval_ms = interval_ms
        self._stop = Event()
        self._thread: Thread | None = None
        self._handle = None
        self._samples = 0
        self._initial_bytes = 0
        self._peak_bytes = 0
        self._final_bytes = 0
        self._pids: set[int] = set()
        self._failure: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def _sample(self) -> int:
        used_bytes = 0
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
            value = process.usedGpuMemory
            if isinstance(value, int) and 0 <= value < (1 << 63):
                used_bytes += value
                self._pids.add(int(process.pid))
        self._samples += 1
        self._peak_bytes = max(self._peak_bytes, used_bytes)
        return used_bytes

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000.0):
            try:
                self._sample()
            except BaseException as exc:
                self._failure = exc
                self._stop.set()

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._initial_bytes = self._sample()
        self._thread = Thread(
            target=self._run,
            name="prism-device-process-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("NVML process-memory sampler was not started")
        self._stop.set()
        self._thread.join()
        try:
            self._final_bytes = self._sample()
        finally:
            pynvml.nvmlShutdown()
            self._thread = None
        if self._failure is not None:
            raise RuntimeError("NVML process-memory sampling failed") from self._failure
        mib = 1024 * 1024
        return {
            "measurement": "NVML total compute-process usedGpuMemory",
            "scope": "post-LLM-init through warmup and measured generation",
            "dedicated_gpu_required": True,
            "device_index": self.device_index,
            "sampling_interval_ms": self.interval_ms,
            "samples": self._samples,
            "observed_pids": sorted(self._pids),
            "after_llm_init_mib": self._initial_bytes / mib,
            "peak_serving_mib": self._peak_bytes / mib,
            "after_benchmark_mib": self._final_bytes / mib,
        }


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


def _planned_online_requests(
    payloads: list[dict],
    *,
    request_ids: list[str],
    offsets_s: list[float],
    sampling: SamplingParams,
) -> tuple[OnlineRequest, ...]:
    """Build requests from the framework-neutral plan without resampling it."""

    if not (len(payloads) == len(request_ids) == len(offsets_s)):
        raise ValueError("planned payload, request-id and arrival counts must match")
    return tuple(
        OnlineRequest(
            request_key=request_id,
            arrival_offset_s=offset_s,
            payload=payload,
            sampling_params=sampling,
        )
        for payload, request_id, offset_s in zip(
            payloads,
            request_ids,
            offsets_s,
            strict=True,
        )
    )


def _prompt_audit(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    prompt_ids = []
    prompt_counts = []
    by_class: dict[str, list[list[int]]] = {}
    for result, case_id in zip(run.requests, request_classes, strict=True):
        seq = llm.scheduler.requests[result.request_id]
        ids = list(seq.token_ids[: seq.num_prompt_tokens])
        prompt_ids.append(ids)
        prompt_counts.append(len(ids))
        by_class.setdefault(case_id, []).append(ids)
    return {
        "prompt_tokens": sum(prompt_counts),
        "prompt_tokens_per_request": prompt_counts,
        "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
        "prompt_token_ids_sha256_by_class": {
            case_id: canonical_json_sha256(rows) for case_id, rows in sorted(by_class.items())
        },
    }


_PREFIX_COUNTER_FIELDS = (
    "pre_admission_hits",
    "visual_hydration_skips",
    "stale_probe_fallbacks",
    "hits",
    "misses",
    "admissions",
    "evictions",
    "rejections",
    "cow_copies",
    "tail_clone_hits",
    "tail_clone_admissions",
    "tail_clone_evictions",
)


def _prefix_metadata_delta(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, int]:
    return {name: int(after[name]) - int(before[name]) for name in _PREFIX_COUNTER_FIELDS}


def _population_prefix_evidence(
    llm,
    run,
    *,
    group_id: str,
    sample_id: str,
    expected_dense_pages: int,
    page_size_tokens: int,
    variant: str,
    metadata_before: dict[str, object],
    metadata_after: dict[str, object],
) -> dict[str, object]:
    if len(run.requests) != 1:
        raise RuntimeError("working-set population runs must contain one request")
    result = run.requests[0]
    if result.state != "finished":
        raise RuntimeError(f"working-set population request did not finish: {result.request_key!r}")
    seq = llm.scheduler.requests[result.request_id]
    prompt_ids = list(seq.token_ids[: seq.num_prompt_tokens])
    decision = dict(seq.visual_pruning_decision_record or {})
    entry = llm.scheduler.block_manager.multimodal_prefix_entry_metadata(seq)
    compacted_tokens = decision.get("compacted_prefix_kv_tokens")
    compacted_pages = (
        None
        if compacted_tokens is None
        else (int(compacted_tokens) + page_size_tokens - 1) // page_size_tokens
    )
    if variant == "vision_only":
        if entry is not None:
            raise RuntimeError("vision-only population unexpectedly retained prefix KV")
    elif entry is None:
        raise RuntimeError(f"prefix population retained no cache entry for group {group_id!r}")
    else:
        entry_pages = int(entry["canonical_blocks"])
        entry_tokens = int(entry["physical_prefix_tokens"])
        if compacted_pages is None:
            compacted_pages = entry_pages
            compacted_tokens = entry_tokens
        elif compacted_pages != entry_pages or int(compacted_tokens) != entry_tokens:
            raise RuntimeError(
                "physical compaction decision differs from retained prefix entry: "
                f"group={group_id}, decision_pages={compacted_pages}, "
                f"entry_pages={entry_pages}, decision_tokens={compacted_tokens}, "
                f"entry_tokens={entry_tokens}"
            )
    if variant in ("dense_prefix", "dense_block_prefix") and compacted_pages != expected_dense_pages:
        raise RuntimeError(
            "dense-prefix population differs from the measured working-set plan: "
            f"group={group_id}, expected={expected_dense_pages}, actual={compacted_pages}"
        )
    return {
        "group_id": group_id,
        "sample_id": sample_id,
        "request_id": result.request_key,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_sha256": canonical_json_sha256([prompt_ids]),
        "expected_dense_prefix_pages": expected_dense_pages,
        "compact_prefix_tokens": (None if compacted_tokens is None else int(compacted_tokens)),
        "compact_prefix_pages": compacted_pages,
        "compact_prefix_page_source": (
            None if variant == "vision_only" else "retained_prefix_entry_physical_tokens"
        ),
        "retained_prefix_entry": entry,
        "dropped_visual_tokens": int(decision.get("dropped_visual_tokens", 0)),
        "physical_compaction": bool(decision.get("physical_compaction", False)),
        "prefix_cache_hit": bool(seq.multimodal_prefix_cache_hit),
        "prefix_cache_counter_delta": _prefix_metadata_delta(
            metadata_before,
            metadata_after,
        ),
        "resident_prefix_pages_before": int(metadata_before["resident_blocks"]),
        "resident_prefix_pages_after": int(metadata_after["resident_blocks"]),
        "resident_prefix_entries_before": int(metadata_before["entries"]),
        "resident_prefix_entries_after": int(metadata_after["entries"]),
    }


def _verify_prism_working_set_runtime(
    llm,
    working_set: MaterializedWorkingSet,
) -> dict[str, object]:
    plan = working_set.plan
    budget = plan["kv_budget"]
    serving = plan["serving"]
    expected = {
        "image_max_pixels": int(plan["processor"]["image_max_pixels"]),
        "max_num_seqs": int(serving["max_num_seqs"]),
        "max_chunk_size": int(serving["max_chunk_size"]),
        "kv_budget_pages": int(budget["pages"]),
        "page_size_tokens": int(budget["page_size_tokens"]),
        "kv_budget_bytes": int(budget["bytes"]),
    }
    actual = {
        "image_max_pixels": int(llm.config.image_max_pixels),
        "max_num_seqs": int(llm.config.max_num_seqs),
        "max_chunk_size": int(llm.config.max_chunk_size),
        "kv_budget_pages": int(llm.config.num_kvcache_blocks),
        "page_size_tokens": int(llm.config.kvcache_block_size),
    }
    for name in (
        "image_max_pixels",
        "max_num_seqs",
        "max_chunk_size",
        "kv_budget_pages",
        "page_size_tokens",
    ):
        if actual[name] != expected[name]:
            raise RuntimeError(
                f"Prism working-set {name} mismatch: {actual[name]} != {expected[name]}"
            )
    prefix_metadata = llm.multimodal_prefix_cache_metadata()
    actual_pool_bytes = int(prefix_metadata["total_pool_blocks"]) * int(
        prefix_metadata["bytes_per_block_all_ranks"]
    )
    if int(prefix_metadata["total_pool_blocks"]) != expected["kv_budget_pages"]:
        raise RuntimeError("Prism KV pool page count differs from the working-set plan")
    if actual_pool_bytes != expected["kv_budget_bytes"]:
        raise RuntimeError(
            "Prism KV pool byte size differs from the working-set plan: "
            f"{actual_pool_bytes} != {expected['kv_budget_bytes']}"
        )
    return {
        "expected": expected,
        "actual": {**actual, "kv_budget_bytes": actual_pool_bytes},
        "processor": verify_working_set_processor(llm.vl_processor, plan),
        "kv_pool_lookup": {
            "total_pool_blocks": int(prefix_metadata["total_pool_blocks"]),
            "prefix_cache_max_blocks": int(prefix_metadata["max_blocks"]),
            "bytes_per_block_all_ranks": int(prefix_metadata["bytes_per_block_all_ranks"]),
        },
    }


def _visual_compaction_summary(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    class_by_id = {
        result.request_id: case_id
        for result, case_id in zip(run.requests, request_classes, strict=True)
    }
    decisions: list[tuple[str, dict[str, object]]] = []
    for result in run.requests:
        request_id = result.request_id
        seq = llm.scheduler.requests[request_id]
        record = seq.visual_pruning_decision_record
        if record and record.get("physical_compaction"):
            decisions.append((class_by_id[request_id], record))

    def aggregate(rows: list[dict[str, object]]) -> dict[str, int]:
        return {
            "decisions": len(rows),
            "effective_reclaims": sum(bool(row.get("released_block_ids")) for row in rows),
            "logical_prompt_tokens": sum(int(row["logical_prompt_tokens"]) for row in rows),
            "physical_prompt_tokens": sum(int(row["physical_prompt_kv_tokens"]) for row in rows),
            "released_blocks": sum(len(row.get("released_block_ids", [])) for row in rows),
            "dense_prompt_blocks": sum(len(row.get("old_block_table", [])) for row in rows),
            "physical_prompt_blocks": sum(len(row.get("new_block_table", [])) for row in rows),
            "dropped_visual_tokens": sum(int(row["dropped_visual_tokens"]) for row in rows),
        }

    by_class: dict[str, list[dict[str, object]]] = {}
    for case_id, record in decisions:
        by_class.setdefault(case_id, []).append(record)
    return {
        **aggregate([record for _, record in decisions]),
        "by_class": {case_id: aggregate(rows) for case_id, rows in sorted(by_class.items())},
    }


def _execution_evidence(llm, run_record: dict[str, object]) -> dict[str, object]:
    decode_batch_counts = Counter(
        int(batch["batch_size"])
        for batch in run_record["engine_metrics"]["batches"]
        if batch["phase"] == "decode"
    )
    max_decode_batch = max(decode_batch_counts, default=1)
    cuda_graph = llm.model_runner.cudagraph_metadata(max_decode_batch)
    capture_sizes = list(cuda_graph["batch_sizes"])
    replay_counts = []
    if cuda_graph["enabled"]:
        for actual_batch_size, count in sorted(decode_batch_counts.items()):
            captured_batch_size = next(size for size in capture_sizes if size >= actual_batch_size)
            replay_counts.append(
                {
                    "actual_batch_size": actual_batch_size,
                    "captured_batch_size": captured_batch_size,
                    "count": count,
                }
            )
    return {
        "observed_decode_batch_counts": [
            {"batch_size": batch_size, "count": count}
            for batch_size, count in sorted(decode_batch_counts.items())
        ],
        "cuda_graph": {
            **cuda_graph,
            "replay_counts": replay_counts,
            "total_observed_replays": (
                sum(decode_batch_counts.values()) if cuda_graph["enabled"] else 0
            ),
        },
        "torch_compile": llm.model_runner.compile_metadata(),
        "vision_tensor_cudagraph": (llm.model_runner.vision_tensor_cudagraph_metadata()),
    }


def _terminal_failure_summary(
    llm,
    run,
    request_classes: list[str],
) -> dict[str, object]:
    failures = []
    reasons: dict[str, int] = {}
    for result, case_id in zip(run.requests, request_classes, strict=True):
        if result.finish_reason not in {"rejected", "cancelled"}:
            continue
        seq = llm.scheduler.requests[result.request_id]
        transition = None if not seq.lifecycle.transitions else seq.lifecycle.transitions[-1]
        detail = None if transition is None else transition.reason
        reason_key = detail or result.finish_reason
        reasons[reason_key] = reasons.get(reason_key, 0) + 1
        failures.append(
            {
                "request_id": result.request_id,
                "request_class": case_id,
                "finish_reason": result.finish_reason,
                "detail": detail,
                "prompt_tokens": seq.num_prompt_tokens,
                "dense_kv_blocks": seq.num_blocks,
            }
        )
    return {
        "count": len(failures),
        "by_reason": reasons,
        "requests": failures,
    }


def _build_engine(args: argparse.Namespace):
    mode = MODE_SPECS[args.mode]
    return LLM(
        args.model,
        enforce_eager=mode.enforce_eager,
        execution_backend=mode.execution,
        decode_compile_region=mode.decode_compile_region,
        decode_compile_mode="max-autotune-no-cudagraphs",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
        allow_unsafe_decode_compile=(mode.decode_compile_region != "none"),
        compression_mode=mode.compression,
        tensor_parallel_size=args.tensor_parallel_size,
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
        scheduler_policy=args.scheduler_policy,
        max_consecutive_prefill_batches=(args.max_consecutive_prefill_batches),
        heavy_prefill_vision_patch_threshold=(args.heavy_prefill_vision_patch_threshold),
        min_decode_batches_between_heavy_prefills=(args.min_decode_batches_between_heavy_prefills),
        max_light_prefill_bypasses_per_heavy=(args.max_light_prefill_bypasses_per_heavy),
        visual_pruning_keep_ratio=args.visual_pruning_keep_ratio,
        visual_pruning_min_keep_tokens=args.visual_pruning_min_keep_tokens,
        visual_pruning_video_min_keep_tokens=(args.visual_pruning_video_min_keep_tokens),
        visual_pruning_strategy=args.visual_pruning_strategy,
        visual_pruning_attention_last_n_layers=(args.visual_pruning_attention_last_n_layers),
        logits_precision=mode.logits_precision or args.logits_precision,
        mlp_projection_mode=args.mlp_projection_mode,
        paged_decode_block_n=args.paged_decode_block_n or mode.paged_decode_block_n or 32,
        paged_decode_num_splits=args.paged_decode_num_splits or 1,
        enable_fused_qk_rmsnorm=mode.fused_qk_rmsnorm,
        enable_fused_qk_mrope=mode.fused_qk_mrope,
        enable_fused_add_rmsnorm=mode.fused_add_rmsnorm,
        enable_packed_kv_projection=mode.packed_kv_projection,
        enable_cooperative_prefill=args.enable_cooperative_prefill,
        cooperative_prefill_layer_quantum=(args.cooperative_prefill_layer_quantum),
        cooperative_prefill_vision_block_quantum=(args.cooperative_prefill_vision_block_quantum),
        enable_vision_tensor_cudagraph=args.enable_vision_tensor_cudagraph,
        enable_visual_embedding_cache=args.enable_visual_embedding_cache,
        vision_attention_backend="sdpa",
        image_max_pixels=args.image_max_pixels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--paged-decode-num-splits",
        type=int,
        default=None,
        help="override the paged decode split-K count (1/2/4/8)",
    )
    parser.add_argument(
        "--paged-decode-block-n",
        type=int,
        default=None,
        help="override the mode's paged decode tile size (16/32/64/128/256)",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--working-set-plan",
        help="run one shared MuirBench media-first working-set plan",
    )
    parser.add_argument(
        "--working-set-id",
        choices=("fit", "knee", "pressure"),
        help="workset selected from --working-set-plan",
    )
    parser.add_argument(
        "--working-set-variant",
        choices=("vision_only", "dense_prefix", "compact_prefix", "dense_block_prefix"),
        help="Prism ablation used with --working-set-plan",
    )
    parser.add_argument(
        "--materialized-root",
        help="root containing the MuirBench media referenced by the plan",
    )
    parser.add_argument("--case", default="single_image_448")
    parser.add_argument(
        "--mode",
        choices=ONLINE_MODES,
        default="off_graph",
    )
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="number of GPUs used to shard the model",
    )
    parser.add_argument(
        "--arrival-process",
        choices=("constant", "poisson", "burst"),
        default="constant",
    )
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--online-cpu-intraop-threads",
        type=int,
        default=8,
        help=(
            "bound host preprocessing parallelism so media preparation cannot "
            "starve the CUDA launch thread"
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument("--max-chunk-size", type=int, default=512)
    parser.add_argument("--max-queue-size", type=int)
    parser.add_argument(
        "--scheduler-policy",
        choices=("fcfs", "vision_aware", "slo_aware"),
        default="fcfs",
    )
    parser.add_argument("--max-consecutive-prefill-batches", type=int, default=1)
    parser.add_argument(
        "--heavy-prefill-vision-patch-threshold",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--min-decode-batches-between-heavy-prefills",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-light-prefill-bypasses-per-heavy",
        type=int,
        default=2,
    )
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
    parser.add_argument("--visual-pruning-keep-ratio", type=float, default=0.5)
    parser.add_argument("--visual-pruning-min-keep-tokens", type=int, default=32)
    parser.add_argument("--visual-pruning-video-min-keep-tokens", type=int)
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
    parser.add_argument("--enable-vision-tensor-cudagraph", action="store_true")
    parser.add_argument(
        "--enable-visual-embedding-cache",
        action="store_true",
        help=(
            "retain exact Vision Encoder outputs for repeated in-process "
            "media identities in a bounded GPU LRU"
        ),
    )
    parser.add_argument(
        "--media-repeat-rate",
        type=float,
        help=(
            "materialize fresh media objects with this exact content-repeat "
            "rate among multimodal requests"
        ),
    )
    parser.add_argument(
        "--vary-media-questions",
        action="store_true",
        help="use deterministic different question suffixes for media requests",
    )
    parser.add_argument(
        "--enable-cooperative-prefill",
        action="store_true",
        help=(
            "yield heavy visual prefills to decode CUDA Graphs at exact "
            "Transformer layer boundaries"
        ),
    )
    parser.add_argument(
        "--cooperative-prefill-layer-quantum",
        type=int,
        default=1,
        help="language Transformer layers executed between decode opportunities",
    )
    parser.add_argument(
        "--cooperative-prefill-vision-block-quantum",
        type=int,
        help=(
            "ViT blocks executed between decode opportunities; defaults to "
            "the language-layer quantum"
        ),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--profile-output",
        help="write the measured online run's semantic CPU/CUDA profile",
    )
    args = parser.parse_args()

    output_paths = [Path(path) for path in (args.output, args.profile_output) if path]
    if len(set(output_paths)) != len(output_paths):
        raise SystemExit("--output and --profile-output must be different files")
    for output_path in output_paths:
        if output_path.exists():
            raise SystemExit(f"refusing to replace existing benchmark output: {output_path}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the online benchmark")
    if args.requests <= 0 or args.warmup_requests < 0:
        raise SystemExit("--requests must be positive and warmup must be >= 0")
    if args.tensor_parallel_size <= 0:
        raise SystemExit("--tensor-parallel-size must be positive")
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be >= 2 for TPOT measurement")
    if args.online_cpu_intraop_threads <= 0:
        raise SystemExit("--online-cpu-intraop-threads must be positive")
    if args.request_rate <= 0 and args.arrival_process != "burst":
        raise SystemExit("--request-rate must be positive")
    if args.media_repeat_rate is not None and not 0.0 <= args.media_repeat_rate <= 1.0:
        raise SystemExit("--media-repeat-rate must be in [0, 1]")

    working_set_args = (
        args.working_set_plan,
        args.working_set_id,
        args.working_set_variant,
        args.materialized_root,
    )
    if any(working_set_args) and not all(working_set_args):
        raise SystemExit(
            "--working-set-plan, --working-set-id, --working-set-variant and "
            "--materialized-root must be used together"
        )
    if args.working_set_plan and (args.media_repeat_rate is not None or args.vary_media_questions):
        raise SystemExit("working-set plans cannot be combined with generated cache workloads")

    torch.set_num_threads(args.online_cpu_intraop_threads)
    working_set: MaterializedWorkingSet | None = None
    working_set_audit: dict[str, object] | None = None
    model_verification: dict[str, object] | None = None
    if args.working_set_plan:
        working_set = materialize_working_set(
            args.working_set_plan,
            workset_id=args.working_set_id,
            materialized_root=args.materialized_root,
        )
        model_verification = verify_working_set_model(working_set.plan, args.model)
        traffic = working_set.plan["traffic"]
        kv_budget = working_set.plan["kv_budget"]
        model_contract = working_set.plan["model"]
        processor_contract = working_set.plan["processor"]
        serving_contract = working_set.plan["serving"]
        args.requests = len(working_set.measured_payloads)
        args.max_tokens = int(traffic["max_new_tokens"])
        args.request_rate = float(traffic["request_rate_per_s"])
        args.arrival_process = str(traffic["arrival_process"])
        args.seed = int(traffic["seed"])
        args.max_model_len = int(model_contract["max_model_len"])
        args.max_num_batched_tokens = args.max_model_len
        args.max_num_seqs = int(serving_contract["max_num_seqs"])
        args.max_chunk_size = int(serving_contract["max_chunk_size"])
        args.image_max_pixels = int(processor_contract["image_max_pixels"])
        args.num_kvcache_blocks = int(kv_budget["pages"])
        args.kvcache_block_size = int(kv_budget["page_size_tokens"])
        args.enable_visual_embedding_cache = True
        args.visual_pruning_strategy = "uniform"
        args.visual_pruning_min_keep_tokens = 768
        args.visual_pruning_video_min_keep_tokens = 256
        if args.working_set_variant == "dense_block_prefix":
            # 真正的 Dense: compression=off + block 级 mm-aware 前缀匹配
            args.mode = "scaled_fp8_kv_compile_graph"
            args.enable_prefix_caching = True
        elif args.working_set_variant == "vision_only":
            args.mode = "scaled_fp8_kv_compile_graph"
            args.enable_prefix_caching = False
        else:
            args.mode = "visual_compact_scaled_fp8_compile_graph"
            args.enable_prefix_caching = True
            args.visual_pruning_keep_ratio = (
                1.0 if args.working_set_variant == "dense_prefix" else 0.6
            )
        manifest = {"name": "muirbench_media_first_working_set"}
        payloads = working_set.measured_payloads
        request_classes = working_set.measured_sample_ids
        workload_case = f"muirbench_{args.working_set_id}"
        cache_workload = None
        warmup_payloads = []
        plan_path = Path(args.working_set_plan)
        working_set_audit = {
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
            "workset_id": args.working_set_id,
            "variant": args.working_set_variant,
            "group_ids": list(working_set.workset["group_ids"]),
            "available_questions": int(working_set.workset["available_questions"]),
            "observed_questions": int(working_set.workset["observed_questions"]),
            "measured_question_switches": int(working_set.workset["measured_question_switches"]),
            "dense_prefix_pages": int(working_set.workset["dense_prefix_pages"]),
            "kv_budget_bytes": int(kv_budget["bytes"]),
            "kv_budget_pages": int(kv_budget["pages"]),
            "page_size_tokens": int(kv_budget["page_size_tokens"]),
            "model_revision": str(model_contract["revision"]),
            "model_verification": model_verification,
            "image_min_pixels": int(processor_contract["image_min_pixels"]),
            "image_max_pixels": int(processor_contract["image_max_pixels"]),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": int(serving_contract["max_num_seqs"]),
            "max_chunk_size": int(serving_contract["max_chunk_size"]),
            "source_prompt_sha256": source_prompt_schedule_sha256(working_set.measured_payloads),
            "population_source_prompt_sha256": source_prompt_schedule_sha256(
                working_set.population_payloads
            ),
            "prompt_token_hash_contract": "exact_token_ids_sha256_across_engines",
            "population_requests": len(working_set.population_payloads),
            "measured_requests": len(working_set.measured_payloads),
        }
    else:
        manifest = load_workload_manifest(args.manifest)
        case = find_workload_case(manifest, args.case)
        payloads = materialize_requests(case, repo_root=REPO_ROOT)
        request_classes = [args.case] * args.requests
        workload_case = args.case
        cache_workload = None
        warmup_payloads = payloads
        if args.media_repeat_rate is not None or args.vary_media_questions:
            warmup_payloads, _ = build_multimodal_cache_workload(
                payloads,
                repeat_rate=1.0,
                vary_questions=False,
            )
            payloads, cache_workload = build_multimodal_cache_workload(
                payloads,
                repeat_rate=(1.0 if args.media_repeat_rate is None else args.media_repeat_rate),
                vary_questions=args.vary_media_questions,
            )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    mode = MODE_SPECS[args.mode]
    llm = _build_engine(args)
    if working_set is not None:
        working_set_audit["runtime_verification"] = _verify_prism_working_set_runtime(
            llm,
            working_set,
        )
    process_memory_sampler = _DeviceProcessMemorySampler()
    profile_record = None
    population_record = None
    population_cache_state = None
    serving_session = OnlineServingSession(llm)
    process_memory_sampler.start()
    try:
        if working_set is not None:
            population_requests = _planned_online_requests(
                working_set.population_payloads,
                request_ids=working_set.population_request_ids,
                offsets_s=working_set.population_offsets_s,
                sampling=sampling,
            )
            population_runs = []
            population_evidence = []
            population_initial_metadata = llm.multimodal_prefix_cache_metadata()
            dense_pages_by_group = {
                str(group["group_id"]): int(group["dense_prefix_pages"])
                for group in working_set.plan["groups"]
            }
            for population_request, group_id, sample_id in zip(
                population_requests,
                working_set.population_group_ids,
                working_set.population_sample_ids,
                strict=True,
            ):
                metadata_before = llm.multimodal_prefix_cache_metadata()
                population_run = serving_session.run((population_request,))
                metadata_after = llm.multimodal_prefix_cache_metadata()
                run_record = population_run.to_record()
                population_runs.append(run_record)
                population_evidence.append(
                    _population_prefix_evidence(
                        llm,
                        population_run,
                        group_id=group_id,
                        sample_id=sample_id,
                        expected_dense_pages=dense_pages_by_group[group_id],
                        page_size_tokens=args.kvcache_block_size,
                        variant=args.working_set_variant,
                        metadata_before=metadata_before,
                        metadata_after=metadata_after,
                    )
                )
            population_final_metadata = llm.multimodal_prefix_cache_metadata()
            actual_compact_pages = [
                int(row["compact_prefix_pages"])
                for row in population_evidence
                if row["compact_prefix_pages"] is not None
            ]
            population_record = {
                "policy": "one_request_per_group_closed_loop",
                "groups": population_evidence,
                "expected_dense_prefix_pages": sum(
                    int(row["expected_dense_prefix_pages"]) for row in population_evidence
                ),
                "compact_prefix_pages": sum(actual_compact_pages),
                "prefix_cache_counter_delta": _prefix_metadata_delta(
                    population_initial_metadata,
                    population_final_metadata,
                ),
                "resident_prefix_pages_after": int(population_final_metadata["resident_blocks"]),
                "resident_prefix_entries_after": int(population_final_metadata["entries"]),
                "runs": population_runs,
            }
            population_cache_state = {
                "visual_embedding_cache": llm.visual_embedding_cache_metadata(),
                "multimodal_prefix_cache": llm.multimodal_prefix_cache_metadata(),
            }
            llm.reset_metrics()
            serving_session.reset_metrics()
        elif args.warmup_requests:
            warmup = _online_requests(
                warmup_payloads,
                count=args.warmup_requests,
                process="burst",
                request_rate=args.request_rate,
                seed=args.seed,
                sampling=sampling,
                key_prefix="warmup",
            )
            serving_session.run(warmup)
            llm.reset_metrics()
            serving_session.reset_metrics()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if working_set is None:
            requests = _online_requests(
                payloads,
                count=args.requests,
                process=args.arrival_process,
                request_rate=args.request_rate,
                seed=args.seed,
                sampling=sampling,
                key_prefix="measured",
            )
        else:
            requests = _planned_online_requests(
                payloads,
                request_ids=working_set.measured_request_ids,
                offsets_s=working_set.measured_offsets_s,
                sampling=sampling,
            )
        trace_sha256 = canonical_json_sha256(
            {
                "classes": request_classes,
                "offsets_s": [request.arrival_offset_s for request in requests],
            }
        )
        profile_context = (
            performance_profile(
                metadata={
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": str(Path(args.model).resolve()),
                    "mode": args.mode,
                    "workload_case": workload_case,
                    "request_count": args.requests,
                    "trace_sha256": trace_sha256,
                    "max_tokens": args.max_tokens,
                    "online_cpu_intraop_threads": torch.get_num_threads(),
                },
                cuda_timing=True,
            )
            if args.profile_output
            else nullcontext(None)
        )
        with profile_context as profile_session:
            run = serving_session.run(requests)
        torch.cuda.synchronize()
        process_device_memory = process_memory_sampler.stop()
        run_record = run.to_record()
        prompt_audit = _prompt_audit(llm, run, request_classes)
        compaction_summary = _visual_compaction_summary(
            llm,
            run,
            request_classes,
        )
        summary = summarize_online_run(run_record)
        git = collect_git_metadata(REPO_ROOT, strict=True)
        config_path = Path(args.model) / "config.json"
        kv_storage = kv_cache_storage_bytes(
            llm.model_runner.kv_cache,
            llm.model_runner.kv_scale_cache,
        )
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
                "config_sha256": (sha256_file(config_path) if config_path.is_file() else None),
            },
            "workload": {
                "manifest": manifest["name"],
                "case": workload_case,
                "source_request_types": [payload["type"] for payload in payloads],
                "request_classes": request_classes,
                "requests": args.requests,
                "max_tokens": args.max_tokens,
                "multimodal_cache_workload": cache_workload,
                "working_set_plan": working_set_audit,
                **prompt_audit,
            },
            "arrival": {
                "process": args.arrival_process,
                "request_rate_per_s": args.request_rate,
                "seed": args.seed,
                "offsets_s": [request.arrival_offset_s for request in requests],
                "trace_sha256": trace_sha256,
            },
            "engine": {
                "mode": args.mode,
                "execution_backend": mode.execution,
                "decode_compile_region": mode.decode_compile_region,
                "compression_mode": mode.compression,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "image_max_pixels": args.image_max_pixels,
                "max_chunk_size": args.max_chunk_size,
                "max_queue_size": args.max_queue_size,
                "scheduler_policy": args.scheduler_policy,
                "max_consecutive_prefill_batches": (args.max_consecutive_prefill_batches),
                "heavy_prefill_vision_patch_threshold": (args.heavy_prefill_vision_patch_threshold),
                "min_decode_batches_between_heavy_prefills": (
                    args.min_decode_batches_between_heavy_prefills
                ),
                "max_light_prefill_bypasses_per_heavy": (args.max_light_prefill_bypasses_per_heavy),
                "num_kvcache_blocks": args.num_kvcache_blocks,
                "kvcache_block_size": args.kvcache_block_size,
                "enable_prefix_caching": args.enable_prefix_caching,
                "logits_precision": mode.logits_precision or args.logits_precision,
                "mlp_projection_mode": args.mlp_projection_mode,
                "visual_pruning_keep_ratio": args.visual_pruning_keep_ratio,
                "visual_pruning_min_keep_tokens": (args.visual_pruning_min_keep_tokens),
                "visual_pruning_video_min_keep_tokens": (args.visual_pruning_video_min_keep_tokens),
                "visual_pruning_strategy": args.visual_pruning_strategy,
                "vision_tensor_cudagraph": (args.enable_vision_tensor_cudagraph),
                "visual_embedding_cache": (llm.visual_embedding_cache_metadata()),
                "multimodal_prefix_cache": (llm.multimodal_prefix_cache_metadata()),
                "cooperative_prefill": args.enable_cooperative_prefill,
                "cooperative_prefill_scope": (
                    "heavy_visual_then_all_loaded"
                    if args.enable_cooperative_prefill
                    else "disabled"
                ),
                "cooperative_prefill_layer_quantum": (args.cooperative_prefill_layer_quantum),
                "cooperative_prefill_vision_block_quantum": (
                    args.cooperative_prefill_layer_quantum
                    if args.cooperative_prefill_vision_block_quantum is None
                    else args.cooperative_prefill_vision_block_quantum
                ),
                "cooperative_prefill_quantum_policy": (
                    "deadline_coalesced_atomic_or_quarter_latched"
                    if args.enable_cooperative_prefill
                    else "disabled"
                ),
                "cooperative_prefill_fine_decode_batch_size": max(
                    1,
                    (args.max_num_seqs + 3) // 4,
                ),
                "cooperative_prefill_sparse_quantum_floor": 4,
                "cooperative_prefill_policy_runtime": (
                    llm.cooperative_prefill_policy_metadata()
                    if args.enable_cooperative_prefill
                    else None
                ),
                "online_media_preprocess": "single_worker_async",
                "online_cpu_intraop_threads": torch.get_num_threads(),
                "media_preprocess_in_ttft": True,
            },
            "memory": {
                "process_device": process_device_memory,
                "torch_allocator_is_headline_eligible": True,
                "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mib": torch.cuda.memory_reserved() / (1024**2),
                "peak_allocated_mib": (torch.cuda.max_memory_allocated() / (1024**2)),
                "kv_cache": {
                    "payload_bytes": kv_storage.payload,
                    "scale_bytes": kv_storage.scales,
                    "total_bytes": kv_storage.total,
                },
            },
            "visual_compaction": compaction_summary,
            "population": (
                None
                if population_record is None
                else {
                    "run": population_record,
                    "cache_state_after": population_cache_state,
                }
            ),
            "execution_evidence": _execution_evidence(llm, run_record),
            "terminal_failures": _terminal_failure_summary(
                llm,
                run,
                request_classes,
            ),
            "run": run_record,
            "summary": summary,
        }
        validate_online_benchmark_record(record)
        if profile_session is not None:
            profile_session.metadata.update(
                {
                    "git_commit": git.commit,
                    "git_dirty": git.dirty,
                    "gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda,
                    "torch": torch.__version__,
                    "completed_requests": sum(
                        request["state"] == "finished" for request in run_record["requests"]
                    ),
                    "terminal_failure_count": record["terminal_failures"]["count"],
                    "output_token_ids_sha256": canonical_json_sha256(
                        [request["token_ids"] for request in run_record["requests"]]
                    ),
                }
            )
            profile_record = profile_session.to_record()
            validate_performance_profile_record(profile_record)
    finally:
        if process_memory_sampler.running:
            try:
                process_memory_sampler.stop()
            except Exception:
                pass
        llm.exit()
        if working_set is not None:
            working_set.close()

    rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if args.output:
        output = Path(args.output)
        _write_text_atomic(output, rendered + "\n")
        print(f"wrote online record to {output}", file=sys.stderr)
    if args.profile_output:
        profile_output = Path(args.profile_output)
        _write_text_atomic(
            profile_output,
            json.dumps(
                profile_record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )
        print(f"wrote online profile to {profile_output}", file=sys.stderr)
    print(rendered)


if __name__ == "__main__":
    main()
