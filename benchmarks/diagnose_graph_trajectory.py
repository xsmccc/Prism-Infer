"""Diagnose eager/CUDA Graph logits on one fixed dynamic-batch trajectory.

This is a correctness diagnostic, not a performance benchmark.  Eager first
generates the natural greedy token history.  CUDA Graph then receives those
same tokens through a forced sampler while its unmodified logits and natural
argmax are compared row by row.  The fixed history prevents one low-margin
argmax change from making every later step incomparable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import (
    collect_git_metadata,
    collect_gpu_metadata,
    expand_case_batch,
    find_workload_case,
    materialize_requests,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.engine.contracts import BatchPhase, DeviceBatch
from prism_infer.layers.sampler import SAMPLING_NUMERICAL_EPSILON


GRAPH_TRAJECTORY_SCHEMA_VERSION = 1
TOP_K_DIAGNOSTIC_TOKENS = 2
LOGITS_BATCH_RANK = 2
MIN_DIAGNOSTIC_MAX_TOKENS = 2


@dataclass(frozen=True, slots=True)
class RequestTrajectory:
    """One request's selected history and pre-sampling logits."""

    token_ids: tuple[int, ...]
    natural_argmax_ids: tuple[int, ...]
    logits: tuple[torch.Tensor, ...]
    logits_sha256: str


@dataclass(slots=True)
class _MutableRequestTrajectory:
    token_ids: list[int] = field(default_factory=list)
    natural_argmax_ids: list[int] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)
    hasher: Any = field(default_factory=hashlib.sha256)

    def freeze(self) -> RequestTrajectory:
        return RequestTrajectory(
            token_ids=tuple(self.token_ids),
            natural_argmax_ids=tuple(self.natural_argmax_ids),
            logits=tuple(self.logits),
            logits_sha256=self.hasher.hexdigest(),
        )


def _tensor_hash_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _top_two(logits: torch.Tensor) -> dict[str, object]:
    values, indices = torch.topk(logits.float(), k=TOP_K_DIAGNOSTIC_TOKENS)
    return {
        "token_ids": [int(value) for value in indices.tolist()],
        "logits": [float(value) for value in values.tolist()],
        "margin": float((values[0] - values[1]).item()),
    }


def compare_logit_rows(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    selected_token_id: int,
) -> dict[str, object]:
    """Compare two vocabulary rows without changing either sampling decision."""

    if baseline.ndim != 1 or candidate.ndim != 1:
        raise ValueError("logit comparison requires rank-1 vocabulary rows")
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"logit row shapes differ: {list(baseline.shape)} != {list(candidate.shape)}"
        )
    if not 0 <= selected_token_id < baseline.numel():
        raise ValueError(f"selected token is outside vocabulary: {selected_token_id}")

    baseline_float = baseline.float()
    candidate_float = candidate.float()
    difference = (candidate_float - baseline_float).abs()
    max_index = int(difference.argmax().item())
    candidate_selected_logit = candidate_float[selected_token_id]
    return {
        "logits_exact": bool(torch.equal(baseline, candidate)),
        "nonzero_logit_count": int(torch.count_nonzero(difference).item()),
        "max_abs_logit_diff": float(difference[max_index].item()),
        "mean_abs_logit_diff": float(difference.mean().item()),
        "rms_logit_diff": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_diff_token_id": max_index,
        "max_diff_baseline_logit": float(baseline_float[max_index].item()),
        "max_diff_candidate_logit": float(candidate_float[max_index].item()),
        "baseline_top2": _top_two(baseline_float),
        "candidate_top2": _top_two(candidate_float),
        "baseline_selected_logit": float(baseline_float[selected_token_id].item()),
        "candidate_selected_logit": float(candidate_selected_logit.item()),
        "candidate_selected_rank": int(
            torch.count_nonzero(candidate_float > candidate_selected_logit).item()
        )
        + 1,
    }


def _decode_row_metadata(device_batch: DeviceBatch) -> list[dict[str, object]]:
    model_inputs = device_batch.model_inputs
    context = device_batch.attention_context
    batch_size = len(device_batch.sequence_ids)
    input_ids = model_inputs.input_ids.detach().cpu()
    positions = model_inputs.position_ids.detach().cpu()
    slot_mapping = context.slot_mapping.detach().cpu()
    context_lens = context.context_lens.detach().cpu()
    block_tables = context.block_tables.detach().cpu()
    rows: list[dict[str, object]] = []
    for row in range(batch_size):
        if positions.ndim == 1:
            row_positions = [int(positions[row].item())]
        else:
            row_positions = [int(value) for value in positions[:, row].tolist()]
        rows.append(
            {
                "input_token_id": int(input_ids[row].item()),
                "position_ids": row_positions,
                "slot_mapping": int(slot_mapping[row].item()),
                "context_len": int(context_lens[row].item()),
                "block_table": [int(value) for value in block_tables[row].tolist()],
            }
        )
    return rows


def _device_batch_record(
    device_batch: DeviceBatch,
    *,
    request_indices: tuple[int, ...],
    engine_step_index: int,
) -> dict[str, object]:
    is_decode = device_batch.phase is BatchPhase.DECODE
    return {
        "engine_step_index": engine_step_index,
        "phase": device_batch.phase.value,
        "request_indices": list(request_indices),
        "actual_batch_size": len(request_indices),
        "execution_bucket": device_batch.execution_bucket,
        "decode_max_context_len": (
            int(device_batch.attention_context.decode_max_context_len.item()) if is_decode else None
        ),
        "rows": _decode_row_metadata(device_batch) if is_decode else None,
    }


class DynamicBatchTrajectorySampler:
    """Record dynamic batches and optionally force a baseline token history."""

    def __init__(
        self,
        sequence_to_request: dict[int, int],
        *,
        runner: Any,
        baseline: dict[int, RequestTrajectory] | None = None,
    ) -> None:
        if not sequence_to_request:
            raise ValueError("trajectory sampler requires at least one request")
        self.sequence_to_request = dict(sequence_to_request)
        self.runner = runner
        self.baseline = baseline
        self._requests = {
            request_index: _MutableRequestTrajectory()
            for request_index in sorted(sequence_to_request.values())
        }
        self._pending_batch: dict[str, object] | None = None
        self._pending_request_indices: tuple[int, ...] = ()
        self._pending_consumed = False
        self._engine_step_index = 0
        self.batch_trace: list[dict[str, object]] = []
        self.comparisons: list[dict[str, object]] = []

    def begin_batch(self, device_batch: DeviceBatch) -> None:
        if self._pending_batch is not None:
            raise RuntimeError("previous diagnostic batch was not closed")
        try:
            request_indices = tuple(
                self.sequence_to_request[sequence_id] for sequence_id in device_batch.sequence_ids
            )
        except KeyError as exc:
            raise RuntimeError(f"unknown sequence id in diagnostic batch: {exc.args[0]}") from exc
        self._pending_request_indices = request_indices
        self._pending_batch = _device_batch_record(
            device_batch,
            request_indices=request_indices,
            engine_step_index=self._engine_step_index,
        )
        self._pending_consumed = False

    def _audit_graph_static_buffers(self) -> dict[str, object] | None:
        batch = self._pending_batch
        if batch is None or batch["phase"] != BatchPhase.DECODE.value:
            return None
        if bool(getattr(self.runner, "enforce_eager", True)):
            return None

        actual = int(batch["actual_batch_size"])
        captured = int(batch["execution_bucket"])
        graph_vars = self.runner.graph_vars
        rows = batch["rows"]
        if not isinstance(rows, list):
            raise RuntimeError("decode diagnostic batch is missing row metadata")

        expected_input_ids = torch.tensor(
            [row["input_token_id"] for row in rows],
            dtype=graph_vars["input_ids"].dtype,
            device=graph_vars["input_ids"].device,
        )
        expected_positions = torch.tensor(
            [row["position_ids"] for row in rows],
            dtype=graph_vars["positions"].dtype,
            device=graph_vars["positions"].device,
        ).T
        if expected_positions.shape[0] == 1:
            expected_positions = expected_positions.expand(
                graph_vars["positions"].shape[0],
                -1,
            )
        expected_slots = torch.tensor(
            [row["slot_mapping"] for row in rows],
            dtype=graph_vars["slot_mapping"].dtype,
            device=graph_vars["slot_mapping"].device,
        )
        expected_context_lens = torch.tensor(
            [row["context_len"] for row in rows],
            dtype=graph_vars["context_lens"].dtype,
            device=graph_vars["context_lens"].device,
        )
        actual_table_width = len(rows[0]["block_table"])
        expected_tables = torch.tensor(
            [row["block_table"] for row in rows],
            dtype=graph_vars["block_tables"].dtype,
            device=graph_vars["block_tables"].device,
        )
        static_tables = graph_vars["block_tables"]
        padding = slice(actual, captured)
        table_tail = static_tables[:actual, actual_table_width:]
        return {
            "active_input_ids_exact": bool(
                torch.equal(graph_vars["input_ids"][:actual], expected_input_ids)
            ),
            "active_position_ids_exact": bool(
                torch.equal(graph_vars["positions"][:, :actual], expected_positions)
            ),
            "active_slot_mapping_exact": bool(
                torch.equal(graph_vars["slot_mapping"][:actual], expected_slots)
            ),
            "active_context_lens_exact": bool(
                torch.equal(graph_vars["context_lens"][:actual], expected_context_lens)
            ),
            "active_block_table_prefix_exact": bool(
                torch.equal(static_tables[:actual, :actual_table_width], expected_tables)
            ),
            "active_block_table_tail_all_minus_one": bool(
                table_tail.numel() == 0 or torch.all(table_tail == -1).item()
            ),
            "padding_rows": captured - actual,
            "padding_slot_mapping_all_minus_one": bool(
                torch.all(graph_vars["slot_mapping"][padding] == -1).item()
            ),
            "padding_context_lens_all_zero": bool(
                torch.all(graph_vars["context_lens"][padding] == 0).item()
            ),
            "padding_block_tables_all_minus_one": bool(
                torch.all(static_tables[padding] == -1).item()
            ),
            "padding_input_ids": [
                int(value) for value in graph_vars["input_ids"][padding].tolist()
            ],
            "padding_position_ids": [
                [int(value) for value in row]
                for row in graph_vars["positions"][:, padding].T.tolist()
            ],
        }

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        if self._pending_batch is None:
            raise RuntimeError("sampler was called without a diagnostic DeviceBatch")
        if self._pending_consumed:
            raise RuntimeError("diagnostic sampler was called twice for one DeviceBatch")
        if logits.ndim != LOGITS_BATCH_RANK or logits.shape[0] != len(
            self._pending_request_indices
        ):
            raise RuntimeError(
                "diagnostic logits do not match the pending batch: "
                f"shape={list(logits.shape)}, requests={self._pending_request_indices}"
            )
        if bool(torch.any(temperatures > SAMPLING_NUMERICAL_EPSILON).item()):
            raise RuntimeError("fixed trajectory diagnostic requires greedy sampling")

        self._pending_batch["graph_static_buffer_audit"] = self._audit_graph_static_buffers()
        selected_tokens: list[int] = []
        for row, request_index in enumerate(self._pending_request_indices):
            request = self._requests[request_index]
            generation_index = len(request.token_ids)
            snapshot = logits[row].detach().cpu().contiguous()
            natural_token = int(snapshot.argmax().item())
            if self.baseline is None:
                selected_token = natural_token
                request.logits.append(snapshot)
            else:
                baseline_request = self.baseline[request_index]
                if generation_index >= len(baseline_request.token_ids):
                    raise RuntimeError(
                        f"request {request_index} exceeded its forced baseline trajectory"
                    )
                selected_token = baseline_request.token_ids[generation_index]
                comparison = compare_logit_rows(
                    baseline_request.logits[generation_index],
                    snapshot,
                    selected_token_id=selected_token,
                )
                comparison.update(
                    {
                        "engine_step_index": self._engine_step_index,
                        "phase": self._pending_batch["phase"],
                        "actual_batch_size": self._pending_batch["actual_batch_size"],
                        "execution_bucket": self._pending_batch["execution_bucket"],
                        "row_index": row,
                        "request_index": request_index,
                        "generation_index": generation_index,
                        "selected_token_id": selected_token,
                        "baseline_natural_argmax_id": (
                            baseline_request.natural_argmax_ids[generation_index]
                        ),
                        "candidate_natural_argmax_id": natural_token,
                    }
                )
                self.comparisons.append(comparison)
            request.token_ids.append(selected_token)
            request.natural_argmax_ids.append(natural_token)
            request.hasher.update(_tensor_hash_bytes(snapshot))
            selected_tokens.append(selected_token)
        self._pending_consumed = True
        return torch.tensor(selected_tokens, dtype=torch.long, device=logits.device)

    def end_batch(self) -> None:
        if self._pending_batch is None:
            raise RuntimeError("no diagnostic batch is open")
        if not self._pending_consumed:
            raise RuntimeError("diagnostic DeviceBatch completed without sampling")
        self.batch_trace.append(self._pending_batch)
        self._pending_batch = None
        self._pending_request_indices = ()
        self._pending_consumed = False
        self._engine_step_index += 1

    def abort_batch(self) -> None:
        """Drop diagnostic state without masking the backend's original error."""

        self._pending_batch = None
        self._pending_request_indices = ()
        self._pending_consumed = False

    def finish(
        self,
        outputs: dict[int, list[int]],
    ) -> dict[int, RequestTrajectory]:
        if self._pending_batch is not None:
            raise RuntimeError("cannot finish with an open diagnostic batch")
        frozen = {
            request_index: request.freeze() for request_index, request in self._requests.items()
        }
        for sequence_id, request_index in self.sequence_to_request.items():
            if sequence_id not in outputs:
                raise RuntimeError(f"missing terminal output for sequence {sequence_id}")
            if tuple(outputs[sequence_id]) != frozen[request_index].token_ids:
                raise RuntimeError(
                    f"terminal output differs from sampler history for request {request_index}"
                )
        return frozen


def _install_batch_trace(llm: LLM, sampler: DynamicBatchTrajectorySampler) -> None:
    """Wrap the selected backend only inside this diagnostic process."""

    backend = llm.model_runner.execution_backend
    original_execute = backend.execute

    def traced_execute(_backend: Any, device_batch: DeviceBatch):
        sampler.begin_batch(device_batch)
        try:
            result = original_execute(device_batch)
        except BaseException:
            sampler.abort_batch()
            raise
        else:
            sampler.end_batch()
            return result

    backend.execute = MethodType(traced_execute, backend)
    llm.model_runner.sampler = sampler


def _submit_requests(
    llm: LLM,
    requests: list[dict[str, Any]],
    sampling: SamplingParams,
) -> list[int]:
    sequence_ids: list[int] = []
    for request in requests:
        request_type = request["type"]
        if request_type == "text":
            sequence_id = llm.add_request(request["prompt"], sampling)
        elif request_type == "image":
            sequence_id = llm.add_vl_request(request["prompt"], request["image"], sampling)
        elif request_type == "images":
            sequence_id = llm.add_images_request(request["prompt"], request["images"], sampling)
        elif request_type == "video":
            sequence_id = llm.add_video_request(request["prompt"], request["video"], sampling)
        else:
            raise ValueError(f"unsupported request type: {request_type!r}")
        sequence_ids.append(sequence_id)
    return sequence_ids


def _build_llm(args: argparse.Namespace, *, enforce_eager: bool) -> LLM:
    return LLM(
        args.model,
        compression_mode="off",
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        logits_precision="model",
        mlp_projection_mode="packed",
        vision_attention_backend=args.vision_attention_backend,
    )


def _run_backend(
    args: argparse.Namespace,
    case: dict[str, Any],
    *,
    enforce_eager: bool,
    baseline: dict[int, RequestTrajectory] | None,
) -> tuple[
    dict[int, RequestTrajectory],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    llm = _build_llm(args, enforce_eager=enforce_eager)
    sampler: DynamicBatchTrajectorySampler | None = None
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    outputs: dict[int, list[int]] = {}
    try:
        requests = materialize_requests(case, repo_root=REPO_ROOT)
        sequence_ids = _submit_requests(llm, requests, sampling)
        sequence_to_request = {
            sequence_id: request_index for request_index, sequence_id in enumerate(sequence_ids)
        }
        sampler = DynamicBatchTrajectorySampler(
            sequence_to_request,
            runner=llm.model_runner,
            baseline=baseline,
        )
        _install_batch_trace(llm, sampler)
        while not llm.is_finished():
            step = llm.step_result()
            for output in step.outputs:
                outputs[output.request_id] = list(output.token_ids)
        trajectories = sampler.finish(outputs)
        graph_metadata = llm.model_runner.cudagraph_metadata(len(sequence_ids))
        return trajectories, sampler.batch_trace, sampler.comparisons, graph_metadata
    finally:
        # The diagnostic hook is intentionally bidirectional while running:
        # runner -> sampler and sampler -> runner.  Break both edges before
        # loading the next 8B backend in this same process.
        runner = getattr(llm, "model_runner", None)
        if sampler is not None:
            if runner is not None and getattr(runner, "sampler", None) is sampler:
                runner.sampler = None
            sampler.runner = None
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _trajectory_record(trajectories: dict[int, RequestTrajectory]) -> list[dict[str, object]]:
    return [
        {
            "request_index": request_index,
            "token_ids": list(trajectory.token_ids),
            "natural_argmax_ids": list(trajectory.natural_argmax_ids),
            "logits_sha256": trajectory.logits_sha256,
        }
        for request_index, trajectory in sorted(trajectories.items())
    ]


def _trace_without_graph_details(trace: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for batch in trace:
        normalized.append(
            {
                key: value
                for key, value in batch.items()
                if key not in {"execution_bucket", "graph_static_buffer_audit"}
            }
        )
    return normalized


def _first_record(
    comparisons: list[dict[str, object]],
    predicate,
) -> dict[str, object] | None:
    return next((record for record in comparisons if predicate(record)), None)


def _summarize(
    baseline: dict[int, RequestTrajectory],
    candidate: dict[int, RequestTrajectory],
    eager_trace: list[dict[str, object]],
    graph_trace: list[dict[str, object]],
    comparisons: list[dict[str, object]],
) -> dict[str, object]:
    first_numeric = _first_record(comparisons, lambda row: not bool(row["logits_exact"]))
    first_argmax = _first_record(
        comparisons,
        lambda row: row["baseline_natural_argmax_id"] != row["candidate_natural_argmax_id"],
    )
    graph_audits = [
        batch["graph_static_buffer_audit"]
        for batch in graph_trace
        if batch["graph_static_buffer_audit"] is not None
    ]
    audit_boolean_fields = (
        "active_input_ids_exact",
        "active_position_ids_exact",
        "active_slot_mapping_exact",
        "active_context_lens_exact",
        "active_block_table_prefix_exact",
        "active_block_table_tail_all_minus_one",
        "padding_slot_mapping_all_minus_one",
        "padding_context_lens_all_zero",
        "padding_block_tables_all_minus_one",
    )
    return {
        "fixed_token_history_exact": all(
            candidate[index].token_ids == baseline[index].token_ids for index in baseline
        ),
        "scheduler_and_device_input_trace_exact": (
            _trace_without_graph_details(eager_trace) == _trace_without_graph_details(graph_trace)
        ),
        "natural_argmax_exact": all(
            candidate[index].natural_argmax_ids == baseline[index].natural_argmax_ids
            for index in baseline
        ),
        "logits_exact": all(bool(row["logits_exact"]) for row in comparisons),
        "first_numeric_difference": first_numeric,
        "first_natural_argmax_difference": first_argmax,
        "max_abs_logit_diff": max(float(row["max_abs_logit_diff"]) for row in comparisons),
        "graph_static_buffer_audits": {
            field: all(bool(audit[field]) for audit in graph_audits)
            for field in audit_boolean_fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", default="h1_eight_image_448")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--num-kvcache-blocks", type=int, default=113)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--vision-attention-backend",
        choices=("sdpa", "flash_attn"),
        default="sdpa",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size < 1 or args.max_tokens < MIN_DIAGNOSTIC_MAX_TOKENS:
        raise SystemExit("--batch-size must be >= 1 and --max-tokens must be >= 2")
    if args.max_num_seqs < args.batch_size:
        raise SystemExit("--max-num-seqs must cover --batch-size")
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite diagnostic artifact: {output_path}")

    manifest = load_workload_manifest(args.manifest)
    source_case = find_workload_case(manifest, args.case)
    case, source_requests, replication_factor = expand_case_batch(
        source_case,
        args.batch_size,
    )

    print("running fixed-trajectory eager baseline", file=sys.stderr)
    eager, eager_trace, eager_comparisons, eager_metadata = _run_backend(
        args,
        case,
        enforce_eager=True,
        baseline=None,
    )
    if eager_comparisons:
        raise RuntimeError("eager baseline unexpectedly produced comparison records")
    print("running teacher-forced CUDA Graph candidate", file=sys.stderr)
    graph, graph_trace, comparisons, graph_metadata = _run_backend(
        args,
        case,
        enforce_eager=False,
        baseline=eager,
    )

    git = collect_git_metadata(REPO_ROOT)
    output = {
        "schema_version": GRAPH_TRAJECTORY_SCHEMA_VERSION,
        "record_type": "cuda_graph_fixed_trajectory_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "correctness_diagnostic_not_performance_claim",
        "environment": {
            **git.as_dict(commit_key="git_commit", dirty_key="git_dirty"),
            **collect_gpu_metadata().environment_dict(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "model": str(Path(args.model).resolve()),
        "workload": {
            "manifest": str(Path(args.manifest).resolve()),
            "case_id": args.case,
            "batch_size": args.batch_size,
            "source_requests": source_requests,
            "replication_factor": replication_factor,
            "max_tokens": args.max_tokens,
        },
        "engine_config": {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "kvcache_block_size": args.kvcache_block_size,
            "vision_attention_backend": args.vision_attention_backend,
            "compression_mode": "off",
            "logits_precision": "model",
            "mlp_projection_mode": "packed",
            "prefix_caching": False,
            "chunked_prefill": False,
        },
        "eager_backend": eager_metadata,
        "graph_backend": graph_metadata,
        "eager_trajectories": _trajectory_record(eager),
        "graph_trajectories": _trajectory_record(graph),
        "eager_batch_trace": eager_trace,
        "graph_batch_trace": graph_trace,
        "comparisons": comparisons,
        "summary": _summarize(
            eager,
            graph,
            eager_trace,
            graph_trace,
            comparisons,
        ),
    }
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
