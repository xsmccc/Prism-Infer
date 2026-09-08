"""Run two independent TP1 replicas on one job's two visible GPUs.

Both replicas load once and use bench_vision_parallel's in-process token client.
Cold and hot concurrent batches have separate ready/START barriers. DP2 throughput
is total returned tokens divided by the global earliest-start/latest-finish span,
not the sum of the two local throughput numbers. Raw per-request JSON and complete
child logs remain in --output-dir. This parent does not import torch or touch CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter_ns
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vision_parallel_report import save_report

BENCHMARK = Path(__file__).with_name("bench_vision_parallel.py")
READY_PREFIX = "VISION_PARALLEL_READY "
PHASES = ("concurrent_cold_prefix_disabled", "concurrent_hot")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--devices",
        nargs=2,
        help="Two CUDA device IDs/UUIDs; default is the job's CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=448 * 448,
        help="Processor pixel limit per image for both replicas",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--hot-prompt")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4, help="Per replica")
    parser.add_argument("--num-kvcache-blocks", type=int, default=64, help="Per replica")
    parser.add_argument("--kvcache-block-size", type=int, choices=(16, 256), default=256)
    parser.add_argument("--concurrent-requests", type=int, default=4, help="Per replica")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="Sequential probes per replica")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=840.0,
        help="Total runtime before cleanup; default leaves 60s in a 15-minute Slurm job",
    )
    return parser


def _resolve_devices(explicit: list[str] | None) -> list[str]:
    devices = explicit or [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("provide exactly two distinct GPUs via CUDA_VISIBLE_DEVICES or --devices")
    return devices


def _child_command(args: argparse.Namespace, replica_id: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(BENCHMARK),
        "--model",
        str(args.model.resolve()),
        "--output",
        str(output),
        "--tensor-parallel-size",
        "1",
        "--vision-mode",
        "replicated",
        "--execution-backend",
        "cuda_graph",
        "--replica-id",
        str(replica_id),
        "--wait-for-start",
    ]
    for name in (
        "image_size",
        "image_max_pixels",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "num_kvcache_blocks",
        "kvcache_block_size",
        "concurrent_requests",
        "max_tokens",
        "warmup",
        "repeat",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for name in ("model_revision", "prompt", "hot_prompt"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), value])
    if args.images:
        command.extend(["--images", *(str(path.resolve()) for path in args.images)])
    return command


def _read_output(
    replica_id: int,
    process: subprocess.Popen[str],
    log_path: Path,
    events: queue.Queue[tuple[str, int, Any]],
) -> None:
    try:
        assert process.stdout is not None
        with log_path.open("x", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if READY_PREFIX in line:
                    payload = json.loads(line.split(READY_PREFIX, 1)[1])
                    events.put(("ready", replica_id, payload))
        events.put(("eof", replica_id, None))
    except Exception as exc:
        events.put(("reader_error", replica_id, f"{type(exc).__name__}: {exc}"))


def _phase_barrier(
    phase: str,
    processes: list[subprocess.Popen[str]],
    events: queue.Queue[tuple[str, int, Any]],
    deadline: float,
) -> dict[str, Any]:
    ready: dict[int, int] = {}
    while len(ready) != len(processes):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {phase}: ready replicas={sorted(ready)}")
        try:
            kind, replica_id, payload = events.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(f"timed out waiting for {phase}") from exc
        if kind != "ready":
            raise RuntimeError(f"replica {replica_id} stopped before {phase}: {kind}, {payload}")
        if payload.get("phase") != phase or replica_id in ready:
            raise RuntimeError(f"unexpected readiness from replica {replica_id}: {payload}")
        ready[replica_id] = perf_counter_ns()
    releases: dict[int, int] = {}
    for replica_id, process in enumerate(processes):
        assert process.stdin is not None
        releases[replica_id] = perf_counter_ns()
        process.stdin.write("START\n")
        process.stdin.flush()
    return {
        "phase": phase,
        "parent_observed_ready_ns": ready,
        "parent_start_write_ns": releases,
        "start_write_skew_ms": (max(releases.values()) - min(releases.values())) / 1e6,
    }


def _aggregate_phase(reports: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    rows_by_replica = [
        [row for row in report.get("requests", []) if row.get("phase") == phase]
        for report in reports
    ]
    rows = [row for group in rows_by_replica for row in group]
    completed = [row for row in rows if row.get("status") == "finished"]
    if not completed:
        return {"submitted_records": len(rows), "completed_requests": 0}
    start = min(row["client_start_ns"] for row in rows)
    finish = max(row["client_finish_ns"] for row in completed)
    duration_s = (finish - start) / 1e9
    tokens = sum(len(row["token_ids"]) for row in completed)
    starts = [min(row["client_start_ns"] for row in group) for group in rows_by_replica if group]
    exact = None
    if len(rows_by_replica) == 2 and len(rows_by_replica[0]) == len(rows_by_replica[1]):
        exact = all(
            left.get("status") == right.get("status") == "finished"
            and left["token_ids"] == right["token_ids"]
            for left, right in zip(rows_by_replica[0], rows_by_replica[1], strict=True)
        )
    return {
        "submitted_records": len(rows),
        "completed_requests": len(completed),
        "all_records_finished": len(rows) == len(completed),
        "client_start_ns": start,
        "client_finish_ns": finish,
        "duration_s": duration_s,
        "output_tokens": tokens,
        "output_tokens_per_s": tokens / duration_s if duration_s > 0 else None,
        "requests_per_s": len(completed) / duration_s if duration_s > 0 else None,
        "replica_first_request_start_skew_ms": (max(starts) - min(starts)) / 1e6,
        "cross_replica_token_ids_exact_by_request_index": exact,
        "full_visual_prefix_reused_requests": sum(
            bool(row.get("full_visual_prefix_reused")) for row in completed
        ),
        "per_request_ttft_ms": [row["ttft_ms"] for row in completed],
        "per_request_tpot_ms": [row["tpot_ms"] for row in completed],
        "per_request_e2e_ms": [row["e2e_ms"] for row in completed],
    }


def _stop_children(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        devices = _resolve_devices(args.devices)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout_seconds <= 0 or args.concurrent_requests < 1:
        parser.error("timeout-seconds and concurrent-requests must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = [output_dir / f"replica_{index}.json" for index in range(2)]
    log_paths = [output_dir / f"replica_{index}.log" for index in range(2)]
    summary_path = output_dir / "dp2_summary.json"
    existing = [str(path) for path in [*raw_paths, *log_paths, summary_path] if path.exists()]
    if existing:
        parser.error("refusing to replace existing evidence: " + ", ".join(existing))
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "two_independent_tp1_replicas",
        "parent_pid": os.getpid(),
        "timeout_seconds": args.timeout_seconds,
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "devices": devices,
        "commands": [],
        "phase_barriers": [],
        "resource_contract": {
            "gpu_count": 2,
            "model_replicas": 2,
            "tensor_parallel_size_per_replica": 1,
            "max_num_seqs_per_replica": args.max_num_seqs,
            "max_num_seqs_total": 2 * args.max_num_seqs,
            "max_num_batched_tokens_per_replica": args.max_num_batched_tokens,
            "max_num_batched_tokens_total": 2 * args.max_num_batched_tokens,
            "concurrent_requests_per_replica": args.concurrent_requests,
            "concurrent_requests_total": 2 * args.concurrent_requests,
            "kv_blocks_per_replica": args.num_kvcache_blocks,
            "kv_budget_note": "two complete TP1 KV pools; twice one engine, not equal total KV",
            "comparison_note": (
                "Compare on the same two physical GPUs. Match total offered requests and record "
                "global sequence slots and KV bytes; do not describe these as automatically equal."
            ),
        },
        "timing_scope": "in_process_client; aggregate earliest_start_to_latest_finish",
        "throughput_scope": "finite_concurrent_batch_with_prefill_and_drain_not_saturation",
        "sequential_probe_note": (
            "Child warmup/sequential/partial probes can overlap on the host. They are retained "
            "as diagnostics, not a new isolated TP1 latency baseline."
        ),
    }
    events: queue.Queue[tuple[str, int, Any]] = queue.Queue()
    processes: list[subprocess.Popen[str]] = []
    readers: list[threading.Thread] = []
    failure: Exception | None = None
    deadline = monotonic() + args.timeout_seconds
    save_report(summary_path, report)
    try:
        for replica_id, device in enumerate(devices):
            command = _child_command(args, replica_id, raw_paths[replica_id])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append(process)
            report["commands"].append(
                {
                    "replica_id": replica_id,
                    "pid": process.pid,
                    "cuda_visible_devices": device,
                    "argv": command,
                }
            )
            reader = threading.Thread(
                target=_read_output,
                args=(replica_id, process, log_paths[replica_id], events),
                daemon=True,
            )
            reader.start()
            readers.append(reader)
        for phase in PHASES:
            report["phase_barriers"].append(_phase_barrier(phase, processes, events, deadline))
            print(f"DP2_START {phase}", flush=True)
            save_report(summary_path, report)
        for replica_id, process in enumerate(processes):
            code = process.wait(timeout=max(0.1, deadline - monotonic()))
            if code != 0:
                raise RuntimeError(f"replica {replica_id} exited with {code}; see its log")
    except Exception as exc:
        failure = exc
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        _stop_children(processes)
        for reader in readers:
            reader.join(timeout=5)
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        child_reports = []
        report["replicas"] = []
        for replica_id, raw_path in enumerate(raw_paths):
            child: dict[str, Any] = {}
            if raw_path.exists():
                try:
                    child = json.loads(raw_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    report.setdefault("artifact_errors", []).append(str(exc))
            child_reports.append(child)
            report["replicas"].append(
                {
                    "replica_id": replica_id,
                    "cuda_visible_devices": devices[replica_id],
                    "raw_json": str(raw_path),
                    "log": str(log_paths[replica_id]),
                    "returncode": (
                        processes[replica_id].returncode if replica_id < len(processes) else None
                    ),
                    "status": child.get("status", "missing"),
                    "gpus": child.get("gpus"),
                    "kv_cache": child.get("kv_cache"),
                    "phase_summaries": child.get("phase_summaries"),
                    "partial_prefix_validation": child.get("partial_prefix_validation"),
                    "error": child.get("error"),
                }
            )
        report["phases"] = {phase: _aggregate_phase(child_reports, phase) for phase in PHASES}
        per_replica_bytes = [
            child.get("kv_cache", {}).get("total_bytes_equal_shards") for child in child_reports
        ]
        report["kv_cache"] = {
            "bytes_per_replica": per_replica_bytes,
            "total_bytes": (
                sum(per_replica_bytes)
                if all(isinstance(v, int) for v in per_replica_bytes)
                else None
            ),
            "budget_relation": "sum of two TP1 pools, not a same-KV-budget TP2 comparison",
        }
        complete = failure is None and all(
            child.get("status") == "complete" for child in child_reports
        )
        report["status"] = "complete" if complete else "failed"
        save_report(summary_path, report)
    print("DP2_RESULT " + str(summary_path), flush=True)
    if report["status"] != "complete":
        raise SystemExit(f"DP2 benchmark failed: {report.get('error', 'missing child result')}")


if __name__ == "__main__":
    main()
