"""Run a balanced two-mode P9 matrix with one fresh process per measurement."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SYSTEM = REPO_ROOT / "benchmarks" / "bench_system.py"
DEFAULT_FRESH_PROCESS_REPEATS = 5
DEFAULT_WARMUP = 2
DEFAULT_MAX_IDLE_MEMORY_MIB = 64.0
DEFAULT_MAX_IDLE_UTILIZATION_PERCENT = 5.0
DEFAULT_RELEASE_TIMEOUT_SECONDS = 15.0
DEFAULT_BOOTSTRAP_SEED = 20260717
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
MIN_FORMAL_FRESH_PROCESS_REPEATS = 5
MIN_FORMAL_WARMUP = 2
REQUIRED_CHILD_SCHEMA_VERSION = 9
PROCESS_MATRIX_SCHEMA_VERSION = 1
PAIR_ORDER_PATTERNS = (("A", "B", "B", "A"), ("B", "A", "A", "B"))
NVIDIA_SMI_QUERY_FIELDS = (
    "uuid",
    "name",
    "memory.used",
    "memory.free",
    "utilization.gpu",
)
MODE_VARIANT_FIELDS = frozenset(
    {
        "name",
        "execution",
        "attention",
        "logits_precision",
        "paged_decode_block_n",
        "fused_qk_rmsnorm",
    }
)
MODEL_TUNING_FIELDS = frozenset(
    {"logits_precision", "paged_decode_block_n", "fused_qk_rmsnorm"}
)
CUDA_GRAPH_EXECUTION_FIELDS = frozenset(
    {
        "cuda_graph_batch_sizes",
        "cuda_graph_capture_ms",
        "cuda_graph_capture_scope",
        "cuda_graph_enabled",
        "cuda_graph_replay_counts",
        "decode_backend",
        "paged_decode_block_n",
    }
)


@dataclass(frozen=True, slots=True)
class GpuState:
    uuid: str
    name: str
    memory_used_mib: float
    memory_free_mib: float
    utilization_percent: float


@dataclass(frozen=True, slots=True)
class ProcessMetric:
    path: tuple[str, ...]
    preferred_direction: str


PROCESS_METRICS = {
    "decode_step_ms": ProcessMetric(("timing_ms", "decode_step", "median"), "lower"),
    "engine_ttft_ms": ProcessMetric(("timing_ms", "engine_ttft", "median"), "lower"),
    "end_to_end_ttft_ms": ProcessMetric(
        ("timing_ms", "end_to_end_ttft", "median"),
        "lower",
    ),
    "end_to_end_ms": ProcessMetric(("timing_ms", "end_to_end", "median"), "lower"),
    "decode_tokens_per_s": ProcessMetric(
        ("throughput", "decode_tokens_per_s", "median"),
        "higher",
    ),
    "engine_output_tokens_per_s": ProcessMetric(
        ("throughput", "engine_output_tokens_per_s", "median"),
        "higher",
    ),
    "peak_allocated_mb": ProcessMetric(
        ("memory_mb", "peak_allocated", "median"),
        "lower",
    ),
    "reserved_mb": ProcessMetric(("memory_mb", "reserved", "median"), "lower"),
}


def balanced_pair_order(repeats_per_mode: int) -> tuple[str, ...]:
    """Truncate alternating ABBA/BAAB blocks to equal per-mode counts."""

    if isinstance(repeats_per_mode, bool) or not isinstance(repeats_per_mode, int):
        raise TypeError("repeats_per_mode must be an integer")
    if repeats_per_mode <= 0:
        raise ValueError("repeats_per_mode must be positive")
    counts = {"A": 0, "B": 0}
    order: list[str] = []
    pattern_index = 0
    while min(counts.values()) < repeats_per_mode:
        for label in PAIR_ORDER_PATTERNS[pattern_index % len(PAIR_ORDER_PATTERNS)]:
            if counts[label] >= repeats_per_mode:
                continue
            order.append(label)
            counts[label] += 1
        pattern_index += 1
    return tuple(order)


def parse_gpu_state(output: str, *, expected_uuid: str) -> GpuState:
    """Select exactly one physical UUID from nounit nvidia-smi CSV output."""

    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == len(NVIDIA_SMI_QUERY_FIELDS):
            rows.append(fields)
    matches = [row for row in rows if row[0] == expected_uuid]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one nvidia-smi row for {expected_uuid}, got {len(matches)}"
        )
    uuid, name, used, free, utilization = matches[0]
    try:
        return GpuState(
            uuid=uuid,
            name=name,
            memory_used_mib=float(used),
            memory_free_mib=float(free),
            utilization_percent=float(utilization),
        )
    except ValueError as exc:
        raise RuntimeError(f"nvidia-smi returned non-numeric state: {matches[0]}") from exc


def validate_idle_gpu(
    state: GpuState,
    *,
    max_memory_used_mib: float,
    max_utilization_percent: float,
) -> None:
    """Fail a process-level measurement before any model allocation."""

    if state.memory_used_mib > max_memory_used_mib:
        raise RuntimeError(
            f"GPU idle memory gate failed: {state.memory_used_mib} > {max_memory_used_mib} MiB"
        )
    if state.utilization_percent > max_utilization_percent:
        raise RuntimeError(
            "GPU idle utilization gate failed: "
            f"{state.utilization_percent} > {max_utilization_percent}%"
        )


def _query_gpu_state(expected_uuid: str) -> GpuState:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return parse_gpu_state(output, expected_uuid=expected_uuid)


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return commit, dirty


def _artifact_paths(output: Path) -> tuple[Path, Path, Path]:
    resolved_output = output.resolve()
    manifest = resolved_output.with_suffix(".manifest.json")
    run_dir = resolved_output.parent / f"{resolved_output.stem}_runs"
    return resolved_output, manifest, run_dir


def _is_git_ignored(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return True
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative.as_posix(),
        ],
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed for artifact path: {path}")
    return completed.returncode == 0


def validate_artifact_destination(output: Path) -> None:
    """Prevent benchmark artifacts from changing the measured Git identity."""

    aggregate, manifest, run_dir = _artifact_paths(output)
    non_ignored = [path for path in (aggregate, manifest, run_dir) if not _is_git_ignored(path)]
    if non_ignored:
        joined = ", ".join(str(path) for path in non_ignored)
        raise RuntimeError(
            "benchmark outputs inside the repository must be gitignored; "
            f"refusing destinations: {joined}"
        )


def _ensure_artifacts_absent(output: Path, manifest: Path, run_dir: Path) -> None:
    existing = [path for path in (output, manifest, run_dir) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"refusing to overwrite existing benchmark artifacts: {joined}")


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _wait_for_release(args: argparse.Namespace) -> GpuState:
    deadline = time.monotonic() + args.release_timeout_seconds
    last_state = _query_gpu_state(args.expected_gpu_uuid)
    while time.monotonic() < deadline:
        try:
            validate_idle_gpu(
                last_state,
                max_memory_used_mib=args.max_idle_memory_mib,
                max_utilization_percent=args.max_idle_utilization_percent,
            )
            return last_state
        except RuntimeError:
            time.sleep(0.25)
            last_state = _query_gpu_state(args.expected_gpu_uuid)
    validate_idle_gpu(
        last_state,
        max_memory_used_mib=args.max_idle_memory_mib,
        max_utilization_percent=args.max_idle_utilization_percent,
    )
    return last_state


def _child_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
) -> list[str]:
    command = [
        args.python,
        str(BENCH_SYSTEM),
        "--model",
        str(Path(args.model).resolve()),
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--case",
        args.case,
        "--modes",
        mode,
        "--max-tokens",
        str(args.max_tokens),
        "--batch-sizes",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--repeat",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--num-kvcache-blocks",
        str(args.num_kvcache_blocks),
        "--kvcache-block-size",
        str(args.kvcache_block_size),
        "--vision-attention-backend",
        args.vision_attention_backend,
        "--logits-precision",
        args.logits_precision,
        "--paged-decode-block-n",
        str(args.paged_decode_block_n),
        "--mlp-projection-mode",
        args.mlp_projection_mode,
        "--disable-prefix-caching",
        "--quiet",
        "--output",
        str(output),
    ]
    return command


def _read_single_record(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"fresh-process child must write one record, got {len(lines)}: {path}")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise RuntimeError(f"child record must be an object: {path}")
    return record


def _nested_number(record: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise RuntimeError(f"child record is missing metric path: {'.'.join(path)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"child metric must be numeric: {'.'.join(path)}={value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RuntimeError(f"child metric must be finite and positive: {'.'.join(path)}={value!r}")
    return numeric


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("percentile fraction must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summarize_process_values(values: Sequence[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("process summary requires non-empty values")
    return {
        "count": len(values),
        "median": median(values),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_median_ratio(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int | list[float]]:
    """Bootstrap the candidate/baseline process-median ratio."""

    if not baseline or not candidate:
        raise ValueError("bootstrap requires non-empty baseline and candidate samples")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    numeric_baseline = [float(value) for value in baseline]
    numeric_candidate = [float(value) for value in candidate]
    if not all(math.isfinite(value) and value > 0.0 for value in numeric_baseline):
        raise ValueError("bootstrap baseline samples must be finite and positive")
    if not all(math.isfinite(value) and value > 0.0 for value in numeric_candidate):
        raise ValueError("bootstrap candidate samples must be finite and positive")
    generator = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        baseline_draw = [
            numeric_baseline[generator.randrange(len(numeric_baseline))] for _ in numeric_baseline
        ]
        candidate_draw = [
            numeric_candidate[generator.randrange(len(numeric_candidate))]
            for _ in numeric_candidate
        ]
        ratios.append(median(candidate_draw) / median(baseline_draw))
    lower = _nearest_rank(ratios, 0.025)
    upper = _nearest_rank(ratios, 0.975)
    return {
        "point_estimate": median(numeric_candidate) / median(numeric_baseline),
        "confidence_interval_95": [lower, upper],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def _metric_samples(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    return {
        name: [_nested_number(record, spec.path) for record in records]
        for name, spec in PROCESS_METRICS.items()
    }


def _mode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = _metric_samples(records)
    return {
        "fresh_process_repeats": len(records),
        "medians": {name: median(values) for name, values in fields.items()},
        "process_statistics": {
            name: _summarize_process_values(values) for name, values in fields.items()
        },
        "raw": fields,
        "kv_cache_bytes": records[0]["kv_cache"]["bytes"],
        "output_sha256": records[0]["correctness"]["output_sha256"],
    }


def _comparison_summary(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str,
    candidate_mode: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    baseline_samples = _metric_samples(baseline)
    candidate_samples = _metric_samples(candidate)
    metrics = {}
    for name, spec in PROCESS_METRICS.items():
        ratio = bootstrap_median_ratio(
            baseline_samples[name],
            candidate_samples[name],
            seed=seed,
            resamples=resamples,
        )
        lower, upper = ratio["confidence_interval_95"]
        point = ratio["point_estimate"]
        if spec.preferred_direction == "lower":
            improvement = 1.0 - point
            improvement_interval = [1.0 - upper, 1.0 - lower]
        else:
            improvement = point - 1.0
            improvement_interval = [lower - 1.0, upper - 1.0]
        metrics[name] = {
            "preferred_direction": spec.preferred_direction,
            "candidate_over_baseline_ratio": point,
            "candidate_over_baseline_ratio_ci95": [lower, upper],
            "improvement_fraction": improvement,
            "improvement_fraction_ci95": improvement_interval,
        }
    return {
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "method": "independent_process_bootstrap_of_median_ratio_percentile",
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "metrics": metrics,
    }


def _without_keys(value: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _all_equal(values: Sequence[object]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def comparability_checks(
    records: Sequence[Mapping[str, Any]],
    *,
    mode_a: str,
    mode_b: str,
    repeats_per_mode: int,
    expected_order: Sequence[str],
    expected_output_tokens: int,
) -> dict[str, bool]:
    """Return same-cell checks with explicit decode-tuning differences only."""

    mode_names = [record["mode"]["name"] for record in records]
    sections = ("environment", "workload", "traffic", "sampling", "measurement")
    checks = {
        "record_count": len(records) == 2 * repeats_per_mode,
        "mode_counts": (
            mode_names.count(mode_a) == repeats_per_mode
            and mode_names.count(mode_b) == repeats_per_mode
        ),
        "run_order": mode_names == list(expected_order),
        **{
            f"{section}_exact": _all_equal([record[section] for record in records])
            for section in sections
        },
        "model_configuration_except_tuning_exact": _all_equal(
            [_without_keys(record["model"], MODEL_TUNING_FIELDS) for record in records]
        ),
        "mode_configuration_except_variant_exact": _all_equal(
            [_without_keys(record["mode"], MODE_VARIANT_FIELDS) for record in records]
        ),
        "execution_configuration_except_cuda_graph_exact": _all_equal(
            [
                _without_keys(record["execution_backend"], CUDA_GRAPH_EXECUTION_FIELDS)
                for record in records
            ]
        ),
        "kv_cache_exact": _all_equal([record["kv_cache"] for record in records]),
        "token_ids_exact": _all_equal([record["correctness"]["token_ids"] for record in records]),
        "decoded_texts_exact": _all_equal(
            [record["correctness"]["decoded_texts"] for record in records]
        ),
        "output_length_exact": all(
            record["correctness"]["output_tokens"] == expected_output_tokens for record in records
        ),
    }
    return checks


def _validate_child_identity(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    mode: str,
    commit: str,
    dirty: bool,
) -> None:
    if record.get("schema_version") != REQUIRED_CHILD_SCHEMA_VERSION:
        raise RuntimeError(
            f"fresh-process matrix requires benchmark schema v{REQUIRED_CHILD_SCHEMA_VERSION}"
        )
    if record.get("record_type") != "system_benchmark":
        raise RuntimeError("fresh-process child wrote the wrong record type")
    if record["mode"]["name"] != mode:
        raise RuntimeError(f"child mode mismatch: {record['mode']['name']} != {mode}")
    environment = record["environment"]
    if environment["gpu_uuid"] != args.expected_gpu_uuid:
        raise RuntimeError("child GPU UUID changed after preflight")
    if environment["git_commit"] != commit or environment["git_dirty"] != dirty:
        raise RuntimeError("child Git identity changed during the process matrix")


def _validate_child_configuration(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    model = record["model"]
    expected_model = {
        "path": str(Path(args.model).resolve()),
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "kvcache_block_size": args.kvcache_block_size,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prefix_caching_enabled": False,
        "logits_precision": record["mode"]["logits_precision"],
        "mlp_projection_mode": args.mlp_projection_mode,
        "paged_decode_block_n": record["mode"]["paged_decode_block_n"],
    }
    mismatched_model = {
        key: (model.get(key), expected)
        for key, expected in expected_model.items()
        if model.get(key) != expected
    }
    if mismatched_model:
        raise RuntimeError(f"child model contract mismatch: {mismatched_model}")
    workload = record["workload"]
    if workload["case_id"] != args.case or workload["max_tokens"] != args.max_tokens:
        raise RuntimeError("child workload identity changed")
    if record["traffic"]["batch_size"] != args.batch_size:
        raise RuntimeError("child batch size changed")
    if record["sampling"] != {
        "temperature": 0.0,
        "ignore_eos": True,
        "max_tokens": args.max_tokens,
    }:
        raise RuntimeError("child sampling contract changed")
    if record["kv_cache"]["blocks"] != args.num_kvcache_blocks:
        raise RuntimeError("child KV block count changed")
    if record["kv_cache"]["block_size"] != args.kvcache_block_size:
        raise RuntimeError("child KV block size changed")


def _validate_child_backend(record: dict[str, Any], *, args: argparse.Namespace) -> None:
    backend = record["execution_backend"]["vision_attention_backend"]
    if backend != args.vision_attention_backend:
        raise RuntimeError(f"child vision backend mismatch: {backend}")
    execution = record["mode"]["execution"]
    execution_backend = record["execution_backend"]
    if execution_backend["decode_backend"] != execution:
        raise RuntimeError("child decode backend does not match mode execution")
    graph_expected = execution == "cuda_graph"
    if execution_backend["cuda_graph_enabled"] is not graph_expected:
        raise RuntimeError("child CUDA Graph activation does not match requested mode")
    if graph_expected == (execution_backend["cuda_graph_capture_scope"] == "none"):
        raise RuntimeError("child CUDA Graph capture scope does not match activation")
    compile_expected = execution.startswith("torch_compile")
    if execution_backend["torch_compile_enabled"] is not compile_expected:
        raise RuntimeError("child torch.compile activation does not match requested mode")


def _validate_child_measurement(record: dict[str, Any], *, args: argparse.Namespace) -> None:
    if record["measurement"] != {
        "warmup": args.warmup,
        "repeat": 1,
        "cuda_synchronize_timing": True,
        "engine_ttft_scope": "sum_of_synchronized_prefill_steps",
        "decode_tpot_scope": "synchronized_engine_decode_step",
        "end_to_end_scope": "request_preprocessing_plus_engine_steps",
    }:
        raise RuntimeError("child measurement contract changed")


def _validate_child_record(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    mode: str,
    commit: str,
    dirty: bool,
) -> None:
    _validate_child_identity(
        record,
        args=args,
        mode=mode,
        commit=commit,
        dirty=dirty,
    )
    _validate_child_configuration(record, args=args)
    _validate_child_backend(record, args=args)
    _validate_child_measurement(record, args=args)


def _execute_child(
    args: argparse.Namespace,
    *,
    index: int,
    label: str,
    mode: str,
    run_dir: Path,
    commit: str,
    dirty: bool,
    child_env: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _query_gpu_state(args.expected_gpu_uuid)
    validate_idle_gpu(
        before,
        max_memory_used_mib=args.max_idle_memory_mib,
        max_utilization_percent=args.max_idle_utilization_percent,
    )
    child_output = run_dir / f"{index:02d}_{label}_{mode}.jsonl"
    child_log = child_output.with_suffix(".stderr.log")
    command = _child_command(args, mode=mode, output=child_output)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=child_env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started
    child_log.write_text(completed.stderr, encoding="utf-8")
    after = _wait_for_release(args)
    if completed.returncode != 0:
        raise RuntimeError(
            f"fresh-process child failed with code {completed.returncode}; see {child_log}"
        )
    record = _read_single_record(child_output)
    _validate_child_record(
        record,
        args=args,
        mode=mode,
        commit=commit,
        dirty=dirty,
    )
    record["protocol"].update(
        {
            "name": "p9_fresh_process_pair_v1",
            "process_scope": "fresh_process_per_mode_repeat",
            "process_index": index,
            "pair_label": label,
            "run_order": "ABBA_BAAB_truncated_balanced",
            "fresh_process_repeats_per_mode": args.fresh_process_repeats,
        }
    )
    run = {
        "index": index,
        "label": label,
        "mode": mode,
        "status": "passed",
        "returncode": completed.returncode,
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "before": asdict(before),
        "after": asdict(after),
        "record": str(child_output),
        "stderr_log": str(child_log),
    }
    return record, run


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    output, manifest_path, run_dir = _artifact_paths(Path(args.output))
    validate_artifact_destination(output)
    commit, dirty = _git_identity()
    if dirty and not args.allow_dirty:
        raise RuntimeError("formal fresh-process matrix requires a clean Git worktree")
    order = balanced_pair_order(args.fresh_process_repeats)
    mode_by_label = {"A": args.mode_a, "B": args.mode_b}
    mode_order = [mode_by_label[label] for label in order]
    planned_commands = [
        _child_command(
            args,
            mode=mode_by_label[label],
            output=run_dir / f"{index:02d}_{label}_{mode_by_label[label]}.jsonl",
        )
        for index, label in enumerate(order)
    ]

    if args.dry_run:
        return {
            "dry_run": True,
            "git_commit": commit,
            "git_dirty": dirty,
            "order": mode_order,
            "commands": planned_commands,
        }

    _ensure_artifacts_absent(output, manifest_path, run_dir)
    run_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    child_env = dict(os.environ)
    child_env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    result = {
        "schema_version": PROCESS_MATRIX_SCHEMA_VERSION,
        "record_type": "p9_fresh_process_matrix",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu_uuid": args.expected_gpu_uuid,
        "cuda_visible_devices": args.cuda_visible_devices,
        "model": str(Path(args.model).resolve()),
        "workload_manifest": str(Path(args.manifest).resolve()),
        "mode_a": args.mode_a,
        "mode_b": args.mode_b,
        "vision_attention_backend": args.vision_attention_backend,
        "case": args.case,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "warmup_per_process": args.warmup,
        "fresh_process_repeats_per_mode": args.fresh_process_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "run_order": mode_order,
        "planned_commands": planned_commands,
        "completed_runs": 0,
        "runs": runs,
        "aggregate_records": str(output),
    }
    _write_json_atomic(manifest_path, result)
    try:
        for index, label in enumerate(order):
            mode = mode_by_label[label]
            record, run = _execute_child(
                args,
                index=index,
                label=label,
                mode=mode,
                run_dir=run_dir,
                commit=commit,
                dirty=dirty,
                child_env=child_env,
            )
            records.append(record)
            runs.append(run)
            result["completed_runs"] = len(runs)
            _write_json_atomic(manifest_path, result)

        final_commit, final_dirty = _git_identity()
        if (final_commit, final_dirty) != (commit, dirty):
            raise RuntimeError("Git identity changed while the process matrix was running")
        checks = comparability_checks(
            records,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            repeats_per_mode=args.fresh_process_repeats,
            expected_order=mode_order,
            expected_output_tokens=args.max_tokens * args.batch_size,
        )
        failed_checks = [name for name, passed in checks.items() if not passed]
        _write_text_atomic(
            output,
            "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
            + "\n",
        )
        grouped = {
            mode: [record for record in records if record["mode"]["name"] == mode]
            for mode in (args.mode_a, args.mode_b)
        }
        formal_checks = {
            "clean_git": not dirty,
            "comparability_pass": not failed_checks,
            "minimum_fresh_process_repeats": (
                args.fresh_process_repeats >= MIN_FORMAL_FRESH_PROCESS_REPEATS
            ),
            "minimum_warmup": args.warmup >= MIN_FORMAL_WARMUP,
            "full_bootstrap_resamples": (args.bootstrap_resamples >= DEFAULT_BOOTSTRAP_RESAMPLES),
            "frozen_bootstrap_seed": args.bootstrap_seed == DEFAULT_BOOTSTRAP_SEED,
        }
        result.update(
            {
                "status": "completed" if not failed_checks else "failed_comparability",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "comparability_checks": checks,
                "comparability_pass": not failed_checks,
                "formal_eligibility_checks": formal_checks,
                "formal_eligible": all(formal_checks.values()),
                "token_exact_across_modes_and_processes": checks["token_ids_exact"],
                "summaries": {
                    mode: _mode_summary(mode_records) for mode, mode_records in grouped.items()
                },
                "comparison": _comparison_summary(
                    grouped[args.mode_a],
                    grouped[args.mode_b],
                    baseline_mode=args.mode_a,
                    candidate_mode=args.mode_b,
                    seed=args.bootstrap_seed,
                    resamples=args.bootstrap_resamples,
                ),
            }
        )
        _write_json_atomic(manifest_path, result)
        if failed_checks:
            raise RuntimeError(
                f"fresh-process comparability gates failed: {failed_checks}; see {manifest_path}"
            )
        return result
    except (Exception, KeyboardInterrupt) as exc:
        if result["status"] == "running":
            result.update(
                {
                    "status": "failed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            _write_json_atomic(manifest_path, result)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode-a", required=True)
    parser.add_argument("--mode-b", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fresh-process-repeats",
        type=int,
        default=DEFAULT_FRESH_PROCESS_REPEATS,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument(
        "--vision-attention-backend",
        choices=("sdpa", "flash_attn"),
        default="sdpa",
    )
    parser.add_argument(
        "--logits-precision",
        choices=("fp32", "model", "selective_fp32"),
        default="model",
    )
    parser.add_argument(
        "--paged-decode-block-n",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=32,
    )
    parser.add_argument("--mlp-projection-mode", choices=("legacy", "packed"), default="packed")
    parser.add_argument(
        "--max-idle-memory-mib",
        type=float,
        default=DEFAULT_MAX_IDLE_MEMORY_MIB,
    )
    parser.add_argument(
        "--max-idle-utilization-percent",
        type=float,
        default=DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
    )
    parser.add_argument(
        "--release-timeout-seconds",
        type=float,
        default=DEFAULT_RELEASE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode_a == args.mode_b:
        raise SystemExit("--mode-a and --mode-b must differ")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    for name in (
        "fresh_process_repeats",
        "max_tokens",
        "batch_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "num_kvcache_blocks",
        "kvcache_block_size",
        "bootstrap_resamples",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.max_num_seqs < args.batch_size:
        raise SystemExit("--max-num-seqs must cover --batch-size")
    if not args.cuda_visible_devices.strip():
        raise SystemExit("--cuda-visible-devices must be non-empty")
    if not Path(args.model).is_dir():
        raise SystemExit(f"--model is not a directory: {args.model}")
    if not Path(args.manifest).is_file():
        raise SystemExit(f"--manifest is not a file: {args.manifest}")
    result = _run_matrix(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
