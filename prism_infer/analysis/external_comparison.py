"""P6 外部框架 offline benchmark 校验与 Prism baseline 对比。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism_infer.analysis.benchmark_schema import validate_benchmark_record
from prism_infer.analysis.pareto_summary import stable_prefix_lengths


def validate_external_record(record: Mapping[str, Any]) -> None:
    """校验 P6 external benchmark 的关键证据字段。"""

    if record.get("schema_version") != 1:
        raise ValueError("external record schema_version must be 1")
    if record.get("record_type") != "external_system_benchmark":
        raise ValueError("external record_type must be external_system_benchmark")
    for section in (
        "environment",
        "model",
        "backend",
        "workload",
        "measurement",
        "correctness",
        "timing_ms",
        "throughput",
        "memory_mb",
    ):
        if not isinstance(record.get(section), Mapping):
            raise ValueError(f"external record missing object section: {section}")
    environment = record["environment"]
    for key in ("framework", "framework_version", "framework_source_commit", "gpu"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"external environment.{key} must be a non-empty string")
    measurement = record["measurement"]
    if measurement.get("repeat", 0) < 1 or measurement.get("warmup", -1) < 0:
        raise ValueError("external warmup/repeat are invalid")
    if measurement.get("cuda_synchronize_timing") is not True:
        raise ValueError("external benchmark must use CUDA synchronized timing")
    correctness = record["correctness"]
    if correctness.get("outputs_identical_across_repeats") is not True:
        raise ValueError("external greedy outputs are not stable across repeats")
    token_ids = correctness.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("external correctness.token_ids must be non-empty")
    for path in (
        ("timing_ms", "end_to_end"),
        ("timing_ms", "engine_ttft"),
        ("timing_ms", "decode_tpot"),
        ("throughput", "e2e_output_tokens_per_s"),
    ):
        summary = record[path[0]].get(path[1])
        if not isinstance(summary, Mapping) or summary.get("count", 0) < 1:
            raise ValueError(f"external {path[0]}.{path[1]} is invalid")
    memory = record["memory_mb"]
    memory_summary = memory.get("peak_allocated", memory.get("process_used"))
    if not isinstance(memory_summary, Mapping) or memory_summary.get("count", 0) < 1:
        raise ValueError("external memory evidence is invalid")


def load_external_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """读取并校验外部框架 JSON records。"""

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_external_record(record)
        records.append(record)
    return records


def _key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    workload = record["workload"]
    return (
        workload["manifest_sha256"],
        workload["case_id"],
        workload["num_requests"],
        workload["max_tokens"],
    )


def compare_external_records(
    prism_records: Sequence[Mapping[str, Any]],
    external_records: Sequence[Mapping[str, Any]],
    *,
    prism_mode: str = "off_eager",
    prism_keep_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """将 external records 与唯一匹配的 Prism benchmark cell 比较。"""

    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in prism_records:
        validate_benchmark_record(record)
        if record["mode"]["name"] != prism_mode:
            continue
        if record["mode"]["visual_pruning_keep_ratio"] != prism_keep_ratio:
            continue
        key = _key(record)
        if key in baselines:
            raise ValueError(f"duplicate Prism external-comparison baseline: {key}")
        baselines[key] = record

    rows: list[dict[str, Any]] = []
    for external in external_records:
        validate_external_record(external)
        key = _key(external)
        if key not in baselines:
            raise ValueError(f"missing matching Prism baseline for external key={key}")
        prism = baselines[key]
        prism_tokens = prism["correctness"]["token_ids"]
        external_tokens = external["correctness"]["token_ids"]
        prefixes = stable_prefix_lengths(prism_tokens, external_tokens)
        prism_tpot = prism["timing_ms"]["decode_step"]["median"]
        external_tpot = external["timing_ms"]["decode_tpot"]["median"]
        prism_throughput = prism["throughput"]["e2e_output_tokens_per_s"]["median"]
        external_throughput = external["throughput"]["e2e_output_tokens_per_s"]["median"]
        prism_peak = prism["memory_mb"]["peak_allocated"]["median"]
        external_memory = external["memory_mb"]
        external_peak = external_memory.get(
            "peak_allocated", external_memory.get("process_used")
        )["median"]
        prism_memory_measurement = prism["memory_mb"].get(
            "measurement", "torch_cuda_allocator"
        )
        external_memory_measurement = external["memory_mb"].get(
            "measurement", "torch_cuda_allocator"
        )
        memory_comparable = prism_memory_measurement == external_memory_measurement
        if min(prism_tpot, prism_throughput, prism_peak) <= 0:
            raise ValueError(f"Prism baseline has a non-positive denominator: {key}")
        input_token_exact = (
            prism["workload"]["prompt_tokens"]
            == external["workload"]["prompt_tokens"]
        )
        rows.append(
            {
                "case_id": external["workload"]["case_id"],
                "num_requests": external["workload"]["num_requests"],
                "max_tokens": external["workload"]["max_tokens"],
                "prompt_tokens_prism": prism["workload"]["prompt_tokens"],
                "prompt_tokens_external": external["workload"]["prompt_tokens"],
                "input_token_exact": input_token_exact,
                "performance_comparable": input_token_exact,
                "framework": external["environment"]["framework"],
                "framework_version": external["environment"]["framework_version"],
                "framework_source_commit": external["environment"][
                    "framework_source_commit"
                ],
                "stable_prefix_per_request": prefixes,
                "token_exact": prism_tokens == external_tokens,
                "prism_tpot_median_ms": prism_tpot,
                "external_tpot_median_ms": external_tpot,
                "external_to_prism_tpot_ratio": (
                    external_tpot / prism_tpot if input_token_exact else None
                ),
                "prism_e2e_output_tokens_per_s": prism_throughput,
                "external_e2e_output_tokens_per_s": external_throughput,
                "external_to_prism_throughput_ratio": (
                    external_throughput / prism_throughput
                    if input_token_exact
                    else None
                ),
                "prism_peak_allocated_mib": prism_peak,
                "external_peak_allocated_mib": external_peak,
                "memory_comparable": memory_comparable,
                "prism_memory_measurement": prism_memory_measurement,
                "external_memory_measurement": external_memory_measurement,
                "external_to_prism_peak_memory_ratio": (
                    external_peak / prism_peak
                    if input_token_exact and memory_comparable
                    else None
                ),
            }
        )
    return rows


def render_external_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染 external vs Prism 紧凑表格。"""

    lines = [
        "| Framework | Case | Prompt P/E | Comparable | Stable prefix | Exact | TPOT P/E | E/P TPOT | "
        "Throughput P/E | E/P throughput | Memory MiB P/E | Memory comparable |",
        "|---|---|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        tpot_ratio = (
            f"{row['external_to_prism_tpot_ratio']:.3f}x"
            if row["performance_comparable"]
            else "n/a"
        )
        throughput_ratio = (
            f"{row['external_to_prism_throughput_ratio']:.3f}x"
            if row["performance_comparable"]
            else "n/a"
        )
        lines.append(
            f"| {row['framework']} {row['framework_version']} | {row['case_id']} | "
            f"{row['prompt_tokens_prism']}/"
            f"{row['prompt_tokens_external']} | "
            f"{'yes' if row['performance_comparable'] else 'no'} | "
            f"{row['stable_prefix_per_request']} | "
            f"{'yes' if row['token_exact'] else 'no'} | "
            f"{row['prism_tpot_median_ms']:.3f}/{row['external_tpot_median_ms']:.3f} | "
            f"{tpot_ratio} | "
            f"{row['prism_e2e_output_tokens_per_s']:.3f}/"
            f"{row['external_e2e_output_tokens_per_s']:.3f} | "
            f"{throughput_ratio} | "
            f"{row['prism_peak_allocated_mib']:.1f}/"
            f"{row['external_peak_allocated_mib']:.1f} | "
            f"{'yes' if row['memory_comparable'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"
