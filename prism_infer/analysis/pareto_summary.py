"""P6 quality-memory-TPOT Pareto 记录汇总。

本模块只读取并校验 Prism benchmark schema，不执行模型。每个压缩记录都与同一
manifest、case、batch、输出长度和 keep ratio 下的明确 baseline 比较。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism_infer.analysis.benchmark_schema import validate_benchmark_record


def load_benchmark_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """读取一个或多个 benchmark JSONL，并拒绝空文件或非法记录。"""

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        file_records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_benchmark_record(record)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
                records.append(record)
                file_records += 1
        if file_records == 0:
            raise ValueError(f"benchmark JSONL has no records: {path}")
    return records


def stable_prefix_lengths(
    baseline: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
) -> list[int]:
    """返回每条请求与 baseline 从首 token 起连续相同的长度。"""

    if len(baseline) != len(candidate):
        raise ValueError(
            "baseline and candidate request counts differ: "
            f"{len(baseline)} != {len(candidate)}"
        )
    lengths: list[int] = []
    for baseline_tokens, candidate_tokens in zip(baseline, candidate, strict=True):
        prefix = 0
        for baseline_token, candidate_token in zip(
            baseline_tokens,
            candidate_tokens,
            strict=False,
        ):
            if baseline_token != candidate_token:
                break
            prefix += 1
        lengths.append(prefix)
    return lengths


def _comparison_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    workload = record["workload"]
    mode = record["mode"]
    return (
        workload["manifest_sha256"],
        workload["case_id"],
        workload["num_requests"],
        workload["max_tokens"],
        mode["visual_pruning_keep_ratio"],
    )


def summarize_pareto_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str = "off_eager",
) -> list[dict[str, Any]]:
    """按 benchmark cell 生成质量、物理 KV 和 TPOT 可比较行。"""

    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    seen_modes: set[tuple[tuple[Any, ...], str]] = set()
    for record in records:
        key = _comparison_key(record)
        mode_name = record["mode"]["name"]
        identity = (key, mode_name)
        if identity in seen_modes:
            raise ValueError(f"duplicate Pareto cell for key={key}, mode={mode_name!r}")
        seen_modes.add(identity)
        if mode_name == baseline_mode:
            baselines[key] = record

    rows: list[dict[str, Any]] = []
    for record in records:
        key = _comparison_key(record)
        if key not in baselines:
            raise ValueError(f"missing {baseline_mode!r} baseline for key={key}")
        baseline = baselines[key]
        baseline_tokens = baseline["correctness"]["token_ids"]
        candidate_tokens = record["correctness"]["token_ids"]
        prefix_lengths = stable_prefix_lengths(baseline_tokens, candidate_tokens)

        baseline_kv = baseline["kv_cache"]
        candidate_kv = record["kv_cache"]
        baseline_active_bytes = baseline_kv["active_prompt_bytes"]
        baseline_physical_tokens = baseline_kv["physical_prompt_tokens"]
        baseline_tpot = baseline["timing_ms"]["decode_step"]["median"]
        if baseline_active_bytes <= 0 or baseline_physical_tokens <= 0 or baseline_tpot <= 0:
            raise ValueError(f"baseline has non-positive comparison denominator: key={key}")

        workload = record["workload"]
        mode = record["mode"]
        rows.append(
            {
                "manifest_name": workload["manifest_name"],
                "manifest_sha256": workload["manifest_sha256"],
                "case_id": workload["case_id"],
                "num_requests": workload["num_requests"],
                "max_tokens": workload["max_tokens"],
                "keep_ratio": mode["visual_pruning_keep_ratio"],
                "mode": mode["name"],
                "logical_prompt_tokens": candidate_kv["logical_prompt_tokens"],
                "physical_prompt_tokens": candidate_kv["physical_prompt_tokens"],
                "physical_token_ratio": (
                    candidate_kv["physical_prompt_tokens"] / baseline_physical_tokens
                ),
                "active_prompt_blocks": candidate_kv["active_prompt_blocks"],
                "active_prompt_bytes": candidate_kv["active_prompt_bytes"],
                "active_prompt_bytes_ratio": (
                    candidate_kv["active_prompt_bytes"] / baseline_active_bytes
                ),
                "stable_prefix_per_request": prefix_lengths,
                "stable_prefix_min": min(prefix_lengths),
                "token_exact": candidate_tokens == baseline_tokens,
                "tpot_median_ms": record["timing_ms"]["decode_step"]["median"],
                "tpot_ratio": (
                    record["timing_ms"]["decode_step"]["median"] / baseline_tpot
                ),
                "git_commit": record["environment"]["git_commit"],
                "git_dirty": record["environment"]["git_dirty"],
            }
        )
    return rows


def render_pareto_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染适合报告引用的紧凑 Markdown 表格。"""

    header = (
        "| Case | Out | Keep | Mode | Physical KV | Blocks | Active bytes | "
        "Stable prefix | Exact | TPOT |\n"
        "|---|---:|---:|---|---:|---:|---:|---:|:---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['max_tokens']} | {row['keep_ratio']:.2f} "
            f"| {row['mode']} | {row['physical_prompt_tokens']} "
            f"({row['physical_token_ratio']:.3f}x) | {row['active_prompt_blocks']} "
            f"| {row['active_prompt_bytes_ratio']:.3f}x | "
            f"{row['stable_prefix_per_request']} | "
            f"{'yes' if row['token_exact'] else 'no'} | {row['tpot_ratio']:.3f}x |"
        )
    return "\n".join(lines) + "\n"
