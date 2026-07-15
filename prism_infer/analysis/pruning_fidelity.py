"""P6.12 dataset-level pruning fidelity 汇总。

本模块只比较同一 workload 下未压缩 baseline 和 pruning candidate 的
greedy token、physical KV 与 per-span 审计记录。输出衡量压缩保真度，
不是 COCO/VQA 任务准确率，也不运行模型或第三方评测器。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism_infer.analysis.benchmark_schema import summarize_values
from prism_infer.analysis.pareto_summary import stable_prefix_lengths


PRUNING_FIDELITY_SCHEMA_VERSION = 1


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


def _candidate_descriptor(record: Mapping[str, Any]) -> dict[str, Any]:
    mode = record["mode"]
    strategy = str(mode["visual_pruning_strategy"])
    last_n_layers = mode.get("visual_pruning_attention_last_n_layers")
    label = f"{mode['name']}/{strategy}"
    if strategy == "attention" and last_n_layers is not None:
        label += f":last{int(last_n_layers)}"
    return {
        "label": label,
        "mode": str(mode["name"]),
        "compression": str(mode["compression"]),
        "strategy": strategy,
        "keep_ratio": float(mode["visual_pruning_keep_ratio"]),
        "min_keep_tokens": int(mode["visual_pruning_min_keep_tokens"]),
        "attention_last_n_layers": (
            int(last_n_layers) if last_n_layers is not None else None
        ),
    }


def _candidate_identity(descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        descriptor["mode"],
        descriptor["compression"],
        descriptor["strategy"],
        descriptor["keep_ratio"],
        descriptor["min_keep_tokens"],
        descriptor["attention_last_n_layers"],
    )


def _read_path(record: Mapping[str, Any], path: tuple[str, str]) -> Any:
    section, key = path
    return record[section][key]


def _assert_comparable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """检查与 pruning fidelity 直接相关的模型、输入和执行条件。"""

    paths = (
        ("model", "path"),
        ("model", "dtype"),
        ("model", "tensor_parallel_size"),
        ("model", "max_model_len"),
        ("model", "max_num_batched_tokens"),
        ("model", "kvcache_block_size"),
        ("model", "num_kvcache_blocks"),
        ("traffic", "kind"),
        ("traffic", "batch_size"),
        ("traffic", "concurrency"),
        ("workload", "prompt_tokens"),
        ("workload", "image_tokens"),
        ("workload", "video_tokens"),
        ("workload", "request_types"),
        ("workload", "input_shapes"),
        ("mode", "execution"),
    )
    mismatches = [
        f"{section}.{key}"
        for section, key in paths
        if _read_path(baseline, (section, key))
        != _read_path(candidate, (section, key))
    ]
    if baseline["model"].get("prefix_caching_enabled") != candidate["model"].get(
        "prefix_caching_enabled"
    ):
        mismatches.append("model.prefix_caching_enabled")
    if mismatches:
        raise ValueError(
            "baseline and candidate are not fidelity-comparable; mismatched fields: "
            f"{mismatches}"
        )


def _require_physical_kv(record: Mapping[str, Any]) -> Mapping[str, Any]:
    kv_cache = record["kv_cache"]
    required = (
        "logical_prompt_tokens",
        "physical_prompt_tokens",
        "active_prompt_bytes",
        "layouts",
    )
    missing = [key for key in required if key not in kv_cache]
    if missing:
        raise ValueError(
            "pruning fidelity requires schema-v4 physical KV evidence; missing "
            f"{missing}"
        )
    return kv_cache


def _span_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    """校验并汇总 candidate 的 per-span retention audit。"""

    kv_cache = _require_physical_kv(record)
    layouts = kv_cache["layouts"]
    if not isinstance(layouts, list):
        raise ValueError("record.kv_cache.layouts must be a list")

    visual_records = 0
    audited_records = 0
    total_visual_spans = 0
    audited_visual_spans = 0
    zero_kept_visual_spans = 0
    keep_ratios: list[float] = []
    for layout_index, layout in enumerate(layouts):
        if not isinstance(layout, Mapping):
            raise ValueError(f"KV layout {layout_index} must be an object")
        decision = layout.get("compression_record")
        if not isinstance(decision, Mapping):
            continue
        spans = decision.get("visual_token_spans")
        if not isinstance(spans, list) or not spans:
            continue
        visual_records += 1
        total_visual_spans += len(spans)
        kept_by_span = decision.get("kept_visual_tokens_by_span")
        if kept_by_span is None:
            continue
        if not isinstance(kept_by_span, list):
            raise ValueError(
                f"layout {layout_index} kept_visual_tokens_by_span must be a list"
            )

        span_by_key: dict[tuple[str, int], int] = {}
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError(f"layout {layout_index} visual span must be an object")
            modality = span.get("modality")
            span_index = span.get("index")
            token_count = span.get("token_count")
            if (
                not isinstance(modality, str)
                or isinstance(span_index, bool)
                or not isinstance(span_index, int)
                or isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 1
            ):
                raise ValueError(f"layout {layout_index} has invalid visual span")
            key = (modality, span_index)
            if key in span_by_key:
                raise ValueError(f"layout {layout_index} has duplicate visual span {key}")
            span_by_key[key] = token_count

        audit_by_key: dict[tuple[str, int], int] = {}
        for audit in kept_by_span:
            if not isinstance(audit, Mapping):
                raise ValueError(f"layout {layout_index} span audit must be an object")
            modality = audit.get("modality")
            span_index = audit.get("span_index")
            kept_tokens = audit.get("kept_tokens")
            if (
                not isinstance(modality, str)
                or isinstance(span_index, bool)
                or not isinstance(span_index, int)
                or isinstance(kept_tokens, bool)
                or not isinstance(kept_tokens, int)
                or kept_tokens < 0
            ):
                raise ValueError(f"layout {layout_index} has invalid span audit")
            key = (modality, span_index)
            if key in audit_by_key:
                raise ValueError(f"layout {layout_index} has duplicate span audit {key}")
            audit_by_key[key] = kept_tokens

        if set(audit_by_key) != set(span_by_key):
            raise ValueError(
                f"layout {layout_index} span audit keys do not match visual spans"
            )
        for key, kept_tokens in audit_by_key.items():
            token_count = span_by_key[key]
            if kept_tokens > token_count:
                raise ValueError(
                    f"layout {layout_index} span {key} keeps {kept_tokens}/{token_count}"
                )
            keep_ratios.append(kept_tokens / token_count)
            zero_kept_visual_spans += int(kept_tokens == 0)
        if sum(audit_by_key.values()) != decision.get("kept_visual_tokens"):
            raise ValueError(
                f"layout {layout_index} span audit sum does not match kept_visual_tokens"
            )
        audited_records += 1
        audited_visual_spans += len(audit_by_key)

    return {
        "available": visual_records > 0 and visual_records == audited_records,
        "visual_records": visual_records,
        "audited_records": audited_records,
        "total_visual_spans": total_visual_spans,
        "audited_visual_spans": audited_visual_spans,
        "zero_kept_visual_spans": zero_kept_visual_spans,
        "minimum_span_keep_ratio": min(keep_ratios) if keep_ratios else None,
    }


def _case_row(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_comparable(baseline, candidate)
    baseline_kv = _require_physical_kv(baseline)
    candidate_kv = _require_physical_kv(candidate)
    baseline_tokens = baseline["correctness"]["token_ids"]
    candidate_tokens = candidate["correctness"]["token_ids"]
    prefix_tokens = stable_prefix_lengths(baseline_tokens, candidate_tokens)
    baseline_lengths = [len(tokens) for tokens in baseline_tokens]
    if any(length < 1 for length in baseline_lengths):
        raise ValueError("pruning fidelity requires non-empty baseline output per request")
    prefix_ratios = [
        prefix / baseline_length
        for prefix, baseline_length in zip(
            prefix_tokens,
            baseline_lengths,
            strict=True,
        )
    ]
    exact_per_request = [
        baseline_request == candidate_request
        for baseline_request, candidate_request in zip(
            baseline_tokens,
            candidate_tokens,
            strict=True,
        )
    ]
    baseline_physical_tokens = int(baseline_kv["physical_prompt_tokens"])
    baseline_active_bytes = int(baseline_kv["active_prompt_bytes"])
    if baseline_physical_tokens < 1 or baseline_active_bytes < 1:
        raise ValueError("baseline physical KV denominators must be positive")
    workload = candidate["workload"]
    return {
        "manifest_name": workload["manifest_name"],
        "manifest_sha256": workload["manifest_sha256"],
        "case_id": workload["case_id"],
        "num_requests": workload["num_requests"],
        "max_tokens": workload["max_tokens"],
        "baseline_mode": baseline["mode"]["name"],
        "candidate": dict(descriptor),
        "baseline_output_lengths": baseline_lengths,
        "stable_prefix_tokens": prefix_tokens,
        "stable_prefix_ratios": prefix_ratios,
        "stable_prefix_ratio_micro": sum(prefix_tokens) / sum(baseline_lengths),
        "exact_per_request": exact_per_request,
        "exact_request_rate": sum(exact_per_request) / len(exact_per_request),
        "token_exact": all(exact_per_request),
        "baseline_physical_prompt_tokens": baseline_physical_tokens,
        "candidate_physical_prompt_tokens": int(
            candidate_kv["physical_prompt_tokens"]
        ),
        "physical_token_ratio": (
            int(candidate_kv["physical_prompt_tokens"])
            / baseline_physical_tokens
        ),
        "baseline_active_prompt_bytes": baseline_active_bytes,
        "candidate_active_prompt_bytes": int(candidate_kv["active_prompt_bytes"]),
        "active_prompt_bytes_ratio": (
            int(candidate_kv["active_prompt_bytes"]) / baseline_active_bytes
        ),
        "span_audit": _span_audit(candidate),
        "baseline_git_commit": baseline["environment"]["git_commit"],
        "baseline_git_dirty": baseline["environment"]["git_dirty"],
        "candidate_git_commit": candidate["environment"]["git_commit"],
        "candidate_git_dirty": candidate["environment"]["git_dirty"],
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        descriptor = row["candidate"]
        key = (
            row["manifest_sha256"],
            *_candidate_identity(descriptor),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        first = group_rows[0]
        prefix_tokens = [
            int(value)
            for row in group_rows
            for value in row["stable_prefix_tokens"]
        ]
        prefix_ratios = [
            float(value)
            for row in group_rows
            for value in row["stable_prefix_ratios"]
        ]
        exact_values = [
            bool(value)
            for row in group_rows
            for value in row["exact_per_request"]
        ]
        baseline_output_tokens = sum(
            sum(int(value) for value in row["baseline_output_lengths"])
            for row in group_rows
        )
        baseline_physical_tokens = sum(
            int(row["baseline_physical_prompt_tokens"])
            for row in group_rows
        )
        baseline_active_bytes = sum(
            int(row["baseline_active_prompt_bytes"])
            for row in group_rows
        )
        if (
            baseline_output_tokens < 1
            or baseline_physical_tokens < 1
            or baseline_active_bytes < 1
        ):
            raise RuntimeError("aggregate baseline denominators must be positive")
        span_audits = [row["span_audit"] for row in group_rows]
        total_visual_records = sum(
            int(audit["visual_records"]) for audit in span_audits
        )
        audited_visual_records = sum(
            int(audit["audited_records"]) for audit in span_audits
        )
        minimum_span_ratios = [
            float(audit["minimum_span_keep_ratio"])
            for audit in span_audits
            if audit["minimum_span_keep_ratio"] is not None
        ]
        aggregates.append(
            {
                "manifest_name": first["manifest_name"],
                "manifest_sha256": first["manifest_sha256"],
                "baseline_mode": first["baseline_mode"],
                "candidate": dict(first["candidate"]),
                "case_count": len(group_rows),
                "request_count": len(exact_values),
                "case_ids": sorted(str(row["case_id"]) for row in group_rows),
                "exact_request_rate": sum(exact_values) / len(exact_values),
                "exact_case_rate": (
                    sum(bool(row["token_exact"]) for row in group_rows)
                    / len(group_rows)
                ),
                "stable_prefix_tokens": summarize_values(prefix_tokens),
                "stable_prefix_ratios": summarize_values(prefix_ratios),
                "stable_prefix_ratio_micro": (
                    sum(prefix_tokens) / baseline_output_tokens
                ),
                "physical_token_ratio": (
                    sum(
                        int(row["candidate_physical_prompt_tokens"])
                        for row in group_rows
                    )
                    / baseline_physical_tokens
                ),
                "active_prompt_bytes_ratio": (
                    sum(
                        int(row["candidate_active_prompt_bytes"])
                        for row in group_rows
                    )
                    / baseline_active_bytes
                ),
                "span_audit": {
                    "available": (
                        total_visual_records > 0
                        and total_visual_records == audited_visual_records
                    ),
                    "visual_records": total_visual_records,
                    "audited_records": audited_visual_records,
                    "total_visual_spans": sum(
                        int(audit["total_visual_spans"])
                        for audit in span_audits
                    ),
                    "audited_visual_spans": sum(
                        int(audit["audited_visual_spans"])
                        for audit in span_audits
                    ),
                    "zero_kept_visual_spans": sum(
                        int(audit["zero_kept_visual_spans"])
                        for audit in span_audits
                    ),
                    "minimum_span_keep_ratio": (
                        min(minimum_span_ratios) if minimum_span_ratios else None
                    ),
                },
                "baseline_git_commits": sorted(
                    {str(row["baseline_git_commit"]) for row in group_rows}
                ),
                "candidate_git_commits": sorted(
                    {str(row["candidate_git_commit"]) for row in group_rows}
                ),
                "all_baselines_clean": not any(
                    bool(row["baseline_git_dirty"]) for row in group_rows
                ),
                "all_candidates_clean": not any(
                    bool(row["candidate_git_dirty"]) for row in group_rows
                ),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (row["manifest_name"], row["candidate"]["label"]),
    )


def summarize_pruning_fidelity_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str = "off_graph",
) -> dict[str, Any]:
    """生成 case-level 和 dataset-level pruning fidelity 汇总。"""

    if not records:
        raise ValueError("pruning fidelity requires at least one benchmark record")
    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    candidates: list[Mapping[str, Any]] = []
    seen_candidates: set[tuple[tuple[Any, ...], tuple[Any, ...]]] = set()
    for record in records:
        key = _comparison_key(record)
        if record["mode"]["name"] == baseline_mode:
            if key in baselines:
                raise ValueError(f"duplicate {baseline_mode!r} baseline for key={key}")
            baselines[key] = record
            continue
        descriptor = _candidate_descriptor(record)
        identity = (key, _candidate_identity(descriptor))
        if identity in seen_candidates:
            raise ValueError(
                "duplicate pruning fidelity candidate for "
                f"key={key}, candidate={descriptor['label']!r}"
            )
        seen_candidates.add(identity)
        candidates.append(record)
    if not baselines:
        raise ValueError(f"no {baseline_mode!r} baseline records were provided")
    if not candidates:
        raise ValueError("no pruning candidate records were provided")

    rows: list[dict[str, Any]] = []
    candidate_coverage: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    for candidate in candidates:
        key = _comparison_key(candidate)
        if key not in baselines:
            raise ValueError(f"missing {baseline_mode!r} baseline for key={key}")
        descriptor = _candidate_descriptor(candidate)
        coverage_key = (
            key[0],
            key[4],
            *_candidate_identity(descriptor),
        )
        candidate_coverage.setdefault(coverage_key, set()).add(key)
        rows.append(_case_row(baselines[key], candidate, descriptor))

    for coverage_key, actual_keys in candidate_coverage.items():
        manifest_sha256, keep_ratio = coverage_key[:2]
        expected_keys = {
            key
            for key in baselines
            if key[0] == manifest_sha256 and key[4] == keep_ratio
        }
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            raise ValueError(
                "candidate does not cover every selected baseline case: "
                f"missing={missing}"
            )

    rows.sort(
        key=lambda row: (
            row["manifest_name"],
            row["candidate"]["label"],
            row["case_id"],
        )
    )
    return {
        "schema_version": PRUNING_FIDELITY_SCHEMA_VERSION,
        "summary_type": "pruning_fidelity",
        "baseline_mode": baseline_mode,
        "limitations": [
            "Measures greedy-token fidelity to an uncompressed baseline, not task accuracy.",
            "Stable-prefix agreement does not measure semantic equivalence after divergence.",
        ],
        "aggregates": _aggregate_rows(rows),
        "cases": rows,
    }


def render_pruning_fidelity_markdown(summary: Mapping[str, Any]) -> str:
    """把 pruning fidelity 汇总渲染为可审计 Markdown。"""

    lines = [
        "# P6.12 Pruning Fidelity Summary",
        "",
        "> Greedy-token fidelity to the uncompressed baseline; not task accuracy.",
        "",
        "## Dataset Aggregates",
        "",
        "| Manifest | Candidate | Cases | Requests | Exact requests | Prefix micro | "
        "Prefix min | Physical KV | Active bytes | Zero spans | Span audit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for aggregate in summary["aggregates"]:
        span_audit = aggregate["span_audit"]
        lines.append(
            f"| {aggregate['manifest_name']} | {aggregate['candidate']['label']} | "
            f"{aggregate['case_count']} | {aggregate['request_count']} | "
            f"{aggregate['exact_request_rate']:.3f} | "
            f"{aggregate['stable_prefix_ratio_micro']:.3f} | "
            f"{aggregate['stable_prefix_ratios']['min']:.3f} | "
            f"{aggregate['physical_token_ratio']:.3f}x | "
            f"{aggregate['active_prompt_bytes_ratio']:.3f}x | "
            f"{span_audit['zero_kept_visual_spans']} | "
            f"{'yes' if span_audit['available'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Candidate | Requests | Stable prefixes | Prefix ratios | Exact | "
            "Physical KV | Active bytes |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["cases"]:
        ratios = ", ".join(f"{value:.3f}" for value in row["stable_prefix_ratios"])
        lines.append(
            f"| {row['case_id']} | {row['candidate']['label']} | "
            f"{row['num_requests']} | {row['stable_prefix_tokens']} | [{ratios}] | "
            f"{row['exact_request_rate']:.3f} | {row['physical_token_ratio']:.3f}x | "
            f"{row['active_prompt_bytes_ratio']:.3f}x |"
        )
    return "\n".join(lines) + "\n"
