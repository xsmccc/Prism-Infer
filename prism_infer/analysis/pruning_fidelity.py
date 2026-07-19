"""P6.12 dataset-level pruning fidelity 汇总。

本模块只比较同一 workload 下未压缩 baseline 和 pruning candidate 的
greedy token、physical KV 与 per-span 审计记录。输出衡量压缩保真度，
不是 COCO/VQA 任务准确率，也不运行模型或第三方评测器。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from prism_infer.analysis.benchmark_schema import summarize_values
from prism_infer.analysis.pareto_summary import stable_prefix_lengths
from prism_infer.analysis.reference_quality import score_reference_batch


PRUNING_FIDELITY_SCHEMA_VERSION = 2
TASK_QUALITY_METRICS = ("token_f1", "rouge_l_f1")
PRUNING_FIDELITY_LIMITATIONS = (
    "Measures greedy-token fidelity to an uncompressed baseline, not task accuracy.",
    "Stable-prefix agreement does not measure semantic equivalence after divergence.",
    "Reference token-F1/ROUGE-L are lexical preflight metrics, not COCO CIDEr/SPICE.",
)
ComparisonKey = tuple[Any, ...]
CandidateIdentity = tuple[Any, ...]
CoverageKey = tuple[Any, ...]
SpanKey = tuple[str, int]


@dataclass(frozen=True)
class _LayoutSpanAudit:
    """Validated retention evidence for one visual KV layout."""

    visual_spans: int
    audited_spans: int
    zero_kept_spans: int
    keep_ratios: tuple[float, ...]

    @property
    def audited(self) -> bool:
        return self.audited_spans == self.visual_spans


@dataclass(frozen=True)
class _IndexedCandidate:
    """Candidate record with its validated comparison identities."""

    record: Mapping[str, Any]
    key: ComparisonKey
    descriptor: Mapping[str, Any]


def _comparison_key(record: Mapping[str, Any]) -> ComparisonKey:
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
        "attention_last_n_layers": (int(last_n_layers) if last_n_layers is not None else None),
    }


def _candidate_identity(descriptor: Mapping[str, Any]) -> CandidateIdentity:
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
        if _read_path(baseline, (section, key)) != _read_path(candidate, (section, key))
    ]
    if baseline["model"].get("prefix_caching_enabled") != candidate["model"].get(
        "prefix_caching_enabled"
    ):
        mismatches.append("model.prefix_caching_enabled")
    for key in ("reference_sources", "task_references"):
        if baseline["workload"].get(key) != candidate["workload"].get(key):
            mismatches.append(f"workload.{key}")
    if mismatches:
        raise ValueError(
            f"baseline and candidate are not fidelity-comparable; mismatched fields: {mismatches}"
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
            f"pruning fidelity requires schema-v4+ physical KV evidence; missing {missing}"
        )
    return kv_cache


def _task_quality_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """计算 baseline/candidate 相对同一任务 reference 的 lexical quality。"""

    task_references = candidate["workload"].get("task_references")
    baseline_batch = score_reference_batch(
        baseline["correctness"].get("decoded_texts"),
        task_references,
    )
    candidate_batch = score_reference_batch(
        candidate["correctness"].get("decoded_texts"),
        task_references,
    )
    if baseline_batch["available"] != candidate_batch["available"]:
        raise ValueError("baseline and candidate task-quality evidence availability differs")
    if not baseline_batch["available"]:
        return {
            "available": False,
            "reason": baseline_batch["reason"],
            "request_count": baseline_batch["request_count"],
        }

    baseline_scores = baseline_batch["scores"]
    candidate_scores = candidate_batch["scores"]
    identities = ("task", "reference_source", "image_id", "reference_count")
    for index, (baseline_score, candidate_score) in enumerate(
        zip(baseline_scores, candidate_scores, strict=True)
    ):
        mismatches = [key for key in identities if baseline_score[key] != candidate_score[key]]
        if mismatches:
            raise ValueError(f"request {index} task-quality identity mismatch: {mismatches}")
    return {
        "available": True,
        "reason": None,
        "request_count": len(baseline_scores),
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "token_f1_delta": [
            candidate_score["token_f1"] - baseline_score["token_f1"]
            for baseline_score, candidate_score in zip(
                baseline_scores,
                candidate_scores,
                strict=True,
            )
        ],
        "rouge_l_f1_delta": [
            candidate_score["rouge_l_f1"] - baseline_score["rouge_l_f1"]
            for baseline_score, candidate_score in zip(
                baseline_scores,
                candidate_scores,
                strict=True,
            )
        ],
    }


def _aggregate_task_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_task_quality_drop: float,
) -> dict[str, Any]:
    """汇总 lexical reference quality，并按 preflight 阈值给出门禁。"""

    unavailable = [
        str(row["task_quality"]["reason"]) for row in rows if not row["task_quality"]["available"]
    ]
    if unavailable:
        return {
            "available": False,
            "reason": "; ".join(sorted(set(unavailable))),
            "request_count": 0,
            "gate": {
                "eligible": False,
                "passed": False,
                "max_macro_score_drop": max_task_quality_drop,
                "failures": ["reference task evidence is incomplete"],
            },
        }

    baseline_scores = [score for row in rows for score in row["task_quality"]["baseline_scores"]]
    candidate_scores = [score for row in rows for score in row["task_quality"]["candidate_scores"]]
    result: dict[str, Any] = {
        "available": True,
        "reason": None,
        "request_count": len(baseline_scores),
        "tasks": sorted({str(score["task"]) for score in baseline_scores}),
        "reference_sources": sorted({str(score["reference_source"]) for score in baseline_scores}),
        "reference_counts": summarize_values(
            [int(score["reference_count"]) for score in baseline_scores]
        ),
    }
    failures: list[str] = []
    for metric in TASK_QUALITY_METRICS:
        baseline_values = [float(score[metric]) for score in baseline_scores]
        candidate_values = [float(score[metric]) for score in candidate_scores]
        baseline_macro = fmean(baseline_values)
        candidate_macro = fmean(candidate_values)
        macro_delta = candidate_macro - baseline_macro
        if baseline_macro <= 0.0:
            failures.append(f"{metric} baseline macro score is zero")
            retention = None
        else:
            retention = candidate_macro / baseline_macro
        if macro_delta < -max_task_quality_drop:
            failures.append(
                f"{metric} macro drop {-macro_delta:.6f} exceeds {max_task_quality_drop:.6f}"
            )
        result[metric] = {
            "baseline": summarize_values(baseline_values),
            "candidate": summarize_values(candidate_values),
            "baseline_macro": baseline_macro,
            "candidate_macro": candidate_macro,
            "macro_delta": macro_delta,
            "retention": retention,
        }
    result["gate"] = {
        "eligible": True,
        "passed": not failures,
        "max_macro_score_drop": max_task_quality_drop,
        "failures": failures,
    }
    return result


def _visual_span_entry(
    span: object,
    *,
    layout_index: int,
) -> tuple[SpanKey, int]:
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
    return (modality, span_index), token_count


def _span_retention_entry(
    audit: object,
    *,
    layout_index: int,
) -> tuple[SpanKey, int]:
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
    return (modality, span_index), kept_tokens


def _index_visual_spans(
    spans: Sequence[object],
    *,
    layout_index: int,
) -> dict[SpanKey, int]:
    indexed: dict[SpanKey, int] = {}
    for span in spans:
        key, token_count = _visual_span_entry(span, layout_index=layout_index)
        if key in indexed:
            raise ValueError(f"layout {layout_index} has duplicate visual span {key}")
        indexed[key] = token_count
    return indexed


def _index_span_retentions(
    audits: Sequence[object],
    *,
    layout_index: int,
) -> dict[SpanKey, int]:
    indexed: dict[SpanKey, int] = {}
    for audit in audits:
        key, kept_tokens = _span_retention_entry(audit, layout_index=layout_index)
        if key in indexed:
            raise ValueError(f"layout {layout_index} has duplicate span audit {key}")
        indexed[key] = kept_tokens
    return indexed


def _validate_span_retentions(
    span_by_key: Mapping[SpanKey, int],
    audit_by_key: Mapping[SpanKey, int],
    *,
    expected_kept_tokens: object,
    layout_index: int,
) -> tuple[tuple[float, ...], int]:
    if set(audit_by_key) != set(span_by_key):
        raise ValueError(f"layout {layout_index} span audit keys do not match visual spans")
    keep_ratios = []
    zero_kept_spans = 0
    for key, kept_tokens in audit_by_key.items():
        token_count = span_by_key[key]
        if kept_tokens > token_count:
            raise ValueError(f"layout {layout_index} span {key} keeps {kept_tokens}/{token_count}")
        keep_ratios.append(kept_tokens / token_count)
        zero_kept_spans += int(kept_tokens == 0)
    if sum(audit_by_key.values()) != expected_kept_tokens:
        raise ValueError(f"layout {layout_index} span audit sum does not match kept_visual_tokens")
    return tuple(keep_ratios), zero_kept_spans


def _audit_layout(layout: object, *, layout_index: int) -> _LayoutSpanAudit | None:
    if not isinstance(layout, Mapping):
        raise ValueError(f"KV layout {layout_index} must be an object")
    decision = layout.get("compression_record")
    if not isinstance(decision, Mapping):
        return None
    spans = decision.get("visual_token_spans")
    if not isinstance(spans, list) or not spans:
        return None
    kept_by_span = decision.get("kept_visual_tokens_by_span")
    if kept_by_span is None:
        return _LayoutSpanAudit(len(spans), 0, 0, ())
    if not isinstance(kept_by_span, list):
        raise ValueError(f"layout {layout_index} kept_visual_tokens_by_span must be a list")
    span_by_key = _index_visual_spans(spans, layout_index=layout_index)
    audit_by_key = _index_span_retentions(kept_by_span, layout_index=layout_index)
    keep_ratios, zero_kept_spans = _validate_span_retentions(
        span_by_key,
        audit_by_key,
        expected_kept_tokens=decision.get("kept_visual_tokens"),
        layout_index=layout_index,
    )
    return _LayoutSpanAudit(
        visual_spans=len(spans),
        audited_spans=len(audit_by_key),
        zero_kept_spans=zero_kept_spans,
        keep_ratios=keep_ratios,
    )


def _span_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    """校验并汇总 candidate 的 per-span retention audit。"""

    layouts = _require_physical_kv(record)["layouts"]
    if not isinstance(layouts, list):
        raise ValueError("record.kv_cache.layouts must be a list")
    audits = [
        audit
        for layout_index, layout in enumerate(layouts)
        if (audit := _audit_layout(layout, layout_index=layout_index)) is not None
    ]
    audited_records = sum(audit.audited for audit in audits)
    keep_ratios = [ratio for audit in audits for ratio in audit.keep_ratios]
    return {
        "available": bool(audits) and len(audits) == audited_records,
        "visual_records": len(audits),
        "audited_records": audited_records,
        "total_visual_spans": sum(audit.visual_spans for audit in audits),
        "audited_visual_spans": sum(audit.audited_spans for audit in audits),
        "zero_kept_visual_spans": sum(audit.zero_kept_spans for audit in audits),
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
        "candidate_physical_prompt_tokens": int(candidate_kv["physical_prompt_tokens"]),
        "physical_token_ratio": (
            int(candidate_kv["physical_prompt_tokens"]) / baseline_physical_tokens
        ),
        "baseline_active_prompt_bytes": baseline_active_bytes,
        "candidate_active_prompt_bytes": int(candidate_kv["active_prompt_bytes"]),
        "active_prompt_bytes_ratio": (
            int(candidate_kv["active_prompt_bytes"]) / baseline_active_bytes
        ),
        "task_quality": _task_quality_comparison(baseline, candidate),
        "span_audit": _span_audit(candidate),
        "baseline_git_commit": baseline["environment"]["git_commit"],
        "baseline_git_dirty": baseline["environment"]["git_dirty"],
        "candidate_git_commit": candidate["environment"]["git_commit"],
        "candidate_git_dirty": candidate["environment"]["git_dirty"],
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_task_quality_drop: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        descriptor = row["candidate"]
        key = (
            row["manifest_sha256"],
            int(row["max_tokens"]),
            *_candidate_identity(descriptor),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        first = group_rows[0]
        case_ids = [str(row["case_id"]) for row in group_rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case_id across benchmark replication cells: {case_ids}")
        prefix_tokens = [int(value) for row in group_rows for value in row["stable_prefix_tokens"]]
        prefix_ratios = [
            float(value) for row in group_rows for value in row["stable_prefix_ratios"]
        ]
        exact_values = [bool(value) for row in group_rows for value in row["exact_per_request"]]
        baseline_output_tokens = sum(
            sum(int(value) for value in row["baseline_output_lengths"]) for row in group_rows
        )
        baseline_physical_tokens = sum(
            int(row["baseline_physical_prompt_tokens"]) for row in group_rows
        )
        baseline_active_bytes = sum(int(row["baseline_active_prompt_bytes"]) for row in group_rows)
        if baseline_output_tokens < 1 or baseline_physical_tokens < 1 or baseline_active_bytes < 1:
            raise RuntimeError("aggregate baseline denominators must be positive")
        span_audits = [row["span_audit"] for row in group_rows]
        total_visual_records = sum(int(audit["visual_records"]) for audit in span_audits)
        audited_visual_records = sum(int(audit["audited_records"]) for audit in span_audits)
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
                "max_tokens": int(first["max_tokens"]),
                "case_count": len(group_rows),
                "request_count": len(exact_values),
                "case_ids": sorted(str(row["case_id"]) for row in group_rows),
                "exact_request_rate": sum(exact_values) / len(exact_values),
                "exact_case_rate": (
                    sum(bool(row["token_exact"]) for row in group_rows) / len(group_rows)
                ),
                "stable_prefix_tokens": summarize_values(prefix_tokens),
                "stable_prefix_ratios": summarize_values(prefix_ratios),
                "stable_prefix_ratio_micro": (sum(prefix_tokens) / baseline_output_tokens),
                "physical_token_ratio": (
                    sum(int(row["candidate_physical_prompt_tokens"]) for row in group_rows)
                    / baseline_physical_tokens
                ),
                "active_prompt_bytes_ratio": (
                    sum(int(row["candidate_active_prompt_bytes"]) for row in group_rows)
                    / baseline_active_bytes
                ),
                "task_quality": _aggregate_task_quality(
                    group_rows,
                    max_task_quality_drop=max_task_quality_drop,
                ),
                "span_audit": {
                    "available": (
                        total_visual_records > 0 and total_visual_records == audited_visual_records
                    ),
                    "visual_records": total_visual_records,
                    "audited_records": audited_visual_records,
                    "total_visual_spans": sum(
                        int(audit["total_visual_spans"]) for audit in span_audits
                    ),
                    "audited_visual_spans": sum(
                        int(audit["audited_visual_spans"]) for audit in span_audits
                    ),
                    "zero_kept_visual_spans": sum(
                        int(audit["zero_kept_visual_spans"]) for audit in span_audits
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
        key=lambda row: (
            row["manifest_name"],
            row["candidate"]["label"],
            row["max_tokens"],
        ),
    )


def _validate_summary_request(
    records: Sequence[Mapping[str, Any]],
    *,
    max_task_quality_drop: float,
) -> None:
    if not 0.0 <= max_task_quality_drop <= 1.0:
        raise ValueError("max_task_quality_drop must be in [0, 1]")
    if not records:
        raise ValueError("pruning fidelity requires at least one benchmark record")


def _index_fidelity_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str,
) -> tuple[dict[ComparisonKey, Mapping[str, Any]], list[_IndexedCandidate]]:
    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    candidates: list[_IndexedCandidate] = []
    seen_candidates: set[tuple[ComparisonKey, CandidateIdentity]] = set()
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
        candidates.append(_IndexedCandidate(record, key, descriptor))
    if not baselines:
        raise ValueError(f"no {baseline_mode!r} baseline records were provided")
    if not candidates:
        raise ValueError("no pruning candidate records were provided")
    return baselines, candidates


def _coverage_cell(key: ComparisonKey) -> tuple[Any, Any]:
    manifest_sha256, _case_id, _num_requests, _max_tokens, keep_ratio = key
    return manifest_sha256, keep_ratio


def _candidate_coverage_key(candidate: _IndexedCandidate) -> CoverageKey:
    return (*_coverage_cell(candidate.key), *_candidate_identity(candidate.descriptor))


def _materialize_candidate_rows(
    candidates: Sequence[_IndexedCandidate],
    baselines: Mapping[ComparisonKey, Mapping[str, Any]],
    *,
    baseline_mode: str,
) -> tuple[list[dict[str, Any]], dict[CoverageKey, set[ComparisonKey]]]:
    rows: list[dict[str, Any]] = []
    candidate_coverage: dict[CoverageKey, set[ComparisonKey]] = {}
    for candidate in candidates:
        if candidate.key not in baselines:
            raise ValueError(f"missing {baseline_mode!r} baseline for key={candidate.key}")
        coverage_key = _candidate_coverage_key(candidate)
        candidate_coverage.setdefault(coverage_key, set()).add(candidate.key)
        rows.append(
            _case_row(
                baselines[candidate.key],
                candidate.record,
                candidate.descriptor,
            )
        )
    return rows, candidate_coverage


def _validate_candidate_coverage(
    candidate_coverage: Mapping[CoverageKey, set[ComparisonKey]],
    baselines: Mapping[ComparisonKey, Mapping[str, Any]],
) -> None:
    for coverage_key, actual_keys in candidate_coverage.items():
        manifest_sha256, keep_ratio, *_candidate_identity_fields = coverage_key
        expected_keys = {
            key for key in baselines if _coverage_cell(key) == (manifest_sha256, keep_ratio)
        }
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            raise ValueError(
                f"candidate does not cover every selected baseline case: missing={missing}"
            )


def _sort_case_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            row["manifest_name"],
            row["candidate"]["label"],
            row["case_id"],
        )
    )


def summarize_pruning_fidelity_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str = "off_graph",
    max_task_quality_drop: float = 0.01,
) -> dict[str, Any]:
    """生成 case-level 和 dataset-level pruning fidelity 汇总。"""

    _validate_summary_request(records, max_task_quality_drop=max_task_quality_drop)
    baselines, candidates = _index_fidelity_records(records, baseline_mode=baseline_mode)
    rows, candidate_coverage = _materialize_candidate_rows(
        candidates,
        baselines,
        baseline_mode=baseline_mode,
    )
    _validate_candidate_coverage(candidate_coverage, baselines)
    _sort_case_rows(rows)
    return {
        "schema_version": PRUNING_FIDELITY_SCHEMA_VERSION,
        "summary_type": "pruning_fidelity",
        "baseline_mode": baseline_mode,
        "limitations": list(PRUNING_FIDELITY_LIMITATIONS),
        "task_quality_gate": {
            "max_macro_score_drop": max_task_quality_drop,
            "metrics": list(TASK_QUALITY_METRICS),
        },
        "aggregates": _aggregate_rows(
            rows,
            max_task_quality_drop=max_task_quality_drop,
        ),
        "cases": rows,
    }


def render_pruning_fidelity_markdown(summary: Mapping[str, Any]) -> str:
    """把 pruning fidelity 与 reference task quality 渲染为可审计 Markdown。"""

    threshold = float(summary["task_quality_gate"]["max_macro_score_drop"])
    lines = [
        "# P6.12 Pruning Fidelity and Task Quality Summary",
        "",
        "> Greedy-token fidelity plus optional multi-reference lexical quality; "
        "token-F1/ROUGE-L are not COCO CIDEr/SPICE.",
        f"> Task gate: every macro score drop must be <= {threshold:.6f}.",
        "",
        "## Dataset Aggregates",
        "",
        "| Manifest | Candidate | Max tokens | Cases | Requests | Exact requests | "
        "Prefix micro | Prefix min | Physical KV | Active bytes | Token F1 B/C | "
        "ROUGE-L B/C | Task gate | Zero spans | Span audit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|",
    ]
    for aggregate in summary["aggregates"]:
        span_audit = aggregate["span_audit"]
        task_quality = aggregate["task_quality"]
        if task_quality["available"]:
            token_f1 = task_quality["token_f1"]
            rouge_l = task_quality["rouge_l_f1"]
            token_f1_cell = f"{token_f1['baseline_macro']:.3f}/{token_f1['candidate_macro']:.3f}"
            rouge_l_cell = f"{rouge_l['baseline_macro']:.3f}/{rouge_l['candidate_macro']:.3f}"
            task_gate = (
                "PASS"
                if task_quality["gate"]["passed"]
                else "FAIL: " + "; ".join(task_quality["gate"]["failures"])
            )
        else:
            token_f1_cell = "n/a"
            rouge_l_cell = "n/a"
            task_gate = f"INELIGIBLE: {task_quality['reason']}"
        lines.append(
            f"| {aggregate['manifest_name']} | {aggregate['candidate']['label']} | "
            f"{aggregate['max_tokens']} | {aggregate['case_count']} | "
            f"{aggregate['request_count']} | "
            f"{aggregate['exact_request_rate']:.3f} | "
            f"{aggregate['stable_prefix_ratio_micro']:.3f} | "
            f"{aggregate['stable_prefix_ratios']['min']:.3f} | "
            f"{aggregate['physical_token_ratio']:.3f}x | "
            f"{aggregate['active_prompt_bytes_ratio']:.3f}x | "
            f"{token_f1_cell} | {rouge_l_cell} | {task_gate} | "
            f"{span_audit['zero_kept_visual_spans']} | "
            f"{'yes' if span_audit['available'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Candidate | Max tokens | Requests | Stable prefixes | "
            "Prefix ratios | Exact | Physical KV | Active bytes | Token F1 B/C | "
            "ROUGE-L B/C |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cases"]:
        ratios = ", ".join(f"{value:.3f}" for value in row["stable_prefix_ratios"])
        task_quality = row["task_quality"]
        if task_quality["available"]:
            baseline_scores = task_quality["baseline_scores"]
            candidate_scores = task_quality["candidate_scores"]
            token_f1_cell = (
                f"{fmean(float(score['token_f1']) for score in baseline_scores):.3f}/"
                f"{fmean(float(score['token_f1']) for score in candidate_scores):.3f}"
            )
            rouge_l_cell = (
                f"{fmean(float(score['rouge_l_f1']) for score in baseline_scores):.3f}/"
                f"{fmean(float(score['rouge_l_f1']) for score in candidate_scores):.3f}"
            )
        else:
            token_f1_cell = "n/a"
            rouge_l_cell = "n/a"
        lines.append(
            f"| {row['case_id']} | {row['candidate']['label']} | "
            f"{row['max_tokens']} | {row['num_requests']} | "
            f"{row['stable_prefix_tokens']} | [{ratios}] | "
            f"{row['exact_request_rate']:.3f} | {row['physical_token_ratio']:.3f}x | "
            f"{row['active_prompt_bytes_ratio']:.3f}x | "
            f"{token_f1_cell} | {rouge_l_cell} |"
        )
    return "\n".join(lines) + "\n"
