"""汇总 P7.4-B CUDA Graph trace、固定成本与 padding matrix。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import validate_benchmark_record
from prism_infer.analysis.performance_profile import (
    validate_performance_profile_record,
)


REPLAY_RANGE = "prism::runner.cudagraph.replay"
EXTERNAL_RANGES = (
    "prism::runner.prepare_inputs",
    "prism::runner.prepare_sample_inputs",
    "prism::runner.cudagraph.copy_inputs",
    "prism::runner.model.compute_logits",
    "prism::runner.sampler",
)
EXPECTED_CATEGORIES = {
    "linear_gemv",
    "paged_decode_attention",
    "copy_cast",
    "elementwise",
    "reduction",
    "layout_index",
    "kv_store",
    "trigonometric",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return records


def _median(stats: dict[str, Any]) -> float:
    value = stats.get("median")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("summary median must be numeric")
    return float(value)


def _require_trace_contract(trace: dict[str, Any]) -> None:
    if trace.get("record_type") != "nsys_profile_summary":
        raise ValueError("trace must be an nsys_profile_summary")
    if int(trace.get("schema_version", 0)) < 2:
        raise ValueError("P7.4-B requires nsys summary schema >= 2")
    targets = trace.get("target_ranges")
    if not isinstance(targets, dict):
        raise ValueError("trace target_ranges must be an object")
    missing = [name for name in (REPLAY_RANGE, *EXTERNAL_RANGES) if name not in targets]
    if missing:
        raise ValueError(f"trace is missing required target ranges: {missing}")
    categories = set(targets[REPLAY_RANGE].get("kernel_categories", {}))
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            "unexpected replay kernel categories: "
            f"expected={sorted(EXPECTED_CATEGORIES)}, actual={sorted(categories)}"
        )
    fraction = sum(
        float(category["kernel_time_fraction"])
        for category in targets[REPLAY_RANGE]["kernel_categories"].values()
    )
    if not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"replay kernel category fractions do not sum to 1: {fraction}")


def _require_padding_contract(records: list[dict[str, Any]]) -> None:
    if len(records) != 8:
        raise ValueError(f"padding matrix must contain 8 cells, got {len(records)}")
    batches = [int(record["traffic"]["batch_size"]) for record in records]
    if sorted(batches) != list(range(1, 9)) or len(set(batches)) != len(batches):
        raise ValueError(f"padding matrix must cover unique batch 1..8: {batches}")
    commits = {str(record["environment"]["git_commit"]) for record in records}
    if len(commits) != 1 or any(record["environment"]["git_dirty"] for record in records):
        raise ValueError("padding matrix must come from one clean commit")
    ceilings = {int(record["model"]["max_num_seqs"]) for record in records}
    bucket_sets = {
        tuple(record["execution_backend"]["cuda_graph_batch_sizes"]) for record in records
    }
    if ceilings != {8} or bucket_sets != {(1, 2, 4, 8)}:
        raise ValueError("padding matrix must use max_num_seqs=8 and buckets [1,2,4,8]")
    baseline_tokens = min(records, key=lambda record: record["traffic"]["batch_size"])[
        "correctness"
    ]["token_ids"][0]
    for record in records:
        backend = record["execution_backend"]
        batch = int(record["traffic"]["batch_size"])
        selected = next(bucket for bucket in (1, 2, 4, 8) if bucket >= batch)
        if int(backend["selected_decode_batch_size"]) != selected:
            raise ValueError(f"selected bucket mismatch for batch={batch}")
        if int(backend["decode_batch_padding"]) != selected - batch:
            raise ValueError(f"padding mismatch for batch={batch}")
        if not record["correctness"]["outputs_identical_across_repeats"]:
            raise ValueError(f"repeat instability for batch={batch}")
        token_rows = record["correctness"]["token_ids"]
        if len(token_rows) != batch or any(row != baseline_tokens for row in token_rows):
            raise ValueError(f"replicated request output mismatch for batch={batch}")


def _summarize_validated(
    trace: dict[str, Any],
    semantic: dict[str, Any],
    padding_records: list[dict[str, Any]],
) -> dict[str, Any]:
    replay = trace["target_ranges"][REPLAY_RANGE]
    decode = trace["phase_summary"]["decode"]
    semantic_decode = semantic["summary_by_phase"]["decode"]
    replay_busy = _median(replay["kernel_time_ms_per_range"])
    engine_busy = _median(decode["kernel_busy_ms"])
    categories = [
        {
            "category": name,
            "kernels_per_step": _median(values["kernels_per_range"]),
            "kernel_time_ms_per_step": _median(values["kernel_time_ms_per_range"]),
            "kernel_time_fraction": float(values["kernel_time_fraction"]),
        }
        for name, values in replay["kernel_categories"].items()
    ]
    categories.sort(key=lambda row: row["kernel_time_ms_per_step"], reverse=True)

    external = []
    for name in EXTERNAL_RANGES:
        target = trace["target_ranges"][name]
        semantic_name = name.removeprefix("prism::")
        semantic_region = semantic_decode[semantic_name]
        external.append(
            {
                "range": semantic_name,
                "cpu_range_ms": _median(target["cpu_range_ms_per_range"]),
                "direct_gpu_busy_ms": _median(target["gpu_busy_ms_per_range"]),
                "direct_gpu_span_ms": _median(target["gpu_span_ms_per_range"]),
                "gpu_tail_after_cpu_ms": _median(target["gpu_tail_after_cpu_ms_per_range"]),
                "semantic_cpu_ms": _median(semantic_region["cpu_ms"]),
                "semantic_cuda_event_ms": (
                    _median(semantic_region["cuda_ms"])
                    if semantic_region["cuda_ms"] is not None
                    else None
                ),
            }
        )

    padding_rows = []
    for record in sorted(
        padding_records,
        key=lambda value: value["traffic"]["batch_size"],
    ):
        backend = record["execution_backend"]
        padding_rows.append(
            {
                "batch_size": int(record["traffic"]["batch_size"]),
                "selected_bucket": int(backend["selected_decode_batch_size"]),
                "padding": int(backend["decode_batch_padding"]),
                "padding_fraction_of_bucket": (
                    int(backend["decode_batch_padding"])
                    / int(backend["selected_decode_batch_size"])
                ),
                "decode_step_median_ms": _median(record["timing_ms"]["decode_step"]),
                "decode_step_p90_ms": float(record["timing_ms"]["decode_step"]["p90"]),
                "decode_tokens_per_s": _median(record["throughput"]["decode_tokens_per_s"]),
                "capture_ms": float(backend["cuda_graph_capture_ms"]),
            }
        )

    return {
        "schema_version": 1,
        "record_type": "p74b_cuda_graph_summary",
        "trace_commit": str(semantic["metadata"]["git_commit"]),
        "trace_clean": not bool(semantic["metadata"]["git_dirty"]),
        "padding_commit": str(padding_records[0]["environment"]["git_commit"]),
        "padding_clean": True,
        "replay": {
            "steps": int(replay["range_count"]),
            "kernels_per_step": _median(replay["kernels_per_range"]),
            "kernel_busy_median_ms": replay_busy,
            "kernel_busy_p90_ms": float(replay["kernel_time_ms_per_range"]["p90"]),
            "gpu_busy_median_ms": _median(replay["gpu_busy_ms_per_range"]),
            "gpu_span_median_ms": _median(replay["gpu_span_ms_per_range"]),
            "gpu_span_minus_busy_ms": (
                _median(replay["gpu_span_ms_per_range"]) - _median(replay["gpu_busy_ms_per_range"])
            ),
            "cpu_launch_range_median_ms": _median(replay["cpu_range_ms_per_range"]),
            "cpu_gpu_busy_overlap_median_ms": _median(replay["cpu_gpu_busy_overlap_ms_per_range"]),
            "cpu_gpu_busy_overlap_fraction_median": _median(
                replay["cpu_gpu_busy_overlap_fraction_per_range"]
            ),
            "gpu_tail_after_cpu_median_ms": _median(replay["gpu_tail_after_cpu_ms_per_range"]),
            "engine_decode_kernel_busy_median_ms": engine_busy,
            "graph_external_kernel_busy_difference_ms": engine_busy - replay_busy,
            "categories": categories,
        },
        "graph_external_ranges": external,
        "padding_matrix": {
            "cells": len(padding_rows),
            "captured_buckets": [1, 2, 4, 8],
            "max_num_seqs": 8,
            "repeat_stable_cells": len(padding_rows),
            "replicated_request_outputs_exact_across_batches": True,
            "rows": padding_rows,
        },
        "claim_boundaries": [
            "Nsight node tracing adds instrumentation; gpu_span_minus_busy is not an occupancy metric.",
            "Sampler CPU time exposes stream synchronization and must not be added as independent fixed work.",
            "Padding cells are one process-level run each; the matrix proves coverage/correctness, not padding speedup or slowdown.",
            "The matrix is replicated single-image offline decode, not online serving goodput.",
        ],
    }


def summarize_p74b(
    trace: dict[str, Any],
    semantic: dict[str, Any],
    padding_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """校验输入并生成 P7.4-B summary。"""

    _require_trace_contract(trace)
    validate_performance_profile_record(semantic)
    for record in padding_records:
        validate_benchmark_record(record)
    _require_padding_contract(padding_records)
    if semantic.get("metadata", {}).get("git_dirty"):
        raise ValueError("trace semantic profile must come from a clean commit")
    return _summarize_validated(trace, semantic, padding_records)


def render_markdown(summary: dict[str, Any]) -> str:
    replay = summary["replay"]
    lines = [
        "# P7.4-B CUDA Graph Profile Summary",
        "",
        f"- trace commit: `{summary['trace_commit']}` (clean: `{summary['trace_clean']}`)",
        f"- padding commit: `{summary['padding_commit']}` (clean: `{summary['padding_clean']}`)",
        f"- replay: `{replay['kernel_busy_median_ms']:.3f} ms` kernel busy, "
        f"`{replay['kernels_per_step']:.0f}` kernels/step",
        "",
        "## Replay kernel categories",
        "",
        "| Category | Kernels/step | Time/step | Fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in replay["categories"]:
        lines.append(
            f"| {row['category']} | {row['kernels_per_step']:.0f} | "
            f"{row['kernel_time_ms_per_step']:.3f} ms | "
            f"{row['kernel_time_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Graph-external ranges",
            "",
            "| Range | CPU range | Direct GPU busy | CUDA-event elapsed | GPU tail after CPU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["graph_external_ranges"]:
        cuda_event = row["semantic_cuda_event_ms"]
        lines.append(
            f"| {row['range']} | {row['cpu_range_ms']:.3f} ms | "
            f"{row['direct_gpu_busy_ms']:.3f} ms | "
            f"{cuda_event:.3f} ms | {row['gpu_tail_after_cpu_ms']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            "## Fixed-8 bucket/padding matrix",
            "",
            "| Batch | Bucket | Padding | Padding/bucket | TPOT median | TPOT p90 | Decode tok/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["padding_matrix"]["rows"]:
        lines.append(
            f"| {row['batch_size']} | {row['selected_bucket']} | {row['padding']} | "
            f"{row['padding_fraction_of_bucket']:.1%} | "
            f"{row['decode_step_median_ms']:.3f} ms | "
            f"{row['decode_step_p90_ms']:.3f} ms | "
            f"{row['decode_tokens_per_s']:.3f} |"
        )
    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in summary["claim_boundaries"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-analysis", required=True)
    parser.add_argument("--semantic-profile", required=True)
    parser.add_argument("--padding-records", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    semantic_records = _read_jsonl(args.semantic_profile)
    if len(semantic_records) != 1:
        raise ValueError("P7.4-B expects exactly one semantic profile record")
    summary = summarize_p74b(
        _read_json(args.trace_analysis),
        semantic_records[0],
        _read_jsonl(args.padding_records),
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    markdown = render_markdown(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
