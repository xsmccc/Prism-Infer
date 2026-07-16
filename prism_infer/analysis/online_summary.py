"""Cross-cell P7.3 online benchmark summary."""

from __future__ import annotations

from typing import Mapping, Sequence

from prism_infer.analysis.online_serving import (
    validate_online_benchmark_record,
)


def summarize_online_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("online summary requires at least one record")
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        validate_online_benchmark_record(record)
        workload = record["workload"]
        arrival = record["arrival"]
        engine = record["engine"]
        summary = record["summary"]
        cell = (
            workload["manifest"],
            workload["case"],
            workload["requests"],
            workload["max_tokens"],
            arrival["process"],
            arrival["request_rate_per_s"],
            arrival["seed"],
            engine["mode"],
            engine["max_chunk_size"],
            engine["max_num_seqs"],
            engine["num_kvcache_blocks"],
            engine["enable_prefix_caching"],
            summary["slo"]["ttft_ms"],
            summary["slo"]["tpot_ms"],
        )
        if cell in seen:
            raise ValueError(f"duplicate online benchmark cell: {cell}")
        seen.add(cell)
        rows.append(
            {
                "manifest": workload["manifest"],
                "case": workload["case"],
                "requests": workload["requests"],
                "max_tokens": workload["max_tokens"],
                "arrival_process": arrival["process"],
                "request_rate_per_s": arrival["request_rate_per_s"],
                "seed": arrival["seed"],
                "mode": engine["mode"],
                "max_chunk_size": engine["max_chunk_size"],
                "max_num_seqs": engine["max_num_seqs"],
                "num_kvcache_blocks": engine["num_kvcache_blocks"],
                "enable_prefix_caching": engine["enable_prefix_caching"],
                "git_commit": record["git_commit"],
                "git_dirty": record["git_dirty"],
                "completed": summary["counts"]["completed"],
                "rejected": summary["counts"]["rejected"],
                "cancelled": summary["counts"]["cancelled"],
                "queue_p99_ms": summary["latency_ms"]["queue"]["p99"],
                "ttft_p99_ms": summary["latency_ms"]["ttft"]["p99"],
                "tpot_p99_ms": summary["latency_ms"]["tpot"]["p99"],
                "request_p99_ms": summary["latency_ms"]["request"]["p99"],
                "requests_per_s": summary["throughput"]["requests_per_s"],
                "output_tokens_per_s": summary["throughput"]["output_tokens_per_s"],
                "goodput_requests_per_s": summary["goodput"]["requests_per_s"],
                "goodput_fraction": summary["goodput"]["fraction_of_completed"],
                "peak_active": summary["scheduler"]["peak_active"],
                "peak_gpu_kv_blocks": summary["scheduler"]["peak_gpu_kv_blocks"],
                "swap_preemptions": summary["scheduler"]["swap_preemptions"],
                "recompute_preemptions": summary["scheduler"]["recompute_preemptions"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["case"],
            row["mode"],
            row["arrival_process"],
            row["request_rate_per_s"],
        )
    )
    return {
        "schema_version": 1,
        "record_type": "prism_online_matrix_summary",
        "cell_count": len(rows),
        "all_clean": all(not bool(row["git_dirty"]) for row in rows),
        "commits": sorted({str(row["git_commit"]) for row in rows}),
        "rows": rows,
    }


def render_online_summary_markdown(summary: Mapping[str, object]) -> str:
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise ValueError("online matrix summary rows must be a list")
    lines = [
        "# Prism Online Serving Matrix",
        "",
        f"- cells: `{summary.get('cell_count')}`",
        f"- all clean: `{summary.get('all_clean')}`",
        f"- commits: `{summary.get('commits')}`",
        "",
        "| Case | Mode | Arrival | Rate | Requests | Good | Queue p99 | TTFT p99 | TPOT p99 | Req/s | Goodput/s | Peak active | Preempt S/R |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['arrival_process']} | "
            f"{float(row['request_rate_per_s']):.3f} | {row['requests']} | "
            f"{float(row['goodput_fraction']):.3f} | "
            f"{float(row['queue_p99_ms']):.3f} | "
            f"{float(row['ttft_p99_ms']):.3f} | "
            f"{float(row['tpot_p99_ms']):.3f} | "
            f"{float(row['requests_per_s']):.3f} | "
            f"{float(row['goodput_requests_per_s']):.3f} | "
            f"{row['peak_active']} | "
            f"{row['swap_preemptions']}/{row['recompute_preemptions']} |"
        )
    return "\n".join(lines) + "\n"
