"""Small shared summaries and incremental JSON saves for vision benchmarks."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def summarize_requests(
    rows: list[dict[str, Any]],
    *,
    concurrent: bool,
) -> dict[str, Any]:
    """Summarize completed requests; rates require one concurrent burst.

    Sequential rows can be separated by other phases or cache resets. Their
    first-start/last-finish span is an observation window, not service time.
    Request dictionaries, including observed/final token comparisons, are kept
    untouched.
    """

    completed = [row for row in rows if row["status"] == "finished"]
    summary: dict[str, Any] = {
        "completed_requests": len(completed),
        "duration_scope": (
            "single_concurrent_burst" if concurrent else "observation_window_including_gaps"
        ),
    }
    if not completed:
        return summary
    start = min(row["client_start_ns"] for row in completed)
    finish = max(row["client_finish_ns"] for row in completed)
    duration_s = (finish - start) / 1e9
    output_tokens = sum(len(row["token_ids"]) for row in completed)
    summary.update(
        {
            "client_start_ns": start,
            "client_finish_ns": finish,
            "duration_s": duration_s,
            "output_tokens": output_tokens,
            "ttft_median_ms": statistics.median(row["ttft_ms"] for row in completed),
            "tpot_median_ms": statistics.median(row["tpot_ms"] for row in completed),
            "e2e_median_ms": statistics.median(row["e2e_ms"] for row in completed),
        }
    )
    if concurrent:
        summary["output_tokens_per_s"] = output_tokens / duration_s if duration_s > 0 else None
        summary["requests_per_s"] = len(completed) / duration_s if duration_s > 0 else None
    return summary


def save_report(path: str | Path, report: dict[str, Any]) -> None:
    """Replace one report after a complete JSON write, preserving the last save on failure."""

    output = Path(path)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
