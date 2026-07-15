#!/usr/bin/env python3
"""Generate dataset-level P6.12 pruning-fidelity JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.pareto_summary import load_benchmark_jsonl
from prism_infer.analysis.pruning_fidelity import (
    render_pruning_fidelity_markdown,
    summarize_pruning_fidelity_records,
)


def _parse_allowlist(value: str | None, *, label: str) -> set[str] | None:
    if value is None:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    if not selected:
        raise ValueError(f"{label} must contain at least one value")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="P6 system benchmark JSONL files")
    parser.add_argument("--baseline-mode", default="off_graph")
    parser.add_argument("--cases", help="comma-separated case allowlist")
    parser.add_argument("--strategies", help="comma-separated candidate strategy allowlist")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    try:
        selected_cases = _parse_allowlist(args.cases, label="--cases")
        selected_strategies = _parse_allowlist(
            args.strategies,
            label="--strategies",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    records = load_benchmark_jsonl(args.inputs)
    if selected_cases is not None:
        records = [
            record
            for record in records
            if record["workload"]["case_id"] in selected_cases
        ]
    if selected_strategies is not None:
        records = [
            record
            for record in records
            if record["mode"]["name"] == args.baseline_mode
            or record["mode"]["visual_pruning_strategy"] in selected_strategies
        ]
    if not records:
        raise SystemExit("record filters selected no benchmark records")

    summary = summarize_pruning_fidelity_records(
        records,
        baseline_mode=args.baseline_mode,
    )
    json_payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_pruning_fidelity_markdown(summary)
    if args.json_output:
        Path(args.json_output).write_text(json_payload, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    print(
        f"summarized {len(records)} records into "
        f"{len(summary['aggregates'])} aggregate rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
