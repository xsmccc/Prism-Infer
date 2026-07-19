#!/usr/bin/env python3
"""Summarize P7.1 external baselines under an explicit comparison profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.external_comparison import (
    COMPARISON_PROFILES,
    compare_external_records,
    load_external_records,
    render_external_markdown,
)
from prism_infer.analysis.pareto_summary import load_benchmark_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism", nargs="+", required=True)
    parser.add_argument("--external", nargs="+", required=True)
    parser.add_argument(
        "--comparison-profile",
        choices=COMPARISON_PROFILES,
        required=True,
    )
    parser.add_argument("--prism-modes", nargs="+", required=True)
    parser.add_argument("--prism-keep-ratio", type=float, default=0.5)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    prism_records = load_benchmark_jsonl(args.prism)
    external_records = load_external_records(args.external)
    rows = []
    for prism_mode in args.prism_modes:
        rows.extend(
            compare_external_records(
                prism_records,
                external_records,
                prism_mode=prism_mode,
                prism_keep_ratio=args.prism_keep_ratio,
                comparison_profile=args.comparison_profile,
            )
        )
    rows.sort(key=lambda row: (row["case_id"], row["prism_mode"]))
    Path(args.json_output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown_output).write_text(
        render_external_markdown(rows),
        encoding="utf-8",
    )
    print(f"compared {len(rows)} P7.1 cells under {args.comparison_profile}")


if __name__ == "__main__":
    main()
