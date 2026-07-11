#!/usr/bin/env python3
"""校验 P6 benchmark JSONL 并生成 Pareto JSON/Markdown 汇总。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.pareto_summary import (
    load_benchmark_jsonl,
    render_pareto_markdown,
    summarize_pareto_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="P6 system benchmark JSONL files")
    parser.add_argument("--baseline-mode", default="off_eager")
    parser.add_argument(
        "--modes",
        help="comma-separated mode allowlist; defaults to every mode in the inputs",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    records = load_benchmark_jsonl(args.inputs)
    if args.modes:
        selected_modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
        if not selected_modes:
            raise SystemExit("--modes must contain at least one mode")
        selected_modes.add(args.baseline_mode)
        records = [
            record for record in records if record["mode"]["name"] in selected_modes
        ]
        if not records:
            raise SystemExit("--modes did not select any input records")
    rows = summarize_pareto_records(records, baseline_mode=args.baseline_mode)
    json_payload = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_pareto_markdown(rows)
    if args.json_output:
        Path(args.json_output).write_text(json_payload, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    print(f"summarized {len(records)} records into {len(rows)} Pareto rows", file=sys.stderr)


if __name__ == "__main__":
    main()
