#!/usr/bin/env python3
"""生成固定 external records 与 Prism off baseline 的比较表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.external_comparison import (
    compare_external_records,
    load_external_records,
    render_external_markdown,
)
from prism_infer.analysis.pareto_summary import load_benchmark_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism", nargs="+", required=True)
    parser.add_argument("--external", nargs="+", required=True)
    parser.add_argument("--prism-mode", default="off_eager")
    parser.add_argument("--prism-keep-ratio", type=float, default=0.5)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    prism_records = load_benchmark_jsonl(args.prism)
    external_records = load_external_records(args.external)
    rows = compare_external_records(
        prism_records,
        external_records,
        prism_mode=args.prism_mode,
        prism_keep_ratio=args.prism_keep_ratio,
    )
    Path(args.json_output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown_output).write_text(
        render_external_markdown(rows),
        encoding="utf-8",
    )
    print(f"compared {len(rows)} external benchmark cells")


if __name__ == "__main__":
    main()
