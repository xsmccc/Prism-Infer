#!/usr/bin/env python3
"""离线分析 Prism-Infer KV trace JSONL。

输入是 `kv_trace()` 生成的 JSONL 文件，输出 summary JSON 和可读 Markdown。
该脚本只读取 trace，不运行模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_infer.analysis.kv_trace import (
    format_summary_markdown,
    read_trace_jsonl,
    render_summary_svg,
    summarize_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Prism-Infer KV trace JSONL")
    parser.add_argument("trace", type=Path, help="trace JSONL path")
    parser.add_argument("--summary-json", type=Path, default=None, help="optional summary JSON output")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown output")
    parser.add_argument("--svg", type=Path, default=None, help="optional SVG chart output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_trace_jsonl(args.trace)
    summary = summarize_trace(records)
    markdown = format_summary_markdown(summary)

    print(markdown)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_summary_svg(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
