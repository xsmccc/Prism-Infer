#!/usr/bin/env python3
"""离线计算 P5.1 visual token importance。

输入是 P4 `kv_trace()` 生成的 JSONL。脚本只读取 trace，不运行模型，
不修改 runtime KV cache，也不声明压缩收益。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_infer.analysis.kv_trace import read_trace_jsonl
from prism_infer.analysis.visual_importance import (
    DEFAULT_KEEP_RATIOS,
    ImportanceWeights,
    format_importance_markdown,
    score_visual_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score visual token importance from Prism-Infer KV trace")
    parser.add_argument("trace", type=Path, help="P4 KV trace JSONL path")
    parser.add_argument("--output-json", type=Path, default=None, help="optional report JSON output")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown output")
    parser.add_argument("--top-k", type=int, default=20, help="number of top/bottom tokens to include")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        action="append",
        default=None,
        help="visual token keep ratio; may be specified multiple times",
    )
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--entropy-focus-weight", type=float, default=0.5)
    parser.add_argument("--k-norm-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratios = tuple(args.keep_ratio) if args.keep_ratio is not None else DEFAULT_KEEP_RATIOS
    weights = ImportanceWeights(
        attention_mass=args.attention_weight,
        entropy_focus=args.entropy_focus_weight,
        k_norm=args.k_norm_weight,
    )
    records = read_trace_jsonl(args.trace)
    report = score_visual_importance(
        records,
        weights=weights,
        keep_ratios=keep_ratios,
        top_k=args.top_k,
    )
    markdown = format_importance_markdown(report, top_k=args.top_k)
    print(markdown)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
