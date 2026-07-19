"""Validate and summarize P7.3 online benchmark JSON records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.online_summary import (
    render_online_summary_markdown,
    summarize_online_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.records]
    summary = summarize_online_records(records)
    rendered_json = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    rendered_markdown = render_online_summary_markdown(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered_json + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered_markdown, encoding="utf-8")
    print(rendered_json)


if __name__ == "__main__":
    main()
