#!/usr/bin/env python3
"""Audit cold and prefix-hit NVTX ranges in an exported Nsight SQLite file."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _ranges(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    return [
        (str(name), int(start), int(end))
        for name, start, end in connection.execute(
            """
            SELECT coalesce(n.text, s.value), n.start, n.end
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON n.textId = s.id
            WHERE n.end IS NOT NULL AND coalesce(n.text, s.value) IS NOT NULL
            ORDER BY n.start
            """
        )
    ]


def _one_parent(
    ranges: list[tuple[str, int, int]],
    name: str,
) -> tuple[int, int]:
    matches = [(start, end) for current, start, end in ranges if current == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} NVTX range; found {len(matches)}")
    return matches[0]


def _nested_names(
    ranges: list[tuple[str, int, int]],
    parent: tuple[int, int],
) -> Counter[str]:
    start, end = parent
    return Counter(
        name for name, item_start, item_end in ranges if item_start >= start and item_end <= end
    )


def _is_vision_range(name: str) -> bool:
    lowered = name.lower()
    return "vision" in lowered or "deepstack" in lowered


def main() -> None:
    args = _parse_args()
    evidence_record: dict[str, Any] = json.loads(args.evidence.read_text(encoding="utf-8"))
    semantic_evidence = evidence_record["evidence"]
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        ranges = _ranges(connection)
    finally:
        connection.close()

    cold_names = _nested_names(ranges, _one_parent(ranges, "prism::cold_request"))
    hit_names = _nested_names(ranges, _one_parent(ranges, "prism::prefix_hit_request"))
    cold_vision = sorted(name for name in cold_names if _is_vision_range(name))
    hit_vision = sorted(name for name in hit_names if _is_vision_range(name))
    hit_deltas = semantic_evidence["prefix_hit_request_counter_deltas"]
    checks = {
        "semantic_prefix_hit_passed": semantic_evidence.get("passed") is True,
        "cold_trace_contains_vision_ranges": bool(cold_vision),
        "prefix_hit_trace_contains_no_vision_or_deepstack_ranges": not hit_vision,
        "prefix_hit_skipped_visual_hydration": hit_deltas.get("visual_hydration_skips") == 1,
        "prefix_hit_used_cached_public_prefix_tokens": semantic_evidence.get(
            "actual_cached_tokens", 0
        )
        > 0,
        "prefix_hit_had_no_stale_fallback": hit_deltas.get("stale_probe_fallbacks") == 0,
    }
    result = {
        "record_type": "working_set_prefix_trace_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "common_prefix_reuse": {
            "cached_tokens": semantic_evidence["actual_cached_tokens"],
            "prompt_tokens": semantic_evidence["prompt_tokens"],
            "cached_fraction": semantic_evidence["cached_fraction"],
        },
        "cold_vision_ranges": {name: cold_names[name] for name in cold_vision},
        "prefix_hit_vision_ranges": {name: hit_names[name] for name in hit_vision},
        "cold_nested_ranges": dict(sorted(cold_names.items())),
        "prefix_hit_nested_ranges": dict(sorted(hit_names.items())),
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite trace audit: {args.output}")
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
