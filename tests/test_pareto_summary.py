"""P6 Pareto 汇总器测试。"""

from copy import deepcopy

import pytest

from prism_infer.analysis.pareto_summary import (
    render_pareto_markdown,
    stable_prefix_lengths,
    summarize_pareto_records,
)


def _record(mode: str, tokens: list[list[int]]) -> dict[str, object]:
    return {
        "environment": {"git_commit": "abc123", "git_dirty": True},
        "mode": {
            "name": mode,
            "visual_pruning_keep_ratio": 0.5,
        },
        "workload": {
            "manifest_name": "test",
            "manifest_sha256": "0" * 64,
            "case_id": "mixed",
            "num_requests": 2,
            "max_tokens": 4,
        },
        "correctness": {"token_ids": tokens},
        "timing_ms": {"decode_step": {"median": 2.0 if mode == "off_eager" else 2.2}},
        "kv_cache": {
            "logical_prompt_tokens": 20,
            "physical_prompt_tokens": 20 if mode == "off_eager" else 12,
            "active_prompt_blocks": 2 if mode == "off_eager" else 1,
            "active_prompt_bytes": 200 if mode == "off_eager" else 100,
        },
    }


def test_stable_prefix_lengths_are_per_request() -> None:
    lengths = stable_prefix_lengths(
        [[1, 2, 3, 4], [5, 6, 7]],
        [[1, 2, 9, 4], [5, 6, 7]],
    )

    print(f"P6 stable prefixes: {lengths}")
    assert lengths == [2, 3]
    with pytest.raises(ValueError, match="request counts differ"):
        stable_prefix_lengths([[1]], [[1], [2]])
    print("P6 stable-prefix contract: PASS")


def test_pareto_summary_uses_matching_baseline_denominators() -> None:
    baseline = _record("off_eager", [[1, 2, 3, 4], [5, 6, 7, 8]])
    compact = _record("visual_compact", [[1, 2, 9, 4], [5, 6, 7, 8]])

    rows = summarize_pareto_records([baseline, compact])
    compact_row = rows[1]
    markdown = render_pareto_markdown(rows)

    print(f"P6 compact Pareto row: {compact_row}")
    assert compact_row["physical_token_ratio"] == 0.6
    assert compact_row["active_prompt_bytes_ratio"] == 0.5
    assert compact_row["stable_prefix_per_request"] == [2, 4]
    assert compact_row["stable_prefix_min"] == 2
    assert compact_row["token_exact"] is False
    assert compact_row["tpot_ratio"] == pytest.approx(1.1)
    assert "visual_compact" in markdown
    print("P6 Pareto baseline comparison: PASS")


def test_pareto_summary_rejects_missing_or_duplicate_baseline() -> None:
    compact = _record("visual_compact", [[1]])
    with pytest.raises(ValueError, match="missing 'off_eager' baseline"):
        summarize_pareto_records([compact])

    baseline = _record("off_eager", [[1]])
    duplicate = deepcopy(baseline)
    with pytest.raises(ValueError, match="duplicate Pareto cell"):
        summarize_pareto_records([baseline, duplicate])
    print("P6 Pareto baseline guards: PASS")
