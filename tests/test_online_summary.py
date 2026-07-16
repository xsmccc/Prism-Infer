"""P7.3 online matrix summary validation tests."""

from copy import deepcopy

import pytest

from prism_infer.analysis.online_serving import summarize_online_run
from prism_infer.analysis.online_summary import (
    render_online_summary_markdown,
    summarize_online_records,
)


def _record(*, mode: str, dirty: bool = False) -> dict:
    run = {
        "duration_s": 1.0,
        "requests": [
            {
                "request_key": "r0",
                "request_id": 1,
                "state": "finished",
                "token_ids": [7, 8],
                "finish_reason": "length",
            }
        ],
        "engine_metrics": {
            "requests": [
                {
                    "request_id": 1,
                    "output_tokens": 2,
                    "queue_ms": 1.0,
                    "ttft_ms": 10.0,
                    "tpot_ms": 5.0,
                    "latency_ms": 15.0,
                    "finish_reason": "length",
                }
            ],
            "batches": [],
        },
        "scheduler_metrics": {
            "policy": "fcfs",
            "peak_active": 1,
            "peak_gpu_kv_blocks": 1,
            "swap_preemptions": 0,
            "recompute_preemptions": 0,
        },
    }
    summary = summarize_online_run(
        run,
        ttft_slo_ms=100.0,
        tpot_slo_ms=20.0,
    )
    return {
        "schema_version": 1,
        "record_type": "prism_online_run",
        "git_commit": "abc123",
        "git_dirty": dirty,
        "hardware": {
            "gpu": "test",
            "gpu_uuid": "GPU-test",
            "total_memory_bytes": 1,
        },
        "workload": {
            "manifest": "test",
            "case": "case",
            "requests": 1,
            "max_tokens": 2,
        },
        "arrival": {
            "process": "constant",
            "request_rate_per_s": 1.0,
            "seed": 7,
            "offsets_s": [0.0],
        },
        "engine": {
            "mode": mode,
            "max_model_len": 32,
            "max_num_batched_tokens": 32,
            "max_num_seqs": 4,
            "max_chunk_size": 8,
            "num_kvcache_blocks": 8,
            "kvcache_block_size": 4,
            "enable_prefix_caching": False,
        },
        "run": run,
        "summary": summary,
    }


def test_online_matrix_summary_is_clean_and_renderable() -> None:
    summary = summarize_online_records(
        (_record(mode="off_graph"), _record(mode="visual_compact_graph"))
    )

    assert summary["cell_count"] == 2
    assert summary["all_clean"]
    assert summary["commits"] == ["abc123"]
    assert summary["rows"][0]["goodput_fraction"] == 1.0
    markdown = render_online_summary_markdown(summary)
    assert "Prism Online Serving Matrix" in markdown
    assert "off_graph" in markdown


def test_online_matrix_summary_rejects_duplicate_and_tamper() -> None:
    record = _record(mode="off_graph")
    with pytest.raises(ValueError, match="duplicate online benchmark cell"):
        summarize_online_records((record, deepcopy(record)))

    tampered = deepcopy(record)
    tampered["summary"]["goodput"]["requests_per_s"] = 99.0
    with pytest.raises(ValueError, match="does not match"):
        summarize_online_records((tampered,))
