"""P6 external benchmark comparison contract 测试。"""

from typing import Any

import prism_infer.analysis.external_comparison as comparison


def _external_record(prompt_tokens: int = 10) -> dict[str, Any]:
    stats = {"count": 3, "median": 2.0}
    return {
        "schema_version": 1,
        "record_type": "external_system_benchmark",
        "environment": {
            "framework": "vllm",
            "framework_version": "1.0",
            "framework_source_commit": "abc123",
            "gpu": "GPU",
        },
        "model": {},
        "backend": {},
        "workload": {
            "manifest_sha256": "0" * 64,
            "case_id": "image",
            "num_requests": 1,
            "max_tokens": 4,
            "prompt_tokens": prompt_tokens,
        },
        "measurement": {
            "warmup": 1,
            "repeat": 3,
            "cuda_synchronize_timing": True,
        },
        "correctness": {
            "outputs_identical_across_repeats": True,
            "token_ids": [[1, 2, 9, 4]],
        },
        "timing_ms": {
            "end_to_end": stats,
            "engine_ttft": stats,
            "decode_tpot": {"count": 3, "median": 1.0},
        },
        "throughput": {
            "e2e_output_tokens_per_s": {"count": 3, "median": 20.0}
        },
        "memory_mb": {"peak_allocated": {"count": 3, "median": 80.0}},
    }


def _prism_record() -> dict[str, Any]:
    return {
        "mode": {"name": "off_eager", "visual_pruning_keep_ratio": 0.5},
        "workload": {
            "manifest_sha256": "0" * 64,
            "case_id": "image",
            "num_requests": 1,
            "max_tokens": 4,
            "prompt_tokens": 10,
        },
        "correctness": {"token_ids": [[1, 2, 3, 4]]},
        "timing_ms": {"decode_step": {"median": 2.0}},
        "throughput": {"e2e_output_tokens_per_s": {"median": 10.0}},
        "memory_mb": {"peak_allocated": {"median": 100.0}},
    }


def test_external_comparison_matches_input_contract(monkeypatch: Any) -> None:
    monkeypatch.setattr(comparison, "validate_benchmark_record", lambda record: None)

    rows = comparison.compare_external_records(
        [_prism_record()],
        [_external_record()],
    )
    row = rows[0]

    print(f"P6 external comparison row: {row}")
    assert row["performance_comparable"] is True
    assert row["stable_prefix_per_request"] == [2]
    assert row["external_to_prism_tpot_ratio"] == 0.5
    assert row["external_to_prism_throughput_ratio"] == 2.0
    assert row["external_to_prism_peak_memory_ratio"] == 0.8
    print("P6 external comparison matched-input contract: PASS")


def test_external_comparison_blocks_mismatched_prompt_ratios(monkeypatch: Any) -> None:
    monkeypatch.setattr(comparison, "validate_benchmark_record", lambda record: None)

    row = comparison.compare_external_records(
        [_prism_record()],
        [_external_record(prompt_tokens=9)],
    )[0]
    markdown = comparison.render_external_markdown([row])

    print(f"P6 external mismatched-input row: {row}")
    assert row["performance_comparable"] is False
    assert row["external_to_prism_tpot_ratio"] is None
    assert row["external_to_prism_throughput_ratio"] is None
    assert "| no |" in markdown
    assert "n/a" in markdown
    print("P6 external mismatched-input guard: PASS")
