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


def _p7_records() -> tuple[dict[str, Any], dict[str, Any]]:
    prism = _prism_record()
    prism.update(
        {
            "environment": {
                "git_commit": "harness123",
                "git_dirty": False,
                "gpu_uuid": "GPU-123",
            },
            "model": {
                "config_sha256": "a" * 64,
                "dtype": "torch.bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": 1280,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 1,
                "kvcache_block_size": 256,
                "prefix_caching_enabled": False,
                "chunked_prefill_enabled": False,
            },
            "mode": {
                "name": "off_graph",
                "execution": "cuda_graph",
                "visual_pruning_keep_ratio": 0.5,
            },
            "traffic": {"kind": "offline_closed_loop"},
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": 4,
            },
            "measurement": {
                "warmup": 2,
                "repeat": 5,
                "cuda_synchronize_timing": True,
            },
            "kv_cache": {
                "bytes": 1024,
                "physical_prompt_tokens": 10,
                "active_prompt_bytes": 1024,
            },
        }
    )
    prism["workload"].update(
        {
            "preprocessing_included_in_e2e": True,
            "output_decoding_included_in_e2e": False,
        }
    )

    external = _external_record()
    external.update(
        {
            "schema_version": 2,
            "protocol": {
                "name": "p7.1_external_offline_v2",
                "comparison_profile": "best_stable",
                "harness_git_commit": "harness123",
                "harness_git_dirty": False,
                "framework_source_dirty": False,
                "process_scope": "fresh_process_per_case_and_backend",
                "command": ["python", "bench.py"],
            },
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": 4,
            },
        }
    )
    external["environment"].update(
        {
            "gpu_uuid": "GPU-123",
            "driver": "1.0",
            "compute_capability": "12.0",
        }
    )
    external["model"].update(
        {
            "config_sha256": "a" * 64,
            "dtype": "torch.bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": 1280,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 1,
            "kv_cache_memory_bytes": 1024,
        }
    )
    external["backend"].update(
        {
            "execution": "cuda_graph",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "compilation_mode": "VLLM_COMPILE",
            "block_size": 256,
            "prefix_caching": False,
        }
    )
    external["workload"].update(
        {
            "traffic": "offline_closed_loop",
            "preprocessing_included_in_e2e": True,
            "output_decoding_included_in_e2e": False,
        }
    )
    external["measurement"] = {
        "warmup": 2,
        "repeat": 5,
        "cuda_synchronize_timing": True,
    }
    return prism, external


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


def test_p7_external_comparison_requires_every_fairness_gate(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(comparison, "validate_benchmark_record", lambda record: None)
    prism, external = _p7_records()

    row = comparison.compare_external_records(
        [prism],
        [external],
        prism_mode="off_graph",
        comparison_profile="best_stable",
    )[0]

    assert row["performance_comparable"] is True
    assert all(row["comparability_checks"].values())
    assert row["non_comparable_reasons"] == []
    assert row["external_cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert row["external_to_prism_tpot_ratio"] == 0.5

    external["protocol"]["harness_git_dirty"] = True
    blocked = comparison.compare_external_records(
        [prism],
        [external],
        prism_mode="off_graph",
        comparison_profile="best_stable",
    )[0]
    assert blocked["performance_comparable"] is False
    assert blocked["external_to_prism_tpot_ratio"] is None
    assert blocked["non_comparable_reasons"] == ["clean_harness"]


def test_p7_external_comparison_rejects_execution_profile_mismatch(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(comparison, "validate_benchmark_record", lambda record: None)
    prism, external = _p7_records()
    external["backend"]["execution"] = "eager"
    external["backend"]["cudagraph_mode"] = "NONE"

    row = comparison.compare_external_records(
        [prism],
        [external],
        prism_mode="off_graph",
        comparison_profile="best_stable",
    )[0]

    assert row["performance_comparable"] is False
    assert "execution_profile" in row["non_comparable_reasons"]
