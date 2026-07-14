"""P6 benchmark schema 与 deterministic workload contract 测试。"""

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.bench_system import (
    MODE_SPECS,
    _annotate_comparisons,
    _describe_case_inputs,
    _expand_case_batch,
    _materialize_requests,
    _parse_keep_ratios,
    _parse_positive_ints,
)
from prism_infer.analysis.benchmark_schema import (
    BENCHMARK_SCHEMA_VERSION,
    canonical_json_sha256,
    load_workload_manifest,
    summarize_values,
    validate_benchmark_record,
    validate_workload_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/workloads/p6_internal_smoke.json"
REAL_MANIFEST = REPO_ROOT / "benchmarks/workloads/p6_real_samples.json"


def _stats() -> dict[str, int | float]:
    return {
        "count": 3,
        "median": 2.0,
        "p90": 3.0,
        "p99": 3.0,
        "min": 1.0,
        "max": 3.0,
    }


def _complete_record() -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "record_type": "system_benchmark",
        "timestamp_utc": "2026-07-11T00:00:00+00:00",
        "environment": {
            "git_commit": "abc123",
            "git_dirty": False,
            "python": "3.12.0",
            "torch": "2.6.0",
            "transformers": "5.13.0",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "model": {
            "path": "/data/models/qwen3-vl",
            "dtype": "torch.bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": 1280,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 1,
            "kvcache_block_size": 256,
            "num_kvcache_blocks": 16,
            "gpu_memory_utilization": 0.9,
        },
        "mode": {
            "name": "off_eager",
            "execution": "eager",
            "attention": "paged_or_sdpa",
            "compression": "off",
            "visual_pruning_keep_ratio": 0.5,
            "visual_pruning_min_keep_tokens": 32,
            "visual_pruning_strategy": "uniform",
        },
        "workload": {
            "manifest_name": "p6_internal_smoke",
            "manifest_sha256": "0" * 64,
            "case_id": "single_image_448",
            "request_types": ["image"],
            "input_shapes": [
                {"type": "image", "visual_shapes": [[448, 448, 3]]}
            ],
            "num_requests": 1,
            "source_num_requests": 1,
            "request_replication_factor": 1,
            "prompt_tokens": 210,
            "image_tokens": 196,
            "video_tokens": 0,
            "image_count": 1,
            "video_count": 0,
            "video_frame_count": 0,
            "max_tokens": 8,
            "preprocessing_included_in_e2e": True,
        },
        "traffic": {
            "kind": "offline_closed_loop",
            "batch_size": 1,
            "concurrency": 1,
            "request_rate_per_s": None,
        },
        "execution_backend": {
            "prefill_backend": "eager",
            "decode_backend": "eager",
            "cuda_graph_enabled": False,
            "cuda_graph_capture_scope": "none",
            "cuda_graph_capture_ms": 0.0,
            "cuda_graph_batch_sizes": [],
            "requested_decode_batch_size": 1,
            "selected_decode_batch_size": 1,
            "decode_batch_padding": 0,
            "torch_compile_enabled": False,
            "torch_compile_region": "none",
            "torch_compile_backend": "none",
            "torch_compile_mode": "none",
            "torch_compile_emulate_precision_casts": False,
            "torch_compile_force_same_precision": False,
            "torch_compile_first_call_ms": 0.0,
        },
        "measurement": {
            "warmup": 1,
            "repeat": 3,
            "cuda_synchronize_timing": True,
        },
        "correctness": {
            "outputs_identical_across_repeats": True,
            "token_ids": [[785, 2168]],
            "output_tokens": 2,
            "output_sha256": canonical_json_sha256([[785, 2168]]),
        },
        "timing_ms": {
            "preprocessing": _stats(),
            "engine_ttft": _stats(),
            "end_to_end_ttft": _stats(),
            "prefill": _stats(),
            "decode_step": _stats(),
            "end_to_end": _stats(),
        },
        "throughput": {
            "engine_output_tokens_per_s": _stats(),
            "e2e_output_tokens_per_s": _stats(),
            "decode_tokens_per_s": _stats(),
            "engine_requests_per_s": _stats(),
            "e2e_requests_per_s": _stats(),
        },
        "memory_mb": {
            "allocated": _stats(),
            "reserved": _stats(),
            "peak_allocated": _stats(),
        },
        "kv_cache": {
            "dtype": "torch.bfloat16",
            "shape": [2, 36, 16, 256, 8, 128],
            "bytes": 603979776,
            "blocks": 16,
            "block_size": 256,
            "capacity_tokens": 4096,
            "logical_prompt_tokens": 210,
            "physical_prompt_tokens": 210,
            "dense_prompt_blocks": 1,
            "active_prompt_blocks": 1,
            "released_prompt_blocks": 0,
            "dense_prompt_bytes": 37748736,
            "active_prompt_bytes": 37748736,
            "layouts": [
                {
                    "schema_version": 1,
                    "mode": "dense",
                    "logical_context_len": 210,
                    "physical_kv_len": 210,
                    "prompt_logical_len": 210,
                    "compressed_prompt_kv_len": 210,
                    "retained_original_positions": [],
                    "block_table": [0],
                    "kv_dtype": "torch.bfloat16",
                    "compression_record": {},
                }
            ],
        },
    }


def test_summarize_values_reports_required_percentiles() -> None:
    summary = summarize_values([4.0, 1.0, 3.0, 2.0])

    print(f"benchmark summary: {summary}")
    assert summary == {
        "count": 4,
        "median": 2.5,
        "p90": 4.0,
        "p99": 4.0,
        "min": 1.0,
        "max": 4.0,
    }
    print("P6 benchmark summary contract: PASS")


def test_default_workload_manifest_is_valid_and_unique() -> None:
    manifest = load_workload_manifest(DEFAULT_MANIFEST)
    case_ids = [case["id"] for case in manifest["cases"]]
    request_types = sorted(
        {
            request["type"]
            for case in manifest["cases"]
            for request in case["requests"]
        }
    )

    print(f"benchmark workload cases: {case_ids}")
    print(f"benchmark workload request types: {request_types}")
    assert len(case_ids) == len(set(case_ids)) == 5
    assert request_types == ["image", "images", "text", "video"]
    print("P6 deterministic workload manifest: PASS")


def test_real_workload_manifest_has_auditable_asset_identity() -> None:
    manifest = load_workload_manifest(REAL_MANIFEST)
    request = manifest["cases"][0]["requests"][0]
    image = request["image"]

    print(f"P6 real workload image metadata: {image}")
    assert request["type"] == "image_file"
    assert image["path"] == "data/p6_real_samples/000000039769.jpg"
    assert image["sha256"] == (
        "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"
    )
    assert (image["width"], image["height"]) == (640, 480)
    print("P6 real workload manifest contract: PASS")


def test_real_workload_materialization_checks_file_identity() -> None:
    manifest = load_workload_manifest(REAL_MANIFEST)
    case = manifest["cases"][0]
    image_path = REPO_ROOT / case["requests"][0]["image"]["path"]
    if not image_path.is_file():
        pytest.skip("run scripts/download_p6_real_samples.sh to install the sample")

    requests = _materialize_requests(case)
    input_shapes, image_count, video_count, frame_count = _describe_case_inputs(case)

    print(
        "P6 real workload materialized: "
        f"type={requests[0]['type']} size={requests[0]['image'].size} "
        f"shapes={input_shapes}"
    )
    assert requests[0]["type"] == "image"
    assert requests[0]["image"].mode == "RGB"
    assert requests[0]["image"].size == (640, 480)
    assert input_shapes == [
        {"type": "image_file", "visual_shapes": [[480, 640, 3]]}
    ]
    assert (image_count, video_count, frame_count) == (1, 0, 0)

    invalid_case = deepcopy(case)
    invalid_case["requests"][0]["image"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _materialize_requests(invalid_case)
    print("P6 real workload file identity: PASS")


def test_real_workload_manifest_rejects_missing_asset_hash() -> None:
    manifest = load_workload_manifest(REAL_MANIFEST)
    invalid = deepcopy(manifest)
    del invalid["cases"][0]["requests"][0]["image"]["sha256"]

    with pytest.raises(ValueError, match="sha256"):
        validate_workload_manifest(invalid)
    print("P6 real workload missing-hash guard: PASS")


def test_execution_matrix_expands_single_request_batch_explicitly() -> None:
    case = {
        "id": "single",
        "requests": [{"type": "text", "prompt": "hello"}],
    }

    expanded, source_size, replication = _expand_case_batch(case, 4)

    print(f"expanded execution batch: {expanded}")
    assert len(expanded["requests"]) == 4
    assert source_size == 1
    assert replication == 4
    assert expanded["requests"][0] is not expanded["requests"][1]
    assert len(case["requests"]) == 1
    print("P6 execution matrix request replication: PASS")


def test_execution_matrix_rejects_partial_request_group() -> None:
    case = {
        "id": "mixed",
        "requests": [
            {"type": "text", "prompt": "one"},
            {"type": "text", "prompt": "two"},
        ],
    }

    with pytest.raises(ValueError, match="positive multiple"):
        _expand_case_batch(case, 3)
    print("P6 execution matrix partial-group guard: PASS")


def test_execution_matrix_axis_parser_preserves_requested_order() -> None:
    values = _parse_positive_ints("8,32,128", label="output", minimum=2)

    print(f"execution matrix parsed axis: {values}")
    assert values == [8, 32, 128]
    with pytest.raises(ValueError, match="duplicate"):
        _parse_positive_ints("1,2,1", label="batch")
    print("P6 execution matrix axis parser: PASS")


def test_p611_physical_compression_modes_have_eager_graph_pairs() -> None:
    """P6.11 benchmark 必须只比较同一种 physical compression 的执行后端。"""

    expected_pairs = (
        ("visual_compact", "visual_compact_graph"),
        ("fp8_kv", "fp8_kv_graph"),
        ("visual_compact_fp8", "visual_compact_fp8_graph"),
    )
    for eager_name, graph_name in expected_pairs:
        eager = MODE_SPECS[eager_name]
        graph = MODE_SPECS[graph_name]
        assert eager.compression == graph.compression
        assert eager.attention == graph.attention
        assert eager.execution == "eager"
        assert eager.enforce_eager
        assert graph.execution == "cuda_graph"
        assert not graph.enforce_eager
    assert MODE_SPECS["visual_prune"].enforce_eager
    print("P6.11 physical compression eager/Graph mode pairs: PASS")


def test_keep_ratio_matrix_parser_guards_range_and_duplicates() -> None:
    values = _parse_keep_ratios("0.25,0.5,0.75,1.0")

    print(f"P6 keep-ratio matrix axis: {values}")
    assert values == [0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        _parse_keep_ratios("0,0.5")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_keep_ratios("0.5,0.5")
    print("P6 keep-ratio matrix parser: PASS")


def test_execution_matrix_comparisons_are_scoped_per_cell() -> None:
    records = []
    for batch_size, max_tokens, mode, tokens in (
        (1, 8, "off_eager", [[1, 2]]),
        (1, 8, "off_graph", [[1, 2]]),
        (2, 8, "off_eager", [[3], [3]]),
        (2, 8, "off_graph", [[3], [3]]),
    ):
        records.append(
            {
                "workload": {
                    "manifest_sha256": "0" * 64,
                    "case_id": "single",
                    "num_requests": batch_size,
                    "max_tokens": max_tokens,
                },
                "mode": {
                    "name": mode,
                    "visual_pruning_keep_ratio": 0.5,
                },
                "correctness": {"token_ids": tokens},
                "kv_cache": {"bytes": 100},
            }
        )

    _annotate_comparisons(records)

    print(f"execution matrix comparisons: {records}")
    assert all(record["comparison_to_first_mode"]["token_exact"] for record in records)
    assert records[1]["comparison_to_first_mode"]["first_mode"] == "off_eager"
    assert records[3]["comparison_to_first_mode"]["first_mode"] == "off_eager"
    print("P6 execution matrix comparison scoping: PASS")


def test_workload_manifest_rejects_duplicate_case_ids() -> None:
    manifest = load_workload_manifest(DEFAULT_MANIFEST)
    invalid = deepcopy(manifest)
    invalid["cases"][1]["id"] = invalid["cases"][0]["id"]

    with pytest.raises(ValueError, match="duplicate workload case id"):
        validate_workload_manifest(invalid)
    print("P6 workload duplicate-id guard: PASS")


def test_complete_benchmark_record_passes_validation() -> None:
    record = _complete_record()
    validate_benchmark_record(record)

    output_hash = canonical_json_sha256(record["correctness"])
    print(f"benchmark record correctness hash: {output_hash}")
    assert len(output_hash) == 64
    print("P6 complete benchmark record: PASS")


def test_v1_benchmark_record_remains_readable() -> None:
    """Schema v2 validator 必须继续读取 P6.1 历史 v1 raw records。"""

    record = _complete_record()
    record["schema_version"] = 1
    del record["execution_backend"]
    del record["workload"]["source_num_requests"]
    del record["workload"]["request_replication_factor"]

    validate_benchmark_record(record)
    print("P6 benchmark schema v1 backward compatibility: PASS")


def test_v2_benchmark_record_validates_cuda_graph_execution() -> None:
    record = _complete_record()
    record["schema_version"] = 2
    record["mode"]["name"] = "off_graph"
    record["mode"]["execution"] = "cuda_graph"
    record["execution_backend"] = {
        "prefill_backend": "eager",
        "decode_backend": "cuda_graph",
        "cuda_graph_enabled": True,
        "cuda_graph_capture_scope": "decode_model_forward",
        "cuda_graph_capture_ms": 123.5,
        "cuda_graph_batch_sizes": [1, 2, 4],
        "requested_decode_batch_size": 3,
        "selected_decode_batch_size": 4,
        "decode_batch_padding": 1,
    }
    record["workload"]["num_requests"] = 3
    record["workload"]["source_num_requests"] = 1
    record["workload"]["request_replication_factor"] = 3
    record["workload"]["request_types"] = ["image"] * 3
    record["workload"]["input_shapes"] *= 3
    record["traffic"]["batch_size"] = 3
    record["traffic"]["concurrency"] = 3
    record["correctness"]["token_ids"] *= 3
    record["correctness"]["output_tokens"] = 6
    record["correctness"]["output_sha256"] = canonical_json_sha256(
        record["correctness"]["token_ids"]
    )

    validate_benchmark_record(record)
    print("P6 benchmark CUDA Graph execution metadata: PASS")


def test_v3_benchmark_record_validates_attention_compile_execution() -> None:
    record = _complete_record()
    record["mode"].update(
        {
            "name": "off_compile_attention",
            "execution": "torch_compile_attention",
        }
    )
    record["execution_backend"].update(
        {
            "decode_backend": "torch_compile_attention",
            "torch_compile_enabled": True,
            "torch_compile_region": "decode_attention",
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default",
            "torch_compile_emulate_precision_casts": True,
            "torch_compile_force_same_precision": True,
            "torch_compile_first_call_ms": 1234.5,
        }
    )

    validate_benchmark_record(record)
    print("P6 benchmark torch.compile execution metadata: PASS")


def test_v3_benchmark_rejects_compile_and_cuda_graph_combination() -> None:
    record = _complete_record()
    record["execution_backend"].update(
        {
            "decode_backend": "torch_compile_attention",
            "cuda_graph_enabled": True,
            "cuda_graph_capture_scope": "decode_model_forward",
            "cuda_graph_capture_ms": 10.0,
            "cuda_graph_batch_sizes": [1],
            "torch_compile_enabled": True,
            "torch_compile_region": "decode_attention",
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default",
            "torch_compile_emulate_precision_casts": True,
            "torch_compile_force_same_precision": True,
            "torch_compile_first_call_ms": 1234.5,
        }
    )

    with pytest.raises(ValueError, match="CUDA Graph execution requires"):
        validate_benchmark_record(record)
    print("P6 benchmark compile/graph mutual exclusion: PASS")


def test_v2_benchmark_rejects_inconsistent_replication() -> None:
    record = _complete_record()
    record["workload"]["request_replication_factor"] = 2

    with pytest.raises(ValueError, match="source_num_requests"):
        validate_benchmark_record(record)
    print("P6 benchmark request replication guard: PASS")


def test_v2_benchmark_rejects_inconsistent_graph_padding() -> None:
    record = _complete_record()
    record["execution_backend"].update(
        {
            "decode_backend": "cuda_graph",
            "cuda_graph_enabled": True,
            "cuda_graph_capture_scope": "decode_model_forward",
            "cuda_graph_capture_ms": 10.0,
            "cuda_graph_batch_sizes": [1, 2, 4],
            "selected_decode_batch_size": 2,
            "decode_batch_padding": 0,
        }
    )

    with pytest.raises(ValueError, match="decode_batch_padding"):
        validate_benchmark_record(record)
    print("P6 benchmark CUDA Graph padding guard: PASS")


def test_v4_benchmark_rejects_inconsistent_physical_kv_bytes() -> None:
    record = _complete_record()
    record["kv_cache"]["active_prompt_bytes"] -= 1

    with pytest.raises(ValueError, match="active prompt bytes"):
        validate_benchmark_record(record)
    print("P6 benchmark physical KV byte accounting guard: PASS")


@pytest.mark.parametrize(
    ("section", "key", "message"),
    [
        ("environment", "git_commit", "environment.git_commit"),
        ("workload", "image_tokens", "workload.image_tokens"),
        ("workload", "input_shapes", "workload.input_shapes"),
        ("timing_ms", "decode_step", "timing_ms.decode_step"),
        ("memory_mb", "peak_allocated", "memory_mb.peak_allocated"),
        ("kv_cache", "bytes", "kv_cache.bytes"),
    ],
)
def test_benchmark_record_rejects_missing_evidence(
    section: str,
    key: str,
    message: str,
) -> None:
    record = _complete_record()
    del record[section][key]

    with pytest.raises(ValueError, match=message):
        validate_benchmark_record(record)
    print(f"P6 benchmark missing-evidence guard: {section}.{key} PASS")


def test_benchmark_record_rejects_output_hash_mismatch() -> None:
    record = _complete_record()
    record["correctness"]["output_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not match token_ids"):
        validate_benchmark_record(record)
    print("P6 benchmark output-hash guard: PASS")


def test_benchmark_record_rejects_invalid_stat_order() -> None:
    record = _complete_record()
    record["timing_ms"]["decode_step"]["p90"] = 0.5

    with pytest.raises(ValueError, match="min <= median <= p90"):
        validate_benchmark_record(record)
    print("P6 benchmark statistic-order guard: PASS")


def test_benchmark_record_rejects_inconsistent_offline_traffic() -> None:
    record = _complete_record()
    record["traffic"]["concurrency"] = 2

    with pytest.raises(ValueError, match="must match workload.num_requests"):
        validate_benchmark_record(record)
    print("P6 benchmark offline-traffic consistency guard: PASS")
