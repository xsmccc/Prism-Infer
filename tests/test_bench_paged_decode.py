"""Paged decode structured benchmark helper tests."""

from __future__ import annotations

import json

import pytest
import torch

from benchmarks.bench_paged_decode import (
    Measurement,
    _build_case_record,
    _parse_positive_int_csv,
    _resolve_cache_dtypes,
    _resolve_output_format,
    _write_output,
)


def test_parse_positive_int_csv_preserves_declared_matrix_order() -> None:
    """Page matrix 顺序应稳定，不能静默排序或去重。"""

    assert _parse_positive_int_csv(
        "16,32,256", option_name="--page-sizes"
    ) == (16, 32, 256)


@pytest.mark.parametrize("raw", ["", "16,", "0,16", "-1", "16,16", "x,16"])
def test_parse_positive_int_csv_rejects_ambiguous_values(raw: str) -> None:
    """空值、非正数、重复或非整数必须在运行 GPU workload 前失败。"""

    with pytest.raises(ValueError):
        _parse_positive_int_csv(raw, option_name="--page-sizes")


def test_resolve_cache_dtypes_rejects_unknown_or_duplicate_modes() -> None:
    """量化模式不能 silent fallback 到 BF16。"""

    assert _resolve_cache_dtypes("bf16") == (("bf16", torch.bfloat16),)
    with pytest.raises(ValueError, match="unsupported cache dtype"):
        _resolve_cache_dtypes("int4")
    with pytest.raises(ValueError, match="unique"):
        _resolve_cache_dtypes("bf16,bf16")


def test_build_case_record_keeps_samples_correctness_and_physical_bytes() -> None:
    """每个结构化 cell 应保留 raw samples、数值门禁与真实 tensor bytes。"""

    q = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    k_cache = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    context_lens = torch.tensor([4], dtype=torch.int32)
    output = torch.zeros_like(q)
    memory = {
        "allocated_before": 64,
        "allocated_after": 80,
        "reserved_after": 128,
        "peak_allocated": 96,
        "peak_delta": 32,
    }
    measurement = Measurement(
        samples_ms=(1.0, 2.0, 3.0),
        output=output,
        memory_bytes=memory,
    )

    record = _build_case_record(
        cache_dtype_name="bf16",
        batch=1,
        context_len=4,
        page_size=4,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        seed=7,
        scale=0.5,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        kernel=measurement,
        reference=measurement,
        max_abs_diff_limit=1e-2,
        mean_abs_diff_limit=1e-3,
    )

    assert record["correctness"]["passed"] is True
    assert record["latency"]["kernel"]["samples_ms"] == [1.0, 2.0, 3.0]
    assert record["latency"]["kernel"]["stats_ms"]["median"] == 2.0
    expected_cache_bytes = k_cache.numel() * k_cache.element_size() * 2
    assert record["memory_bytes"]["physical_kv_payload"] == expected_cache_bytes


def test_write_output_supports_json_and_jsonl_without_silent_overwrite(
    tmp_path,
) -> None:
    """正式 artifact 可选 JSON/JSONL，默认不覆盖已有证据。"""

    run = {"status": "passed", "git": {"commit": "abc", "dirty": False}}
    cases = [{"case_id": "bf16_page16_batch1_context4"}]
    json_path = tmp_path / "record.json"
    jsonl_path = tmp_path / "record.jsonl"

    _write_output(
        json_path,
        output_format="json",
        overwrite=False,
        run=run,
        cases=cases,
    )
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["record_type"] == "prism_paged_decode_benchmark_run"
    assert document["cases"] == cases

    _write_output(
        jsonl_path,
        output_format="jsonl",
        overwrite=False,
        run=run,
        cases=cases,
    )
    line = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert line["record_type"] == "prism_paged_decode_benchmark_case"
    assert line["case"] == cases[0]
    assert _resolve_output_format(json_path, "auto") == "json"
    assert _resolve_output_format(jsonl_path, "auto") == "jsonl"

    with pytest.raises(FileExistsError):
        _write_output(
            json_path,
            output_format="json",
            overwrite=False,
            run=run,
            cases=cases,
        )
