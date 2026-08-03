"""Focused contracts for the SGLang working-set harness."""

from types import SimpleNamespace

import pytest

sglang_harness = pytest.importorskip("benchmarks.bench_online_sglang")


def _working_set() -> SimpleNamespace:
    return SimpleNamespace(
        plan={
            "kv_budget": {
                "bytes": 4_282_122_240,
                "pages": 220,
                "page_size_tokens": 256,
            },
            "serving": {"max_num_seqs": 8, "max_chunk_size": 8192},
        }
    )


def test_working_set_kv_dtype_fails_closed() -> None:
    assert sglang_harness._resolve_working_set_kv_cache_dtype("auto") == "fp8_e4m3"
    assert sglang_harness._resolve_working_set_kv_cache_dtype("fp8_e4m3") == "fp8_e4m3"
    with pytest.raises(ValueError, match="requires --kv-cache-dtype"):
        sglang_harness._resolve_working_set_kv_cache_dtype("bfloat16")


def test_runtime_preserves_effective_token_and_byte_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_kwargs = {"min_pixels": 65_536, "max_pixels": 602_112}
    monkeypatch.setattr(
        sglang_harness,
        "working_set_processor_kwargs",
        lambda _plan: processor_kwargs,
    )
    kv_bytes_per_token = 73_728
    raw_capacity = 4_282_122_240 // kv_bytes_per_token
    max_total_tokens = (raw_capacity // 256) * 256
    server_args = SimpleNamespace(
        max_total_tokens=max_total_tokens,
        page_size=256,
        max_running_requests=8,
        chunked_prefill_size=8192,
        kv_cache_dtype="fp8_e4m3",
        mm_process_config={"image": processor_kwargs},
    )
    engine = SimpleNamespace(server_args=server_args)
    verification = sglang_harness._verify_sglang_working_set_runtime(
        engine,
        _working_set(),
        kv_bytes_per_token=kv_bytes_per_token,
        processor_verification={"image_size": {"shortest_edge": 65_536, "longest_edge": 602_112}},
        mm_process_config={"image": processor_kwargs},
    )
    assert verification["max_total_tokens"] == max_total_tokens
    assert verification["kv_capacity_bytes"] <= 4_282_122_240
    assert verification["unused_budget_bytes"] < 256 * kv_bytes_per_token
    assert verification["chunked_prefill_size"] == 8192
    assert all(verification["checks"].values())

    server_args.kv_cache_dtype = "bfloat16"
    with pytest.raises(RuntimeError, match="kv_cache_dtype"):
        sglang_harness._verify_sglang_working_set_runtime(
            engine,
            _working_set(),
            kv_bytes_per_token=kv_bytes_per_token,
            processor_verification={
                "image_size": {"shortest_edge": 65_536, "longest_edge": 602_112}
            },
            mm_process_config={"image": processor_kwargs},
        )


def test_resources_close_sampler_engine_and_images() -> None:
    closed: list[str] = []
    resources = sglang_harness._RunResources()
    resources.memory_sampler = SimpleNamespace(close=lambda: closed.append("sampler"))
    resources.engine = SimpleNamespace(shutdown=lambda: closed.append("engine"))
    resources.working_set = SimpleNamespace(close=lambda: closed.append("images"))

    resources.close(suppress_errors=False)

    assert closed == ["sampler", "engine", "images"]
