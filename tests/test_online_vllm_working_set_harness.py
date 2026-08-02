"""Focused contracts for the vLLM working-set harness."""

from types import SimpleNamespace

import pytest

vllm_harness = pytest.importorskip("benchmarks.bench_online_vllm")


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


def test_working_set_formal_uses_clean_fixed_plan_contract() -> None:
    vllm_harness._validate_formal_contract(
        formal=True,
        git_dirty=False,
        has_working_set_plan=True,
        h3_conformance=None,
    )
    with pytest.raises(SystemExit, match="clean harness worktree"):
        vllm_harness._validate_formal_contract(
            formal=True,
            git_dirty=True,
            has_working_set_plan=True,
            h3_conformance=None,
        )


def test_working_set_kv_dtype_fails_closed() -> None:
    assert vllm_harness._resolve_working_set_kv_cache_dtype("auto") == "fp8_per_token_head"
    assert (
        vllm_harness._resolve_working_set_kv_cache_dtype("fp8_per_token_head")
        == "fp8_per_token_head"
    )
    with pytest.raises(ValueError, match="requires --kv-cache-dtype"):
        vllm_harness._resolve_working_set_kv_cache_dtype("bfloat16")


def test_runtime_audits_initialized_blocks_and_allocated_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_kwargs = {"min_pixels": 65_536, "max_pixels": 602_112}
    monkeypatch.setattr(
        vllm_harness,
        "working_set_processor_kwargs",
        lambda _plan: processor_kwargs,
    )
    text_config = SimpleNamespace(
        num_hidden_layers=36,
        num_key_value_heads=8,
        num_attention_heads=32,
        hidden_size=4096,
        head_dim=128,
    )
    model_config = SimpleNamespace(
        hf_text_config=text_config,
        multimodal_config=SimpleNamespace(
            mm_processor_kwargs=processor_kwargs,
            mm_processor_cache_gb=1.0,
        ),
    )
    cache_config = SimpleNamespace(
        kv_cache_memory_bytes=4_282_122_240,
        cache_dtype="fp8_per_token_head",
        num_gpu_blocks=220,
        block_size=256,
        enable_prefix_caching=True,
    )
    config = SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=8192,
        ),
        model_config=model_config,
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="TRITON_ATTN")),
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))
    verification = vllm_harness._verify_vllm_working_set_runtime(
        llm,
        _working_set(),
        processor_verification={"image_size": {"shortest_edge": 65_536, "longest_edge": 602_112}},
    )
    assert verification["num_gpu_blocks"] == 220
    assert verification["kv_bytes_per_block"] == 19_464_192
    assert verification["kv_allocated_bytes"] == 4_282_122_240
    assert verification["max_num_batched_tokens"] == 8192
    assert verification["attention_backend"] == "TRITON_ATTN"
    assert all(verification["checks"].values())

    cache_config.num_gpu_blocks = 219
    with pytest.raises(RuntimeError, match="num_gpu_blocks"):
        vllm_harness._verify_vllm_working_set_runtime(
            llm,
            _working_set(),
            processor_verification={
                "image_size": {"shortest_edge": 65_536, "longest_edge": 602_112}
            },
        )


def test_resources_close_sampler_engine_and_images() -> None:
    closed: list[str] = []
    resources = vllm_harness._RunResources()
    resources.memory_sampler = SimpleNamespace(close=lambda: closed.append("sampler"))
    resources.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: closed.append("engine"))
        )
    )
    resources.working_set = SimpleNamespace(close=lambda: closed.append("images"))

    resources.close(suppress_errors=False)

    assert closed == ["sampler", "engine", "images"]
