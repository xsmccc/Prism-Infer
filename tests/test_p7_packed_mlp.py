"""P7.5 packed gate/up projection contracts。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from prism_infer.models.qwen3_vl import Qwen3VLTextMLP, Qwen3VLTextModel
from benchmarks.bench_packed_mlp import (
    _formal_environment_issues,
    _summarize,
)


def _legacy_forward(mlp: Qwen3VLTextMLP, x: torch.Tensor) -> torch.Tensor:
    return F.linear(
        F.silu(F.linear(x, mlp.gate_proj.weight))
        * F.linear(x, mlp.up_proj.weight),
        mlp.down_proj.weight,
    )


def test_packed_gate_up_shares_storage_and_matches_legacy_forward() -> None:
    torch.manual_seed(20260717)
    mlp = Qwen3VLTextMLP(
        hidden_size=16,
        intermediate_size=24,
        dtype=torch.float32,
    ).eval()
    x = torch.randn(4, 16)

    packed_ptr = mlp.gate_up_proj.weight.untyped_storage().data_ptr()
    assert mlp.gate_proj.weight.untyped_storage().data_ptr() == packed_ptr
    assert mlp.up_proj.weight.untyped_storage().data_ptr() == packed_ptr
    assert mlp.up_proj.weight.storage_offset() == 24 * 16

    with torch.inference_mode():
        actual = mlp(x)
        expected = _legacy_forward(mlp, x)
    # CPU BLAS may select a different reduction shape for the packed GEMM.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_packed_gate_up_preserves_legacy_state_dict_contract() -> None:
    torch.manual_seed(20260717)
    source = Qwen3VLTextMLP(16, 24, torch.float32).state_dict()
    assert set(source) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }

    restored = Qwen3VLTextMLP(16, 24, torch.float32).eval()
    restored.load_state_dict(source, strict=True)
    assert torch.equal(restored.gate_proj.weight, source["gate_proj.weight"])
    assert torch.equal(restored.up_proj.weight, source["up_proj.weight"])
    assert (
        restored.gate_proj.weight.untyped_storage().data_ptr()
        == restored.gate_up_proj.weight.untyped_storage().data_ptr()
    )


def test_packed_gate_up_rebinds_views_after_dtype_conversion() -> None:
    mlp = Qwen3VLTextMLP(16, 24, torch.float32).to(dtype=torch.float64)
    packed_ptr = mlp.gate_up_proj.weight.untyped_storage().data_ptr()
    assert mlp.gate_proj.weight.dtype == torch.float64
    assert mlp.up_proj.weight.dtype == torch.float64
    assert mlp.gate_proj.weight.untyped_storage().data_ptr() == packed_ptr
    assert mlp.up_proj.weight.untyped_storage().data_ptr() == packed_ptr


def test_packed_gate_up_executes_one_projection() -> None:
    mlp = Qwen3VLTextMLP(16, 24, torch.float32).eval()
    calls = {"packed": 0, "gate": 0, "up": 0}
    hooks = [
        mlp.gate_up_proj.register_forward_hook(
            lambda *_: calls.__setitem__("packed", calls["packed"] + 1)
        ),
        mlp.gate_proj.register_forward_hook(
            lambda *_: calls.__setitem__("gate", calls["gate"] + 1)
        ),
        mlp.up_proj.register_forward_hook(
            lambda *_: calls.__setitem__("up", calls["up"] + 1)
        ),
    ]
    try:
        with torch.inference_mode():
            output = mlp(torch.randn(2, 16))
    finally:
        for hook in hooks:
            hook.remove()
    assert list(output.shape) == [2, 16]
    assert calls == {"packed": 1, "gate": 0, "up": 0}


def test_legacy_mode_executes_two_projections_with_identical_weights() -> None:
    packed = Qwen3VLTextMLP(
        16,
        24,
        torch.float32,
        projection_mode="packed",
    ).eval()
    legacy = Qwen3VLTextMLP(
        16,
        24,
        torch.float32,
        projection_mode="legacy",
    ).eval()
    legacy.load_state_dict(packed.state_dict(), strict=True)
    calls = {"packed": 0, "gate": 0, "up": 0}
    hooks = [
        legacy.gate_up_proj.register_forward_hook(
            lambda *_: calls.__setitem__("packed", calls["packed"] + 1)
        ),
        legacy.gate_proj.register_forward_hook(
            lambda *_: calls.__setitem__("gate", calls["gate"] + 1)
        ),
        legacy.up_proj.register_forward_hook(
            lambda *_: calls.__setitem__("up", calls["up"] + 1)
        ),
    ]
    x = torch.randn(2, 16)
    try:
        with torch.inference_mode():
            packed_output = packed(x)
            legacy_output = legacy(x)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(packed_output, legacy_output, rtol=1e-5, atol=1e-7)
    assert calls == {"packed": 0, "gate": 1, "up": 1}


def test_mlp_projection_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="projection_mode"):
        Qwen3VLTextMLP(16, 24, torch.float32, projection_mode="auto")


def test_projection_mode_propagates_to_every_decoder_layer() -> None:
    model = Qwen3VLTextModel(
        vocab_size=32,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        num_layers=3,
        intermediate_size=16,
        dtype=torch.float32,
        head_dim=4,
        mrope_section=[1, 1, 0],
        mlp_projection_mode="legacy",
    )
    assert [layer.mlp.projection_mode for layer in model.layers] == [
        "legacy",
        "legacy",
        "legacy",
    ]


def test_packed_mlp_benchmark_formal_environment_gate() -> None:
    clean_idle = {
        "memory_used_mb": 4,
        "utilization_gpu_percent": 0,
    }
    assert not _formal_environment_issues(
        git_dirty=False,
        gpu_baseline=clean_idle,
        max_baseline_memory_mb=1024,
        max_baseline_utilization_percent=5,
    )

    contaminated = {
        "memory_used_mb": 17282,
        "utilization_gpu_percent": 34,
    }
    issues = _formal_environment_issues(
        git_dirty=True,
        gpu_baseline=contaminated,
        max_baseline_memory_mb=1024,
        max_baseline_utilization_percent=5,
    )
    assert issues == [
        "git worktree is dirty",
        "baseline GPU memory exceeds limit: 17282 > 1024 MiB",
        "baseline GPU utilization exceeds limit: 34 > 5%",
    ]


def test_packed_mlp_benchmark_statistics_use_nearest_rank() -> None:
    assert _summarize([1.0, 2.0, 3.0, 4.0, 5.0]) == {
        "count": 5,
        "median": 3.0,
        "p90": 5.0,
        "p99": 5.0,
        "min": 1.0,
        "max": 5.0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_shape_bf16_cuda_packed_gate_up_is_bitwise_exact() -> None:
    torch.manual_seed(20260717)
    mlp = Qwen3VLTextMLP(
        hidden_size=4096,
        intermediate_size=12288,
        dtype=torch.bfloat16,
    ).cuda().eval()
    try:
        with torch.inference_mode():
            for batch in (1, 2, 4, 8, 210, 408, 988):
                x = torch.randn(
                    batch,
                    4096,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                actual = mlp(x)
                expected = _legacy_forward(mlp, x)
                assert torch.equal(actual, expected), (
                    f"packed gate/up diverged for batch={batch}: "
                    f"max_diff={(actual - expected).abs().max().item()}"
                )
    finally:
        del mlp
        torch.cuda.empty_cache()
