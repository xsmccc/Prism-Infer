"""P6.2 分层 performance profile collector 测试。"""

from copy import deepcopy

import pytest
import torch

from prism_infer.analysis.performance_profile import (
    get_performance_profile_session,
    is_performance_profile_enabled,
    performance_profile,
    profile_region,
    validate_performance_profile_record,
)
from prism_infer.layers.attention import Attention
from prism_infer.utils.context import reset_context, set_context


def test_disabled_profile_region_is_noop() -> None:
    """默认关闭时 profile region 不创建 session，也不改变代码结果。"""

    assert not is_performance_profile_enabled()
    assert get_performance_profile_session() is None
    value = 3
    with profile_region("disabled.cpu", cuda=False):
        value *= 7
    assert value == 21
    assert get_performance_profile_session() is None
    print("P6.2 disabled performance profile no-op: PASS")


def test_disabled_profile_region_does_not_break_dynamo_graph() -> None:
    """默认关闭的 profiling 不应切碎 ``torch.compile`` tensor graph。"""

    def function(x: torch.Tensor) -> torch.Tensor:
        with profile_region("compiled.unit", cuda=False):
            return torch.sin(x) + 1

    default_device = torch.get_default_device()
    torch.set_default_device(None)
    try:
        torch._dynamo.reset()
        output = torch._dynamo.explain(function)(torch.ones(2))
        torch._dynamo.reset()
    finally:
        torch.set_default_device(default_device)
    assert output.graph_count == 1
    assert output.graph_break_count == 0
    assert output.op_count == 2
    print("P6.3 compile/profile graph isolation: PASS")


def test_cpu_profile_builds_steps_regions_and_summary() -> None:
    """CPU collector 应生成自洽的 step、region 和分位数 summary。"""

    with performance_profile(
        metadata={"case": "cpu_unit"},
        cuda_timing=False,
    ) as session:
        with profile_region("preprocess.unit", cuda=False):
            sum(range(16))
        session.begin_step()
        session.annotate_step(phase="decode", batch_size=1)
        with profile_region(
            "runner.unit",
            cuda=False,
            metadata={"shape": [1, 4]},
        ):
            sum(range(32))
        session.end_step()

    record = session.to_record()
    validate_performance_profile_record(record)

    assert record["metadata"] == {"case": "cpu_unit"}
    assert record["cuda_timing"] is False
    assert record["steps"] == [{"step_id": 0, "phase": "decode", "batch_size": 1, "status": "ok"}]
    assert [region["name"] for region in record["regions"]] == [
        "preprocess.unit",
        "runner.unit",
    ]
    assert record["regions"][0]["step_id"] is None
    assert record["regions"][0]["phase"] == "preprocess"
    assert record["regions"][1]["step_id"] == 0
    assert record["regions"][1]["phase"] == "decode"
    assert record["summary"]["runner.unit"]["calls"] == 1
    assert record["summary"]["runner.unit"]["cuda_ms"] is None
    assert record["summary_by_phase"]["decode"]["runner.unit"]["calls"] == 1
    print("P6.2 CPU performance profile schema: PASS")


def _run_profiled_prefill(enable_profile: bool) -> tuple[torch.Tensor, dict | None]:
    """运行一个 deterministic CPU prefill attention case。"""

    torch.manual_seed(20260711)
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 1, 8)
    v = torch.randn(4, 1, 8)
    attention = Attention(num_heads=2, num_kv_heads=1, head_dim=8, scale=8**-0.5)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    def run() -> torch.Tensor:
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            slot_mapping=torch.arange(4, dtype=torch.int32),
        )
        try:
            return attention(q, k, v)
        finally:
            reset_context()

    if not enable_profile:
        return run(), None
    with performance_profile(cuda_timing=False) as session:
        session.begin_step()
        session.annotate_step(phase="prefill", batch_size=1)
        output = run()
        session.end_step()
    return output, session.to_record()


def test_profiled_attention_is_exact_noop() -> None:
    """开启 CPU profiling 不应改变 attention 数值输出。"""

    output_off, _ = _run_profiled_prefill(False)
    output_on, record = _run_profiled_prefill(True)
    diff = (output_off - output_on).abs()

    assert record is not None
    assert torch.equal(output_off, output_on)
    assert "attention.prefill.sdpa" in record["summary"]
    print(f"profile off output shape: {list(output_off.shape)}")
    print(f"profile on output shape: {list(output_on.shape)}")
    print(f"profile on/off max diff: {diff.max().item():.6e}")
    print("P6.2 performance profile attention exact no-op: PASS")


def test_profile_validator_rejects_tampered_summary() -> None:
    """profile validator 必须拒绝与 raw regions 不一致的 summary。"""

    with performance_profile(cuda_timing=False) as session:
        session.begin_step()
        session.annotate_step(phase="decode", batch_size=1)
        with profile_region("runner.unit", cuda=False):
            sum(range(8))
        session.end_step()
    record = session.to_record()
    invalid = deepcopy(record)
    invalid["summary"]["runner.unit"]["calls"] = 2

    with pytest.raises(ValueError, match="summary does not match raw regions"):
        validate_performance_profile_record(invalid)
    print("P6.2 performance profile tamper guard: PASS")
