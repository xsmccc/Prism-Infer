"""Focused GPU checks for the SM120 block-4 gate-up kernel."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from prism_infer.ops.block4_gate_up import (
    block4_gate_up_swiglu,
    compress_block4_gate_up_weight,
)
from prism_infer.ops.swiglu import fused_silu_mul


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="block-4 gate-up requires an SM120 CUDA device",
)


@torch.no_grad()
def test_block4_gate_up_matches_decompressed_bf16_reference() -> None:
    torch.manual_seed(20260723)
    hidden_states = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    weight_fp8, scales = compress_block4_gate_up_weight(weight)
    decompressed = (
        (weight_fp8.float().reshape(weight.shape[0], -1, 4) * scales.float().unsqueeze(-1))
        .reshape_as(weight_fp8)
        .to(torch.bfloat16)
    )

    expected = fused_silu_mul(F.linear(hidden_states, decompressed))
    actual = block4_gate_up_swiglu(hidden_states, weight_fp8, scales)

    assert (
        weight_fp8.numel() * weight_fp8.element_size() + (scales.numel() * scales.element_size())
        == weight.numel() * weight.element_size() * 3 // 4
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0)
