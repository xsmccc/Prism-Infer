import importlib.util

import pytest
import torch
import torch.nn.functional as F

from prism_infer.ops.cutlass_swiglu import maybe_cutlass_dual_swiglu
from prism_infer.ops.swiglu import HAS_SWIGLU_TRITON, fused_silu_mul


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_SWIGLU_TRITON,
    reason="fused SwiGLU requires CUDA and Triton",
)
def test_fused_swiglu_is_bit_exact_for_decode_and_prefill() -> None:
    torch.manual_seed(0)
    for rows in (1, 5, 37):
        packed = torch.randn((rows, 256), dtype=torch.bfloat16, device="cuda")
        gate, up = packed.chunk(2, dim=-1)
        expected = F.silu(gate) * up
        actual = fused_silu_mul(packed)
        assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or importlib.util.find_spec("cutlass_library") is None,
    reason="CUTLASS dual-GEMM SwiGLU requires an SM120 GPU and nvidia-cutlass",
)
def test_cutlass_dual_gemm_swiglu_is_bit_exact_for_large_prefill() -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(
        (1024, 4096),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Match the trained projection scale; unit-variance weights drive SiLU
    # into a subnormal regime outside this model-specific kernel's contract.
    packed_weight = (
        torch.randn(
            (24576, 4096),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.02
    ).to(torch.bfloat16)
    expected = fused_silu_mul(F.linear(hidden_states, packed_weight))
    actual = maybe_cutlass_dual_swiglu(hidden_states, packed_weight)
    assert actual is not None
    assert torch.equal(actual, expected)
