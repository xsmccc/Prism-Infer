import pytest
import torch
import torch.nn.functional as F

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
