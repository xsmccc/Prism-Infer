import pytest
import torch

from prism_infer.ops.add_rmsnorm import (
    HAS_ADD_RMSNORM_TRITON,
    fused_add_rmsnorm_prefill,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_ADD_RMSNORM_TRITON,
    reason="fused add RMSNorm requires CUDA and Triton",
)
def test_prefill_fused_add_rmsnorm_keeps_native_reduction_exact() -> None:
    torch.manual_seed(0)
    eps = 1e-6
    weight = torch.randn((4096,), dtype=torch.bfloat16, device="cuda")
    for rows in (5, 37):
        values = torch.randn((rows, 4096), dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(values)
        expected_added = values + residual
        expected_float = expected_added.float()
        expected_variance = expected_float.pow(2).mean(-1, keepdim=True)
        expected = weight * (
            expected_float * torch.rsqrt(expected_variance + eps)
        ).to(torch.bfloat16)

        actual, actual_added = fused_add_rmsnorm_prefill(
            values,
            residual,
            weight,
            eps=eps,
        )

        assert torch.equal(actual_added, expected_added)
        assert torch.equal(actual, expected)
