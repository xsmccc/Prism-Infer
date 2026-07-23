import pytest
import torch

from prism_infer.ops.qk_rmsnorm import (
    HAS_QK_RMSNORM_TRITON,
    fused_qk_rmsnorm_prefill,
)


def _rmsnorm(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values_float = values.float()
    variance = values_float.pow(2).mean(-1, keepdim=True)
    normalized = values_float * torch.rsqrt(variance + eps)
    return weight * normalized.to(values.dtype)


def _mrope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = values.shape[-1] // 2
    rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cos + rotated * sin


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_QK_RMSNORM_TRITON,
    reason="prefill fused Q/K RMSNorm requires CUDA and Triton",
)
def test_prefill_qk_rmsnorm_mrope_keeps_native_reductions_exact() -> None:
    torch.manual_seed(0)
    eps = 1e-6
    q_weight = torch.randn((128,), dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn((128,), dtype=torch.bfloat16, device="cuda")
    for rows in (5, 37):
        q = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((rows, 8, 128), dtype=torch.bfloat16, device="cuda")
        cos = torch.randn((rows, 128), dtype=torch.bfloat16, device="cuda")
        sin = torch.randn((rows, 128), dtype=torch.bfloat16, device="cuda")
        expected_q = _mrope(_rmsnorm(q, q_weight, eps), cos, sin)
        expected_k = _mrope(_rmsnorm(k, k_weight, eps), cos, sin)

        actual_q, actual_k = fused_qk_rmsnorm_prefill(
            q,
            k,
            q_weight,
            k_weight,
            eps=eps,
            cos=cos,
            sin=sin,
        )

        assert torch.equal(actual_q, expected_q)
        assert torch.equal(actual_k, expected_k)
