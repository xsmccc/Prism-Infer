"""Bit-exact fused residual add and RMSNorm for Qwen3-VL decode."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice

    HAS_ADD_RMSNORM_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    libdevice = None
    HAS_ADD_RMSNORM_TRITON = False


ADD_RMSNORM_XBLOCK = 1
ADD_RMSNORM_NUM_WARPS = 8
SUPPORTED_ADD_RMSNORM_HIDDEN_SIZE = 4096
MAX_EXACT_ADD_RMSNORM_BATCH = 4


if HAS_ADD_RMSNORM_TRITON:

    @triton.jit
    def _fused_add_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        added_ptr,
        ROWS: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        EPS: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        offsets = tl.arange(0, HIDDEN_SIZE)[None, :]
        valid = rows < ROWS
        addresses = rows * HIDDEN_SIZE + offsets

        x = tl.load(x_ptr + addresses, mask=valid, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + addresses, mask=valid, other=0.0).to(tl.float32)
        # PyTorch's BF16 add materializes before RMSNorm. Preserve that rounding
        # boundary so the fused path remains exact over long greedy trajectories.
        added = (x + residual).to(tl.bfloat16).to(tl.float32)
        squared = tl.where(valid, added * added, 0.0)
        variance = tl.sum(squared, axis=1)[:, None] / HIDDEN_SIZE
        inverse_rms = libdevice.rsqrt(variance + EPS)
        normalized = (added * inverse_rms).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + offsets).to(tl.bfloat16).to(tl.float32)
        output = (normalized * weight).to(tl.bfloat16).to(tl.float32)

        tl.store(added_ptr + addresses, added, mask=valid)
        tl.store(output_ptr + addresses, output, mask=valid)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact ``RMSNorm(residual + x)`` and the materialized sum."""

    if not HAS_ADD_RMSNORM_TRITON:
        raise RuntimeError("fused add RMSNorm requires Triton")
    if x.ndim != 2 or residual.ndim != 2:
        raise ValueError("fused add RMSNorm expects rank-2 tensors")
    if x.shape != residual.shape:
        raise ValueError("fused add RMSNorm requires matching input shapes")
    if x.device != residual.device or not x.is_cuda:
        raise ValueError("fused add RMSNorm requires inputs on the same CUDA device")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("fused add RMSNorm requires BF16 inputs")
    if x.shape[0] > MAX_EXACT_ADD_RMSNORM_BATCH:
        raise ValueError(
            "fused add RMSNorm exactness is validated for batch <= "
            f"{MAX_EXACT_ADD_RMSNORM_BATCH}, got {x.shape[0]}"
        )
    if x.shape[1] != SUPPORTED_ADD_RMSNORM_HIDDEN_SIZE:
        raise ValueError(
            "fused add RMSNorm supports hidden_size="
            f"{SUPPORTED_ADD_RMSNORM_HIDDEN_SIZE}, got {x.shape[1]}"
        )
    if not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError("fused add RMSNorm requires contiguous inputs")
    if (
        tuple(weight.shape) != (x.shape[1],)
        or weight.device != x.device
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous BF16 on the input device")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be a positive finite float, got {eps!r}")

    output = torch.empty_like(x)
    added = torch.empty_like(x)
    grid = (triton.cdiv(x.shape[0], ADD_RMSNORM_XBLOCK),)
    _fused_add_rmsnorm_kernel[grid](
        x,
        residual,
        weight,
        output,
        added,
        ROWS=x.shape[0],
        HIDDEN_SIZE=x.shape[1],
        EPS=eps,
        XBLOCK=ADD_RMSNORM_XBLOCK,
        num_warps=ADD_RMSNORM_NUM_WARPS,
    )
    return output, added


__all__ = ["HAS_ADD_RMSNORM_TRITON", "fused_add_rmsnorm"]
