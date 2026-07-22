"""Bit-exact fused Q/K RMSNorm for Qwen3-VL decode."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice

    HAS_QK_RMSNORM_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    libdevice = None
    HAS_QK_RMSNORM_TRITON = False


QK_RMSNORM_XBLOCK = 8
QK_RMSNORM_NUM_WARPS = 4
SUPPORTED_QK_RMSNORM_HEAD_DIM = 128
MAX_EXACT_QK_RMSNORM_BATCH = 4


if HAS_QK_RMSNORM_TRITON:

    @triton.jit
    def _fused_qk_rmsnorm_kernel(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        q_out_ptr,
        k_out_ptr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EPS: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        offsets = tl.arange(0, HEAD_DIM)[None, :]
        total_rows = Q_ROWS + K_ROWS
        valid = rows < total_rows
        is_q = rows < Q_ROWS

        q = tl.load(
            q_ptr + rows * HEAD_DIM + offsets,
            mask=valid & is_q,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + (rows - Q_ROWS) * HEAD_DIM + offsets,
            mask=valid & ~is_q,
            other=0.0,
        ).to(tl.float32)
        values = tl.where(is_q, q, k)

        q_weight = tl.load(q_weight_ptr + offsets).to(tl.float32)
        k_weight = tl.load(k_weight_ptr + offsets).to(tl.float32)
        weight = tl.where(is_q, q_weight, k_weight).to(tl.bfloat16).to(tl.float32)

        squared = tl.where(valid, values * values, 0.0)
        variance = tl.sum(squared, axis=1)[:, None] / HEAD_DIM
        inverse_rms = libdevice.rsqrt(variance + EPS)
        normalized = (
            (values * inverse_rms).to(tl.float32).to(tl.bfloat16).to(tl.float32)
        )
        output = (weight * normalized).to(tl.bfloat16).to(tl.float32)

        tl.store(
            q_out_ptr + rows * HEAD_DIM + offsets,
            output,
            mask=valid & is_q,
        )
        tl.store(
            k_out_ptr + (rows - Q_ROWS) * HEAD_DIM + offsets,
            output,
            mask=valid & ~is_q,
        )


def fused_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize contiguous BF16 decode Q/K in one Graph-safe Triton launch."""

    if not HAS_QK_RMSNORM_TRITON:
        raise RuntimeError("fused Q/K RMSNorm requires Triton")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("fused Q/K RMSNorm expects rank-3 Q and K tensors")
    if q.device != k.device or not q.is_cuda:
        raise ValueError("fused Q/K RMSNorm requires Q and K on the same CUDA device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("fused Q/K RMSNorm requires BF16 Q and K")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("fused Q/K RMSNorm requires matching batch and head dimensions")
    if q.shape[0] > MAX_EXACT_QK_RMSNORM_BATCH:
        raise ValueError(
            "fused Q/K RMSNorm exactness is validated for batch <= "
            f"{MAX_EXACT_QK_RMSNORM_BATCH}, got {q.shape[0]}"
        )
    if q.shape[-1] != SUPPORTED_QK_RMSNORM_HEAD_DIM:
        raise ValueError(
            "fused Q/K RMSNorm supports head_dim="
            f"{SUPPORTED_QK_RMSNORM_HEAD_DIM}, got {q.shape[-1]}"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("fused Q/K RMSNorm requires contiguous Q and K")
    expected_weight_shape = (q.shape[-1],)
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(f"{name} must have shape {expected_weight_shape}")
        if (
            weight.device != q.device
            or weight.dtype != q.dtype
            or not weight.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous BF16 on the Q/K device")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be a positive finite float, got {eps!r}")

    head_dim = q.shape[-1]
    q_rows = q.numel() // head_dim
    k_rows = k.numel() // head_dim
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    grid = (triton.cdiv(q_rows + k_rows, QK_RMSNORM_XBLOCK),)
    _fused_qk_rmsnorm_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        q_out,
        k_out,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        HEAD_DIM=head_dim,
        EPS=eps,
        XBLOCK=QK_RMSNORM_XBLOCK,
        num_warps=QK_RMSNORM_NUM_WARPS,
    )
    return q_out, k_out


__all__ = ["HAS_QK_RMSNORM_TRITON", "fused_qk_rmsnorm"]
