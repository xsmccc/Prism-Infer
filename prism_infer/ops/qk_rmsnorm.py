"""Bit-exact fused Q/K RMSNorm for Qwen3-VL decode."""

from __future__ import annotations

import math
import os

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


QK_RMSNORM_XBLOCK = 1
QK_RMSNORM_NUM_WARPS = 1
SUPPORTED_QK_RMSNORM_HEAD_DIM = 128
MAX_EXACT_QK_RMSNORM_BATCH = 4
MIN_PREFILL_QK_RMSNORM_ROWS = 1024
PREFILL_QK_RMSNORM_NUM_WARPS = 1
PREFILL_Q_HEADS = 32
PREFILL_K_HEADS = 8


if HAS_QK_RMSNORM_TRITON:

    @triton.jit
    def _round_bfloat16(value):
        """Materialize an RN BF16 boundary so LLVM cannot contract M-RoPE ops."""

        rounded = tl.inline_asm_elementwise(
            asm="cvt.rn.bf16.f32 $0, $1;",
            constraints="=h,f",
            args=[value],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )
        return rounded.to(tl.float32)

    @triton.jit
    def _prefill_qk_square_kernel(
        q_ptr,
        k_ptr,
        q_squared_ptr,
        k_squared_ptr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, HEAD_DIM)
        is_q = row < Q_ROWS
        k_row = row - Q_ROWS
        valid_k = ~is_q & (k_row < K_ROWS)
        q = tl.load(
            q_ptr + row * HEAD_DIM + offsets,
            mask=is_q,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + k_row * HEAD_DIM + offsets,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        values = tl.where(is_q, q, k)
        squared = values * values
        tl.store(
            q_squared_ptr + row * HEAD_DIM + offsets,
            squared,
            mask=is_q,
        )
        tl.store(
            k_squared_ptr + k_row * HEAD_DIM + offsets,
            squared,
            mask=valid_k,
        )

    @triton.jit
    def _prefill_qk_normalize_mrope_kernel(
        q_ptr,
        k_ptr,
        q_variance_ptr,
        k_variance_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, HEAD_DIM)
        is_q = row < Q_ROWS
        k_row = row - Q_ROWS
        valid_k = ~is_q & (k_row < K_ROWS)
        q = tl.load(
            q_ptr + row * HEAD_DIM + offsets,
            mask=is_q,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + k_row * HEAD_DIM + offsets,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        values = tl.where(is_q, q, k)
        q_variance = tl.load(q_variance_ptr + row, mask=is_q, other=0.0)
        k_variance = tl.load(
            k_variance_ptr + k_row,
            mask=valid_k,
            other=0.0,
        )
        variance = tl.where(is_q, q_variance, k_variance)
        inverse_rms = libdevice.rsqrt(variance + EPS)

        q_weight = (
            tl.load(q_weight_ptr + offsets).to(tl.bfloat16).to(tl.float32)
        )
        k_weight = (
            tl.load(k_weight_ptr + offsets).to(tl.bfloat16).to(tl.float32)
        )
        weight = tl.where(is_q, q_weight, k_weight)
        normalized = _round_bfloat16(values * inverse_rms)
        output = _round_bfloat16(normalized * weight)

        rotated_offsets = tl.where(
            offsets < HEAD_DIM // 2,
            offsets + HEAD_DIM // 2,
            offsets - HEAD_DIM // 2,
        )
        q_rotated = tl.load(
            q_ptr + row * HEAD_DIM + rotated_offsets,
            mask=is_q,
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.load(
            k_ptr + k_row * HEAD_DIM + rotated_offsets,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        rotated_values = tl.where(is_q, q_rotated, k_rotated)
        q_rotated_weight = (
            tl.load(q_weight_ptr + rotated_offsets)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        k_rotated_weight = (
            tl.load(k_weight_ptr + rotated_offsets)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rotated_weight = tl.where(
            is_q,
            q_rotated_weight,
            k_rotated_weight,
        )
        rotated_normalized = _round_bfloat16(rotated_values * inverse_rms)
        rotated_output = _round_bfloat16(
            rotated_normalized * rotated_weight
        )
        rotated_output = tl.where(
            offsets < HEAD_DIM // 2,
            -rotated_output,
            rotated_output,
        )

        token_row = tl.where(
            is_q,
            row // Q_HEADS,
            k_row // K_HEADS,
        )
        cos = (
            tl.load(cos_ptr + token_row * HEAD_DIM + offsets)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        sin = (
            tl.load(sin_ptr + token_row * HEAD_DIM + offsets)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        direct_product = _round_bfloat16(output * cos)
        rotated_product = _round_bfloat16(rotated_output * sin)
        output = _round_bfloat16(direct_product + rotated_product)

        tl.store(
            q_out_ptr + row * HEAD_DIM + offsets,
            output,
            mask=is_q,
        )
        tl.store(
            k_out_ptr + k_row * HEAD_DIM + offsets,
            output,
            mask=valid_k,
        )

    @triton.jit
    def _fused_qk_rmsnorm_kernel(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        v_ptr,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EPS: tl.constexpr,
        XBLOCK: tl.constexpr,
        APPLY_MROPE: tl.constexpr,
        STORE_KV: tl.constexpr,
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

        if APPLY_MROPE:
            rotated_offsets = tl.where(
                offsets < HEAD_DIM // 2,
                offsets + HEAD_DIM // 2,
                offsets - HEAD_DIM // 2,
            )
            q_rotated = tl.load(
                q_ptr + rows * HEAD_DIM + rotated_offsets,
                mask=valid & is_q,
                other=0.0,
            ).to(tl.float32)
            k_rotated = tl.load(
                k_ptr + (rows - Q_ROWS) * HEAD_DIM + rotated_offsets,
                mask=valid & ~is_q,
                other=0.0,
            ).to(tl.float32)
            rotated_values = tl.where(is_q, q_rotated, k_rotated)
            q_rotated_weight = (
                tl.load(q_weight_ptr + rotated_offsets)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            k_rotated_weight = (
                tl.load(k_weight_ptr + rotated_offsets)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            rotated_weight = tl.where(is_q, q_rotated_weight, k_rotated_weight)
            rotated_normalized = (
                (rotated_values * inverse_rms)
                .to(tl.float32)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            rotated_output = (
                (rotated_weight * rotated_normalized)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            rotated_output = tl.where(
                offsets < HEAD_DIM // 2,
                -rotated_output,
                rotated_output,
            )
            token_rows = tl.where(
                is_q,
                rows // Q_HEADS,
                (rows - Q_ROWS) // K_HEADS,
            )
            cos = (
                tl.load(
                    cos_ptr + token_rows * HEAD_DIM + offsets,
                    mask=valid,
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            sin = (
                tl.load(
                    sin_ptr + token_rows * HEAD_DIM + offsets,
                    mask=valid,
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            direct_product = _round_bfloat16(output * cos)
            rotated_product = _round_bfloat16(rotated_output * sin)
            output = _round_bfloat16(direct_product + rotated_product)

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
        if STORE_KV:
            token_rows = (rows - Q_ROWS) // K_HEADS
            kv_heads = (rows - Q_ROWS) % K_HEADS
            slots = tl.load(
                slot_mapping_ptr + token_rows,
                mask=valid & ~is_q,
                other=-1,
            )
            cache_offsets = (
                slots * K_HEADS * HEAD_DIM + kv_heads * HEAD_DIM + offsets
            )
            store_mask = valid & ~is_q & (slots >= 0)
            values_v = tl.load(
                v_ptr + (rows - Q_ROWS) * HEAD_DIM + offsets,
                mask=store_mask,
                other=0.0,
            )
            tl.store(k_cache_ptr + cache_offsets, output, mask=store_mask)
            tl.store(v_cache_ptr + cache_offsets, values_v, mask=store_mask)


def fused_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K and optionally apply exact M-RoPE in one Triton launch."""

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
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must either both be provided or both be omitted")
    apply_mrope = cos is not None
    store_kv_args = (v, k_cache, v_cache, slot_mapping)
    store_kv = any(value is not None for value in store_kv_args)
    if store_kv and any(value is None for value in store_kv_args):
        raise ValueError("v, K/V caches, and slot_mapping must be provided together")
    head_dim = q.shape[-1]
    if apply_mrope:
        expected_position_shape = (q.shape[0], head_dim)
        for name, value in (("cos", cos), ("sin", sin)):
            if (
                tuple(value.shape) != expected_position_shape
                or value.device != q.device
                or value.dtype != q.dtype
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous BF16 with shape "
                    f"{expected_position_shape} on the Q/K device"
                )
    if store_kv:
        if v.shape != k.shape or v.dtype != k.dtype or v.device != k.device:
            raise ValueError("fused KV store requires V to match K shape, dtype, and device")
        if not v.is_contiguous():
            raise ValueError("fused KV store requires contiguous V")
        for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
            if (
                cache.dtype != torch.bfloat16
                or cache.device != q.device
                or not cache.is_contiguous()
                or tuple(cache.shape[-2:]) != tuple(k.shape[-2:])
            ):
                raise ValueError(
                    f"{name} must be contiguous BF16 with K head dimensions"
                )
        if (
            slot_mapping.dtype != torch.int32
            or slot_mapping.device != q.device
            or slot_mapping.ndim != 1
            or slot_mapping.numel() != q.shape[0]
        ):
            raise ValueError("slot_mapping must provide one CUDA int32 slot per token")

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
        cos if apply_mrope else q,
        sin if apply_mrope else q,
        q_out,
        k_out,
        v if store_kv else q,
        k_cache if store_kv else q,
        v_cache if store_kv else q,
        slot_mapping if store_kv else q,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        Q_HEADS=q.shape[1],
        K_HEADS=k.shape[1],
        HEAD_DIM=head_dim,
        EPS=eps,
        XBLOCK=QK_RMSNORM_XBLOCK,
        APPLY_MROPE=apply_mrope,
        STORE_KV=store_kv,
        num_warps=QK_RMSNORM_NUM_WARPS,
    )
    return q_out, k_out


def fused_qk_rmsnorm_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep native RMS reductions while fusing prefill Q/K elementwise work."""

    if not HAS_QK_RMSNORM_TRITON:
        raise RuntimeError("prefill fused Q/K RMSNorm requires Triton")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("prefill fused Q/K RMSNorm expects rank-3 Q and K")
    if q.device != k.device or not q.is_cuda:
        raise ValueError(
            "prefill fused Q/K RMSNorm requires one CUDA device"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("prefill fused Q/K RMSNorm requires BF16 Q and K")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "prefill fused Q/K RMSNorm requires matching token and head dimensions"
        )
    if q.shape[-1] != SUPPORTED_QK_RMSNORM_HEAD_DIM:
        raise ValueError(
            "prefill fused Q/K RMSNorm supports head_dim="
            f"{SUPPORTED_QK_RMSNORM_HEAD_DIM}, got {q.shape[-1]}"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError(
            "prefill fused Q/K RMSNorm requires contiguous Q and K"
        )
    expected_weight_shape = (q.shape[-1],)
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if (
            tuple(weight.shape) != expected_weight_shape
            or weight.device != q.device
            or weight.dtype != q.dtype
            or not weight.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous BF16 with shape "
                f"{expected_weight_shape} on the Q/K device"
            )
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            f"eps must be a positive finite float, got {eps!r}"
        )
    expected_position_shape = (q.shape[0], q.shape[-1])
    for name, value in (("cos", cos), ("sin", sin)):
        if (
            tuple(value.shape) != expected_position_shape
            or value.device != q.device
            or value.dtype != q.dtype
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous BF16 with shape "
                f"{expected_position_shape} on the Q/K device"
            )

    head_dim = q.shape[-1]
    q_rows = q.numel() // head_dim
    k_rows = k.numel() // head_dim
    q_squared = torch.empty_like(q, dtype=torch.float32)
    k_squared = torch.empty_like(k, dtype=torch.float32)
    grid = (q_rows + k_rows,)
    _prefill_qk_square_kernel[grid](
        q,
        k,
        q_squared,
        k_squared,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        HEAD_DIM=head_dim,
        num_warps=PREFILL_QK_RMSNORM_NUM_WARPS,
    )
    # These two reductions intentionally remain native PyTorch operations.
    # Their tree order is part of the established BF16 inference trajectory.
    q_variance = q_squared.mean(-1, keepdim=True)
    k_variance = k_squared.mean(-1, keepdim=True)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    _prefill_qk_normalize_mrope_kernel[grid](
        q,
        k,
        q_variance,
        k_variance,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        Q_HEADS=q.shape[1],
        K_HEADS=k.shape[1],
        HEAD_DIM=head_dim,
        EPS=eps,
        num_warps=PREFILL_QK_RMSNORM_NUM_WARPS,
    )
    return q_out, k_out


def can_use_prefill_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether the validated large-prefill Q/K path applies."""

    if os.environ.get("PRISM_DISABLE_PREFILL_QK_RMSNORM") == "1":
        return False
    if not HAS_QK_RMSNORM_TRITON or torch.compiler.is_compiling():
        return False
    if q.ndim != 3 or k.ndim != 3:
        return False
    if q.shape[0] < MIN_PREFILL_QK_RMSNORM_ROWS:
        return False
    if q.shape[1:] != (PREFILL_Q_HEADS, SUPPORTED_QK_RMSNORM_HEAD_DIM):
        return False
    if k.shape != (
        q.shape[0],
        PREFILL_K_HEADS,
        SUPPORTED_QK_RMSNORM_HEAD_DIM,
    ):
        return False
    expected_position_shape = (
        q.shape[0],
        SUPPORTED_QK_RMSNORM_HEAD_DIM,
    )
    if cos.shape != expected_position_shape or sin.shape != expected_position_shape:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if cos.dtype != q.dtype or sin.dtype != q.dtype:
        return False
    if not q.is_cuda or q.device != k.device:
        return False
    if cos.device != q.device or sin.device != q.device:
        return False
    return (
        q.is_contiguous()
        and k.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
    )


def maybe_fused_qk_rmsnorm_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the exact large-prefill fusion or request the native fallback."""

    if not can_use_prefill_qk_rmsnorm(q, k, cos, sin):
        return None
    return fused_qk_rmsnorm_prefill(
        q,
        k,
        q_weight,
        k_weight,
        eps=eps,
        cos=cos,
        sin=sin,
    )


__all__ = [
    "HAS_QK_RMSNORM_TRITON",
    "MIN_PREFILL_QK_RMSNORM_ROWS",
    "can_use_prefill_qk_rmsnorm",
    "fused_qk_rmsnorm",
    "fused_qk_rmsnorm_prefill",
    "maybe_fused_qk_rmsnorm_prefill",
]
