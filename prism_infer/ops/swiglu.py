"""Bit-exact fused SwiGLU activation for small-batch decode."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_SWIGLU_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    HAS_SWIGLU_TRITON = False


SWIGLU_BLOCK_SIZE = 64
SWIGLU_NUM_WARPS = 1
MAX_SWIGLU_BATCH = 4


if HAS_SWIGLU_TRITON:

    @triton.jit
    def _fused_swiglu_kernel(
        packed_ptr,
        output_ptr,
        INTERMEDIATE_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < INTERMEDIATE_SIZE
        packed_row = row * (2 * INTERMEDIATE_SIZE)
        gate = tl.load(
            packed_ptr + packed_row + offsets,
            mask=valid,
        ).to(tl.float32)
        up = tl.load(
            packed_ptr + packed_row + INTERMEDIATE_SIZE + offsets,
            mask=valid,
        ).to(tl.float32)

        # PyTorch eager materializes BF16 SiLU before the BF16 multiply.  Keep
        # that rounding boundary while fusing both operations into one kernel.
        activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        output = (activated * up).to(tl.bfloat16)
        tl.store(
            output_ptr + row * INTERMEDIATE_SIZE + offsets,
            output,
            mask=valid,
        )


def fused_silu_mul(packed: torch.Tensor) -> torch.Tensor:
    """Return ``SiLU(gate) * up`` with eager-BF16-equivalent rounding."""

    if not HAS_SWIGLU_TRITON:
        raise RuntimeError("fused SwiGLU requires Triton")
    if packed.ndim != 2:
        raise ValueError("fused SwiGLU expects a rank-2 packed tensor")
    if not packed.is_cuda or packed.dtype != torch.bfloat16:
        raise ValueError("fused SwiGLU requires CUDA BF16 input")
    if not packed.is_contiguous():
        raise ValueError("fused SwiGLU requires contiguous input")
    if not 1 <= packed.shape[0] <= MAX_SWIGLU_BATCH:
        raise ValueError(
            f"fused SwiGLU supports batch sizes 1 through {MAX_SWIGLU_BATCH}, "
            f"got {packed.shape[0]}"
        )
    if packed.shape[1] % 2:
        raise ValueError("fused SwiGLU requires an even packed feature dimension")

    intermediate_size = packed.shape[1] // 2
    output = torch.empty(
        (packed.shape[0], intermediate_size),
        dtype=packed.dtype,
        device=packed.device,
    )
    _fused_swiglu_kernel[
        (packed.shape[0], triton.cdiv(intermediate_size, SWIGLU_BLOCK_SIZE))
    ](
        packed,
        output,
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_SIZE=SWIGLU_BLOCK_SIZE,
        num_warps=SWIGLU_NUM_WARPS,
    )
    return output


__all__ = ["HAS_SWIGLU_TRITON", "fused_silu_mul"]
