"""Decode-only block-4 FP8-weight gate-up projection fused with SwiGLU."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_BLOCK4_GATE_UP_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    HAS_BLOCK4_GATE_UP_TRITON = False


BLOCK4_GROUP_SIZE = 4
BLOCK4_GATE_UP_ROWS = 4
BLOCK4_GATE_UP_K = 256
BLOCK4_GATE_UP_NUM_WARPS = 2
SUPPORTED_COMPUTE_CAPABILITY = (12, 0)


if HAS_BLOCK4_GATE_UP_TRITON:

    @triton.jit
    def _block4_gate_up_swiglu_kernel(
        hidden_ptr,
        weight_fp8_ptr,
        scale_ptr,
        output_ptr,
        HIDDEN_SIZE: tl.constexpr,
        INTERMEDIATE_SIZE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_rows = row_offsets < INTERMEDIATE_SIZE
        group_offsets = tl.arange(0, BLOCK_K // 4)
        scale_stride = HIDDEN_SIZE // 4
        gate_accumulator = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        for group_start in tl.static_range(0, HIDDEN_SIZE // 4, BLOCK_K // 4):
            scale_columns = group_start + group_offsets
            up_rows = row_offsets + INTERMEDIATE_SIZE
            gate_scale = tl.load(
                scale_ptr + row_offsets[:, None] * scale_stride + scale_columns[None, :],
                mask=valid_rows[:, None],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                scale_ptr + up_rows[:, None] * scale_stride + scale_columns[None, :],
                mask=valid_rows[:, None],
                other=0.0,
            ).to(tl.float32)
            gate_contribution = tl.zeros(
                (BLOCK_ROWS, BLOCK_K // 4),
                dtype=tl.float32,
            )
            up_contribution = tl.zeros(
                (BLOCK_ROWS, BLOCK_K // 4),
                dtype=tl.float32,
            )

            for lane in tl.static_range(0, 4):
                columns = scale_columns * 4 + lane
                hidden = tl.load(hidden_ptr + columns).to(tl.float32)
                gate_quantized = tl.load(
                    weight_fp8_ptr + row_offsets[:, None] * HIDDEN_SIZE + columns[None, :],
                    mask=valid_rows[:, None],
                    other=0.0,
                ).to(tl.float32)
                up_quantized = tl.load(
                    weight_fp8_ptr + up_rows[:, None] * HIDDEN_SIZE + columns[None, :],
                    mask=valid_rows[:, None],
                    other=0.0,
                ).to(tl.float32)
                gate_weight = (gate_quantized * gate_scale).to(tl.bfloat16).to(tl.float32)
                up_weight = (up_quantized * up_scale).to(tl.bfloat16).to(tl.float32)
                gate_contribution += gate_weight * hidden[None, :]
                up_contribution += up_weight * hidden[None, :]

            gate_accumulator += tl.sum(gate_contribution, axis=1)
            up_accumulator += tl.sum(up_contribution, axis=1)

        gate = gate_accumulator.to(tl.bfloat16).to(tl.float32)
        up = up_accumulator.to(tl.bfloat16).to(tl.float32)
        activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        output = (activated * up).to(tl.bfloat16)
        tl.store(output_ptr + row_offsets, output, mask=valid_rows)


@torch.no_grad()
def compress_block4_gate_up_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous FP8 values and FP16 scales for groups of four weights."""

    if not HAS_BLOCK4_GATE_UP_TRITON:
        raise RuntimeError("block-4 gate-up compression requires Triton")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("block-4 gate-up compression requires FP8 E4M3FN")
    if weight.ndim != 2 or weight.shape[0] % 2:
        raise ValueError("block-4 gate-up weight must be a packed rank-2 matrix")
    if weight.shape[1] % BLOCK4_GATE_UP_K:
        raise ValueError(f"block-4 gate-up hidden size must be divisible by {BLOCK4_GATE_UP_K}")
    if not weight.is_cuda or weight.dtype != torch.bfloat16:
        raise ValueError("block-4 gate-up compression requires a CUDA BF16 weight")
    if not weight.is_contiguous():
        raise ValueError("block-4 gate-up compression requires a contiguous weight")
    if torch.cuda.get_device_capability(weight.device) != SUPPORTED_COMPUTE_CAPABILITY:
        raise RuntimeError("block-4 gate-up supports only compute capability 12.0")

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    groups = weight.float().reshape(
        weight.shape[0],
        -1,
        BLOCK4_GROUP_SIZE,
    )
    scales = (groups.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(1e-12).to(torch.float16)
    quantized = (groups / scales.float()).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (
        quantized.reshape_as(weight).contiguous(),
        scales.squeeze(-1).contiguous(),
    )


def block4_gate_up_swiglu(
    hidden_states: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the canonical batch-one decode gate-up projection and SwiGLU."""

    if not HAS_BLOCK4_GATE_UP_TRITON:
        raise RuntimeError("block-4 gate-up requires Triton")
    if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
        raise ValueError("block-4 gate-up requires one rank-2 decode row")
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise ValueError("block-4 gate-up requires a CUDA BF16 activation")
    if not hidden_states.is_contiguous():
        raise ValueError("block-4 gate-up requires a contiguous activation")
    if torch.cuda.get_device_capability(hidden_states.device) != SUPPORTED_COMPUTE_CAPABILITY:
        raise RuntimeError("block-4 gate-up supports only compute capability 12.0")
    if weight_fp8.dtype != torch.float8_e4m3fn or scales.dtype != torch.float16:
        raise ValueError("block-4 gate-up requires FP8 weights and FP16 scales")
    if weight_fp8.ndim != 2 or weight_fp8.shape[0] % 2:
        raise ValueError("block-4 gate-up requires a packed rank-2 weight")
    if hidden_states.shape[1] != weight_fp8.shape[1]:
        raise ValueError("block-4 gate-up activation/weight hidden sizes differ")
    if hidden_states.shape[1] % BLOCK4_GATE_UP_K:
        raise ValueError(f"block-4 gate-up hidden size must be divisible by {BLOCK4_GATE_UP_K}")
    expected_scales = (
        weight_fp8.shape[0],
        weight_fp8.shape[1] // BLOCK4_GROUP_SIZE,
    )
    if scales.shape != expected_scales:
        raise ValueError(
            f"block-4 gate-up scale shape must be {expected_scales}, got {tuple(scales.shape)}"
        )
    if weight_fp8.device != hidden_states.device or scales.device != hidden_states.device:
        raise ValueError("block-4 gate-up tensors must share one CUDA device")
    if not weight_fp8.is_contiguous() or not scales.is_contiguous():
        raise ValueError("block-4 gate-up weight and scales must be contiguous")

    intermediate_size = weight_fp8.shape[0] // 2
    output = torch.empty(
        (1, intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    _block4_gate_up_swiglu_kernel[(triton.cdiv(intermediate_size, BLOCK4_GATE_UP_ROWS),)](
        hidden_states,
        weight_fp8,
        scales,
        output,
        HIDDEN_SIZE=hidden_states.shape[1],
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_ROWS=BLOCK4_GATE_UP_ROWS,
        BLOCK_K=BLOCK4_GATE_UP_K,
        num_warps=BLOCK4_GATE_UP_NUM_WARPS,
    )
    return output


__all__ = [
    "HAS_BLOCK4_GATE_UP_TRITON",
    "block4_gate_up_swiglu",
    "compress_block4_gate_up_weight",
]
