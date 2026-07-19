"""
Interleaved M-RoPE: 多模态 3D Rotary Position Embedding.

参照 HF Qwen3VLTextRotaryEmbedding、vLLM Qwen3_VisionTransformer。
输出 [1, seqlen, head_dim] cos/sin，batch 维度通过广播处理。
"""

from math import isfinite

import torch
import torch.nn as nn

from prism_infer.models.qwen3_vl_architecture import (
    CANONICAL_MROPE_SECTION,
    CANONICAL_TEXT_HEAD_DIM,
    CANONICAL_TEXT_ROPE_THETA,
    MROPE_AXIS_COUNT,
    MROPE_POSITION_TENSOR_RANK,
    ROTARY_PAIR_SIZE,
)


POSITION_ID_MATRIX_RANK = 2


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """标准 RoPE: q*cos + rotate_half(q)*sin. cos/sin 已包含三轴交错."""
    # cos/sin: [1, seqlen, head_dim] → [1, 1, seqlen, head_dim] 与 q:[B,Heads,S,D] 广播
    c = cos.unsqueeze(1)  # [1, 1, S, D] or [B, 1, S, D]
    s = sin.unsqueeze(1)
    qr = q * c + rotate_half(q) * s
    kr = k * c + rotate_half(k) * s
    return qr, kr


class MRope(nn.Module):
    """Interleaved M-RoPE: 三轴交错后输出标准 2D cos/sin."""

    def __init__(
        self,
        head_dim: int = CANONICAL_TEXT_HEAD_DIM,
        theta: float = CANONICAL_TEXT_ROPE_THETA,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
            or head_dim % ROTARY_PAIR_SIZE
        ):
            raise ValueError(f"head_dim must be a positive even integer: {head_dim!r}")
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or float(theta) <= 0.0
            or not isfinite(float(theta))
        ):
            raise ValueError(f"theta must be positive and finite: {theta!r}")
        section = CANONICAL_MROPE_SECTION if mrope_section is None else tuple(mrope_section)
        if len(section) != MROPE_AXIS_COUNT or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in section
        ):
            raise ValueError(f"mrope_section must contain {MROPE_AXIS_COUNT} positive integers")
        if sum(section) * ROTARY_PAIR_SIZE != head_dim:
            raise ValueError(
                f"mrope_section sum ({sum(section)}) * {ROTARY_PAIR_SIZE} != head_dim ({head_dim})"
            )
        self.head_dim = head_dim
        self.theta = float(theta)
        self.mrope_section = section

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, head_dim, ROTARY_PAIR_SIZE).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """三轴频率按 T-H-W 交错合并.

        freqs: [3, batch, seqlen, head_dim/2]
        返回: [batch, seqlen, head_dim/2]
        """
        out = freqs[0].clone()  # [batch, seqlen, dim/2] — 从 T 轴开始
        for axis in range(1, MROPE_AXIS_COUNT):
            length = self.mrope_section[axis] * MROPE_AXIS_COUNT
            index = slice(axis, length, MROPE_AXIS_COUNT)
            out[..., index] = freqs[axis, ..., index]
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple:
        """生成 interleaved cos/sin: [batch, seqlen, head_dim].

        position_ids:
          - [batch, seqlen]: 纯文本 1D positions, 与 HF 行为一致
          - [3, batch, seqlen]: 多模态 T/H/W positions, 与 HF 行为一致
          - [3, seqlen]: 兼容调用，视为单 batch 的 T/H/W positions
        """
        device = position_ids.device
        dtype = x.dtype

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        if position_ids.ndim == POSITION_ID_MATRIX_RANK:
            # HF treats [batch, seqlen] as text positions and expands to 3 axes.
            # Keep the historical convenience form [3, seqlen] for one-sample
            # multimodal positions when x is not batch=3.
            if position_ids.shape[0] == MROPE_AXIS_COUNT and x.shape[0] != MROPE_AXIS_COUNT:
                position_ids = position_ids[:, None, :]
            else:
                position_ids = position_ids[None, ...].expand(
                    MROPE_AXIS_COUNT, position_ids.shape[0], -1
                )
        elif (
            position_ids.ndim != MROPE_POSITION_TENSOR_RANK
            or position_ids.shape[0] != MROPE_AXIS_COUNT
        ):
            raise ValueError(
                "position_ids must have shape [batch, seqlen], [3, seqlen], or [3, batch, seqlen]"
            )

        position_ids = position_ids.to(device)
        batch = position_ids.shape[1]

        # Match HF Qwen3VLTextRotaryEmbedding exactly:
        # inv_freq [dim/2] -> [3, batch, dim/2, 1]
        # position_ids [3, batch, seqlen] -> [3, batch, 1, seqlen]
        inv_freq = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(MROPE_AXIS_COUNT, batch, -1, 1)
            .to(device)
        )
        pos = position_ids[:, :, None, :].float()

        # freqs: [3, batch, seqlen, dim/2]
        freqs = (inv_freq @ pos).transpose(2, 3)

        # 三轴交错: [3, batch, seqlen, dim/2] → [batch, seqlen, dim/2]
        freqs = self._interleave(freqs)

        # [batch, seqlen, dim/2] → [batch, seqlen, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
