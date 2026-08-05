"""Fused activation layers used by Qwen text and vision blocks."""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """Apply SwiGLU to concatenated gate and up projections."""

    def __init__(self) -> None:
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, -1)
        return F.silu(gate) * up
