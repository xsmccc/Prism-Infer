"""Rotary position embeddings for text and multimodal token positions."""

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position encoding to one query or key tensor."""

    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Precompute and apply a shared cosine/sine position table."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if isinstance(head_size, bool) or not isinstance(head_size, int) or head_size <= 0:
            raise ValueError(f"head_size must be a positive integer, got {head_size!r}")
        if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int) or rotary_dim <= 0:
            raise ValueError(f"rotary_dim must be a positive integer, got {rotary_dim!r}")
        if rotary_dim != head_size:
            raise ValueError(
                "partial rotary dimensions are unsupported: "
                f"rotary_dim={rotary_dim}, head_size={head_size}"
            )
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if base <= 0:
            raise ValueError(f"RoPE base must be positive, got {base}")
        self.head_size = head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Return the process-shared legacy RoPE module for one configuration."""

    if rope_scaling is not None:
        raise NotImplementedError("legacy RotaryEmbedding does not support rope_scaling")
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
