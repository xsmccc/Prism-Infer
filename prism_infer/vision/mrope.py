"""
Interleaved M-RoPE: 多模态 3D Rotary Position Embedding.

参照 HF Qwen3VLTextRotaryEmbedding。
输出 [batch, seqlen, head_dim] cos/sin，已包含三轴交错。
"""
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> tuple:
    """标准 RoPE: q*cos + rotate_half(q)*sin. cos/sin 已包含三轴交错."""
    orig_dtype = q.dtype
    q_f, k_f = q.float(), k.float()
    c = cos.float().unsqueeze(1)  # [B, 1, S, D] or [S, 1, D]
    s = sin.float().unsqueeze(1)
    if q_f.dim() == 3:  # [heads, S, D] — squeeze batch
        c = c.squeeze(0)
        s = s.squeeze(0)
    qr = q_f * c + rotate_half(q_f) * s
    kr = k_f * c + rotate_half(k_f) * s
    return qr.to(orig_dtype), kr.to(orig_dtype)


class MRope(nn.Module):
    """Interleaved M-RoPE: 三轴交错后输出标准 2D cos/sin."""

    def __init__(self, head_dim: int = 128, theta: float = 5000000.0,
                 mrope_section: list[int] | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.mrope_section = mrope_section or [24, 20, 20]
        assert sum(self.mrope_section) * 2 == head_dim

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """三轴频率按 T-H-W 交错合并.

        freqs: [3, batch, seqlen, 64]
        返回: [batch, seqlen, 64]
        """
        out = freqs[0].clone()  # 从 T 轴开始
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            out[:, :, idx] = freqs[dim, :, :, idx]
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple:
        """生成 interleaved cos/sin: [batch, seqlen, head_dim].

        position_ids: [3, seqlen] (T/H/W 三轴) 或 [1, seqlen] (纯文本)
        """
        device = position_ids.device
        dtype = x.dtype
        batch = x.shape[0]
        seqlen = position_ids.shape[-1]

        # 确保 position_ids 是 [3, seqlen] 格式
        if position_ids.ndim == 2 and position_ids.shape[0] != 3:
            position_ids = position_ids[None, :].expand(3, -1, -1)

        # inv_freq [64] → [3, batch, seqlen, 64, 1]
        inv_freq = self.inv_freq[None, None, None, :, None].expand(
            3, batch, seqlen, -1, 1).to(device)

        # positions [3, seqlen] → [3, batch, seqlen, 1, 1]
        pos = position_ids.float()[:, None, :, None, None].to(device)

        # freqs: [3, batch, seqlen, 64, 1] → [3, batch, seqlen, 64]
        freqs = (inv_freq @ pos).squeeze(-1)

        # 三轴交错: [3, batch, seqlen, 64] → [batch, seqlen, 64]
        freqs = self._interleave(freqs)

        # 扩展为 [batch, seqlen, 128]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
