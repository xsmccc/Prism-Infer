"""
Interleaved M-RoPE: 多模态 3D Rotary Position Embedding.

参照 HF Qwen3VLTextRotaryEmbedding、vLLM Qwen3_VisionTransformer。
输出 [1, seqlen, head_dim] cos/sin，batch 维度通过广播处理。
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
    # cos/sin: [1, seqlen, head_dim] → [1, 1, seqlen, head_dim] 与 q:[B,Heads,S,D] 广播
    c = cos.float().unsqueeze(1)  # [1, 1, S, D] or [B, 1, S, D]
    s = sin.float().unsqueeze(1)
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
        assert sum(self.mrope_section) * 2 == head_dim, \
            f"mrope_section sum ({sum(self.mrope_section)}) * 2 != head_dim ({head_dim})"

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """三轴频率按 T-H-W 交错合并.

        freqs: [3, 1, seqlen, head_dim/2]
        返回: [1, seqlen, head_dim/2]
        """
        out = freqs[0].clone()  # [1, seqlen, dim/2] — 从 T 轴开始
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            out[:, :, idx] = freqs[dim, :, :, idx]
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple:
        """生成 interleaved cos/sin: [1, seqlen, head_dim].

        position_ids: [3, seqlen] | [1, seqlen] | [3, batch, seqlen]
        返回 cos/sin 不含 batch 维 (广播), 调用方自行 expand 到 batch.
        """
        device = position_ids.device
        dtype = x.dtype
        seqlen = position_ids.shape[-1]

        # 统一为 [3, seqlen]: 去掉 batch 维, 各样本位置相同
        if position_ids.ndim == 3:
            position_ids = position_ids[:, 0, :]   # [3, batch, seqlen] → [3, seqlen]
        elif position_ids.ndim == 2 and position_ids.shape[0] != 3:
            position_ids = position_ids.expand(3, -1)  # [1, seqlen] → [3, seqlen]

        # inv_freq [dim/2] → [3, 1, seqlen, dim/2, 1]
        inv_freq = self.inv_freq[None, None, None, :, None].expand(
            3, 1, seqlen, -1, 1).to(device)

        # positions [3, seqlen] → [3, 1, seqlen, 1, 1]
        pos = position_ids.float()[:, None, :, None, None].to(device)

        # freqs: [3, 1, seqlen, dim/2]
        freqs = (inv_freq @ pos).squeeze(-1)

        # 三轴交错: [3, 1, seqlen, dim/2] → [1, seqlen, dim/2]
        freqs = self._interleave(freqs)

        # [1, seqlen, dim/2] → [1, seqlen, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
