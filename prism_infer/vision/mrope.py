"""
Interleaved M-RoPE: 多模态 3D Rotary Position Embedding.

参照 HF Qwen3VLTextRotaryEmbedding。
输出 [3, seqlen, head_dim] cos/sin — 三轴各一套。
"""
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MRope(nn.Module):
    """Interleaved M-RoPE: 输出 [3, seqlen, head_dim] cos/sin."""

    def __init__(self, head_dim: int = 128, theta: float = 5000000.0,
                 mrope_section: list[int] | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.mrope_section = mrope_section or [24, 20, 20]
        assert sum(self.mrope_section) * 2 == head_dim

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple:
        """生成 cos/sin: [3, seqlen, head_dim]"""
        device = position_ids.device
        seqlen = position_ids.shape[1]
        dtype = x.dtype

        inv_freq = self.inv_freq[None, None, :].expand(3, seqlen, -1).to(device)
        pos = position_ids.float().unsqueeze(-1).to(device)
        freqs = pos * inv_freq  # [3, seqlen, 64]
        emb = torch.cat((freqs, freqs), dim=-1)  # [3, seqlen, 128]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def build_axis_map(head_dim: int, mrope_section: list[int]) -> list[tuple]:
    """构建每个 head_dim 位置 → 所属轴的映射."""
    sections = []
    offset = 0
    for _ in range(2):
        for axis, slen in enumerate(mrope_section):
            sections.append((offset, offset + slen, axis))
            offset += slen
    return sorted(sections)


def apply_mrope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor,
                mrope_section: list[int] | None = None) -> tuple:
    """应用 M-RoPE: 按 mrope_section 分配三轴 cos/sin 到 head_dim 各段.

    q:   [batch, n_heads, seqlen, head_dim]
    k:   [batch, n_kv_heads, seqlen, head_dim]
    cos: [3, seqlen, head_dim] — axis 0=T, axis 1=H, axis 2=W
    sin: 同上
    """
    mrope_section = mrope_section or [24, 20, 20]
    head_dim = q.shape[-1]
    orig_dtype = q.dtype

    q_f, k_f = q.float(), k.float()
    c_f, s_f = cos.float(), sin.float()

    q_out, k_out = q_f.clone(), k_f.clone()
    axis_map = build_axis_map(head_dim, mrope_section)

    for start, end, axis in axis_map:
        # cos[axis]: [seqlen, head_dim] → 取 [start:end] → [seqlen, slen]
        c = c_f[axis, :, start:end]  # [seqlen, slen]
        s = s_f[axis, :, start:end]
        # 广播: [seqlen, slen] → [1, 1, seqlen, slen]
        c = c[None, None, :, :]
        s = s[None, None, :, :]
        q_s = q_f[:, :, :, start:end]
        k_s = k_f[:, :, :, start:end]
        q_out[:, :, :, start:end] = q_s * c + rotate_half(q_s) * s
        k_out[:, :, :, start:end] = k_s * c + rotate_half(k_s) * s

    return q_out.to(orig_dtype), k_out.to(orig_dtype)
