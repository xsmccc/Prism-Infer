"""Vision RoPE 初始化路径回归测试。

覆盖 P2-005: 当默认 device 为 CUDA 时，VisionEncoder 仍必须复现 HF
`Qwen3VLVisionRotaryEmbedding` 的 CPU 初始化数值，然后再迁移到目标设备。
否则极小的 inv_freq 差异会在 bf16 RoPE 后变成 full logits 误差。
"""

import torch

from conftest import require_transformers
from prism_infer.vision.vision_encoder import VisionEncoder


def test_vision_rotary_embedding_matches_hf_when_default_device_is_cuda():
    """默认 device 为 CUDA 时，Vision RoPE 频率仍与 HF 初始化路径 exact match。"""

    transformers = require_transformers()
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_thw = torch.tensor([[1, 28, 28]], device=device)

    hf_rotary = Qwen3VLVisionRotaryEmbedding(36).to(device)
    default_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        our = VisionEncoder(torch.bfloat16).to(device)
    finally:
        torch.set_default_device(default_device)

    with torch.no_grad():
        hf_inv = hf_rotary.inv_freq
        our_inv = our.rotary_pos_emb.inv_freq
        hf_freq = _hf_vision_freq_table(hf_rotary, 28, device)
        our_freq = our.rotary_pos_emb(28)
        hf_rot = _hf_rot_pos_emb_from_freq(hf_freq, grid_thw, merge_size=2)
        our_rot = our.rot_pos_emb(grid_thw)

    inv_diff = (hf_inv.float() - our_inv.float()).abs()
    freq_diff = (hf_freq.float() - our_freq.float()).abs()
    rot_diff = (hf_rot.float() - our_rot.float()).abs()

    print(f"  inv_freq shape: {list(our_inv.shape)}")
    print(f"  rot_pos_emb shape: {list(our_rot.shape)}")
    print(f"  inv_freq max diff: {inv_diff.max().item():.6e}")
    print(f"  freq_table max diff: {freq_diff.max().item():.6e}")
    print(f"  rot_pos_emb max diff: {rot_diff.max().item():.6e}")

    assert torch.equal(hf_inv, our_inv)
    assert torch.equal(hf_freq, our_freq)
    assert torch.equal(hf_rot, our_rot)


def test_patch_merger_layernorm_eps_matches_hf():
    """PatchMerger 和 DeepStack merger 的 LayerNorm eps 必须为 1e-6。"""

    vision = VisionEncoder(torch.bfloat16)
    eps_values = [vision.merger.norm.eps]
    eps_values.extend(merger.norm.eps for merger in vision.deepstack_merger_list)

    print(f"  merger eps values: {eps_values}")
    assert eps_values == [1e-6, 1e-6, 1e-6, 1e-6]


def _hf_vision_freq_table(hf_rotary, seqlen: int, device: str) -> torch.Tensor:
    """Call HF vision rotary embedding across old/new forward signatures."""

    try:
        return hf_rotary(seqlen)
    except (AttributeError, TypeError):
        position_ids = torch.arange(seqlen, device=device, dtype=torch.long)
        return hf_rotary(position_ids)


def _hf_rot_pos_emb_from_freq(
    freq_table: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_size: int,
) -> torch.Tensor:
    """用 HF `rot_pos_emb` 的索引顺序从频率表构造 RoPE embedding。"""

    device = freq_table.device
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return freq_table[pos_ids].flatten(1)
