"""Qwen3-VL Vision Encoder 自实现。

对齐参考:
  HF transformers 本地源码
  `.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:46-275`
  `.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:564-753`

架构:
  PatchEmbed (Conv3d) -> PosEmbed -> 27x ViTBlock -> 4x PatchMerger
  Output: (main_features [196, 4096], [ds0, ds1, ds2] each [196, 4096])
"""

from math import isqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_infer.models.qwen3_vl_architecture import (
    CANONICAL_VISION_HIDDEN_SIZE,
    CANONICAL_VISION_IN_CHANNELS,
    CANONICAL_VISION_INTERMEDIATE_SIZE,
    CANONICAL_VISION_NUM_HEADS,
    CANONICAL_VISION_OUTPUT_SIZE,
    CANONICAL_VISION_PATCH_SIZE,
    CANONICAL_VISION_ROPE_THETA,
    CANONICAL_VISION_SPATIAL_MERGE_SIZE,
    CANONICAL_VISION_TEMPORAL_PATCH_SIZE,
    Qwen3VLVisionArchitecture,
)
from prism_infer.vision.backends import (
    VisionAttentionBackendName,
    normalize_vision_attention_backend,
)

try:
    from flash_attn import flash_attn_varlen_func as vision_flash_attn_varlen_func

    HAS_VISION_FLASH_ATTN = True
except ImportError:
    vision_flash_attn_varlen_func = None
    HAS_VISION_FLASH_ATTN = False


UNBATCHED_ATTENTION_TENSOR_RANK = 3
VISION_TOKEN_MATRIX_RANK = 2
SINGLE_SEQUENCE_CU_SEQLENS_COUNT = 2
QKV_PROJECTION_COUNT = 3


def _config_dtype(config: object | None) -> torch.dtype:
    if config is None:
        return torch.bfloat16
    for name in ("torch_dtype", "dtype"):
        if isinstance(config, dict) and name in config:
            value = config[name]
            break
        if hasattr(config, name):
            value = getattr(config, name)
            break
    else:
        return torch.bfloat16
    if isinstance(value, str):
        value = getattr(torch, value.replace("torch.", ""), None)
    if not isinstance(value, torch.dtype):
        raise TypeError(f"vision dtype must resolve to torch.dtype, got {value!r}")
    return value


# ═══════════════════════════════════════════════════════════════
# PatchEmbed — Conv3d Patch Embedding
# Ref: HF modeling_qwen3_vl.py:59-76
# ═══════════════════════════════════════════════════════════════
class PatchEmbed(nn.Module):
    """Conv3d patch embedding: pixel_values → patch features.

    pixel_values shape: [N, 1536] where 1536 = in_channels * temporal * patch * patch
    patch features shape: [N, hidden_size=1152]
    """

    def __init__(
        self,
        in_channels: int = CANONICAL_VISION_IN_CHANNELS,
        hidden_size: int = CANONICAL_VISION_HIDDEN_SIZE,
        temporal_patch_size: int = CANONICAL_VISION_TEMPORAL_PATCH_SIZE,
        patch_size: int = CANONICAL_VISION_PATCH_SIZE,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        # Conv3d: in_channels=3(RGB), out_channels=1152, kernel=(2,16,16), stride=(2,16,16)
        self.proj = nn.Conv3d(
            in_channels=in_channels,  # 3
            out_channels=hidden_size,  # 1152
            kernel_size=kernel_size,  # (2,16,16)
            stride=kernel_size,
            bias=True,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C*T*H*W] = [N, 1536] where 1536 = 3*2*16*16
        L, C_all = x.shape
        # 转为权重精度 (bf16)，避免 fp32×bf16 混合精度误差
        target_dtype = self.proj.weight.dtype
        x = x.to(dtype=target_dtype)
        # reshape to [L, C, T, H, W] = [L, 3, 2, 16, 16]
        x = x.view(L, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        # Conv3d: [L, 3, 2, 16, 16] → [L, 1152, 1, 1, 1]
        x = self.proj(x)
        # squeeze → [L, 1152]
        x = x.view(L, self.hidden_size)
        return x


# ═══════════════════════════════════════════════════════════════
# ViTMLP — Vision Transformer FFN
# Ref: HF modeling_qwen3_vl.py:46-56
# ═══════════════════════════════════════════════════════════════
class ViTMLP(nn.Module):
    """ViT FFN: Linear(1152→4304) + GELU-Tanh + Linear(4304→1152).

    Qwen3-VL ViT 使用 GELU-Tanh 而非 LLM 的 SiLU。
    """

    def __init__(
        self,
        dim: int = CANONICAL_VISION_HIDDEN_SIZE,
        hidden_dim: int = CANONICAL_VISION_INTERMEDIATE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")  # GELU-Tanh
        x = self.linear_fc2(x)
        return x


class VisionRotaryEmbedding(nn.Module):
    """Vision 2D RoPE 的一维频率表.

    HF 在构造 `Qwen3VLVisionRotaryEmbedding` 时注册 `inv_freq` buffer，
    forward 只按最大高宽取外积。这里保持同样的数据流，避免每次 forward
    动态重算频率导致 bf16 量化边界发生变化。
    Ref: transformers/models/qwen3_vl/modeling_qwen3_vl.py:79-90
    """

    def __init__(self, dim: int, theta: float = CANONICAL_VISION_ROPE_THETA) -> None:
        super().__init__()
        target_device = torch.empty((), dtype=torch.float).device
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim))
        self.register_buffer("inv_freq", inv_freq.to(target_device), persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


# ═══════════════════════════════════════════════════════════════
# ViTAttention — Vision Transformer Self-Attention (合并 QKV, 16头, 双向)
# Ref: HF modeling_qwen3_vl.py:168-248
# ═══════════════════════════════════════════════════════════════
class ViTAttention(nn.Module):
    """ViT Self-Attention: 合并 QKV 投影 + 16 头 + 双向注意力.

    与 LLM Attention 的关键区别:
      - QKV 是合并投影 (一次 GEMM 出 Q+K+V)，LLM 是分开的
      - 双向注意力 (无 causal mask)，LLM 是单向的
      - 16 Q-heads = 16 KV-heads (无 GQA)，LLM 有 GQA (32:8)
      - head_dim = 72 (1152/16)，LLM head_dim = 128
      - 使用 2D RoPE
    """

    def __init__(
        self,
        dim: int = CANONICAL_VISION_HIDDEN_SIZE,
        num_heads: int = CANONICAL_VISION_NUM_HEADS,
        dtype: torch.dtype = torch.bfloat16,
        *,
        attention_backend: VisionAttentionBackendName | str = VisionAttentionBackendName.SDPA,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 72
        self.scale = self.head_dim**-0.5
        self.attention_backend = normalize_vision_attention_backend(attention_backend)
        if (
            self.attention_backend is VisionAttentionBackendName.FLASH_ATTN
            and not HAS_VISION_FLASH_ATTN
        ):
            raise RuntimeError(
                "vision_attention_backend='flash_attn' was requested, but flash-attn "
                "is not installed"
            )

        # 合并 QKV: 3 * dim = 3 * 1152 = 3456
        self.qkv = nn.Linear(dim, dim * QKV_PROJECTION_COUNT, bias=True, dtype=dtype)
        # 输出投影
        self.proj = nn.Linear(dim, dim, bias=True, dtype=dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE helper: 将最后一维的后半取负，交换前后半"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """对 x 应用 Rotary Position Embedding. HF 在 float32 下计算以减少精度损失。

        x:   [B, heads, N, head_dim] 或 [heads, N, head_dim]
        cos/sin: [N, head_dim]
        返回同 shape 同 dtype 的旋转后 tensor
        """
        orig_dtype = x.dtype
        # HF 在 float32 下做 RoPE 以减少精度损失
        x_f = x.float()
        cos_f = cos.float()
        sin_f = sin.float()
        # cos: [N, head_dim] → [1, 1, N, head_dim]
        cos_f = cos_f.unsqueeze(0).unsqueeze(0)
        sin_f = sin_f.unsqueeze(0).unsqueeze(0)
        if x_f.dim() == UNBATCHED_ATTENTION_TENSOR_RANK:
            cos_f = cos_f.squeeze(0)
            sin_f = sin_f.squeeze(0)
        result = x_f * cos_f + ViTAttention.rotate_half(x_f) * sin_f
        return result.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """x: [N, 1152] or [B, N, 1152] → same shape.

        cu_seqlens: [num_images + 1]，多图时按每张图分段做双向
        attention，避免不同图片 patch 之间互相注意。
        """
        squeeze_out = False
        if x.dim() == VISION_TOKEN_MATRIX_RANK:
            x = x.unsqueeze(0)  # [N, 1152] → [1, N, 1152]
            squeeze_out = True

        B, N, D = x.shape

        # 1. QKV 合并投影: [B, N, 1152] → [B, N, 3456]
        qkv = self.qkv(x)

        # 2. 拆分为 Q, K, V: 3456 = 1152×3 → 3 × [B, N, 1152]
        q, k, v = qkv.chunk(QKV_PROJECTION_COUNT, dim=-1)

        # 3. Reshape 为多头: [B, N, 1152] → [B, N, 16, 72] → [B, 16, N, 72]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. 2D RoPE (可选)
        if cos is not None and sin is not None:
            q = self.apply_rotary_emb(q, cos, sin)
            k = self.apply_rotary_emb(k, cos, sin)

        # 5. Scaled Dot-Product Attention (双向, 无 causal mask)。
        # 多图时 HF eager 路径按 cu_seqlens 分段计算，不能让不同图片
        # patch 之间互相注意。单图或未传 cu_seqlens 时保持原路径。
        if self.attention_backend is VisionAttentionBackendName.FLASH_ATTN:
            if cu_seqlens is None:
                raise RuntimeError("vision FlashAttention requires cu_seqlens")
            if not q.is_cuda or B != 1 or q.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "vision FlashAttention requires a single packed CUDA batch in fp16 or bf16"
                )
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            q_varlen = q[0].transpose(0, 1).contiguous()
            k_varlen = k[0].transpose(0, 1).contiguous()
            v_varlen = v[0].transpose(0, 1).contiguous()
            o_varlen = vision_flash_attn_varlen_func(
                q_varlen,
                k_varlen,
                v_varlen,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scale,
                causal=False,
                deterministic=True,
            )
            o = o_varlen.transpose(0, 1).unsqueeze(0)
        elif cu_seqlens is not None and cu_seqlens.numel() > SINGLE_SEQUENCE_CU_SEQLENS_COUNT:
            outputs = []
            for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                q_i = q[:, :, start:end, :]
                k_i = k[:, :, start:end, :]
                v_i = v[:, :, start:end, :]
                outputs.append(
                    F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False, scale=self.scale)
                )
            o = torch.cat(outputs, dim=2)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)

        # 6. 合并多头 + 输出投影: [B, 16, N, 72] → [B, N, 1152]
        o = o.transpose(1, 2).reshape(B, N, D)
        o = self.proj(o)

        if squeeze_out:
            o = o.squeeze(0)
        return o


# ═══════════════════════════════════════════════════════════════
# ViTBlock — Vision Transformer Block (Pre-Norm + 残差)
# Ref: HF modeling_qwen3_vl.py:251-275
# ═══════════════════════════════════════════════════════════════
class ViTBlock(nn.Module):
    """ViT Transformer Block: LayerNorm → Attention → +res → LayerNorm → MLP → +res.

    与 LLM DecoderLayer 的区别:
      - 使用 LayerNorm (不是 RMSNorm)
      - 使用 ViTAttention (合并 QKV + 双向 + 2D RoPE)
      - 使用 ViTMLP (GELU-Tanh)
      - 无 cross-attention
    """

    def __init__(
        self,
        dim: int = CANONICAL_VISION_HIDDEN_SIZE,
        num_heads: int = CANONICAL_VISION_NUM_HEADS,
        mlp_hidden: int = CANONICAL_VISION_INTERMEDIATE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        *,
        attention_backend: VisionAttentionBackendName | str = VisionAttentionBackendName.SDPA,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06, dtype=dtype)
        self.attn = ViTAttention(
            dim,
            num_heads,
            dtype,
            attention_backend=attention_backend,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-06, dtype=dtype)
        self.mlp = ViTMLP(dim, mlp_hidden, dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # Pre-norm + Attention + residual
        x = x + self.attn(
            self.norm1(x),
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════
# PatchMerger — 空间合并 (784→196) + 维度映射 (1152→4096)
# Ref: HF modeling_qwen3_vl.py:93-106
# ═══════════════════════════════════════════════════════════════
class PatchMerger(nn.Module):
    """将 ViT patch 特征合并为 LLM 维度的 visual token.

    空间合并: 相邻 2×2=4 个 patch 特征拼接 (1152×4=4608)
    维度映射: LayerNorm → Linear(4608→4608) → GELU → Linear(4608→4096)
    """

    def __init__(
        self,
        dim: int = CANONICAL_VISION_HIDDEN_SIZE,
        out_dim: int = CANONICAL_VISION_OUTPUT_SIZE,
        merge_size: int = CANONICAL_VISION_SPATIAL_MERGE_SIZE,
        post_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.merge_size = merge_size
        self.post_norm = post_norm
        merged_dim = dim * merge_size * merge_size  # 1152 * 4 = 4608

        # post_norm=True (deepstack): norm 在合并之后, 对 4608 维做 LN
        # post_norm=False (main): norm 在合并之前, 对 1152 维做 LN
        norm_dim = merged_dim if post_norm else dim
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6, dtype=dtype)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(merged_dim, out_dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            # DeepStack: 先合并，再 norm
            x = self.norm(x.view(-1, self.dim * self.merge_size * self.merge_size))
        else:
            # Main: 先 norm (per-patch)，再合并
            x = self.norm(x)
            x = x.view(-1, self.dim * self.merge_size * self.merge_size)
        x = F.gelu(self.linear_fc1(x))
        x = self.linear_fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════
# VisionEncoder — 完整 ViT 编码器 (PatchEmbed → 27 Blocks → 4 Mergers)
# ═══════════════════════════════════════════════════════════════
class VisionEncoder(nn.Module):
    """Qwen3-VL Vision Encoder: patch→27层→4路Merger→(main, [ds0,ds1,ds2]).

    参照 HF Qwen3VLVisionModel 的结构自实现。
    """

    def __init__(
        self,
        config=None,
        dtype: torch.dtype | None = None,
        *,
        attention_backend: VisionAttentionBackendName | str = VisionAttentionBackendName.SDPA,
    ):
        super().__init__()
        # Backward compatibility: VisionEncoder(torch.bfloat16)
        if isinstance(config, torch.dtype):
            dtype = config
            config = None

        dtype = dtype or _config_dtype(config)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"vision dtype must be torch.dtype, got {dtype!r}")
        self.attention_backend = normalize_vision_attention_backend(attention_backend)

        architecture = (
            Qwen3VLVisionArchitecture.canonical()
            if config is None
            else Qwen3VLVisionArchitecture.from_config(config)
        )
        dim = architecture.hidden_size
        in_channels = architecture.in_channels
        temporal_patch_size = architecture.temporal_patch_size
        patch_size = architecture.patch_size
        num_heads = architecture.num_heads
        mlp_hidden = architecture.intermediate_size
        num_layers = architecture.depth
        out_dim = architecture.output_size
        pos_embed_size = architecture.num_position_embeddings
        spatial_merge_size = architecture.spatial_merge_size
        deepstack_indexes = architecture.deepstack_visual_indexes

        # Patch Embed
        self.patch_embed = PatchEmbed(in_channels, dim, temporal_patch_size, patch_size, dtype)

        # 可学习位置编码 (最多 2304 个 patch)
        self.pos_embed = nn.Embedding(pos_embed_size, dim, dtype=dtype)
        self.num_grid_per_side = isqrt(pos_embed_size)

        # 27 ViT Blocks
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim,
                    num_heads,
                    mlp_hidden,
                    dtype,
                    attention_backend=self.attention_backend,
                )
                for _ in range(num_layers)
            ]
        )

        # 4 Mergers: 1 主 (post_norm=False) + 3 DeepStack (post_norm=True)
        self.merger = PatchMerger(dim, out_dim, spatial_merge_size, post_norm=False, dtype=dtype)
        self.deepstack_merger_list = nn.ModuleList(
            [
                PatchMerger(dim, out_dim, spatial_merge_size, post_norm=True, dtype=dtype)
                for _ in range(len(deepstack_indexes))
            ]
        )
        self.deepstack_visual_indexes = list(deepstack_indexes)
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(self.head_dim // 2)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """生成 2D RoPE 的频率 embedding (参照 HF rot_pos_emb).

        grid_thw: [[T, H, W]], e.g. [[1, 28, 28]]
        返回: [total_patches, head_dim/2] = [784, 36]
        """
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # [max_hw, head_dim/4]
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // self.spatial_merge_size, width // self.spatial_merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(self.spatial_merge_size, device=device)
            intra_col = torch.arange(self.spatial_merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * self.spatial_merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * self.spatial_merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            n = coords.shape[0]
            pos_ids[offset : offset + n] = coords
            offset += n

        embeddings = freq_table[pos_ids]  # [N, 2, dim/2]
        return embeddings.flatten(1)  # [N, dim]

    def _pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """双线性插值 + spatial merge 重排 (完全参照 HF fast_pos_embed_interpolate)."""
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
        weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

        for t, h, w in grid_thw.to(device).tolist():
            t, h, w = int(t), int(h), int(w)

            h_grid = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_grid = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_floor = h_grid.int()
            w_floor = w_grid.int()
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)

            h_frac = h_grid - h_floor
            w_frac = w_grid - w_floor

            h_floor_offset = h_floor * self.num_grid_per_side
            h_ceil_offset = h_ceil * self.num_grid_per_side

            corner_indices = [
                (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
                (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
            ]
            corner_weights = [
                ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
                ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
                (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
                (h_frac[:, None] * w_frac[None, :]).flatten(),
            ]

            h_idx = torch.arange(h, device=device).view(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            w_idx = torch.arange(w, device=device).view(
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            reorder = (
                (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
                .transpose(1, 2)
                .flatten()
                .repeat(t)
            )

            for i in range(4):
                idx_parts[i].append(corner_indices[i][reorder])
                weight_parts[i].append(corner_weights[i][reorder])

        idx_tensor = torch.stack([torch.cat(part) for part in idx_parts])
        weight_tensor = torch.stack([torch.cat(part) for part in weight_parts])
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        return pos_embeds.sum(0).to(dtype=dtype)

    def _get_position_embeddings(self, grid_thw: torch.Tensor, hidden_states: torch.Tensor):
        """生成 RoPE 的 (cos, sin) 对.

        rot_pos_emb 输出 [N, head_dim/2] = [N, 36],
        cat 后 [N, head_dim] = [N, 72].
        """
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [N, 36]
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [N, 72]
        return emb.cos(), emb.sin()  # each [N, 72]

    def prepare_tensor_region_inputs(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """准备 Vision tensor region 的输入。

        grid 驱动的位置插值、2D RoPE index 和分段边界包含 Python 动态控制流，
        保留在 compile region 外；同时只物化一次最大 segment 长度，避免每层
        为 varlen attention 重复同步。

        pixel_values: [N, patch_input_dim]
        grid_thw: [num_images, 3]
        返回: x、cos/sin、cu_seqlens，以及静态 max_seqlen。
        """

        x = self.patch_embed(pixel_values)  # [N, dim]
        x = x + self._pos_embed_interpolate(grid_thw)
        cos, sin = self._get_position_embeddings(grid_thw, x)
        segment_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        )
        max_seqlen = int(segment_lengths.max().item())
        cu_seqlens = segment_lengths.cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(x.device)
        return x, cos, sin, cu_seqlens, max_seqlen

    def forward_tensor_region(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """执行 ViT blocks、DeepStack mergers 和 main merger。

        x: [N, dim]
        cos/sin: [N, head_dim]
        cu_seqlens: [segments + 1]
        返回: main [N / merge_size^2, out_dim] 和 DeepStack tensor list。
        """

        deepstack_features: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            x = block(
                x,
                cos=cos,
                sin=sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](x))

        main = self.merger(x)
        return main, deepstack_features

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """前向传播。

        pixel_values: [N, 1536]
        grid_thw: [[T, H, W]]
        返回: (main [N/4, 4096], [ds0, ds1, ds2] each [N/4, 4096])
        """
        tensor_inputs = self.prepare_tensor_region_inputs(pixel_values, grid_thw)
        return self.forward_tensor_region(*tensor_inputs)
