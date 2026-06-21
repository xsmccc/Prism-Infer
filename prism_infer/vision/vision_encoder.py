"""
Vision Encoder for Qwen3-VL-8B — 自实现 (参照 vLLM qwen3_vl.py)。

Ref: vLLM v0.22.0, vllm/model_executor/models/qwen3_vl.py L348-L700+

Architecture:
  PatchEmbed (Conv3d) → PosEmbed → 27× ViTBlock → 4× PatchMerger
  Output: (main_features [196, 4096], [ds0, ds1, ds2] each [196, 4096])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(config, *names, default=None):
    """Read an attribute/key from a HF config-like object without importing transformers."""
    if config is None:
        return default
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


# ═══════════════════════════════════════════════════════════════
# PatchEmbed — Conv3d Patch Embedding
# Ref: vLLM qwen3_vl.py L348-L374
# ═══════════════════════════════════════════════════════════════
class PatchEmbed(nn.Module):
    """Conv3d patch embedding: pixel_values → patch features.

    pixel_values shape: [N, 1536] where 1536 = in_channels * temporal * patch * patch
    patch features shape: [N, hidden_size=1152]
    """

    def __init__(self, in_channels: int = 3, hidden_size: int = 1152,
                 temporal_patch_size: int = 2, patch_size: int = 16,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        # Conv3d: in_channels=3(RGB), out_channels=1152, kernel=(2,16,16), stride=(2,16,16)
        self.proj = nn.Conv3d(
            in_channels=in_channels, # 3
            out_channels=hidden_size, # 1152
            kernel_size=kernel_size, # (2,16,16)
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
        x = x.view(L, self.in_channels, self.temporal_patch_size,
                    self.patch_size, self.patch_size)
        # Conv3d: [L, 3, 2, 16, 16] → [L, 1152, 1, 1, 1]
        x = self.proj(x)
        # squeeze → [L, 1152]
        x = x.view(L, self.hidden_size)
        return x


# ═══════════════════════════════════════════════════════════════
# ViTMLP — Vision Transformer FFN
# Ref: vLLM qwen3_vl.py L377-L411
# ═══════════════════════════════════════════════════════════════
class ViTMLP(nn.Module):
    """ViT FFN: Linear(1152→4304) + GELU-Tanh + Linear(4304→1152).

    Qwen3-VL ViT 使用 GELU-Tanh 而非 LLM 的 SiLU。
    """

    def __init__(self, dim: int = 1152, hidden_dim: int = 4304,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1(x)
        x = F.gelu(x, approximate='tanh')  # GELU-Tanh
        x = self.linear_fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════
# ViTAttention — Vision Transformer Self-Attention (合并 QKV, 16头, 双向)
# Ref: HF Qwen3VLVisionAttention, vLLM qwen2_5_vl.py L313-L397
# ═══════════════════════════════════════════════════════════════
class ViTAttention(nn.Module):
    """ViT Self-Attention: 合并 QKV 投影 + 16 头 + 双向注意力.

    与 LLM Attention 的关键区别:
      - QKV 是合并投影 (一次 GEMM 出 Q+K+V)，LLM 是分开的
      - 双向注意力 (无 causal mask)，LLM 是单向的
      - 16 Q-heads = 16 KV-heads (无 GQA)，LLM 有 GQA (32:8)
      - head_dim = 72 (1152/16)，LLM head_dim = 128
      - 使用 2D RoPE (当前未实现，后续添加)
    """

    def __init__(self, dim: int = 1152, num_heads: int = 16,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 72
        self.scale = self.head_dim ** -0.5

        # 合并 QKV: 3 * dim = 3 * 1152 = 3456
        self.qkv = nn.Linear(dim, dim * 3, bias=True, dtype=dtype)
        # 输出投影
        self.proj = nn.Linear(dim, dim, bias=True, dtype=dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE helper: 将最后一维的后半取负，交换前后半"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
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
        if x_f.dim() == 3:
            cos_f = cos_f.squeeze(0)
            sin_f = sin_f.squeeze(0)
        result = x_f * cos_f + ViTAttention.rotate_half(x_f) * sin_f
        return result.to(orig_dtype)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor = None,
                sin: torch.Tensor = None) -> torch.Tensor:
        """x: [N, 1152] or [B, N, 1152] → same shape."""
        squeeze_out = False
        if x.dim() == 2:
            x = x.unsqueeze(0)   # [N, 1152] → [1, N, 1152]
            squeeze_out = True

        B, N, D = x.shape

        # 1. QKV 合并投影: [B, N, 1152] → [B, N, 3456]
        qkv = self.qkv(x)

        # 2. 拆分为 Q, K, V: 3456 = 1152×3 → 3 × [B, N, 1152]
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Reshape 为多头: [B, N, 1152] → [B, N, 16, 72] → [B, 16, N, 72]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. 2D RoPE (可选)
        if cos is not None and sin is not None:
            q = self.apply_rotary_emb(q, cos, sin)
            k = self.apply_rotary_emb(k, cos, sin)

        # 5. Scaled Dot-Product Attention (双向, 无 causal mask)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale)

        # 6. 合并多头 + 输出投影: [B, 16, N, 72] → [B, N, 1152]
        o = o.transpose(1, 2).reshape(B, N, D)
        o = self.proj(o)

        if squeeze_out:
            o = o.squeeze(0)
        return o


# ═══════════════════════════════════════════════════════════════
# ViTBlock — Vision Transformer Block (Pre-Norm + 残差)
# Ref: vLLM qwen3_vl.py L414-L465
# ═══════════════════════════════════════════════════════════════
class ViTBlock(nn.Module):
    """ViT Transformer Block: LayerNorm → Attention → +res → LayerNorm → MLP → +res.

    与 LLM DecoderLayer 的区别:
      - 使用 LayerNorm (不是 RMSNorm)
      - 使用 ViTAttention (合并 QKV + 双向 + 2D RoPE)
      - 使用 ViTMLP (GELU-Tanh)
      - 无 cross-attention
    """

    def __init__(self, dim: int = 1152, num_heads: int = 16,
                 mlp_hidden: int = 4304, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06, dtype=dtype)
        self.attn = ViTAttention(dim, num_heads, dtype)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06, dtype=dtype)
        self.mlp = ViTMLP(dim, mlp_hidden, dtype)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor = None,
                sin: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm + Attention + residual
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin)
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════
# PatchMerger — 空间合并 (784→196) + 维度映射 (1152→4096)
# Ref: HF Qwen3VLVisionPatchMerger, vLLM qwen3_vl.py L468-L517
# ═══════════════════════════════════════════════════════════════
class PatchMerger(nn.Module):
    """将 ViT patch 特征合并为 LLM 维度的 visual token.

    空间合并: 相邻 2×2=4 个 patch 特征拼接 (1152×4=4608)
    维度映射: LayerNorm → Linear(4608→4608) → GELU → Linear(4608→4096)
    """

    def __init__(self, dim: int = 1152, out_dim: int = 4096,
                 merge_size: int = 2, post_norm: bool = False,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.merge_size = merge_size
        self.post_norm = post_norm
        merged_dim = dim * merge_size * merge_size  # 1152 * 4 = 4608

        # post_norm=True (deepstack): norm 在合并之后, 对 4608 维做 LN
        # post_norm=False (main): norm 在合并之前, 对 1152 维做 LN
        norm_dim = merged_dim if post_norm else dim
        self.norm = nn.LayerNorm(norm_dim, dtype=dtype)
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

    参照 HF Qwen3VLVisionModel + vLLM Qwen3_VisionTransformer 实现。
    """

    def __init__(self, config=None, dtype: torch.dtype | None = None):
        super().__init__()
        # Backward compatibility: VisionEncoder(torch.bfloat16)
        if isinstance(config, torch.dtype):
            dtype = config
            config = None

        dtype = dtype or _cfg_get(config, "torch_dtype", "dtype",
                                  default=torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))

        dim = _cfg_get(config, "hidden_size", "embed_dim", default=1152)
        in_channels = _cfg_get(config, "in_channels", default=3)
        temporal_patch_size = _cfg_get(config, "temporal_patch_size", default=2)
        patch_size = _cfg_get(config, "patch_size", default=16)
        num_heads = _cfg_get(config, "num_heads", "num_attention_heads", default=16)
        mlp_hidden = _cfg_get(config, "intermediate_size", "mlp_hidden_size",
                              default=4304)
        num_layers = _cfg_get(config, "depth", "num_hidden_layers",
                              "num_layers", default=27)
        out_dim = _cfg_get(config, "out_hidden_size", "output_hidden_size",
                           default=4096)
        pos_embed_size = _cfg_get(config, "num_position_embeddings",
                                  "max_position_embeddings", default=2304)
        spatial_merge_size = _cfg_get(config, "spatial_merge_size", default=2)
        deepstack_indexes = _cfg_get(config, "deepstack_visual_indexes",
                                     "deepstack_visual_indices",
                                     default=[8, 16, 24])

        # Patch Embed
        self.patch_embed = PatchEmbed(
            in_channels, dim, temporal_patch_size, patch_size, dtype)

        # 可学习位置编码 (最多 2304 个 patch)
        self.pos_embed = nn.Embedding(pos_embed_size, dim, dtype=dtype)
        self.num_grid_per_side = int(pos_embed_size ** 0.5)  # 48 for Qwen3-VL-8B

        # 27 ViT Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, mlp_hidden, dtype) for _ in range(num_layers)
        ])

        # 4 Mergers: 1 主 (post_norm=False) + 3 DeepStack (post_norm=True)
        self.merger = PatchMerger(dim, out_dim, spatial_merge_size,
                                  post_norm=False, dtype=dtype)
        self.deepstack_merger_list = nn.ModuleList([
            PatchMerger(dim, out_dim, spatial_merge_size,
                        post_norm=True, dtype=dtype)
            for _ in range(len(deepstack_indexes))
        ])
        self.deepstack_visual_indexes = list(deepstack_indexes)
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = dim // num_heads

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """生成 2D RoPE 的频率 embedding (参照 HF rot_pos_emb).

        grid_thw: [[T, H, W]], e.g. [[1, 28, 28]]
        返回: [total_patches, head_dim/2] = [784, 36]
        """
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        dim = self.head_dim // 2  # HF: rot_pos_emb 返回 head_dim/2
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long,
                              device=next(self.parameters()).device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // self.spatial_merge_size, width // self.spatial_merge_size

            block_rows = torch.arange(merged_h)
            block_cols = torch.arange(merged_w)
            intra_row = torch.arange(self.spatial_merge_size)
            intra_col = torch.arange(self.spatial_merge_size)

            row_idx = block_rows[:, None, None, None] * self.spatial_merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * self.spatial_merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            n = coords.shape[0]
            pos_ids[offset:offset + n] = coords
            offset += n

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self._compute_rope_freqs(dim, max_hw, pos_ids.device)
        embeddings = freq_table[pos_ids]  # [N, 2, dim/2]
        return embeddings.flatten(1)       # [N, dim]

    def _pos_embed_interpolate(self, grid_thw: torch.Tensor,
                                 device: torch.device) -> torch.Tensor:
        """双线性插值 + spatial merge 重排 (完全参照 HF fast_pos_embed_interpolate)."""
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clamp(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clamp(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Spatial merge permute (参照 HF)
        grid_hs, grid_ws = grid_thw[:, 1], grid_thw[:, 2]
        patch_pos_embeds = patch_pos_embeds.split(
            [int(h) * int(w) for h, w in zip(grid_hs, grid_ws)])
        outputs = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_thw[:, 0], grid_hs, grid_ws):
            h, w = int(h), int(w)
            pos_embed = pos_embed.repeat(int(t), 1) if int(t) > 1 else pos_embed
            pos_embed = (
                pos_embed.view(int(t), h // self.spatial_merge_size, self.spatial_merge_size,
                               w // self.spatial_merge_size, self.spatial_merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            outputs.append(pos_embed)
        return torch.cat(outputs, dim=0).to(dtype=dtype)

    @staticmethod
    def _compute_rope_freqs(dim: int, max_pos: int,
                            device: torch.device) -> torch.Tensor:
        """计算 RoPE 频率表: [max_pos, dim/2].

        freqs[pos, i] = pos / theta^(2i/dim)
        theta = 10000 (标准 RoPE base)
        """
        theta = 10000.0
        freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        pos = torch.arange(max_pos, device=device).float()
        return pos[:, None] * freq[None, :]  # [max_pos, dim/2]

    def _get_position_embeddings(self, grid_thw: torch.Tensor,
                                  hidden_states: torch.Tensor):
        """生成 RoPE 的 (cos, sin) 对.

        rot_pos_emb 输出 [N, head_dim/2] = [N, 36],
        cat 后 [N, head_dim] = [N, 72].
        """
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [N, 36]
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [N, 72]
        return emb.cos(), emb.sin()  # each [N, 72]

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> tuple:
        """前向传播.

        pixel_values: [N, 1536]
        grid_thw: [[T, H, W]]
        返回: (main [N/4, 4096], [ds0, ds1, ds2] each [N/4, 4096])
        """
        # 1. Patch Embed + Position Embed (双线性插值)
        x = self.patch_embed(pixel_values)  # [N, 1152]
        x = x + self._pos_embed_interpolate(grid_thw, x.device)

        # 2. RoPE cos/sin
        cos, sin = self._get_position_embeddings(grid_thw, x)

        # 3. 27 ViT Blocks + DeepStack 提取
        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos=cos, sin=sin)
            if i in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(i)
                ds = self.deepstack_merger_list[ds_idx](x)
                deepstack_features.append(ds)

        # 4. 主 Merger
        main = self.merger(x)

        return (main, deepstack_features)
