"""
Qwen3-VL-8B 完整模型定义 (Vision + LLM + DeepStack).

参照 HF Qwen3VLForConditionalGeneration 结构。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_infer.vision.vision_encoder import VisionEncoder
from prism_infer.vision.mrope import MRope, apply_mrope


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextRMSNorm
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextRMSNorm(nn.Module):
    """LLM 端的 RMSNorm (与 ViT 的 LayerNorm 不同)."""

    def __init__(self, dim: int, eps: float = 1e-06, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # 先 cast 再乘 weight, 与 HF 运算顺序一致 (避免 bf16×fp32→fp32 放大误差)
        return self.weight * x.to(dtype=input_dtype)


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextAttention — LLM Self-Attention
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextAttention(nn.Module):
    """LLM Attention: 分开 QKV + QK-Norm + M-RoPE + GQA."""

    def __init__(self, hidden_size: int = 4096, num_heads: int = 32,
                 num_kv_heads: int = 8, head_dim: int = 128,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_key_value_groups = num_heads // num_kv_heads  # 4

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)

        self.q_norm = Qwen3VLTextRMSNorm(head_dim, dtype=dtype)
        self.k_norm = Qwen3VLTextRMSNorm(head_dim, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # M-RoPE (先 RoPE 再 QK-Norm, 与 HF 顺序一致)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_mrope(q, k, cos, sin)

        # QK-Norm: 对 Q 和 K 沿 head_dim 做 L2 归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # GQA: 将 KV head 复制以匹配 Q head 数
        # (nump_key_value_groups=32/8=4, 每个 KV head 服务 4 个 Q head)
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled Dot-Product Attention
        # 如果没有显式 mask, 用 causal; 如果有, 只用 mask (避免 is_causal 被忽略)
        sdpa_kwargs = {"scale": self.scale}
        if attention_mask is not None:
            sdpa_kwargs["attn_mask"] = attention_mask
        else:
            sdpa_kwargs["is_causal"] = True
        o = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

        o = o.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(o)


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextMLP — LLM FFN (SwiGLU)
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextMLP(nn.Module):
    """LLM MLP: Gate-Up-Down (SwiGLU)."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextDecoderLayer
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextDecoderLayer(nn.Module):
    """LLM Decoder Layer: RMSNorm→Attn→+res→RMSNorm→MLP→+res."""

    def __init__(self, hidden_size: int = 4096, num_heads: int = 32,
                 num_kv_heads: int = 8, intermediate_size: int = 12288,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, dtype=dtype)
        self.self_attn = Qwen3VLTextAttention(
            hidden_size, num_heads, num_kv_heads, dtype=dtype)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size, dtype=dtype)
        self.mlp = Qwen3VLTextMLP(hidden_size, intermediate_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states,
                                        position_embeddings=position_embeddings,
                                        attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextModel — LLM Backbone (36 layers)
# Ref: HF Qwen3VLTextModel, modeling_qwen3_vl.py L741-L857
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextModel(nn.Module):
    """LLM backbone: embed_tokens → 36 layers → final norm + DeepStack injection.

    DeepStack 注入发生在 LLM layers 0, 1, 2 之后 (与 HF 一致).
    注意: [8, 16, 24] 是 ViT 侧提取 deepstack 特征的层号, 不是注入层号.
    Ref: HF Qwen3VLTextModel.forward L834-L840
    """

    def __init__(self, vocab_size: int = 151936, hidden_size: int = 4096,
                 num_heads: int = 32, num_kv_heads: int = 8,
                 num_layers: int = 36, intermediate_size: int = 12288,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen3VLTextDecoderLayer(hidden_size, num_heads, num_kv_heads,
                                     intermediate_size, dtype)
            for _ in range(num_layers)
        ])
        self.norm = Qwen3VLTextRMSNorm(hidden_size, dtype=dtype)
        # M-RoPE: LLM 3D 位置编码 (head_dim=128, theta=5M, mrope_section=[24,20,20])
        self.rotary_emb = MRope(head_dim=128, theta=5000000.0,
                                mrope_section=[24, 20, 20])

    def forward(self, input_ids: torch.Tensor | None = None,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                # DeepStack 参数 (参照 HF L772-L774)
                visual_pos_masks: torch.Tensor | None = None,
                deepstack_visual_embeds: list[torch.Tensor] | None = None,
                ) -> torch.Tensor:
        """
        visual_pos_masks: [batch, seqlen] bool, 视觉 token 的位置 mask
        deepstack_visual_embeds: list of 3 tensors, each [N_vis, 4096]
            分别注入到 layer 0, 1, 2 之后
        """
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("必须提供 input_ids 或 inputs_embeds")

        # 自动计算 M-RoPE position embeddings (参照 HF L819-L820)
        if position_embeddings is None:
            if position_ids is None:
                seqlen = hidden_states.shape[1]
                device = hidden_states.device
                # 纯文本: [1, seqlen] position_ids
                position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states,
                                   position_embeddings=position_embeddings,
                                   attention_mask=attention_mask)

            # DeepStack 注入: 在 layers 0, 1, 2 之后 (HF L835-L840)
            # deepstack_visual_embeds 是 list of 3, 对应 layer 0,1,2
            if (deepstack_visual_embeds is not None
                    and layer_idx < len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor,
                           visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor) -> torch.Tensor:
        """将 deepstack 视觉特征加到 hidden_states 的视觉 token 位置.
        Ref: HF Qwen3VLTextModel._deepstack_process L849-L857
        """
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        # hidden_states[visual_pos_masks, :] += visual_embeds
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + visual_embeds
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Qwen3VLModel — Vision + LLM + DeepStack 注入
# Ref: HF Qwen3VLModel, modeling_qwen3_vl.py L1162-L1264
# ═══════════════════════════════════════════════════════════════
class Qwen3VLModel(nn.Module):
    """Qwen3-VL 模型主体: Vision Encoder + LLM Backbone + DeepStack 注入.

    数据流 (与 HF 一致):
      1. 文本 embedding → inputs_embeds
      2. Vision Encoder → 主特征 + 3 路 DeepStack
      3. 用 image_token_id 找到视觉占位符位置 → masked_scatter 替换
      4. LLM forward (每 layer 之后检查是否需要注入 deepstack)
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.visual = VisionEncoder(dtype)
        self.language_model = Qwen3VLTextModel(dtype=dtype)
        # <|image_pad|> token ID (HF config.image_token_id)
        self.image_token_id = 151655

    def forward(self, input_ids: torch.Tensor,
                pixel_values: torch.Tensor | None = None,
                image_grid_thw: torch.Tensor | None = None,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: [batch, seqlen]
        pixel_values: [N_patches, 1536] or None
        image_grid_thw: [[T, H, W]] or None
        """
        # 1. 文本 embedding
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None

        # 2. Vision encoding + 特征注入 (参照 HF L1191-L1237)
        if pixel_values is not None and image_grid_thw is not None:
            main_vis, deepstack_vis = self.visual(pixel_values, image_grid_thw)
            # main_vis: [N_vis, 4096], deepstack_vis: list of 3 × [N_vis, 4096]

            # 找到视觉 token 占位符位置 (HF L1091: input_ids == image_token_id)
            visual_pos_masks_2d = (input_ids == self.image_token_id)  # [batch, seqlen]
            visual_pos_masks = visual_pos_masks_2d.unsqueeze(-1)      # [batch, seqlen, 1]

            # 验证 token 数量匹配 (参照 HF L1097-L1099)
            n_vis_tokens = visual_pos_masks_2d.sum().item()
            expected_elements = n_vis_tokens * inputs_embeds.shape[-1]
            actual_elements = main_vis.numel()
            if expected_elements != actual_elements:
                raise ValueError(
                    f"视觉 token 数量不匹配: input 中有 {n_vis_tokens} 个 "
                    f"<|image_pad|> ({n_vis_tokens}×{inputs_embeds.shape[-1]}"
                    f"={expected_elements} elements), "
                    f"但 Vision Encoder 输出 {main_vis.shape[0]} 个 token "
                    f"({actual_elements} elements)"
                )

            # 用视觉特征替换占位符 (HF L1201: masked_scatter)
            inputs_embeds = inputs_embeds.masked_scatter(
                visual_pos_masks.to(inputs_embeds.device), main_vis)

            deepstack_visual_embeds = deepstack_vis
            # visual_pos_masks 转回 2D 给 _deepstack_process 使用
            visual_pos_masks = visual_pos_masks_2d

        # 3. LLM forward (DeepStack 注入在 Qwen3VLTextModel.forward 内部)
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Qwen3VLForCausalLM — 最外层
# ═══════════════════════════════════════════════════════════════
class Qwen3VLForCausalLM(nn.Module):
    """Qwen3-VL-8B 完整模型."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.model = Qwen3VLModel(dtype)
        self.lm_head = nn.Linear(4096, 151936, bias=False, dtype=dtype)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight  # tie weights

    def forward(self, input_ids: torch.Tensor,
                pixel_values: torch.Tensor | None = None,
                image_grid_thw: torch.Tensor | None = None,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, pixel_values, image_grid_thw,
                          position_embeddings, attention_mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
