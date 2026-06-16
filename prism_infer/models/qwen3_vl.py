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
        return (self.weight * x).to(dtype=input_dtype)


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

        # QK-Norm: 对 Q 和 K 沿 head_dim 做 L2 归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # M-RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_mrope(q, k, cos, sin)

        # GQA: 将 KV head 复制以匹配 Q head 数
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled Dot-Product Attention
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            is_causal=True, scale=self.scale)

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
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextModel(nn.Module):
    """LLM backbone: embed_tokens → 36 layers → final norm."""

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

    def forward(self, input_ids: torch.Tensor,
                position_embeddings: tuple | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                   position_embeddings=position_embeddings,
                                   attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Qwen3VLModel — Vision + LLM + DeepStack 注入
# ═══════════════════════════════════════════════════════════════
class Qwen3VLModel(nn.Module):
    """Qwen3-VL 模型主体: Vision Encoder + LLM Backbone + DeepStack 注入."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.visual = VisionEncoder(dtype)
        self.language_model = Qwen3VLTextModel(dtype=dtype)
        self.deepstack_visual_indexes = [8, 16, 24]

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
        hidden_states = self.language_model.embed_tokens(input_ids)

        # 2. Vision encoding (如果有图)
        if pixel_values is not None and image_grid_thw is not None:
            main_vis, deepstack_vis = self.visual(pixel_values, image_grid_thw)

            # 将主视觉特征替换 <|image_pad|> 占位符
            # 占位符在 input_ids 中是一段连续的 <|image_pad|>
            # 找到视觉占位符的位置并替换
            vis_token_count = main_vis.shape[0]  # 196
            # 简化: 假设视觉特征替换序列开头 (实际应该根据 token 位置)
            # FIXME: 后续实现精确的视觉 token 位置检测
            hidden_states[:, :vis_token_count] = main_vis.unsqueeze(0)

            # DeepStack 注入: 在特定层之后注入视觉特征
            for ds_idx, layer_idx in enumerate(self.deepstack_visual_indexes):
                # 在 attention 之后、MLP 之前注入
                # 这是简化实现, 实际上 HF 在 decoder layer 内部注入
                # FIXME: 实现精确的 per-layer DeepStack 注入
                pass

        # 3. LLM forward
        hidden_states = self.language_model.norm(
            self._run_layers(hidden_states, position_embeddings, attention_mask))
        return hidden_states

    def _run_layers(self, hidden_states, position_embeddings, attention_mask):
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states,
                                   position_embeddings=position_embeddings,
                                   attention_mask=attention_mask)
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
