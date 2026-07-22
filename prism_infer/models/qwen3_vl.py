"""
Qwen3-VL-8B 完整模型定义 (Vision + LLM + DeepStack).

参照 HF Qwen3VLForConditionalGeneration 结构。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_infer.layers.attention import Attention
from prism_infer.models.qwen3_vl_architecture import (
    CANONICAL_MROPE_SECTION,
    CANONICAL_TEXT_HEAD_DIM,
    CANONICAL_TEXT_HIDDEN_SIZE,
    CANONICAL_TEXT_INTERMEDIATE_SIZE,
    CANONICAL_TEXT_NUM_HEADS,
    CANONICAL_TEXT_NUM_KV_HEADS,
    CANONICAL_TEXT_NUM_LAYERS,
    CANONICAL_TEXT_RMS_NORM_EPS,
    CANONICAL_TEXT_ROPE_THETA,
    CANONICAL_TEXT_VOCAB_SIZE,
    MROPE_AXIS_COUNT,
    Qwen3VLArchitecture,
    Qwen3VLTextArchitecture,
)
from prism_infer.observability import is_trace_enabled, profile_region
from prism_infer.ops.add_rmsnorm import fused_add_rmsnorm
from prism_infer.ops.qk_rmsnorm import fused_qk_rmsnorm
from prism_infer.utils.context import get_context
from prism_infer.vision.backends import VisionAttentionBackendName
from prism_infer.vision.mrope import MRope, apply_mrope
from prism_infer.vision.vision_encoder import VisionEncoder

FLATTENED_TOKEN_FEATURES_RANK = 2
SELECTIVE_FP32_LOGITS_TOP_K = 16
BATCHED_TOKEN_FEATURES_RANK = 3
MROPE_POSITION_MATRIX_RANK = 2
POSITION_EMBEDDING_TENSOR_COUNT = 2


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


def _normalize_dtype(dtype):
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
    return dtype


def _text_config(config):
    return _cfg_get(config, "text_config", default=config)


def _vision_config(config):
    return _cfg_get(config, "vision_config", "vision_config_dict", default=None)


class _PackedLinear(nn.Module):
    """只保存一次连续权重、但不改变外部 state-dict 合同的线性层。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        nn.init.uniform_(
            self.weight,
            -1.0 / math.sqrt(input_size),
            1.0 / math.sqrt(input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        # gate_proj/up_proj 的 Parameter view 仍输出 HF-compatible keys；packed
        # storage 是执行细节，不能再保存一份重复权重。
        return None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # 旧 state dict 会在 sibling gate_proj/up_proj 中直接写入共享 storage。
        return None


class _LinearWeightView(nn.Module):
    """把 packed weight 的一个连续 row slice 暴露为兼容的 Linear module。"""

    def __init__(
        self,
        packed_weight: nn.Parameter,
        start: int,
        length: int,
    ) -> None:
        super().__init__()
        self.start = start
        self.length = length
        self.weight = nn.Parameter(
            packed_weight.narrow(0, start, length),
            requires_grad=packed_weight.requires_grad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def rebind(self, packed_weight: nn.Parameter) -> None:
        """在 Module.to/_apply 后恢复共享 storage，避免转换产生副本。"""

        self._parameters["weight"] = nn.Parameter(
            packed_weight.narrow(0, self.start, self.length),
            requires_grad=packed_weight.requires_grad,
        )


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextRMSNorm
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextRMSNorm(nn.Module):
    """LLM 端的 RMSNorm (与 ViT 的 LayerNorm 不同)."""

    def __init__(self, dim: int, eps: float = CANONICAL_TEXT_RMS_NORM_EPS, dtype=torch.bfloat16):
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

    def __init__(
        self,
        hidden_size: int = CANONICAL_TEXT_HIDDEN_SIZE,
        num_heads: int = CANONICAL_TEXT_NUM_HEADS,
        num_kv_heads: int = CANONICAL_TEXT_NUM_KV_HEADS,
        head_dim: int = CANONICAL_TEXT_HEAD_DIM,
        dtype: torch.dtype = torch.bfloat16,
        rms_norm_eps: float = CANONICAL_TEXT_RMS_NORM_EPS,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.num_key_value_groups = num_heads // num_kv_heads  # 4

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)

        self.q_norm = Qwen3VLTextRMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)
        self.k_norm = Qwen3VLTextRMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)
        self.engine_attn = Attention(num_heads, head_dim, self.scale, num_kv_heads)
        self._compiled_decode_qkv_forward = None
        self.fused_qk_rmsnorm_enabled = False
        self.fused_qk_mrope_enabled = False

    def enable_decode_compile(
        self,
        *,
        mode: str,
        emulate_precision_casts: bool,
        force_same_precision: bool,
    ) -> None:
        """Compile the pure decode QKV-preparation region only.

        KV-cache writes are an explicit state commit and paged attention reads
        aliased per-layer views.  They intentionally stay outside AOT
        functionalization; otherwise Inductor clones the backing storage of a
        mutated view.  The fullgraph region still fuses QK-Norm/M-RoPE around
        the QKV projections, while Prism's validated Triton store/decode
        kernels remain the execution backend.
        """

        if self._compiled_decode_qkv_forward is not None:
            raise RuntimeError("decode attention compile was already enabled")
        compile_options = dict(torch._inductor.list_mode_options(mode))
        compile_options["emulate_precision_casts"] = emulate_precision_casts
        compile_options["force_same_precision"] = force_same_precision
        self._compiled_decode_qkv_forward = torch.compile(
            self._forward_engine_qkv,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options=compile_options,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim == FLATTENED_TOKEN_FEATURES_RANK:
            context = get_context()
            if not context.is_prefill and self._compiled_decode_qkv_forward is not None:
                if is_trace_enabled():
                    raise RuntimeError("decode torch.compile does not support KV trace collection")
                compression_metadata = context.compression_metadata
                if compression_metadata is not None and compression_metadata.enabled:
                    raise RuntimeError(
                        "decode torch.compile requires compression_mode='off', "
                        f"got {compression_metadata.mode!r}"
                    )
                required_context = {
                    "slot_mapping": context.slot_mapping,
                    "context_lens": context.context_lens,
                    "block_tables": context.block_tables,
                    "decode_max_context_len": context.decode_max_context_len,
                }
                missing = [name for name, value in required_context.items() if value is None]
                if missing:
                    raise RuntimeError(
                        f"decode torch.compile is missing context tensors: {', '.join(missing)}"
                    )
                prepared_positions = self._prepare_engine_position_embeddings(
                    position_embeddings,
                    num_tokens=hidden_states.shape[0],
                )
                q, k, v = self._compiled_decode_qkv_forward(
                    hidden_states,
                    prepared_positions,
                )
                output = self.engine_attn.forward_decode_explicit(
                    q,
                    k,
                    v,
                    context,
                )
                return self._project_engine_output(output)
            return self._forward_engine(hidden_states, position_embeddings)

        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm: HF 4.57.1 是先 QK-Norm 再 M-RoPE.
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        # M-RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_mrope(q, k, cos, sin)

        sdpa_kwargs = {"scale": self.scale}
        if attention_mask is not None:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            sdpa_kwargs["attn_mask"] = attention_mask
        else:
            if q.is_cuda:
                sdpa_kwargs["enable_gqa"] = True
            else:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            sdpa_kwargs["is_causal"] = True
        o = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

        o = o.transpose(1, 2).contiguous().reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(o)

    def _forward_engine(
        self, hidden_states: torch.Tensor, position_embeddings: tuple | None = None
    ) -> torch.Tensor:
        """engine flatten 路径: [num_tokens, hidden] -> [num_tokens, hidden]."""

        prepared_positions = self._prepare_engine_position_embeddings(
            position_embeddings,
            num_tokens=hidden_states.shape[0],
        )
        q, k, v = self._forward_engine_qkv(hidden_states, prepared_positions)
        o = self.engine_attn(q, k, v)
        return self._project_engine_output(o)

    def _forward_engine_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure QKV/QK-Norm/M-RoPE region shared by eager and compile."""

        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
        fused_mrope = False
        if (
            self.fused_qk_rmsnorm_enabled
            and q.is_cuda
            and not torch.compiler.is_compiling()
            and not get_context().is_prefill
            and q.shape[0] <= 4
        ):
            fused_positions = {}
            if self.fused_qk_mrope_enabled and position_embeddings is not None:
                cos, sin = position_embeddings
                fused_positions = {"cos": cos, "sin": sin}
                fused_mrope = True
            q, k = fused_qk_rmsnorm(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                eps=float(self.q_norm.eps),
                **fused_positions,
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if position_embeddings is not None and not fused_mrope:
            cos, sin = position_embeddings
            q, k = self._apply_mrope_engine(q, k, cos, sin)
        return q, k, v

    def _project_engine_output(self, output: torch.Tensor) -> torch.Tensor:
        """Apply the output projection outside the stateful cache boundary."""

        return self.o_proj(output.contiguous().reshape(output.shape[0], -1))

    def _prepare_engine_position_embeddings(
        self,
        position_embeddings: tuple | None,
        *,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Validate and normalize M-RoPE tensors before entering Dynamo."""

        if position_embeddings is None:
            return None
        if (
            not isinstance(position_embeddings, tuple)
            or len(position_embeddings) != POSITION_EMBEDDING_TENSOR_COUNT
        ):
            raise TypeError("engine position_embeddings must be a (cos, sin) tensor tuple")
        cos, sin = position_embeddings
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            raise TypeError("engine M-RoPE cos/sin values must be tensors")
        if cos.shape != sin.shape or cos.dtype != sin.dtype or cos.device != sin.device:
            raise ValueError("engine M-RoPE cos/sin tensors must match in shape, dtype, and device")
        if cos.ndim == BATCHED_TOKEN_FEATURES_RANK:
            if cos.shape[0] != 1:
                raise ValueError(f"engine M-RoPE expects batch=1 cos/sin, got {list(cos.shape)}")
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
        expected_shape = (num_tokens, self.head_dim)
        if cos.ndim != FLATTENED_TOKEN_FEATURES_RANK or tuple(cos.shape) != expected_shape:
            raise ValueError(
                "engine M-RoPE expects cos/sin [num_tokens, head_dim], "
                f"expected={list(expected_shape)}, got={list(cos.shape)}"
            )
        return cos, sin

    @staticmethod
    def _apply_mrope_engine(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把 M-RoPE 应用到 engine flatten q/k。

        q: [num_tokens, num_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        cos/sin: [1, num_tokens, head_dim] or [num_tokens, head_dim]
        """

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (
            q * cos
            + torch.cat((-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), dim=-1) * sin
        )
        k = (
            k * cos
            + torch.cat((-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), dim=-1) * sin
        )
        return q, k


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextMLP — LLM FFN (SwiGLU)
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextMLP(nn.Module):
    """LLM MLP: Gate-Up-Down (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int = CANONICAL_TEXT_HIDDEN_SIZE,
        intermediate_size: int = CANONICAL_TEXT_INTERMEDIATE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        projection_mode: str = "packed",
    ):
        super().__init__()
        if projection_mode not in ("legacy", "packed"):
            raise ValueError(
                f"projection_mode must be 'legacy' or 'packed', got {projection_mode!r}"
            )
        self.projection_mode = projection_mode
        self.gate_up_proj = _PackedLinear(
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
        )
        self.gate_proj = _LinearWeightView(
            self.gate_up_proj.weight,
            0,
            intermediate_size,
        )
        self.up_proj = _LinearWeightView(
            self.gate_up_proj.weight,
            intermediate_size,
            intermediate_size,
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection_mode == "packed":
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.gate_proj.rebind(self.gate_up_proj.weight)
        self.up_proj.rebind(self.gate_up_proj.weight)
        return self


# ═══════════════════════════════════════════════════════════════
# Qwen3VLTextDecoderLayer
# ═══════════════════════════════════════════════════════════════
class Qwen3VLTextDecoderLayer(nn.Module):
    """LLM Decoder Layer: RMSNorm→Attn→+res→RMSNorm→MLP→+res."""

    def __init__(
        self,
        hidden_size: int = CANONICAL_TEXT_HIDDEN_SIZE,
        num_heads: int = CANONICAL_TEXT_NUM_HEADS,
        num_kv_heads: int = CANONICAL_TEXT_NUM_KV_HEADS,
        intermediate_size: int = CANONICAL_TEXT_INTERMEDIATE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        head_dim: int = CANONICAL_TEXT_HEAD_DIM,
        mlp_projection_mode: str = "packed",
        rms_norm_eps: float = CANONICAL_TEXT_RMS_NORM_EPS,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLTextRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )
        self.self_attn = Qwen3VLTextAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )
        self.mlp = Qwen3VLTextMLP(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            projection_mode=mlp_projection_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_decode_fused_add_rmsnorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        position_embeddings: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry residual state so both decoder adds fuse with the following norm."""

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_add_rmsnorm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                eps=float(self.input_layernorm.eps),
            )
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        hidden_states, residual = fused_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            eps=float(self.post_attention_layernorm.eps),
        )
        return self.mlp(hidden_states), residual


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

    def __init__(
        self,
        vocab_size: int = CANONICAL_TEXT_VOCAB_SIZE,
        hidden_size: int = CANONICAL_TEXT_HIDDEN_SIZE,
        num_heads: int = CANONICAL_TEXT_NUM_HEADS,
        num_kv_heads: int = CANONICAL_TEXT_NUM_KV_HEADS,
        num_layers: int = CANONICAL_TEXT_NUM_LAYERS,
        intermediate_size: int = CANONICAL_TEXT_INTERMEDIATE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        head_dim: int = CANONICAL_TEXT_HEAD_DIM,
        rope_theta: float = CANONICAL_TEXT_ROPE_THETA,
        mrope_section: list[int] | None = None,
        mlp_projection_mode: str = "packed",
        rms_norm_eps: float = CANONICAL_TEXT_RMS_NORM_EPS,
    ):
        super().__init__()
        dtype = _normalize_dtype(dtype)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    dtype,
                    head_dim,
                    mlp_projection_mode,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )
        self.fused_add_rmsnorm_enabled = False
        # M-RoPE: LLM 3D 位置编码 (head_dim=128, theta=5M, mrope_section=[24,20,20])
        self.rotary_emb = MRope(
            head_dim=head_dim,
            theta=rope_theta,
            mrope_section=(
                list(CANONICAL_MROPE_SECTION) if mrope_section is None else mrope_section
            ),
        )

    @classmethod
    def from_config(
        cls,
        config,
        dtype: torch.dtype | None = None,
        *,
        mlp_projection_mode: str = "packed",
    ):
        architecture = Qwen3VLTextArchitecture.from_config(config)
        text_config = _text_config(config)
        dtype = _normalize_dtype(
            dtype
            or _cfg_get(
                text_config,
                "torch_dtype",
                default=_cfg_get(config, "torch_dtype", default=torch.bfloat16),
            )
        )
        return cls(
            vocab_size=architecture.vocab_size,
            hidden_size=architecture.hidden_size,
            num_heads=architecture.num_heads,
            num_kv_heads=architecture.num_kv_heads,
            num_layers=architecture.num_layers,
            intermediate_size=architecture.intermediate_size,
            dtype=dtype,
            head_dim=architecture.head_dim,
            rope_theta=architecture.rope_theta,
            mrope_section=list(architecture.mrope_section),
            mlp_projection_mode=mlp_projection_mode,
            rms_norm_eps=architecture.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
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
                device = hidden_states.device
                if hidden_states.ndim == FLATTENED_TOKEN_FEATURES_RANK:
                    # engine flatten: [num_tokens, hidden]
                    position_ids = torch.arange(hidden_states.shape[0], device=device)
                else:
                    seqlen = hidden_states.shape[1]
                    # full-sequence: [batch, seqlen]
                    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
            elif (
                hidden_states.ndim == FLATTENED_TOKEN_FEATURES_RANK
                and position_ids.ndim == MROPE_POSITION_MATRIX_RANK
                and position_ids.shape[0] == MROPE_AXIS_COUNT
            ):
                # engine flatten VL: [3, num_tokens] 明确视为单个 flatten 序列，
                # 避免 num_tokens == 3 时被 MRope 误判为 [batch, seqlen]。
                position_ids = position_ids[:, None, :]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        use_fused_add_rmsnorm = (
            self.fused_add_rmsnorm_enabled
            and hidden_states.ndim == FLATTENED_TOKEN_FEATURES_RANK
            and hidden_states.is_cuda
            and not torch.compiler.is_compiling()
            and not get_context().is_prefill
            and hidden_states.shape[0] <= 4
            and attention_mask is None
            and deepstack_visual_embeds is None
        )
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            if use_fused_add_rmsnorm:
                hidden_states, residual = layer.forward_decode_fused_add_rmsnorm(
                    hidden_states,
                    residual,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )

            # DeepStack 注入: 在 layers 0, 1, 2 之后 (HF L835-L840)
            # deepstack_visual_embeds 是 list of 3, 对应 layer 0,1,2
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        if use_fused_add_rmsnorm:
            hidden_states, _ = fused_add_rmsnorm(
                hidden_states,
                residual,
                self.norm.weight,
                eps=float(self.norm.eps),
            )
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
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

    def __init__(
        self,
        config=None,
        dtype: torch.dtype | None = None,
        *,
        mlp_projection_mode: str = "packed",
        vision_encoder_microbatch_patches: int | None = None,
        vision_attention_backend: VisionAttentionBackendName | str = (
            VisionAttentionBackendName.SDPA
        ),
    ):
        super().__init__()
        architecture = (
            Qwen3VLArchitecture.canonical()
            if config is None
            else Qwen3VLArchitecture.from_config(config)
        )
        self.architecture = architecture
        text_config = _text_config(config)
        dtype = _normalize_dtype(
            dtype
            or _cfg_get(
                config,
                "torch_dtype",
                "dtype",
                default=_cfg_get(text_config, "torch_dtype", "dtype", default=torch.bfloat16),
            )
        )
        vision_config = _vision_config(config)
        self.visual = VisionEncoder(
            vision_config,
            dtype=dtype,
            attention_backend=vision_attention_backend,
        )
        self.language_model = (
            Qwen3VLTextModel.from_config(
                config,
                dtype=dtype,
                mlp_projection_mode=mlp_projection_mode,
            )
            if config is not None
            else Qwen3VLTextModel(
                dtype=dtype,
                mlp_projection_mode=mlp_projection_mode,
            )
        )
        self.image_token_id = architecture.image_token_id
        self.video_token_id = architecture.video_token_id
        if vision_encoder_microbatch_patches is not None and (
            isinstance(vision_encoder_microbatch_patches, bool)
            or not isinstance(vision_encoder_microbatch_patches, int)
            or vision_encoder_microbatch_patches <= 0
        ):
            raise ValueError("vision_encoder_microbatch_patches must be a positive integer or None")
        self.vision_encoder_microbatch_patches = vision_encoder_microbatch_patches

    @staticmethod
    def _plan_visual_microbatches(
        grid_thw: torch.Tensor,
        *,
        patch_limit: int,
    ) -> tuple[tuple[int, int, tuple[tuple[int, int, int], ...]], ...]:
        """Partition contiguous temporal segments without splitting attention."""

        rows = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
        plans: list[tuple[int, int, tuple[tuple[int, int, int], ...]]] = []
        chunk_rows: list[tuple[int, int, int]] = []
        chunk_start = 0
        chunk_patches = 0
        source_offset = 0

        def flush() -> None:
            nonlocal chunk_rows, chunk_start, chunk_patches
            if not chunk_rows:
                return
            plans.append((chunk_start, source_offset, tuple(chunk_rows)))
            chunk_rows = []
            chunk_start = source_offset
            chunk_patches = 0

        for temporal, height, width in rows:
            if temporal <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"grid_thw values must be positive, got {rows}")
            frame_patches = height * width
            if frame_patches > patch_limit:
                raise ValueError(
                    "one visual attention segment exceeds the encoder "
                    f"microbatch limit: patches={frame_patches} limit={patch_limit}"
                )
            remaining = temporal
            while remaining:
                frame_capacity = (patch_limit - chunk_patches) // frame_patches
                if frame_capacity == 0:
                    flush()
                    frame_capacity = patch_limit // frame_patches
                take = min(remaining, frame_capacity)
                patch_count = take * frame_patches
                chunk_rows.append((take, height, width))
                chunk_patches += patch_count
                source_offset += patch_count
                remaining -= take
                if chunk_patches == patch_limit:
                    flush()
        flush()
        return tuple(plans)

    def _encode_visual_payload(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """用自实现 VisionEncoder 编码 image/video patch payload。"""

        patch_limit = self.vision_encoder_microbatch_patches
        if patch_limit is None or int(pixel_values.shape[0]) <= patch_limit:
            return self.visual(pixel_values, grid_thw)

        plans = self._plan_visual_microbatches(
            grid_thw,
            patch_limit=patch_limit,
        )
        expected_patches = plans[-1][1] if plans else 0
        if expected_patches != int(pixel_values.shape[0]):
            raise ValueError(
                "visual grid/payload patch count mismatch: "
                f"grid={expected_patches} payload={int(pixel_values.shape[0])}"
            )

        main_parts: list[torch.Tensor] = []
        deepstack_parts: list[list[torch.Tensor]] | None = None
        for microbatch_index, (start, end, rows) in enumerate(plans):
            chunk_grid = torch.tensor(
                rows,
                dtype=grid_thw.dtype,
                device=grid_thw.device,
            )
            with profile_region(
                "model.vision.encoder_microbatch",
                metadata={
                    "index": microbatch_index,
                    "patches": end - start,
                    "segments": sum(row[0] for row in rows),
                },
            ):
                main, deepstack = self.visual(
                    pixel_values[start:end],
                    chunk_grid,
                )
            main_parts.append(main)
            if deepstack_parts is None:
                deepstack_parts = [[] for _ in deepstack]
            if len(deepstack) != len(deepstack_parts):
                raise RuntimeError("vision microbatches returned inconsistent DeepStack")
            for layer_parts, layer_output in zip(deepstack_parts, deepstack):
                layer_parts.append(layer_output)

        return (
            torch.cat(main_parts, dim=0),
            [torch.cat(parts, dim=0) for parts in (deepstack_parts or [])],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [batch, seqlen]
        pixel_values: [N_image_patches, 1536] or None
        image_grid_thw: [[T, H, W]] or None
        pixel_values_videos: [N_video_patches, 1536] or None
        video_grid_thw: [[T, H, W]] or None
        """
        # 1. 文本 embedding
        with profile_region("model.token_embedding"):
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        visual_masks: list[torch.Tensor] = []
        deepstack_by_modality: list[list[torch.Tensor]] = []

        # 2. Vision encoding + 特征注入 (参照 HF L1191-L1237)
        if pixel_values is not None and image_grid_thw is not None:
            with profile_region(
                "model.vision.image",
                metadata={"patch_tokens": int(pixel_values.shape[0])},
            ):
                main_vis, deepstack_vis = self._encode_visual_payload(
                    pixel_values,
                    image_grid_thw,
                )
            # main_vis: [N_vis, 4096], deepstack_vis: list of 3 × [N_vis, 4096]
            main_vis = main_vis.to(inputs_embeds.device, inputs_embeds.dtype)

            # 找到视觉 token 占位符位置 (HF L1091: input_ids == image_token_id)
            visual_pos_masks_base = input_ids == self.image_token_id
            visual_pos_masks = visual_pos_masks_base.unsqueeze(-1).expand_as(inputs_embeds)

            # 验证 token 数量匹配 (参照 HF L1097-L1099)
            n_vis_tokens = visual_pos_masks_base.sum().item()
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
                visual_pos_masks.to(inputs_embeds.device), main_vis
            )

            visual_masks.append(visual_pos_masks_base)
            deepstack_by_modality.append(deepstack_vis)

        if pixel_values_videos is not None and video_grid_thw is not None:
            with profile_region(
                "model.vision.video",
                metadata={"patch_tokens": int(pixel_values_videos.shape[0])},
            ):
                video_vis, deepstack_video = self._encode_visual_payload(
                    pixel_values_videos,
                    video_grid_thw,
                )
            video_vis = video_vis.to(inputs_embeds.device, inputs_embeds.dtype)

            video_pos_masks_base = input_ids == self.video_token_id
            video_pos_masks = video_pos_masks_base.unsqueeze(-1).expand_as(inputs_embeds)
            n_video_tokens = video_pos_masks_base.sum().item()
            expected_elements = n_video_tokens * inputs_embeds.shape[-1]
            actual_elements = video_vis.numel()
            if expected_elements != actual_elements:
                raise ValueError(
                    f"视频 token 数量不匹配: input 中有 {n_video_tokens} 个 "
                    f"<|video_pad|> ({n_video_tokens}×{inputs_embeds.shape[-1]}"
                    f"={expected_elements} elements), "
                    f"但 Vision Encoder 输出 {video_vis.shape[0]} 个 token "
                    f"({actual_elements} elements)"
                )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_pos_masks.to(inputs_embeds.device), video_vis
            )

            visual_masks.append(video_pos_masks_base)
            deepstack_by_modality.append(deepstack_video)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if visual_masks:
            visual_pos_masks = visual_masks[0]
            for mask in visual_masks[1:]:
                visual_pos_masks = visual_pos_masks | mask
            if len(deepstack_by_modality) == 1:
                deepstack_visual_embeds = deepstack_by_modality[0]
            else:
                deepstack_visual_embeds = []
                for layer_embeds in zip(*deepstack_by_modality):
                    joint = layer_embeds[0].new_zeros(
                        visual_pos_masks.sum(),
                        layer_embeds[0].shape[-1],
                    )
                    for mask, embeds in zip(visual_masks, layer_embeds):
                        modality_mask = mask[visual_pos_masks]
                        joint[modality_mask, :] = embeds.to(joint.device, joint.dtype)
                    deepstack_visual_embeds.append(joint)

        # 3. LLM forward (DeepStack 注入在 Qwen3VLTextModel.forward 内部)
        with profile_region("model.language_model"):
            hidden_states = self.language_model(
                inputs_embeds=inputs_embeds,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Qwen3VLForCausalLM — 最外层
# ═══════════════════════════════════════════════════════════════
class Qwen3VLForCausalLM(nn.Module):
    """Qwen3-VL-8B 完整模型."""

    def __init__(
        self,
        config=None,
        dtype: torch.dtype | None = None,
        *,
        mlp_projection_mode: str = "packed",
        vision_encoder_microbatch_patches: int | None = None,
        vision_attention_backend: VisionAttentionBackendName | str = (
            VisionAttentionBackendName.SDPA
        ),
    ):
        # Backward compatibility: Qwen3VLForCausalLM(torch.bfloat16)
        if isinstance(config, torch.dtype):
            dtype = config
            config = None
        super().__init__()
        text_config = _text_config(config)
        dtype = _normalize_dtype(
            dtype
            or _cfg_get(
                config,
                "torch_dtype",
                "dtype",
                default=_cfg_get(text_config, "torch_dtype", "dtype", default=torch.bfloat16),
            )
        )
        self.model = Qwen3VLModel(
            config,
            dtype=dtype,
            mlp_projection_mode=mlp_projection_mode,
            vision_encoder_microbatch_patches=vision_encoder_microbatch_patches,
            vision_attention_backend=vision_attention_backend,
        )
        architecture = self.model.architecture
        hidden_size = architecture.text.hidden_size
        vocab_size = architecture.text.vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
        # HF uses the loaded model dtype for lm_head.  Keep fp32 as an explicit
        # historical-reproduction mode; converting the full weight every decode
        # step is both slower and less numerically faithful to HF BF16 logits.
        self.logits_precision = "model"
        if architecture.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == FLATTENED_TOKEN_FEATURES_RANK:
            context = get_context()
            if context.is_prefill and context.cu_seqlens_q is not None:
                hidden_states = hidden_states[context.cu_seqlens_q[1:] - 1].contiguous()
        if (
            self.logits_precision == "fp32"
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        ):
            return F.linear(hidden_states.float(), self.lm_head.weight.float())
        logits = self.lm_head(hidden_states)
        if (
            self.logits_precision == "selective_fp32"
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        ):
            candidate_ids = logits.topk(
                k=SELECTIVE_FP32_LOGITS_TOP_K,
                dim=-1,
                sorted=False,
            ).indices
            candidate_weights = F.embedding(candidate_ids, self.lm_head.weight)
            candidate_logits = torch.bmm(
                candidate_weights.float(),
                hidden_states.float().unsqueeze(-1),
            ).squeeze(-1)
            logits = logits.float()
            logits.scatter_(-1, candidate_ids, candidate_logits)
        return logits
