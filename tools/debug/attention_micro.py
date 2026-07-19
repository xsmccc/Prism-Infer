"""Micro debug for Qwen3-VL text attention.

手动运行脚本，用同一份 layer0 输入和 position embeddings 比较 HF 与 Prism-Infer
attention 内部节点，定位 full logits 的第一处误差来源。

内存策略: 先运行 HF layer0 并把激活/局部权重拷回 CPU，释放 HF GPU 模型后，
再构建 Prism-Infer 模型，只加载 layer0 相关权重。
"""
import gc

import torch
import torch.nn.functional as F

from _common import get_model_path
from prism_infer.vision.mrope import apply_mrope


MODEL_PATH = get_model_path()
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEQ_LEN = 64
VOCAB_SIZE = 151936


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """与 Transformers integrations/sdpa_attention.py repeat_kv 等价。"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def diff(name: str, hf: torch.Tensor, ours: torch.Tensor) -> None:
    hf_f = hf.detach().float().cpu()
    our_f = ours.detach().float().cpu()
    delta = (hf_f - our_f).abs()
    print(
        f"{name:<20} shape={list(hf.shape)} "
        f"max={delta.max().item():.6e} mean={delta.mean().item():.6e} "
        f"hf_std={hf_f.std().item():.6e} our_std={our_f.std().item():.6e}"
    )


def collect_hf(input_ids: torch.Tensor):
    """运行 HF layer0 attention 相关节点，并返回 CPU 激活和局部权重。"""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    print("Loading HF on GPU...")
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=DTYPE, trust_remote_code=True, local_files_only=True
    ).cuda().eval()

    hf_lm = hf.model.language_model
    hf_layer = hf_lm.layers[0]
    hf_attn = hf_layer.self_attn

    with torch.no_grad():
        hidden = hf_lm.embed_tokens(input_ids)
        pos = torch.arange(SEQ_LEN, device=DEVICE).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = hf_lm.rotary_emb(hidden, pos)

        norm = hf_layer.input_layernorm(hidden)
        input_shape = norm.shape[:-1]
        hidden_shape = (*input_shape, -1, hf_attn.head_dim)

        q = hf_attn.q_norm(hf_attn.q_proj(norm).view(hidden_shape)).transpose(1, 2)
        k = hf_attn.k_norm(hf_attn.k_proj(norm).view(hidden_shape)).transpose(1, 2)
        v = hf_attn.v_proj(norm).view(hidden_shape).transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        sdpa_gqa = F.scaled_dot_product_attention(
            q_rope, k_rope, v,
            attn_mask=None,
            dropout_p=0.0,
            scale=hf_attn.scaling,
            is_causal=True,
            enable_gqa=True,
        )
        sdpa_repeat = F.scaled_dot_product_attention(
            q_rope,
            repeat_kv(k_rope, hf_attn.num_key_value_groups),
            repeat_kv(v, hf_attn.num_key_value_groups),
            attn_mask=None,
            dropout_p=0.0,
            scale=hf_attn.scaling,
            is_causal=True,
        )
        flat = sdpa_gqa.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        attn_out = hf_attn.o_proj(flat)
        layer0_out = hf_layer(hidden, position_embeddings=(cos, sin), attention_mask=None)

        values = {
            "hidden": hidden.detach().cpu(),
            "cos": cos.detach().cpu(),
            "sin": sin.detach().cpu(),
            "norm": norm.detach().cpu(),
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "q_rope": q_rope.detach().cpu(),
            "k_rope": k_rope.detach().cpu(),
            "sdpa_gqa": sdpa_gqa.detach().cpu(),
            "sdpa_repeat": sdpa_repeat.detach().cpu(),
            "flat": flat.detach().cpu(),
            "attn_out": attn_out.detach().cpu(),
            "layer0_out": layer0_out.detach().cpu(),
        }
        config = hf.config
        state = {
            key: value.detach().cpu()
            for key, value in hf.state_dict().items()
            if (
                key.startswith("model.language_model.embed_tokens.")
                or key.startswith("model.language_model.layers.0.")
            )
        }

    del hf, hf_lm, hf_layer, hf_attn
    gc.collect()
    torch.cuda.empty_cache()
    return config, state, values


def run_our(input_ids: torch.Tensor, config, state, hf_values: dict[str, torch.Tensor]) -> None:
    """构建 Prism-Infer 模型并比较 layer0 attention 内部节点。"""
    from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

    print("Building Prism-Infer on GPU...")
    default_device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        our = Qwen3VLForCausalLM(config=config, dtype=DTYPE).eval()
    finally:
        torch.set_default_device(default_device)

    our_sd = our.state_dict()
    loaded = 0
    for key, value in state.items():
        if key in our_sd:
            our_sd[key].copy_(value.to(DEVICE), non_blocking=True)
            loaded += 1
    print(f"Loaded local weights: {loaded}/{len(state)}")

    our_lm = our.model.language_model
    our_layer = our_lm.layers[0]
    our_attn = our_layer.self_attn

    with torch.no_grad():
        hidden = our_lm.embed_tokens(input_ids)
        diff("embed", hf_values["hidden"], hidden)

        pos = torch.arange(SEQ_LEN, device=DEVICE).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = our_lm.rotary_emb(hidden, pos)
        diff("cos", hf_values["cos"], cos)
        diff("sin", hf_values["sin"], sin)

        norm = our_layer.input_layernorm(hidden)
        diff("input_norm", hf_values["norm"], norm)

        bsz, q_len, _ = norm.shape
        q = our_attn.q_proj(norm).view(bsz, q_len, our_attn.num_heads, our_attn.head_dim)
        k = our_attn.k_proj(norm).view(bsz, q_len, our_attn.num_kv_heads, our_attn.head_dim)
        v = our_attn.v_proj(norm).view(bsz, q_len, our_attn.num_kv_heads, our_attn.head_dim).transpose(1, 2)
        q = our_attn.q_norm(q).transpose(1, 2)
        k = our_attn.k_norm(k).transpose(1, 2)
        diff("q_norm", hf_values["q"], q)
        diff("k_norm", hf_values["k"], k)
        diff("v", hf_values["v"], v)

        q_rope, k_rope = apply_mrope(q, k, cos, sin)
        diff("q_rope", hf_values["q_rope"], q_rope)
        diff("k_rope", hf_values["k_rope"], k_rope)

        sdpa_gqa = F.scaled_dot_product_attention(
            q_rope, k_rope, v,
            attn_mask=None,
            dropout_p=0.0,
            scale=our_attn.scale,
            is_causal=True,
            enable_gqa=True,
        )
        diff("sdpa_gqa", hf_values["sdpa_gqa"], sdpa_gqa)

        sdpa_repeat = F.scaled_dot_product_attention(
            q_rope,
            repeat_kv(k_rope, our_attn.num_key_value_groups),
            repeat_kv(v, our_attn.num_key_value_groups),
            attn_mask=None,
            dropout_p=0.0,
            scale=our_attn.scale,
            is_causal=True,
        )
        diff("sdpa_repeat", hf_values["sdpa_repeat"], sdpa_repeat)
        diff("hf_gqa_vs_repeat", hf_values["sdpa_gqa"], hf_values["sdpa_repeat"])
        diff("our_gqa_vs_repeat", sdpa_gqa, sdpa_repeat)

        flat = sdpa_gqa.transpose(1, 2).contiguous().reshape(bsz, q_len, -1).contiguous()
        diff("attn_flat", hf_values["flat"], flat)

        attn_out = our_attn.o_proj(flat)
        diff("attn_out", hf_values["attn_out"], attn_out)

        layer0_out = our_layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
        diff("layer0_out", hf_values["layer0_out"], layer0_out)

    del our
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
    config, state, hf_values = collect_hf(input_ids)
    run_our(input_ids, config, state, hf_values)


if __name__ == "__main__":
    main()
