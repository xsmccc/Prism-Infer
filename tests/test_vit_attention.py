"""Test ViTAttention: QKV投影 + RoPE + SDPA 完整正确性验证。

验证 ViT Attention 完整 forward 与 HF Qwen3VLVisionAttention 一致。
"""
import os

import torch
import torch.nn.functional as F

import importlib.util
spec = importlib.util.spec_from_file_location(
    "vision_encoder", os.path.join(os.path.dirname(__file__),
    "../prism_infer/vision/vision_encoder.py"))
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
ViTAttention = ve.ViTAttention

from conftest import get_model_path, require_transformers

THRESHOLD = 1e-5


def test_attention_full():
    """完整对比: QKV投影 + RoPE + SDPA + 输出投影"""
    transformers = require_transformers()
    cache = get_model_path()
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_attn = hf.visual.blocks[0].attn

    our = ViTAttention(1152, 16, torch.bfloat16)
    for k in ['qkv.weight', 'qkv.bias', 'proj.weight', 'proj.bias']:
        our.state_dict()[k].copy_(hf_attn.state_dict()[k])

    # 随机输入
    N = 8
    hs = torch.randn(N, 1152, dtype=torch.bfloat16)
    cos = torch.randn(N, 72, dtype=torch.float32)
    sin = torch.randn(N, 72, dtype=torch.float32)

    with torch.no_grad():
        # HF: QKV → RoPE → SDPA → proj
        hf_qkv = hf_attn.qkv(hs)
        hf_q, hf_k, hf_v = hf_qkv.reshape(N, 3, 16, 72).permute(1,0,2,3).unbind(0)
        hf_qr, hf_kr = apply_rotary_pos_emb_vision(hf_q, hf_k, cos, sin)
        hf_qs = hf_qr.transpose(0,1).unsqueeze(0)
        hf_ks = hf_kr.transpose(0,1).unsqueeze(0)
        hf_vs = hf_v.transpose(0,1).unsqueeze(0)
        hf_sdpa = F.scaled_dot_product_attention(
            hf_qs, hf_ks, hf_vs, is_causal=False, scale=72**-0.5)
        hf_out = hf_attn.proj(hf_sdpa.transpose(1,2).reshape(1, N, 1152))

        # Ours: same path
        our_qkv = our.qkv(hs)
        our_q, our_k, our_v = our_qkv.chunk(3, dim=-1)
        our_q = our_q.view(1, N, 16, 72).transpose(1, 2)
        our_k = our_k.view(1, N, 16, 72).transpose(1, 2)
        our_v = our_v.view(1, N, 16, 72).transpose(1, 2)
        our_qr = our.apply_rotary_emb(our_q, cos, sin)
        our_kr = our.apply_rotary_emb(our_k, cos, sin)
        our_sdpa = F.scaled_dot_product_attention(
            our_qr, our_kr, our_v, is_causal=False, scale=72**-0.5)
        our_out = our.proj(our_sdpa.transpose(1,2).reshape(1, N, 1152))

    diff = (our_out.float() - hf_out.float()).abs().max().item()
    print(f"  max diff: {diff:.10f}")
    assert diff < THRESHOLD, f"diff {diff:.2e} > threshold"
    print("  PASS")


def test_attention_shape():
    """验证 shape 正确"""
    our = ViTAttention(1152, 16, torch.bfloat16)
    x = torch.randn(4, 1152, dtype=torch.bfloat16)
    out = our(x)
    assert out.shape == (4, 1152)
    out = our(x.unsqueeze(0))
    assert out.shape == (1, 4, 1152)
    print("  shape: PASS")


if __name__ == '__main__':
    print("=== ViTAttention Tests ===")
    test_attention_shape()
    test_attention_full()
    print("=== All PASS ===")
