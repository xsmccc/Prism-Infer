"""Test MRope: cos/sin 生成 + apply_mrope 应用."""
import torch
from prism_infer.vision.mrope import MRope, apply_mrope, rotate_half

from conftest import get_model_path, require_transformers


def test_cos_sin_generation():
    """cos/sin 与 HF 一致"""
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_rope = hf.model.language_model.rotary_emb
    our = MRope(128, 5000000.0, [24, 20, 20])

    pos = torch.zeros(3, 1, 6, dtype=torch.long)
    pos[0, 0, 4:] = torch.tensor([0, 1])
    pos[1, 0, :4] = torch.tensor([5, 6, 7, 8])
    pos[2, 0, :4] = torch.tensor([10, 11, 12, 13])
    hs = torch.randn(1, 6, 4096, dtype=torch.bfloat16)

    with torch.no_grad():
        hf_c, hf_s = hf_rope(hs, pos)
        our_c, our_s = our(hs, pos)

    cd = (our_c.float() - hf_c.float()).abs().max().item()
    sd = (our_s.float() - hf_s.float()).abs().max().item()
    print(f"  cos diff: {cd:.10f}")
    print(f"  sin diff: {sd:.10f}")
    assert cd < 1e-5 and sd < 1e-5
    print("  cos_sin: PASS")


def test_apply_mrope():
    """apply_mrope applies the full-head RoPE formula to q/k."""
    mrope = MRope(128, 5000000.0, [24, 20, 20])
    pos_ids = torch.zeros(3, 1, 1, dtype=torch.long)
    pos_ids[0, 0] = 5; pos_ids[1, 0] = 10; pos_ids[2, 0] = 15
    hs = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
    with torch.no_grad():
        cos, sin = mrope(hs, pos_ids)

    q = torch.randn(1, 32, 1, 128, dtype=torch.float32)
    k = torch.randn(1, 8, 1, 128, dtype=torch.float32)
    qr, kr = apply_mrope(q, k, cos, sin)

    c = cos.float().unsqueeze(1)
    s = sin.float().unsqueeze(1)
    expected_q = q * c + rotate_half(q) * s
    expected_k = k * c + rotate_half(k) * s
    q_diff = (expected_q - qr).abs().max().item()
    k_diff = (expected_k - kr).abs().max().item()
    assert q_diff < 1e-6, f"q diff {q_diff:.2e}"
    assert k_diff < 1e-6, f"k diff {k_diff:.2e}"
    print("  apply_mrope: PASS")


if __name__ == '__main__':
    print("=== MRope Tests ===")
    test_cos_sin_generation()
    test_apply_mrope()
    print("=== All PASS ===")
