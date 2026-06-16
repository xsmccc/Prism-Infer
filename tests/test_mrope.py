"""Test MRope: cos/sin 生成 + apply_mrope 应用."""
import os, sys, torch
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/home/xsmccc/nano-vllm')
from prism_infer.vision.mrope import MRope, apply_mrope, rotate_half

from transformers import Qwen3VLForConditionalGeneration
CACHE = '/home/xsmccc/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def test_cos_sin_generation():
    """cos/sin 与 HF 一致"""
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        CACHE, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_rope = hf.model.language_model.rotary_emb
    our = MRope(128, 5000000.0, [24, 20, 20])

    pos = torch.zeros(3, 6, dtype=torch.long)
    pos[0, 4:] = torch.tensor([0, 1])
    pos[1, :4] = torch.tensor([5, 6, 7, 8])
    pos[2, :4] = torch.tensor([10, 11, 12, 13])
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
    """apply_mrope 对每段 head_dim 正确应用对应轴的 RoPE"""
    mrope = MRope(128, 5000000.0, [24, 20, 20])
    pos_ids = torch.zeros(3, 1, dtype=torch.long)
    pos_ids[0] = 5; pos_ids[1] = 10; pos_ids[2] = 15
    hs = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
    with torch.no_grad():
        cos, sin = mrope(hs, pos_ids)

    q = torch.randn(1, 32, 1, 128, dtype=torch.float32)
    k = torch.randn(1, 8, 1, 128, dtype=torch.float32)
    qr, kr = apply_mrope(q, k, cos, sin)

    for start, end, axis in [(0,24,0),(24,44,1),(44,64,2),(64,88,0),(88,108,1),(108,128,2)]:
        c = cos[axis, :, start:end]
        s = sin[axis, :, start:end]
        qs = q[:, :, :, start:end]
        expected = qs * c[None,None,:,:] + rotate_half(qs) * s[None,None,:,:]
        d = (expected - qr[:, :, :, start:end]).abs().max().item()
        assert d < 1e-6, f"[{start}:{end}] diff {d:.2e}"
    print("  apply_mrope: PASS")


if __name__ == '__main__':
    print("=== MRope Tests ===")
    test_cos_sin_generation()
    test_apply_mrope()
    print("=== All PASS ===")
