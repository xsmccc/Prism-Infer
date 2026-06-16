"""Test PatchEmbed: Conv3d patch embedding 正确性验证。

Ref: prism_infer/vision/vision_encoder.py
Ground truth: HF Qwen3VLForConditionalGeneration.model.visual.patch_embed
"""
import os, sys
import torch
import importlib.util

os.environ['HF_HUB_OFFLINE'] = '1'

# 直接导入避免触发 flash_attn
spec = importlib.util.spec_from_file_location(
    "vision_encoder", os.path.join(os.path.dirname(__file__),
    "../prism_infer/vision/vision_encoder.py"))
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
PatchEmbed = ve.PatchEmbed

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

CACHE = '/home/xsmccc/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
THRESHOLD = 1e-5


def test_patch_embed_shape():
    """验证输出 shape 正确"""
    our = PatchEmbed(3, 1152, 2, 16, torch.bfloat16)
    pv = torch.randn(784, 1536, dtype=torch.bfloat16)
    with torch.no_grad():
        out = our(pv)
    assert out.shape == (784, 1152), f"Expected (784, 1152), got {out.shape}"
    print("  shape: PASS")


def test_patch_embed_accuracy():
    """验证输出值与 HF 一致"""
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        CACHE, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_pe = hf.visual.patch_embed

    our = PatchEmbed(3, 1152, 2, 16, torch.bfloat16)
    our.load_state_dict(hf_pe.state_dict())

    # 用真实图片输入
    img = Image.new('RGB', (448, 448), color=(100, 150, 200))
    p = AutoProcessor.from_pretrained(CACHE, trust_remote_code=True, local_files_only=True)
    pv = p(text=p.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image', 'image': img}]}],
        tokenize=False, add_generation_prompt=True),
        images=[img], return_tensors='pt')['pixel_values']

    with torch.no_grad():
        hf_out = hf_pe(pv).float()
        our_out = our(pv).float()

    max_diff = (our_out - hf_out).abs().max().item()
    mean_hf = hf_out.mean().item()
    mean_our = our_out.mean().item()

    print(f"  max diff: {max_diff:.10f}")
    print(f"  mean (HF/Ours): {mean_hf:.6f} / {mean_our:.6f}")
    assert max_diff < THRESHOLD, f"max diff {max_diff:.2e} exceeds threshold {THRESHOLD:.0e}"
    print("  accuracy: PASS")


if __name__ == '__main__':
    print("=== PatchEmbed Tests ===")
    test_patch_embed_shape()
    test_patch_embed_accuracy()
    print("=== All PASS ===")
