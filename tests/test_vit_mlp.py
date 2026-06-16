"""Test ViTMLP: Vision Transformer FFN 正确性验证。

Ref: prism_infer/vision/vision_encoder.py
Ground truth: HF Qwen3-VL visual.blocks[0].mlp
"""
import os, torch

os.environ['HF_HUB_OFFLINE'] = '1'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vision_encoder", os.path.join(os.path.dirname(__file__),
    "../prism_infer/vision/vision_encoder.py"))
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
ViTMLP = ve.ViTMLP

from transformers import Qwen3VLForConditionalGeneration

CACHE = '/home/xsmccc/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
THRESHOLD = 1e-5


def test_mlp_shape():
    our = ViTMLP(1152, 4304, torch.bfloat16)
    x = torch.randn(4, 1152, dtype=torch.bfloat16)
    with torch.no_grad():
        out = our(x)
    assert out.shape == (4, 1152), f"Expected (4, 1152), got {out.shape}"
    print("  shape: PASS")


def test_mlp_accuracy():
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        CACHE, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_mlp = hf.visual.blocks[0].mlp

    our = ViTMLP(1152, 4304, torch.bfloat16)
    our.load_state_dict(hf_mlp.state_dict())

    x = torch.randn(16, 1152, dtype=torch.bfloat16)
    with torch.no_grad():
        hf_out = hf_mlp(x).float()
        our_out = our(x).float()

    max_diff = (our_out - hf_out).abs().max().item()
    print(f"  max diff: {max_diff:.10f}")
    print(f"  mean (HF/Ours): {hf_out.mean().item():.6f} / {our_out.mean().item():.6f}")
    assert max_diff < THRESHOLD, f"max diff {max_diff:.2e} exceeds threshold {THRESHOLD:.0e}"
    print("  accuracy: PASS")


if __name__ == '__main__':
    print("=== ViTMLP Tests ===")
    test_mlp_shape()
    test_mlp_accuracy()
    print("=== All PASS ===")
