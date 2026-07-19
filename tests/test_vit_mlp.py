"""Test ViTMLP: Vision Transformer FFN 正确性验证。

Ref: prism_infer/vision/vision_encoder.py
Ground truth: HF Qwen3-VL visual.blocks[0].mlp
"""

import os

import pytest
import torch

import importlib.util

spec = importlib.util.spec_from_file_location(
    "vision_encoder",
    os.path.join(os.path.dirname(__file__), "../prism_infer/vision/vision_encoder.py"),
)
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
ViTMLP = ve.ViTMLP

from conftest import get_model_path, hf_qwen3_vl_visual, require_transformers

THRESHOLD = 1e-5


def test_mlp_shape():
    our = ViTMLP(1152, 4304, torch.bfloat16)
    x = torch.randn(4, 1152, dtype=torch.bfloat16)
    with torch.no_grad():
        out = our(x)
    assert out.shape == (4, 1152), f"Expected (4, 1152), got {out.shape}"
    print("  shape: PASS")


@pytest.mark.model
@pytest.mark.integration
def test_mlp_accuracy():
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True, local_files_only=True
    )
    hf_mlp = hf_qwen3_vl_visual(hf).blocks[0].mlp

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


if __name__ == "__main__":
    print("=== ViTMLP Tests ===")
    test_mlp_shape()
    test_mlp_accuracy()
    print("=== All PASS ===")
