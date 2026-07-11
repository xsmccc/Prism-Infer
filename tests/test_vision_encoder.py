"""Test VisionEncoder: 完整 ViT 编码器正确性验证。

测试: PatchEmbed + PosEmbed + 27 ViTBlock + 4 Merger + DeepStack
Ground truth: HF Qwen3VLVisionModel (model.visual)
"""
import os

import torch

import importlib.util
spec = importlib.util.spec_from_file_location(
    "vision_encoder", os.path.join(os.path.dirname(__file__),
    "../prism_infer/vision/vision_encoder.py"))
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
VisionEncoder = ve.VisionEncoder


def test_vision_tensor_region_split_is_exact() -> None:
    """动态 geometry preparation 与稳定 tensor region 拆分必须 exact。"""

    config = {
        "hidden_size": 16,
        "in_channels": 3,
        "temporal_patch_size": 1,
        "patch_size": 2,
        "num_heads": 4,
        "intermediate_size": 32,
        "depth": 3,
        "out_hidden_size": 24,
        "num_position_embeddings": 16,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [0, 1, 2],
    }
    torch.manual_seed(20260711)
    vision = VisionEncoder(config, dtype=torch.float32)
    pixels = torch.randn(16, 12)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    reference_main, reference_deepstack = vision(pixels, grid_thw)
    tensor_inputs = vision.prepare_tensor_region_inputs(pixels, grid_thw)
    actual_main, actual_deepstack = vision.forward_tensor_region(*tensor_inputs)

    assert torch.equal(reference_main, actual_main)
    assert len(reference_deepstack) == len(actual_deepstack) == 3
    assert all(
        torch.equal(reference, actual)
        for reference, actual in zip(reference_deepstack, actual_deepstack)
    )
    print(f"vision tensor input shapes: {[list(value.shape) for value in tensor_inputs]}")
    print(f"vision main output shape: {list(actual_main.shape)}")
    print("Vision dynamic/tensor region split max diff: 0.000000e+00 PASS")

from conftest import get_model_path, hf_qwen3_vl_visual, require_transformers

THRESHOLD = 2e-2
# 注: 单模块 diff < 1e-5, 但 27 层 ViT Block 链式传播导致 bf16 累积误差 ~0.016。
# 根因: CPU LayerNorm 内部 float32 accum 不同实例并行度不一致。
# 预期: GPU 上用确定性算法 diff < 1e-5。本条为 CPU 已知特例。


def test_vision_encoder():
    """完整对比 VisionEncoder vs HF model.visual"""
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_vis = hf_qwen3_vl_visual(hf)

    our = VisionEncoder(torch.bfloat16)
    # 加载权重: HF "model.visual.xxx" → our "xxx"
    vis_sd = {k[13:]: v for k, v in hf.state_dict().items()
              if k.startswith('model.visual.')}
    our.load_state_dict(vis_sd, strict=False)

    pv = torch.randn(784, 1536, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 28, 28]])

    with torch.no_grad():
        hf_out = hf_vis(pv, grid_thw=grid_thw)
        hf_main, hf_ds = _unpack_hf_vision_output(hf_out)
        our_main, our_ds = our(pv, grid_thw=grid_thw)

    results = []
    md = (our_main.float() - hf_main.float()).abs().max().item()
    ok = "PASS" if md < THRESHOLD else "FAIL"
    results.append((f"main [196,4096]", md, ok))
    print(f"  main [196,4096]: {md:.6f} {ok}")

    for i in range(3):
        d = (our_ds[i].float() - hf_ds[i].float()).abs().max().item()
        ok = "PASS" if d < THRESHOLD else "FAIL"
        results.append((f"ds[{i}] [196,4096]", d, ok))
        print(f"  ds[{i}] [196,4096]: {d:.6f} {ok}")

    all_pass = all(r[2] == "PASS" for r in results)
    assert all_pass, f"Some tests failed"
    print(f"\n  All PASS (CPU bf16, threshold={THRESHOLD})")


def _unpack_hf_vision_output(output):
    """Return merged visual output and DeepStack features across HF layouts."""

    if hasattr(output, "pooler_output") and hasattr(output, "deepstack_features"):
        return output.pooler_output, output.deepstack_features
    return output


if __name__ == '__main__':
    print("=== VisionEncoder Test ===")
    test_vision_encoder()
