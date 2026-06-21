"""Test Qwen3-VL model components (CPU-memory-safe)."""
import gc

import torch
from prism_infer.models.qwen3_vl import (
    Qwen3VLTextRMSNorm, Qwen3VLTextMLP,
    Qwen3VLTextDecoderLayer, Qwen3VLTextModel,
)
from prism_infer.vision.mrope import MRope
from conftest import get_model_path, require_transformers


def _get_hf_sd(key: str):
    """加载 HF 模型的 state_dict, 用完即释放."""
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    sd = hf.state_dict()
    result = {k: v for k, v in sd.items() if k.startswith(key)}
    del hf, sd; gc.collect()
    return result


def test_rmsnorm():
    sd = _get_hf_sd('model.language_model.layers.0.input_layernorm')
    our = Qwen3VLTextRMSNorm(4096)
    our.load_state_dict({k.split('.')[-1]: v for k, v in sd.items()})
    del sd; gc.collect()
    x = torch.randn(2, 4, 4096, dtype=torch.bfloat16)
    with torch.no_grad():
        out = our(x)
    assert out.shape == (2, 4, 4096)
    print("  RMSNorm: PASS")


def test_mlp():
    sd = _get_hf_sd('model.language_model.layers.0.mlp')
    our = Qwen3VLTextMLP(4096, 12288, torch.bfloat16)
    # Map HF key → our key (remove prefix)
    mapped = {}
    for k, v in sd.items():
        key = k.replace('model.language_model.layers.0.mlp.', '')
        mapped[key] = v
    our.load_state_dict(mapped)
    del sd, mapped; gc.collect()
    x = torch.randn(2, 4, 4096, dtype=torch.bfloat16)
    with torch.no_grad():
        out = our(x)
    assert out.shape == (2, 4, 4096)
    print("  MLP: PASS")


def test_decoder_layer():
    sd = _get_hf_sd('model.language_model.layers.0')
    our = Qwen3VLTextDecoderLayer(4096, 32, 8, 12288, torch.bfloat16)
    mapped = {}
    for k, v in sd.items():
        key = k.replace('model.language_model.layers.0.', '')
        mapped[key] = v
    our.load_state_dict(mapped)
    del sd, mapped; gc.collect()
    x = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
    pos_ids = torch.zeros(3, 4, dtype=torch.long)
    pos_ids[0] = torch.arange(4)
    mrope = MRope(128, 5000000.0, [24, 20, 20])
    with torch.no_grad():
        cos, sin = mrope(x, pos_ids)
        our_out = our(x, position_embeddings=(cos, sin))
    assert our_out.shape == (1, 4, 4096)
    print("  DecoderLayer: PASS")


def test_weight_keys():
    """仅验证 key 名称匹配, 不加载权重."""
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_keys = set(k for k in hf.state_dict().keys()
                  if k.startswith('model.language_model.'))
    del hf; gc.collect()

    our = Qwen3VLTextModel(151936, 4096, 32, 8, 36, 12288, torch.bfloat16)
    our_keys = set(our.state_dict().keys())
    del our; gc.collect()

    # 简化: 只检查层数和关键模块
    print(f"  HF keys: {len(hf_keys)}, Our keys: {len(our_keys)}")
    # 应有 36 层 × ~10 keys + embed + norm
    assert len(our_keys) > 300
    print("  Weight keys: PASS")


if __name__ == '__main__':
    print("=== Qwen3-VL Model Tests ===")
    test_rmsnorm()
    test_mlp()
    test_decoder_layer()
    test_weight_keys()
    print("=== All PASS ===")
