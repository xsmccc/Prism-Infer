"""Test ViTAttention with RoPE: 验证 RoPE 旋转逻辑正确。

用 HF 生成的 cos/sin 作为 ground truth，验证我们的 apply_rotary_emb 输出。
"""
import os

import torch

import importlib.util
spec = importlib.util.spec_from_file_location(
    "vision_encoder", os.path.join(os.path.dirname(__file__),
    "../prism_infer/vision/vision_encoder.py"))
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)
ViTAttention = ve.ViTAttention

from conftest import get_model_path, hf_qwen3_vl_visual, require_transformers
from PIL import Image

THRESHOLD = 1e-5


def test_attention_with_rope():
    """完整对比: 用 HF 的 cos/sin，验证我们的 attention 输出与 HF 一致"""
    transformers = require_transformers()
    cache = get_model_path()
    hf = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        cache, dtype=torch.bfloat16, device_map='cpu',
        trust_remote_code=True, local_files_only=True)
    hf_visual = hf_qwen3_vl_visual(hf)
    hf_attn = hf_visual.blocks[0].attn

    our = ViTAttention(1152, 16, torch.bfloat16)
    our_sd = our.state_dict()
    hf_sd = hf_attn.state_dict()
    for key in ['qkv.weight', 'qkv.bias', 'proj.weight', 'proj.bias']:
        our_sd[key].copy_(hf_sd[key])

    # 生成真实图片的 RoPE cos/sin
    img = Image.new('RGB', (448, 448), color=(100, 150, 200))
    p = transformers.AutoProcessor.from_pretrained(
        cache, trust_remote_code=True, local_files_only=True)
    pv = p(text=p.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image', 'image': img}]}],
        tokenize=False, add_generation_prompt=True),
        images=[img], return_tensors='pt')['pixel_values']

    grid_thw = torch.tensor([[1, 28, 28]])

    # Hook 获取 HF attention 的完整输出
    hf_out_dict = {}
    def hf_hook(module, args, kwargs, output):
        hf_out_dict['args'] = args
        hf_out_dict['kwargs'] = kwargs

    h = hf_attn.register_forward_hook(hf_hook, with_kwargs=True)
    with torch.no_grad():
        hf_visual(pv, grid_thw=grid_thw)
    h.remove()

    # 获取传给 HF attention 的实际 cos/sin
    hf_attn_in = hf_out_dict['args'][0]           # the input tensor
    actual_cos = hf_out_dict['kwargs'].get('position_embeddings', (None,None))[0]
    actual_sin = hf_out_dict['kwargs'].get('position_embeddings', (None,None))[1]
    print(f"HF attention input shape: {list(hf_attn_in.shape)}")
    print(f"HF cos shape: {list(actual_cos.shape)}")

    # 用 HF 的 cos/sin 跑我们的 attention
    with torch.no_grad():
        our_out = our(hf_attn_in, cos=actual_cos, sin=actual_sin)
        # HF 完整 attention 输出
        hf_attn_out = hf_visual.blocks[0].attn(
            hf_attn_in,
            cu_seqlens=hf_out_dict['kwargs']['cu_seqlens'],
            position_embeddings=(actual_cos, actual_sin),
        )

    diff = (our_out.float() - hf_attn_out.float()).abs().max().item()
    print(f"\n  ViT Attention (with RoPE) max diff: {diff:.10f}")
    if diff < THRESHOLD:
        print("  PASS — RoPE application correct")
    else:
        print(f"  FAIL — diff {diff:.2e} > threshold {THRESHOLD:.0e}")


if __name__ == '__main__':
    print("=== ViTAttention + RoPE Test ===")
    test_attention_with_rope()
