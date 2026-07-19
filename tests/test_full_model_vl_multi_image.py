"""多图 full-model logits 验证。

对比 HF Qwen3VLForConditionalGeneration 与 Prism-Infer Qwen3VLForCausalLM
在同一多图输入上的最后 token logits。两个模型分开加载/释放，避免同时占用 GPU。
"""

import gc
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, "/data/Prism-Infer")
from conftest import get_model_path, require_transformers
from prism_infer.engine.vl_inputs import prepare_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config


pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.slow,
]


MODEL_PATH = get_model_path()
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def _demo_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (448, 448), color=(100, 150, 200)),
        Image.new("RGB", (448, 448), color=(200, 120, 80)),
    ]


def _prepare_inputs():
    transformers = require_transformers()
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = transformers.AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    inputs = prepare_image_inputs(processor, "Compare these images.", _demo_images())
    position_ids, _ = get_qwen3_vl_rope_index_from_config(
        inputs.input_ids,
        config=config,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    return inputs, position_ids


def run_hf_vl_forward(inputs) -> torch.Tensor:
    """运行 HF 多图 forward，返回最后 token logits。"""

    from transformers import Qwen3VLForConditionalGeneration

    print(f"  加载 HF VL 模型到 GPU... (当前显存: {_gpu_mem():.1f} GB)")
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            trust_remote_code=True,
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        out = model(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            pixel_values=inputs.pixel_values.to(DEVICE),
            image_grid_thw=inputs.image_grid_thw.to(DEVICE),
        )
    logits = out.logits[:, -1, :].clone()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  HF VL 已释放 (显存: {_gpu_mem():.1f} GB)")
    return logits


def run_our_vl_forward(inputs, position_ids: torch.Tensor) -> torch.Tensor:
    """运行 Prism-Infer 多图 forward，返回最后 token logits。"""

    from transformers import Qwen3VLForConditionalGeneration
    from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

    print(f"  加载 HF 到 CPU 提取权重... (显存: {_gpu_mem():.1f} GB)")
    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    print("  创建 Prism-Infer VL 模型到 GPU...")
    default_device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        our = Qwen3VLForCausalLM(config=hf_cpu.config, dtype=DTYPE).eval()
    finally:
        torch.set_default_device(default_device)

    our_sd = our.state_dict()
    hf_sd = hf_cpu.state_dict()
    loaded, missing = 0, []
    for key in our_sd:
        if key in hf_sd:
            our_sd[key].copy_(hf_sd[key].to(DEVICE), non_blocking=True)
            loaded += 1
        else:
            missing.append(key)
    unexpected = [key for key in hf_sd if key not in our_sd]
    del hf_cpu, hf_sd
    gc.collect()
    print(f"  权重: {loaded}/{len(our_sd)} loaded")
    print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    with torch.no_grad():
        hidden = our(
            input_ids=inputs.input_ids.to(DEVICE),
            pixel_values=inputs.pixel_values.to(DEVICE),
            image_grid_thw=inputs.image_grid_thw.to(DEVICE),
            position_ids=position_ids.to(DEVICE),
        )
        logits = our.compute_logits(hidden)[:, -1, :].clone()

    del our, hidden
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Prism-Infer VL 已释放 (显存: {_gpu_mem():.1f} GB)")
    return logits


def compare_last_logits(hf_logits: torch.Tensor, our_logits: torch.Tensor) -> str:
    """打印并返回最后 token logits 对齐结果。"""

    hf_f = hf_logits.float().cpu()
    our_f = our_logits.float().cpu()
    diff = (hf_f - our_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    hf_nan = torch.isnan(hf_f).sum().item()
    our_nan = torch.isnan(our_f).sum().item()
    hf_mean, hf_std = hf_f.mean().item(), hf_f.std().item()
    our_mean, our_std = our_f.mean().item(), our_f.std().item()

    print("\n=== Multi-Image VL Last Logits 对比 ===")
    print(f"  Shape: HF={list(hf_logits.shape)}, Our={list(our_logits.shape)}")
    print(f"  NaN: HF={hf_nan}, Our={our_nan}")
    print(f"  HF mean/std:  {hf_mean:.6e} / {hf_std:.6e}")
    print(f"  Our mean/std: {our_mean:.6e} / {our_std:.6e}")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if hf_nan > 0 or our_nan > 0:
        print("  FAIL: NaN detected")
        return "FAIL"
    if max_diff < 1e-2:
        print("  PASS (max diff < 0.01)")
        return "PASS"
    print("  FAIL: max diff >= 0.01")
    return "FAIL"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 才能运行多图 full-model logits 验证")

    print("=" * 60)
    print("Prism-Infer Multi-Image VL Full Model Verification")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE}")
    print("=" * 60)

    vl_inputs, vl_position_ids = _prepare_inputs()
    print(f"input_ids shape: {list(vl_inputs.input_ids.shape)}")
    print(f"pixel_values shape: {list(vl_inputs.pixel_values.shape)}")
    print(f"image_grid_thw shape: {list(vl_inputs.image_grid_thw.shape)}")
    print(
        f"image tokens: {vl_inputs.image_token_count} / expected {vl_inputs.expected_image_tokens}"
    )
    print(f"position_ids shape: {list(vl_position_ids.shape)}")

    print("\n[1/2] HF multi-image VL forward...")
    hf = run_hf_vl_forward(vl_inputs)
    print("\n[2/2] Prism-Infer multi-image VL forward...")
    ours = run_our_vl_forward(vl_inputs, vl_position_ids)
    result = compare_last_logits(hf, ours)

    if result != "PASS":
        raise SystemExit(1)
