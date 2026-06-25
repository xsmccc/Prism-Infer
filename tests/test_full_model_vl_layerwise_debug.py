"""图文 full-model layerwise debug。

直接运行本脚本定位 HF 与 Prism-Infer 单图输入 logits 差异。pytest 收集时跳过。
"""

import gc
import sys
from collections.abc import Callable

if "pytest" in sys.modules:
    import pytest

    pytest.skip("manual GPU debug script", allow_module_level=True)

import torch
from PIL import Image

sys.path.insert(0, "/data/Prism-Infer")
from conftest import get_model_path, require_transformers
from prism_infer.engine.vl_inputs import prepare_single_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config


MODEL_PATH = get_model_path()
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "logits"):
        return value.logits
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _record_tensor(store: dict[str, torch.Tensor], name: str, value) -> None:
    tensor = _first_tensor(value)
    if tensor is None:
        return
    store[name] = tensor.detach().float().cpu()


def _register_common_hooks(model, store: dict[str, torch.Tensor], prefix: str) -> list:
    handles = []
    lm = model.model.language_model

    def save(name: str) -> Callable:
        return lambda _module, _args, output: _record_tensor(store, name, output)

    def save_input(name: str) -> Callable:
        return lambda _module, args: _record_tensor(store, name, args[0])

    handles.append(model.model.visual.register_forward_hook(save(f"{prefix}.visual")))
    handles.append(lm.embed_tokens.register_forward_hook(save(f"{prefix}.embed")))
    handles.append(lm.rotary_emb.register_forward_hook(save(f"{prefix}.rope")))
    handles.append(lm.norm.register_forward_hook(save(f"{prefix}.final_norm")))

    for idx, layer in enumerate(lm.layers):
        base = f"{prefix}.layer_{idx:02d}"
        handles.append(layer.register_forward_pre_hook(save_input(f"{base}.input")))
        handles.append(layer.input_layernorm.register_forward_hook(save(f"{base}.input_norm")))
        handles.append(layer.self_attn.register_forward_hook(save(f"{base}.attn")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save(f"{base}.post_attn_norm")))
        handles.append(layer.mlp.register_forward_hook(save(f"{base}.mlp")))
        handles.append(layer.register_forward_hook(save(f"{base}.output")))

    handles.append(model.lm_head.register_forward_hook(save(f"{prefix}.logits")))
    return handles


def _remove_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()


def _prepare_inputs():
    transformers = require_transformers()
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = transformers.AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    inputs = prepare_single_image_inputs(processor, "Describe this image.", image)
    position_ids, _ = get_qwen3_vl_rope_index_from_config(
        inputs.input_ids,
        config=config,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    return inputs, position_ids


def run_hf_trace(inputs) -> dict[str, torch.Tensor]:
    from transformers import Qwen3VLForConditionalGeneration

    print(f"[HF] loading VL model on GPU... mem={_gpu_mem():.1f} GiB")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda().eval()

    store: dict[str, torch.Tensor] = {}
    handles = _register_common_hooks(model, store, "hf")
    with torch.no_grad():
        _ = model(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            pixel_values=inputs.pixel_values.to(DEVICE),
            image_grid_thw=inputs.image_grid_thw.to(DEVICE),
        )
    _remove_hooks(handles)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[HF] done. mem={_gpu_mem():.1f} GiB, tensors={len(store)}")
    return store


def run_our_trace(inputs, position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    from transformers import Qwen3VLForConditionalGeneration
    from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

    print(f"[Our] loading HF on CPU for weights... mem={_gpu_mem():.1f} GiB")
    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    print("[Our] building Prism-Infer VL model on GPU...")
    default_device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        model = Qwen3VLForCausalLM(config=hf_cpu.config, dtype=DTYPE).eval()
    finally:
        torch.set_default_device(default_device)

    our_sd = model.state_dict()
    hf_sd = hf_cpu.state_dict()
    loaded = 0
    missing = []
    for key in our_sd:
        if key in hf_sd:
            our_sd[key].copy_(hf_sd[key].to(DEVICE), non_blocking=True)
            loaded += 1
        else:
            missing.append(key)
    unexpected = [key for key in hf_sd if key not in our_sd]
    del hf_cpu, hf_sd
    gc.collect()

    print(f"[Our] weights: {loaded}/{len(our_sd)} loaded")
    print(f"[Our] missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"[Our] unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    store: dict[str, torch.Tensor] = {}
    handles = _register_common_hooks(model, store, "our")
    with torch.no_grad():
        hidden = model(
            input_ids=inputs.input_ids.to(DEVICE),
            pixel_values=inputs.pixel_values.to(DEVICE),
            image_grid_thw=inputs.image_grid_thw.to(DEVICE),
            position_ids=position_ids.to(DEVICE),
        )
        _ = model.compute_logits(hidden)
    _remove_hooks(handles)

    del model, hidden
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Our] done. mem={_gpu_mem():.1f} GiB, tensors={len(store)}")
    return store


def _stats(tensor: torch.Tensor) -> tuple[float, float]:
    return tensor.mean().item(), tensor.std().item()


def compare_traces(hf: dict[str, torch.Tensor], ours: dict[str, torch.Tensor]) -> None:
    print("\n=== VL Layerwise Diff ===")
    print("name                         max_diff     mean_diff    hf_mean      hf_std       our_mean     our_std")
    print("-" * 104)

    ordered_names = ["visual", "embed", "rope"]
    for idx in range(36):
        base = f"layer_{idx:02d}"
        ordered_names.extend([
            f"{base}.input",
            f"{base}.input_norm",
            f"{base}.attn",
            f"{base}.post_attn_norm",
            f"{base}.mlp",
            f"{base}.output",
        ])
    ordered_names.extend(["final_norm", "logits"])

    worst_name = None
    worst_diff = -1.0
    for name in ordered_names:
        hf_key = f"hf.{name}"
        our_key = f"our.{name}"
        if hf_key not in hf or our_key not in ours:
            print(f"{name:<28} missing hf={hf_key in hf} our={our_key in ours}")
            continue
        hf_tensor = hf[hf_key]
        our_tensor = ours[our_key]
        if hf_tensor.shape != our_tensor.shape:
            print(f"{name:<28} shape mismatch hf={list(hf_tensor.shape)} our={list(our_tensor.shape)}")
            continue
        diff = (hf_tensor - our_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        hf_mean, hf_std = _stats(hf_tensor)
        our_mean, our_std = _stats(our_tensor)
        if max_diff > worst_diff:
            worst_diff = max_diff
            worst_name = name
        print(
            f"{name:<28} {max_diff:>10.6e} {mean_diff:>12.6e} "
            f"{hf_mean:>11.4e} {hf_std:>11.4e} {our_mean:>11.4e} {our_std:>11.4e}"
        )

    print("-" * 104)
    print(f"Worst tensor: {worst_name}, max_diff={worst_diff:.6e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 才能运行 VL layerwise debug")

    print("=" * 80)
    print("Prism-Infer VL Full Model Layerwise Debug")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE}")
    print("=" * 80)

    vl_inputs, vl_position_ids = _prepare_inputs()
    print(f"input_ids shape: {list(vl_inputs.input_ids.shape)}")
    print(f"pixel_values shape: {list(vl_inputs.pixel_values.shape)}")
    print(f"image_grid_thw shape: {list(vl_inputs.image_grid_thw.shape)}")
    print(f"position_ids shape: {list(vl_position_ids.shape)}")

    hf_trace = run_hf_trace(vl_inputs)
    our_trace = run_our_trace(vl_inputs, vl_position_ids)
    compare_traces(hf_trace, our_trace)
