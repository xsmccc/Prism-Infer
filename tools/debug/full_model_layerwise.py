"""Full-model layerwise debug for Qwen3-VL logits alignment.

手动 GPU 定位脚本。直接运行时会:
1. 运行 HF full forward 并记录关键激活到 CPU。
2. 释放 HF GPU 模型。
3. 加载 Prism-Infer 模型、复制 HF 权重、运行 forward 并记录同名激活。
4. 按层打印 max diff / mean diff / mean / std，用于定位误差开始放大的位置。
"""
import gc
from collections.abc import Callable

import torch

from _common import get_model_path


MODEL_PATH = get_model_path()
DTYPE = torch.bfloat16
DEVICE = "cuda"
VOCAB_SIZE = 151936
SEQ_LEN = 64


def _gpu_mem() -> float:
    """返回当前 CUDA allocated 显存，单位 GiB。"""
    return torch.cuda.memory_allocated() / 1024**3


def _first_tensor(value):
    """从 module output 中提取第一个 tensor。"""
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
    """记录 tensor 到 CPU float32，避免保留 GPU 引用。"""
    tensor = _first_tensor(value)
    if tensor is None:
        return
    store[name] = tensor.detach().float().cpu()


def _register_common_hooks(model, store: dict[str, torch.Tensor], prefix: str) -> list:
    """在 HF/Prism-Infer 的同构模块上注册 hook。"""
    handles = []
    lm = model.model.language_model

    def save(name: str) -> Callable:
        return lambda _module, _args, output: _record_tensor(store, name, output)

    def save_input(name: str) -> Callable:
        return lambda _module, args: _record_tensor(store, name, args[0])

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


def run_hf_trace(input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """运行 HF forward 并返回激活 trace。"""
    from transformers import Qwen3VLForConditionalGeneration

    print(f"[HF] loading model on GPU... mem={_gpu_mem():.1f} GiB")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda().eval()

    store: dict[str, torch.Tensor] = {}
    handles = _register_common_hooks(model, store, "hf")
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    _remove_hooks(handles)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[HF] done and released. mem={_gpu_mem():.1f} GiB, tensors={len(store)}")
    return store


def run_our_trace(input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """运行 Prism-Infer forward 并返回激活 trace。"""
    from transformers import Qwen3VLForConditionalGeneration
    from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

    print(f"[Our] loading HF on CPU for weights... mem={_gpu_mem():.1f} GiB")
    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    print("[Our] building Prism-Infer model on GPU...")
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
    print(f"[Our] model ready. mem={_gpu_mem():.1f} GiB")

    store: dict[str, torch.Tensor] = {}
    handles = _register_common_hooks(model, store, "our")
    with torch.no_grad():
        hidden_states = model(input_ids=input_ids)
        _ = model.compute_logits(hidden_states)
    _remove_hooks(handles)

    del model, hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Our] done and released. mem={_gpu_mem():.1f} GiB, tensors={len(store)}")
    return store


def _stats(tensor: torch.Tensor) -> tuple[float, float]:
    return tensor.mean().item(), tensor.std().item()


def compare_traces(hf: dict[str, torch.Tensor], ours: dict[str, torch.Tensor]) -> None:
    """比较 HF 与 Prism-Infer 同名 trace。"""
    print("\n=== Layerwise Diff ===")
    print("name                         max_diff     mean_diff    hf_mean      hf_std       our_mean     our_std")
    print("-" * 104)

    ordered_names = ["embed", "rope"]
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
        raise RuntimeError("需要 CUDA 才能运行 full-model layerwise debug")

    print("=" * 80)
    print("Prism-Infer Full Model Layerwise Debug")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE}")
    print(f"Seq len: {SEQ_LEN}")
    print("=" * 80)

    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)

    hf_trace = run_hf_trace(input_ids)
    our_trace = run_our_trace(input_ids)
    compare_traces(hf_trace, our_trace)
