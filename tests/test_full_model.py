"""
全模型验证: 在 GPU 上对比我们的 Qwen3VLForCausalLM vs HF Qwen3VLForConditionalGeneration.

策略: 两个模型永不同时占用 GPU, 各自加载→跑 forward→释放。
"""
import gc, torch
import sys
sys.path.insert(0, '/data/Prism-Infer')
from conftest import get_model_path

MODEL_PATH = get_model_path()
DTYPE = torch.bfloat16
DEVICE = 'cuda'
VOCAB_SIZE = 151936


def _gpu_mem():
    return torch.cuda.memory_allocated() / 1024**3


def run_hf_forward(input_ids):
    """加载 HF 模型到 GPU, 跑 forward, 释放, 返回 logits."""
    from transformers import Qwen3VLForConditionalGeneration
    print(f"  加载 HF 模型到 GPU... (当前显存: {_gpu_mem():.1f} GB)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=DTYPE, trust_remote_code=True, local_files_only=True)
    model = model.cuda().eval()
    print(f"  HF 加载完成 (显存: {_gpu_mem():.1f} GB)")

    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits.clone()
    del model, out; gc.collect(); torch.cuda.empty_cache()
    print(f"  HF 已释放 (显存: {_gpu_mem():.1f} GB)")
    return logits


def run_our_forward(input_ids):
    """创建我们的模型, 从 HF CPU 加载权重到 GPU, 跑 forward, 释放, 返回 logits."""
    from transformers import Qwen3VLForConditionalGeneration
    from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

    # 1. 加载 HF 到 CPU (仅用于提取权重)
    print(f"  加载 HF 到 CPU 提取权重... (显存: {_gpu_mem():.1f} GB)")
    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=DTYPE, trust_remote_code=True, local_files_only=True)
    # 放在 CPU 上, 不占 GPU
    hf_cpu.eval()

    # 2. 直接在 GPU 上创建我们的模型，避免 CPU 同时持有两份 8B 权重。
    print(f"  创建 Prism-Infer 模型到 GPU...")
    default_device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        our = Qwen3VLForCausalLM(config=hf_cpu.config, dtype=DTYPE).eval()
    finally:
        torch.set_default_device(default_device)

    # 3. 逐参数复制: HF(CPU) → Our(GPU)
    our_sd = our.state_dict()
    hf_sd = hf_cpu.state_dict()
    loaded, missing = 0, []
    for key in our_sd:
        if key in hf_sd:
            our_sd[key].copy_(hf_sd[key].to(DEVICE), non_blocking=True)
            loaded += 1
        else:
            missing.append(key)
    unexpected = [k for k in hf_sd if k not in our_sd]

    # 4. 释放 HF CPU 模型
    del hf_cpu, hf_sd; gc.collect()
    print(f"  权重: {loaded}/{len(our_sd)} loaded")
    print(f"  Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
    print(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"  加载完成 (显存: {_gpu_mem():.1f} GB)")

    # 5. 跑 forward
    with torch.no_grad():
        hidden = our(input_ids=input_ids)
    logits = our.compute_logits(hidden).clone()
    del our, hidden; gc.collect(); torch.cuda.empty_cache()
    print(f"  Our 已释放 (显存: {_gpu_mem():.1f} GB)")
    return logits


def compare_logits(hf_logits, our_logits):
    """对比两个 logits tensor."""
    hf_f = hf_logits.float().cpu()
    our_f = our_logits.float().cpu()
    max_diff = (hf_f - our_f).abs().max().item()
    mean_diff = (hf_f - our_f).abs().mean().item()
    hf_nan = torch.isnan(hf_f).sum().item()
    our_nan = torch.isnan(our_f).sum().item()

    print(f"\n=== Logits 对比 ===")
    print(f"  Shape: HF={list(hf_logits.shape)}, Our={list(our_logits.shape)}")
    print(f"  NaN: HF={hf_nan}, Our={our_nan}")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff > 1.0:
        print(f"  ❌ FAIL: max diff 过大")
        return "FAIL"
    elif hf_nan > 0 or our_nan > 0:
        print(f"  ❌ FAIL: NaN detected")
        return "FAIL"
    elif max_diff < 0.01:
        print(f"  ✅ PASS (max diff < 0.01)")
        return "PASS"
    else:
        print(f"  ⚠️  MARGINAL (0.01 <= max diff < 1.0)")
        return "MARGINAL"


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Prism-Infer Full Model Verification")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Dtype: {DTYPE}")
    print(f"策略: 两个模型各自独立加载/释放, 永不同时占用 GPU")
    print("=" * 60)

    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 64)).to(DEVICE)
    attention_mask = None  # 纯文本用 causal, 让 scaled_dot_product_attention 自动处理

    # Step 1: 跑 HF model
    print("\n[1/2] HF forward...")
    hf_logits = run_hf_forward(input_ids)

    # Step 2: 跑 Our model
    print("\n[2/2] Our forward...")
    our_logits = run_our_forward(input_ids)

    # Step 3: 对比
    result = compare_logits(hf_logits, our_logits)

    print(f"\n{'=' * 60}")
    print(f"Result: {result}")
    print("=" * 60)
