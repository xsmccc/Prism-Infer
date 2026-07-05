"""P3.4 batch-size 数值敏感性记录。

Qwen3-VL bf16 full forward 在 batch=1 与 batch=4 duplicate 输入上会选择
不同 GEMM/attention 计算路径，HF 和 Prism-Infer 都会产生同量级 logits 差异。
该测试用于解释 mixed batch 长输出中 text-only row 不能强制要求与单请求
32-token 完全一致；Prism-Infer 的 CUDA logits 路径使用 fp32 lm_head 做
greedy tie-break，因此这里验证同量级差异而不是逐位相同。
"""

import gc

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from conftest import get_model_path, require_transformers
from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    pytest = __import__("pytest")
    pytest.skip("batch numeric sensitivity test requires CUDA")


def _load_our_from_hf(hf_model) -> Qwen3VLForCausalLM:
    default_device = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        our = Qwen3VLForCausalLM(config=hf_model.config, dtype=torch.bfloat16).eval()
    finally:
        torch.set_default_device(default_device)
    our_sd = our.state_dict()
    hf_sd = hf_model.state_dict()
    for key in our_sd:
        our_sd[key].copy_(hf_sd[key].to("cuda"), non_blocking=True)
    return our


def _compare_duplicate_batch(model, input_ids: torch.Tensor, *, is_hf: bool) -> tuple[float, float, int, int]:
    single_ids = input_ids[:1]
    batch_ids = input_ids.expand(4, -1).contiguous()
    attention_mask = torch.ones_like(single_ids)
    batch_mask = torch.ones_like(batch_ids)
    with torch.inference_mode():
        if is_hf:
            single_logits = model(input_ids=single_ids, attention_mask=attention_mask).logits[:, -1, :]
            batch_logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits[:, -1, :]
        else:
            single_hidden = model(input_ids=single_ids)
            batch_hidden = model(input_ids=batch_ids)
            single_logits = model.compute_logits(single_hidden)[:, -1, :]
            batch_logits = model.compute_logits(batch_hidden)[:, -1, :]
    diff = (single_logits[0].float().cpu() - batch_logits[0].float().cpu()).abs()
    return (
        diff.max().item(),
        diff.mean().item(),
        int(single_logits[0].argmax().item()),
        int(batch_logits[0].argmax().item()),
    )


def test_hf_and_prism_share_text_duplicate_batch_numeric_sensitivity() -> None:
    """HF 与 Prism 对 duplicate batch 的 bf16 数值敏感性应同量级。"""

    _require_cuda()
    require_transformers()
    model_path = get_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    input_ids = torch.tensor([tokenizer.encode("Hello")], device="cuda", dtype=torch.long)

    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    hf_max, hf_mean, hf_single_arg, hf_batch_arg = _compare_duplicate_batch(
        hf_model,
        input_ids,
        is_hf=True,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    our_model = _load_our_from_hf(hf_cpu)
    del hf_cpu
    gc.collect()
    torch.cuda.empty_cache()
    our_max, our_mean, our_single_arg, our_batch_arg = _compare_duplicate_batch(
        our_model,
        input_ids,
        is_hf=False,
    )
    del our_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"duplicate input_ids shape: {list(input_ids.shape)}")
    print(f"HF duplicate batch max diff: {hf_max:.6e}")
    print(f"HF duplicate batch mean diff: {hf_mean:.6e}")
    print(f"HF argmax single/batch: {hf_single_arg} / {hf_batch_arg}")
    print(f"Prism duplicate batch max diff: {our_max:.6e}")
    print(f"Prism duplicate batch mean diff: {our_mean:.6e}")
    print(f"Prism argmax single/batch: {our_single_arg} / {our_batch_arg}")

    assert abs(hf_max - our_max) < 1e-2
    assert abs(hf_mean - our_mean) < 1e-3
    assert hf_single_arg == our_single_arg
    assert hf_batch_arg == our_batch_arg
    print("HF/Prism duplicate batch numeric sensitivity: PASS")
