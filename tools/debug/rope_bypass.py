"""Manually compare Prism MRope with the Hugging Face RoPE implementation."""

import torch, gc; torch.manual_seed(42)
from _common import get_model_path
from transformers import Qwen3VLForConditionalGeneration
MODEL_PATH = get_model_path()

input_ids_gpu = torch.randint(0, 5000, (1, 16)).cuda()  # shorter seq

# === Step 1: HF forward on GPU ===
print("[1/3] HF forward...")
hf = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True)
hf = hf.cuda().eval()
with torch.no_grad():
    hf_out = hf(input_ids=input_ids_gpu)
hf_logits = hf_out.logits.detach().float().cpu()
del hf_out, hf; gc.collect(); torch.cuda.empty_cache()
assert torch.cuda.memory_allocated() < 1e9, f"HF not freed: {torch.cuda.memory_allocated()/1e9:.1f}GB"
print(f"  HF done, GPU free")

# === Step 2: Load HF on CPU for weights + RoPE ===
print("[2/3] Extract RoPE + weights...")
hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True)
hf_rot = hf_cpu.model.language_model.rotary_emb
pos_ids = torch.arange(16).unsqueeze(0)  # seqlen=16
x_dummy = torch.randn(1, 16, 4096, dtype=torch.bfloat16)
hf_cos, hf_sin = hf_rot(x_dummy, pos_ids)
hf_sd = hf_cpu.state_dict()

from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM
our = Qwen3VLForCausalLM(dtype=torch.bfloat16).cuda().eval()
osd = our.state_dict()
for k in osd:
    if k in hf_sd: osd[k].copy_(hf_sd[k].cuda())
del hf_cpu, hf_sd; gc.collect()
hf_cos_gpu = hf_cos.cuda(); hf_sin_gpu = hf_sin.cuda()
print(f"  Our model ready on GPU")

# === Step 3: Methods 1 and 2 ===
print("[3/3] Run comparisons...")
with torch.no_grad():
    # Method 1: Our MRope
    out1 = our(input_ids=input_ids_gpu)
    logits1 = our.compute_logits(out1).float().cpu()

    # Method 2: HF RoPE
    hidden = our.model.language_model.embed_tokens(input_ids_gpu)
    pos_emb = (hf_cos_gpu, hf_sin_gpu)
    for layer in our.model.language_model.layers:
        hidden = layer(hidden, position_embeddings=pos_emb, attention_mask=None)
    hidden = our.model.language_model.norm(hidden)
    logits2 = our.compute_logits(hidden).float().cpu()

print(f'=== Results (seqlen=16) ===')
print(f'Our (our MRope) vs HF:  max={(logits1-hf_logits).abs().max():.4f}  mean={(logits1-hf_logits).abs().mean():.4f}')
print(f'Our (HF RoPE)  vs HF:  max={(logits2-hf_logits).abs().max():.4f}  mean={(logits2-hf_logits).abs().mean():.4f}')
print(f'Our (our) vs Our (HF):  max={(logits1-logits2).abs().max():.4f}')
