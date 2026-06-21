"""Single-layer debug: isolate which submodule causes the 27 max diff."""
import sys

if "pytest" in sys.modules:
    import pytest
    pytest.skip("manual GPU debug script", allow_module_level=True)

import torch, gc
from conftest import get_model_path

torch.manual_seed(42)
torch.cuda.empty_cache()

MODEL_PATH = get_model_path()

from transformers import Qwen3VLForConditionalGeneration

# Load HF on CPU for weight extraction
print("Loading HF (CPU)...")
hf = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True)

# Extract HF weights
hf_sd = hf.state_dict()

# Build our model on GPU
import sys; sys.path.insert(0, '/data/Prism-Infer')
from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM

print("Building our model (GPU)...")
our = Qwen3VLForCausalLM(dtype=torch.bfloat16).cuda().eval()
our_sd = our.state_dict()
for key in our_sd:
    if key in hf_sd:
        our_sd[key].copy_(hf_sd[key].cuda())
print(f"  Loaded {len(our_sd)} weights")

# Test data
x_cpu = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
x_cuda = x_cpu.clone().cuda()
pos_ids = torch.arange(4).unsqueeze(0)  # [1, 4]

# Get HF position embeddings (CPU) and move to GPU for our model
hf_rot = hf.model.language_model.rotary_emb
our_rot = our.model.language_model.rotary_emb

with torch.no_grad():
    hf_cos, hf_sin = hf_rot(x_cpu, pos_ids)
    # HF returns [B, S, D] or similar - normalize
    if hf_cos.dim() > 3:
        hf_cos = hf_cos.reshape(1, 4, -1)
        hf_sin = hf_sin.reshape(1, 4, -1)
    print(f"HF cos: {hf_cos.shape}, sin: {hf_sin.shape}")

    our_cos, our_sin = our_rot(x_cpu, pos_ids)
    print(f"Our cos: {our_cos.shape}, sin: {our_sin.shape}")

cos_diff = (hf_cos.float() - our_cos.float()).abs().max().item()
sin_diff = (hf_sin.float() - our_sin.float()).abs().max().item()
print(f"RoPE cos diff: {cos_diff:.6e}")
print(f"RoPE sin diff: {sin_diff:.6e}")

# === Test 1: Attention with SAME RoPE ===
print("\n=== Attention (same HF RoPE) ===")
hf_attn = hf.model.language_model.layers[0].self_attn
our_attn = our.model.language_model.layers[0].self_attn

# HF on CPU: keep cos/sin on CPU
hf_cos_attn = hf_cos.unsqueeze(1)  # [1, 1, 4, 128] on CPU
hf_sin_attn = hf_sin.unsqueeze(1)
# Our on GPU: move cos/sin to GPU
our_cos_gpu = hf_cos.cuda()  # [1, 4, 128] on GPU
our_sin_gpu = hf_sin.cuda()

with torch.no_grad():
    hf_out = hf_attn(x_cpu, position_embeddings=(hf_cos_attn, hf_sin_attn), attention_mask=None)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

with torch.no_grad():
    our_out = our_attn(x_cuda, position_embeddings=(our_cos_gpu, our_sin_gpu), attention_mask=None)

diff = (hf_out.float() - our_out.float().cpu()).abs().max().item()
print(f"  Max diff (same RoPE): {diff:.6e}")
if diff < 1e-2:
    print(f"  ✅ PASS — attention implementation matches HF")
elif diff < 5:
    print(f"  ⚠️  MODERATE — small differences in attention")
else:
    print(f"  ❌ FAIL — attention doesn't match")

# === Test 2: Full layer with SAME RoPE ===
print("\n=== Full Layer 0 (same HF RoPE) ===")
hf_layer = hf.model.language_model.layers[0]
our_layer = our.model.language_model.layers[0]

with torch.no_grad():
    hf_full = hf_layer(x_cpu, position_embeddings=(hf_cos_attn, hf_sin_attn), attention_mask=None)
    if isinstance(hf_full, tuple):
        hf_full = hf_full[0]

with torch.no_grad():
    our_full = our_layer(x_cuda, position_embeddings=(our_cos_gpu, our_sin_gpu), attention_mask=None)

diff = (hf_full.float() - our_full.float().cpu()).abs().max().item()
print(f"  Max diff (same RoPE): {diff:.6e}")
if diff < 1e-2:
    print(f"  ✅ PASS")
elif diff < 5:
    print(f"  ⚠️  MODERATE")
else:
    print(f"  ❌ FAIL")

# === Test 3: MLP only ===
print("\n=== MLP only ===")
hf_mlp = hf.model.language_model.layers[0].mlp
our_mlp = our_layer.mlp

with torch.no_grad():
    hf_mlp_out = hf_mlp(x_cpu)
    our_mlp_out = our_mlp(x_cuda)
diff = (hf_mlp_out.float() - our_mlp_out.float().cpu()).abs().max().item()
print(f"  Max diff: {diff:.6e}  {'PASS' if diff < 1e-5 else 'FAIL'}")

del hf, our
torch.cuda.empty_cache()
print("\nDone")
