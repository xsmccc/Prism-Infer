"""Inspect the repaired Scaled-FP8 prefix-hit Attention path without a model."""

import json

import torch
from torch.profiler import ProfilerActivity, profile

from prism_infer.layers.attention import Attention
from prism_infer.utils.context import Context


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    q = torch.randn(32, 32, 128, device="cuda", dtype=torch.bfloat16)
    attention = Attention(num_heads=32, num_kv_heads=8, head_dim=128, scale=128**-0.5)
    attention.k_cache = torch.randn(8, 256, 8, 128, device="cuda").to(torch.float8_e4m3fn)
    attention.v_cache = torch.randn_like(attention.k_cache, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    attention.k_scale_cache = torch.ones(8, 256, 8, device="cuda")
    attention.v_scale_cache = torch.ones_like(attention.k_scale_cache)
    context = Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 32], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 2048], device="cuda", dtype=torch.int32),
        cu_seqlens_q_host=(0, 32),
        cu_seqlens_k_host=(0, 2048),
        block_tables=torch.arange(8, device="cuda", dtype=torch.int32).view(1, -1),
    )
    attention._forward_prefill_paged(q, context)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(3):
            attention._forward_prefill_paged(q, context)
        torch.cuda.synchronize()
    counts = {event.key: event.count for event in profiler.key_averages()}
    inspected = [
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::nonzero",
        "aten::repeat_interleave",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_attention_math",
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "calls": 3,
                "query_shape": list(q.shape),
                "context_tokens": 2048,
                "kv_heads": 8,
                "page_size": 256,
                "operators": {name: counts.get(name, 0) for name in inspected},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
