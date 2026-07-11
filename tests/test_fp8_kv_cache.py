"""P5.3 FP8 KV baseline 与 P6.2-D vectorized store tests。"""

import pytest
import torch
import torch.nn.functional as F

from prism_infer.engine.compression import CompressionMetadata
from prism_infer.layers.attention import (
    HAS_TRITON,
    Attention,
    _store_kvcache_eager,
    store_kvcache,
)
from prism_infer.ops.paged_decode import HAS_TRITON as HAS_PAGED_DECODE_TRITON
from prism_infer.utils.context import reset_context, set_context


def _require_fp8() -> None:
    if hasattr(torch, "float8_e4m3fn"):
        return
    pytest.skip("torch.float8_e4m3fn is required for fp8_kv tests")


def _require_fp8_triton() -> None:
    _require_fp8()
    if torch.cuda.is_available() and HAS_TRITON:
        return
    pytest.skip("vectorized fp8_kv store requires CUDA and Triton")


def test_fp8_kv_store_uses_half_bf16_cache_bytes_and_matches_roundtrip() -> None:
    """FP8 KV store should physically use 1 byte/element and match fp8 round-trip."""

    _require_fp8()
    torch.manual_seed(20260709)
    dtype = torch.bfloat16
    key = torch.randn(5, 2, 8, dtype=dtype)
    value = torch.randn(5, 2, 8, dtype=dtype)
    k_cache = torch.empty(2, 4, 2, 8, dtype=torch.float8_e4m3fn)
    v_cache = torch.empty_like(k_cache)
    slot_mapping = torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    expected_k = key.to(torch.float8_e4m3fn).to(dtype)
    expected_v = value.to(torch.float8_e4m3fn).to(dtype)
    actual_k = torch.stack(
        [k_cache[0, 0], k_cache[0, 1], k_cache[0, 2], k_cache[1, 0], k_cache[1, 1]],
        dim=0,
    ).to(dtype)
    actual_v = torch.stack(
        [v_cache[0, 0], v_cache[0, 1], v_cache[0, 2], v_cache[1, 0], v_cache[1, 1]],
        dim=0,
    ).to(dtype)
    k_diff = (actual_k.float() - expected_k.float()).abs()
    v_diff = (actual_v.float() - expected_v.float()).abs()
    bf16_cache_bytes = k_cache.numel() * torch.empty((), dtype=dtype).element_size()
    fp8_cache_bytes = k_cache.numel() * k_cache.element_size()

    print(f"fp8 store key shape: {list(key.shape)}")
    print(f"fp8 store cache shape: {list(k_cache.shape)}")
    print(f"bf16 cache bytes for one tensor: {bf16_cache_bytes}")
    print(f"fp8 cache bytes for one tensor: {fp8_cache_bytes}")
    print(f"fp8 store k roundtrip max diff: {k_diff.max().item():.6e}")
    print(f"fp8 store v roundtrip max diff: {v_diff.max().item():.6e}")

    assert k_cache.element_size() == 1
    assert fp8_cache_bytes * 2 == bf16_cache_bytes
    assert k_diff.max().item() == 0
    assert v_diff.max().item() == 0
    print("fp8 kv store roundtrip: PASS")


def test_fp8_kv_triton_store_matches_eager_reference_qwen_shape() -> None:
    """FP8 Triton store 应在 Qwen prefill 形状下 exact 对齐 eager reference。"""

    _require_fp8_triton()
    torch.manual_seed(20260711)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 210
    num_kv_heads = 8
    head_dim = 128
    cache_shape = (3, 256, num_kv_heads, head_dim)
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    initial_k = torch.full(
        cache_shape,
        -4.0,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    initial_v = torch.full_like(initial_k, 4.0)
    # 跨非连续 physical blocks；最后一个 token 是 padding，不应写 cache。
    slot_mapping = torch.cat(
        (
            torch.arange(104, device=device, dtype=torch.int32),
            torch.arange(256, 361, device=device, dtype=torch.int32),
            torch.tensor([-1], device=device, dtype=torch.int32),
        )
    )

    reference_k = initial_k.clone()
    reference_v = initial_v.clone()
    _store_kvcache_eager(
        key,
        value,
        reference_k,
        reference_v,
        slot_mapping,
    )
    actual_k = initial_k.clone()
    actual_v = initial_v.clone()
    store_kvcache(key, value, actual_k, actual_v, slot_mapping)
    torch.cuda.synchronize()

    k_diff = (actual_k.to(dtype).float() - reference_k.to(dtype).float()).abs()
    v_diff = (actual_v.to(dtype).float() - reference_v.to(dtype).float()).abs()
    untouched_slot = 200
    flat_actual_k = actual_k.reshape(-1, num_kv_heads, head_dim)
    flat_actual_v = actual_v.reshape(-1, num_kv_heads, head_dim)

    print(f"fp8 Triton store key shape: {list(key.shape)}")
    print(f"fp8 Triton store cache shape: {list(actual_k.shape)}")
    print(f"fp8 Triton store slot mapping shape: {list(slot_mapping.shape)}")
    print(
        "fp8 Triton store output mean/std: "
        f"{actual_k.to(dtype).float().mean().item():.6e}/"
        f"{actual_k.to(dtype).float().std().item():.6e}"
    )
    print(
        "fp8 eager store reference mean/std: "
        f"{reference_k.to(dtype).float().mean().item():.6e}/"
        f"{reference_k.to(dtype).float().std().item():.6e}"
    )
    print(f"fp8 Triton store K max diff: {k_diff.max().item():.6e}")
    print(f"fp8 Triton store V max diff: {v_diff.max().item():.6e}")

    assert actual_k.shape == reference_k.shape == cache_shape
    assert actual_v.shape == reference_v.shape == cache_shape
    assert k_diff.max().item() == 0.0
    assert v_diff.max().item() == 0.0
    assert torch.all(flat_actual_k[untouched_slot] == -4.0)
    assert torch.all(flat_actual_v[untouched_slot] == 4.0)
    print("fp8 Triton KV store eager-reference correctness: PASS")


def _reference_fp8_decode(
    q: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    ref_k_cache = k_cache.clone()
    ref_v_cache = v_cache.clone()
    ref_k_cache[1, 2] = current_k[0].to(torch.float8_e4m3fn)
    ref_v_cache[1, 2] = current_v[0].to(torch.float8_e4m3fn)

    keys = torch.cat([ref_k_cache[0, :4], ref_k_cache[1, :3]], dim=0).to(q.dtype)
    values = torch.cat([ref_v_cache[0, :4], ref_v_cache[1, :3]], dim=0).to(q.dtype)
    keys = keys.repeat_interleave(2, dim=1)
    values = values.repeat_interleave(2, dim=1)

    q_i = q[0].unsqueeze(0).unsqueeze(2)
    k_i = keys.transpose(0, 1).unsqueeze(0)
    v_i = values.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(
        q_i,
        k_i,
        v_i,
        is_causal=False,
        scale=8 ** -0.5,
    ).squeeze(0).squeeze(1).unsqueeze(0)


def test_fp8_kv_decode_matches_dequantized_reference() -> None:
    """FP8 decode path must match an independent dequantized SDPA reference."""

    _require_fp8()
    torch.manual_seed(20260709)
    dtype = torch.float32
    q = torch.randn(1, 4, 8, dtype=dtype)
    current_k = torch.randn(1, 2, 8, dtype=dtype)
    current_v = torch.randn(1, 2, 8, dtype=dtype)
    k_cache_source = torch.randn(2, 4, 2, 8, dtype=dtype)
    v_cache_source = torch.randn(2, 4, 2, 8, dtype=dtype)
    k_cache = k_cache_source.to(torch.float8_e4m3fn)
    v_cache = v_cache_source.to(torch.float8_e4m3fn)
    metadata = CompressionMetadata(
        mode="fp8_kv",
        is_prefill=False,
        num_sequences=1,
        total_prompt_tokens=6,
        total_image_tokens=0,
        total_video_tokens=0,
        block_size=4,
    )
    attn = Attention(num_heads=4, num_kv_heads=2, head_dim=8, scale=8 ** -0.5)
    attn.k_cache = k_cache.clone()
    attn.v_cache = v_cache.clone()

    try:
        set_context(
            False,
            slot_mapping=torch.tensor([6], dtype=torch.int32),
            context_lens=torch.tensor([7], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
            compression_metadata=metadata,
        )
        with torch.inference_mode():
            output = attn(q, current_k, current_v)
    finally:
        reset_context()

    reference = _reference_fp8_decode(q, current_k, current_v, k_cache, v_cache)
    diff = (output - reference).abs()

    print(f"fp8 decode q shape: {list(q.shape)}")
    print(f"fp8 decode k_cache shape: {list(k_cache.shape)}")
    print(f"fp8 decode output shape: {list(output.shape)}")
    print(f"fp8 decode reference shape: {list(reference.shape)}")
    print(f"fp8 decode output mean/std: {output.mean().item():.6e}/{output.std().item():.6e}")
    print(
        "fp8 decode reference mean/std: "
        f"{reference.mean().item():.6e}/{reference.std().item():.6e}"
    )
    print(f"fp8 decode max diff: {diff.max().item():.6e}")

    assert output.shape == reference.shape == (1, 4, 8)
    assert diff.max().item() < 1e-5
    print("fp8 kv decode reference: PASS")


def test_fp8_kv_cuda_attention_uses_paged_kernel_reference_semantics() -> None:
    """Engine FP8 CUDA decode 应走 paged kernel 并对齐量化后 SDPA reference。"""

    _require_fp8_triton()
    if not HAS_PAGED_DECODE_TRITON:
        pytest.skip("paged decode Triton kernel is required")
    torch.manual_seed(20260711)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 2
    num_heads = 8
    num_kv_heads = 2
    head_dim = 128
    block_size = 16
    context_len = 33
    blocks_per_seq = 3
    num_blocks = batch * blocks_per_seq
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=dtype)
    current_k = torch.randn(batch, num_kv_heads, head_dim, device=device, dtype=dtype)
    current_v = torch.randn_like(current_k)
    initial_k = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    ).to(torch.float8_e4m3fn)
    initial_v = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(
        num_blocks,
        device=device,
        dtype=torch.int32,
    ).view(batch, blocks_per_seq)
    context_lens = torch.full(
        (batch,), context_len, device=device, dtype=torch.int32
    )
    slot_mapping = torch.tensor(
        [
            int(block_tables[row, 2].item()) * block_size
            for row in range(batch)
        ],
        device=device,
        dtype=torch.int32,
    )
    metadata = CompressionMetadata(
        mode="fp8_kv",
        is_prefill=False,
        num_sequences=batch,
        total_prompt_tokens=batch * (context_len - 1),
        total_image_tokens=0,
        total_video_tokens=0,
        block_size=block_size,
    )
    attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
    ).cuda()
    attn.k_cache = initial_k.clone()
    attn.v_cache = initial_v.clone()

    try:
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            compression_metadata=metadata,
        )
        with torch.inference_mode():
            output = attn(q, current_k, current_v)
    finally:
        reset_context()

    reference_outputs = []
    k_dequant = attn.k_cache.to(dtype)
    v_dequant = attn.v_cache.to(dtype)
    for row in range(batch):
        ids = block_tables[row].long()
        keys = k_dequant[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]
        values = v_dequant[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]
        keys = keys.repeat_interleave(num_heads // num_kv_heads, dim=1)
        values = values.repeat_interleave(num_heads // num_kv_heads, dim=1)
        reference_outputs.append(
            F.scaled_dot_product_attention(
                q[row].view(1, num_heads, 1, head_dim),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                is_causal=False,
                scale=head_dim ** -0.5,
            ).squeeze(0).squeeze(1)
        )
    reference = torch.stack(reference_outputs)
    diff = (output - reference).abs()
    torch.cuda.synchronize()

    print(f"FP8 engine paged q/cache shapes: {list(q.shape)}/{list(initial_k.shape)}")
    print(f"FP8 engine paged output shape: {list(output.shape)}")
    print(
        "FP8 engine paged output/reference mean/std: "
        f"{output.float().mean().item():.6e}/{output.float().std().item():.6e} vs "
        f"{reference.float().mean().item():.6e}/{reference.float().std().item():.6e}"
    )
    print(
        f"FP8 engine paged max/mean diff: {diff.max().item():.6e}/"
        f"{diff.float().mean().item():.6e}"
    )
    assert output.shape == reference.shape == q.shape
    assert diff.max().item() < 1e-2
    assert diff.float().mean().item() < 1e-3
    print("P6.5 FP8 engine paged dispatch: PASS")
