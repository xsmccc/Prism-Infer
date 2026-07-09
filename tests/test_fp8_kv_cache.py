"""P5.3 FP8 KV cache baseline tests."""

import pytest
import torch
import torch.nn.functional as F

from prism_infer.engine.compression import CompressionMetadata
from prism_infer.layers.attention import Attention, store_kvcache
from prism_infer.utils.context import reset_context, set_context


def _require_fp8() -> None:
    if hasattr(torch, "float8_e4m3fn"):
        return
    pytest.skip("torch.float8_e4m3fn is required for fp8_kv tests")


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
