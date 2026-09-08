"""Exact CUDA checks against the existing paged gather/dequantization path."""

import pytest
import torch

from prism_infer.ops.paged_attention_reference import gather_paged_kv_vectorized
from prism_infer.ops.paged_kv_gather import HAS_TRITON, gather_paged_kv_fused

pytestmark = pytest.mark.gpu
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _require_cuda_and_triton():
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required")


def _assert_exact(actual, expected):
    for output, reference in zip(actual, expected, strict=True):
        assert output.is_contiguous()
        # Also retain signed-zero and other exact output bit patterns.
        assert torch.equal(output.view(torch.uint8), reference.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    ("payload_dtype", "target_dtype", "scaled"),
    [
        (FP8, torch.bfloat16, True),
        (FP8, torch.float16, True),
        (FP8, torch.float32, True),
        (FP8, torch.bfloat16, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.float16, False),
        (torch.float16, torch.bfloat16, False),
    ],
)
def test_exact_real_layer_views_reordered_pages_and_partial_tail(
    payload_dtype,
    target_dtype,
    scaled,
):
    """Qwen3-VL-8B TP1 geometry; actual hot history is approximately 1607 tokens."""
    torch.manual_seed(20260908)
    cache = torch.empty((2, 36, 7, 256, 8, 128), device="cuda", dtype=payload_dtype)
    k_cache, v_cache = cache[0, 35], cache[1, 35]
    k_cache.copy_(torch.randn(k_cache.shape, device="cuda") * 4)
    v_cache.copy_(torch.randn(v_cache.shape, device="cuda") * 4)
    scales = torch.empty((2, 36, 7, 256, 8), device="cuda", dtype=torch.float32)
    k_scale, v_scale = scales[0, 35], scales[1, 35]
    k_scale.copy_(torch.rand(k_scale.shape, device="cuda") * 2 + 1e-6)
    v_scale.copy_(torch.rand(v_scale.shape, device="cuda") * 2 + 1e-6)
    table = torch.tensor([4, 1, 6, 0, 5, 2, 3, -1, -1], device="cuda", dtype=torch.int32)
    options = {
        "target_dtype": target_dtype,
        "k_scale_cache": k_scale if scaled else None,
        "v_scale_cache": v_scale if scaled else None,
    }
    # One specialization must handle these lengths, including the one-token tail.
    for context_len in (1, 257, 1607):
        expected = gather_paged_kv_vectorized(k_cache, v_cache, table, context_len, **options)
        actual = gather_paged_kv_fused(k_cache, v_cache, table, context_len, **options)
        _assert_exact(actual, expected)


@pytest.mark.parametrize("payload_dtype", [FP8, torch.bfloat16])
def test_exact_distinct_noncontiguous_strides_and_int64_table(payload_dtype):
    torch.manual_seed(17)
    k_storage = torch.empty((18, 32, 6, 34), device="cuda", dtype=payload_dtype)
    v_storage = torch.empty((27, 32, 3, 51), device="cuda", dtype=payload_dtype)
    k_cache = k_storage[::2, ::2, ::2, ::2]
    v_cache = v_storage[::3, ::2, :, ::3]
    k_cache.copy_(torch.randn(k_cache.shape, device="cuda") * 3)
    v_cache.copy_(torch.randn(v_cache.shape, device="cuda") * 3)
    k_scale = torch.rand((18, 32, 6), device="cuda")[::2, ::2, ::2] + 0.01
    v_scale = torch.rand((27, 16, 6), device="cuda")[::3, :, ::2] + 0.01
    # Rebuild scale views after filling so the gather must respect their strides.
    ks = torch.empty((18, 32, 6), device="cuda")[::2, ::2, ::2]
    vs = torch.empty((27, 16, 6), device="cuda")[::3, :, ::2]
    ks.copy_(k_scale)
    vs.copy_(v_scale)
    table = torch.tensor([3, -1, 0, -1, 5, -1, 3, -1], device="cuda", dtype=torch.int64)[::2]
    options = {
        "target_dtype": torch.float16,
        "k_scale_cache": ks if payload_dtype == FP8 else None,
        "v_scale_cache": vs if payload_dtype == FP8 else None,
    }
    expected = gather_paged_kv_vectorized(k_cache, v_cache, table, 61, **options)
    actual = gather_paged_kv_fused(k_cache, v_cache, table, 61, **options)
    _assert_exact(actual, expected)


@pytest.mark.parametrize(
    ("target_dtype", "scale_value"),
    [(torch.bfloat16, 1.003), (torch.float16, 1.0004)],
)
def test_scale_is_rounded_before_multiplication(target_dtype, scale_value):
    k_cache = torch.full((1, 16, 1, 16), 1.5, device="cuda").to(FP8)
    v_cache = -k_cache.to(torch.float32)
    v_cache = v_cache.to(FP8)
    scales = torch.full((1, 16, 1), scale_value, device="cuda", dtype=torch.float32)
    table = torch.tensor([0], device="cuda", dtype=torch.int32)
    options = {"target_dtype": target_dtype, "k_scale_cache": scales, "v_scale_cache": scales}
    expected = gather_paged_kv_vectorized(k_cache, v_cache, table, 7, **options)
    actual = gather_paged_kv_fused(k_cache, v_cache, table, 7, **options)
    late_rounding = (k_cache[0, :7].float() * scales[0, :7].unsqueeze(-1)).to(target_dtype)
    assert not torch.equal(late_rounding, expected[0])
    _assert_exact(actual, expected)


def test_page_stride_exceeding_signed_int32_is_addressed_exactly():
    """Touch two tiny pages in a >2 GiB FP8 backing allocation; do not fill the gap."""
    page_stride = 2**31 + 128
    storage = torch.empty(page_stride + 128, device="cuda", dtype=FP8)
    strides = (page_stride, 16, 16, 1)
    k_cache = storage.as_strided((2, 1, 1, 16), strides)
    v_cache = storage.as_strided((2, 1, 1, 16), strides, storage_offset=64)
    for page in range(2):
        k_cache[page].copy_(torch.arange(16, device="cuda").reshape(1, 1, 16) + page)
        v_cache[page].copy_(-torch.arange(16, device="cuda").reshape(1, 1, 16) - page)
    table = torch.tensor([1, 0, -1], device="cuda", dtype=torch.int32)
    options = {"target_dtype": torch.bfloat16}
    expected = gather_paged_kv_vectorized(k_cache, v_cache, table, 2, **options)
    actual = gather_paged_kv_fused(k_cache, v_cache, table, 2, **options)
    _assert_exact(actual, expected)
