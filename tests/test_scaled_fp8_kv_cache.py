"""P9-C dynamically scaled FP8 KV cache correctness and lifecycle gates."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from prism_infer.config import (
    KVCacheFormat,
    KVScaleMode,
    QuantizationConfig,
)
from prism_infer.engine.compression import CompressionMetadata
from prism_infer.engine.kv_layout import KVCompactionPlan
from prism_infer.engine.kv_quantization import (
    FP8_E4M3FN_MAX,
    KV_SCALE_DTYPE,
    PER_TOKEN_HEAD_SCALE_FLOOR,
    kv_block_storage_bytes,
)
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.layers.attention import Attention, HAS_TRITON, store_kvcache
from prism_infer.ops.paged_decode import (
    HAS_TRITON as HAS_PAGED_DECODE_TRITON,
    paged_decode_attention,
)
from prism_infer.utils.context import reset_context, set_context


def _require_fp8() -> None:
    if hasattr(torch, "float8_e4m3fn"):
        return
    pytest.skip("torch.float8_e4m3fn is required")


def _require_fp8_cuda() -> None:
    _require_fp8()
    if torch.cuda.is_available() and HAS_TRITON and HAS_PAGED_DECODE_TRITON:
        return
    pytest.skip("scaled FP8 CUDA tests require CUDA and Triton")


def _reference_quantize(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent per-token/per-head E4M3FN quantization reference."""

    tensor_float = tensor.float()
    scales = torch.clamp(
        tensor_float.abs().amax(dim=-1) / FP8_E4M3FN_MAX,
        min=PER_TOKEN_HEAD_SCALE_FLOOR,
    )
    payload = torch.clamp(
        tensor_float / scales.unsqueeze(-1),
        min=-FP8_E4M3FN_MAX,
        max=FP8_E4M3FN_MAX,
    ).to(torch.float8_e4m3fn)
    return payload, scales


def _dequantize(
    payload: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return payload.float() * scales.float().unsqueeze(-1)


def _reference_paged_decode(
    q: torch.Tensor,
    k_payload: torch.Tensor,
    v_payload: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    attention_scale: float,
) -> torch.Tensor:
    k_cache = _dequantize(k_payload, k_scales)
    v_cache = _dequantize(v_payload, v_scales)
    num_heads = q.shape[1]
    num_kv_heads = k_payload.shape[2]
    groups = num_heads // num_kv_heads
    outputs = []
    for row in range(q.shape[0]):
        context_len = int(context_lens[row].item())
        block_ids = block_tables[row].long()
        keys = k_cache[block_ids].reshape(-1, num_kv_heads, q.shape[-1])[:context_len]
        values = v_cache[block_ids].reshape(-1, num_kv_heads, q.shape[-1])[:context_len]
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        output = F.scaled_dot_product_attention(
            q[row].float().view(1, num_heads, 1, q.shape[-1]),
            keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0),
            is_causal=False,
            scale=attention_scale,
        )
        outputs.append(output.squeeze(0).squeeze(1))
    return torch.stack(outputs).to(q.dtype)


def _scaled_metadata(*, is_prefill: bool = False) -> CompressionMetadata:
    return CompressionMetadata(
        mode="scaled_fp8_kv",
        is_prefill=is_prefill,
        num_sequences=1,
        total_prompt_tokens=4,
        total_image_tokens=0,
        total_video_tokens=0,
        block_size=4,
    )


def test_scaled_fp8_quantization_config_is_explicit() -> None:
    resolved = QuantizationConfig().resolve(compression_mode="scaled_fp8_kv")
    assert resolved.kv_cache_format is KVCacheFormat.FP8_E4M3FN
    assert resolved.kv_scale_mode is KVScaleMode.PER_TOKEN_HEAD

    with pytest.raises(ValueError, match="kv_scale_mode conflicts"):
        QuantizationConfig(kv_scale_mode=KVScaleMode.UNIT).resolve(compression_mode="scaled_fp8_kv")


def test_scaled_fp8_storage_ratio_includes_fp32_scales() -> None:
    _require_fp8()
    shape = {
        "num_layers": 36,
        "page_size": 256,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    bf16 = kv_block_storage_bytes(
        **shape,
        payload_dtype=torch.bfloat16,
        token_head_scales=False,
    )
    scaled_fp8 = kv_block_storage_bytes(
        **shape,
        payload_dtype=torch.float8_e4m3fn,
        token_head_scales=True,
    )

    assert scaled_fp8.payload / bf16.total == 0.5
    assert scaled_fp8.scales / bf16.total == 0.015625
    assert scaled_fp8.total / bf16.total == 0.515625


def test_scaled_fp8_eager_store_matches_independent_reference() -> None:
    _require_fp8()
    torch.manual_seed(20260717)
    key = torch.randn(5, 2, 8, dtype=torch.float32) * 37.0
    value = torch.randn_like(key) * 19.0
    cache_shape = (2, 4, 2, 8)
    scale_shape = cache_shape[:-1]
    k_cache = torch.full(cache_shape, -4.0, dtype=torch.float8_e4m3fn)
    v_cache = torch.full(cache_shape, 4.0, dtype=torch.float8_e4m3fn)
    k_scales = torch.full(scale_shape, 17.0, dtype=KV_SCALE_DTYPE)
    v_scales = torch.full(scale_shape, 23.0, dtype=KV_SCALE_DTYPE)
    slot_mapping = torch.tensor([0, 1, 4, 6, -1], dtype=torch.int32)

    store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_scales,
        v_scales,
    )

    expected_k, expected_k_scales = _reference_quantize(key[:-1])
    expected_v, expected_v_scales = _reference_quantize(value[:-1])
    written_slots = torch.tensor([0, 1, 4, 6], dtype=torch.long)
    flat_k = k_cache.reshape(-1, 2, 8)
    flat_v = v_cache.reshape(-1, 2, 8)
    flat_k_scales = k_scales.reshape(-1, 2)
    flat_v_scales = v_scales.reshape(-1, 2)

    assert torch.equal(flat_k.index_select(0, written_slots).float(), expected_k.float())
    assert torch.equal(flat_v.index_select(0, written_slots).float(), expected_v.float())
    torch.testing.assert_close(flat_k_scales.index_select(0, written_slots), expected_k_scales)
    torch.testing.assert_close(flat_v_scales.index_select(0, written_slots), expected_v_scales)
    assert torch.all(flat_k[3] == -4.0)
    assert torch.all(flat_k_scales[3] == 17.0)


def test_scaled_fp8_store_and_attention_fail_closed_on_scale_mismatch() -> None:
    _require_fp8()
    key = torch.randn(1, 2, 8)
    value = torch.randn_like(key)
    k_cache = torch.empty(1, 4, 2, 8, dtype=torch.float8_e4m3fn)
    v_cache = torch.empty_like(k_cache)
    scales = torch.empty(1, 4, 2, dtype=torch.float32)
    slots = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="provided together"):
        store_kvcache(key, value, k_cache, v_cache, slots, scales, None)
    with pytest.raises(ValueError, match="shape must equal"):
        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slots,
            scales[:, :, :1],
            scales[:, :, :1],
        )
    with pytest.raises(ValueError, match="torch.float32"):
        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slots,
            scales.to(torch.float16),
            scales.to(torch.float16),
        )

    attention = Attention(num_heads=4, num_kv_heads=2, head_dim=8, scale=8**-0.5)
    attention.k_cache = k_cache
    attention.v_cache = v_cache
    try:
        set_context(
            False,
            slot_mapping=slots,
            context_lens=torch.tensor([1], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            compression_metadata=_scaled_metadata(),
        )
        with pytest.raises(RuntimeError, match="metadata/scale cache mismatch"):
            attention(torch.randn(1, 4, 8), key, value)
    finally:
        reset_context()


def test_scaled_fp8_eager_decode_matches_independent_sdpa() -> None:
    _require_fp8()
    torch.manual_seed(20260718)
    q = torch.randn(1, 4, 8, dtype=torch.float32)
    current_k = torch.randn(1, 2, 8, dtype=torch.float32) * 11.0
    current_v = torch.randn_like(current_k) * 7.0
    k_source = torch.randn(2, 4, 2, 8, dtype=torch.float32) * 13.0
    v_source = torch.randn_like(k_source) * 9.0
    k_payload, k_scales = _reference_quantize(k_source)
    v_payload, v_scales = _reference_quantize(v_source)

    attention = Attention(num_heads=4, num_kv_heads=2, head_dim=8, scale=8**-0.5)
    attention.k_cache = k_payload.clone()
    attention.v_cache = v_payload.clone()
    attention.k_scale_cache = k_scales.clone()
    attention.v_scale_cache = v_scales.clone()
    try:
        set_context(
            False,
            slot_mapping=torch.tensor([6], dtype=torch.int32),
            context_lens=torch.tensor([7], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
            compression_metadata=_scaled_metadata(),
        )
        with torch.inference_mode():
            actual = attention(q, current_k, current_v)
    finally:
        reset_context()

    expected_k_payload = k_payload.clone()
    expected_v_payload = v_payload.clone()
    expected_k_scales = k_scales.clone()
    expected_v_scales = v_scales.clone()
    current_k_payload, current_k_scales = _reference_quantize(current_k)
    current_v_payload, current_v_scales = _reference_quantize(current_v)
    expected_k_payload[1, 2] = current_k_payload[0]
    expected_v_payload[1, 2] = current_v_payload[0]
    expected_k_scales[1, 2] = current_k_scales[0]
    expected_v_scales[1, 2] = current_v_scales[0]
    expected = _reference_paged_decode(
        q,
        expected_k_payload,
        expected_v_payload,
        expected_k_scales,
        expected_v_scales,
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        8**-0.5,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _make_compaction_plan(dtype: torch.dtype) -> KVCompactionPlan:
    return KVCompactionPlan(
        seq_id=7,
        logical_prompt_len=4,
        physical_prompt_len=3,
        old_block_table=(0, 1),
        new_block_table=(0,),
        released_block_ids=(1,),
        retained_original_positions=(0, 2, 3),
        source_slots=(0, 2, 3),
        destination_slots=(0, 1, 2),
        kv_dtype=str(dtype),
        compression_record={"physical_compaction": False},
    )


def _make_lifecycle_runner() -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.block_size = 4
    runner.world_size = 1
    runner.uses_token_head_scales = True
    runner.kv_cache = torch.arange(
        2 * 2 * 2 * 4 * 1 * 3,
        dtype=torch.float32,
    ).view(2, 2, 2, 4, 1, 3)
    runner.kv_scale_cache = torch.arange(
        2 * 2 * 2 * 4 * 1,
        dtype=torch.float32,
    ).view(2, 2, 2, 4, 1)
    return runner


def test_scaled_fp8_cow_and_compaction_move_payload_and_scales_together() -> None:
    cow_runner = _make_lifecycle_runner()
    expected_payload = cow_runner.kv_cache[:, :, 0].clone()
    expected_scales = cow_runner.kv_scale_cache[:, :, 0].clone()
    cow_runner.copy_kv_blocks([(0, 1)])
    assert torch.equal(cow_runner.kv_cache[:, :, 1], expected_payload)
    assert torch.equal(cow_runner.kv_scale_cache[:, :, 1], expected_scales)

    compact_runner = _make_lifecycle_runner()
    plan = _make_compaction_plan(compact_runner.kv_cache.dtype)
    source = torch.tensor(plan.source_slots, dtype=torch.long)
    destination = torch.tensor(plan.destination_slots, dtype=torch.long)
    payload_before = compact_runner.kv_cache.reshape(2, 2, -1, 1, 3).clone()
    scales_before = compact_runner.kv_scale_cache.reshape(2, 2, -1, 1).clone()
    expected_payload = payload_before.index_select(2, source)
    expected_scales = scales_before.index_select(2, source)

    compact_runner.compact_kv_cache([plan])

    payload_after = compact_runner.kv_cache.reshape(2, 2, -1, 1, 3)
    scales_after = compact_runner.kv_scale_cache.reshape(2, 2, -1, 1)
    assert torch.equal(payload_after.index_select(2, destination), expected_payload)
    assert torch.equal(scales_after.index_select(2, destination), expected_scales)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_fp8_swap_moves_payload_and_scales_together() -> None:
    runner = object.__new__(ModelRunner)
    runner.uses_token_head_scales = True
    runner.kv_cache = torch.randn(2, 2, 2, 4, 1, 3, device="cuda")
    runner.kv_scale_cache = torch.randn(2, 2, 2, 4, 1, device="cuda")
    runner.cpu_kv_cache = torch.empty(2, 2, 1, 4, 1, 3, pin_memory=True)
    runner.cpu_kv_scale_cache = torch.empty(2, 2, 1, 4, 1, pin_memory=True)
    expected_payload = runner.kv_cache[:, :, 0].cpu()
    expected_scales = runner.kv_scale_cache[:, :, 0].cpu()

    runner.swap_blocks([(0, 0)], "out")
    runner.kv_cache[:, :, 1].zero_()
    runner.kv_scale_cache[:, :, 1].zero_()
    runner.swap_blocks([(0, 1)], "in")

    torch.testing.assert_close(runner.kv_cache[:, :, 1].cpu(), expected_payload)
    torch.testing.assert_close(runner.kv_scale_cache[:, :, 1].cpu(), expected_scales)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_fp8_triton_store_matches_reference() -> None:
    _require_fp8_cuda()
    torch.manual_seed(20260719)
    key = torch.randn(33, 2, 128, device="cuda", dtype=torch.bfloat16) * 31.0
    value = torch.randn_like(key) * 17.0
    cache_shape = (3, 16, 2, 128)
    k_cache = torch.full(cache_shape, -4.0, device="cuda", dtype=torch.float8_e4m3fn)
    v_cache = torch.full(cache_shape, 4.0, device="cuda", dtype=torch.float8_e4m3fn)
    k_scales = torch.full(cache_shape[:-1], 11.0, device="cuda", dtype=torch.float32)
    v_scales = torch.full(cache_shape[:-1], 13.0, device="cuda", dtype=torch.float32)
    slots = torch.cat(
        (
            torch.arange(16, device="cuda", dtype=torch.int32),
            torch.arange(24, 40, device="cuda", dtype=torch.int32),
            torch.tensor([-1], device="cuda", dtype=torch.int32),
        )
    )

    store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        slots,
        k_scales,
        v_scales,
    )
    torch.cuda.synchronize()

    expected_k, expected_k_scales = _reference_quantize(key[:-1])
    expected_v, expected_v_scales = _reference_quantize(value[:-1])
    written = slots[:-1].long()
    actual_k = k_cache.reshape(-1, 2, 128).index_select(0, written)
    actual_v = v_cache.reshape(-1, 2, 128).index_select(0, written)
    actual_k_scales = k_scales.reshape(-1, 2).index_select(0, written)
    actual_v_scales = v_scales.reshape(-1, 2).index_select(0, written)

    assert torch.equal(actual_k.float(), expected_k.float())
    assert torch.equal(actual_v.float(), expected_v.float())
    torch.testing.assert_close(actual_k_scales, expected_k_scales, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_v_scales, expected_v_scales, rtol=1e-6, atol=1e-7)
    assert torch.all(k_cache.reshape(-1, 2, 128)[20] == -4.0)
    assert torch.all(k_scales.reshape(-1, 2)[20] == 11.0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_fp8_paged_decode_matches_independent_reference() -> None:
    _require_fp8_cuda()
    torch.manual_seed(20260720)
    batch = 2
    num_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 16
    context_lens = torch.tensor([17, 33], device="cuda", dtype=torch.int32)
    blocks_per_sequence = 3
    block_tables = torch.arange(
        batch * blocks_per_sequence,
        device="cuda",
        dtype=torch.int32,
    ).view(batch, blocks_per_sequence)
    q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_source = (
        torch.randn(
            batch * blocks_per_sequence,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 23.0
    )
    v_source = torch.randn_like(k_source) * 15.0
    k_payload, k_scales = _reference_quantize(k_source)
    v_payload, v_scales = _reference_quantize(v_source)
    attention_scale = head_dim**-0.5

    actual = paged_decode_attention(
        q,
        k_payload,
        v_payload,
        block_tables,
        context_lens,
        attention_scale,
        k_scale_cache=k_scales,
        v_scale_cache=v_scales,
    )
    expected = _reference_paged_decode(
        q,
        k_payload,
        v_payload,
        k_scales,
        v_scales,
        block_tables,
        context_lens,
        attention_scale,
    )
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    assert diff.max().item() < 2e-2
    assert diff.mean().item() < 2e-3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_fp8_attention_cuda_graph_replay_updates_scales() -> None:
    _require_fp8_cuda()
    torch.manual_seed(20260721)
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    attention = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
    )
    attention.k_cache = torch.empty(
        1, 4, num_kv_heads, head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    attention.v_cache = torch.empty_like(attention.k_cache)
    attention.k_scale_cache = torch.empty(1, 4, num_kv_heads, device="cuda", dtype=torch.float32)
    attention.v_scale_cache = torch.empty_like(attention.k_scale_cache)
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([1], device="cuda", dtype=torch.int32)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)

    try:
        set_context(
            False,
            slot_mapping=slots,
            context_lens=context_lens,
            block_tables=block_tables,
            compression_metadata=_scaled_metadata(),
        )
        # Compile both Triton kernels before stream capture.
        attention(q, k, v)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = attention(q, k, v)

        replay_q = torch.randn_like(q)
        replay_k = torch.randn_like(k) * 29.0
        replay_v = torch.randn_like(v) * 21.0
        q.copy_(replay_q)
        k.copy_(replay_k)
        v.copy_(replay_v)
        graph.replay()
        torch.cuda.synchronize()

        expected_k_payload, expected_k_scales = _reference_quantize(replay_k)
        expected_v_payload, expected_v_scales = _reference_quantize(replay_v)
        torch.testing.assert_close(attention.k_scale_cache[0, 0], expected_k_scales[0])
        torch.testing.assert_close(attention.v_scale_cache[0, 0], expected_v_scales[0])
        assert torch.equal(attention.k_cache[0, 0].float(), expected_k_payload[0].float())
        assert torch.equal(attention.v_cache[0, 0].float(), expected_v_payload[0].float())
        expected_output = _reference_paged_decode(
            replay_q,
            expected_k_payload.view(1, 1, num_kv_heads, head_dim),
            expected_v_payload.view(1, 1, num_kv_heads, head_dim),
            expected_k_scales.view(1, 1, num_kv_heads),
            expected_v_scales.view(1, 1, num_kv_heads),
            block_tables,
            context_lens,
            head_dim**-0.5,
        )
        torch.testing.assert_close(
            captured_output,
            expected_output,
            rtol=2e-2,
            atol=2e-2,
        )
    finally:
        reset_context()
