"""P5.0 compression-off baseline tests."""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.compression import (
    CompressionMetadata,
    build_compression_metadata,
    ensure_compression_off,
    normalize_compression_mode,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.layers.attention import Attention
from prism_infer.sampling_params import SamplingParams
from prism_infer.utils.context import reset_context, set_context


def test_compression_mode_off_validation():
    """P5.0 only accepts the explicit off baseline."""

    assert normalize_compression_mode(None) == "off"
    assert normalize_compression_mode("OFF") == "off"
    with pytest.raises(ValueError, match="only supports compression_mode='off'"):
        normalize_compression_mode("visual_prune")
    print("compression mode off validation: PASS")


def test_compression_metadata_counts_visual_tokens():
    """Off metadata should expose visual-token counts without changing behavior."""

    config = SimpleNamespace(compression_mode="off", kvcache_block_size=256)
    image_seq = Sequence(
        [1, 99, 99, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        image_token_id=99,
        image_token_count=2,
    )
    video_seq = Sequence(
        [1, 98, 98, 98, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
        video_token_id=98,
        video_token_count=3,
    )

    metadata = build_compression_metadata(
        config,
        [image_seq, video_seq],
        is_prefill=True,
    )

    print(f"compression metadata mode: {metadata.mode}")
    print(f"compression metadata visual tokens: {metadata.total_visual_tokens}")
    print(f"compression metadata prompt tokens: {metadata.total_prompt_tokens}")

    assert metadata.mode == "off"
    assert not metadata.enabled
    assert metadata.is_prefill
    assert metadata.num_sequences == 2
    assert metadata.total_image_tokens == 2
    assert metadata.total_video_tokens == 3
    assert metadata.total_visual_tokens == 5
    assert metadata.total_prompt_tokens == 9
    assert metadata.block_size == 256
    print("compression metadata visual-token counts: PASS")


def _run_prefill_attention(compression_metadata: CompressionMetadata | None) -> torch.Tensor:
    torch.manual_seed(20260705)
    dtype = torch.float32
    seqlen = 5
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    q = torch.randn(seqlen, num_heads, head_dim, dtype=dtype)
    k = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype)
    v = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype)
    attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
    )
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32)

    try:
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            slot_mapping=torch.arange(seqlen, dtype=torch.int32),
            compression_metadata=compression_metadata,
        )
        with torch.inference_mode():
            return attn(q, k, v)
    finally:
        reset_context()


def test_compression_off_attention_output_identical():
    """Carrying off metadata must not change attention outputs."""

    metadata = CompressionMetadata(
        mode="off",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=2,
        total_video_tokens=0,
        block_size=256,
    )

    baseline = _run_prefill_attention(None)
    with_metadata = _run_prefill_attention(metadata)
    diff = (baseline - with_metadata).abs()

    print(f"compression off attention output shape: {list(with_metadata.shape)}")
    print(f"compression off attention max diff: {diff.max().item():.6e}")

    assert torch.equal(baseline, with_metadata)
    print("compression off attention equivalence: PASS")


def test_attention_rejects_unimplemented_compression_mode():
    """Compression modes must fail loudly until an implementation exists."""

    metadata = CompressionMetadata(
        mode="visual_prune",
        is_prefill=True,
        num_sequences=1,
        total_prompt_tokens=5,
        total_image_tokens=2,
        total_video_tokens=0,
        block_size=256,
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        ensure_compression_off(metadata)
    print("compression non-off guard: PASS")
