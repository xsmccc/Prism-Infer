"""Flash-decoding split-K paged decode alignment tests."""

import pytest
import torch

from prism_infer.engine.kv_quantization import KV_SCALE_DTYPE
from prism_infer.ops.paged_decode import paged_decode_attention
from prism_infer.ops.paged_attention_reference import paged_decode_attention_reference

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64
PAGE_SIZE = 32
SCALE = HEAD_DIM**-0.5


def _case(*, fp8: bool, batch: int, context_len: int, splits: int, seed: int = 3):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    k_cache = (
        torch.randn(pages * 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device)
        .to(torch.bfloat16)
        .to(cache_dtype)
    )
    v_cache = (
        torch.randn(pages * 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device)
        .to(torch.bfloat16)
        .to(cache_dtype)
    )
    if fp8:
        k_scale = (torch.rand(pages * 2, PAGE_SIZE, NUM_KV_HEADS, device=device) * 0.5 + 0.75).to(
            KV_SCALE_DTYPE
        )
        v_scale = (torch.rand(pages * 2, PAGE_SIZE, NUM_KV_HEADS, device=device) * 0.5 + 0.75).to(
            KV_SCALE_DTYPE
        )
    else:
        k_scale = v_scale = None
    q = torch.randn(batch, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.5
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, -1).repeat(batch, 1)
    context_lens = torch.full((batch,), context_len, dtype=torch.int32, device=device)
    max_context_len = context_lens.max()

    from prism_infer.utils.context import Context

    context = Context(
        is_prefill=False,
        block_tables=block_tables,
        context_lens=context_lens,
        decode_max_context_len=max_context_len,
    )
    with torch.no_grad():
        out = paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale=SCALE,
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
            max_context_len=max_context_len,
            block_n=32,
            num_splits=splits,
        )
        ref = paged_decode_attention_reference(
            q,
            k_cache,
            v_cache,
            context,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            scale=SCALE,
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
            profile_prefix="test",
        )
    return out, ref


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("batch,context_len", [(1, 300), (1, 1024), (2, 2000), (2, 4096)])
def test_split_k_matches_reference(fp8, splits, batch, context_len):
    out, ref = _case(fp8=fp8, batch=batch, context_len=context_len, splits=splits)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"fp8={fp8} splits={splits} bs={batch} ctx={context_len} max_abs_diff={diff:.6e}")
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("  PASS")


def test_split_k_equals_single_split():
    """split-K 输出与单段 kernel 一致 (数值同阶)。"""

    single, ref = _case(fp8=True, batch=1, context_len=2048, splits=1)
    multi, _ = _case(fp8=True, batch=1, context_len=2048, splits=4)
    assert torch.allclose(single.float(), multi.float(), atol=1e-2, rtol=1e-2)
    print("  PASS")


def test_split_k_short_context_empty_splits():
    """短上下文: 多余 split 空转 (l=0) 不得产生 NaN/错误输出。"""

    out, ref = _case(fp8=True, batch=1, context_len=64, splits=4)
    assert torch.isfinite(out.float()).all()
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("  PASS")
