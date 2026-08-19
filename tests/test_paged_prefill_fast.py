"""Paged prefill fast-path (vectorized gather + masked SDPA) tests.

覆盖: bf16 / FP8+scale、前缀命中、纯 cold、多序列 batch、GQA、
跨页上下文、q_len=1 与 chunked 语义。数值与参考路径对照
(mem-efficient 后端 vs math 后端, 允许 flash 级误差)。
"""

import pytest
import torch

from prism_infer.engine.kv_quantization import KV_SCALE_DTYPE
from prism_infer.layers.attention import Attention
from prism_infer.ops.paged_attention_reference import (
    paged_prefill_attention_reference,
)
from prism_infer.utils.context import Context

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64
BLOCK_SIZE = 32
SCALE = HEAD_DIM**-0.5


def _make_attention() -> Attention:
    attention = Attention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        scale=SCALE,
        num_kv_heads=NUM_KV_HEADS,
    )
    return attention


def _fill_caches(
    *,
    num_blocks: int,
    fp8: bool,
    device: torch.device,
):
    cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    torch.manual_seed(1234)
    k_cache = (
        torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device)
        .to(torch.bfloat16)
        .to(cache_dtype)
    )
    v_cache = (
        torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device)
        .to(torch.bfloat16)
        .to(cache_dtype)
    )
    if fp8:
        k_scale = (
            torch.rand(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, device=device) * 0.5 + 0.75
        ).to(KV_SCALE_DTYPE)
        v_scale = (
            torch.rand(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, device=device) * 0.5 + 0.75
        ).to(KV_SCALE_DTYPE)
    else:
        k_scale = v_scale = None
    return k_cache, v_cache, k_scale, v_scale


def _context(block_tables, cu_q, cu_k, max_q, max_k, k_lens) -> Context:
    return Context(
        is_prefill=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        block_tables=block_tables,
        context_lens=torch.tensor(k_lens, dtype=torch.int32, device=block_tables.device),
    )


def _run_case(
    *,
    fp8: bool,
    q_lens: list[int],
    k_lens: list[int],
    seed: int = 7,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total_q = sum(q_lens)
    total_k = sum(k_lens)
    num_blocks = (total_k + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_cache, v_cache, k_scale, v_scale = _fill_caches(
        num_blocks=num_blocks,
        fp8=fp8,
        device=device,
    )
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.5

    cu_q = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(len(k_lens) + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.tensor(q_lens, device=device).cumsum(0).to(torch.int32)
    cu_k[1:] = torch.tensor(k_lens, device=device).cumsum(0).to(torch.int32)

    # 每序列连续块编号 (顺序分配, 前缀/后缀共享同一 block 表)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(1, -1).repeat(len(q_lens), 1)
    context = _context(block_tables, cu_q, cu_k, max(q_lens), max(k_lens), k_lens)
    attention = _make_attention()
    attention.k_cache = k_cache
    attention.v_cache = v_cache
    attention.k_scale_cache = k_scale
    attention.v_scale_cache = v_scale

    with torch.no_grad():
        print(f"  cu_q={context.cu_seqlens_q.tolist()} cu_k={context.cu_seqlens_k.tolist()}")
        fast = attention._forward_prefill_paged_fast(q, context)
        ref = paged_prefill_attention_reference(
            q,
            k_cache,
            v_cache,
            context,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            scale=SCALE,
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
        )
    return fast, ref


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "q_lens,k_lens",
    [
        ([7], [7]),          # 纯 cold (prefix 0)
        ([7], [39]),         # 前缀命中
        ([1], [40]),         # q_len=1
        ([5], [64]),         # 跨页上下文
        ([3, 5, 9], [35, 37, 41]),  # 多序列 batch
    ],
)
def test_paged_prefill_fast_matches_reference(fp8, q_lens, k_lens):
    fast, ref = _run_case(fp8=fp8, q_lens=q_lens, k_lens=k_lens)
    print(f"fp8={fp8} q={q_lens} k={k_lens}")
    print(f"  max abs diff: {(fast.float() - ref.float()).abs().max().item():.6e}")
    assert torch.allclose(
        fast.float(),
        ref.float(),
        atol=5e-2,
        rtol=5e-2,
    ), "fast path diverges from reference"
    print("  PASS")


@pytest.mark.parametrize("fp8", [False, True])
def test_paged_prefill_fast_gqa_layout(fp8):
    """GQA 展开后 (kv_heads -> q_heads) 与参考路径一致。"""

    fast, ref = _run_case(fp8=fp8, q_lens=[6], k_lens=[48])
    assert fast.shape == ref.shape
    assert torch.allclose(fast.float(), ref.float(), atol=5e-2, rtol=5e-2)
    print("  GQA PASS")


def test_paged_prefill_fast_chunked_semantics():
    """chunked prefill 的中间 chunk (前缀=已计算 chunk) 因果对齐正确。"""

    # 模拟 chunk 2: 前缀 32 tokens 已入缓存, 本 chunk 查询 8 tokens
    fast, ref = _run_case(fp8=True, q_lens=[8], k_lens=[40])
    assert torch.allclose(fast.float(), ref.float(), atol=5e-2, rtol=5e-2)
    # 与冷路径 (无前缀) 的注意力结果必须不同 (前缀可见)
    fast_cold, _ = _run_case(fp8=True, q_lens=[8], k_lens=[8], seed=7)
    assert not torch.allclose(fast.float(), fast_cold.float(), atol=1e-3, rtol=1e-3)
    print("  chunked semantics PASS")
