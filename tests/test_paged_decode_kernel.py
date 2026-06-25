"""P3.6 paged decode Triton kernel correctness 验证。"""

import torch
import torch.nn.functional as F

from prism_infer.ops.paged_decode import HAS_TRITON, paged_decode_attention


def _require_kernel() -> None:
    if torch.cuda.is_available() and HAS_TRITON:
        return
    pytest = __import__("pytest")
    pytest.skip("paged decode kernel requires CUDA and Triton")


def _reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """用 PyTorch SDPA 作为 paged decode correctness reference。"""

    outputs = []
    block_size = k_cache.shape[1]
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    groups = num_heads // num_kv_heads
    for seq_idx in range(q.shape[0]):
        context_len = int(context_lens[seq_idx].item())
        keys = []
        values = []
        remaining = context_len
        for block_id in block_tables[seq_idx].tolist():
            if remaining <= 0:
                break
            if block_id < 0:
                break
            take = min(block_size, remaining)
            keys.append(k_cache[block_id, :take])
            values.append(v_cache[block_id, :take])
            remaining -= take
        assert remaining == 0
        k = torch.cat(keys, dim=0)
        v = torch.cat(values, dim=0)
        if groups != 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        # q_i: [1, heads, 1, dim], k/v: [1, heads, context_len, dim]
        q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
        k_i = k.transpose(0, 1).unsqueeze(0)
        v_i = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            is_causal=False,
            scale=scale,
        )
        outputs.append(out.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


def _run_case(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    context_lens_list: list[int],
) -> None:
    torch.manual_seed(20260625 + head_dim + block_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_context = max(context_lens_list)
    max_blocks = (max_context + block_size - 1) // block_size
    num_blocks = batch * max_blocks

    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)

    rows = []
    next_block = 0
    for context_len in context_lens_list:
        blocks = (context_len + block_size - 1) // block_size
        row = list(range(next_block, next_block + blocks))
        row.extend([-1] * (max_blocks - blocks))
        rows.append(row)
        next_block += blocks
    block_tables = torch.tensor(rows, device=device, dtype=torch.int32)
    context_lens = torch.tensor(context_lens_list, device=device, dtype=torch.int32)
    scale = head_dim ** -0.5

    with torch.inference_mode():
        kernel_out = paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale,
        )
        ref_out = _reference_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale,
        )
    torch.cuda.synchronize()

    diff = (kernel_out - ref_out).abs()
    print(f"paged kernel q shape: {list(q.shape)}")
    print(f"paged kernel k_cache shape: {list(k_cache.shape)}")
    print(f"paged kernel v_cache shape: {list(v_cache.shape)}")
    print(f"paged kernel block_tables shape: {list(block_tables.shape)}")
    print(f"paged kernel context_lens: {context_lens_list}")
    print(f"paged kernel output shape: {list(kernel_out.shape)}")
    print(f"paged reference output shape: {list(ref_out.shape)}")
    print(f"paged kernel mean/std: {kernel_out.float().mean().item():.6e} / {kernel_out.float().std().item():.6e}")
    print(f"paged reference mean/std: {ref_out.float().mean().item():.6e} / {ref_out.float().std().item():.6e}")
    print(f"paged kernel max diff: {diff.max().item():.6e}")
    print(f"paged kernel mean diff: {diff.float().mean().item():.6e}")

    assert list(kernel_out.shape) == [batch, num_heads, head_dim]
    assert diff.max().item() < 1e-2
    print("paged decode Triton kernel correctness: PASS")


def test_paged_decode_kernel_matches_sdpa_reference_gqa_small() -> None:
    """小形状 GQA paged decode kernel 应对齐 SDPA reference。"""

    _require_kernel()
    _run_case(
        batch=3,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        block_size=4,
        context_lens_list=[1, 5, 9],
    )


def test_paged_decode_kernel_matches_sdpa_reference_qwen_shape() -> None:
    """Qwen3-VL decode head_dim=128/GQA 形状应对齐 SDPA reference。"""

    _require_kernel()
    _run_case(
        batch=2,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        context_lens_list=[17, 33],
    )
