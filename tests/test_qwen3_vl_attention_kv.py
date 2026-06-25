"""P2.4 Qwen3-VL engine attention KV cache 验证。"""

import torch

try:
    import pytest
except ImportError:
    pytest = None

from prism_infer.layers.attention import HAS_FLASH_ATTN
from prism_infer.models.qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextModel
from prism_infer.utils.context import reset_context, set_context


def _skip_if_needed() -> None:
    if torch.cuda.is_available() and HAS_FLASH_ATTN:
        return
    message = "Qwen3-VL engine attention test requires CUDA and flash-attn"
    if pytest is not None:
        pytest.skip(message)
    raise SystemExit(f"SKIP: {message}")


def test_engine_attention_prefill_matches_full_sequence_and_writes_kv():
    """flatten prefill attention 应对齐 full-sequence 路径并写入 paged KV cache。"""

    _skip_if_needed()
    torch.manual_seed(20260624)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    seqlen = 7
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16

    attn = Qwen3VLTextAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    ).to(device).eval()
    hidden = torch.randn(1, seqlen, hidden_size, device=device, dtype=dtype)

    with torch.inference_mode():
        full_out = attn(hidden)

        k_cache = torch.empty(1, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.empty_like(k_cache)
        attn.engine_attn.k_cache = k_cache
        attn.engine_attn.v_cache = v_cache
        slot_mapping = torch.arange(seqlen, device=device, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            slot_mapping=slot_mapping,
        )
        engine_out = attn(hidden.squeeze(0))

        expected_k = attn.k_norm(
            attn.k_proj(hidden.squeeze(0)).view(seqlen, num_kv_heads, head_dim)
        )
        expected_v = attn.v_proj(hidden.squeeze(0)).view(seqlen, num_kv_heads, head_dim)

    torch.cuda.synchronize()
    reset_context()

    out_diff = (engine_out - full_out.squeeze(0)).abs()
    k_diff = (k_cache.view(seqlen, num_kv_heads, head_dim) - expected_k).abs()
    v_diff = (v_cache.view(seqlen, num_kv_heads, head_dim) - expected_v).abs()

    print(f"hidden input shape: {list(hidden.shape)}")
    print(f"full output shape: {list(full_out.shape)}")
    print(f"engine output shape: {list(engine_out.shape)}")
    print(f"k_cache shape: {list(k_cache.shape)}")
    print(f"attention output max diff: {out_diff.max().item():.6e}")
    print(f"attention output mean diff: {out_diff.float().mean().item():.6e}")
    print(f"k_cache max diff: {k_diff.max().item():.6e}")
    print(f"v_cache max diff: {v_diff.max().item():.6e}")

    assert list(engine_out.shape) == [seqlen, hidden_size]
    assert out_diff.max().item() < 1e-2
    assert k_diff.max().item() == 0
    assert v_diff.max().item() == 0
    print("engine attention prefill KV: PASS")


def test_engine_mrope_position_ids_three_token_ambiguity():
    """engine flatten `[3, N]` position_ids 在 N=3 时仍应按单序列 VL 处理。"""

    torch.manual_seed(20260624)
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqlen = 3
    hidden_size = 64

    model = Qwen3VLTextModel(
        vocab_size=128,
        hidden_size=hidden_size,
        num_heads=4,
        num_kv_heads=2,
        num_layers=0,
        intermediate_size=128,
        dtype=dtype,
        head_dim=16,
        mrope_section=[2, 3, 3],
    ).to(device).eval()
    hidden = torch.randn(seqlen, hidden_size, device=device, dtype=dtype)
    position_ids = torch.tensor(
        [[0, 1, 2], [0, 2, 4], [0, 3, 6]],
        device=device,
        dtype=torch.long,
    )

    class CaptureRope(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shape: list[int] | None = None

        def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.seen_shape = list(pos.shape)
            return torch.zeros(1, seqlen, 16, device=device, dtype=dtype), torch.zeros(1, seqlen, 16, device=device, dtype=dtype)

    capture = CaptureRope()
    model.rotary_emb = capture
    with torch.inference_mode():
        output = model(inputs_embeds=hidden, position_ids=position_ids)

    print(f"engine position_ids shape: {list(position_ids.shape)}")
    print(f"normalized position_ids shape: {capture.seen_shape}")
    print(f"guard output shape: {list(output.shape)}")

    assert capture.seen_shape == [3, 1, seqlen]
    assert list(output.shape) == [seqlen, hidden_size]
    print("engine M-RoPE ambiguity guard: PASS")


def test_engine_attention_decode_reads_paged_kv_cache():
    """decode fallback 必须从 block_table 指向的 paged KV cache 读取完整历史。"""

    torch.manual_seed(20260624)
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    context_len = 6

    attn = Qwen3VLTextAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    ).to(device).eval()
    hidden = torch.randn(context_len, hidden_size, device=device, dtype=dtype)

    with torch.inference_mode():
        q_all = attn.q_norm(attn.q_proj(hidden).view(context_len, num_heads, head_dim))
        k_all = attn.k_norm(attn.k_proj(hidden).view(context_len, num_kv_heads, head_dim))
        v_all = attn.v_proj(hidden).view(context_len, num_kv_heads, head_dim)

        k_cache = torch.empty(1, context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.empty_like(k_cache)
        k_cache[0, :context_len] = k_all
        v_cache[0, :context_len] = v_all
        attn.engine_attn.k_cache = k_cache
        attn.engine_attn.v_cache = v_cache

        q_last = q_all[-1:].contiguous()
        set_context(
            False,
            slot_mapping=torch.tensor([context_len - 1], device=device, dtype=torch.int32),
            context_lens=torch.tensor([context_len], device=device, dtype=torch.int32),
            block_tables=torch.tensor([[0]], device=device, dtype=torch.int32),
        )
        engine_o = attn.engine_attn(q_last, k_all[-1:].contiguous(), v_all[-1:].contiguous())

        keys = k_all.repeat_interleave(num_heads // num_kv_heads, dim=1)
        values = v_all.repeat_interleave(num_heads // num_kv_heads, dim=1)
        q_ref = q_last.unsqueeze(0).transpose(1, 2)
        k_ref = keys.unsqueeze(0).transpose(1, 2)
        v_ref = values.unsqueeze(0).transpose(1, 2)
        ref_o = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            is_causal=False,
            scale=attn.scale,
        ).transpose(1, 2).squeeze(0)

    reset_context()
    diff = (engine_o - ref_o).abs()
    print(f"decode q shape: {list(q_last.shape)}")
    print(f"decode engine output shape: {list(engine_o.shape)}")
    print(f"decode reference output shape: {list(ref_o.shape)}")
    print(f"decode engine mean/std: {engine_o.float().mean().item():.6e} / {engine_o.float().std().item():.6e}")
    print(f"decode reference mean/std: {ref_o.float().mean().item():.6e} / {ref_o.float().std().item():.6e}")
    print(f"decode output max diff: {diff.max().item():.6e}")
    print(f"decode output mean diff: {diff.float().mean().item():.6e}")

    assert list(engine_o.shape) == [1, num_heads, head_dim]
    assert diff.max().item() < 1e-2
    print("engine attention decode paged KV kernel: PASS")
