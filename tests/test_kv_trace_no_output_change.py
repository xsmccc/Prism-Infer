"""P4 KV trace on/off 不改变 attention 输出验证。"""

import torch

from prism_infer.analysis.kv_trace import kv_trace
from prism_infer.layers.attention import Attention
from prism_infer.utils.context import reset_context, set_context


def _run_prefill_attention(enable_trace: bool):
    torch.manual_seed(20260626)
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
    attn.layer_idx = 0
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32)
    metadata = None
    session = None
    if enable_trace:
        from prism_infer.analysis.kv_trace import build_trace_metadata
        from prism_infer.engine.sequence import Sequence
        from prism_infer.sampling_params import SamplingParams

        seq = Sequence(
            [1, 99, 99, 2, 3],
            SamplingParams(temperature=0.0, max_tokens=1),
            image_token_id=99,
            image_token_count=2,
        )
        seq.block_table = [0]
        trace_cm = kv_trace(metadata={"case": "unit_prefill"}, top_k_tokens=3)
        session = trace_cm.__enter__()
        metadata = build_trace_metadata(
            [seq],
            is_prefill=True,
            input_ids=torch.arange(seqlen, dtype=torch.long),
            position_ids=torch.arange(seqlen, dtype=torch.long),
            slot_mapping=torch.arange(seqlen, dtype=torch.int32),
            block_tables=None,
            context_lens=None,
            block_size=256,
        )
    else:
        trace_cm = None

    try:
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            slot_mapping=torch.arange(seqlen, dtype=torch.int32),
            trace_metadata=metadata,
        )
        with torch.inference_mode():
            output = attn(q, k, v)
    finally:
        reset_context()
        if trace_cm is not None:
            trace_cm.__exit__(None, None, None)
    return output, session


def _run_decode_attention(enable_trace: bool):
    torch.manual_seed(20260626)
    dtype = torch.float32
    context_len = 5
    block_size = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    q = torch.randn(1, num_heads, head_dim, dtype=dtype)
    k_cache_values = torch.randn(context_len, num_kv_heads, head_dim, dtype=dtype)
    v_cache_values = torch.randn(context_len, num_kv_heads, head_dim, dtype=dtype)
    k_cache = torch.zeros(1, block_size, num_kv_heads, head_dim, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    k_cache[0, :context_len] = k_cache_values
    v_cache[0, :context_len] = v_cache_values
    k_current = k_cache_values[-1:].contiguous()
    v_current = v_cache_values[-1:].contiguous()

    attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
    )
    attn.layer_idx = 0
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    metadata = None
    session = None
    if enable_trace:
        from prism_infer.analysis.kv_trace import build_trace_metadata
        from prism_infer.engine.sequence import Sequence
        from prism_infer.sampling_params import SamplingParams

        seq = Sequence(
            [1, 99, 99, 2, 3],
            SamplingParams(temperature=0.0, max_tokens=1),
            image_token_id=99,
            image_token_count=2,
        )
        seq.block_table = [0]
        seq.num_tokens = context_len
        trace_cm = kv_trace(metadata={"case": "unit_decode"}, top_k_tokens=3)
        session = trace_cm.__enter__()
        metadata = build_trace_metadata(
            [seq],
            is_prefill=False,
            input_ids=torch.tensor([seq.last_token], dtype=torch.long),
            position_ids=torch.tensor([context_len - 1], dtype=torch.long),
            slot_mapping=torch.tensor([-1], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            context_lens=torch.tensor([context_len], dtype=torch.int32),
            block_size=block_size,
        )
    else:
        trace_cm = None

    try:
        set_context(
            False,
            slot_mapping=torch.tensor([-1], dtype=torch.int32),
            context_lens=torch.tensor([context_len], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            trace_metadata=metadata,
        )
        with torch.inference_mode():
            output = attn(q, k_current, v_current)
    finally:
        reset_context()
        if trace_cm is not None:
            trace_cm.__exit__(None, None, None)
    return output, session


def test_kv_trace_prefill_on_off_output_identical():
    """开启 KV trace 不应改变 decode attention 输出。"""

    output_off, _ = _run_decode_attention(enable_trace=False)
    output_on, session = _run_decode_attention(enable_trace=True)
    diff = (output_on - output_off).abs()

    assert session is not None
    records = session.records
    record = records[0]
    attention = record["attention"]["sequence_stats"][0]

    print(f"trace off output shape: {list(output_off.shape)}")
    print(f"trace on output shape: {list(output_on.shape)}")
    print(f"trace output max diff: {diff.max().item():.6e}")
    print(f"trace output mean diff: {diff.mean().item():.6e}")
    print(f"trace records: {len(records)}")
    print(f"trace q shape: {record['tensor_stats']['q']['shape']}")
    print(f"trace visual attention mass: {attention['visual_mass_mean']:.6e}")

    assert list(output_on.shape) == [1, 2, 4]
    assert diff.max().item() == 0.0
    assert len(records) == 1
    assert record["record_type"] == "attention_layer"
    assert record["batch"]["sequences"][0]["spans"][1]["modality"] == "image"
    assert 0.0 <= attention["visual_mass_mean"] <= 1.0
    print("KV trace prefill on/off equality: PASS")
