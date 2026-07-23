"""P6.3 attention-only compile 配置与 decode dispatch 测试。"""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.config import Config
from prism_infer.models.qwen3_vl import Qwen3VLTextAttention
from prism_infer.ops.kv_cache_store import HAS_TRITON as HAS_STORE_TRITON
from prism_infer.ops.paged_decode import HAS_TRITON as HAS_PAGED_DECODE_TRITON
from prism_infer.utils.context import reset_context, set_context


def _patch_auto_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """避免配置 contract 测试访问外部模型仓库。"""

    config = SimpleNamespace(
        max_position_embeddings=1024,
        text_config=SimpleNamespace(max_position_embeddings=1024),
    )
    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda _path: config,
    )


def test_attention_compile_config_requires_eager_off_baseline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auto_config(monkeypatch)
    valid = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        compression_mode="off",
        decode_compile_region="attention",
        allow_unsafe_decode_compile=True,
    )
    assert valid.decode_compile_region == "attention"

    with pytest.raises(ValueError, match="mutually exclusive"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            enforce_eager=False,
            compression_mode="off",
            decode_compile_region="attention",
            allow_unsafe_decode_compile=True,
        )
    with pytest.raises(ValueError, match="compression_mode='off'"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            enforce_eager=True,
            compression_mode="fp8_kv",
            decode_compile_region="attention",
            allow_unsafe_decode_compile=True,
        )
    with pytest.raises(ValueError, match="rejected P6.3 preflight candidate"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            enforce_eager=True,
            compression_mode="off",
            decode_compile_region="attention",
        )
    print("P6.3 compile/Graph/compression config isolation: PASS")


def test_compile_graph_config_uses_supported_stateless_region(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auto_config(monkeypatch)
    config = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        execution_backend="compile_graph",
        compression_mode="off",
        decode_compile_region="stateless",
    )

    assert config.enforce_eager is False
    assert config.decode_compile_region == "stateless"
    assert config.allow_unsafe_decode_compile is False

    with pytest.raises(ValueError, match="compile_graph.*requires.*stateless"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            execution_backend="compile_graph",
            compression_mode="off",
            decode_compile_region="attention",
        )


@pytest.mark.parametrize(
    "compression_mode",
    ("scaled_fp8_kv", "visual_compact_scaled_fp8"),
)
def test_compile_graph_config_allows_graph_safe_scaled_kv(
    compression_mode: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stateless compiled projections do not own or functionalize KV state."""

    _patch_auto_config(monkeypatch)
    config = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        execution_backend="compile_graph",
        compression_mode=compression_mode,
        decode_compile_region="stateless",
    )

    assert config.execution_backend == "compile_graph"
    assert config.compression_mode == compression_mode
    assert config.decode_compile_region == "stateless"


def test_block4_gate_up_requires_explicit_graph_backend(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SM120 decode kernel must never look enabled on an inactive backend."""

    _patch_auto_config(monkeypatch)
    valid = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        execution_backend="compile_graph",
        decode_compile_region="stateless",
        mlp_projection_mode="packed",
        enable_decode_block4_gate_up=True,
    )
    assert valid.enable_decode_block4_gate_up is True

    with pytest.raises(ValueError, match="requires a CUDA Graph backend"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            execution_backend="eager",
            mlp_projection_mode="packed",
            enable_decode_block4_gate_up=True,
        )


def test_p611_config_rejects_logical_prune_cuda_graph(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """logical retained-slot metadata 不能静默绕过用户请求的 CUDA Graph。"""

    _patch_auto_config(monkeypatch)
    with pytest.raises(ValueError, match="requires execution backend 'eager'"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            enforce_eager=False,
            compression_mode="visual_prune",
        )
    print("P6.11 logical visual-prune CUDA Graph rejection: PASS")


@pytest.mark.parametrize(
    "compression_mode",
    (
        "fp8_kv",
        "scaled_fp8_kv",
        "visual_compact",
        "visual_compact_fp8",
        "visual_compact_scaled_fp8",
    ),
)
def test_p611_config_allows_physical_compression_cuda_graph(
    compression_mode: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """物理 KV dtype/layout 模式必须允许显式 CUDA Graph execution。"""

    _patch_auto_config(monkeypatch)
    config = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        enforce_eager=False,
        compression_mode=compression_mode,
    )
    assert config.compression_mode == compression_mode
    print(f"P6.11 physical compression Graph config={compression_mode}: PASS")


def test_p74_logits_precision_is_explicit_and_validated(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型原生 logits 是默认路径，fp32 只保留历史复现能力。"""

    _patch_auto_config(monkeypatch)
    default = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )
    candidate = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        logits_precision="fp32",
    )
    assert default.logits_precision == "model"
    assert candidate.logits_precision == "fp32"

    with pytest.raises(ValueError, match="logits_precision"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            logits_precision="int8",
        )


def test_p75_mlp_projection_mode_is_explicit_and_validated(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auto_config(monkeypatch)
    default = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )
    legacy = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        mlp_projection_mode="legacy",
    )
    assert default.mlp_projection_mode == "packed"
    assert legacy.mlp_projection_mode == "legacy"

    with pytest.raises(ValueError, match="mlp_projection_mode"):
        Config(
            str(tmp_path),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            mlp_projection_mode="auto",
        )


def test_attention_compile_dispatch_is_decode_only() -> None:
    attention = Qwen3VLTextAttention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
    )
    attention._compiled_decode_qkv_forward = lambda hidden, _positions: (
        hidden + 1,
        hidden,
        hidden,
    )
    attention.engine_attn.forward_decode_explicit = lambda query, *_args: query
    attention._project_engine_output = lambda output: output
    attention._forward_engine = lambda hidden, _positions: hidden + 2
    hidden_states = torch.zeros(1, 8)
    slot_mapping = torch.tensor([0], dtype=torch.int32)

    try:
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=torch.tensor([1], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            decode_max_context_len=torch.tensor(1, dtype=torch.int32),
        )
        decode_output = attention(hidden_states)
        reset_context()
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=1,
            slot_mapping=slot_mapping,
        )
        prefill_output = attention(hidden_states)
    finally:
        reset_context()

    assert torch.equal(decode_output, hidden_states + 1)
    assert torch.equal(prefill_output, hidden_states + 2)
    print("P6.3 attention compile decode-only dispatch: PASS")


def test_attention_compile_merges_mode_and_precision_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = Qwen3VLTextAttention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
    )
    captured: dict[str, object] = {}

    def fake_compile(function: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        torch._inductor,
        "list_mode_options",
        lambda mode: {"triton.cudagraphs": mode == "reduce-overhead"},
    )
    attention.enable_decode_compile(
        mode="reduce-overhead",
        emulate_precision_casts=True,
        force_same_precision=True,
    )

    assert "mode" not in captured
    assert captured["options"] == {
        "triton.cudagraphs": True,
        "emulate_precision_casts": True,
        "force_same_precision": True,
    }
    print(f"P6.3 merged compile options: {captured['options']} PASS")


@pytest.mark.gpu
def test_compile_qkv_split_handles_nonzero_offset_cache_views() -> None:
    """A compiled pure region must not functionalize aliased KV-cache views."""

    if not torch.cuda.is_available() or not HAS_STORE_TRITON or not HAS_PAGED_DECODE_TRITON:
        pytest.skip("CUDA and Triton are required for compile primitive coverage")

    torch.manual_seed(20260719)
    device = torch.device("cuda:0")
    hidden_size = 64
    batch = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    page_size = 16
    num_blocks = 2
    attention = Qwen3VLTextAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )
    attention = attention.to(device)
    hidden_states = torch.randn(batch, hidden_size, device=device, dtype=torch.bfloat16)
    cos = torch.randn(1, batch, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    initial_cache = torch.randn(
        2,
        2,
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    slot_mapping = torch.tensor([4, 13], device=device, dtype=torch.int32)
    context_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
    block_tables = torch.tensor([[0, -1], [1, -1]], device=device, dtype=torch.int32)
    max_context_len = torch.tensor(6, device=device, dtype=torch.int32)

    try:
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            decode_max_context_len=max_context_len,
        )
        reference_cache = initial_cache.clone()
        attention.engine_attn.k_cache = reference_cache[0, 1]
        attention.engine_attn.v_cache = reference_cache[1, 1]
        reference = attention(hidden_states, (cos, sin))

        actual_cache = initial_cache.clone()
        attention.engine_attn.k_cache = actual_cache[0, 1]
        attention.engine_attn.v_cache = actual_cache[1, 1]
        attention.enable_decode_compile(
            mode="default",
            emulate_precision_casts=True,
            force_same_precision=True,
        )
        actual = attention(hidden_states, (cos, sin))
        torch.cuda.synchronize()
    finally:
        reset_context()

    assert attention.engine_attn.k_cache.storage_offset() > 0
    assert attention.engine_attn.v_cache.storage_offset() > 0
    torch.testing.assert_close(actual_cache, reference_cache, rtol=0.0, atol=0.0)
    # The legacy attention-only compile path is explicitly opt-in because
    # Inductor may change BF16 reduction order.  This test protects the aliased
    # cache contract; numerical equivalence is sufficient for its output.
    torch.testing.assert_close(actual, reference, rtol=0.1, atol=0.002)
