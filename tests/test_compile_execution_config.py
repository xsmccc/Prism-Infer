"""P6.3 attention-only compile 配置与 decode dispatch 测试。"""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.config import Config
from prism_infer.models.qwen3_vl import Qwen3VLTextAttention
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


def test_p611_config_rejects_logical_prune_cuda_graph(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """logical retained-slot metadata 不能静默绕过用户请求的 CUDA Graph。"""

    _patch_auto_config(monkeypatch)
    with pytest.raises(ValueError, match="requires enforce_eager=True"):
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
    ("fp8_kv", "visual_compact", "visual_compact_fp8"),
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


def test_attention_compile_dispatch_is_decode_only() -> None:
    attention = Qwen3VLTextAttention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
    )
    attention._compiled_decode_forward = (
        lambda hidden, _positions: hidden + 1
    )
    attention._forward_engine = lambda hidden, _positions: hidden + 2
    hidden_states = torch.zeros(1, 8)
    slot_mapping = torch.tensor([0], dtype=torch.int32)

    try:
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=torch.tensor([1], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
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
