"""CPU metadata tests; no FlashInfer installation or CUDA execution required."""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.model_runner import ModelRunner
from prism_infer.layers import attention as attention_module
from prism_infer.layers.attention import Attention
from prism_infer.ops.flashinfer_paged import (
    FlashInferPagedDecode,
    build_flashinfer_decode_plan_inputs,
)


class _FakeDecodeWrapper:
    def __init__(self):
        self.plans = []

    def plan(self, **kwargs):
        self.plans.append(
            {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
        )


def _decode_adapter(max_batch, max_blocks, *, block_size=4):
    adapter = FlashInferPagedDecode.__new__(FlashInferPagedDecode)
    adapter.block_size = block_size
    adapter.max_batch = max_batch
    adapter.max_blocks = max_blocks
    adapter.num_qo_heads = 4
    adapter.num_kv_heads = 2
    adapter.head_dim = 8
    adapter.dtype = torch.bfloat16
    adapter.indptr_buf = torch.zeros(max_batch + 1, dtype=torch.int32)
    adapter.indices_buf = torch.full((max_batch * max_blocks,), -1, dtype=torch.int32)
    adapter.last_page_len_buf = torch.zeros(max_batch, dtype=torch.int32)
    adapter._wrapper = _FakeDecodeWrapper()
    return adapter


def _planned_pages(plan):
    indptr = plan["indptr"].tolist()
    indices = plan["indices"].tolist()
    return [indices[start:end] for start, end in zip(indptr, indptr[1:], strict=False)]


def test_unequal_decode_lengths_pack_only_valid_pages():
    lengths = torch.tensor([4, 10], dtype=torch.int32)
    tables = torch.tensor([[7, -1, -1], [8, 9, 10]], dtype=torch.int32)
    indptr, indices, last_page_len = build_flashinfer_decode_plan_inputs(
        context_lens=lengths, block_tables=tables, block_size=4
    )
    assert indptr.tolist() == [0, 1, 4]
    assert indices.tolist() == [7, 8, 9, 10]
    assert last_page_len.tolist() == [4, 2]

    adapter = _decode_adapter(2, 3)
    adapter.stage_plan_inputs(lengths, tables)
    plan = adapter._wrapper.plans[-1]
    assert _planned_pages(plan) == [[7], [8, 9, 10]]
    assert plan["last_page_len"].tolist() == [4, 2]


def test_same_shape_decode_crossing_page_boundary_replans_with_stable_buffers():
    adapter = _decode_adapter(2, 3)
    buffers = (adapter.indptr_buf, adapter.indices_buf, adapter.last_page_len_buf)
    addresses = [buffer.data_ptr() for buffer in buffers]
    adapter.stage_plan_inputs(
        torch.tensor([4, 9], dtype=torch.int32),
        torch.tensor([[7, -1, -1], [8, 9, 10]], dtype=torch.int32),
    )
    adapter.stage_plan_inputs(
        torch.tensor([5, 10], dtype=torch.int32),
        torch.tensor([[7, 11, -1], [8, 9, 10]], dtype=torch.int32),
    )
    assert len(adapter._wrapper.plans) == 2
    before, after = adapter._wrapper.plans
    assert before["indptr"].tolist() == [0, 1, 4]
    assert after["indptr"].tolist() == [0, 2, 5]
    assert _planned_pages(after) == [[7, 11], [8, 9, 10]]
    assert after["last_page_len"].tolist() == [1, 2]
    assert [
        buffer.data_ptr()
        for buffer in (adapter.indptr_buf, adapter.indices_buf, adapter.last_page_len_buf)
    ] == addresses


def test_graph_padding_stages_captured_bucket_without_mutating_context():
    actual_wrapper = _decode_adapter(9, 3)
    captured_wrapper = _decode_adapter(16, 3)
    attention = SimpleNamespace(
        flashinfer_decode_enabled=True,
        _flashinfer_decode_wrappers={9: actual_wrapper, 16: captured_wrapper},
    )
    context = SimpleNamespace(
        context_lens=torch.tensor([4] * 8 + [10], dtype=torch.int32),
        block_tables=torch.tensor([[7, -1, -1]] * 8 + [[8, 9, 10]], dtype=torch.int32),
        slot_mapping=torch.arange(9, dtype=torch.int32),
    )
    original_tensors = {name: value.clone() for name, value in vars(context).items()}
    Attention.stage_flashinfer_decode_metadata(attention, context, captured_batch_size=16)

    assert actual_wrapper._wrapper.plans == []
    plan = captured_wrapper._wrapper.plans[-1]
    assert _planned_pages(plan) == [[7]] * 8 + [[8, 9, 10]] + [[0]] * 7
    assert plan["last_page_len"].tolist() == [4] * 8 + [2] + [1] * 7
    for name, value in original_tensors.items():
        assert torch.equal(getattr(context, name), value)

    # A subsequent full bucket must replace the dummy rows rather than retain them.
    captured_wrapper.stage_plan_inputs(
        torch.full((16,), 5, dtype=torch.int32),
        torch.tensor([[12, 13, -1]] * 16, dtype=torch.int32),
    )
    assert len(captured_wrapper._wrapper.plans) == 2
    assert _planned_pages(captured_wrapper._wrapper.plans[-1]) == [[12, 13]] * 16


def test_eager_decode_grows_wrapper_when_page_table_width_increases(monkeypatch):
    wrappers = []

    class Wrapper:
        def __init__(self, **kwargs):
            self.max_blocks = kwargs["max_blocks"]
            wrappers.append(self)

        def stage_plan_inputs(self, lengths, tables):
            assert tables.shape[1] <= self.max_blocks

        def run(self, q, k_cache, v_cache):
            return q

    monkeypatch.setattr(attention_module, "FlashInferPagedDecode", Wrapper)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attention = Attention(4, 8, 8**-0.5, 2)
    attention.k_cache = attention.v_cache = torch.zeros(4, 4, 2, 8, dtype=torch.bfloat16)
    q = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    for table in ([[0]], [[0, 1]]):
        context = SimpleNamespace(
            context_lens=torch.tensor([len(table[0]) * 4]),
            block_tables=torch.tensor(table, dtype=torch.int32),
        )
        assert attention._forward_decode_paged_flashinfer(q, context) is q
    assert [wrapper.max_blocks for wrapper in wrappers] == [1, 2]


@pytest.mark.parametrize(
    "available, dtype, message",
    [
        (False, torch.bfloat16, "unavailable"),
        (True, torch.float8_e4m3fn, "unscaled BF16/FP16"),
    ],
)
def test_explicit_flashinfer_request_does_not_silently_choose_another_backend(
    monkeypatch, available, dtype, message
):
    monkeypatch.setattr(attention_module, "HAS_FLASHINFER", available)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        enable_flashinfer_paged=False, enable_flashinfer_decode=True, decode_compile_region="none"
    )
    runner.kv_cache_dtype = dtype
    runner._attention_layers = lambda: pytest.fail("invalid backend reached binding")
    with pytest.raises((RuntimeError, ValueError), match=message):
        runner._configure_flashinfer_paged()


def test_flashinfer_decode_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(attention_module, "HAS_FLASHINFER", True)
    attention = Attention(4, 8, 8**-0.5, 2)
    attention.flashinfer_decode_enabled = True
    attention.k_cache = attention.v_cache = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)

    def fail(*args):
        raise RuntimeError("native decode failure")

    attention._forward_decode_paged_flashinfer = fail
    with pytest.raises(RuntimeError, match="native decode failure"):
        attention._forward_decode_paged(SimpleNamespace(is_cuda=True), SimpleNamespace())
