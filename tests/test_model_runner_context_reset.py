"""ModelRunner 异常路径上下文清理回归测试。"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.contracts import DeviceModelInputs, PreparedModelInputs
from prism_infer.engine.sequence import Sequence
from prism_infer.utils.context import Context, get_context, reset_context


def _runner(enable_chunked_prefill: bool = False) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        enable_chunked_prefill=enable_chunked_prefill,
        max_chunk_size=2,
        execution_backend="eager",
    )
    runner.rank = 0
    return runner


def test_model_runner_run_resets_context_when_forward_raises() -> None:
    """run_model 抛异常时必须 reset_context，避免下一轮 attention 读到脏状态。"""

    runner = _runner(enable_chunked_prefill=False)
    seq = Sequence([1, 2, 3], block_size=4, request_id=0)

    def prepare_prefill(self, seqs, *, prefill_slices):
        return PreparedModelInputs(
            model_inputs=DeviceModelInputs(
                input_ids=torch.tensor([1]),
                position_ids=torch.tensor([0]),
            ),
            attention_context=Context(
                is_prefill=True,
                slot_mapping=torch.tensor([7], dtype=torch.int32),
            ),
        )

    def prepare_sample(self, seqs):
        return torch.tensor([1.0])

    def run_model_eager(self, model_inputs, *, is_prefill):
        raise RuntimeError("synthetic forward failure")

    runner._prepare_prefill_batch = MethodType(prepare_prefill, runner)
    runner.prepare_sample = MethodType(prepare_sample, runner)
    runner.run_model_eager = MethodType(run_model_eager, runner)

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runner.run([seq], True)

    assert get_context().slot_mapping is None
    print("ModelRunner context reset on exception: PASS")


def test_model_runner_run_never_mutates_chunked_sequence_during_prepare() -> None:
    """Chunk preparation receives an immutable range without disguising Sequence."""

    runner = _runner(enable_chunked_prefill=True)
    seq = Sequence([1, 2, 3, 4, 5], block_size=4, request_id=0)
    seq.block_table = [0]
    seq.num_computed_tokens = 0
    seq.num_cached_tokens = 0
    original_num_tokens = seq.num_tokens
    original_token_ids = list(seq.token_ids)
    original_cached = seq.num_cached_tokens

    observed_slices = []

    def prepare_prefill(self, seqs, *, prefill_slices):
        observed_slices.extend(prefill_slices)
        assert seqs[0].num_tokens == original_num_tokens
        assert seqs[0].token_ids == original_token_ids
        assert seqs[0].num_cached_tokens == original_cached
        raise RuntimeError("synthetic prepare failure")

    runner._prepare_prefill_batch = MethodType(prepare_prefill, runner)

    with pytest.raises(RuntimeError, match="synthetic prepare failure"):
        runner.run([seq], True)

    assert seq.num_tokens == original_num_tokens
    assert seq.token_ids == original_token_ids
    assert seq.num_cached_tokens == original_cached
    assert not hasattr(seq, "_orig_num_tokens")
    assert len(observed_slices) == 1
    assert observed_slices[0].token_start == 0
    assert observed_slices[0].token_end == 2
    assert get_context().slot_mapping is None
    reset_context()
    print("ModelRunner chunked state restore on exception: PASS")


def test_model_runner_chunk_progress_does_not_become_prefix_hit_state() -> None:
    """Chunk progress and shared prefix-cache hits are separate contracts."""

    runner = _runner(enable_chunked_prefill=True)
    seq = Sequence([1, 2, 3], block_size=4, request_id=0)
    seq.block_table = [0]

    observed_ranges = []

    def prepare_prefill(self, seqs, *, prefill_slices):
        observed_ranges.append((prefill_slices[0].token_start, prefill_slices[0].token_end))
        return PreparedModelInputs(
            model_inputs=DeviceModelInputs(
                input_ids=torch.tensor([1]),
                position_ids=torch.tensor([0]),
            ),
            attention_context=Context(is_prefill=True),
        )

    def prepare_sample(self, seqs):
        return torch.tensor([0.0])

    def run_model_eager(self, model_inputs, *, is_prefill):
        return torch.tensor([[1.0]])

    runner._prepare_prefill_batch = MethodType(prepare_prefill, runner)
    runner.prepare_sample = MethodType(prepare_sample, runner)
    runner.run_model_eager = MethodType(run_model_eager, runner)
    runner.sampler = lambda logits, temperatures: torch.tensor([7])

    first = runner.run([seq], True, [2])
    assert first == [None]
    assert seq.num_computed_tokens == 2
    assert seq.num_cached_tokens == 0

    second = runner.run([seq], True, [1])
    assert second == [7]
    assert seq.num_computed_tokens == 3
    assert seq.num_cached_tokens == 0
    assert observed_ranges == [(0, 2), (2, 3)]
    print("chunk progress/prefix-hit separation: PASS")
