"""ModelRunner 异常路径上下文清理回归测试。"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.sequence import Sequence
from prism_infer.utils.context import get_context, reset_context, set_context


def _runner(enable_chunked_prefill: bool = False) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        enable_chunked_prefill=enable_chunked_prefill,
        max_chunk_size=2,
    )
    runner.rank = 0
    return runner


def test_model_runner_run_resets_context_when_forward_raises() -> None:
    """run_model 抛异常时必须 reset_context，避免下一轮 attention 读到脏状态。"""

    runner = _runner(enable_chunked_prefill=False)
    seq = Sequence([1, 2, 3])

    def prepare_prefill(self, seqs):
        set_context(True, slot_mapping=torch.tensor([7], dtype=torch.int32))
        return SimpleNamespace(input_ids=torch.tensor([1]), position_ids=torch.tensor([0]))

    def prepare_sample(self, seqs):
        return torch.tensor([1.0])

    def run_model(self, model_inputs, is_prefill):
        raise RuntimeError("synthetic forward failure")

    runner.prepare_prefill = MethodType(prepare_prefill, runner)
    runner.prepare_sample = MethodType(prepare_sample, runner)
    runner.run_model = MethodType(run_model, runner)

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runner.run([seq], True)

    assert get_context().slot_mapping is None
    print("ModelRunner context reset on exception: PASS")


def test_model_runner_run_restores_chunked_sequence_when_prepare_raises() -> None:
    """chunked prefill 准备阶段异常时必须恢复 Sequence 临时状态。"""

    runner = _runner(enable_chunked_prefill=True)
    seq = Sequence([1, 2, 3, 4, 5])
    seq.block_table = [0]
    seq.num_computed_tokens = 0
    seq.num_cached_tokens = 0
    original_num_tokens = seq.num_tokens
    original_token_ids = list(seq.token_ids)
    original_cached = seq.num_cached_tokens

    def prepare_prefill(self, seqs):
        set_context(True, slot_mapping=torch.tensor([3], dtype=torch.int32))
        raise RuntimeError("synthetic prepare failure")

    runner.prepare_prefill = MethodType(prepare_prefill, runner)

    with pytest.raises(RuntimeError, match="synthetic prepare failure"):
        runner.run([seq], True)

    assert seq.num_tokens == original_num_tokens
    assert seq.token_ids == original_token_ids
    assert seq.num_cached_tokens == original_cached
    assert not hasattr(seq, "_orig_num_tokens")
    assert get_context().slot_mapping is None
    reset_context()
    print("ModelRunner chunked state restore on exception: PASS")
