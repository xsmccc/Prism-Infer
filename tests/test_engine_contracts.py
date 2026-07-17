"""P7.2 engine boundary and lifecycle contract tests."""

from dataclasses import FrozenInstanceError
from contextlib import contextmanager
from collections.abc import Iterator
from itertools import count
from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    ExecutionResult,
    KVTransferPlan,
    RequestOutput,
)
from prism_infer.engine.executor import ModelExecutor
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.request import RequestState
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.scheduler_policy import FCFSSchedulerPolicy
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


_REQUEST_IDS = count()


def _sequence(
    token_ids: list[int],
    sampling_params: SamplingParams | None = None,
    **kwargs: object,
) -> Sequence:
    return Sequence(
        token_ids,
        sampling_params,
        block_size=4,
        request_id=next(_REQUEST_IDS),
        **kwargs,
    )


@contextmanager
def _explicit_page_contract() -> Iterator[None]:
    assert not hasattr(Sequence, "block_size")
    assert not hasattr(Sequence, "set_block_size")
    yield
    assert not hasattr(Sequence, "block_size")
    assert not hasattr(Sequence, "set_block_size")


def _scheduler_config(**overrides):
    values = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 16,
        "max_model_len": 32,
        "enable_chunked_prefill": False,
        "max_chunk_size": 4,
        "max_queue_size": None,
        "max_consecutive_prefill_batches": 1,
        "eos": -1,
        "num_kvcache_blocks": 16,
        "kvcache_block_size": 4,
        "num_cpu_blocks": 4,
        "enable_prefix_caching": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_request_fsm_records_valid_transitions_and_rejects_invalid() -> None:
    seq = _sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    seq.transition_to(RequestState.PREFILLING, reason="scheduled")
    seq.transition_to(RequestState.DECODING, reason="first token")
    seq.transition_to(RequestState.FINISHED, reason="length")

    assert seq.status is RequestState.FINISHED
    assert [transition.target for transition in seq.lifecycle.transitions] == [
        RequestState.PREFILLING,
        RequestState.DECODING,
        RequestState.FINISHED,
    ]
    with pytest.raises(RuntimeError, match="invalid request state transition"):
        seq.transition_to(RequestState.WAITING, reason="illegal resurrection")

    invalid = _sequence([3])
    with pytest.raises(RuntimeError, match="WAITING->FINISHED"):
        invalid.transition_to(RequestState.FINISHED, reason="skip execution")


def test_batch_plan_is_immutable_and_keeps_legacy_adapter() -> None:
    seq = _sequence([1, 2])
    plan = BatchPlan(
        phase=BatchPhase.PREFILL,
        sequences=(seq,),
        scheduled_token_counts=(2,),
        policy_name="test",
    )

    seqs, is_prefill, cow, swap_in, swap_out = plan
    assert seqs == [seq]
    assert is_prefill
    assert cow == swap_in == swap_out == []
    assert plan.num_scheduled_tokens == 2
    with pytest.raises(FrozenInstanceError):
        plan.phase = BatchPhase.DECODE


def test_fcfs_policy_admission_and_chunk_budget_are_pure() -> None:
    policy = FCFSSchedulerPolicy(
        max_model_len=8,
        max_num_batched_tokens=8,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        max_chunk_size=3,
        max_queue_size=1,
    )
    valid = _sequence(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=2),
    )
    too_long = _sequence(
        [1, 2, 3, 4, 5, 6, 7],
        SamplingParams(max_tokens=2),
    )

    assert policy.admit(valid, queued_requests=0).accepted
    assert not policy.admit(valid, queued_requests=1).accepted
    assert not policy.admit(too_long, queued_requests=0).accepted
    assert policy.prefill_token_count(valid, available_tokens=8) == 3
    assert policy.prefill_token_count(valid, available_tokens=2) == 2

    visual = _sequence(
        [1, 99, 99, 2, 99, 99, 3],
        SamplingParams(max_tokens=1),
        video_token_id=99,
        video_token_count=4,
    )
    visual_policy = FCFSSchedulerPolicy(
        max_model_len=8,
        max_num_batched_tokens=8,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        max_chunk_size=5,
    )
    assert visual_policy.prefill_token_count(
        visual, available_tokens=4
    ) == 1
    visual.num_computed_tokens = 1
    assert visual_policy.prefill_token_count(
        visual, available_tokens=5
    ) == 5


def test_scheduler_emits_named_plan_and_advances_fsm() -> None:
    with _explicit_page_contract():
        scheduler = Scheduler(_scheduler_config())
        seq = _sequence(
            [1, 2, 3],
            SamplingParams(temperature=0.0, max_tokens=2),
        )
        scheduler.add(seq)
        plan = scheduler.schedule()

        assert plan.phase is BatchPhase.PREFILL
        assert plan.sequences == (seq,)
        assert plan.scheduled_token_counts == (3,)
        assert seq.status is RequestState.PREFILLING

        outputs = scheduler.postprocess(plan, [9])
        assert outputs == ()
        assert seq.status is RequestState.DECODING
        decode = scheduler.schedule()
        assert decode.phase is BatchPhase.DECODE
        finished = scheduler.postprocess(decode, [10])
        assert finished[0].request_id == seq.seq_id
        assert finished[0].finish_reason == "length"
        assert scheduler.is_finished()


def test_scheduler_admission_rejection_and_swapped_cancel_are_terminal() -> None:
    with _explicit_page_contract():
        scheduler = Scheduler(
            _scheduler_config(
                max_model_len=4,
                num_kvcache_blocks=2,
                num_cpu_blocks=2,
            )
        )
        rejected = _sequence(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=1),
        )
        decision = scheduler.add(rejected, raise_on_reject=False)
        assert not decision.accepted
        assert rejected.status is RequestState.REJECTED

        active = _sequence(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=1),
        )
        scheduler.block_manager.allocate(active)
        scheduler.block_manager.swap_out(active)
        active.status = RequestState.SWAPPED
        scheduler.swapped.append(active)
        assert len(scheduler.block_manager.cpu_free_block_ids) == 1

        assert scheduler.cancel(active.seq_id)
        assert active.status is RequestState.CANCELLED
        assert len(scheduler.block_manager.cpu_free_block_ids) == 2
        assert not scheduler.cancel(active.seq_id)


def test_online_prefix_hit_prefill_uses_remaining_token_budget() -> None:
    with _explicit_page_contract():
        scheduler = Scheduler(
            _scheduler_config(
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_chunk_size=8,
            )
        )
        sampling = SamplingParams(
            temperature=0.0, max_tokens=3, ignore_eos=True
        )
        first = _sequence([1, 2, 3, 4, 5], sampling)
        scheduler.add(first)
        first_prefill = scheduler.schedule()
        scheduler.postprocess(first_prefill, [9])

        second = _sequence([1, 2, 3, 4, 5], sampling)
        scheduler.add(second)
        # Fair interleave gives the existing decoder one turn before new prefill.
        decode = scheduler.schedule()
        assert decode.phase is BatchPhase.DECODE
        scheduler.postprocess(decode, [10])
        second_prefill = scheduler.schedule()

        assert second_prefill.phase is BatchPhase.PREFILL
        assert second_prefill.sequences == (second,)
        assert second.num_cached_tokens == 4
        assert second.num_computed_tokens == 4
        assert second_prefill.scheduled_token_counts == (1,)
        assert second.block_table[0] == first.block_table[0]


def test_scheduler_swap_preemption_round_trip_is_measured() -> None:
    with _explicit_page_contract():
        scheduler = Scheduler(
            _scheduler_config(
                num_kvcache_blocks=2,
                num_cpu_blocks=2,
                max_num_seqs=2,
            )
        )
        sampling = SamplingParams(
            temperature=0.0, max_tokens=2, ignore_eos=True
        )
        first = _sequence([1, 2, 3, 4], sampling)
        second = _sequence([5, 6, 7, 8], sampling)
        scheduler.add(first)
        scheduler.add(second)
        prefill = scheduler.schedule()
        scheduler.postprocess(prefill, [9, 10])

        decode_first = scheduler.schedule()
        assert decode_first.kv_transfers.swap_out
        assert scheduler.swap_preemptions == 1
        assert len(scheduler.swapped) == 1
        scheduler.postprocess(decode_first, [11])

        decode_second = scheduler.schedule()
        assert decode_second.kv_transfers.swap_in
        assert scheduler.swap_in_operations == 1
        scheduler.postprocess(decode_second, [12])

        assert scheduler.is_finished()
        metrics = scheduler.metrics_snapshot()
        assert metrics["completed_requests"] == 2
        assert metrics["peak_swapped"] == 1
        assert metrics["peak_cpu_kv_blocks"] == 1


class _FakeRunner:
    kv_cache_dtype = "torch.bfloat16"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def call(self, method_name: str, *args: object):
        self.calls.append((method_name, *args))
        if method_name == "run_plan":
            return ExecutionResult(token_ids=(7,))
        return None


def test_engine_exit_drops_executor_runner_reference() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    runner = _FakeRunner()
    engine.model_runner = runner
    engine.executor = SimpleNamespace(runner=runner)
    engine.ps = []

    engine.exit()

    assert runner.calls == [("exit",)]
    assert not hasattr(engine, "executor")
    assert not hasattr(engine, "model_runner")


def test_engine_exit_cleans_references_before_reraising_runner_failure() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    released: list[bool] = []

    class BackendStub:
        def release(self) -> None:
            released.append(True)

    class FailingRunner:
        def __init__(self) -> None:
            self.execution_backend = BackendStub()

        def call(self, method_name: str) -> None:
            assert method_name == "exit"
            raise RuntimeError("synthetic runner exit failure")

    runner = FailingRunner()
    engine.model_runner = runner
    engine.executor = SimpleNamespace(runner=runner)
    engine.ps = []
    engine.control_senders = []

    with pytest.raises(RuntimeError, match="synthetic runner exit failure"):
        engine.exit()

    assert released == [True]
    assert not hasattr(runner, "execution_backend")
    assert not hasattr(engine, "executor")
    assert not hasattr(engine, "model_runner")


def test_model_runner_exit_breaks_backend_ownership_cycle(
    monkeypatch,
) -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size = 1
    released: list[bool] = []

    class BackendStub:
        def release(self) -> None:
            released.append(True)

    runner.execution_backend = BackendStub()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "prism_infer.engine.model_runner.dist.destroy_process_group",
        lambda: None,
    )

    runner.exit()

    assert released == [True]
    assert not hasattr(runner, "execution_backend")


def test_executor_applies_immutable_kv_plan_before_model_run() -> None:
    seq = _sequence([1])
    plan = BatchPlan(
        phase=BatchPhase.DECODE,
        sequences=(seq,),
        scheduled_token_counts=(1,),
        kv_transfers=KVTransferPlan(
            copy_on_write=((1, 2),),
            swap_out=((3, 4),),
            swap_in=((5, 6),),
        ),
    )
    runner = _FakeRunner()
    executor = ModelExecutor(
        SimpleNamespace(compression_mode="off"),
        runner,
        SimpleNamespace(),
    )

    result = executor.execute(plan)

    assert result.token_ids == (7,)
    assert [call[0] for call in runner.calls] == [
        "copy_kv_blocks",
        "swap_blocks",
        "swap_blocks",
        "run_plan",
    ]
    assert runner.calls[-1][1] is plan


def test_engine_metrics_observe_without_driving_scheduler() -> None:
    seq = _sequence(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    plan = BatchPlan(
        phase=BatchPhase.PREFILL,
        sequences=(seq,),
        scheduled_token_counts=(2,),
        created_ns=1_500_000,
    )
    metrics = EngineMetrics()
    metrics.on_request_submitted(seq, timestamp_ns=1_000_000)
    metrics.on_batch_planned(plan)
    execution = ExecutionResult(token_ids=(7,))
    metrics.on_batch_completed(
        plan,
        execution,
        started_ns=2_000_000,
        finished_ns=3_000_000,
    )
    metrics.on_requests_finished(
        (
            RequestOutput(
                request_id=seq.seq_id,
                token_ids=(7,),
                finish_reason="length",
            ),
        ),
        timestamp_ns=3_100_000,
    )

    snapshot = metrics.snapshot()
    request = snapshot["requests"][0]
    assert request["queue_ms"] == pytest.approx(0.5)
    assert request["ttft_ms"] == pytest.approx(2.0)
    assert request["latency_ms"] == pytest.approx(2.1)
    assert snapshot["batches"][0]["duration_ms"] == pytest.approx(1.0)
