"""Cancellation of paused prefill must not consume another request's output."""

from itertools import count
from types import SimpleNamespace

from prism_infer.engine.contracts import BatchPhase, ExecutionResult
from prism_infer.engine.llm_engine import LLMEngine, _PendingPrefill
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.engine.request import RequestState
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


class _Executor:
    def __init__(self):
        self.finish_calls = 0

    def execute(self, plan):
        if plan.is_prefill:
            for seq, part in zip(plan.sequences, plan.prefill_slices, strict=True):
                seq.num_computed_tokens = part.token_end
        return ExecutionResult(token_ids=tuple(100 + seq.seq_id for seq in plan.sequences))

    def finish_prefill(self, plan, _handle):
        self.finish_calls += 1
        return self.execute(plan)


def _engine():
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=32,
        enable_chunked_prefill=False,
        max_chunk_size=4,
        max_queue_size=None,
        max_consecutive_prefill_batches=2,
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=4,
        num_cpu_blocks=0,
        enable_prefix_caching=False,
        enable_visual_embedding_cache=False,
        compression_mode="scaled_fp8_kv",
    )
    engine.clock_ns = count(start=1, step=1000).__next__
    engine.scheduler = Scheduler(engine.config, clock_ns=engine.clock_ns)
    engine.metrics = EngineMetrics()
    engine.executor = _Executor()
    engine._pending_prefill = None
    engine._decode_after_slo_prefill_interrupt = False
    return engine


def _sequence(request_id, max_tokens):
    return Sequence(
        [10 + request_id, 20 + request_id],
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        block_size=4,
        request_id=request_id,
    )


def _pause_prefill(engine, requests):
    # A real Decode request remains runnable while the new Prefill is paused.
    decoding = _sequence(0, 4)
    engine._submit_sequence(decoding)
    initial = engine.scheduler.schedule()
    engine._execute_planned_step(initial, None)
    assert decoding.status is RequestState.DECODING
    for request in requests:
        engine._submit_sequence(request)
    plan = engine.scheduler.schedule()
    assert plan.phase is BatchPhase.PREFILL
    assert plan.sequences == tuple(requests)
    engine.metrics.on_batch_planned(plan)
    engine._pending_prefill = _PendingPrefill(
        plan=plan,
        runner_handle=object(),
        started_ns=engine.clock_ns(),
    )
    return plan


def test_cancel_one_paused_prefill_preserves_survivor_first_and_final_token():
    engine = _engine()
    cancelled = _sequence(1, 2)
    survivor = _sequence(2, 1)
    paused = _pause_prefill(engine, [cancelled, survivor])
    survivor_pages = tuple(survivor.block_table)
    survivor_start = paused.prefill_slices[1].token_start

    assert engine.cancel_request(cancelled.seq_id)

    assert engine.executor.finish_calls == 0
    assert engine._pending_prefill is None
    assert cancelled.status is RequestState.CANCELLED
    assert cancelled.completion_token_ids == []
    assert survivor.status is RequestState.PREFILLING
    assert survivor.num_computed_tokens == survivor_start
    assert tuple(survivor.block_table) == survivor_pages
    assert survivor.completion_token_ids == []

    # The prior Prefill may yield one Decode turn; then the survivor must be
    # selected even though enable_chunked_prefill=False, without reallocating KV.
    for _ in range(3):
        resumed = engine.scheduler.schedule()
        result = engine._execute_planned_step(resumed, None)
        if survivor.seq_id in resumed.sequence_ids:
            assert resumed.phase is BatchPhase.PREFILL
            assert resumed.prefill_slices[0].token_start == survivor_start
            assert result.execution.token_ids == (102,)
            assert len(result.outputs) == 1
            assert result.outputs[0].request_id == survivor.seq_id
            assert result.outputs[0].token_ids == (102,)
            assert result.outputs[0].finish_reason == "length"
            break
    else:
        raise AssertionError("surviving Prefill was never rescheduled")
    assert survivor.status is RequestState.FINISHED


def test_cancel_single_paused_prefill_does_not_finish_or_sample_it():
    engine = _engine()
    request = _sequence(1, 1)
    _pause_prefill(engine, [request])

    assert engine.cancel_request(request.seq_id)

    assert engine.executor.finish_calls == 0
    assert engine._pending_prefill is None
    assert request.status is RequestState.CANCELLED
    assert request.completion_token_ids == []
    assert request.block_table == []
    remaining = engine.scheduler.schedule()
    assert remaining.phase is BatchPhase.DECODE
    assert remaining.sequence_ids == (0,)
