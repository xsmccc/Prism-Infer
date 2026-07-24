"""P7.3 deterministic online arrival, batching and SLO tests."""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.analysis.online_serving import (
    ONLINE_BENCHMARK_SCHEMA_VERSION,
    percentile,
    summarize_online_run,
    validate_online_benchmark_record,
)
from prism_infer.engine.contracts import ExecutionResult
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.engine.online import OnlineRequest, OnlineServingSession
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


class _FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)


class _DeterministicExecutor:
    def __init__(self, clock: _FakeClock, step_s: float = 0.010) -> None:
        self.clock = clock
        self.step_s = step_s

    def execute(self, plan) -> ExecutionResult:
        token_ids: list[int | None] = []
        if plan.is_prefill:
            for seq, count in zip(plan.sequences, plan.scheduled_token_counts):
                seq.num_computed_tokens += count
                seq.num_cached_tokens = seq.num_computed_tokens
                token_ids.append(None if not seq.is_prefill_finished else 1000 + seq.seq_id)
        else:
            token_ids = [2000 + seq.seq_id for seq in plan.sequences]
        self.clock.advance(self.step_s)
        return ExecutionResult(token_ids=tuple(token_ids))


def _engine(
    clock: _FakeClock,
    *,
    max_queue_size: int | None = None,
) -> LLMEngine:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=32,
        enable_chunked_prefill=True,
        max_chunk_size=2,
        max_queue_size=max_queue_size,
        max_consecutive_prefill_batches=1,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
        num_cpu_blocks=8,
        enable_prefix_caching=False,
        compression_mode="off",
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine.clock_ns = clock
    engine.scheduler = Scheduler(config, clock_ns=clock)
    engine.executor = _DeterministicExecutor(clock)
    engine.metrics = EngineMetrics()
    return engine


def _request(
    key: str,
    arrival_s: float,
    *,
    max_tokens: int = 3,
    cancel_s: float | None = None,
) -> OnlineRequest:
    return OnlineRequest(
        request_key=key,
        arrival_offset_s=arrival_s,
        payload={"type": "text", "prompt": [1, 2, 3, 4, 5]},
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
        cancel_offset_s=cancel_s,
    )


def test_online_request_rejects_nonfinite_time_and_incomplete_media_payload() -> None:
    sampling = SamplingParams(max_tokens=1)
    with pytest.raises(ValueError, match="finite non-negative"):
        OnlineRequest(
            request_key="nan-arrival",
            arrival_offset_s=float("nan"),
            payload={"type": "text", "prompt": "hello"},
            sampling_params=sampling,
        )
    with pytest.raises(ValueError, match="requires 'image'"):
        OnlineRequest(
            request_key="missing-image",
            arrival_offset_s=0.0,
            payload={"type": "image", "prompt": "describe"},
            sampling_params=sampling,
        )


def test_online_arrival_loop_refreshes_clock_after_host_preprocessing() -> None:
    clock = _FakeClock()
    session = OnlineServingSession(
        SimpleNamespace(is_finished=lambda: True),
        clock_ns=clock,
        sleep_fn=clock.advance,
    )
    requests = (_request("first", 0.0), _request("second", 0.5))
    state = session._new_run_state(requests)
    submitted: list[str] = []

    def submit(request: OnlineRequest, _arrival_ns: int) -> int:
        submitted.append(request.request_key)
        clock.advance(1.0)
        return len(submitted)

    session._submit = submit
    session._submit_ready_arrivals(state)

    assert submitted == ["first", "second"]
    assert not state.pending


def test_online_session_preserves_arrival_and_continuous_batching() -> None:
    clock = _FakeClock()
    engine = _engine(clock)
    session = OnlineServingSession(
        engine,
        clock_ns=clock,
        sleep_fn=clock.advance,
    )

    result = session.run((_request("a", 0.0), _request("b", 0.015)))

    assert [request.state for request in result.requests] == [
        "finished",
        "finished",
    ]
    assert all(len(request.token_ids) == 3 for request in result.requests)
    request_metrics = {record["request_id"]: record for record in result.engine_metrics["requests"]}
    request_b = next(request for request in result.requests if request.request_key == "b")
    assert request_metrics[request_b.request_id]["submitted_ns"] == 15_000_000
    assert request_metrics[request_b.request_id]["queue_ms"] >= 5.0

    batches = result.engine_metrics["batches"]
    phases = [batch["phase"] for batch in batches]
    assert "prefill" in phases and "decode" in phases
    # Once request A begins decoding, a queued/chunking B request cannot starve it.
    first_decode = phases.index("decode")
    assert phases[first_decode - 1] == "prefill"
    assert result.scheduler_metrics["peak_active"] == 2

    summary = summarize_online_run(
        result.to_record(),
        ttft_slo_ms=100.0,
        tpot_slo_ms=100.0,
    )
    assert summary["counts"] == {
        "submitted": 2,
        "completed": 2,
        "rejected": 0,
        "cancelled": 0,
        "good": 2,
    }
    assert summary["goodput"]["fraction_of_completed"] == 1.0

    record = {
        "schema_version": 1,
        "record_type": "prism_online_run",
        "git_commit": "abc123",
        "git_dirty": False,
        "hardware": {
            "gpu": "test-gpu",
            "gpu_uuid": "GPU-test",
            "total_memory_bytes": 1,
        },
        "workload": {
            "manifest": "test",
            "case": "test",
            "requests": 2,
            "max_tokens": 3,
        },
        "arrival": {
            "process": "constant",
            "request_rate_per_s": 1.0,
            "seed": 1,
            "offsets_s": [0.0, 0.015],
        },
        "engine": {
            "mode": "off_eager",
            "max_model_len": 32,
            "max_num_batched_tokens": 8,
            "max_num_seqs": 4,
            "max_chunk_size": 2,
            "num_kvcache_blocks": 32,
            "kvcache_block_size": 4,
            "enable_prefix_caching": False,
        },
        "run": result.to_record(),
        "summary": summary,
    }
    validate_online_benchmark_record(record)
    record["summary"] = {**summary, "counts": {**summary["counts"], "good": 0}}
    with pytest.raises(ValueError, match="does not match"):
        validate_online_benchmark_record(record)


def test_visual_prefill_guard_preserves_text_admission_and_bounds_visual_delay() -> None:
    clock = _FakeClock()
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=32,
        enable_chunked_prefill=True,
        max_chunk_size=16,
        max_queue_size=None,
        max_consecutive_prefill_batches=1,
        min_decode_batches_between_visual_prefills=2,
        max_vision_patches_per_batch=16,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
        num_cpu_blocks=8,
        enable_prefix_caching=False,
    )
    scheduler = Scheduler(config, clock_ns=clock)
    sampling = SamplingParams(max_tokens=8, ignore_eos=True)
    decode_seq = Sequence(
        [1, 2],
        sampling,
        block_size=4,
        request_id=0,
    )
    scheduler.add(decode_seq)
    first_prefill = scheduler.schedule()
    decode_seq.num_computed_tokens = decode_seq.num_prompt_tokens
    scheduler.postprocess(first_prefill, (10,))

    visual_seq = Sequence(
        [1, 99, 2],
        sampling,
        block_size=4,
        request_id=1,
        pixel_values=torch.ones((4, 2)),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        image_token_id=99,
        image_token_count=1,
    )
    text_seq = Sequence(
        [3, 4],
        sampling,
        block_size=4,
        request_id=2,
    )
    scheduler.add(visual_seq)
    scheduler.add(text_seq)

    first_decode = scheduler.schedule()
    assert not first_decode.is_prefill
    scheduler.postprocess(first_decode, (11,))

    text_prefill = scheduler.schedule()
    assert text_prefill.is_prefill
    assert text_prefill.sequences == (text_seq,)
    text_seq.num_computed_tokens = text_seq.num_prompt_tokens
    scheduler.postprocess(text_prefill, (12,))

    second_decode = scheduler.schedule()
    assert not second_decode.is_prefill
    scheduler.postprocess(second_decode, (13, 14))

    visual_prefill = scheduler.schedule()
    assert visual_prefill.is_prefill
    assert visual_prefill.sequences == (visual_seq,)
    metrics = scheduler.metrics_snapshot()
    assert metrics["visual_prefill_deferrals"] == 1
    assert metrics["text_prefill_bypasses"] == 1
    assert metrics["max_decode_batches_between_visual_prefills"] == 2


def test_online_schema_v2_requires_projection_mode() -> None:
    clock = _FakeClock()
    result = OnlineServingSession(
        _engine(clock),
        clock_ns=clock,
        sleep_fn=clock.advance,
    ).run((_request("a", 0.0),))
    summary = summarize_online_run(
        result.to_record(),
        ttft_slo_ms=100.0,
        tpot_slo_ms=100.0,
    )
    record = {
        "schema_version": ONLINE_BENCHMARK_SCHEMA_VERSION,
        "record_type": "prism_online_run",
        "git_commit": "abc123",
        "git_dirty": False,
        "hardware": {
            "gpu": "test-gpu",
            "gpu_uuid": "GPU-test",
            "total_memory_bytes": 1,
        },
        "workload": {
            "manifest": "test",
            "case": "test",
            "requests": 1,
            "max_tokens": 3,
        },
        "arrival": {
            "process": "burst",
            "request_rate_per_s": 1.0,
            "seed": 1,
            "offsets_s": [0.0],
        },
        "engine": {
            "mode": "off_eager",
            "max_model_len": 32,
            "max_num_batched_tokens": 8,
            "max_num_seqs": 4,
            "max_chunk_size": 2,
            "num_kvcache_blocks": 32,
            "kvcache_block_size": 4,
            "enable_prefix_caching": False,
            "mlp_projection_mode": "packed",
        },
        "run": result.to_record(),
        "summary": summary,
    }
    validate_online_benchmark_record(record)

    del record["engine"]["mlp_projection_mode"]
    with pytest.raises(ValueError, match="mlp_projection_mode"):
        validate_online_benchmark_record(record)


def test_online_admission_rejection_and_cancellation_are_accounted() -> None:
    clock = _FakeClock()
    engine = _engine(clock, max_queue_size=1)
    session = OnlineServingSession(
        engine,
        clock_ns=clock,
        sleep_fn=clock.advance,
    )

    result = session.run(
        (
            _request("z-accepted", 0.0, max_tokens=8, cancel_s=0.025),
            _request("a-rejected", 0.0, max_tokens=2),
        )
    )

    by_key = {request.request_key: request for request in result.requests}
    assert by_key["z-accepted"].state == "cancelled"
    assert by_key["a-rejected"].state == "rejected"
    assert result.scheduler_metrics["cancelled_requests"] == 1
    assert result.scheduler_metrics["rejected_requests"] == 1
    summary = summarize_online_run(
        result.to_record(),
        ttft_slo_ms=100.0,
        tpot_slo_ms=100.0,
    )
    assert summary["counts"]["cancelled"] == 1
    assert summary["counts"]["rejected"] == 1
    assert summary["counts"]["completed"] == 0


def test_online_percentile_and_schema_fail_closed() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="duration_s"):
        summarize_online_run(
            {"duration_s": 0, "engine_metrics": {"requests": []}},
            ttft_slo_ms=10,
            tpot_slo_ms=10,
        )
