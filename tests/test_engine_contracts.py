"""Engine boundary and request-lifecycle tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from itertools import count
from multiprocessing import Pipe
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
from prism_infer.engine.scheduler_policy import (
    FCFSSchedulerPolicy,
    VisionAwareSchedulerPolicy,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.tp_control import (
    TPControlPlane,
    TPMethod,
    TPResponse,
    TPResponseStatus,
)
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
    assert len(plan.prefill_slices) == 1
    assert plan.prefill_slices[0].sequence_id == seq.seq_id
    assert plan.prefill_slices[0].token_start == 0
    assert plan.prefill_slices[0].token_end == 2
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
    assert visual_policy.prefill_token_count(visual, available_tokens=4) == 1
    visual.num_computed_tokens = 1
    assert visual_policy.prefill_token_count(visual, available_tokens=5) == 5


def test_vision_aware_policy_bypasses_heavy_prefill_with_bounded_credit() -> None:
    policy = VisionAwareSchedulerPolicy(
        max_model_len=8,
        max_num_batched_tokens=8,
        max_num_seqs=2,
        enable_chunked_prefill=False,
        max_chunk_size=8,
        heavy_prefill_vision_patch_threshold=4,
        min_decode_batches_between_heavy_prefills=3,
    )
    heavy = _sequence(
        [1],
        pixel_values=torch.zeros((4, 1)),
        image_grid_thw=torch.ones((1, 3), dtype=torch.int64),
    )
    light = _sequence([2])

    assert (
        policy.waiting_prefill_index(
            (heavy, light),
            has_decode=True,
            decode_batches_since_heavy_prefill=0,
            light_prefill_bypasses_since_heavy=0,
        )
        == 1
    )
    assert (
        policy.waiting_prefill_index(
            (heavy,),
            has_decode=True,
            decode_batches_since_heavy_prefill=0,
            light_prefill_bypasses_since_heavy=0,
        )
        is None
    )
    assert (
        policy.waiting_prefill_index(
            (heavy, light),
            has_decode=True,
            decode_batches_since_heavy_prefill=3,
            light_prefill_bypasses_since_heavy=0,
        )
        == 0
    )
    assert (
        policy.waiting_prefill_index(
            (heavy, light),
            has_decode=True,
            decode_batches_since_heavy_prefill=0,
            light_prefill_bypasses_since_heavy=2,
        )
        == 0
    )


def test_slo_policy_isolates_tight_text_from_visual_prefill() -> None:
    scheduler = Scheduler(
        _scheduler_config(
            scheduler_policy="slo_aware",
            heavy_prefill_vision_patch_threshold=4,
        ),
        clock_ns=lambda: 0,
    )
    visual = _sequence(
        [1],
        SamplingParams(max_tokens=1),
        pixel_values=torch.zeros((4, 1)),
        image_grid_thw=torch.ones((1, 3), dtype=torch.int64),
    )
    visual.submitted_ns = 0
    visual.ttft_slo_ms = 1_352.0
    text = _sequence([2], SamplingParams(max_tokens=1))
    text.submitted_ns = 0
    text.ttft_slo_ms = 170.0
    scheduler.add(visual)
    scheduler.add(text)

    plan = scheduler.schedule()

    assert plan.sequences == (text,)
    assert tuple(scheduler.waiting) == (visual,)
    metrics = scheduler.metrics_snapshot()
    assert metrics["deadline_prefill_reorders"] == 1
    assert metrics["cost_tier_batch_deferrals"] == 1


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


def test_scheduler_caps_gpu_resident_sequences_at_max_num_seqs() -> None:
    scheduler = Scheduler(
        _scheduler_config(
            max_num_seqs=2,
            num_kvcache_blocks=8,
        )
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True)
    sequences = [_sequence([index] * 4, sampling) for index in range(1, 5)]
    for seq in sequences:
        scheduler.add(seq)

    prefill = scheduler.schedule()
    assert prefill.sequences == tuple(sequences[:2])
    assert len(scheduler.running) == 2
    assert list(scheduler.waiting) == sequences[2:]
    scheduler.postprocess(prefill, [11, 12])

    first_decode = scheduler.schedule()
    assert first_decode.phase is BatchPhase.DECODE
    scheduler.postprocess(first_decode, [13, 14])

    # Prefill is policy-eligible again, but no new request may become GPU
    # resident until one of the two existing decoders leaves.
    second_decode = scheduler.schedule()
    assert second_decode.phase is BatchPhase.DECODE
    assert second_decode.sequences == tuple(sequences[:2])
    assert len(scheduler.running) == 2
    assert list(scheduler.waiting) == sequences[2:]
    scheduler.postprocess(second_decode, [15, 16])

    next_prefill = scheduler.schedule()
    assert next_prefill.sequences == tuple(sequences[2:])
    assert scheduler.metrics_snapshot()["peak_running"] == 2


def test_scheduler_bounds_aggregate_vision_patches_and_isolates_oversized_request() -> None:
    def image_sequence(patches: int) -> Sequence:
        return _sequence(
            [1, 99, 99, 2],
            SamplingParams(max_tokens=1),
            pixel_values=torch.zeros(patches, 4),
            image_grid_thw=torch.tensor([[1, 2, patches // 2]]),
            image_token_id=99,
            image_token_count=2,
        )

    scheduler = Scheduler(_scheduler_config(max_vision_patches_per_batch=8))
    first = image_sequence(6)
    second = image_sequence(6)
    scheduler.add(first)
    scheduler.add(second)

    plan = scheduler.schedule()

    assert plan.sequences == (first,)
    assert plan.num_scheduled_vision_patches == 6
    assert list(scheduler.waiting) == [second]

    oversized_scheduler = Scheduler(_scheduler_config(max_vision_patches_per_batch=4))
    oversized = image_sequence(6)
    trailing = _sequence([3, 4], SamplingParams(max_tokens=1))
    oversized_scheduler.add(oversized)
    oversized_scheduler.add(trailing)

    dedicated = oversized_scheduler.schedule()

    assert dedicated.sequences == (oversized,)
    assert dedicated.num_scheduled_vision_patches == 6
    assert list(oversized_scheduler.waiting) == [trailing]


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
        sampling = SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True)
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
        sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
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
    events: list[str] = []

    class BackendStub:
        def release(self) -> None:
            events.append("release")

    runner.execution_backend = BackendStub()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(
        "prism_infer.engine.model_runner.dist.destroy_process_group",
        lambda: events.append("destroy_process_group"),
    )

    runner.exit()

    assert events == ["synchronize", "release", "destroy_process_group"]
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
    assert runner.calls[:-1] == [
        ("swap_blocks", [(5, 6)], "in"),
        ("swap_blocks", [(3, 4)], "out"),
        ("copy_kv_blocks", [(1, 2)]),
    ]
    assert runner.calls[-1][0] == "run_plan"
    assert runner.calls[-1][1] is plan


class _CPUTransferRunner:
    """Real page-copy methods with CPU-backed FP8 payload and FP32 scales."""

    kv_cache_dtype = torch.float8_e4m3fn
    _bound_gpu_scale_cache = ModelRunner._bound_gpu_scale_cache
    copy_kv_blocks = ModelRunner.copy_kv_blocks
    copy_kv_block_prefixes = ModelRunner.copy_kv_block_prefixes
    swap_blocks = ModelRunner.swap_blocks

    def __init__(self, gpu_pages: int, cpu_pages: int) -> None:
        self.kv_cache = torch.zeros(2, 1, gpu_pages, 4, 1, 2, dtype=self.kv_cache_dtype)
        self.cpu_kv_cache = torch.zeros(2, 1, cpu_pages, 4, 1, 2, dtype=self.kv_cache_dtype)
        self.kv_scale_cache = torch.zeros(2, 1, gpu_pages, 4, 1, dtype=torch.float32)
        self.cpu_kv_scale_cache = torch.zeros(2, 1, cpu_pages, 4, 1, dtype=torch.float32)

    def call(self, method_name: str, *args: object):
        if method_name != "run_plan":
            return getattr(self, method_name)(*args)
        plan = args[0]
        for index, seq in enumerate(plan.sequences):
            if plan.is_prefill:
                token_slice = plan.prefill_slices[index]
                start, end = token_slice.token_start, token_slice.token_end
            else:
                start, end = seq.num_tokens - 1, seq.num_tokens
            for position in range(start, end):
                block = seq.block_table[position // 4]
                row = position % 4
                # Write distinct finite FP8 bit patterns, not FP8 arithmetic;
                # transfer correctness requires payload bytes AND scales exact.
                self.kv_cache.view(torch.uint8)[:, :, block, row].fill_(
                    16 + seq.token_ids[position] % 48
                )
                self.kv_scale_cache[:, :, block, row].fill_(0.125 + seq.token_ids[position] % 257)
            if plan.is_prefill:
                seq.num_computed_tokens = end
        return ExecutionResult(token_ids=(88,) * plan.batch_size)


def _cpu_transfer_engine(monkeypatch, *, gpu_pages: int, cpu_pages: int):
    # swap_blocks synchronizes CUDA after its real copy_ calls. Only that
    # device fence is replaced: all storage/copies in these tests are on CPU.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    config = _scheduler_config(
        num_kvcache_blocks=gpu_pages,
        num_cpu_blocks=cpu_pages,
        max_model_len=128,
        enable_prefix_caching=True,
        compression_mode="scaled_fp8_kv",
        tensor_parallel_size=1,
        max_consecutive_prefill_batches=2,
    )
    scheduler = Scheduler(config)
    runner = _CPUTransferRunner(gpu_pages, cpu_pages)
    return scheduler, ModelExecutor(config, runner, scheduler.block_manager), runner


def _prefill_transfer_requests(scheduler, executor, *requests):
    for seq in requests:
        scheduler.add(seq)
    plan = scheduler.schedule()
    assert plan.is_prefill
    result = executor.execute(plan)
    scheduler.postprocess(plan, result.token_ids)


def test_executor_preserves_swap_out_source_reused_by_cow(monkeypatch) -> None:
    scheduler, executor, runner = _cpu_transfer_engine(monkeypatch, gpu_pages=4, cpu_pages=2)
    crossing = _sequence([10, 11, 12, 13])
    shared_tail = _sequence(
        [1, 151655, 700],
        pixel_values=torch.zeros(4, 3),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        image_token_id=151655,
        image_token_count=1,
        multimodal_media_token_hashes=(bytes(range(32)),),
    )
    shared_tail.multimodal_prefix_cache_key = "transfer-test-image"
    victim = _sequence(list(range(20, 28)))
    _prefill_transfer_requests(scheduler, executor, crossing, shared_tail, victim)
    expected_payload = runner.kv_cache.view(torch.uint8)[:, :, victim.block_table].clone()
    expected_scales = runner.kv_scale_cache[:, :, victim.block_table].clone()

    # crossing needs a new page, preempting the two-page victim. The next
    # request has a cache-owned tail, whose CoW consumes the other freed page.
    plan = scheduler.schedule()
    transfers = plan.kv_transfers
    assert plan.sequences == (crossing, shared_tail)
    assert len(transfers.copy_on_write) == 1
    assert transfers.copy_on_write[0][1] in {src for src, _ in transfers.swap_out}
    executor.execute(plan)

    assert torch.equal(
        runner.cpu_kv_cache.view(torch.uint8)[:, :, victim.cpu_block_table],
        expected_payload,
    )
    assert torch.equal(runner.cpu_kv_scale_cache[:, :, victim.cpu_block_table], expected_scales)


def test_executor_consumes_swap_in_before_cpu_slot_is_reused(monkeypatch) -> None:
    scheduler, executor, runner = _cpu_transfer_engine(monkeypatch, gpu_pages=3, cpu_pages=1)
    crossing = _sequence([10, 11, 12, 13])
    resident = _sequence([30, 31])
    swapped = _sequence([40, 41])
    _prefill_transfer_requests(scheduler, executor, crossing, resident, swapped)

    # Establish a completed earlier preemption using the real scheduler API.
    scheduler.running.remove(swapped)
    prior_swap_out = []
    scheduler.preempt(swapped, prior_swap_out)
    runner.call("swap_blocks", prior_swap_out, "out")
    expected_payload = runner.cpu_kv_cache.view(torch.uint8).clone()
    expected_scales = runner.cpu_kv_scale_cache.clone()

    # A short ordinary request reuses the released GPU page and finishes.
    # Its bytes must not overwrite swapped's still-live CPU copy next round.
    probe = _sequence([999], SamplingParams(max_tokens=1))
    _prefill_transfer_requests(scheduler, executor, probe)
    assert probe.is_finished
    plan = scheduler.schedule()
    transfers = plan.kv_transfers
    assert plan.sequences == (crossing, resident)
    assert len(transfers.swap_in) == len(transfers.swap_out) == 1
    assert transfers.swap_out[0] == tuple(reversed(transfers.swap_in[0]))
    executor.execute(plan)

    assert torch.equal(runner.cpu_kv_cache.view(torch.uint8), expected_payload)
    assert torch.equal(runner.cpu_kv_scale_cache, expected_scales)


def test_tp_control_dispatches_prefix_row_copies_and_receives_ack() -> None:
    sender, receiver = Pipe(duplex=True)
    rank0 = TPControlPlane(rank=0, world_size=2, channel=[sender])
    rank1 = TPControlPlane(rank=1, world_size=2, channel=receiver)
    copies = [(1, 2, 145), (3, 4, 1)]
    dispatched: list[object] = []
    worker = SimpleNamespace(copy_kv_block_prefixes=dispatched.append)
    try:
        sent, _ = rank0.broadcast("copy_kv_block_prefixes", (copies,))
        received = rank1.read_command()
        assert received.method is TPMethod.COPY_KV_BLOCK_PREFIXES
        assert received.args == (copies,)
        ModelRunner._invoke_local(worker, received.method.value, received.args)
        assert dispatched == [copies]
        rank1.send_response(TPResponse.ok(received, worker_rank=1))
        rank0.await_responses(sent)
    finally:
        sender.close()
        receiver.close()


def test_tp_control_validates_initial_topology_and_timeout_updates() -> None:
    sender, receiver = Pipe(duplex=True)
    try:
        with pytest.raises(ValueError, match="channel count"):
            TPControlPlane(rank=0, world_size=3, channel=[sender])
        with pytest.raises(ValueError, match="timeout must be positive"):
            TPControlPlane(rank=0, world_size=2, channel=[sender], timeout_seconds=0)

        runner = ModelRunner.__new__(ModelRunner)
        runner.tp_control = TPControlPlane(rank=0, world_size=2, channel=[sender])
        runner.control_timeout_seconds = 0.25
        assert runner._control_timeout() == 0.25
        assert runner._rank0_control_channels() == [sender]
        with pytest.raises(ValueError, match="timeout must be positive"):
            runner.control_timeout_seconds = 0
        assert runner._control_timeout() == 0.25

        worker = TPControlPlane(rank=1, world_size=2, channel=receiver)
        with pytest.raises(RuntimeError, match="only TP rank 0"):
            worker.rank0_channels()
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("wrong_field", ["request_id", "worker_rank"])
def test_tp_control_still_rejects_mismatched_ack(wrong_field) -> None:
    sender, receiver = Pipe(duplex=True)
    rank0 = TPControlPlane(rank=0, world_size=2, channel=[sender])
    rank1 = TPControlPlane(rank=1, world_size=2, channel=receiver)
    try:
        command, _ = rank0.broadcast("copy_kv_blocks", ([(1, 2)],))
        received = rank1.read_command()
        fields = {"request_id": received.request_id, "worker_rank": 1}
        fields[wrong_field] += 1
        rank1.send_response(TPResponse(status=TPResponseStatus.OK, **fields))
        message = "stale or out-of-order" if wrong_field == "request_id" else "rank mismatch"
        with pytest.raises(RuntimeError, match=message):
            rank0.await_responses(command)
    finally:
        sender.close()
        receiver.close()


def test_visual_embedding_cache_rejects_text_model_before_construction(monkeypatch) -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(enable_visual_embedding_cache=True)

    def unexpected_model_construction(_config):
        pytest.fail("visual embedding cache incompatibility reached model construction")

    monkeypatch.setattr(
        "prism_infer.engine.model_runner.Qwen3ForCausalLM",
        unexpected_model_construction,
    )
    with pytest.raises(ValueError, match="visual embedding cache.*Qwen3-VL"):
        runner._create_model(SimpleNamespace(model_type="qwen3"))


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
    seq.num_cached_tokens = 1
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
    assert request["cached_tokens"] == 1
    assert request["ttft_ms"] == pytest.approx(2.0)
    assert request["latency_ms"] == pytest.approx(2.1)
    assert snapshot["batches"][0]["duration_ms"] == pytest.approx(1.0)
