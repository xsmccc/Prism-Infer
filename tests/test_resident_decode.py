"""Decode interleaving must not preempt requests while Prefill owns the pool."""

from types import SimpleNamespace

from prism_infer.engine.request import RequestState
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


def _scheduler() -> Scheduler:
    return Scheduler(
        SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=32,
            max_model_len=128,
            enable_chunked_prefill=False,
            max_chunk_size=32,
            max_queue_size=None,
            max_consecutive_prefill_batches=1,
            eos=-1,
            num_kvcache_blocks=3,
            kvcache_block_size=4,
            num_cpu_blocks=3,
            enable_prefix_caching=False,
        )
    )


def _complete_prefill(scheduler, plan):
    for seq in plan.sequences:
        seq.num_computed_tokens = seq.num_prompt_tokens
    return scheduler.postprocess(plan, [42] * plan.batch_size)


def test_full_pool_finishes_prefill_without_preempting_last_decoder():
    scheduler = _scheduler()
    decoder = Sequence([1, 2, 3, 4], block_size=4, request_id=0)
    scheduler.add(decoder)
    _complete_prefill(scheduler, scheduler.schedule())
    pending = Sequence(
        list(range(10, 18)), SamplingParams(max_tokens=1), block_size=4, request_id=1
    )
    scheduler.add(pending)
    prefill = scheduler.schedule_prefill()
    pages = tuple(decoder.block_table)

    assert scheduler.schedule_resident_decode() is None
    assert decoder.status is RequestState.DECODING
    assert tuple(decoder.block_table) == pages
    assert not scheduler.swapped and not scheduler.waiting
    assert scheduler.swap_preemptions == scheduler.recompute_preemptions == 0

    outputs = _complete_prefill(scheduler, prefill)
    assert outputs[0].request_id == pending.seq_id
    decode = scheduler.schedule_resident_decode()
    assert decode.sequence_ids == (decoder.seq_id,)
    assert len(decoder.block_table) == 2


def test_blocked_decoder_does_not_preempt_another_runnable_decoder():
    scheduler = _scheduler()
    blocked = Sequence([1, 2, 3, 4], block_size=4, request_id=0)
    runnable = Sequence([10, 11], block_size=4, request_id=1)
    scheduler.add(blocked)
    scheduler.add(runnable)
    _complete_prefill(scheduler, scheduler.schedule())
    pending = Sequence([20, 21, 22, 23], block_size=4, request_id=2)
    scheduler.add(pending)
    scheduler.schedule_prefill()

    decode = scheduler.schedule_resident_decode()
    assert decode.sequence_ids == (runnable.seq_id,)
    assert blocked.status is RequestState.DECODING
    assert pending.status is RequestState.PREFILLING
    assert decode.kv_transfers.is_empty
    assert not scheduler.swapped and not scheduler.waiting
