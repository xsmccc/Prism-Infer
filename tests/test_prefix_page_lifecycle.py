"""Regressions for computed-page publication, writable tails and reclamation."""

from types import SimpleNamespace

import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams

PAD = 151655
MEDIA_A = bytes(range(32))


def _scheduler(manager: BlockManager) -> Scheduler:
    return Scheduler(
        SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            enable_chunked_prefill=True,
            max_chunk_size=4,
            max_model_len=128,
            eos=-1,
            max_queue_size=32,
            max_consecutive_prefill_batches=4,
        ),
        kv_manager=manager,
    )


def _image_sequence(tokens, image_sizes, media_hashes, request_id, media_key):
    seq = Sequence(
        tokens,
        SamplingParams(max_tokens=8),
        block_size=4,
        request_id=request_id,
        pixel_values=torch.zeros(sum(image_sizes), 3),
        image_grid_thw=torch.tensor([[1, size, 1] for size in image_sizes]),
        image_token_id=PAD,
        image_token_count=sum(image_sizes),
        multimodal_media_token_hashes=media_hashes,
        image_merge_size=1,
    )
    seq.multimodal_prefix_cache_key = media_key
    return seq


def _finish_prefill(manager, seq):
    seq.num_computed_tokens = seq.num_prompt_tokens
    manager.publish_computed_blocks(seq, seq.num_cached_tokens, seq.num_prompt_tokens)
    manager.store_multimodal_prefix(seq)


def test_chunked_prefill_publishes_only_completed_pages():
    manager = BlockManager(32, 4, block_level_mm_prefix=True)
    scheduler = _scheduler(manager)
    requests = [Sequence(list(range(13)), block_size=4, request_id=i) for i in range(2)]
    for seq in requests:
        scheduler.add(seq)

    plan = scheduler.schedule()
    assert plan.scheduled_token_counts == (4, 4)
    assert [seq.num_cached_tokens for seq in requests] == [0, 0]
    assert manager.hash_to_block_id == {}

    # The runner completed only the first chunk; cancellation must not retain
    # hashes for the other pages allocated for the full prompt.
    for seq in requests:
        seq.num_computed_tokens = 4
    scheduler.postprocess(plan, [None, None])
    for seq in requests:
        scheduler.cancel(seq.seq_id)
    later = Sequence(list(range(13)), block_size=4, request_id=2)
    manager.allocate(later)
    assert later.num_cached_tokens == 4
    assert all(manager.blocks[block].hash == -1 for block in later.block_table[1:])


def test_decode_does_not_publish_a_page_before_its_last_kv_write():
    manager = BlockManager(8, 4, block_level_mm_prefix=True)
    scheduler = _scheduler(manager)
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=4), block_size=4, request_id=0)
    scheduler.add(seq)
    prefill = scheduler.schedule()
    seq.num_computed_tokens = 3
    scheduler.postprocess(prefill, [4])
    decode = scheduler.schedule()
    assert manager.hash_to_block_id == {}
    scheduler.postprocess(decode, [5])
    assert manager.blocks[seq.block_table[0]].token_ids == [1, 2, 3, 4]


def test_entry_tail_drops_previous_question_hash_and_supports_shorter_question():
    manager = BlockManager(32, 4, block_level_mm_prefix=True)
    first = _image_sequence([PAD] * 5 + [11, 12, 13, 14], [5], (MEDIA_A,), 0, "A")
    manager.allocate(first)
    _finish_prefill(manager, first)
    canonical_tail = first.block_table[1]
    old_tokens = list(manager.blocks[canonical_tail].token_ids)
    manager.deallocate(first)

    second = _image_sequence([PAD] * 5 + [21], [5], (MEDIA_A,), 1, "A")
    copies = manager.allocate(second)
    assert copies == ((canonical_tail, second.block_table[1], 1),)
    assert manager.blocks[second.block_table[1]].hash == -1
    _finish_prefill(manager, second)
    second.append_token(300)
    for token in (301, 302, 303):
        if second.physical_kv_len % 4 != 1:
            manager.copy_on_write(second)
        manager.may_append(second)
        manager.publish_computed_blocks(second, second.num_tokens - 1, second.num_tokens)
        second.append_token(token)
    assert manager.blocks[canonical_tail].token_ids == old_tokens
    assert manager.blocks[second.block_table[1]].token_ids == [PAD, 21, 300, 301]


def test_cache_only_shared_pages_are_reclaimable_but_active_pages_are_not():
    manager = BlockManager(6, 4, block_level_mm_prefix=True)
    first = None
    for request_id, key in enumerate(("A+B", "A+C")):
        seq = _image_sequence(
            [PAD] * 4 + [999] + [PAD] * 4 + [11],
            [4, 4],
            (MEDIA_A, bytes([request_id + 100]) * 32),
            request_id,
            key,
        )
        manager.allocate(seq)
        _finish_prefill(manager, seq)
        if first is None:
            first = seq
        else:
            assert seq.block_table[0] == first.block_table[0]
        # Retain the first active request until both entries have been created.
        if request_id:
            manager.deallocate(seq)

    cold = Sequence(list(range(23)), SamplingParams(max_tokens=1), block_size=4, request_id=2)
    assert not manager.can_allocate(cold)
    manager.deallocate(first)
    assert manager.can_allocate(cold)
    scheduler = _scheduler(manager)
    scheduler.add(cold)
    assert scheduler.schedule().sequences == (cold,)


def test_image_boundary_recompute_makes_every_overlapping_page_private():
    manager = BlockManager(32, 4, block_level_mm_prefix=True)
    tokens = [10] + [PAD] * 10 + [11, 12, 13]
    first = _image_sequence(tokens, [10], (MEDIA_A,), 0, None)
    manager.allocate(first)
    _finish_prefill(manager, first)
    second = _image_sequence([10] + [PAD] * 10 + [21, 22, 23], [10], (MEDIA_A,), 1, None)
    copies = manager.allocate(second)
    assert second.num_cached_tokens == 1
    assert copies == ((first.block_table[0], second.block_table[0], 1),)
    assert set(first.block_table).isdisjoint(second.block_table)
    assert all(manager.blocks[block].hash == -1 for block in second.block_table)
