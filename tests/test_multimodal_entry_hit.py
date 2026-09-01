"""Multimodal entry-hit allocation tests."""

from __future__ import annotations

import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.sequence import Sequence

PAD = 151655
SEP = 999
H_A = bytes(range(32))
H_B = bytes(255 - value for value in range(32))


def make_vl_seq(
    manager: BlockManager,
    token_ids: list[int],
    request_id: int,
) -> Sequence:
    pad_count = token_ids.count(PAD)
    return Sequence(
        token_ids,
        block_size=manager.block_size,
        request_id=request_id,
        pixel_values=torch.zeros(pad_count, 3),
        image_grid_thw=torch.tensor([[1, 3, 1], [1, 4, 1]], dtype=torch.long),
        position_ids=torch.arange(len(token_ids), dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1),
        rope_delta=torch.zeros(1, 1),
        image_token_id=PAD,
        image_token_count=pad_count,
        multimodal_media_token_hashes=(H_A, H_B),
        image_merge_size=1,
    )


def dense_manager() -> BlockManager:
    return BlockManager(
        num_blocks=32,
        block_size=4,
        enable_prefix_caching=True,
        block_level_mm_prefix=True,
    )


def store_entry(manager: BlockManager, seq: Sequence) -> None:
    seq.multimodal_prefix_cache_key = "test-media-key"
    seq.num_computed_tokens = seq.num_prompt_tokens
    assert manager.store_multimodal_prefix(seq)


def test_entry_hit_suffix_blocks_carry_hashes() -> None:
    manager = dense_manager()
    first = make_vl_seq(manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11], 0)
    manager.allocate(first)
    store_entry(manager, first)

    second = make_vl_seq(manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [12], 1)
    second.multimodal_prefix_cache_key = "test-media-key"
    manager.allocate(second)

    assert second.num_cached_tokens > 0
    assert second.multimodal_prefix_cache_hit
    for block_index, block_id in enumerate(second.block_table):
        if len(second.block(block_index)) == manager.block_size:
            assert manager.blocks[block_id].hash != -1


def test_entry_hit_boundary_aligned_decode() -> None:
    manager = dense_manager()
    first = make_vl_seq(
        manager,
        [PAD] * 3 + [SEP] + [PAD] * 4 + [11, 12, 13],
        0,
    )
    manager.allocate(first)
    store_entry(manager, first)

    second = make_vl_seq(
        manager,
        [PAD] * 3 + [SEP] + [PAD] * 4 + [21, 22, 23, 24],
        1,
    )
    second.multimodal_prefix_cache_key = "test-media-key"
    manager.allocate(second)

    assert second.num_cached_tokens > 0
    second.append_token(299)
    for token in (300, 301, 302, 303, 304, 305):
        if second.physical_kv_len % manager.block_size != 1:
            manager.copy_on_write(second)
        manager.may_append(second)
        second.append_token(token)
    assert second.physical_kv_len == 19
