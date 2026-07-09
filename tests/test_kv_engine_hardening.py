"""P4.5 KV Engine hardening 回归测试。"""

from types import SimpleNamespace

import pickle

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.sequence import Sequence
from prism_infer.layers.attention import store_kvcache


def _sequence(token_ids: list[int], block_size: int) -> Sequence:
    """构造与 BlockManager block_size 一致的轻量 Sequence。"""

    old_block_size = Sequence.block_size
    Sequence.set_block_size(block_size)
    try:
        return Sequence(token_ids)
    finally:
        Sequence.set_block_size(old_block_size)


def test_store_kvcache_eager_writes_canonical_4d_paged_cache() -> None:
    """CPU fallback 必须按 flat slot 写入 4D paged cache。"""

    num_blocks = 3
    block_size = 4
    num_kv_heads = 2
    head_dim = 3
    key = torch.arange(5 * num_kv_heads * head_dim, dtype=torch.float32).view(
        5, num_kv_heads, head_dim
    )
    value = key + 1000
    k_cache = torch.full(
        (num_blocks, block_size, num_kv_heads, head_dim),
        -1.0,
        dtype=torch.float32,
    )
    v_cache = torch.full_like(k_cache, -1.0)
    slot_mapping = torch.tensor([0, 3, 4, 9, -1], dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    expected_k = torch.full_like(k_cache, -1.0)
    expected_v = torch.full_like(v_cache, -1.0)
    for src_idx, slot in enumerate(slot_mapping.tolist()):
        if slot == -1:
            continue
        block_id = slot // block_size
        offset = slot % block_size
        expected_k[block_id, offset] = key[src_idx]
        expected_v[block_id, offset] = value[src_idx]

    k_diff = (k_cache - expected_k).abs()
    v_diff = (v_cache - expected_v).abs()
    print(f"store key input shape: {list(key.shape)}")
    print(f"store cache shape: {list(k_cache.shape)}")
    print(f"store slot_mapping: {slot_mapping.tolist()}")
    print(f"store k_cache max diff: {k_diff.max().item():.6e}")
    print(f"store v_cache max diff: {v_diff.max().item():.6e}")

    assert k_diff.max().item() == 0.0
    assert v_diff.max().item() == 0.0
    print("KV layout 4D eager store: PASS")


def test_block_manager_deallocate_clears_hash_index() -> None:
    """释放最后一个引用后，prefix hash 不能继续指向 free block。"""

    block_size = 4
    old_block_size = Sequence.block_size
    Sequence.set_block_size(block_size)
    try:
        manager = BlockManager(num_blocks=4, block_size=block_size)
        seq = Sequence([1, 2, 3, 4])
        manager.allocate(seq)
        assert len(seq.block_table) == 1
        block_id = seq.block_table[0]
        block_hash = manager.blocks[block_id].hash
        assert manager.hash_to_block_id[block_hash] == block_id

        manager.deallocate(seq)

        print(f"deallocated block id: {block_id}")
        print(f"released block hash: {block_hash}")
        print(f"hash index keys after deallocate: {list(manager.hash_to_block_id.keys())}")
        print(f"free block ids after deallocate: {sorted(manager.free_block_id_set)}")

        assert block_hash not in manager.hash_to_block_id
        assert block_id in manager.free_block_id_set
        assert manager.blocks[block_id].ref_count == 0
        print("BlockManager hash cleanup: PASS")
    finally:
        Sequence.set_block_size(old_block_size)


def test_block_manager_rejects_sequence_block_size_mismatch() -> None:
    """Sequence 页表粒度与 BlockManager 物理粒度不一致时必须显式失败。"""

    old_block_size = Sequence.block_size
    Sequence.set_block_size(8)
    try:
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Sequence.block_size must match"):
            manager.allocate(seq)
        print("BlockManager block size mismatch gate: PASS")
    finally:
        Sequence.set_block_size(old_block_size)


def test_block_manager_swap_uses_separate_cpu_block_table() -> None:
    """swap_out 后 GPU block_table 必须清空，CPU block id 只能进入 cpu_block_table。"""

    block_size = 4
    old_block_size = Sequence.block_size
    Sequence.set_block_size(block_size)
    try:
        manager = BlockManager(num_blocks=4, block_size=block_size, num_cpu_blocks=4)
        seq = Sequence([10, 11, 12, 13, 14, 15])
        manager.allocate(seq)
        gpu_table_before = list(seq.block_table)

        swap_out_map = manager.swap_out(seq)
        print(f"swap out map: {swap_out_map}")
        print(f"gpu block_table after swap_out: {seq.block_table}")
        print(f"cpu block_table after swap_out: {seq.cpu_block_table}")

        assert seq.block_table == []
        assert len(seq.cpu_block_table) == len(gpu_table_before)
        assert [gpu_id for gpu_id, _ in swap_out_map] == gpu_table_before
        assert manager.can_swap_in(seq)
        assert len(seq.cpu_block_hashes) == len(seq.cpu_block_table)
        assert len(seq.cpu_block_token_ids) == len(seq.cpu_block_table)

        swap_in_map = manager.swap_in(seq)
        print(f"swap in map: {swap_in_map}")
        print(f"gpu block_table after swap_in: {seq.block_table}")
        print(f"cpu block_table after swap_in: {seq.cpu_block_table}")

        assert seq.cpu_block_table == []
        assert len(seq.block_table) == len(gpu_table_before)
        assert [cpu_id for cpu_id, _ in swap_in_map] == [cpu_id for _, cpu_id in swap_out_map]
        for block_id in seq.block_table:
            assert block_id in manager.used_block_ids
            assert block_id not in manager.free_block_id_set
        print("BlockManager swap table split: PASS")
    finally:
        Sequence.set_block_size(old_block_size)


def test_block_manager_swap_in_restores_hash_from_metadata_after_decode_pickle() -> None:
    """swap_in 不能依赖 decode 反序列化对象保留完整 token_ids。"""

    block_size = 4
    old_block_size = Sequence.block_size
    Sequence.set_block_size(block_size)
    try:
        manager = BlockManager(num_blocks=4, block_size=block_size, num_cpu_blocks=4)
        seq = Sequence([10, 11, 12, 13, 14])
        manager.allocate(seq)
        full_block_hash = manager.blocks[seq.block_table[0]].hash
        full_block_tokens = list(manager.blocks[seq.block_table[0]].token_ids)
        seq.append_token(99)
        swap_out_map = manager.swap_out(seq)

        restored = pickle.loads(pickle.dumps(seq))
        assert not hasattr(restored, "token_ids")

        swap_in_map = manager.swap_in(restored)
        restored_block_id = restored.block_table[0]

        print(f"swap out map before pickle: {swap_out_map}")
        print(f"swap in map after pickle: {swap_in_map}")
        print(f"restored full block hash: {manager.blocks[restored_block_id].hash}")
        print(f"restored full block tokens: {manager.blocks[restored_block_id].token_ids}")

        assert manager.blocks[restored_block_id].hash == full_block_hash
        assert manager.blocks[restored_block_id].token_ids == full_block_tokens
        assert manager.hash_to_block_id[full_block_hash] == restored_block_id
        print("BlockManager swap hash metadata restore: PASS")
    finally:
        Sequence.set_block_size(old_block_size)


def test_prepare_prefill_rejects_prefix_cache_before_attention() -> None:
    """prefix-cache prefill 未实现前，应在 ModelRunner 准备阶段显式失败。"""

    block_size = 4
    old_block_size = Sequence.block_size
    Sequence.set_block_size(block_size)
    try:
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = block_size
        runner.config = SimpleNamespace(enable_chunked_prefill=False)
        seq = Sequence([1, 2, 3, 4, 5, 6])
        seq.block_table = [0, 1]
        seq.num_cached_tokens = block_size

        with pytest.raises(RuntimeError, match="paged prefix-cache prefill is not supported"):
            runner.prepare_prefill([seq])
        print("prefix-cache prefill early gate: PASS")
    finally:
        Sequence.set_block_size(old_block_size)
