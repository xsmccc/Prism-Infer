"""Per-image block index tests (Phase 3: out-of-order / subset reuse).

Pins the contract added by the per-image block index:

- 乱序复用: [图A,图B] 缓存后 [图B] 前缀独立命中 (链式 miss, 索引 hit)
- 排列复用: 两张满块图任意顺序全部索引复用
- 链式优先: 完全重复请求仍走 hash 链 (索引不介入)
- 共享语义: 多个请求复用同一物理块 refcount 正确
- 碰撞安全: 相同 pad 不同图 → 索引不复用
- 小图/混合块: 不满 block 或跨图块的块不进索引
- 池层: 注册/探测/认领/淘汰/残留清理
- 纯文本: 不受影响

布局约定 (block_size=4): 图 A=3 pad, 图 B=4 pad, 使 B 独占整块
([PAD]*3 + [SEP] + [PAD]*4 的第 2 块全部是 B 的 pad)。
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.block_pool import GpuBlockPool
from prism_infer.engine.sequence import Sequence

PAD = 151655
SEP = 999  # 代表 <|vision_end|><|vision_start|> 分隔符，使相邻图成为独立 pad run

# 逐图媒体 SHA256（模拟），两两不同
H_A = bytes(range(32))
H_B = bytes(255 - value for value in range(32))
H_C = bytes(value ^ 0x5A for value in range(32))
H_D = bytes(value ^ 0x33 for value in range(32))


def _surrogate(image_hash: bytes) -> int:
    return int.from_bytes(hashlib.sha256(image_hash).digest()[:8], "little")


def make_vl_seq(
    manager: BlockManager,
    token_ids: list[int],
    pads_per_image: list[int],
    media_hashes: tuple[bytes, ...],
    request_id: int,
) -> Sequence:
    """Build a multimodal sequence with explicit per-image pad counts."""

    pad_count = token_ids.count(PAD)
    n_images = len(media_hashes)
    if pad_count != sum(pads_per_image) or n_images != len(pads_per_image):
        raise ValueError("test pad layout mismatch")
    grid = torch.tensor(
        [[1, pads, 1] for pads in pads_per_image], dtype=torch.long
    )
    pixel = torch.zeros(pad_count, 3)
    return Sequence(
        token_ids,
        block_size=manager.block_size,
        request_id=request_id,
        pixel_values=pixel,
        image_grid_thw=grid,
        position_ids=torch.arange(len(token_ids), dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1),
        rope_delta=torch.zeros(1, 1),
        image_token_id=PAD,
        image_token_count=pad_count,
        multimodal_media_token_hashes=media_hashes,
        image_merge_size=1,
    )


def dense_manager(num_blocks: int, block_size: int = 4) -> BlockManager:
    return BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_caching=True,
        block_level_mm_prefix=True,
    )


# ---------------------------------------------------------------------------
# 池层
# ---------------------------------------------------------------------------

class TestPoolImageIndex:
    KEY = (7, 0)

    def _setup(self):
        pool = GpuBlockPool(num_blocks=4, retain_hashes_on_free=True)
        block = pool.allocate_free()
        pool.register_hash(block.block_id, 111, [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7})
        pool.register_image_owner(block.block_id, self.KEY)
        return pool, block

    def test_register_and_peek(self):
        pool, block = self._setup()
        found = pool.peek_image_block((7, 0), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7})
        assert found is block
        assert pool.peek_image_block((7, 1), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7}) is None
        assert pool.peek_image_block((8, 0), [PAD] * 4, {0: 8, 1: 8, 2: 8, 3: 8}) is None

    def test_claim_then_retain_revives_cached_block(self):
        pool, block = self._setup()
        pool.release_reference(block.block_id)
        assert block.ref_count == 0
        assert block.block_id in pool.cached_block_ids
        claimed = pool.claim_image_block((7, 0), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7})
        assert claimed is block
        assert block.ref_count == 0  # claim 不动引用计数
        revived = pool.retain(claimed.block_id)  # 调用方 retain 复活
        assert revived is block
        assert block.ref_count == 1
        assert block.block_id not in pool.cached_block_ids

    def test_claim_skips_stale_entries(self):
        pool = GpuBlockPool(num_blocks=4, retain_hashes_on_free=True)
        stale = pool.allocate_free()  # 先入队: 无内容
        pool.register_image_owner(stale.block_id, (7, 0))
        good = pool.allocate_free()
        pool.register_hash(good.block_id, 111, [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7})
        pool.register_image_owner(good.block_id, (7, 0))
        claimed = pool.claim_image_block((7, 0), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7})
        assert claimed is good
        # 残留条目被逐出索引
        assert stale.block_id not in pool.image_to_block_ids.get((7, 0), ())

    def test_eviction_removes_index(self):
        pool, block = self._setup()
        pool.release_reference(block.block_id)
        evicted = pool._evict_oldest_cached()
        assert evicted is block
        assert pool.peek_image_block((7, 0), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7}) is None
        assert block.image_owner is None

    def test_clear_hash_removes_index(self):
        pool, block = self._setup()
        pool.clear_hash(block.block_id)
        assert pool.peek_image_block((7, 0), [PAD] * 4, {0: 7, 1: 7, 2: 7, 3: 7}) is None


# ---------------------------------------------------------------------------
# BlockManager 层
# ---------------------------------------------------------------------------

class TestImageIndexReuse:
    def test_leading_image_reuse_across_contexts(self):
        """[图A,图B,Q] 缓存后, [图B,Q2] 前缀独立命中图B的满块。"""

        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(
            manager,
            [PAD] * 3 + [SEP] + [PAD] * 4 + [11],
            [3, 4],
            (H_A, H_B),
            0,
        )
        manager.allocate(first)
        assert first.num_cached_tokens == 0  # 冷请求

        second = make_vl_seq(manager, [PAD] * 4 + [21, 22], [4], (H_B,), 1)
        candidate = manager.probe_multimodal_prefix(second, would_hydrate_visual=False)
        assert candidate == 4  # 图B 独占第 1 块, 对齐图边界
        manager.allocate(second)
        assert second.num_cached_tokens == 4
        assert second.block_table[0] == first.block_table[1]
        assert manager._block_level_image_index_reused_blocks == 1

    def test_permutation_two_full_images(self):
        """[A(3),B(4),C(3),D(4)] 缓存后, [D,B] 两张满块图全部索引复用。"""

        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(
            manager,
            [PAD] * 3 + [SEP] + [PAD] * 4 + [PAD] * 3 + [SEP] + [PAD] * 4 + [11],
            [3, 4, 3, 4],
            (H_A, H_B, H_C, H_D),
            0,
        )
        manager.allocate(first)
        # 块布局: [A混合, B满, C混合, D满, 尾]
        assert first.block_table[1] != first.block_table[3]

        second = make_vl_seq(
            manager, [PAD] * 4 + [PAD] * 4 + [21], [4, 4], (H_D, H_B), 1
        )
        manager.allocate(second)
        assert second.num_cached_tokens == 8
        assert second.block_table[0] == first.block_table[3]  # D
        assert second.block_table[1] == first.block_table[1]  # B
        assert manager._block_level_image_index_reused_blocks == 2

    def test_exact_repeat_prefers_chain(self):
        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11], [3, 4], (H_A, H_B), 0
        )
        manager.allocate(first)
        repeat = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11], [3, 4], (H_A, H_B), 1
        )
        manager.allocate(repeat)
        assert repeat.num_cached_tokens == 8
        assert manager._block_level_image_index_reused_blocks == 0  # 全走链

    def test_collision_safe_no_reuse(self):
        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(manager, [PAD] * 4 + [11], [4], (H_A,), 0)
        manager.allocate(first)
        other = make_vl_seq(manager, [PAD] * 4 + [12], [4], (H_C,), 1)
        manager.allocate(other)
        assert other.num_cached_tokens == 0
        assert manager._block_level_image_index_reused_blocks == 0

    def test_shared_claims_refcount(self):
        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11], [3, 4], (H_A, H_B), 0
        )
        manager.allocate(first)
        second = make_vl_seq(manager, [PAD] * 4 + [21], [4], (H_B,), 1)
        manager.allocate(second)
        third = make_vl_seq(manager, [PAD] * 4 + [22], [4], (H_B,), 2)
        manager.allocate(third)
        shared_id = first.block_table[1]
        assert second.block_table[0] == shared_id
        assert third.block_table[0] == shared_id
        assert manager.blocks[shared_id].ref_count == 3

    def test_tiny_image_not_indexed(self):
        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(manager, [PAD] * 2 + [SEP] + [11, 12], [2], (H_A,), 0)
        manager.allocate(first)
        second = make_vl_seq(manager, [PAD] * 2 + [21, 22], [2], (H_A,), 1)
        manager.allocate(second)
        # 不满一个 block 的图没有独立满块, 索引不可用
        assert manager._block_level_image_index_reused_blocks == 0

    def test_mixed_block_not_owned(self):
        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(manager, [PAD, PAD, 11, 12, 13], [2], (H_A,), 0)
        manager.allocate(first)
        second = make_vl_seq(manager, [PAD, PAD, 21, 22, 23], [2], (H_A,), 1)
        manager.allocate(second)
        assert manager._block_level_image_index_reused_blocks == 0

    def test_eviction_drops_index_claims(self):
        manager = dense_manager(num_blocks=2)
        first = make_vl_seq(manager, [PAD] * 4 + [11], [4], (H_B,), 0)
        manager.allocate(first)
        first_block = first.block_table[0]
        manager.deallocate(first)  # 满块保留为 cached, 索引仍在

        # 压力分配迫使 cached 块淘汰, 索引必须同步清除
        seq_b = Sequence([5, 6, 7, 8, 9, 10, 11, 12], block_size=4, request_id=1)
        manager.allocate(seq_b)
        assert first_block not in manager._gpu_pool.cached_block_ids
        sur = _surrogate(H_B)
        mm = {0: sur, 1: sur, 2: sur, 3: sur}
        assert manager._gpu_pool.peek_image_block(sur, [PAD] * 4, mm) is None

        manager.deallocate(seq_b)  # 腾出容量给 late
        late = make_vl_seq(manager, [PAD] * 4 + [77], [4], (H_B,), 9)
        manager.allocate(late)
        assert manager._block_level_image_index_reused_blocks == 0

    def test_text_only_unaffected(self):
        manager = dense_manager(num_blocks=8)
        first = Sequence([1, 2, 3, 4, 5], block_size=4, request_id=0)
        manager.allocate(first)
        repeat = Sequence([1, 2, 3, 4, 6], block_size=4, request_id=1)
        manager.allocate(repeat)
        assert repeat.num_cached_tokens == 4
        assert manager._block_level_image_index_reused_blocks == 0


class TestEntryHitSuffixHashing:
    def _store_entry(self, manager, first, key="test-media-key"):
        first.multimodal_prefix_cache_key = key
        first.num_computed_tokens = first.num_prompt_tokens  # 模拟 prefill 完成
        assert manager.store_multimodal_prefix(first)

    def test_entry_hit_suffix_blocks_carry_hashes(self):
        """入口整段复用分配的 suffix 满块必须带 hash (may_append 不变量)。"""

        manager = dense_manager(num_blocks=32)
        first = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11], [3, 4], (H_A, H_B), 0
        )
        manager.allocate(first)
        self._store_entry(manager, first)

        # 同图同布局 + 不同问题 -> entry 命中
        second = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [12], [3, 4], (H_A, H_B), 1
        )
        second.multimodal_prefix_cache_key = "test-media-key"
        manager.allocate(second)
        assert second.num_cached_tokens > 0
        assert second.multimodal_prefix_cache_hit
        # 所有满块 (含 suffix) 都有 hash
        for block_index, block_id in enumerate(second.block_table):
            tokens = second.block(block_index)
            if len(tokens) == manager.block_size:
                assert manager.blocks[block_id].hash != -1

    def test_entry_hit_boundary_aligned_decode(self):
        """boundary-aligned prompt + entry 命中 + decode 首 token 不崩。"""

        manager = dense_manager(num_blocks=32)
        # 12 tokens = 3 满块 (boundary-aligned)
        first = make_vl_seq(
            manager, [PAD] * 3 + [SEP] + [PAD] * 4 + [11, 12, 13], [3, 4], (H_A, H_B), 0
        )
        manager.allocate(first)
        self._store_entry(manager, first)

        second = make_vl_seq(
            manager,
            [PAD] * 3 + [SEP] + [PAD] * 4 + [21, 22, 23, 24],
            [3, 4],
            (H_A, H_B),
            1,
        )
        second.multimodal_prefix_cache_key = "test-media-key"
        manager.allocate(second)
        assert second.num_cached_tokens > 0
        # 生产路径: 首 token 在 prefill 收尾时采样, 不经过 may_append ->
        # 后续第一次 may_append 直接是 rem==1 (boundary-aligned 崩溃现场)
        second.append_token(299)
        for token in (300, 301, 302, 303, 304, 305):
            if second.physical_kv_len % manager.block_size != 1:
                manager.copy_on_write(second)
            manager.may_append(second)
            second.append_token(token)
        assert second.physical_kv_len == 12 + 7

    def test_same_image_different_in_image_offset_not_reused(self):
        """同一张图的不同图内切片不能互换 (mrope 位置是图内 2D 坐标)。"""

        manager = dense_manager(num_blocks=16)
        # 图 A 独占块 0,1 (图内块序 0,1)
        first = make_vl_seq(manager, [PAD] * 8 + [11], [8], (H_A,), 0)
        manager.allocate(first)
        # 图 A 前移一格: 块 0 混合, 块 1 (tokens 4-7) 图内 offset 3, 不对齐
        second = make_vl_seq(manager, [1, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, 22], [8], (H_A,), 1)
        manager.allocate(second)
        assert second.block_table[1] != first.block_table[1]
        assert manager._block_level_image_index_reused_blocks == 0

    def test_block_zero_aligned_reuse_prefers_chain(self):
        """块 0 对齐复用: content-only 链 hash 直接命中, 不经过索引。"""

        manager = dense_manager(num_blocks=16)
        first = make_vl_seq(manager, [PAD] * 8 + [11], [8], (H_A,), 0)
        manager.allocate(first)
        second = make_vl_seq(manager, [PAD] * 4 + [21, 22], [4], (H_A,), 1)
        candidate = manager.probe_multimodal_prefix(second, would_hydrate_visual=False)
        assert candidate == 4
        manager.allocate(second)
        assert second.num_cached_tokens == 4
        assert second.block_table[0] == first.block_table[0]  # 图内块序 0
        assert manager._block_level_reused_blocks == 1
        assert manager._block_level_image_index_reused_blocks == 0
