"""Block-level mm-aware prefix matching (dense mode) tests.

Pins the dense-mode prefix-cache contract added by the block-level upgrade:

- 子集复用: 图1-8 请求缓存的 block 被 图1-4 请求逐块复用
- 同图不同问: 纯图 block 命中, 问题 block miss (部分复用)
- 多轮追问: 图 + Q1 缓存可被 图 + Q1 + A1 + Q4 复用 (文本 hash 链)
- 碰撞安全: 相同 pad token 序列但媒体不同 → 绝不复用
- 布局变化: 链式失效, 不错误复用
- 无媒体身份的多模态序列: 不哈希 (安全回退)
- pool lazy retention: deallocate 后 hash 保留, 可复活, 压力下淘汰
- probe 语义: 部分命中 / 整段命中 / hydration skip 判定
- decode boundary-aligned: prompt 整除 block_size 时可正常 decode
- swap 往返保留 mm 元数据
"""

from __future__ import annotations

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.block_pool import NO_BLOCK_HASH
from prism_infer.engine.sequence import Sequence

PAD = 151655
SEP = 999  # 代表 <|vision_end|><|vision_start|> 分隔符，使相邻图成为独立 pad run

# 逐图媒体 SHA256（模拟），两两不同
H_A = bytes(range(32))
H_B = bytes(255 - value for value in range(32))
H_C = bytes(value ^ 0x5A for value in range(32))


def make_mm_seq(
    manager: BlockManager,
    token_ids: list[int],
    media_hashes: tuple[bytes, ...] | None,
    request_id: int,
) -> Sequence:
    """Build a multimodal sequence with per-image media identity."""
    return Sequence(
        token_ids,
        block_size=manager.block_size,
        request_id=request_id,
        position_ids=(
            torch.arange(len(token_ids), dtype=torch.long)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        ),
        rope_delta=torch.zeros(1, 1),
        image_token_id=PAD,
        image_token_count=token_ids.count(PAD),
        multimodal_media_token_hashes=media_hashes,
    )


def decode_tokens(manager: BlockManager, seq: Sequence, tokens: list[int]) -> None:
    """Replicate the scheduler decode loop: CoW -> may_append -> append_token."""
    for token in tokens:
        if seq.physical_kv_len % manager.block_size != 1:
            manager.copy_on_write(seq)
        manager.may_append(seq)
        seq.append_token(token)


def dense_manager(num_blocks: int, block_size: int = 4) -> BlockManager:
    return BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_caching=True,
        block_level_mm_prefix=True,
    )


# ---------------------------------------------------------------------------
# 1. 子集复用
# ---------------------------------------------------------------------------

def test_subset_reuse_图1_8_then_图1_4() -> None:
    """图1-8 请求缓存的 block 必须被 图1-4 请求逐块复用。"""

    manager = dense_manager(num_blocks=16)
    full = make_mm_seq(manager, [PAD] * 4 + [SEP] + [PAD] * 4 + [11, 12, 13], (H_A, H_B), 0)
    manager.allocate(full)
    assert full.num_cached_tokens == 0  # 冷请求

    subset = make_mm_seq(manager, [PAD] * 4 + [21, 22, 23], (H_A,), 1)
    manager.allocate(subset)

    print(f"full table: {full.block_table}")
    print(f"subset table: {subset.block_table}")
    print(f"subset cached tokens: {subset.num_cached_tokens}")

    assert subset.block_table[0] == full.block_table[0]  # 图1 的 block 复用
    assert subset.num_cached_tokens == 4
    assert subset.block_table[1] != full.block_table[1]  # 问题部分是新块


def test_same_image_different_question_partial_reuse() -> None:
    """同图不同问: 纯图 block 命中, 问题 block miss。"""

    manager = dense_manager(num_blocks=16)
    first = make_mm_seq(manager, [PAD] * 4 + [31, 32, 33], (H_A,), 0)
    manager.allocate(first)

    second = make_mm_seq(manager, [PAD] * 4 + [41, 42, 43, 44], (H_A,), 1)
    manager.allocate(second)

    print(f"first table: {first.block_table}")
    print(f"second table: {second.block_table}")
    print(f"second cached tokens: {second.num_cached_tokens}")

    assert second.block_table[0] == first.block_table[0]
    assert second.num_cached_tokens == 4
    assert len(second.block_table) == 2


# ---------------------------------------------------------------------------
# 2. 多轮追问
# ---------------------------------------------------------------------------

def test_multiturn_text_growth_reuse() -> None:
    """图+Q1 缓存的 block 必须被 图+Q1+A1+Q4 请求复用。"""

    manager = dense_manager(num_blocks=16)
    # R1: 图 + Q1 (7 tokens -> block 0 满, block 1 是 3/4 的 partial)
    r1 = make_mm_seq(manager, [PAD] * 4 + [51, 52, 53], (H_A,), 0)
    manager.allocate(r1)
    decode_tokens(manager, r1, [61, 62, 63, 64])  # 生成 A1, block 1 完成时注册 hash
    r1_table = list(r1.block_table)
    manager.deallocate(r1)  # 池层 lazy retention 保留满块 hash

    # R2: 图 + Q1 + A1 + Q4
    r2 = make_mm_seq(
        manager,
        [PAD] * 4 + [51, 52, 53] + [61, 62, 63, 64] + [71, 72],
        (H_A,),
        1,
    )
    manager.allocate(r2)

    print(f"r1 table: {r1_table}")
    print(f"r2 table: {r2.block_table}")
    print(f"r2 cached tokens: {r2.num_cached_tokens}")

    assert r2.block_table[0] == r1_table[0]  # 图 block 复用
    assert r2.block_table[1] == r1_table[1]  # Q1+A1 首 token 的 block 复用
    assert r2.num_cached_tokens == 8
    assert r2.block_table[2] != r1_table[2]  # Q4 部分是新块


# ---------------------------------------------------------------------------
# 3. 碰撞安全
# ---------------------------------------------------------------------------

def test_collision_safety_same_pads_different_media() -> None:
    """相同 pad token 序列但媒体不同 → 绝不复用。"""

    manager = dense_manager(num_blocks=16)
    first = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], (H_A,), 0)
    manager.allocate(first)

    impostor = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], (H_B,), 1)
    manager.allocate(impostor)

    print(f"first table: {first.block_table}")
    print(f"impostor table: {impostor.block_table}")

    assert impostor.block_table[0] != first.block_table[0]
    assert impostor.num_cached_tokens == 0


def test_multimodal_without_media_identity_never_hashes() -> None:
    """没有逐图媒体身份的多模态序列不参与 block 哈希 (安全回退)。"""

    manager = dense_manager(num_blocks=16)
    first = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], None, 0)
    manager.allocate(first)
    second = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], None, 1)
    manager.allocate(second)

    assert first.block_table != second.block_table
    assert second.num_cached_tokens == 0
    assert all(
        manager.blocks[block_id].hash == NO_BLOCK_HASH
        for block_id in (*first.block_table, *second.block_table)
    )


# ---------------------------------------------------------------------------
# 4. 布局变化
# ---------------------------------------------------------------------------

def test_layout_change_cascades_miss() -> None:
    """标签变化使前缀块 hash 变化, 后续块链式失效, 不错误复用。"""

    manager = dense_manager(num_blocks=16)
    layout_a = make_mm_seq(manager, [10, PAD, PAD, PAD, 11, 12], (H_A,), 0)
    manager.allocate(layout_a)

    layout_b = make_mm_seq(manager, [20, PAD, PAD, PAD, 11, 12], (H_A,), 1)
    manager.allocate(layout_b)

    print(f"layout_a table: {layout_a.block_table}")
    print(f"layout_b table: {layout_b.block_table}")

    assert layout_b.num_cached_tokens == 0
    assert layout_b.block_table[0] != layout_a.block_table[0]


def test_pure_image_block_reuses_across_different_suffix() -> None:
    """纯图 block 不含标签, 不同后缀下仍命中。"""

    manager = dense_manager(num_blocks=16)
    first = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], (H_A,), 0)
    manager.allocate(first)

    other = make_mm_seq(manager, [PAD] * 4 + [99, 98, 97, 96], (H_A,), 1)
    manager.allocate(other)

    assert other.block_table[0] == first.block_table[0]
    assert other.num_cached_tokens == 4


# ---------------------------------------------------------------------------
# 5. Pool lazy retention
# ---------------------------------------------------------------------------

def test_pool_lazy_retention_survives_deallocate_and_revives() -> None:
    """deallocate 后满块 hash 保留为 cached, 新请求可复活复用。"""

    manager = dense_manager(num_blocks=4)
    first = Sequence([1, 2, 3, 4], block_size=4, request_id=0)
    manager.allocate(first)
    first_block = first.block_table[0]
    first_hash = manager.blocks[first_block].hash
    manager.deallocate(first)

    # 块仍是 cached 状态: 可回收但不丢 hash
    assert manager.blocks[first_block].hash == first_hash
    assert manager.hash_to_block_id[first_hash] == first_block

    second = Sequence([1, 2, 3, 4], block_size=4, request_id=1)
    manager.allocate(second)
    print(f"first block: {first_block}, second table: {second.block_table}")
    assert second.block_table[0] == first_block
    assert second.num_cached_tokens == 4


def test_pool_cached_blocks_evict_under_pressure() -> None:
    """压力下 cached 块按 FIFO 淘汰, 淘汰必须清 hash。"""

    manager = dense_manager(num_blocks=2)
    seq_a = Sequence([1, 2, 3, 4], block_size=4, request_id=0)
    manager.allocate(seq_a)
    hash_a = manager.blocks[seq_a.block_table[0]].hash
    manager.deallocate(seq_a)  # block 0 cached; 1 个真 free

    seq_b = Sequence([5, 6, 7, 8, 9, 10, 11, 12], block_size=4, request_id=1)
    manager.allocate(seq_b)  # 需要 2 个新块: 先用真 free, 再淘汰 cached block 0

    print(f"seq_b table: {seq_b.block_table}")
    assert hash_a not in manager.hash_to_block_id
    assert seq_b.block_table[0] == 1  # 真 free 块优先
    assert seq_b.block_table[1] == 0  # 被淘汰后重新分配


# ---------------------------------------------------------------------------
# 6. Probe 语义
# ---------------------------------------------------------------------------

def test_probe_block_level_partial_and_full_hits() -> None:
    """probe 返回块级最长缓存前缀; hydration skip 只统计整段视觉命中。"""

    manager = dense_manager(num_blocks=16)
    full = make_mm_seq(manager, [PAD] * 8 + [11, 12, 13], (H_A,), 0)
    manager.allocate(full)

    # 整段命中: 同图不同问 -> probe == boundary(8), hydration skip +1
    same = make_mm_seq(manager, [PAD] * 8 + [21, 22, 23], (H_A,), 1)
    candidate = manager.probe_multimodal_prefix(same, would_hydrate_visual=True)
    assert candidate == 8
    assert same.multimodal_prefix_pre_admission_hit is True
    assert manager.multimodal_prefix_cache_metadata()["visual_hydration_skips"] == 1

    # 部分命中: 第二张图未缓存 -> probe == 4 (< boundary 9), 不记 hydration skip
    partial = make_mm_seq(
        manager,
        [PAD] * 4 + [SEP] + [PAD] * 4 + [31, 32, 33],
        (H_A, H_C),
        2,
    )
    candidate = manager.probe_multimodal_prefix(partial, would_hydrate_visual=True)
    assert candidate == 4
    assert partial.multimodal_prefix_pre_admission_hit is True
    assert manager.multimodal_prefix_cache_metadata()["visual_hydration_skips"] == 1


def test_snap_to_image_boundary_prevents_mid_image_split() -> None:
    """命中停在图片 span 内部时必须 snap 到图边界并私有化共享块。"""

    manager = dense_manager(num_blocks=16)
    first = make_mm_seq(manager, [PAD] * 6 + [11, 12], (H_A,), 0)  # run [0, 6), 2 满块
    manager.allocate(first)

    # 走查命中 block 0 (candidate=4) 但 4 落在 run [0,6) 内部 -> snap 到 0
    other = make_mm_seq(manager, [PAD] * 6 + [13, 14, 15, 16], (H_A,), 1)
    pairs = manager.allocate(other)

    print(f"first table: {first.block_table}")
    print(f"other table: {other.block_table}")
    print(f"other cached tokens: {other.num_cached_tokens}")
    print(f"copy pairs: {pairs}")

    assert other.num_cached_tokens == 0  # 不能切开图片 span, 整段重算
    assert len(pairs) == 1  # 共享块已私有化 (CoW)
    assert other.block_table[0] != first.block_table[0]
    assert manager.blocks[first.block_table[0]].ref_count == 1  # 缓存未被污染


# ---------------------------------------------------------------------------
# 7. Decode boundary-aligned 修复
# ---------------------------------------------------------------------------

def test_decode_after_boundary_aligned_prompt() -> None:
    """prompt 整除 block_size 时 decode 第一步不得 raise (历史潜在 bug)。"""

    manager = dense_manager(num_blocks=8)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], block_size=4, request_id=0)
    manager.allocate(seq)
    decode_tokens(manager, seq, [9, 10, 11])
    print(f"table: {seq.block_table}, hashes: {[manager.blocks[b].hash for b in seq.block_table]}")
    assert seq.num_tokens == 11
    assert manager.blocks[seq.block_table[1]].hash != NO_BLOCK_HASH


def test_decode_after_boundary_aligned_shared_prefix() -> None:
    """boundary-aligned + 共享尾块: CoW 后 decode 不得 raise。"""

    manager = dense_manager(num_blocks=8)
    first = Sequence([1, 2, 3, 4, 5, 6, 7, 8], block_size=4, request_id=0)
    manager.allocate(first)
    second = Sequence([1, 2, 3, 4, 5, 6, 7, 8], block_size=4, request_id=1)
    manager.allocate(second)
    assert second.block_table == first.block_table  # 整段共享
    decode_tokens(manager, second, [9, 10, 11])
    print(f"second table: {second.block_table}")
    # 尾块 CoW 分离, 且 first 的缓存不受污染
    assert second.block_table[1] != first.block_table[1]


# ---------------------------------------------------------------------------
# 8. Swap 往返
# ---------------------------------------------------------------------------

def test_swap_roundtrip_keeps_mm_metadata() -> None:
    """swap_out/swap_in 后 mm block 的 hash 与 mm 元数据可恢复。"""

    manager = BlockManager(
        num_blocks=8,
        block_size=4,
        num_cpu_blocks=8,
        enable_prefix_caching=True,
        block_level_mm_prefix=True,
    )
    seq = make_mm_seq(manager, [PAD] * 4 + [11, 12, 13], (H_A,), 0)
    manager.allocate(seq)
    full_hash = manager.blocks[seq.block_table[0]].hash
    full_mm = dict(manager.blocks[seq.block_table[0]].mm_token_hashes or {})

    manager.swap_out(seq)
    assert seq.block_table == []
    manager.swap_in(seq)

    print(f"restored table: {seq.block_table}")
    assert manager.blocks[seq.block_table[0]].hash == full_hash
    assert manager.blocks[seq.block_table[0]].mm_token_hashes == full_mm


# ---------------------------------------------------------------------------
# 9. 视觉 payload 范围切片 (纯函数, CPU)
# ---------------------------------------------------------------------------

def test_image_payload_split_for_range_pure_math() -> None:
    """整图子集切片: patch 行、token 行、切片边界正确。"""

    pixel_values = torch.arange(320 * 3, dtype=torch.float32).view(320, 3)
    grid = torch.tensor([[1, 8, 8], [1, 16, 16]])  # 图1: 64 patches / 16 tokens; 图2: 256 / 64
    merge_size = 2
    token_ids = [PAD] * 16 + [SEP] + [PAD] * 64 + [1, 2, 3]
    image_token_id = PAD

    from prism_infer.engine.sequence import split_image_payloads_for_range

    # 范围 [0, 16): 只含图1
    payload, sliced_grid, in_range = split_image_payloads_for_range(
        pixel_values=pixel_values,
        grid=grid,
        merge_size=merge_size,
        token_ids=token_ids,
        image_token_id=image_token_id,
        range_start=0,
        range_end=16,
    )
    assert in_range == 16
    assert sliced_grid.tolist() == [[1, 8, 8]]
    assert payload.shape[0] == 64

    # 范围 [17, 81): 只含图2
    payload, sliced_grid, in_range = split_image_payloads_for_range(
        pixel_values=pixel_values,
        grid=grid,
        merge_size=merge_size,
        token_ids=token_ids,
        image_token_id=image_token_id,
        range_start=17,
        range_end=81,
    )
    assert in_range == 64
    assert sliced_grid.tolist() == [[1, 16, 16]]
    assert payload.shape[0] == 256

    # 范围 [0, 81): 两图全含
    payload, sliced_grid, in_range = split_image_payloads_for_range(
        pixel_values=pixel_values,
        grid=grid,
        merge_size=merge_size,
        token_ids=token_ids,
        image_token_id=image_token_id,
        range_start=0,
        range_end=81,
    )
    assert in_range == 80
    assert sliced_grid.tolist() == [[1, 8, 8], [1, 16, 16]]
    assert payload.shape[0] == 320


def test_image_payload_split_rejects_mid_image_range() -> None:
    """切开单张图内部的范围必须报错 (而非静默错配)。"""

    pixel_values = torch.zeros(320, 3)
    grid = torch.tensor([[1, 8, 8], [1, 16, 16]])
    token_ids = [PAD] * 16 + [SEP] + [PAD] * 64 + [1, 2, 3]
    from prism_infer.engine.sequence import split_image_payloads_for_range

    with pytest.raises(ValueError, match="splits image"):
        split_image_payloads_for_range(
            pixel_values=pixel_values,
            grid=grid,
            merge_size=2,
            token_ids=token_ids,
            image_token_id=PAD,
            range_start=8,  # 图1 内部
            range_end=81,
        )
