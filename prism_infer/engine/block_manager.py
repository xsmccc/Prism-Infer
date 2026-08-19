"""Paged KV-cache allocation, prefix sharing, compaction, and swap management."""

import copy
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import xxhash

from prism_infer.engine.block_pool import (
    NO_BLOCK_HASH,
    Block,
    CpuBlockPool,
    GpuBlockPool,
)
from prism_infer.engine.kv_compaction_coordinator import KVCompactionCoordinator
from prism_infer.engine.kv_layout import (
    KV_LAYOUT_VISUAL_COMPACT,
    KVCacheLayoutDescriptor,
    KVCompactionPlan,
)
from prism_infer.engine.sequence import Sequence


@dataclass(slots=True)
class _MultimodalPrefixCacheEntry:
    """One compacted multimodal prefix retained by physical page reference."""

    key: str
    prompt_prefix_token_ids: tuple[int, ...]
    logical_prefix_len: int
    physical_prefix_len: int
    retained_original_positions: tuple[int, ...]
    block_ids: tuple[int, ...]
    kv_dtype: str
    compression_record: dict[str, object]
    benefit_tokens: int
    lifetime_hits: int = 0
    tail_clone_block_ids: list[int] = field(default_factory=list)

    @property
    def resident_blocks(self) -> int:
        return len(self.block_ids) + len(self.tail_clone_block_ids)

    @property
    def benefit_per_block(self) -> float:
        return self.benefit_tokens * (1 + self.lifetime_hits) / self.resident_blocks


class BlockManager:
    """Own physical KV blocks and the prefix-cache residency index."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_cpu_blocks: int = 0,
        *,
        enable_prefix_caching: bool = True,
        block_level_mm_prefix: bool = False,
    ):
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
        if not isinstance(enable_prefix_caching, bool):
            raise TypeError("enable_prefix_caching must be bool")
        if not isinstance(block_level_mm_prefix, bool):
            raise TypeError("block_level_mm_prefix must be bool")
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.block_level_mm_prefix = block_level_mm_prefix
        self._gpu_pool = GpuBlockPool(
            num_blocks,
            retain_hashes_on_free=block_level_mm_prefix,
        )
        self._cpu_pool = CpuBlockPool(num_cpu_blocks)
        self._compaction = KVCompactionCoordinator(
            block_size=block_size,
            gpu_pool=self._gpu_pool,
        )
        self._multimodal_prefix_cache: OrderedDict[
            str,
            _MultimodalPrefixCacheEntry,
        ] = OrderedDict()
        self._multimodal_prefix_cache_max_blocks = num_blocks
        self._multimodal_prefix_cache_blocks = 0
        self._multimodal_prefix_pre_admission_hits = 0
        self._multimodal_prefix_visual_hydration_skips = 0
        self._multimodal_prefix_stale_probe_fallbacks = 0
        self._multimodal_prefix_cache_hits = 0
        self._multimodal_prefix_cache_misses = 0
        self._multimodal_prefix_cache_admissions = 0
        self._multimodal_prefix_cache_evictions = 0
        self._multimodal_prefix_cache_rejections = 0
        self._multimodal_prefix_cache_cow_copies = 0
        self._multimodal_prefix_cache_cow_rows = 0
        self._multimodal_prefix_cache_tail_clone_hits = 0
        self._multimodal_prefix_cache_tail_clone_admissions = 0
        self._multimodal_prefix_cache_tail_clone_evictions = 0
        self._multimodal_prefix_cache_tail_clone_reused_rows = 0
        self._block_level_probe_hits = 0
        self._block_level_reused_blocks = 0
        self._block_level_miss_blocks = 0
        self._block_level_image_index_reused_blocks = 0
        self._block_level_image_index_reused_blocks = 0

    @staticmethod
    def _multimodal_prefix_boundary(seq: Sequence) -> int | None:
        """Return the strict logical prefix ending at the visual payload."""

        return seq.multimodal_prefix_boundary

    @staticmethod
    def _multimodal_prefix_cache_id(
        media_key: str,
        prompt_prefix_token_ids: tuple[int, ...],
    ) -> str:
        token_bytes = np.asarray(prompt_prefix_token_ids, dtype=np.int64).tobytes()
        token_digest = hashlib.sha256(token_bytes).hexdigest()
        return f"{media_key}:{len(prompt_prefix_token_ids)}:{token_digest}"

    def _hashable_sequence(self, seq: Sequence) -> bool:
        """Whether this sequence's blocks may join the prefix-hash index.

        纯文本序列恒可哈希。多模态序列只有在 block 级 mm 模式开启、逐图媒体
        身份可用（无视频）、且 pad run 数与逐图 hash 数一致时才可哈希——
        纯 pad token 序列对任何图片相同，没有媒体身份的哈希会把不同图片
        错误匹配；run/hash 数不一致说明身份映射不可靠，整体回退不哈希。
        """

        if not seq.is_multimodal:
            return True
        if not (
            self.block_level_mm_prefix
            and seq.multimodal_media_token_hashes is not None
        ):
            return False
        if seq.pixel_values_videos is not None or seq.video_grid_thw is not None:
            return False
        spans = seq.image_token_spans()
        return spans is not None and len(spans) == len(seq.multimodal_media_token_hashes)

    @staticmethod
    def _block_image_owner(
        mm_token_hashes: dict[int, int] | None,
        block_size: int,
    ) -> int | None:
        """Full block entirely inside one image's pad run -> its surrogate."""

        if mm_token_hashes is None or len(mm_token_hashes) != block_size:
            return None
        owners = set(mm_token_hashes.values())
        if len(owners) != 1:
            return None
        return next(iter(owners))

    def _block_hash_and_mm(
        self,
        seq: Sequence,
        block_index: int,
        prefix_hash: int,
    ) -> tuple[int, dict[int, int] | None]:
        """Compute the mm-aware hash of one full block chained to its prefix."""

        token_ids = seq.block(block_index)
        mm_token_hashes = None
        if seq.is_multimodal:
            if not self._hashable_sequence(seq):
                return NO_BLOCK_HASH, None
            # 不含 pad 的块返回 None, 按纯文本块哈希 (多轮追问的文本增长部分)
            mm_token_hashes = seq.mm_token_hashes_for_block(block_index, self.block_size)
        return self.compute_hash(token_ids, prefix_hash, mm_token_hashes), mm_token_hashes

    def _cached_prefix_length(self, seq: Sequence) -> int:
        """Walk full prompt blocks through the hash index; return cached tokens.

        多模态序列的结果向下对齐到图片边界：block 边界可能切进某张图的
        pad span 内部，而 prefill 范围不允许切开单张图。
        """

        if not self.enable_prefix_caching or not self._hashable_sequence(seq):
            return 0
        prefix_hash = NO_BLOCK_HASH
        cached = 0
        full_blocks = seq.num_prompt_tokens // self.block_size
        for block_index in range(full_blocks):
            block_hash, mm_token_hashes = self._block_hash_and_mm(
                seq,
                block_index,
                prefix_hash,
            )
            if block_hash == NO_BLOCK_HASH:
                break
            token_ids = seq.block(block_index)
            cached_block = self._gpu_pool.lookup(block_hash, token_ids, mm_token_hashes)
            if cached_block is None:
                owner = self._block_image_owner(mm_token_hashes, self.block_size)
                if owner is not None:
                    cached_block = self._gpu_pool.peek_image_block(
                        owner,
                        token_ids,
                        mm_token_hashes,
                    )
                if cached_block is None:
                    break
            cached += self.block_size
            prefix_hash = block_hash
        return self._snap_to_image_boundary(seq, cached)

    @staticmethod
    def _snap_to_image_boundary(seq: Sequence, candidate: int) -> int:
        """Snap a cached-prefix length down to the nearest image-span start."""

        if not seq.is_multimodal or seq.image_token_id is None or candidate <= 0:
            return candidate
        spans = seq.image_token_spans()
        if spans is None:
            return candidate
        for span_start, span_end in spans:
            if span_start < candidate < span_end:
                return span_start
        return candidate

    def _privatize_block(
        self,
        seq: Sequence,
        block_index: int,
    ) -> tuple[int, int] | None:
        """Give the sequence a private copy of a shared block (row-aligned CoW)."""

        block_id = seq.block_table[block_index]
        block = self.blocks[block_id]
        if block.ref_count <= 1:
            return None
        self._ensure_free_blocks(1)
        new_block = self._allocate_free_block()
        new_block_id = new_block.block_id
        if block.hash != NO_BLOCK_HASH:
            self._gpu_pool.register_hash(
                new_block_id,
                block.hash,
                block.token_ids,
                block.mm_token_hashes,
            )
        self._gpu_pool.release_reference(block_id)
        seq.block_table[block_index] = new_block_id
        return (block_id, new_block_id)

    def _matching_multimodal_prefix(
        self,
        seq: Sequence,
    ) -> tuple[str, _MultimodalPrefixCacheEntry] | None:
        if not self.enable_prefix_caching:
            return None
        media_key = seq.multimodal_prefix_cache_key
        boundary = self._multimodal_prefix_boundary(seq)
        if media_key is None or boundary is None:
            return None
        prompt_prefix = tuple(seq.prompt_token_ids[:boundary])
        cache_id = self._multimodal_prefix_cache_id(media_key, prompt_prefix)
        entry = self._multimodal_prefix_cache.get(cache_id)
        if entry is None:
            return None
        if (
            entry.key != media_key
            or entry.logical_prefix_len != boundary
            or entry.prompt_prefix_token_ids != prompt_prefix
        ):
            raise RuntimeError("multimodal prefix cache digest collision")
        return cache_id, entry

    def probe_multimodal_prefix(
        self,
        seq: Sequence,
        *,
        would_hydrate_visual: bool = False,
    ) -> int:
        """Probe before admission so a prefix hit can bypass Vision hydration.

        block 级 mm 模式下走 hash 链 walk（部分命中返回部分长度）；
        其余保持 entry 级整段语义。hydration skip 只在候选覆盖整个视觉
        边界时计数（部分命中仍需视觉编码）。
        """

        self._assert_sequence_block_size(seq)
        seq.multimodal_prefix_cache_enabled = self.enable_prefix_caching
        if self.block_level_mm_prefix and self._hashable_sequence(seq):
            walk_candidate = self._cached_prefix_length(seq)
            match = self._matching_multimodal_prefix(seq)
            entry_candidate = 0 if match is None else match[1].logical_prefix_len
            candidate_tokens = max(walk_candidate, entry_candidate)
            if candidate_tokens:
                self._block_level_probe_hits += 1
        else:
            match = self._matching_multimodal_prefix(seq)
            candidate_tokens = 0 if match is None else match[1].logical_prefix_len
        seq.prefix_cache_candidate_tokens = candidate_tokens
        if candidate_tokens:
            seq.multimodal_prefix_pre_admission_hit = True
            self._multimodal_prefix_pre_admission_hits += 1
            boundary = self._multimodal_prefix_boundary(seq)
            if would_hydrate_visual and (
                boundary is None or candidate_tokens >= boundary
            ):
                self._multimodal_prefix_visual_hydration_skips += 1
        return candidate_tokens

    def cached_prefix_tokens(self, seq: Sequence) -> int:
        """Publish a read-only cache candidate for cache-aware scheduling."""

        self._assert_sequence_block_size(seq)
        seq.multimodal_prefix_cache_enabled = self.enable_prefix_caching
        if self.block_level_mm_prefix and self._hashable_sequence(seq):
            walk_candidate = self._cached_prefix_length(seq)
            match = self._matching_multimodal_prefix(seq)
            entry_candidate = 0 if match is None else match[1].logical_prefix_len
            candidate_tokens = max(walk_candidate, entry_candidate)
        else:
            match = self._matching_multimodal_prefix(seq)
            candidate_tokens = 0 if match is None else match[1].logical_prefix_len
        seq.prefix_cache_candidate_tokens = candidate_tokens
        return candidate_tokens

    def _required_free_blocks_for_prefix_hit(
        self,
        seq: Sequence,
        entry: _MultimodalPrefixCacheEntry,
    ) -> int:
        suffix_tokens = seq.num_prompt_tokens - entry.logical_prefix_len
        final_physical_tokens = entry.physical_prefix_len + suffix_tokens
        final_blocks = (final_physical_tokens + self.block_size - 1) // self.block_size
        suffix_blocks = final_blocks - len(entry.block_ids)
        has_idle_tail_clone = any(
            self.blocks[block_id].ref_count == 1 for block_id in entry.tail_clone_block_ids
        )
        partial_page_copy = int(
            entry.physical_prefix_len % self.block_size != 0 and not has_idle_tail_clone
        )
        return suffix_blocks + partial_page_copy

    def _reclaimable_cache_blocks(
        self,
        *,
        protected_cache_id: str | None = None,
    ) -> int:
        return sum(
            1
            for cache_id, entry in self._multimodal_prefix_cache.items()
            if cache_id != protected_cache_id
            for block_id in (*entry.block_ids, *entry.tail_clone_block_ids)
            if self.blocks[block_id].ref_count == 1
        )

    def _eviction_candidate(
        self,
        *,
        protected_cache_id: str | None = None,
    ) -> str | None:
        candidates = [
            (entry.benefit_per_block, order, cache_id)
            for order, (cache_id, entry) in enumerate(self._multimodal_prefix_cache.items())
            if cache_id != protected_cache_id
        ]
        return min(candidates)[2] if candidates else None

    def _evict_multimodal_prefix(self, cache_id: str) -> None:
        entry = self._multimodal_prefix_cache.pop(cache_id)
        for block_id in (*entry.block_ids, *entry.tail_clone_block_ids):
            self._gpu_pool.release_reference(block_id)
        self._multimodal_prefix_cache_blocks -= entry.resident_blocks
        self._multimodal_prefix_cache_evictions += 1

    def _drop_idle_tail_clone(
        self,
        *,
        protected_cache_id: str | None = None,
    ) -> bool:
        """Release one derived tail page before evicting reusable prefix KV."""

        for cache_id, entry in self._multimodal_prefix_cache.items():
            if cache_id == protected_cache_id:
                continue
            for index, block_id in enumerate(entry.tail_clone_block_ids):
                if self.blocks[block_id].ref_count != 1:
                    continue
                entry.tail_clone_block_ids.pop(index)
                self._gpu_pool.release_reference(block_id)
                self._multimodal_prefix_cache_blocks -= 1
                self._multimodal_prefix_cache_tail_clone_evictions += 1
                return True
        return False

    def _ensure_free_blocks(
        self,
        required_blocks: int,
        *,
        protected_cache_id: str | None = None,
    ) -> None:
        while self._gpu_pool.free_count < required_blocks:
            if self._drop_idle_tail_clone(
                protected_cache_id=protected_cache_id,
            ):
                continue
            cache_id = self._eviction_candidate(
                protected_cache_id=protected_cache_id,
            )
            if cache_id is None:
                break
            self._evict_multimodal_prefix(cache_id)
        if self._gpu_pool.free_count < required_blocks:
            raise RuntimeError(
                "insufficient GPU KV-cache capacity after cache eviction: "
                f"required={required_blocks}, available={self._gpu_pool.free_count}"
            )

    def _allocate_dense(self, seq: Sequence) -> tuple[tuple[int, int, int], ...]:
        self._ensure_free_blocks(seq.num_blocks)
        block_hash = NO_BLOCK_HASH
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            mm_token_hashes = None
            if (
                self.enable_prefix_caching
                and len(token_ids) == self.block_size
                and self._hashable_sequence(seq)
            ):
                block_hash, mm_token_hashes = self._block_hash_and_mm(
                    seq,
                    i,
                    block_hash,
                )
            else:
                block_hash = NO_BLOCK_HASH
            cached_block = self._gpu_pool.lookup(block_hash, token_ids, mm_token_hashes)
            image_index_hit = False
            if cached_block is None:
                owner = self._block_image_owner(mm_token_hashes, self.block_size)
                if owner is not None:
                    cached_block = self._gpu_pool.claim_image_block(
                        owner,
                        token_ids,
                        mm_token_hashes,
                    )
                    image_index_hit = cached_block is not None
            if cached_block is None:
                cache_miss = True
                if block_hash != NO_BLOCK_HASH:
                    self._block_level_miss_blocks += 1
            else:
                self._block_level_reused_blocks += 1
                if image_index_hit:
                    self._block_level_image_index_reused_blocks += 1
            if cache_miss:
                block = self._allocate_free_block()
            else:
                seq.num_cached_tokens += self.block_size
                block = self._gpu_pool.retain(cached_block.block_id)
            if block_hash != NO_BLOCK_HASH:
                if cache_miss or block.ref_count == 1:
                    # 链式复用同值登记幂等; 索引复用的共享块保留原链映射
                    self._gpu_pool.register_hash(
                        block.block_id,
                        block_hash,
                        token_ids,
                        mm_token_hashes,
                    )
                owner = self._block_image_owner(mm_token_hashes, self.block_size)
                if owner is not None:
                    self._gpu_pool.register_image_owner(block.block_id, owner)
            seq.block_table.append(block.block_id)
        copy_prefix: list[tuple[int, int, int]] = []
        if seq.is_multimodal:
            # block 边界可能切进图片 span 内部: 缓存前缀向下对齐到图片边界,
            # 尾部重算落在共享块上时先私有化 (CoW) 再交给 executor 复制。
            snapped = self._snap_to_image_boundary(seq, seq.num_cached_tokens)
            if snapped != seq.num_cached_tokens:
                pair = self._privatize_block(seq, snapped // self.block_size)
                if pair is not None:
                    copy_prefix.append((*pair, self.block_size))
                seq.num_cached_tokens = snapped
        return tuple(copy_prefix)

    def _allocate_multimodal_prefix_hit(
        self,
        seq: Sequence,
        cache_id: str,
        entry: _MultimodalPrefixCacheEntry,
    ) -> tuple[tuple[int, int, int], ...]:
        required_free_blocks = self._required_free_blocks_for_prefix_hit(
            seq,
            entry,
        )
        self._ensure_free_blocks(
            required_free_blocks,
            protected_cache_id=cache_id,
        )
        tail_rows = entry.physical_prefix_len % self.block_size
        immutable_block_ids = entry.block_ids[:-1] if tail_rows else entry.block_ids
        for block_id in immutable_block_ids:
            self._gpu_pool.retain(block_id)
            seq.block_table.append(block_id)

        copy_prefix: list[tuple[int, int, int]] = []
        if tail_rows:
            idle_tail_clone = next(
                (
                    block_id
                    for block_id in entry.tail_clone_block_ids
                    if self.blocks[block_id].ref_count == 1
                ),
                None,
            )
            if idle_tail_clone is not None:
                self._gpu_pool.retain(idle_tail_clone)
                seq.block_table.append(idle_tail_clone)
                self._multimodal_prefix_cache_tail_clone_hits += 1
                self._multimodal_prefix_cache_tail_clone_reused_rows += tail_rows
            else:
                canonical_tail = entry.block_ids[-1]
                self._gpu_pool.retain(canonical_tail)
                seq.block_table.append(canonical_tail)
                pair = self.copy_on_write(seq)
                if pair is None:
                    raise RuntimeError("shared compact prefix tail did not trigger CoW")
                copy_prefix.append((*pair, tail_rows))
                self._multimodal_prefix_cache_cow_copies += 1
                self._multimodal_prefix_cache_cow_rows += tail_rows
                copied_tail = pair[1]
                if self._multimodal_prefix_cache_blocks < self._multimodal_prefix_cache_max_blocks:
                    self._gpu_pool.retain(copied_tail)
                    entry.tail_clone_block_ids.append(copied_tail)
                    self._multimodal_prefix_cache_blocks += 1
                    self._multimodal_prefix_cache_tail_clone_admissions += 1

        suffix_tokens = seq.num_prompt_tokens - entry.logical_prefix_len
        final_physical_tokens = entry.physical_prefix_len + suffix_tokens
        final_blocks = (final_physical_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < final_blocks:
            seq.block_table.append(self._allocate_free_block().block_id)

        compression_record = dict(entry.compression_record)
        compression_record.update(
            {
                "multimodal_prefix_cache_hit": True,
                "cached_logical_prefix_tokens": entry.logical_prefix_len,
                "cached_physical_prefix_tokens": entry.physical_prefix_len,
                "logical_prompt_tokens": seq.num_prompt_tokens,
                "physical_prompt_kv_tokens": final_physical_tokens,
            }
        )
        seq.num_cached_tokens = entry.logical_prefix_len
        seq.num_computed_tokens = entry.logical_prefix_len
        seq.prefix_cache_candidate_tokens = entry.logical_prefix_len
        seq.multimodal_prefix_cache_hit = True
        seq.visual_pruning_decision_record = compression_record
        seq.install_cached_prefix_layout(
            KVCacheLayoutDescriptor(
                mode=KV_LAYOUT_VISUAL_COMPACT,
                logical_context_len=entry.logical_prefix_len,
                physical_kv_len=entry.physical_prefix_len,
                prompt_logical_len=seq.num_prompt_tokens,
                compressed_prompt_kv_len=entry.physical_prefix_len,
                retained_original_positions=entry.retained_original_positions,
                kv_dtype=entry.kv_dtype,
                compression_record=compression_record,
                compacted_logical_len=entry.logical_prefix_len,
            )
        )
        self._multimodal_prefix_cache.move_to_end(cache_id)
        entry.lifetime_hits += 1
        self._multimodal_prefix_cache_hits += 1
        return tuple(copy_prefix)

    def _allocate_dense_entry_hit(
        self,
        seq: Sequence,
        cache_id: str,
        entry: _MultimodalPrefixCacheEntry,
    ) -> tuple[tuple[int, int, int], ...]:
        """Dense block 级模式下的 entry 整段复用 (不安装 compact layout)。

        复用 entry 的不可变前缀块, 尾部不满块无条件私有化 (entry 块必须
        保持不可变), 后缀分配新块。kv_layout 保持 None, decode 期 hash 链
        与 block 级 walk 语义不受影响。
        """

        required_free_blocks = self._required_free_blocks_for_prefix_hit(
            seq,
            entry,
        )
        self._ensure_free_blocks(
            required_free_blocks,
            protected_cache_id=cache_id,
        )
        tail_rows = entry.physical_prefix_len % self.block_size
        immutable_block_ids = entry.block_ids[:-1] if tail_rows else entry.block_ids
        for block_id in immutable_block_ids:
            self._gpu_pool.retain(block_id)
            seq.block_table.append(block_id)
        self._block_level_reused_blocks += len(immutable_block_ids)

        copy_prefix: list[tuple[int, int, int]] = []
        if tail_rows:
            idle_tail_clone = next(
                (
                    block_id
                    for block_id in entry.tail_clone_block_ids
                    if self.blocks[block_id].ref_count == 1
                ),
                None,
            )
            if idle_tail_clone is not None:
                # 复用空闲 tail clone (内容为上次请求写过的尾块, 前缀行未变)
                self._gpu_pool.retain(idle_tail_clone)
                seq.block_table.append(idle_tail_clone)
                self._multimodal_prefix_cache_tail_clone_hits += 1
            else:
                canonical_tail = entry.block_ids[-1]
                block = self.blocks[canonical_tail]
                new_block = self._allocate_free_block()
                new_block_id = new_block.block_id
                if block.hash != NO_BLOCK_HASH:
                    self._gpu_pool.register_hash(
                        new_block_id,
                        block.hash,
                        block.token_ids,
                        block.mm_token_hashes,
                    )
                seq.block_table.append(new_block_id)
                copy_prefix.append((canonical_tail, new_block_id, self.block_size))
                self._multimodal_prefix_cache_cow_copies += 1
                self._multimodal_prefix_cache_cow_rows += tail_rows
                if (
                    self._multimodal_prefix_cache_blocks
                    < self._multimodal_prefix_cache_max_blocks
                ):
                    self._gpu_pool.retain(new_block_id)
                    entry.tail_clone_block_ids.append(new_block_id)
                    self._multimodal_prefix_cache_blocks += 1
                    self._multimodal_prefix_cache_tail_clone_admissions += 1

        suffix_tokens = seq.num_prompt_tokens - entry.logical_prefix_len
        final_physical_tokens = entry.physical_prefix_len + suffix_tokens
        final_blocks = (final_physical_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < final_blocks:
            seq.block_table.append(self._allocate_free_block().block_id)

        compression_record = dict(entry.compression_record)
        compression_record.update(
            {
                "multimodal_prefix_cache_hit": True,
                "cached_logical_prefix_tokens": entry.logical_prefix_len,
                "cached_physical_prefix_tokens": entry.physical_prefix_len,
                "logical_prompt_tokens": seq.num_prompt_tokens,
                "physical_prompt_kv_tokens": final_physical_tokens,
            }
        )
        seq.num_cached_tokens = entry.logical_prefix_len
        seq.num_computed_tokens = entry.logical_prefix_len
        seq.prefix_cache_candidate_tokens = entry.logical_prefix_len
        seq.multimodal_prefix_cache_hit = True
        seq.visual_pruning_decision_record = compression_record
        self._multimodal_prefix_cache.move_to_end(cache_id)
        entry.lifetime_hits += 1
        self._multimodal_prefix_cache_hits += 1
        return tuple(copy_prefix)

    # Compatibility views for existing diagnostics/tests. Allocator mutations
    # remain centralized in GpuBlockPool and CpuBlockPool.
    @property
    def blocks(self) -> list[Block]:
        return self._gpu_pool.blocks

    @property
    def hash_to_block_id(self) -> dict[int, int]:
        return self._gpu_pool.hash_to_block_id

    @property
    def free_block_ids(self):
        return self._gpu_pool.free_block_ids

    @property
    def free_block_id_set(self) -> set[int]:
        return self._gpu_pool.free_block_id_set

    @property
    def used_block_ids(self) -> set[int]:
        return self._gpu_pool.used_block_ids

    @property
    def num_cpu_blocks(self) -> int:
        return self._cpu_pool.capacity

    @property
    def cpu_free_block_ids(self):
        return self._cpu_pool.free_block_ids

    @classmethod
    def compute_hash(
        cls,
        token_ids: list[int],
        prefix: int = -1,
        mm_token_hashes: dict[int, int] | None = None,
    ) -> int:
        """Hash one token block and its predecessor for prefix sharing.

        ``mm_token_hashes`` 把 block 内 pad 位置映射到逐图 surrogate int64，
        使 pad token 序列相同但媒体不同的 block 产生不同 hash。
        位置编码 (m-rope) 有意不参与 hash——同一媒体在不同布局下位置不同，
        复用后 attention 重新计算位置。
        """

        h = xxhash.xxh64()
        if prefix != NO_BLOCK_HASH:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        if mm_token_hashes:
            for position in sorted(mm_token_hashes):
                h.update(position.to_bytes(4, "little"))
                h.update(mm_token_hashes[position].to_bytes(8, "little"))
        return h.intdigest()

    def _remove_hash_index_for_block(self, block: Block) -> None:
        """释放/替换 block 前清理仍指向该 block 的 prefix-cache 索引。"""

        self._gpu_pool.remove_hash_index(block)

    def _allocate_block(self, block_id: int) -> Block:
        return self._gpu_pool.allocate(block_id)

    def _allocate_free_block(self) -> Block:
        """从空闲队列头分配一个真实空闲 block，跳过过期队列项。"""

        return self._gpu_pool.allocate_free()

    def _assert_sequence_block_size(self, seq: Sequence) -> None:
        """确保 Sequence 页表计算粒度与 BlockManager 物理粒度一致。"""

        if seq.block_size != self.block_size:
            raise ValueError(
                "Sequence.block_size must match BlockManager.block_size, "
                f"got sequence={seq.block_size}, manager={self.block_size}"
            )

    def _deallocate_block(self, block_id: int) -> None:
        self._gpu_pool.deallocate(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        self._assert_sequence_block_size(seq)
        match = self._matching_multimodal_prefix(seq)
        if match is None:
            required_blocks = seq.num_blocks
            protected_cache_id = None
        else:
            protected_cache_id, entry = match
            required_blocks = self._required_free_blocks_for_prefix_hit(
                seq,
                entry,
            )
        return (
            self._gpu_pool.free_count
            + self._reclaimable_cache_blocks(
                protected_cache_id=protected_cache_id,
            )
            >= required_blocks
        )

    def allocate(self, seq: Sequence) -> tuple[tuple[int, int, int], ...]:
        self._assert_sequence_block_size(seq)
        if seq.block_table or seq.cpu_block_table:
            raise RuntimeError(f"sequence {seq.seq_id} already owns a KV block table")
        if not self.can_allocate(seq):
            match = self._matching_multimodal_prefix(seq)
            required_blocks = (
                seq.num_blocks
                if match is None
                else self._required_free_blocks_for_prefix_hit(seq, match[1])
            )
            raise RuntimeError(
                "insufficient GPU KV-cache capacity for atomic allocation: "
                f"required={required_blocks}, available={self._gpu_pool.free_count}"
            )
        if self.block_level_mm_prefix and self._hashable_sequence(seq):
            # block 级模式: entry 命中优先 (同组同布局整段复用, 不装 compact
            # layout, 保持 decode 期 hash 链), block walk 兜底 (子集/多轮/部分)。
            entry_match = self._matching_multimodal_prefix(seq)
            if entry_match is not None:
                cache_id, entry = entry_match
                try:
                    return self._allocate_dense_entry_hit(seq, cache_id, entry)
                except BaseException:
                    if seq.block_table:
                        self.deallocate(seq)
                    raise
            if (
                self.enable_prefix_caching
                and seq.multimodal_prefix_cache_key is not None
                and self._cached_prefix_length(seq) == 0
            ):
                self._multimodal_prefix_cache_misses += 1
            return self._allocate_dense(seq)
        match = self._matching_multimodal_prefix(seq)
        if match is None:
            if seq.multimodal_prefix_pre_admission_hit and not seq.multimodal_prefix_stale_fallback:
                seq.multimodal_prefix_stale_fallback = True
                self._multimodal_prefix_stale_probe_fallbacks += 1
            if self.enable_prefix_caching and seq.multimodal_prefix_cache_key is not None:
                self._multimodal_prefix_cache_misses += 1
            return self._allocate_dense(seq)
        cache_id, entry = match
        try:
            return self._allocate_multimodal_prefix_hit(
                seq,
                cache_id,
                entry,
            )
        except BaseException:
            if seq.block_table:
                self.deallocate(seq)
            raise

    def deallocate(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        if seq.cpu_block_table:
            self._cpu_pool.validate_owned(seq.cpu_block_table)
        for block_id in reversed(seq.block_table):
            self._gpu_pool.release_reference(block_id)
        self._cpu_pool.release_many(seq.cpu_block_table)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.cpu_block_table.clear()
        seq.cpu_block_hashes.clear()
        seq.cpu_block_token_ids.clear()
        seq.cpu_block_mm_hashes.clear()
        seq.kv_layout = None
        seq.prefix_cache_candidate_tokens = 0
        seq.multimodal_prefix_cache_hit = False

    def store_multimodal_prefix(self, seq: Sequence) -> bool:
        """Retain multimodal prefix pages after a cold prefill.

        compact 模式按压缩后 retained positions 存储；dense 模式（block 级）
        按整段视觉前缀的物理满块存储——entry 仅作 O(1) 探测 fast path 与
        benefit-gated 保护，真正的跨请求存活由池层 lazy retention 保证。
        """

        self._assert_sequence_block_size(seq)
        media_key = seq.multimodal_prefix_cache_key
        layout = seq.kv_layout
        boundary = self._multimodal_prefix_boundary(seq)
        if not self.enable_prefix_caching or media_key is None or boundary is None:
            return False
        if seq.multimodal_prefix_cache_hit:
            return False
        if not seq.is_prefill_finished or seq.num_tokens != seq.num_prompt_tokens:
            raise RuntimeError("multimodal prefix admission requires completed prompt prefill")
        if layout is not None:
            retained_positions = tuple(
                position for position in layout.retained_original_positions if position < boundary
            )
            physical_prefix_len = len(retained_positions)
            kv_dtype = layout.kv_dtype
            compression_record = dict(layout.compression_record)
        elif self.block_level_mm_prefix:
            retained_positions = tuple(range(boundary))
            physical_prefix_len = boundary
            kv_dtype = "dense"
            compression_record = {
                "multimodal_prefix_cache_hit": False,
                "compression_mode": "dense_block_level",
            }
        else:
            return False
        if physical_prefix_len <= 0:
            self._multimodal_prefix_cache_rejections += 1
            return False
        prefix_blocks = (physical_prefix_len + self.block_size - 1) // self.block_size
        if prefix_blocks > self._multimodal_prefix_cache_max_blocks or prefix_blocks > len(
            seq.block_table
        ):
            self._multimodal_prefix_cache_rejections += 1
            return False

        prompt_prefix = tuple(seq.prompt_token_ids[:boundary])
        cache_id = self._multimodal_prefix_cache_id(
            media_key,
            prompt_prefix,
        )
        existing = self._multimodal_prefix_cache.get(cache_id)
        if existing is not None:
            if existing.key != media_key or existing.prompt_prefix_token_ids != prompt_prefix:
                raise RuntimeError("multimodal prefix cache digest collision")
            self._multimodal_prefix_cache.move_to_end(cache_id)
            return False

        candidate_utility = boundary / prefix_blocks
        while (
            self._multimodal_prefix_cache_blocks + prefix_blocks
            > self._multimodal_prefix_cache_max_blocks
        ):
            if self._drop_idle_tail_clone():
                continue
            eviction_id = self._eviction_candidate()
            if eviction_id is None:
                self._multimodal_prefix_cache_rejections += 1
                return False
            eviction_entry = self._multimodal_prefix_cache[eviction_id]
            if candidate_utility <= eviction_entry.benefit_per_block:
                self._multimodal_prefix_cache_rejections += 1
                return False
            self._evict_multimodal_prefix(eviction_id)

        block_ids = tuple(seq.block_table[:prefix_blocks])
        for block_id in block_ids:
            self._gpu_pool.retain(block_id)
        self._multimodal_prefix_cache[cache_id] = _MultimodalPrefixCacheEntry(
            key=media_key,
            prompt_prefix_token_ids=prompt_prefix,
            logical_prefix_len=boundary,
            physical_prefix_len=physical_prefix_len,
            retained_original_positions=retained_positions,
            block_ids=block_ids,
            kv_dtype=kv_dtype,
            compression_record=compression_record,
            benefit_tokens=boundary,
        )
        self._multimodal_prefix_cache_blocks += prefix_blocks
        self._multimodal_prefix_cache_admissions += 1
        return True

    def clear_multimodal_prefix_cache(self) -> None:
        """Release every cache-owned page reference."""

        for cache_id in tuple(self._multimodal_prefix_cache):
            self._evict_multimodal_prefix(cache_id)
        self._multimodal_prefix_cache_evictions = 0

    def multimodal_prefix_cache_metadata(self) -> dict[str, object]:
        """Return resident page state and measured-run counters."""

        return {
            "enabled": self.enable_prefix_caching,
            "identity": "sha256_model_processor_media_layout_prompt_prefix_v1",
            "lookup_strategy": "direct_boundary_prompt_sha256",
            "scope": "compacted_scaled_fp8_multimodal_prefix_kv",
            "admission_policy": "lifetime_hits_times_logical_tokens_per_page",
            "total_pool_blocks": self._gpu_pool.capacity,
            "max_blocks": self._multimodal_prefix_cache_max_blocks,
            "resident_blocks": self._multimodal_prefix_cache_blocks,
            "entries": len(self._multimodal_prefix_cache),
            "pre_admission_hits": self._multimodal_prefix_pre_admission_hits,
            "visual_hydration_skips": self._multimodal_prefix_visual_hydration_skips,
            "stale_probe_fallbacks": self._multimodal_prefix_stale_probe_fallbacks,
            "hits": self._multimodal_prefix_cache_hits,
            "misses": self._multimodal_prefix_cache_misses,
            "admissions": self._multimodal_prefix_cache_admissions,
            "evictions": self._multimodal_prefix_cache_evictions,
            "rejections": self._multimodal_prefix_cache_rejections,
            "cow_copies": self._multimodal_prefix_cache_cow_copies,
            "cow_copied_rows": self._multimodal_prefix_cache_cow_rows,
            "cow_dense_equivalent_rows": (
                self._multimodal_prefix_cache_cow_copies * self.block_size
            ),
            "tail_clone_hits": self._multimodal_prefix_cache_tail_clone_hits,
            "tail_clone_admissions": (self._multimodal_prefix_cache_tail_clone_admissions),
            "tail_clone_evictions": (self._multimodal_prefix_cache_tail_clone_evictions),
            "tail_clone_reused_rows": (self._multimodal_prefix_cache_tail_clone_reused_rows),
            "copy_avoided_rows": (
                self._multimodal_prefix_cache_cow_copies * self.block_size
                - self._multimodal_prefix_cache_cow_rows
                + self._multimodal_prefix_cache_tail_clone_reused_rows
            ),
            "resident_tail_clone_blocks": sum(
                len(entry.tail_clone_block_ids) for entry in self._multimodal_prefix_cache.values()
            ),
            "resident_lifetime_hits": sum(
                entry.lifetime_hits for entry in self._multimodal_prefix_cache.values()
            ),
            "block_level_matching": self.block_level_mm_prefix,
            "block_level_probe_hits": self._block_level_probe_hits,
            "block_level_reused_blocks": self._block_level_reused_blocks,
            "block_level_miss_blocks": self._block_level_miss_blocks,
            "block_level_image_index_reused_blocks": (
                self._block_level_image_index_reused_blocks
            ),
            "pool_cached_blocks": len(self._gpu_pool.cached_block_ids),
            "pool_cached_evictions": self._gpu_pool.cached_evictions,
        }

    def multimodal_prefix_entry_metadata(
        self,
        seq: Sequence,
    ) -> dict[str, object] | None:
        """Return exact O(1) residency evidence for one sequence prefix."""

        self._assert_sequence_block_size(seq)
        match = self._matching_multimodal_prefix(seq)
        if match is None:
            return None
        _, entry = match
        return {
            "logical_prefix_tokens": entry.logical_prefix_len,
            "physical_prefix_tokens": entry.physical_prefix_len,
            "canonical_blocks": len(entry.block_ids),
            "tail_clone_blocks": len(entry.tail_clone_block_ids),
            "resident_blocks": entry.resident_blocks,
            "lifetime_hits": entry.lifetime_hits,
        }

    def multimodal_prefix_compression_record(
        self,
        seq: Sequence,
    ) -> dict[str, object] | None:
        """Return the immutable compression decision retained by an exact prefix entry."""

        self._assert_sequence_block_size(seq)
        match = self._matching_multimodal_prefix(seq)
        if match is None:
            return None
        return copy.deepcopy(match[1].compression_record)

    def reset_multimodal_prefix_cache_metrics(self) -> None:
        """Reset counters while retaining warm compacted prefix pages."""

        self._multimodal_prefix_pre_admission_hits = 0
        self._multimodal_prefix_visual_hydration_skips = 0
        self._multimodal_prefix_stale_probe_fallbacks = 0
        self._multimodal_prefix_cache_hits = 0
        self._multimodal_prefix_cache_misses = 0
        self._multimodal_prefix_cache_admissions = 0
        self._multimodal_prefix_cache_evictions = 0
        self._multimodal_prefix_cache_rejections = 0
        self._multimodal_prefix_cache_cow_copies = 0
        self._multimodal_prefix_cache_cow_rows = 0
        self._multimodal_prefix_cache_tail_clone_hits = 0
        self._multimodal_prefix_cache_tail_clone_admissions = 0
        self._multimodal_prefix_cache_tail_clone_evictions = 0
        self._multimodal_prefix_cache_tail_clone_reused_rows = 0
        self._block_level_probe_hits = 0
        self._block_level_reused_blocks = 0
        self._block_level_miss_blocks = 0

    def build_compaction_plan(
        self,
        seq: Sequence,
        *,
        kv_dtype: str,
    ) -> KVCompactionPlan | None:
        """Build a device-copy plan without publishing sequence mutations."""

        self._assert_sequence_block_size(seq)
        return self._compaction.build_plan(seq, kv_dtype=kv_dtype)

    def commit_compaction(
        self,
        seq: Sequence,
        plan: KVCompactionPlan,
    ) -> None:
        """Publish a compaction plan after its device copy succeeds."""

        self._assert_sequence_block_size(seq)
        self._compaction.commit(seq, plan)

    def can_append(self, seq: Sequence) -> bool:
        self._assert_sequence_block_size(seq)
        if seq.physical_kv_len % self.block_size == 1:
            required_blocks = 1
        elif not seq.block_table or self.blocks[seq.block_table[-1]].ref_count <= 1:
            required_blocks = 0
        else:
            match = self._matching_multimodal_prefix(seq)
            cache_owns_tail = bool(
                match is not None
                and seq.block_table[-1] in (*match[1].block_ids, *match[1].tail_clone_block_ids)
            )
            required_blocks = int(
                not (cache_owns_tail and self.blocks[seq.block_table[-1]].ref_count == 2)
            )
        return self._gpu_pool.free_count + self._reclaimable_cache_blocks() >= required_blocks

    def may_append(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 取当前最后一个 block
        physical_remainder = seq.physical_kv_len % self.block_size
        if physical_remainder == 1:
            # The appended token starts a new physical block.
            # 上一个 block 刚填满(hash 已算好), 需要分配新块
            if (
                not seq.has_compact_kv_layout
                and self.enable_prefix_caching
                and self._hashable_sequence(seq)
            ):
                if last_block.hash == NO_BLOCK_HASH:
                    raise RuntimeError("completed dense KV block is missing its hash")
            self._ensure_free_blocks(1)
            block = self._allocate_free_block()
            block_table.append(block.block_id)
        elif physical_remainder == 0:
            # The appended token completes the current block.
            # 计算这个 block 的哈希, 注册到缓存索引
            if (
                not seq.has_compact_kv_layout
                and self.enable_prefix_caching
                and self._hashable_sequence(seq)
            ):
                token_ids = seq.block(seq.num_blocks - 1)
                if last_block.hash == NO_BLOCK_HASH:
                    prefix = (
                        self.blocks[block_table[-2]].hash
                        if len(block_table) > 1
                        else NO_BLOCK_HASH
                    )
                    block_hash, mm_token_hashes = self._block_hash_and_mm(
                        seq,
                        seq.num_blocks - 1,
                        prefix,
                    )
                    if block_hash != NO_BLOCK_HASH:
                        self._gpu_pool.register_hash(
                            last_block.block_id,
                            block_hash,
                            token_ids,
                            mm_token_hashes,
                        )
                elif last_block.token_ids != token_ids:
                    # prompt 整除 block_size 时, 分配期已注册满块 hash;
                    # decode 首 token 再次"完成"该块, 内容一致则跳过注册。
                    raise RuntimeError("mutable KV block unexpectedly has a prefix hash")
                else:
                    pass  # 边界对齐的已注册满块: 内容一致, 无操作
        else:
            # A partially filled private block needs no metadata update.
            if last_block.hash != NO_BLOCK_HASH:
                raise RuntimeError("partial KV block unexpectedly has a prefix hash")

    def can_swap_out(self, seq: Sequence) -> bool:
        """Return whether the CPU pool can hold this sequence's KV blocks."""

        self._assert_sequence_block_size(seq)
        return bool(seq.block_table) and self._cpu_pool.can_allocate(len(seq.block_table))

    def swap_out(self, seq: Sequence) -> list[tuple[int, int]]:
        """Move sequence ownership to CPU blocks and return payload copy pairs."""

        self._assert_sequence_block_size(seq)
        if seq.cpu_block_table:
            raise RuntimeError(f"seq {seq.seq_id} already has CPU block table")
        if not seq.block_table:
            raise RuntimeError(f"seq {seq.seq_id} has no GPU block table to swap out")
        if seq.kv_layout is not None:
            seq.kv_layout.validate(
                block_size=self.block_size,
                block_table=seq.block_table,
            )
        gpu_block_table = self._gpu_pool.validate_owned(seq.block_table)
        cpu_block_table = self._cpu_pool.allocate_many(len(gpu_block_table))
        swap_map = list(zip(gpu_block_table, cpu_block_table, strict=False))
        cpu_block_hashes: list[int] = []
        cpu_block_token_ids: list[list[int]] = []
        cpu_block_mm_hashes: list[dict[int, int] | None] = []
        for gpu_id in gpu_block_table:
            block = self.blocks[gpu_id]
            cpu_block_hashes.append(block.hash)
            cpu_block_token_ids.append(list(block.token_ids))
            cpu_block_mm_hashes.append(
                None if block.mm_token_hashes is None else dict(block.mm_token_hashes)
            )
        for gpu_id in gpu_block_table:
            self._gpu_pool.release_reference(gpu_id)
        seq.cpu_block_table = list(cpu_block_table)
        seq.cpu_block_hashes = cpu_block_hashes
        seq.cpu_block_token_ids = cpu_block_token_ids
        seq.cpu_block_mm_hashes = cpu_block_mm_hashes
        seq.block_table.clear()
        if seq.kv_layout is not None:
            seq.kv_layout.validate(
                block_size=self.block_size,
                block_table=seq.cpu_block_table,
            )
        return swap_map

    def can_swap_in(self, seq: Sequence) -> bool:
        """是否有足够的 GPU block 来换入这个序列"""
        self._assert_sequence_block_size(seq)
        return bool(seq.cpu_block_table) and (
            self._gpu_pool.free_count + self._reclaimable_cache_blocks() >= len(seq.cpu_block_table)
        )

    def _validate_swap_in_metadata(
        self,
        seq: Sequence,
        cpu_block_table: tuple[int, ...],
    ) -> tuple[list[int], list[list[int]], list[dict[int, int] | None]]:
        block_hashes = seq.cpu_block_hashes
        block_token_ids = seq.cpu_block_token_ids
        block_mm_hashes = getattr(seq, "cpu_block_mm_hashes", [])
        if len(block_hashes) != len(cpu_block_table) or len(block_token_ids) != len(
            cpu_block_table
        ):
            raise RuntimeError(
                "swapped sequence is missing CPU block hash metadata; "
                "cannot restore prefix-cache index safely"
            )
        if len(block_mm_hashes) != len(cpu_block_table):
            raise RuntimeError(
                "swapped sequence is missing CPU block mm metadata; "
                "cannot restore prefix-cache index safely"
            )
        for block_hash, token_ids, mm_hashes in zip(
            block_hashes,
            block_token_ids,
            block_mm_hashes,
            strict=False,
        ):
            if block_hash == NO_BLOCK_HASH:
                if token_ids:
                    raise RuntimeError("unhashed swapped block contains stale token metadata")
                continue
            if len(token_ids) != self.block_size:
                raise RuntimeError(
                    "swapped full block metadata is inconsistent: "
                    f"hash={block_hash}, token_count={len(token_ids)}, "
                    f"block_size={self.block_size}"
                )
            if mm_hashes is not None and len(mm_hashes) > self.block_size:
                raise RuntimeError(
                    "swapped block mm metadata exceeds block size: "
                    f"hash={block_hash}, mm_positions={len(mm_hashes)}, "
                    f"block_size={self.block_size}"
                )
        return block_hashes, block_token_ids, block_mm_hashes

    def swap_in(self, seq: Sequence) -> list[tuple[int, int]]:
        """CPU → GPU: 把序列的 KV Cache 从 CPU 内存搬回 GPU 显存
        返回: [(cpu_block_id, gpu_block_id), ...] 需要在 GPU 上执行的搬运对
        """
        self._assert_sequence_block_size(seq)
        if seq.block_table:
            raise RuntimeError(f"seq {seq.seq_id} already has GPU block table")
        if not seq.cpu_block_table:
            raise RuntimeError(f"seq {seq.seq_id} has no CPU block table to swap in")
        if seq.kv_layout is not None:
            seq.kv_layout.validate(
                block_size=self.block_size,
                block_table=seq.cpu_block_table,
            )
        cpu_block_table = self._cpu_pool.validate_owned(seq.cpu_block_table)
        self._ensure_free_blocks(len(cpu_block_table))
        cpu_block_hashes, cpu_block_token_ids, cpu_block_mm_hashes = (
            self._validate_swap_in_metadata(
                seq,
                cpu_block_table,
            )
        )
        new_blocks = self._gpu_pool.allocate_many(len(cpu_block_table))
        new_block_table = [block.block_id for block in new_blocks]
        for block_id, block_hash, token_ids, mm_hashes in zip(
            new_block_table,
            cpu_block_hashes,
            cpu_block_token_ids,
            cpu_block_mm_hashes,
            strict=False,
        ):
            if block_hash != NO_BLOCK_HASH:
                self._gpu_pool.register_hash(block_id, block_hash, token_ids, mm_hashes)
        swap_map = list(zip(cpu_block_table, new_block_table, strict=False))
        self._cpu_pool.release_many(cpu_block_table)
        seq.block_table = new_block_table
        seq.cpu_block_table.clear()
        seq.cpu_block_hashes.clear()
        seq.cpu_block_token_ids.clear()
        seq.cpu_block_mm_hashes.clear()
        if seq.kv_layout is not None:
            seq.kv_layout.validate(
                block_size=self.block_size,
                block_table=seq.block_table,
            )
        return swap_map

    def copy_on_write(self, seq: Sequence) -> tuple[int, int] | None:
        """Make the writable tail block private and return its GPU copy pair."""

        self._assert_sequence_block_size(seq)
        if not seq.block_table:
            return None
        last_block_id = seq.block_table[-1]
        last_block = self.blocks[last_block_id]

        if last_block.ref_count <= 1:
            return None

        match = self._matching_multimodal_prefix(seq)
        protected_cache_id = None if match is None else match[0]
        try:
            self._ensure_free_blocks(
                1,
                protected_cache_id=protected_cache_id,
            )
        except RuntimeError:
            if match is None or last_block_id not in (
                *match[1].block_ids,
                *match[1].tail_clone_block_ids,
            ):
                raise
            self._evict_multimodal_prefix(match[0])
            last_block = self.blocks[last_block_id]
            if last_block.ref_count <= 1:
                return None
            self._ensure_free_blocks(1)
        last_block = self.blocks[last_block_id]
        if last_block.ref_count <= 1:
            return None
        new_block = self._allocate_free_block()
        new_block_id = new_block.block_id
        # 复制旧 block 的元数据 (hash, token_ids, mm 身份) 到新 block
        # CoW 只是逻辑分离, GPU 上的 KV 数据由调用者 (model_runner.copy_kv_blocks) 复制
        if last_block.hash != NO_BLOCK_HASH:
            self._gpu_pool.register_hash(
                new_block_id,
                last_block.hash,
                last_block.token_ids,
                last_block.mm_token_hashes,
            )
        self._gpu_pool.release_reference(last_block_id)

        seq.block_table[-1] = new_block_id  # 更新页表

        return (last_block_id, new_block_id)
