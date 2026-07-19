# ═══════════════════════════════════════════════════════════════
# block_manager.py —— KV Cache 物理块管理器 (PagedAttention 核心)
#
# 核心思想: 把 GPU 显存里的 KV Cache 切成固定大小的 block (如16 tokens/block),
#           用类似 OS 虚拟内存分页的方式管理:
#           - 每个 block 有唯一 block_id
#           - 序列通过 block_table (页表) 间接引用物理 block
#           - 支持 Prefix Caching: 相同前缀的 block 可以复用 (ref_count > 1)
#
# C++ 类比: Block ≈ 内存页, BlockManager ≈ 页帧分配器,
#           block_table ≈ 页表, hash_to_block_id ≈ TLB/缓存索引
# ═══════════════════════════════════════════════════════════════

import numpy as np
import xxhash

from prism_infer.engine.block_pool import (
    Block,
    CpuBlockPool,
    GpuBlockPool,
    NO_BLOCK_HASH,
)
from prism_infer.engine.kv_compaction_coordinator import KVCompactionCoordinator
from prism_infer.engine.kv_layout import KVCompactionPlan
from prism_infer.engine.sequence import Sequence


# ─── BlockManager: 物理块分配器 ──────────────────────────────
# 管理所有 Block 的分配、释放、Prefix Caching
# C++ 类比: class PageFrameAllocator + prefix hash table
class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_cpu_blocks: int = 0,
        *,
        enable_prefix_caching: bool = True,
    ):
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
        if not isinstance(enable_prefix_caching, bool):
            raise TypeError("enable_prefix_caching must be bool")
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._gpu_pool = GpuBlockPool(num_blocks)
        self._cpu_pool = CpuBlockPool(num_cpu_blocks)
        self._compaction = KVCompactionCoordinator(
            block_size=block_size,
            gpu_pool=self._gpu_pool,
        )

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

    # ── 计算 block 内容的哈希指纹 (类方法，不需要实例) ──
    # prefix: 前一个 block 的哈希 → 形成链式哈希 (保证相同前缀才能匹配)
    # C++ 类比: static hash_t compute_hash(vector<int>& tokens, hash_t prefix)
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != NO_BLOCK_HASH:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    # ── 分配一个物理块 (内部方法) ──
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

    # ── 释放一个物理块 (内部方法) ──
    def _deallocate_block(self, block_id: int) -> None:
        self._gpu_pool.deallocate(block_id)

    # ═══════════════════════════════════════════════════════════
    # 以下四个方法 = 对外接口, 被 scheduler.py 调用
    # ═══════════════════════════════════════════════════════════

    # ── can_allocate: Prefill 前检查是否有足够空闲块 ──
    # scheduler._schedule_prefill() 调用
    def can_allocate(self, seq: Sequence) -> bool:
        self._assert_sequence_block_size(seq)
        return self._gpu_pool.free_count >= seq.num_blocks

    # ── allocate: 为一条新序列分配所有 KV Cache 块 (Prefill 阶段) ──
    # 带 Prefix Caching: 如果之前有相同前缀的 block，直接复用，跳过计算
    #
    # 流程: 遍历序列的每个 block → 算哈希 → 查缓存 → 命中则复用, 未命中则新分配
    def allocate(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        if seq.block_table or seq.cpu_block_table:
            raise RuntimeError(f"sequence {seq.seq_id} already owns a KV block table")
        if not self.can_allocate(seq):
            raise RuntimeError(
                "insufficient GPU KV-cache capacity for atomic allocation: "
                f"required={seq.num_blocks}, available={self._gpu_pool.free_count}"
            )
        block_hash = NO_BLOCK_HASH
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            block_hash = (
                self.compute_hash(token_ids, block_hash)
                if self.enable_prefix_caching
                and not seq.is_multimodal
                and len(token_ids) == self.block_size
                else NO_BLOCK_HASH
            )
            cached_block = self._gpu_pool.lookup(block_hash, token_ids)
            if cached_block is None:
                cache_miss = True
            if cache_miss:
                block = self._allocate_free_block()
            else:
                seq.num_cached_tokens += self.block_size
                block = self._gpu_pool.retain(cached_block.block_id)
            if block_hash != NO_BLOCK_HASH:
                self._gpu_pool.register_hash(
                    block.block_id,
                    block_hash,
                    token_ids,
                )
            seq.block_table.append(block.block_id)

    # ── deallocate: 释放一条序列的所有 KV Cache 块 ──
    # scheduler.preempt() 或 scheduler.postprocess() (序列结束时) 调用
    # 倒序释放: 最后一个块通常是不满的, 没有缓存价值
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
        seq.kv_layout = None

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
        return self._gpu_pool.free_count >= (seq.physical_kv_len % self.block_size == 1)
        # 注意: (len(seq) % block_size == 1) 是 bool, 转成 int 就是 0 或 1
        # >= 1 → 需要 1 个空闲块
        # >= 0 → 永远 True (不需要新块)

    # ── may_append: Decode 阶段实际追加 token 后更新 block 状态 ──
    # scheduler.postprocess() 在 append_token 之后调用
    # 三种情况, 取决于追加后序列长度对 block_size 的余数:
    def may_append(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 取当前最后一个 block
        physical_remainder = seq.physical_kv_len % self.block_size
        if physical_remainder == 1:
            # ---- 情况1: 余数=1 → 刚好溢出到新 block ----
            # 上一个 block 刚填满(hash 已算好), 需要分配新块
            if (
                not seq.has_compact_kv_layout
                and self.enable_prefix_caching
                and not seq.is_multimodal
            ):
                if last_block.hash == NO_BLOCK_HASH:
                    raise RuntimeError("completed dense KV block is missing its hash")
            block = self._allocate_free_block()
            block_table.append(block.block_id)
        elif physical_remainder == 0:
            # ---- 情况2: 余数=0 → 当前 block 刚好填满 ----
            # 计算这个 block 的哈希, 注册到缓存索引
            if last_block.hash != NO_BLOCK_HASH:
                raise RuntimeError("mutable KV block unexpectedly has a prefix hash")
            if (
                not seq.has_compact_kv_layout
                and self.enable_prefix_caching
                and not seq.is_multimodal
            ):
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = (
                    self.blocks[block_table[-2]].hash if len(block_table) > 1 else NO_BLOCK_HASH
                )
                block_hash = self.compute_hash(token_ids, prefix)
                self._gpu_pool.register_hash(
                    last_block.block_id,
                    block_hash,
                    token_ids,
                )
        else:
            # ---- 情况3: 余数>1且!=0 → block 还没满, 什么都不用做 ----
            if last_block.hash != NO_BLOCK_HASH:
                raise RuntimeError("partial KV block unexpectedly has a prefix hash")

    # ════════════════════════════════════════════════════════════
    # Swap 相关方法: GPU ↔ CPU KV Cache 块搬运
    # C++ 类比: OS 的 swap 分区管理
    #   swap_out = 页面换出 (GPU→CPU, 释放 GPU 物理页)
    #   swap_in  = 页面换入 (CPU→GPU, 占用 GPU 物理页)
    # ════════════════════════════════════════════════════════════

    def can_swap_out(self, seq: Sequence) -> bool:
        """是否有足够的 CPU block 来换出这个序列"""
        self._assert_sequence_block_size(seq)
        return bool(seq.block_table) and self._cpu_pool.can_allocate(len(seq.block_table))

    def swap_out(self, seq: Sequence) -> list[tuple[int, int]]:
        """GPU → CPU: 把序列的 KV Cache 从 GPU 显存搬到 CPU 内存
        返回: [(gpu_block_id, cpu_block_id), ...] 需要在 GPU 上执行的搬运对
        """
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
        swap_map = list(zip(gpu_block_table, cpu_block_table))
        cpu_block_hashes: list[int] = []
        cpu_block_token_ids: list[list[int]] = []
        for gpu_id in gpu_block_table:
            block = self.blocks[gpu_id]
            cpu_block_hashes.append(block.hash)
            cpu_block_token_ids.append(list(block.token_ids))
        for gpu_id in gpu_block_table:
            self._gpu_pool.release_reference(gpu_id)
        seq.cpu_block_table = list(cpu_block_table)
        seq.cpu_block_hashes = cpu_block_hashes
        seq.cpu_block_token_ids = cpu_block_token_ids
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
        return bool(seq.cpu_block_table) and self._gpu_pool.free_count >= len(seq.cpu_block_table)

    def _validate_swap_in_metadata(
        self,
        seq: Sequence,
        cpu_block_table: tuple[int, ...],
    ) -> tuple[list[int], list[list[int]]]:
        block_hashes = seq.cpu_block_hashes
        block_token_ids = seq.cpu_block_token_ids
        if len(block_hashes) != len(cpu_block_table) or len(block_token_ids) != len(
            cpu_block_table
        ):
            raise RuntimeError(
                "swapped sequence is missing CPU block hash metadata; "
                "cannot restore prefix-cache index safely"
            )
        for block_hash, token_ids in zip(block_hashes, block_token_ids):
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
        return block_hashes, block_token_ids

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
        if self._gpu_pool.free_count < len(cpu_block_table):
            raise RuntimeError(
                "insufficient free GPU KV-cache blocks for atomic swap-in: "
                f"required={len(cpu_block_table)}, "
                f"available={self._gpu_pool.free_count}"
            )
        cpu_block_hashes, cpu_block_token_ids = self._validate_swap_in_metadata(
            seq,
            cpu_block_table,
        )
        new_blocks = self._gpu_pool.allocate_many(len(cpu_block_table))
        new_block_table = [block.block_id for block in new_blocks]
        for block_id, block_hash, token_ids in zip(
            new_block_table,
            cpu_block_hashes,
            cpu_block_token_ids,
        ):
            if block_hash != NO_BLOCK_HASH:
                self._gpu_pool.register_hash(block_id, block_hash, token_ids)
        swap_map = list(zip(cpu_block_table, new_block_table))
        self._cpu_pool.release_many(cpu_block_table)
        seq.block_table = new_block_table
        seq.cpu_block_table.clear()
        seq.cpu_block_hashes.clear()
        seq.cpu_block_token_ids.clear()
        if seq.kv_layout is not None:
            seq.kv_layout.validate(
                block_size=self.block_size,
                block_table=seq.block_table,
            )
        return swap_map

        # ── copy_on_write: 写时复制 (CoW) ──

    # 当某个 block 被多个序列共享 (ref_count > 1) 时,
    # 写入前必须先复制一份独立的 block, 避免污染其他序列的 KV Cache
    #
    # C++ 类比: Linux fork() 后的 Copy-on-Write 页面
    #   - 多进程共享同一物理页 (ref_count > 1)
    #   - 进程写入 → page fault → 复制新页 → 各自独立
    # 这里:
    #   - 多序列共享同一 KV Cache block (ref_count > 1)
    #   - 序列要写 KV → CoW → 复制新 block → 更新 block_table
    #
    # 返回: (old_block_id, new_block_id) 如果发生了复制, 否则 None
    #        调用者需要用这个信息在 GPU 上复制 KV 数据
    def copy_on_write(self, seq: Sequence) -> tuple[int, int] | None:
        self._assert_sequence_block_size(seq)
        if not seq.block_table:
            return None
        last_block_id = seq.block_table[-1]
        last_block = self.blocks[last_block_id]

        if last_block.ref_count <= 1:
            return None  # 独占, 不需要复制

        # 需要 CoW: 分配新 block, 旧 block 引用计数 -1
        new_block = self._allocate_free_block()
        new_block_id = new_block.block_id
        # 复制旧 block 的元数据 (hash, token_ids) 到新 block
        # CoW 只是逻辑分离, GPU 上的 KV 数据由调用者 (model_runner.copy_kv_blocks) 复制
        if last_block.hash != NO_BLOCK_HASH:
            self._gpu_pool.register_hash(
                new_block_id,
                last_block.hash,
                last_block.token_ids,
            )
        self._gpu_pool.release_reference(last_block_id)

        seq.block_table[-1] = new_block_id  # 更新页表

        return (last_block_id, new_block_id)
