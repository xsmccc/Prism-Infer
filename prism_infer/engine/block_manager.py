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

from collections import deque

import numpy as np
import xxhash          # 超快哈希库，用于 Prefix Caching 的 block 内容指纹

from prism_infer.engine.sequence import Sequence


# ─── Block: 一个物理 KV Cache 块 ─────────────────────────────
# 每个 Block 对应 GPU 显存里一段固定大小的 KV Cache 空间
# C++ 类比: struct PageFrame { int id; int ref_count; hash_t hash; vector<int> tokens; };
class Block:

    def __init__(self, block_id: int):
        self.block_id = block_id       # 物理块编号 (0, 1, 2, ...)
        self.ref_count = 0             # 引用计数 (多个序列共享同一 block 时 > 1)
        self.hash = -1                 # 内容哈希 (-1 = 未满/未计算)
        self.token_ids = []            # 这个 block 里存的 token 内容 (用于验证 hash 命中)

    # ── 更新哈希和内容（block 填满时调用）──
    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash            # 设置哈希指纹
        self.token_ids = token_ids  # 设置内容副本

    # ── 重置为初始状态（被分配给新序列时调用）──
    def reset(self) -> None:
        self.ref_count = 1             # 新分配 → 引用计数 = 1
        self.hash = -1                 # 清空哈希
        self.token_ids = []            # 清空 token 记录


# ─── BlockManager: 物理块分配器 ──────────────────────────────
# 管理所有 Block 的分配、释放、Prefix Caching
# C++ 类比: class PageFrameAllocator + prefix hash table
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, num_cpu_blocks: int = 0):
        self.block_size = block_size                            # 每个 block 容纳的 token 数 (如 16)
        # ── GPU 端 block 管理 ──
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有物理块数组 (类似 C++ 的页帧数组)
        self.hash_to_block_id: dict[int, int] = dict()         # 哈希 → block_id 的映射 (Prefix Caching 索引)
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块 ID 队列
        self.free_block_id_set: set[int] = set(range(num_blocks))   # O(1) 判断指定 block 是否空闲
        self.used_block_ids: set[int] = set()                   # 正在被使用的块 ID 集合
        # ── CPU 端 block 管理 (Swap 用) ──
        # C++ 类比: swap 分区的页帧管理
        self.num_cpu_blocks = num_cpu_blocks
        self.cpu_free_block_ids: deque[int] = deque(range(num_cpu_blocks))  # CPU 空闲块 ID

    # ── 计算 block 内容的哈希指纹 (类方法，不需要实例) ──
    # prefix: 前一个 block 的哈希 → 形成链式哈希 (保证相同前缀才能匹配)
    # C++ 类比: static hash_t compute_hash(vector<int>& tokens, hash_t prefix)
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()                                      # 创建 xxHash64 哈希器
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))              # 先把前缀哈希喂进去 (链式哈希)
        h.update(np.array(token_ids).tobytes())                 # 再把 token 内容喂进去
        return h.intdigest()                                    # 返回 64-bit 整数哈希值

    # ── 分配一个物理块 (内部方法) ──
    def _remove_hash_index_for_block(self, block: Block) -> None:
        """释放/替换 block 前清理仍指向该 block 的 prefix-cache 索引。"""

        if block.hash == -1:
            return
        if self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0                             # 确保这个块没被占用
        if block_id not in self.free_block_id_set:
            raise RuntimeError(f"block {block_id} is not free")
        self._remove_hash_index_for_block(block)
        block.reset()                                           # 重置为干净状态, ref_count=1
        self.free_block_id_set.remove(block_id)                  # 从空闲集合中移除
        self.used_block_ids.add(block_id)                       # 加入使用中集合
        return self.blocks[block_id]

    def _allocate_free_block(self) -> Block:
        """从空闲队列头分配一个真实空闲 block，跳过过期队列项。"""

        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_id_set:
                return self._allocate_block(block_id)
        raise RuntimeError("no free KV cache block available")

    def _assert_sequence_block_size(self, seq: Sequence) -> None:
        """确保 Sequence 页表计算粒度与 BlockManager 物理粒度一致。"""

        if seq.block_size != self.block_size:
            raise ValueError(
                "Sequence.block_size must match BlockManager.block_size, "
                f"got sequence={seq.block_size}, manager={self.block_size}"
            )

    # ── 释放一个物理块 (内部方法) ──
    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0                             # 引用计数为 0 才能真正释放
        self._remove_hash_index_for_block(block)
        self.used_block_ids.remove(block_id)                    # 从使用中移除
        self.free_block_id_set.add(block_id)
        self.free_block_ids.append(block_id)                    # 归还到空闲队列

    # ═══════════════════════════════════════════════════════════
    # 以下四个方法 = 对外接口, 被 scheduler.py 调用
    # ═══════════════════════════════════════════════════════════

    # ── can_allocate: Prefill 前检查是否有足够空闲块 ──
    # scheduler._schedule_prefill() 调用
    def can_allocate(self, seq: Sequence) -> bool:
        self._assert_sequence_block_size(seq)
        return len(self.free_block_id_set) >= seq.num_blocks       # 空闲块数 >= 序列需要的块数?

    # ── allocate: 为一条新序列分配所有 KV Cache 块 (Prefill 阶段) ──
    # 带 Prefix Caching: 如果之前有相同前缀的 block，直接复用，跳过计算
    #
    # 流程: 遍历序列的每个 block → 算哈希 → 查缓存 → 命中则复用, 未命中则新分配
    def allocate(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        assert not seq.block_table                              # 确保还没分配过
        h = -1                                                  # 链式哈希的前缀 (第一个 block 无前缀)
        cache_miss = False                                      # 一旦 miss，后续所有 block 都一定 miss
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)                            # 取出第 i 个 block 的 token 内容
            # 只有满块才算哈希 (不满的块随时会变, 缓存无意义)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)        # 用哈希查找是否有缓存的 block
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True                               # 哈希未命中 or 内容不匹配 → miss
            if cache_miss:
                # ---- Cache Miss: 分配新块 ----
                block = self._allocate_free_block()
                block_id = block.block_id
            else:
                # ---- Cache Hit: 复用已有块, 跳过 KV 计算 ----
                seq.num_cached_tokens += self.block_size        # 这些 token 不用重新算 attention
                if block_id in self.used_block_ids:
                    # 块正在被别的序列使用 → 共享, 引用计数 +1
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块虽然有缓存但无人使用 → 重新激活
                    block = self._allocate_block(block_id)
            # 如果是满块, 记录哈希和内容 (供后续查找)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)                    # 加入序列的页表

    # ── deallocate: 释放一条序列的所有 KV Cache 块 ──
    # scheduler.preempt() 或 scheduler.postprocess() (序列结束时) 调用
    # 倒序释放: 最后一个块通常是不满的, 没有缓存价值
    def deallocate(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        for block_id in reversed(seq.block_table):              # 倒序遍历页表
            block = self.blocks[block_id]
            block.ref_count -= 1                                # 引用计数 -1
            if block.ref_count == 0:                            # 没有其他序列引用了
                self._deallocate_block(block_id)                # 真正释放回空闲池
            # 如果 ref_count > 0 → 别的序列还在用, 不释放 (Prefix Caching 共享)
        seq.num_cached_tokens = 0                               # 重置缓存计数
        seq.block_table.clear()                                 # 清空页表
        seq.cpu_block_table.clear()                             # 释放后不应残留 swap 页表
        seq.cpu_block_hashes.clear()
        seq.cpu_block_token_ids.clear()

    # ── can_append: Decode 阶段检查是否能追加 1 个 token ──
    # scheduler._schedule_decode() 的 while not 循环调用
    # len(seq) % block_size == 1 意味着刚好跨入新 block, 需要分配 1 个新块
    # 其他情况不需要新块 (当前块还没满), 返回 True
    def can_append(self, seq: Sequence) -> bool:
        self._assert_sequence_block_size(seq)
        return len(self.free_block_id_set) >= (len(seq) % self.block_size == 1)
        # 注意: (len(seq) % block_size == 1) 是 bool, 转成 int 就是 0 或 1
        # >= 1 → 需要 1 个空闲块
        # >= 0 → 永远 True (不需要新块)

    # ── may_append: Decode 阶段实际追加 token 后更新 block 状态 ──
    # scheduler.postprocess() 在 append_token 之后调用
    # 三种情况, 取决于追加后序列长度对 block_size 的余数:
    def may_append(self, seq: Sequence) -> None:
        self._assert_sequence_block_size(seq)
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]               # 取当前最后一个 block
        if len(seq) % self.block_size == 1:
            # ---- 情况1: 余数=1 → 刚好溢出到新 block ----
            # 上一个 block 刚填满(hash 已算好), 需要分配新块
            assert last_block.hash != -1                        # 上一个块应该是满的(有hash)
            block = self._allocate_free_block()
            block_id = block.block_id
            block_table.append(block_id)                        # 新块加入页表
        elif len(seq) % self.block_size == 0:
            # ---- 情况2: 余数=0 → 当前 block 刚好填满 ----
            # 计算这个 block 的哈希, 注册到缓存索引
            assert last_block.hash == -1                        # 填满前应该是没 hash 的
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)                     # 记录哈希
            self.hash_to_block_id[h] = last_block.block_id     # 注册到缓存索引
        else:
            # ---- 情况3: 余数>1且!=0 → block 还没满, 什么都不用做 ----
            assert last_block.hash == -1                        # 没满的块不应该有 hash

    # ════════════════════════════════════════════════════════════
    # Swap 相关方法: GPU ↔ CPU KV Cache 块搬运
    # C++ 类比: OS 的 swap 分区管理
    #   swap_out = 页面换出 (GPU→CPU, 释放 GPU 物理页)
    #   swap_in  = 页面换入 (CPU→GPU, 占用 GPU 物理页)
    # ════════════════════════════════════════════════════════════

    def can_swap_out(self, seq: Sequence) -> bool:
        """是否有足够的 CPU block 来换出这个序列"""
        self._assert_sequence_block_size(seq)
        return len(self.cpu_free_block_ids) >= len(seq.block_table)

    def swap_out(self, seq: Sequence) -> list[tuple[int, int]]:
        """GPU → CPU: 把序列的 KV Cache 从 GPU 显存搬到 CPU 内存
        返回: [(gpu_block_id, cpu_block_id), ...] 需要在 GPU 上执行的搬运对
        """
        self._assert_sequence_block_size(seq)
        if seq.cpu_block_table:
            raise RuntimeError(f"seq {seq.seq_id} already has CPU block table")
        swap_map = []
        cpu_block_table = []
        cpu_block_hashes = []
        cpu_block_token_ids = []
        for gpu_id in seq.block_table:
            cpu_id = self.cpu_free_block_ids.popleft()
            swap_map.append((gpu_id, cpu_id))
            block = self.blocks[gpu_id]
            cpu_block_hashes.append(block.hash)
            cpu_block_token_ids.append(list(block.token_ids))
            # 释放 GPU block
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(gpu_id)
            cpu_block_table.append(cpu_id)
        seq.cpu_block_table = cpu_block_table
        seq.cpu_block_hashes = cpu_block_hashes
        seq.cpu_block_token_ids = cpu_block_token_ids
        seq.block_table = []
        return swap_map

    def can_swap_in(self, seq: Sequence) -> bool:
        """是否有足够的 GPU block 来换入这个序列"""
        self._assert_sequence_block_size(seq)
        return bool(seq.cpu_block_table) and len(self.free_block_id_set) >= len(seq.cpu_block_table)

    def swap_in(self, seq: Sequence) -> list[tuple[int, int]]:
        """CPU → GPU: 把序列的 KV Cache 从 CPU 内存搬回 GPU 显存
        返回: [(cpu_block_id, gpu_block_id), ...] 需要在 GPU 上执行的搬运对
        """
        self._assert_sequence_block_size(seq)
        if seq.block_table:
            raise RuntimeError(f"seq {seq.seq_id} already has GPU block table")
        if not seq.cpu_block_table:
            raise RuntimeError(f"seq {seq.seq_id} has no CPU block table to swap in")
        cpu_block_hashes = getattr(seq, "cpu_block_hashes", [])
        cpu_block_token_ids = getattr(seq, "cpu_block_token_ids", [])
        if (
            len(cpu_block_hashes) != len(seq.cpu_block_table)
            or len(cpu_block_token_ids) != len(seq.cpu_block_table)
        ):
            raise RuntimeError(
                "swapped sequence is missing CPU block hash metadata; "
                "cannot restore prefix-cache index safely"
            )
        swap_map = []
        new_block_table = []
        for cpu_id in seq.cpu_block_table:
            block = self._allocate_free_block()
            gpu_id = block.block_id
            swap_map.append((cpu_id, gpu_id))
            self.cpu_free_block_ids.append(cpu_id)
            new_block_table.append(gpu_id)
        seq.block_table = new_block_table
        seq.cpu_block_table = []

        # 恢复满块的 hash 信息 (Prefix Cache 需要)。
        # 注意: decode 序列跨进程序列化后可能只保留 last_token，不能依赖
        # seq.block(i) 重新读取完整 token_ids。
        for i, block_id in enumerate(new_block_table):
            block_hash = cpu_block_hashes[i]
            token_ids = cpu_block_token_ids[i]
            if block_hash != -1:
                if len(token_ids) != self.block_size:
                    raise RuntimeError(
                        "swapped full block metadata is inconsistent: "
                        f"hash={block_hash}, token_count={len(token_ids)}, "
                        f"block_size={self.block_size}"
                    )
                block = self.blocks[block_id]
                block.update(block_hash, list(token_ids))
                self.hash_to_block_id[block_hash] = block_id
        seq.cpu_block_hashes = []
        seq.cpu_block_token_ids = []
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
        new_block.hash = last_block.hash
        new_block.token_ids = list(last_block.token_ids)  # 深拷贝
        if new_block.hash != -1:
            self.hash_to_block_id[new_block.hash] = new_block_id  # 更新 hash 索引
        
        last_block.ref_count -= 1  # 旧 block 不再被这个 seq 引用
        # 注意: 不调用 _deallocate_block, 因为 ref_count > 0 (还有其他序列在用)
        
        seq.block_table[-1] = new_block_id  # 更新页表
        
        return (last_block_id, new_block_id)
