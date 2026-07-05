from collections import deque

from prism_infer.config import Config
from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 两个调度上限：并发序列数上限 + 单批 token 数上限
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = getattr(config, 'enable_chunked_prefill', False)
        self.max_chunk_size = getattr(config, 'max_chunk_size', 512)
        self.eos = config.eos
        Sequence.set_block_size(config.kvcache_block_size)
        # 负责 KV Cache 物理块的分配/释放
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, getattr(config, 'num_cpu_blocks', 0))
        # waiting: 等待 prefill 的序列；running: 正在 decode 的序列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()  # KV Cache 在 CPU 上的序列 (Swap)

    def is_finished(self):
        # 两个队列都空，表示所有请求都处理完成
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        # 新请求先进入 waiting 队尾（FIFO）
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
        # ---------- prefill 分支：优先从 waiting 拉新序列 ----------
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        if not self.enable_chunked_prefill:
            # ── 传统模式: 整个 Prefill 一次跑完, 不和 Decode 混跑 ──
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                    break
                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
            if scheduled_seqs:
                return scheduled_seqs, True, [], [], []
        else:
            # ── Chunked Prefill: 分块处理, 每次最多 max_chunk_size 个 token ──
            # 先处理还没 Prefill 完的 running 序列 (上一轮 chunk 的后续)
            for seq in list(self.running):
                if not seq.is_prefill_finished:
                    chunk = min(seq.remaining_prefill_tokens, self.max_chunk_size,
                                self.max_num_batched_tokens - num_batched_tokens)
                    if chunk <= 0:
                        break
                    num_batched_tokens += chunk
                    num_seqs += 1
                    scheduled_seqs.append(seq)

            # 再从 waiting 拉新序列
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                if not self.block_manager.can_allocate(seq):
                    break
                chunk = min(len(seq) - seq.num_cached_tokens, self.max_chunk_size,
                            self.max_num_batched_tokens - num_batched_tokens)
                if chunk <= 0:
                    break
                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += chunk
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)

            if scheduled_seqs:
                return scheduled_seqs, True, [], [], []

        # ---------- decode 分支：从 running 中继续生成 ----------
        cow_pairs = []  # CoW 需要 GPU 端复制的 (src, dst) block 对
        swap_in_map = []   # CPU→GPU 搬运对
        swap_out_map = []  # GPU→CPU 搬运对
        # ── 先尝试换入 swapped 序列 (如果 GPU 有足够空间) ──
        while self.swapped and self.block_manager.can_swap_in(self.swapped[0]):
            seq = self.swapped.popleft()
            pairs = self.block_manager.swap_in(seq)
            swap_in_map.extend(pairs)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            print(f"[SwapIn] seq={seq.seq_id} blocks={len(pairs)} CPU→GPU")
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 空间不足时，驱逐 running 队尾（后进先出）释放空间
                    self.preempt(self.running.pop(), swap_out_map)
                else:
                    # 连当前序列也放不下，只能驱逐它，结束本轮尝试
                    self.preempt(seq, swap_out_map)
                    break
            else:
                # while ... else: 仅在未触发 break 时执行（说明可正常 append）
                num_seqs += 1
                # ── CoW 检查: 写共享 block 前先复制 ──
                cow_pair = self.block_manager.copy_on_write(seq)
                if cow_pair is not None:
                    cow_pairs.append(cow_pair)
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs   # vllm在这里用的swap，换到CPU内存
        # popleft 取出的序列仍需保留在 running，放回队头准备下一轮 decode
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, cow_pairs, swap_in_map, swap_out_map

    def preempt(self, seq: Sequence, swap_out_map: list = None):
        # 驱逐: 优先 swap_out (保留 KV Cache), 其次彻底释放
        if swap_out_map is not None and self.block_manager.can_swap_out(seq):
            # Swap Out: GPU→CPU, 保留 KV Cache 到 CPU 内存
            pairs = self.block_manager.swap_out(seq)
            swap_out_map.extend(pairs)
            seq.status = SequenceStatus.SWAPPED
            self.swapped.append(seq)
            print(f"[SwapOut] seq={seq.seq_id} blocks={len(pairs)} GPU→CPU")
        else:
            # 彻底释放: CPU 也满了, 只能丢弃 KV Cache
            seq.status = SequenceStatus.WAITING
            self.block_manager.deallocate(seq)
            self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        # 将本轮生成结果写回序列，并检查是否达到终止条件
        for seq, token_id in zip(seqs, token_ids):
            if token_id is None:
                # Chunked Prefill 中间 chunk: 不采样, 跳过
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
