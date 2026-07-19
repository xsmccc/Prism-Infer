"""P4.5 Scheduler swap table 语义回归测试。"""

from types import SimpleNamespace

import pytest

from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence, SequenceStatus


def test_scheduler_swap_in_capacity_uses_cpu_block_table() -> None:
    """swapped 序列应根据 cpu_block_table 判断换入容量。"""

    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=16,
        enable_chunked_prefill=False,
        max_chunk_size=4,
        max_queue_size=None,
        max_consecutive_prefill_batches=1,
        eos=-1,
        num_kvcache_blocks=3,
        kvcache_block_size=4,
        num_cpu_blocks=3,
        enable_prefix_caching=True,
    )
    scheduler = Scheduler(config)
    seq = Sequence([1, 2, 3, 4], block_size=4, request_id=0)
    scheduler.block_manager.allocate(seq)
    # scheduler 的 decode 分支处理的是上一轮 postprocess 已追加的 last token；
    # 此时新 token 的 KV slot 还未由 may_append 预留。
    seq.append_token(5)
    swap_map = scheduler.block_manager.swap_out(seq)
    seq.status = SequenceStatus.SWAPPED
    scheduler.swapped.append(seq)

    scheduled, is_prefill, cow_pairs, swap_in_map, swap_out_map = scheduler.schedule()

    print(f"scheduler initial swap map: {swap_map}")
    print(f"scheduler swap_in_map: {swap_in_map}")
    print(f"scheduler seq block_table after swap_in: {seq.block_table}")
    print(f"scheduler seq cpu_block_table after swap_in: {seq.cpu_block_table}")

    assert scheduled == [seq]
    assert not is_prefill
    assert cow_pairs == []
    assert swap_out_map == []
    assert [cpu_id for cpu_id, _ in swap_in_map] == [cpu_id for _, cpu_id in swap_map]
    assert seq.status == SequenceStatus.RUNNING
    assert seq.cpu_block_table == []
    assert seq.block_table
    print("Scheduler swap table capacity: PASS")


def test_scheduler_empty_decode_raises_runtime_error() -> None:
    """decode 分支没有可运行序列时必须显式报错，不能依赖 assert。"""

    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=16,
        enable_chunked_prefill=False,
        max_chunk_size=4,
        max_queue_size=None,
        max_consecutive_prefill_batches=1,
        eos=-1,
        num_kvcache_blocks=1,
        kvcache_block_size=256,
        num_cpu_blocks=0,
        enable_prefix_caching=True,
    )
    scheduler = Scheduler(config)

    with pytest.raises(RuntimeError, match="no runnable sequences"):
        scheduler.schedule()
    print("Scheduler empty decode explicit error: PASS")
