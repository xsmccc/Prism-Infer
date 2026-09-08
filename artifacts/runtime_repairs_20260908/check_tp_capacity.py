"""Exercise the real NCCL capacity reduction with deliberately unequal limits.

This checks capacity selection, not model performance. mem_get_info is injected;
the collective runs on both allocated GPUs with the production selection method.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from prism_infer.engine.model_runner import ModelRunner


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank, runner.world_size = rank, dist.get_world_size()
    runner.kv_cache_dtype = torch.bfloat16
    runner.kv_payload_block_bytes, runner.kv_scale_block_bytes = 256, 0
    local_limits = (10, 6)
    result = {}
    try:
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(local_limits[rank] * 256, 16384)
            ),
            patch.object(
                torch.cuda,
                "memory_stats",
                return_value={"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0},
            ),
        ):
            runner.config = SimpleNamespace(gpu_memory_utilization=1.0, num_kvcache_blocks=-1)
            selected = runner._select_num_kv_blocks(256)
            assert selected == 6
            runner.config = SimpleNamespace(gpu_memory_utilization=1.0, num_kvcache_blocks=7)
            try:
                runner._select_num_kv_blocks(256)
            except RuntimeError as error:
                assert "requested=7, max=6" in str(error)
                rejected = True
            else:
                rejected = False
            assert rejected
            result = {
                "rank": rank,
                "injected_local_pages": local_limits[rank],
                "selected_pages": selected,
                "explicit_over_capacity_rejected": rejected,
            }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(os.environ["REPAIR_CAPACITY_OUTPUT"]).write_text(
                json.dumps(
                    {"scope": "real NCCL with injected capacity limits", "ranks": gathered},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
