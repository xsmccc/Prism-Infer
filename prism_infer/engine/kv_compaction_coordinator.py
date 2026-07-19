"""Transactional sequence-page orchestration for visual KV compaction."""

from __future__ import annotations

from prism_infer.engine.block_pool import GpuBlockPool
from prism_infer.engine.kv_layout import (
    KV_LAYOUT_VISUAL_COMPACT,
    KVCacheLayoutDescriptor,
    KVCompactionPlan,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.visual_pruning import build_retained_context_indices


class KVCompactionCoordinator:
    """Build and commit physical compaction without executing device copies.

    The model runner executes the copy represented by ``KVCompactionPlan``.
    Only after that succeeds does this coordinator atomically publish the new
    page table and release suffix blocks.
    """

    def __init__(self, *, block_size: int, gpu_pool: GpuBlockPool):
        self.block_size = block_size
        self.gpu_pool = gpu_pool

    def build_plan(
        self,
        seq: Sequence,
        *,
        kv_dtype: str,
    ) -> KVCompactionPlan | None:
        record = seq.visual_pruning_decision_record
        if record is None:
            if seq.image_token_count or seq.video_token_count:
                raise RuntimeError("visual_compact prefill requires a pruning decision record")
            return None
        if seq.kv_layout is not None:
            raise RuntimeError("sequence KV cache was already compacted")
        if seq.num_tokens != seq.num_prompt_tokens:
            raise RuntimeError("visual KV compaction must run before first token append")
        if seq.num_cached_tokens != 0:
            raise RuntimeError("visual KV compaction does not support prefix-cache prefill")
        if seq.cpu_block_table:
            raise RuntimeError("visual KV compaction requires GPU-resident blocks")
        if len(seq.block_table) != seq.num_blocks:
            raise RuntimeError("visual KV compaction requires a complete prompt block table")
        self.gpu_pool.validate_owned(seq.block_table)
        shared_blocks = [
            block_id
            for block_id in seq.block_table
            if self.gpu_pool.blocks[block_id].ref_count != 1
        ]
        if shared_blocks:
            raise RuntimeError(
                "visual KV compaction does not mutate prefix-shared blocks; "
                f"shared_block_ids={shared_blocks}"
            )

        retained_positions = build_retained_context_indices(
            record,
            seq.num_prompt_tokens,
        )
        if not retained_positions:
            raise RuntimeError("visual KV compaction retained zero prompt tokens")
        physical_prompt_len = len(retained_positions)
        new_num_blocks = (physical_prompt_len + self.block_size - 1) // self.block_size
        old_block_table = tuple(seq.block_table)
        new_block_table = old_block_table[:new_num_blocks]
        released_block_ids = old_block_table[new_num_blocks:]
        source_slots = tuple(
            old_block_table[position // self.block_size] * self.block_size
            + position % self.block_size
            for position in retained_positions
        )
        destination_slots = tuple(
            new_block_table[index // self.block_size] * self.block_size + index % self.block_size
            for index in range(physical_prompt_len)
        )
        plan = KVCompactionPlan(
            seq_id=seq.seq_id,
            logical_prompt_len=seq.num_prompt_tokens,
            physical_prompt_len=physical_prompt_len,
            old_block_table=old_block_table,
            new_block_table=new_block_table,
            released_block_ids=released_block_ids,
            retained_original_positions=retained_positions,
            source_slots=source_slots,
            destination_slots=destination_slots,
            kv_dtype=kv_dtype,
            compression_record=dict(record),
        )
        plan.validate(block_size=self.block_size)
        return plan

    def commit(self, seq: Sequence, plan: KVCompactionPlan) -> None:
        plan.validate(block_size=self.block_size)
        if plan.seq_id != seq.seq_id:
            raise RuntimeError("compaction plan sequence id mismatch")
        if seq.kv_layout is not None:
            raise RuntimeError("sequence KV cache was already compacted")
        if tuple(seq.block_table) != plan.old_block_table:
            raise RuntimeError("sequence block table changed before compaction commit")
        self.gpu_pool.validate_owned(plan.old_block_table)
        if any(self.gpu_pool.blocks[block_id].ref_count != 1 for block_id in plan.old_block_table):
            raise RuntimeError("compaction commit found a shared or released block")
        if seq.visual_pruning_decision_record != plan.compression_record:
            raise RuntimeError("visual pruning decision changed before compaction commit")

        compact_record = dict(plan.compression_record)
        compact_record.update(
            {
                "physical_compaction": True,
                "logical_prompt_tokens": plan.logical_prompt_len,
                "physical_prompt_kv_tokens": plan.physical_prompt_len,
                "old_block_table": list(plan.old_block_table),
                "new_block_table": list(plan.new_block_table),
                "released_block_ids": list(plan.released_block_ids),
            }
        )
        layout = KVCacheLayoutDescriptor(
            mode=KV_LAYOUT_VISUAL_COMPACT,
            logical_context_len=seq.num_tokens,
            physical_kv_len=plan.physical_prompt_len,
            prompt_logical_len=seq.num_prompt_tokens,
            compressed_prompt_kv_len=plan.physical_prompt_len,
            retained_original_positions=plan.retained_original_positions,
            kv_dtype=plan.kv_dtype,
            compression_record=compact_record,
        )
        layout.validate(
            block_size=self.block_size,
            block_table=list(plan.new_block_table),
        )

        # Compacted pages no longer represent original contiguous token blocks.
        for block_id in plan.new_block_table:
            self.gpu_pool.clear_hash(block_id)
        for block_id in plan.released_block_ids:
            if self.gpu_pool.blocks[block_id].ref_count != 1:
                raise RuntimeError("released compact suffix block still has references")
            self.gpu_pool.release_reference(block_id)

        seq.block_table = list(plan.new_block_table)
        seq.visual_pruning_decision_record = compact_record
        seq.install_kv_layout(layout)


__all__ = ["KVCompactionCoordinator"]
