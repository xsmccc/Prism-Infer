"""Execution boundary between immutable scheduler plans and ModelRunner."""

from __future__ import annotations

from typing import Protocol

from prism_infer.analysis.performance_profile import profile_region
from prism_infer.engine.compression import (
    COMPRESSION_VISUAL_COMPACT,
    COMPRESSION_VISUAL_COMPACT_FP8,
)
from prism_infer.engine.contracts import (
    BatchPlan,
    ExecutionResult,
    KVCacheManager,
)


class RunnerBackend(Protocol):
    kv_cache_dtype: object

    def call(self, method_name: str, *args: object) -> object: ...


class ModelExecutor:
    """Apply KV transfers, run the model and commit physical compaction."""

    def __init__(self, config, runner: RunnerBackend, kv_manager: KVCacheManager):
        self.config = config
        self.runner = runner
        self.kv_manager = kv_manager

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        transfers = plan.kv_transfers
        if transfers.copy_on_write:
            with profile_region("engine.kv.copy_on_write"):
                self.runner.call(
                    "copy_kv_blocks", list(transfers.copy_on_write)
                )
        if transfers.swap_out:
            with profile_region("engine.kv.swap_out"):
                self.runner.call(
                    "swap_blocks", list(transfers.swap_out), "out"
                )
        if transfers.swap_in:
            with profile_region("engine.kv.swap_in"):
                self.runner.call(
                    "swap_blocks", list(transfers.swap_in), "in"
                )

        with profile_region("engine.model_runner"):
            runner_result = self.runner.call("run_plan", plan)
        if not isinstance(runner_result, ExecutionResult):
            raise RuntimeError(
                "rank-0 runner must return ExecutionResult, got "
                f"{type(runner_result).__name__}"
            )
        if len(runner_result.token_ids) != plan.batch_size:
            raise RuntimeError(
                "rank-0 runner result must match the planned batch size: "
                f"{len(runner_result.token_ids)} != {plan.batch_size}"
            )

        compaction_count = 0
        if (
            plan.is_prefill
            and self.config.compression_mode
            in (
                COMPRESSION_VISUAL_COMPACT,
                COMPRESSION_VISUAL_COMPACT_FP8,
            )
        ):
            with profile_region("engine.kv.visual_compact"):
                plans = [
                    compaction_plan
                    for seq in plan.sequences
                    if seq.is_prefill_finished
                    if (
                        compaction_plan
                        := self.kv_manager.build_compaction_plan(
                            seq,
                            kv_dtype=str(self.runner.kv_cache_dtype),
                        )
                    )
                    is not None
                ]
                self.runner.call("compact_kv_cache", plans)
                plans_by_seq_id = {
                    compaction_plan.seq_id: compaction_plan
                    for compaction_plan in plans
                }
                for seq in plan.sequences:
                    compaction_plan = plans_by_seq_id.get(seq.seq_id)
                    if compaction_plan is not None:
                        self.kv_manager.commit_compaction(
                            seq, compaction_plan
                        )
                compaction_count = len(plans)

        return ExecutionResult(
            token_ids=runner_result.token_ids,
            compaction_count=compaction_count,
        )
