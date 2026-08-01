"""Execution boundary between immutable scheduler plans and ModelRunner."""

from __future__ import annotations

from typing import Protocol

from prism_infer.engine.compression import (
    COMPRESSION_VISUAL_COMPACT,
    COMPRESSION_VISUAL_COMPACT_FP8,
    COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
)
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    ExecutionResult,
    KVCacheManager,
)
from prism_infer.observability import profile_region


class RunnerBackend(Protocol):
    kv_cache_dtype: object

    def call(self, method_name: str, *args: object) -> object: ...


class ModelExecutor:
    """Apply KV transfers, run the model and commit physical compaction."""

    def __init__(self, config, runner: RunnerBackend, kv_manager: KVCacheManager):
        self.config = config
        self.runner = runner
        self.kv_manager = kv_manager

    def begin_prefill(self, plan: BatchPlan) -> object:
        """Prepare a prefill for layer-boundary execution."""

        if plan.phase is not BatchPhase.PREFILL:
            raise ValueError("begin_prefill requires a prefill plan")
        transfers = plan.kv_transfers
        if transfers.swap_in or transfers.swap_out:
            raise RuntimeError("cooperative prefill does not support KV swapping")
        if transfers.copy_prefix:
            with profile_region("engine.kv.copy_prefix"):
                self.runner.call(
                    "copy_kv_block_prefixes",
                    list(transfers.copy_prefix),
                )
        if transfers.copy_on_write:
            with profile_region("engine.kv.copy_on_write"):
                self.runner.call(
                    "copy_kv_blocks",
                    list(transfers.copy_on_write),
                )
        begin = getattr(self.runner, "begin_prefill_plan", None)
        if begin is None:
            raise RuntimeError("runner does not support cooperative prefill")
        return begin(plan)

    def advance_prefill(
        self,
        pending: object,
        *,
        max_layers: int,
        max_vision_blocks: int,
    ) -> bool:
        """Run one bounded vision-block or language-layer quantum."""

        advance = getattr(self.runner, "advance_prefill_plan", None)
        if advance is None:
            raise RuntimeError("runner does not support cooperative prefill")
        return bool(
            advance(
                pending,
                max_layers,
                max_vision_blocks=max_vision_blocks,
            )
        )

    def finish_prefill(
        self,
        plan: BatchPlan,
        pending: object,
    ) -> ExecutionResult:
        """Finish a cooperative prefill and apply physical KV compaction."""

        finish = getattr(self.runner, "finish_prefill_plan", None)
        if finish is None:
            raise RuntimeError("runner does not support cooperative prefill")
        runner_result = finish(pending)
        if not isinstance(runner_result, ExecutionResult):
            raise RuntimeError(
                f"rank-0 runner must return ExecutionResult, got {type(runner_result).__name__}"
            )
        return self._commit_visual_compaction(plan, runner_result)

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        transfers = plan.kv_transfers
        if plan.phase is BatchPhase.DECODE and transfers.is_empty:
            fast_execute = getattr(
                self.runner,
                "execute_single_greedy_decode_cudagraph",
                None,
            )
            if fast_execute is not None:
                with profile_region("engine.model_runner"):
                    if self.config.tensor_parallel_size > 1:
                        state_builder = getattr(
                            self.runner,
                            "single_greedy_decode_state",
                            None,
                        )
                        state = None if state_builder is None else state_builder(plan)
                        fast_result = (
                            None
                            if state is None
                            else self.runner.call(
                                "execute_single_greedy_decode_cudagraph_state",
                                state,
                            )
                        )
                    else:
                        fast_result = fast_execute(plan)
                if fast_result is not None:
                    return fast_result
        if transfers.copy_prefix:
            with profile_region("engine.kv.copy_prefix"):
                self.runner.call(
                    "copy_kv_block_prefixes",
                    list(transfers.copy_prefix),
                )
        if transfers.copy_on_write:
            with profile_region("engine.kv.copy_on_write"):
                self.runner.call("copy_kv_blocks", list(transfers.copy_on_write))
        if transfers.swap_out:
            with profile_region("engine.kv.swap_out"):
                self.runner.call("swap_blocks", list(transfers.swap_out), "out")
        if transfers.swap_in:
            with profile_region("engine.kv.swap_in"):
                self.runner.call("swap_blocks", list(transfers.swap_in), "in")

        with profile_region("engine.model_runner"):
            runner_result = self.runner.call("run_plan", plan)
        if not isinstance(runner_result, ExecutionResult):
            raise RuntimeError(
                f"rank-0 runner must return ExecutionResult, got {type(runner_result).__name__}"
            )
        if len(runner_result.token_ids) != plan.batch_size:
            raise RuntimeError(
                "rank-0 runner result must match the planned batch size: "
                f"{len(runner_result.token_ids)} != {plan.batch_size}"
            )

        return self._commit_visual_compaction(plan, runner_result)

    def _commit_visual_compaction(
        self,
        plan: BatchPlan,
        runner_result: ExecutionResult,
    ) -> ExecutionResult:
        """Apply post-prefill physical compaction to a completed runner result."""

        compaction_count = 0
        if plan.is_prefill and self.config.compression_mode in (
            COMPRESSION_VISUAL_COMPACT,
            COMPRESSION_VISUAL_COMPACT_FP8,
            COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
        ):
            with profile_region("engine.kv.visual_compact"):
                plans = [
                    compaction_plan
                    for seq in plan.sequences
                    if seq.kv_layout is None
                    if (
                        seq.is_prefill_finished
                        or (
                            seq.multimodal_prefix_cache_enabled
                            and seq.num_computed_tokens == seq.multimodal_prefix_boundary
                        )
                    )
                    if (
                        compaction_plan := self.kv_manager.build_compaction_plan(
                            seq,
                            kv_dtype=str(self.runner.kv_cache_dtype),
                        )
                    )
                    is not None
                ]
                self.runner.call("compact_kv_cache", plans)
                plans_by_seq_id = {
                    compaction_plan.seq_id: compaction_plan for compaction_plan in plans
                }
                for seq in plan.sequences:
                    compaction_plan = plans_by_seq_id.get(seq.seq_id)
                    if compaction_plan is not None:
                        self.kv_manager.commit_compaction(seq, compaction_plan)
                compaction_count = len(plans)
                for seq in plan.sequences:
                    if (
                        seq.is_prefill_finished
                        and seq.kv_layout is not None
                        and not seq.multimodal_prefix_cache_hit
                    ):
                        self.kv_manager.store_multimodal_prefix(seq)

        return ExecutionResult(
            token_ids=runner_result.token_ids,
            compaction_count=compaction_count,
        )
