"""Request scheduling and KV ownership coordination.

P7.2 keeps queue mutation here while moving policy decisions, immutable batch
handoff and GPU execution behind explicit contracts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Iterable

from prism_infer.config import DEFAULT_MAX_VISION_PATCHES_PER_BATCH, Config
from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    KVCacheManager,
    KVTransferPlan,
    RequestOutput,
)
from prism_infer.engine.request import RequestState
from prism_infer.engine.scheduler_policy import (
    AdmissionDecision,
    FCFSSchedulerPolicy,
    SchedulerPolicy,
)
from prism_infer.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class _PrefillCandidate:
    """One sequence's immutable contribution to a prefill batch."""

    sequence: Sequence
    token_count: int
    vision_patches: int


@dataclass(slots=True)
class _PrefillBatchBuilder:
    """Accumulate scheduler-local prefill resources before publishing a plan."""

    max_tokens: int
    max_sequences: int
    max_vision_patches: int
    sequences: list[Sequence] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    num_tokens: int = 0
    num_vision_patches: int = 0

    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.num_tokens

    @property
    def has_sequence_capacity(self) -> bool:
        return len(self.sequences) < self.max_sequences

    @property
    def requires_dedicated_batch(self) -> bool:
        return len(self.sequences) == 1 and self.num_vision_patches > self.max_vision_patches

    def accepts(self, candidate: _PrefillCandidate) -> bool:
        if not self.has_sequence_capacity:
            return False
        if candidate.token_count > self.available_tokens:
            return False
        return (
            not self.sequences
            or self.num_vision_patches + candidate.vision_patches <= self.max_vision_patches
        )

    def add(self, candidate: _PrefillCandidate) -> None:
        if candidate.token_count <= 0:
            raise ValueError("prefill candidate token_count must be positive")
        if not self.accepts(candidate):
            raise ValueError("prefill candidate exceeds batch resource limits")
        self.sequences.append(candidate.sequence)
        self.token_counts.append(candidate.token_count)
        self.num_tokens += candidate.token_count
        self.num_vision_patches += candidate.vision_patches


class Scheduler:
    """Own request queues and turn policy decisions into immutable plans."""

    def __init__(
        self,
        config: Config,
        *,
        policy: SchedulerPolicy | None = None,
        kv_manager: KVCacheManager | None = None,
        clock_ns=perf_counter_ns,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_chunk_size = config.max_chunk_size
        self.max_model_len = config.max_model_len
        vision_patch_budget = getattr(
            config,
            "max_vision_patches_per_batch",
            DEFAULT_MAX_VISION_PATCHES_PER_BATCH,
        )
        if (
            isinstance(vision_patch_budget, bool)
            or not isinstance(vision_patch_budget, int)
            or vision_patch_budget <= 0
        ):
            raise ValueError("max_vision_patches_per_batch must be a positive integer")
        self.max_vision_patches_per_batch = vision_patch_budget
        self.eos = config.eos
        self.clock_ns = clock_ns
        self.block_manager: KVCacheManager = kv_manager or BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.num_cpu_blocks,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        self.policy = policy or FCFSSchedulerPolicy(
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_chunk_size=self.max_chunk_size,
            max_queue_size=config.max_queue_size,
            max_consecutive_prefill_batches=(config.max_consecutive_prefill_batches),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.rejected: deque[Sequence] = deque()
        self.cancelled: deque[Sequence] = deque()
        self.requests: dict[int, Sequence] = {}
        self.consecutive_prefill_batches = 0
        self.admitted_requests = 0
        self.rejected_requests = 0
        self.cancelled_requests = 0
        self.completed_requests = 0
        self.swap_in_operations = 0
        self.swap_preemptions = 0
        self.recompute_preemptions = 0
        self.peak_waiting = 0
        self.peak_running = 0
        self.peak_swapped = 0
        self.peak_active = 0
        self.peak_gpu_kv_blocks = 0
        self.peak_cpu_kv_blocks = 0

    def is_finished(self) -> bool:
        return not self.waiting and not self.running and not self.swapped

    @property
    def num_active_requests(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def add(
        self,
        seq: Sequence,
        *,
        raise_on_reject: bool = True,
    ) -> AdmissionDecision:
        if seq.seq_id in self.requests:
            raise RuntimeError(f"duplicate request id: {seq.seq_id}")
        self.requests[seq.seq_id] = seq
        decision = self.policy.admit(
            seq,
            queued_requests=self.num_active_requests,
        )
        if not decision.accepted:
            seq.transition_to(
                RequestState.REJECTED,
                reason=decision.reason or "admission rejected",
            )
            self.rejected.append(seq)
            self.rejected_requests += 1
            self._observe_state()
            if raise_on_reject:
                raise ValueError(decision.reason or "request rejected")
            return decision
        self.waiting.append(seq)
        self.admitted_requests += 1
        self._observe_state()
        return decision

    def get_request(self, request_id: int) -> Sequence | None:
        return self.requests.get(request_id)

    def _observe_state(self) -> None:
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))
        self.peak_running = max(self.peak_running, len(self.running))
        self.peak_swapped = max(self.peak_swapped, len(self.swapped))
        self.peak_active = max(self.peak_active, self.num_active_requests)
        used_gpu = len(getattr(self.block_manager, "used_block_ids", ()))
        self.peak_gpu_kv_blocks = max(self.peak_gpu_kv_blocks, used_gpu)
        num_cpu_blocks = int(getattr(self.block_manager, "num_cpu_blocks", 0))
        free_cpu = len(getattr(self.block_manager, "cpu_free_block_ids", ()))
        self.peak_cpu_kv_blocks = max(
            self.peak_cpu_kv_blocks,
            num_cpu_blocks - free_cpu,
        )

    def metrics_snapshot(self) -> dict[str, int | str]:
        return {
            "policy": self.policy.name,
            "admitted_requests": self.admitted_requests,
            "rejected_requests": self.rejected_requests,
            "cancelled_requests": self.cancelled_requests,
            "completed_requests": self.completed_requests,
            "swap_in_operations": self.swap_in_operations,
            "swap_preemptions": self.swap_preemptions,
            "recompute_preemptions": self.recompute_preemptions,
            "peak_waiting": self.peak_waiting,
            "peak_running": self.peak_running,
            "peak_swapped": self.peak_swapped,
            "peak_active": self.peak_active,
            "peak_gpu_kv_blocks": self.peak_gpu_kv_blocks,
            "peak_cpu_kv_blocks": self.peak_cpu_kv_blocks,
        }

    def reset_metrics(self) -> None:
        if not self.is_finished():
            raise RuntimeError("scheduler metrics can reset only while idle")
        self.requests.clear()
        self.rejected.clear()
        self.cancelled.clear()
        self.consecutive_prefill_batches = 0
        for name in (
            "admitted_requests",
            "rejected_requests",
            "cancelled_requests",
            "completed_requests",
            "swap_in_operations",
            "swap_preemptions",
            "recompute_preemptions",
            "peak_waiting",
            "peak_running",
            "peak_swapped",
            "peak_active",
            "peak_gpu_kv_blocks",
            "peak_cpu_kv_blocks",
        ):
            setattr(self, name, 0)

    def _prefill_candidate(
        self,
        seq: Sequence,
        *,
        available_tokens: int,
    ) -> _PrefillCandidate | None:
        token_count = self.policy.prefill_token_count(
            seq,
            available_tokens=available_tokens,
        )
        if token_count <= 0:
            return None
        token_start = max(seq.num_cached_tokens, seq.num_computed_tokens)
        return _PrefillCandidate(
            sequence=seq,
            token_count=token_count,
            vision_patches=seq.vision_patch_count_for_prefill_range(
                token_start,
                token_start + token_count,
            ),
        )

    def _collect_running_prefills(self, batch: _PrefillBatchBuilder) -> None:
        if not self.enable_chunked_prefill:
            return
        # Continue prior chunks before admitting new work. Decode requests
        # remain in the shared running queue but are ignored by this pass.
        for seq in tuple(self.running):
            if seq.status is not RequestState.PREFILLING:
                continue
            if not batch.has_sequence_capacity:
                break
            candidate = self._prefill_candidate(
                seq,
                available_tokens=batch.available_tokens,
            )
            if candidate is None:
                break
            if not batch.accepts(candidate):
                continue
            batch.add(candidate)
            if batch.requires_dedicated_batch:
                break

    def _allocate_waiting_prefill(
        self,
        seq: Sequence,
        *,
        available_tokens: int,
        batch: _PrefillBatchBuilder,
    ) -> _PrefillCandidate:
        self.block_manager.allocate(seq)
        # Prefix-cache hits are already materialized. Chunk progress begins
        # after that prefix rather than recomputing it.
        seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
        candidate = self._prefill_candidate(
            seq,
            available_tokens=available_tokens,
        )
        if candidate is None:
            # A fully cached prompt still needs a model step to produce the
            # next-token logits; current prefix caching intentionally never
            # caches an incomplete tail, so this is an invariant violation.
            self.block_manager.deallocate(seq)
            raise RuntimeError("scheduler admitted a fully cached prompt without a computable tail")
        if not batch.accepts(candidate):
            self.block_manager.deallocate(seq)
            raise RuntimeError("KV allocation changed prefill candidate beyond batch limits")
        try:
            seq.transition_to(
                RequestState.PREFILLING,
                reason="admitted to prefill batch",
            )
        except BaseException:
            self.block_manager.deallocate(seq)
            raise
        self.waiting.popleft()
        self.running.append(seq)
        return candidate

    def _admit_waiting_prefills(self, batch: _PrefillBatchBuilder) -> None:
        while self.waiting and batch.has_sequence_capacity and not batch.requires_dedicated_batch:
            seq = self.waiting[0]
            candidate = self._prefill_candidate(
                seq,
                available_tokens=batch.available_tokens,
            )
            if candidate is None or not self.block_manager.can_allocate(seq):
                break
            if not batch.accepts(candidate):
                break
            allocated = self._allocate_waiting_prefill(
                seq,
                available_tokens=batch.available_tokens,
                batch=batch,
            )
            batch.add(allocated)

    def _prefill_plan(self) -> BatchPlan | None:
        batch = _PrefillBatchBuilder(
            max_tokens=self.max_num_batched_tokens,
            max_sequences=self.max_num_seqs,
            max_vision_patches=self.max_vision_patches_per_batch,
        )
        self._collect_running_prefills(batch)
        self._admit_waiting_prefills(batch)
        if not batch.sequences:
            return None
        return BatchPlan(
            phase=BatchPhase.PREFILL,
            sequences=tuple(batch.sequences),
            scheduled_token_counts=tuple(batch.token_counts),
            policy_name=self.policy.name,
            created_ns=self.clock_ns(),
        )

    def _decode_plan(self) -> BatchPlan:
        # The latency-critical offline/interactive path has one resident
        # decoding request and no swapped work.  Keep it in-place instead of
        # rebuilding/removing/restoring temporary deques every token.  Any
        # capacity pressure or mixed state falls through to the general path.
        if not self.swapped and len(self.running) == 1:
            seq = self.running[0]
            if (
                seq.status is RequestState.DECODING
                and self.block_manager.can_append(seq)
            ):
                cow_pair = self.block_manager.copy_on_write(seq)
                self.block_manager.may_append(seq)
                self._observe_state()
                return BatchPlan(
                    phase=BatchPhase.DECODE,
                    sequences=(seq,),
                    scheduled_token_counts=(1,),
                    kv_transfers=KVTransferPlan(
                        copy_on_write=(() if cow_pair is None else (cow_pair,)),
                    ),
                    policy_name=self.policy.name,
                    created_ns=self.clock_ns(),
                )

        cow_pairs: list[tuple[int, int]] = []
        swap_in_map: list[tuple[int, int]] = []
        swap_out_map: list[tuple[int, int]] = []
        scheduled: list[Sequence] = []

        while self.swapped and self.block_manager.can_swap_in(self.swapped[0]):
            seq = self.swapped.popleft()
            pairs = self.block_manager.swap_in(seq)
            swap_in_map.extend(pairs)
            self.swap_in_operations += 1
            seq.transition_to(
                RequestState.DECODING,
                reason="KV swap-in completed",
            )
            self.running.append(seq)

        decode_candidates = deque(
            seq for seq in self.running if seq.status is RequestState.DECODING
        )
        # Remove selected decode candidates from the shared queue.  Prefilling
        # requests stay in place for the next chunk.
        for seq in decode_candidates:
            self.running.remove(seq)

        while decode_candidates and len(scheduled) < self.max_num_seqs:
            seq = decode_candidates.popleft()
            while not self.block_manager.can_append(seq):
                victim = self.policy.preemption_candidate(tuple(decode_candidates))
                if victim is not None:
                    decode_candidates.remove(victim)
                    self.preempt(victim, swap_out_map)
                else:
                    self.preempt(seq, swap_out_map)
                    break
            else:
                cow_pair = self.block_manager.copy_on_write(seq)
                if cow_pair is not None:
                    cow_pairs.append(cow_pair)
                self.block_manager.may_append(seq)
                scheduled.append(seq)

        # Candidates beyond max_num_seqs were not preempted; restore queue order.
        self.running.extend(decode_candidates)
        if not scheduled:
            raise RuntimeError("scheduler decode step produced no runnable sequences")
        self.running.extendleft(reversed(scheduled))
        self._observe_state()
        return BatchPlan(
            phase=BatchPhase.DECODE,
            sequences=tuple(scheduled),
            scheduled_token_counts=(1,) * len(scheduled),
            kv_transfers=KVTransferPlan(
                copy_on_write=tuple(cow_pairs),
                swap_in=tuple(swap_in_map),
                swap_out=tuple(swap_out_map),
            ),
            policy_name=self.policy.name,
            created_ns=self.clock_ns(),
        )

    def schedule(self) -> BatchPlan:
        has_prefill = bool(self.waiting) or any(
            seq.status is RequestState.PREFILLING for seq in self.running
        )
        has_decode = bool(self.swapped) or any(
            seq.status is RequestState.DECODING for seq in self.running
        )
        if self.policy.should_schedule_prefill(
            has_prefill=has_prefill,
            has_decode=has_decode,
            consecutive_prefill_batches=self.consecutive_prefill_batches,
        ):
            prefill_plan = self._prefill_plan()
            if prefill_plan is not None:
                self.consecutive_prefill_batches += 1
                self._observe_state()
                return prefill_plan
        plan = self._decode_plan()
        self.consecutive_prefill_batches = 0
        return plan

    def preempt(
        self,
        seq: Sequence,
        swap_out_map: list[tuple[int, int]] | None = None,
    ) -> None:
        if swap_out_map is not None and self.block_manager.can_swap_out(seq):
            pairs = self.block_manager.swap_out(seq)
            swap_out_map.extend(pairs)
            seq.transition_to(
                RequestState.SWAPPED,
                reason="decode KV capacity preemption",
            )
            self.swapped.append(seq)
            self.swap_preemptions += 1
        else:
            self.block_manager.deallocate(seq)
            seq.num_computed_tokens = 0
            seq.transition_to(
                RequestState.WAITING,
                reason="recompute preemption",
            )
            self.waiting.appendleft(seq)
            self.recompute_preemptions += 1
        self._observe_state()

    def postprocess(
        self,
        plan_or_seqs: BatchPlan | Iterable[Sequence],
        token_ids: Iterable[int | None],
    ) -> tuple[RequestOutput, ...]:
        if isinstance(plan_or_seqs, BatchPlan):
            plan = plan_or_seqs
            seqs = plan.sequences
        else:
            seqs = tuple(plan_or_seqs)
            plan = None

        outputs: list[RequestOutput] = []
        for seq, token_id in zip(seqs, token_ids):
            if token_id is None:
                continue
            seq.append_token(token_id)
            finished = (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens
            if finished:
                seq.transition_to(
                    RequestState.FINISHED,
                    reason=("eos" if not seq.ignore_eos and token_id == self.eos else "length"),
                )
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                outputs.append(
                    RequestOutput(
                        request_id=seq.seq_id,
                        token_ids=tuple(seq.completion_token_ids),
                        finish_reason=(
                            "eos" if not seq.ignore_eos and token_id == self.eos else "length"
                        ),
                    )
                )
                self.completed_requests += 1
            elif seq.status is RequestState.PREFILLING:
                seq.transition_to(
                    RequestState.DECODING,
                    reason="prefill completed and first token sampled",
                )
            elif plan is not None and plan.phase is BatchPhase.PREFILL:
                raise RuntimeError("prefill result received for request outside PREFILLING state")
        self._observe_state()
        return tuple(outputs)

    def cancel(self, request_id: int) -> bool:
        """Cancel one queued/running/swapped request and release its KV state."""

        for queue in (self.waiting, self.running, self.swapped):
            for seq in tuple(queue):
                if seq.seq_id != request_id:
                    continue
                queue.remove(seq)
                if seq.block_table or seq.cpu_block_table:
                    self.block_manager.deallocate(seq)
                seq.transition_to(
                    RequestState.CANCELLED,
                    reason="cancelled by caller",
                )
                self.cancelled.append(seq)
                self.cancelled_requests += 1
                self._observe_state()
                return True
        return False
