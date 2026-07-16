"""Stable contracts between scheduling, KV management, execution and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Protocol, Sequence as TypingSequence, runtime_checkable

if TYPE_CHECKING:
    from prism_infer.engine.kv_layout import KVCompactionPlan
    from prism_infer.engine.sequence import Sequence


BlockPair = tuple[int, int]


class BatchPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class KVTransferPlan:
    """Immutable GPU/CPU KV movement requested by one scheduler decision."""

    copy_on_write: tuple[BlockPair, ...] = ()
    swap_in: tuple[BlockPair, ...] = ()
    swap_out: tuple[BlockPair, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.copy_on_write or self.swap_in or self.swap_out)


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """One immutable scheduler-to-executor handoff.

    The referenced ``Sequence`` objects remain mutable request state, but the
    membership, phase, token budgets and KV operations of this decision cannot
    be changed after scheduling.
    """

    phase: BatchPhase
    sequences: tuple["Sequence", ...]
    scheduled_token_counts: tuple[int, ...]
    kv_transfers: KVTransferPlan = field(default_factory=KVTransferPlan)
    policy_name: str = "fcfs"
    created_ns: int = field(default_factory=perf_counter_ns)

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("BatchPlan requires at least one sequence")
        if len(self.scheduled_token_counts) != len(self.sequences):
            raise ValueError(
                "scheduled_token_counts must match sequences: "
                f"{len(self.scheduled_token_counts)} != {len(self.sequences)}"
            )
        if any(count <= 0 for count in self.scheduled_token_counts):
            raise ValueError("scheduled token counts must all be positive")
        if self.phase is BatchPhase.DECODE and any(
            count != 1 for count in self.scheduled_token_counts
        ):
            raise ValueError("decode BatchPlan must schedule one token per request")

    @property
    def is_prefill(self) -> bool:
        return self.phase is BatchPhase.PREFILL

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def sequence_ids(self) -> tuple[int, ...]:
        return tuple(seq.seq_id for seq in self.sequences)

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(self.scheduled_token_counts)

    def as_legacy_tuple(
        self,
    ) -> tuple[
        list["Sequence"],
        bool,
        list[BlockPair],
        list[BlockPair],
        list[BlockPair],
    ]:
        """Compatibility adapter for P1-P6 benchmark/test call sites."""

        return (
            list(self.sequences),
            self.is_prefill,
            list(self.kv_transfers.copy_on_write),
            list(self.kv_transfers.swap_in),
            list(self.kv_transfers.swap_out),
        )

    def __iter__(self):
        # Allows ``seqs, is_prefill, ... = scheduler.schedule()`` while new code
        # consumes the named immutable fields.
        return iter(self.as_legacy_tuple())


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    token_ids: tuple[int | None, ...]
    compaction_count: int = 0


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: int
    token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True, slots=True)
class StepResult:
    plan: BatchPlan
    outputs: tuple[RequestOutput, ...]
    execution: ExecutionResult
    elapsed_ns: int

    @property
    def legacy_num_tokens(self) -> int:
        return (
            self.plan.num_scheduled_tokens
            if self.plan.is_prefill
            else -self.plan.batch_size
        )

    def as_legacy_tuple(self) -> tuple[list[tuple[int, list[int]]], int]:
        return (
            [
                (output.request_id, list(output.token_ids))
                for output in self.outputs
            ],
            self.legacy_num_tokens,
        )


@runtime_checkable
class KVCacheManager(Protocol):
    """Scheduler-visible physical KV ownership contract."""

    def can_allocate(self, seq: "Sequence") -> bool: ...

    def allocate(self, seq: "Sequence") -> None: ...

    def deallocate(self, seq: "Sequence") -> None: ...

    def can_append(self, seq: "Sequence") -> bool: ...

    def may_append(self, seq: "Sequence") -> None: ...

    def copy_on_write(self, seq: "Sequence") -> BlockPair | None: ...

    def can_swap_out(self, seq: "Sequence") -> bool: ...

    def swap_out(self, seq: "Sequence") -> list[BlockPair]: ...

    def can_swap_in(self, seq: "Sequence") -> bool: ...

    def swap_in(self, seq: "Sequence") -> list[BlockPair]: ...

    def build_compaction_plan(
        self,
        seq: "Sequence",
        *,
        kv_dtype: str,
    ) -> "KVCompactionPlan | None": ...

    def commit_compaction(
        self,
        seq: "Sequence",
        plan: "KVCompactionPlan",
    ) -> None: ...

class EngineExecutor(Protocol):
    def execute(self, plan: BatchPlan) -> ExecutionResult: ...


class MetricsSink(Protocol):
    """Observer contract; implementations must not drive scheduling."""

    def on_request_submitted(self, seq: "Sequence", *, timestamp_ns: int) -> None: ...

    def on_batch_planned(self, plan: BatchPlan) -> None: ...

    def on_batch_completed(
        self,
        plan: BatchPlan,
        result: ExecutionResult,
        *,
        started_ns: int,
        finished_ns: int,
    ) -> None: ...

    def on_requests_finished(
        self,
        outputs: TypingSequence[RequestOutput],
        *,
        timestamp_ns: int,
    ) -> None: ...
