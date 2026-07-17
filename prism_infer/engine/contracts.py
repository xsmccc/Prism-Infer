"""Stable contracts between scheduling, KV management, execution and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Protocol, Sequence as TypingSequence, runtime_checkable

import torch

from prism_infer.engine.request import validate_request_id
from prism_infer.utils.context import Context

if TYPE_CHECKING:
    from prism_infer.engine.kv_layout import KVCompactionPlan
    from prism_infer.engine.sequence import Sequence


BlockPair = tuple[int, int]


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_block_pairs(pairs: object, *, name: str) -> None:
    if not isinstance(pairs, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                for block_id in pair
            )
        ):
            raise ValueError(
                f"{name} must contain non-negative integer block pairs"
            )


class BatchPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class KVTransferPlan:
    """Immutable GPU/CPU KV movement requested by one scheduler decision."""

    copy_on_write: tuple[BlockPair, ...] = ()
    swap_in: tuple[BlockPair, ...] = ()
    swap_out: tuple[BlockPair, ...] = ()

    def __post_init__(self) -> None:
        _validate_block_pairs(self.copy_on_write, name="copy_on_write")
        _validate_block_pairs(self.swap_in, name="swap_in")
        _validate_block_pairs(self.swap_out, name="swap_out")

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
        if not isinstance(self.phase, BatchPhase):
            raise TypeError(
                f"BatchPlan.phase must be BatchPhase, got {type(self.phase).__name__}"
            )
        if not isinstance(self.sequences, tuple):
            raise TypeError("BatchPlan.sequences must be an immutable tuple")
        if not self.sequences:
            raise ValueError("BatchPlan requires at least one sequence")
        sequence_ids = tuple(
            validate_request_id(
                getattr(seq, "seq_id", None),
                name="BatchPlan sequence id",
            )
            for seq in self.sequences
        )
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("BatchPlan sequence ids must be unique")
        if not isinstance(self.scheduled_token_counts, tuple):
            raise TypeError(
                "BatchPlan.scheduled_token_counts must be an immutable tuple"
            )
        if len(self.scheduled_token_counts) != len(self.sequences):
            raise ValueError(
                "scheduled_token_counts must match sequences: "
                f"{len(self.scheduled_token_counts)} != {len(self.sequences)}"
            )
        for count in self.scheduled_token_counts:
            _positive_int(count, name="scheduled token count")
        if not isinstance(self.kv_transfers, KVTransferPlan):
            raise TypeError("BatchPlan.kv_transfers must be KVTransferPlan")
        if not isinstance(self.policy_name, str) or not self.policy_name:
            raise ValueError("BatchPlan.policy_name must be a non-empty string")
        if (
            isinstance(self.created_ns, bool)
            or not isinstance(self.created_ns, int)
            or self.created_ns < 0
        ):
            raise ValueError("BatchPlan.created_ns must be a non-negative integer")
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
class DeviceModelInputs:
    """Tensor-only model arguments at the execution boundary."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError("DeviceModelInputs.input_ids must be a tensor")
        if not isinstance(self.position_ids, torch.Tensor):
            raise TypeError("DeviceModelInputs.position_ids must be a tensor")
        if self.input_ids.numel() == 0:
            raise ValueError("DeviceModelInputs.input_ids must not be empty")
        if self.position_ids.numel() == 0:
            raise ValueError("DeviceModelInputs.position_ids must not be empty")
        for name in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"DeviceModelInputs.{name} must be a tensor or None")


@dataclass(frozen=True, slots=True)
class DeviceBatch:
    """Immutable tensor boundary consumed by an execution backend.

    It intentionally contains no mutable ``Sequence`` objects.  Request-state
    mutation remains outside compile/CUDA Graph regions.
    """

    phase: BatchPhase
    sequence_ids: tuple[int, ...]
    scheduled_token_counts: tuple[int, ...]
    model_inputs: DeviceModelInputs
    attention_context: Context
    temperatures: torch.Tensor | None
    execution_bucket: int
    kv_scale_views: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.phase, BatchPhase):
            raise TypeError(
                f"DeviceBatch.phase must be BatchPhase, got {type(self.phase).__name__}"
            )
        if not isinstance(self.sequence_ids, tuple):
            raise TypeError("DeviceBatch.sequence_ids must be an immutable tuple")
        for sequence_id in self.sequence_ids:
            validate_request_id(sequence_id, name="DeviceBatch sequence id")
        batch_size = len(self.sequence_ids)
        if batch_size == 0:
            raise ValueError("DeviceBatch requires at least one request")
        if len(set(self.sequence_ids)) != batch_size:
            raise ValueError("DeviceBatch sequence_ids must be unique")
        if not isinstance(self.scheduled_token_counts, tuple):
            raise TypeError(
                "DeviceBatch.scheduled_token_counts must be an immutable tuple"
            )
        if len(self.scheduled_token_counts) != batch_size:
            raise ValueError(
                "DeviceBatch scheduled_token_counts must match sequence_ids"
            )
        for count in self.scheduled_token_counts:
            _positive_int(count, name="DeviceBatch scheduled token count")
        if self.phase is BatchPhase.DECODE and any(
            count != 1 for count in self.scheduled_token_counts
        ):
            raise ValueError("decode DeviceBatch must schedule one token per request")
        if not isinstance(self.model_inputs, DeviceModelInputs):
            raise TypeError("DeviceBatch.model_inputs must be DeviceModelInputs")
        if not isinstance(self.attention_context, Context):
            raise TypeError("DeviceBatch.attention_context must be Context")
        if self.attention_context.is_prefill != (
            self.phase is BatchPhase.PREFILL
        ):
            raise ValueError("DeviceBatch phase/context mismatch")
        if self.temperatures is not None and not isinstance(
            self.temperatures,
            torch.Tensor,
        ):
            raise TypeError("DeviceBatch.temperatures must be a tensor or None")
        if self.temperatures is not None and self.temperatures.numel() != batch_size:
            raise ValueError(
                "DeviceBatch temperatures must contain one value per request"
            )
        _positive_int(self.execution_bucket, name="DeviceBatch execution_bucket")
        if self.execution_bucket < batch_size:
            raise ValueError(
                "DeviceBatch execution_bucket must cover the request batch"
            )
        if not isinstance(self.kv_scale_views, tuple) or any(
            not isinstance(view, torch.Tensor) for view in self.kv_scale_views
        ):
            raise TypeError("DeviceBatch.kv_scale_views must be a tuple of tensors")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    token_ids: tuple[int | None, ...]
    compaction_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple):
            raise TypeError("ExecutionResult.token_ids must be an immutable tuple")
        if any(
            token_id is not None
            and (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            )
            for token_id in self.token_ids
        ):
            raise ValueError(
                "ExecutionResult token ids must be non-negative integers or None"
            )
        if (
            isinstance(self.compaction_count, bool)
            or not isinstance(self.compaction_count, int)
            or self.compaction_count < 0
        ):
            raise ValueError(
                "ExecutionResult.compaction_count must be a non-negative integer"
            )


@runtime_checkable
class ExecutionBackend(Protocol):
    """Explicit prepare/execute lifecycle for eager/compile/Graph backends."""

    name: str

    def prepare(self, plan: BatchPlan) -> DeviceBatch: ...

    def warmup(self, bucket: int | None = None) -> None: ...

    def capture(self, bucket: int | None = None) -> None: ...

    def execute(self, device_batch: DeviceBatch) -> ExecutionResult: ...

    def release(self) -> None: ...


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
