"""Low-overhead engine/request metrics implementing the P7 contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Sequence as TypingSequence

from prism_infer.engine.contracts import (
    BatchPlan,
    ExecutionResult,
    RequestOutput,
)
from prism_infer.engine.sequence import Sequence


@dataclass(slots=True)
class RequestMetrics:
    request_id: int
    submitted_ns: int
    prompt_tokens: int
    max_tokens: int
    first_scheduled_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    finished_ns: int | None = None
    token_timestamps_ns: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    def to_record(self) -> dict[str, object]:
        queue_ms = (
            None
            if self.first_scheduled_ns is None
            else (self.first_scheduled_ns - self.submitted_ns) / 1e6
        )
        ttft_ms = (
            None
            if self.first_token_ns is None
            else (self.first_token_ns - self.submitted_ns) / 1e6
        )
        latency_ms = (
            None
            if self.finished_ns is None
            else (self.finished_ns - self.submitted_ns) / 1e6
        )
        inter_token_ms = [
            (current - previous) / 1e6
            for previous, current in zip(
                self.token_timestamps_ns, self.token_timestamps_ns[1:]
            )
        ]
        tpot_ms = (
            None
            if not inter_token_ms
            else sum(inter_token_ms) / len(inter_token_ms)
        )
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "output_tokens": len(self.token_timestamps_ns),
            "submitted_ns": self.submitted_ns,
            "first_scheduled_ns": self.first_scheduled_ns,
            "first_token_ns": self.first_token_ns,
            "finished_ns": self.finished_ns,
            "queue_ms": queue_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "inter_token_ms": inter_token_ms,
            "latency_ms": latency_ms,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    phase: str
    batch_size: int
    scheduled_tokens: int
    sequence_ids: tuple[int, ...]
    policy_name: str
    started_ns: int
    finished_ns: int
    compaction_count: int

    def to_record(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "batch_size": self.batch_size,
            "scheduled_tokens": self.scheduled_tokens,
            "sequence_ids": list(self.sequence_ids),
            "policy_name": self.policy_name,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "duration_ms": (self.finished_ns - self.started_ns) / 1e6,
            "compaction_count": self.compaction_count,
        }


class EngineMetrics:
    """In-memory metrics ledger suitable for offline and online adapters."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[int, RequestMetrics] = {}
        self._batches: list[BatchMetrics] = []

    def on_request_submitted(
        self,
        seq: Sequence,
        *,
        timestamp_ns: int,
    ) -> None:
        with self._lock:
            if seq.seq_id in self._requests:
                raise RuntimeError(
                    f"duplicate metrics request id: {seq.seq_id}"
                )
            self._requests[seq.seq_id] = RequestMetrics(
                request_id=seq.seq_id,
                submitted_ns=timestamp_ns,
                prompt_tokens=seq.num_prompt_tokens,
                max_tokens=seq.max_tokens,
            )

    def on_batch_planned(self, plan: BatchPlan) -> None:
        with self._lock:
            for seq in plan.sequences:
                request = self._requests.get(seq.seq_id)
                if request is not None and request.first_scheduled_ns is None:
                    request.first_scheduled_ns = plan.created_ns

    def on_batch_completed(
        self,
        plan: BatchPlan,
        result: ExecutionResult,
        *,
        started_ns: int,
        finished_ns: int,
    ) -> None:
        with self._lock:
            self._batches.append(
                BatchMetrics(
                    phase=plan.phase.value,
                    batch_size=plan.batch_size,
                    scheduled_tokens=plan.num_scheduled_tokens,
                    sequence_ids=plan.sequence_ids,
                    policy_name=plan.policy_name,
                    started_ns=started_ns,
                    finished_ns=finished_ns,
                    compaction_count=result.compaction_count,
                )
            )
            for seq, token_id in zip(plan.sequences, result.token_ids):
                if token_id is None:
                    continue
                request = self._requests.get(seq.seq_id)
                if request is None:
                    continue
                request.token_timestamps_ns.append(finished_ns)
                request.last_token_ns = finished_ns
                if request.first_token_ns is None:
                    request.first_token_ns = finished_ns

    def on_requests_finished(
        self,
        outputs: TypingSequence[RequestOutput],
        *,
        timestamp_ns: int,
    ) -> None:
        with self._lock:
            for output in outputs:
                request = self._requests.get(output.request_id)
                if request is None:
                    continue
                request.finished_ns = timestamp_ns
                request.finish_reason = output.finish_reason

    def mark_terminal(
        self,
        request_id: int,
        *,
        reason: str,
        timestamp_ns: int,
    ) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is not None:
                request.finished_ns = timestamp_ns
                request.finish_reason = reason

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "requests": [
                    self._requests[request_id].to_record()
                    for request_id in sorted(self._requests)
                ],
                "batches": [batch.to_record() for batch in self._batches],
            }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._batches.clear()


class NullMetrics:
    """No-op implementation for callers that explicitly disable metrics."""

    def on_request_submitted(self, seq: Sequence, *, timestamp_ns: int) -> None:
        return None
    def on_batch_planned(self, plan: BatchPlan) -> None:
        return None

    def on_batch_completed(
        self,
        plan: BatchPlan,
        result: ExecutionResult,
        *,
        started_ns: int,
        finished_ns: int,
    ) -> None:
        return None

    def on_requests_finished(
        self,
        outputs: TypingSequence[RequestOutput],
        *,
        timestamp_ns: int,
    ) -> None:
        return None
