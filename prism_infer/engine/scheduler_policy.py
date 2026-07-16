"""Pure scheduling policy decisions, separate from mutable queue/KV state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence as TypingSequence

from prism_infer.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str | None = None


class SchedulerPolicy(Protocol):
    name: str

    def admit(
        self,
        seq: Sequence,
        *,
        queued_requests: int,
    ) -> AdmissionDecision: ...

    def prefill_token_count(
        self,
        seq: Sequence,
        *,
        available_tokens: int,
    ) -> int: ...

    def preemption_candidate(
        self,
        candidates: TypingSequence[Sequence],
    ) -> Sequence | None: ...

    def should_schedule_prefill(
        self,
        *,
        has_prefill: bool,
        has_decode: bool,
        consecutive_prefill_batches: int,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class FCFSSchedulerPolicy:
    """FIFO admission/prefill with newest-running recompute preemption."""

    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    enable_chunked_prefill: bool
    max_chunk_size: int
    max_queue_size: int | None = None
    max_consecutive_prefill_batches: int = 1
    name: str = "fcfs"

    def __post_init__(self) -> None:
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("scheduler batch limits must be positive")
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if self.max_queue_size is not None and self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive when set")
        if self.max_consecutive_prefill_batches <= 0:
            raise ValueError(
                "max_consecutive_prefill_batches must be positive"
            )

    def admit(
        self,
        seq: Sequence,
        *,
        queued_requests: int,
    ) -> AdmissionDecision:
        requested_length = seq.num_prompt_tokens + seq.max_tokens
        if requested_length > self.max_model_len:
            return AdmissionDecision(
                False,
                "request length exceeds max_model_len: "
                f"prompt={seq.num_prompt_tokens} max_tokens={seq.max_tokens} "
                f"limit={self.max_model_len}",
            )
        if (
            self.max_queue_size is not None
            and queued_requests >= self.max_queue_size
        ):
            return AdmissionDecision(
                False,
                f"request queue is full: limit={self.max_queue_size}",
            )
        for start, end in self._visual_spans(seq):
            span_tokens = end - start
            if self.enable_chunked_prefill and span_tokens > self.max_chunk_size:
                return AdmissionDecision(
                    False,
                    "visual token span exceeds atomic prefill chunk: "
                    f"span_tokens={span_tokens} "
                    f"max_chunk_size={self.max_chunk_size}",
                )
        return AdmissionDecision(True)

    @staticmethod
    def _visual_spans(seq: Sequence) -> tuple[tuple[int, int], ...]:
        visual_ids = {
            token_id
            for token_id in (seq.image_token_id, seq.video_token_id)
            if token_id is not None
        }
        visual_positions = [
            index
            for index, token_id in enumerate(seq.prompt_token_ids)
            if token_id in visual_ids
        ]
        if not visual_positions:
            return ()
        # A single processor payload can map to multiple placeholder runs
        # (video temporal groups, multiple images).  VisionEncoder emits one
        # concatenated feature tensor, so all runs and separators between them
        # are one atomic region for prefill.
        return ((visual_positions[0], visual_positions[-1] + 1),)

    def prefill_token_count(
        self,
        seq: Sequence,
        *,
        available_tokens: int,
    ) -> int:
        if available_tokens <= 0:
            return 0
        remaining = seq.remaining_prefill_tokens
        if not self.enable_chunked_prefill:
            return remaining if remaining <= available_tokens else 0
        start = seq.num_computed_tokens
        count = min(remaining, self.max_chunk_size, available_tokens)
        end = start + count
        for span_start, span_end in self._visual_spans(seq):
            if span_end <= start or span_start >= end:
                continue
            if start < span_start and end < span_end:
                # Stop immediately before visual placeholders; the next chunk
                # can consume that visual span atomically with its payload.
                return span_start - start
            if span_start <= start < span_end and end < span_end:
                required = span_end - start
                if (
                    required > self.max_chunk_size
                    or required > available_tokens
                ):
                    return 0
                end = span_end
        return end - start

    def preemption_candidate(
        self,
        candidates: TypingSequence[Sequence],
    ) -> Sequence | None:
        # Preserve the historical LIFO victim policy while making the choice
        # independently testable and replaceable.
        return candidates[-1] if candidates else None

    def should_schedule_prefill(
        self,
        *,
        has_prefill: bool,
        has_decode: bool,
        consecutive_prefill_batches: int,
    ) -> bool:
        if not has_prefill:
            return False
        if not has_decode:
            return True
        return (
            consecutive_prefill_batches
            < self.max_consecutive_prefill_batches
        )
