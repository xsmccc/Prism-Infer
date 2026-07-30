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

    def waiting_prefill_index(
        self,
        candidates: TypingSequence[Sequence],
        *,
        has_decode: bool,
        decode_batches_since_heavy_prefill: int,
        light_prefill_bypasses_since_heavy: int,
    ) -> int | None: ...

    def is_heavy_prefill(self, vision_patches: int) -> bool: ...


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
            raise ValueError("max_consecutive_prefill_batches must be positive")

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
        if self.max_queue_size is not None and queued_requests >= self.max_queue_size:
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
            index for index, token_id in enumerate(seq.prompt_token_ids) if token_id in visual_ids
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
        start = seq.effective_prefill_start
        count = min(remaining, self.max_chunk_size, available_tokens)
        end = start + count
        multimodal_prefix_boundary = seq.multimodal_prefix_boundary
        if (
            seq.multimodal_prefix_cache_enabled
            and seq.kv_layout is None
            and multimodal_prefix_boundary is not None
            and start < multimodal_prefix_boundary < end
        ):
            end = multimodal_prefix_boundary
        for span_start, span_end in self._visual_spans(seq):
            if span_end <= start or span_start >= end:
                continue
            if start < span_start and end < span_end:
                # Stop immediately before visual placeholders; the next chunk
                # can consume that visual span atomically with its payload.
                return span_start - start
            if span_start <= start < span_end and end < span_end:
                required = span_end - start
                if required > self.max_chunk_size or required > available_tokens:
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
        return consecutive_prefill_batches < self.max_consecutive_prefill_batches

    def waiting_prefill_index(
        self,
        candidates: TypingSequence[Sequence],
        *,
        has_decode: bool,
        decode_batches_since_heavy_prefill: int,
        light_prefill_bypasses_since_heavy: int,
    ) -> int | None:
        del (
            has_decode,
            decode_batches_since_heavy_prefill,
            light_prefill_bypasses_since_heavy,
        )
        return 0 if candidates else None

    def is_heavy_prefill(self, vision_patches: int) -> bool:
        del vision_patches
        return False


@dataclass(frozen=True, slots=True)
class VisionAwareSchedulerPolicy(FCFSSchedulerPolicy):
    """Protect decode cadence from long, atomic vision-prefill batches.

    A heavy request may be bypassed by the oldest lightweight request while
    decode credit accumulates. Decode credit or the explicit bypass cap then
    restores the oldest request's priority.
    """

    heavy_prefill_vision_patch_threshold: int = 4096
    min_decode_batches_between_heavy_prefills: int = 32
    max_light_prefill_bypasses_per_heavy: int = 2
    name: str = "vision_aware"

    def __post_init__(self) -> None:
        FCFSSchedulerPolicy.__post_init__(self)
        if self.heavy_prefill_vision_patch_threshold <= 0:
            raise ValueError("heavy_prefill_vision_patch_threshold must be positive")
        if self.min_decode_batches_between_heavy_prefills <= 0:
            raise ValueError("min_decode_batches_between_heavy_prefills must be positive")
        if self.max_light_prefill_bypasses_per_heavy <= 0:
            raise ValueError("max_light_prefill_bypasses_per_heavy must be positive")

    def waiting_prefill_index(
        self,
        candidates: TypingSequence[Sequence],
        *,
        has_decode: bool,
        decode_batches_since_heavy_prefill: int,
        light_prefill_bypasses_since_heavy: int,
    ) -> int | None:
        if not candidates:
            return None
        if (
            not has_decode
            or decode_batches_since_heavy_prefill
            >= self.min_decode_batches_between_heavy_prefills
            or light_prefill_bypasses_since_heavy
            >= self.max_light_prefill_bypasses_per_heavy
            or not self.is_heavy_prefill(candidates[0].prefill_vision_patch_count)
        ):
            return 0
        for index, candidate in enumerate(candidates[1:], start=1):
            if not self.is_heavy_prefill(candidate.prefill_vision_patch_count):
                return index
        return None

    def is_heavy_prefill(self, vision_patches: int) -> bool:
        return vision_patches >= self.heavy_prefill_vision_patch_threshold


@dataclass(frozen=True, slots=True)
class SLOAwareSchedulerPolicy(FCFSSchedulerPolicy):
    """Order waiting prefills by TTFT deadline and isolate cost tiers.

    Requests without an explicit TTFT SLO retain FCFS ordering. Under decode
    load, co-batching is restricted to comparable prefill-cost tiers so a
    short text request cannot inherit the latency of a multi-image or video
    prefill.
    """

    heavy_prefill_vision_patch_threshold: int = 4096
    prefill_reserve_ms_by_tier: tuple[float, float, float] = (
        120.0,
        250.0,
        700.0,
    )
    name: str = "slo_aware"

    def __post_init__(self) -> None:
        FCFSSchedulerPolicy.__post_init__(self)
        if self.heavy_prefill_vision_patch_threshold <= 0:
            raise ValueError(
                "heavy_prefill_vision_patch_threshold must be positive"
            )
        if any(value <= 0 for value in self.prefill_reserve_ms_by_tier):
            raise ValueError("prefill reserve estimates must be positive")

    @staticmethod
    def ttft_deadline_ns(seq: Sequence) -> int | None:
        if seq.submitted_ns is None or seq.ttft_slo_ms is None:
            return None
        return seq.submitted_ns + int(seq.ttft_slo_ms * 1_000_000)

    def prefill_cost_tier(self, seq: Sequence) -> int:
        vision_patches = seq.prefill_vision_patch_count
        if vision_patches == 0 and seq.remaining_prefill_tokens <= 256:
            return 0
        if vision_patches < self.heavy_prefill_vision_patch_threshold:
            return 1
        return 2

    def prefill_reserve_ns(self, seq: Sequence) -> int:
        tier = self.prefill_cost_tier(seq)
        return int(self.prefill_reserve_ms_by_tier[tier] * 1_000_000)

    def can_co_batch(self, anchor: Sequence, candidate: Sequence) -> bool:
        """Keep text isolated while allowing visual requests to share work."""

        anchor_tier = self.prefill_cost_tier(anchor)
        candidate_tier = self.prefill_cost_tier(candidate)
        return (anchor_tier == 0) == (candidate_tier == 0)

    def waiting_prefill_index(
        self,
        candidates: TypingSequence[Sequence],
        *,
        has_decode: bool,
        decode_batches_since_heavy_prefill: int,
        light_prefill_bypasses_since_heavy: int,
    ) -> int | None:
        del (
            has_decode,
            decode_batches_since_heavy_prefill,
            light_prefill_bypasses_since_heavy,
        )
        if not candidates:
            return None

        def priority(index: int) -> tuple[int, int, int]:
            candidate = candidates[index]
            deadline_ns = self.ttft_deadline_ns(candidate)
            submitted_ns = candidate.submitted_ns
            return (
                (
                    deadline_ns - self.prefill_reserve_ns(candidate)
                    if deadline_ns is not None
                    else 2**63 - 1
                ),
                submitted_ns if submitted_ns is not None else 2**63 - 1,
                index,
            )

        return min(range(len(candidates)), key=priority)

    def is_heavy_prefill(self, vision_patches: int) -> bool:
        return vision_patches >= self.heavy_prefill_vision_patch_threshold
