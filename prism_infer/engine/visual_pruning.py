"""Visual-token pruning decision helpers for P5.2 preparation.

This module does not enable runtime compression by itself.  It only computes
auditable visual-token retention decisions and can build an experimental
prefill slot mask for focused tests.  Active compression still requires a
decode path that understands the retention mask or a verified physical KV
compaction path.

Inputs:
    seq.token_ids: [seq_len]
    slot_mapping: [total_prefill_tokens]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite

import torch


@dataclass(frozen=True)
class VisualPruningConfig:
    """Visual-token pruning decision configuration."""

    keep_ratio: float = 0.6
    min_keep_tokens: int = 32
    strategy: str = "uniform"

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.min_keep_tokens < 1:
            raise ValueError(f"min_keep_tokens must be >= 1, got {self.min_keep_tokens}")
        if self.strategy not in ("uniform", "score"):
            raise ValueError(f"unsupported strategy: {self.strategy!r}")


@dataclass(frozen=True)
class VisualTokenSpan:
    """A contiguous image/video placeholder-token span in one sequence."""

    modality: str
    start: int
    end: int
    index: int

    @property
    def token_count(self) -> int:
        return self.end - self.start

    def to_record(self) -> dict[str, int | str]:
        return {
            "modality": self.modality,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class PruningDecision:
    """Per-sequence visual-token pruning decision record."""

    seq_id: int
    prompt_token_count: int
    total_visual_tokens: int
    kept_visual_tokens: int
    dropped_visual_tokens: int
    keep_ratio_target: float
    keep_ratio_actual: float
    strategy: str
    visual_token_spans: tuple[VisualTokenSpan, ...]
    kept_token_indices: tuple[int, ...]
    dropped_token_indices: tuple[int, ...]
    physical_compaction: bool = False

    def to_record(self) -> dict[str, object]:
        """Convert the decision into a JSON-serializable audit record."""

        return {
            "seq_id": self.seq_id,
            "prompt_token_count": self.prompt_token_count,
            "total_visual_tokens": self.total_visual_tokens,
            "kept_visual_tokens": self.kept_visual_tokens,
            "dropped_visual_tokens": self.dropped_visual_tokens,
            "keep_ratio_target": self.keep_ratio_target,
            "keep_ratio_actual": self.keep_ratio_actual,
            "strategy": self.strategy,
            "physical_compaction": self.physical_compaction,
            "visual_token_spans": [span.to_record() for span in self.visual_token_spans],
            "kept_token_indices": list(self.kept_token_indices),
            "dropped_token_indices": list(self.dropped_token_indices),
        }


def find_visual_token_spans(seq) -> tuple[VisualTokenSpan, ...]:
    """Scan one sequence and return contiguous image/video token spans."""

    token_ids = getattr(seq, "token_ids", None)
    if token_ids is None:
        raise ValueError("visual pruning requires full seq.token_ids")
    token_ids = [int(token_id) for token_id in token_ids]

    image_token_id = getattr(seq, "image_token_id", None)
    video_token_id = getattr(seq, "video_token_id", None)
    image_token_count = int(getattr(seq, "image_token_count", 0))
    video_token_count = int(getattr(seq, "video_token_count", 0))
    if image_token_count and image_token_id is None:
        raise ValueError("image_token_count is set but image_token_id is missing")
    if video_token_count and video_token_id is None:
        raise ValueError("video_token_count is set but video_token_id is missing")
    if (
        image_token_id is not None
        and video_token_id is not None
        and int(image_token_id) == int(video_token_id)
    ):
        raise ValueError("image_token_id and video_token_id must be different")

    def modality(token_id: int) -> str | None:
        if image_token_id is not None and token_id == int(image_token_id):
            return "image"
        if video_token_id is not None and token_id == int(video_token_id):
            return "video"
        return None

    spans: list[VisualTokenSpan] = []
    span_counts = {"image": 0, "video": 0}
    active_modality: str | None = None
    active_start = 0
    for idx, token_id in enumerate(token_ids):
        current_modality = modality(token_id)
        if current_modality == active_modality:
            continue
        if active_modality is not None:
            span_index = span_counts[active_modality]
            span_counts[active_modality] += 1
            spans.append(VisualTokenSpan(active_modality, active_start, idx, span_index))
        active_modality = current_modality
        active_start = idx
    if active_modality is not None:
        span_index = span_counts[active_modality]
        spans.append(VisualTokenSpan(active_modality, active_start, len(token_ids), span_index))

    actual_image_tokens = sum(span.token_count for span in spans if span.modality == "image")
    actual_video_tokens = sum(span.token_count for span in spans if span.modality == "video")
    if actual_image_tokens != image_token_count:
        raise ValueError(
            "image token count mismatch for pruning decision: "
            f"metadata={image_token_count}, token_ids={actual_image_tokens}"
        )
    if actual_video_tokens != video_token_count:
        raise ValueError(
            "video token count mismatch for pruning decision: "
            f"metadata={video_token_count}, token_ids={actual_video_tokens}"
        )
    return tuple(spans)


def _visual_token_indices(spans: tuple[VisualTokenSpan, ...]) -> tuple[int, ...]:
    indices: list[int] = []
    for span in spans:
        indices.extend(range(span.start, span.end))
    return tuple(indices)


def _target_keep_count(total_visual_tokens: int, config: VisualPruningConfig) -> int:
    if total_visual_tokens <= 0:
        return 0
    if total_visual_tokens <= config.min_keep_tokens:
        return total_visual_tokens
    target = max(config.min_keep_tokens, ceil(total_visual_tokens * config.keep_ratio))
    return min(total_visual_tokens, target)


def _uniform_keep_indices(
    visual_token_indices: tuple[int, ...],
    target_keep: int,
) -> tuple[int, ...]:
    total_visual_tokens = len(visual_token_indices)
    if target_keep >= total_visual_tokens:
        return visual_token_indices
    if target_keep == 1:
        return (visual_token_indices[total_visual_tokens // 2],)

    selected_ordinals: list[int] = []
    selected_set: set[int] = set()
    for keep_idx in range(target_keep):
        ordinal = round(keep_idx * (total_visual_tokens - 1) / (target_keep - 1))
        if ordinal not in selected_set:
            selected_ordinals.append(ordinal)
            selected_set.add(ordinal)
    if len(selected_ordinals) < target_keep:
        for ordinal in range(total_visual_tokens):
            if ordinal in selected_set:
                continue
            selected_ordinals.append(ordinal)
            selected_set.add(ordinal)
            if len(selected_ordinals) == target_keep:
                break

    return tuple(visual_token_indices[ordinal] for ordinal in sorted(selected_ordinals))


def _score_keep_indices(
    visual_token_indices: tuple[int, ...],
    target_keep: int,
    token_scores: Mapping[int, float] | None,
) -> tuple[int, ...]:
    if token_scores is None:
        raise ValueError("strategy='score' requires token_scores keyed by sequence token index")

    scored_indices: list[tuple[float, int]] = []
    for token_index in visual_token_indices:
        if token_index not in token_scores:
            raise ValueError(f"missing score for visual token index {token_index}")
        score = float(token_scores[token_index])
        if not isfinite(score):
            raise ValueError(f"non-finite score for visual token index {token_index}: {score}")
        scored_indices.append((score, token_index))

    selected = sorted(scored_indices, key=lambda item: (-item[0], item[1]))[:target_keep]
    return tuple(sorted(token_index for _, token_index in selected))


def compute_pruning_decision(
    seq,
    config: VisualPruningConfig,
    *,
    token_scores: Mapping[int, float] | None = None,
) -> PruningDecision | None:
    """Compute a visual-token retention decision for one sequence."""

    spans = find_visual_token_spans(seq)
    visual_token_indices = _visual_token_indices(spans)
    total_visual_tokens = len(visual_token_indices)
    if total_visual_tokens == 0:
        return None

    target_keep = _target_keep_count(total_visual_tokens, config)
    if config.strategy == "uniform":
        kept_token_indices = _uniform_keep_indices(visual_token_indices, target_keep)
    elif config.strategy == "score":
        kept_token_indices = _score_keep_indices(
            visual_token_indices,
            target_keep,
            token_scores,
        )
    else:
        raise ValueError(f"unsupported strategy: {config.strategy!r}")

    kept_set = set(kept_token_indices)
    dropped_token_indices = tuple(
        token_index for token_index in visual_token_indices if token_index not in kept_set
    )
    kept_count = len(kept_token_indices)
    dropped_count = len(dropped_token_indices)
    actual_ratio = kept_count / total_visual_tokens

    return PruningDecision(
        seq_id=seq.seq_id,
        prompt_token_count=int(getattr(seq, "num_prompt_tokens", len(getattr(seq, "token_ids", [])))),
        total_visual_tokens=total_visual_tokens,
        kept_visual_tokens=kept_count,
        dropped_visual_tokens=dropped_count,
        keep_ratio_target=config.keep_ratio,
        keep_ratio_actual=actual_ratio,
        strategy=config.strategy,
        visual_token_spans=spans,
        kept_token_indices=kept_token_indices,
        dropped_token_indices=dropped_token_indices,
    )


def apply_pruning_to_slot_mapping(
    slot_mapping: torch.Tensor,
    decisions: list[PruningDecision | None],
    seq_start_indices: list[int],
) -> torch.Tensor:
    """Build an experimental prefill slot mask from pruning decisions.

    slot_mapping: [total_prefill_tokens]
    The returned mask is not enough for complete active compression because
    decode must also understand the retained-token layout.
    """

    if slot_mapping.ndim != 1:
        raise ValueError(f"slot_mapping must be 1D, got shape {list(slot_mapping.shape)}")
    if len(decisions) != len(seq_start_indices):
        raise ValueError(
            "decisions and seq_start_indices length mismatch: "
            f"{len(decisions)} vs {len(seq_start_indices)}"
        )

    slot_mapping = slot_mapping.clone()

    for seq_idx, decision in enumerate(decisions):
        if decision is None:
            continue

        seq_start = seq_start_indices[seq_idx]
        for local_token_idx in decision.dropped_token_indices:
            flat_idx = seq_start + local_token_idx
            if flat_idx < 0 or flat_idx >= slot_mapping.numel():
                raise ValueError(
                    "pruning decision points outside slot_mapping: "
                    f"seq_idx={seq_idx}, local_token_idx={local_token_idx}, "
                    f"flat_idx={flat_idx}, slot_mapping_len={slot_mapping.numel()}"
                )
            slot_mapping[flat_idx] = -1

    return slot_mapping


def build_retained_context_indices(
    decision_record: dict[str, object] | None,
    context_len: int,
) -> tuple[int, ...]:
    """Build retained context indices for logical decode-time visual pruning.

    decision_record: one `PruningDecision.to_record()` dictionary.
    context_len: current decode context length, including generated tokens.

    The returned indices keep all non-visual prompt tokens and all generated
    tokens, while dropping only prompt visual-token positions listed in the
    decision record.
    """

    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    if decision_record is None:
        return tuple(range(context_len))
    if bool(decision_record.get("physical_compaction", False)):
        raise NotImplementedError(
            "physical visual KV compaction is not supported by logical pruning"
        )

    prompt_token_count = int(decision_record.get("prompt_token_count", -1))
    if prompt_token_count < 0:
        raise ValueError("visual pruning record missing prompt_token_count")
    if context_len < prompt_token_count:
        raise ValueError(
            "decode context shorter than pruning prompt: "
            f"context_len={context_len}, prompt_token_count={prompt_token_count}"
        )

    visual_token_indices: set[int] = set()
    for span in decision_record.get("visual_token_spans", []):
        if not isinstance(span, dict):
            raise ValueError(f"visual span record must be a dict, got {type(span)!r}")
        start = int(span["start"])
        end = int(span["end"])
        if start < 0 or end < start or end > prompt_token_count:
            raise ValueError(
                "visual span outside prompt range: "
                f"start={start}, end={end}, prompt_token_count={prompt_token_count}"
            )
        visual_token_indices.update(range(start, end))

    kept_indices = {int(idx) for idx in decision_record.get("kept_token_indices", [])}
    dropped_indices = {int(idx) for idx in decision_record.get("dropped_token_indices", [])}
    if kept_indices & dropped_indices:
        raise ValueError("kept and dropped visual token indices overlap")
    if kept_indices | dropped_indices != visual_token_indices:
        raise ValueError(
            "visual pruning record does not cover exactly the visual prompt tokens"
        )

    return tuple(idx for idx in range(context_len) if idx not in dropped_indices)
