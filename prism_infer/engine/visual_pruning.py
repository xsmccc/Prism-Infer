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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class VisualPruningConfig:
    """Visual-token pruning decision configuration."""

    keep_ratio: float = 0.6
    min_keep_tokens: int = 32
    strategy: str = "uniform"
    attention_last_n_layers: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.min_keep_tokens < 1:
            raise ValueError(f"min_keep_tokens must be >= 1, got {self.min_keep_tokens}")
        if self.strategy not in ("uniform", "score", "attention"):
            raise ValueError(f"unsupported strategy: {self.strategy!r}")
        if self.attention_last_n_layers < 1:
            raise ValueError(
                "attention_last_n_layers must be >= 1, got "
                f"{self.attention_last_n_layers}"
            )


@dataclass(frozen=True)
class _RuntimeScoreSequence:
    """一个 flattened prefill row 的视觉索引。"""

    seq_id: int
    token_start: int
    token_end: int
    visual_token_indices: tuple[int, ...]


class RuntimeVisualTokenScorer:
    """在指定 decoder layers 聚合最后 query 对视觉 token 的 attention mass。

    q: [total_tokens, local_q_heads, head_dim]
    k: [total_tokens, local_kv_heads, head_dim]

    scorer 仅保存每条序列的 device tensor；CPU materialization 延迟到完整
    prefill forward 结束，避免在每个 attention layer 内触发 host synchronization。
    TP 下每个 rank 先对本地 Q heads 求均值，finalize 时再 all-reduce 得到全局均值。
    """

    def __init__(
        self,
        seqs: Sequence[object],
        *,
        layer_ids: Sequence[int],
    ) -> None:
        normalized_layers = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        if not normalized_layers or normalized_layers[0] < 0:
            raise ValueError(f"layer_ids must be non-empty and non-negative: {layer_ids}")

        specs: list[_RuntimeScoreSequence] = []
        token_start = 0
        for seq in seqs:
            token_count = len(seq)
            spans = find_visual_token_spans(seq)
            visual_indices = _visual_token_indices(spans)
            specs.append(
                _RuntimeScoreSequence(
                    seq_id=int(seq.seq_id),
                    token_start=token_start,
                    token_end=token_start + token_count,
                    visual_token_indices=visual_indices,
                )
            )
            token_start += token_count

        self.layer_ids = normalized_layers
        self.sequences = tuple(specs)
        self.total_tokens = token_start
        self._score_sums: dict[int, torch.Tensor] = {}
        self._observed_layers: set[int] = set()

    def observe(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float,
    ) -> None:
        """聚合一个目标 layer 的 query-aware visual attention score。"""

        if layer_id not in self.layer_ids:
            return
        if layer_id in self._observed_layers:
            raise RuntimeError(f"runtime visual scorer observed layer {layer_id} twice")
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "runtime visual scorer expects q/k [tokens, heads, dim], got "
                f"{list(q.shape)} and {list(k.shape)}"
            )
        if q.shape[0] != self.total_tokens or k.shape[0] != self.total_tokens:
            raise ValueError(
                "runtime visual scorer token count mismatch: "
                f"expected={self.total_tokens}, q={q.shape[0]}, k={k.shape[0]}"
            )
        if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                "runtime visual scorer requires compatible GQA heads/dim: "
                f"q={list(q.shape)}, k={list(k.shape)}"
            )

        groups = q.shape[1] // k.shape[1]
        for spec in self.sequences:
            if not spec.visual_token_indices:
                continue
            # q_last: [local_q_heads, head_dim]
            q_last = q[spec.token_end - 1].detach().float()
            # keys: [seq_tokens, local_q_heads, head_dim]
            keys = k[spec.token_start:spec.token_end].detach().float()
            keys = keys.repeat_interleave(groups, dim=1)
            # probs: [local_q_heads, seq_tokens]
            logits = torch.einsum("hd,thd->ht", q_last, keys) * float(scale)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            local_indices = torch.tensor(
                spec.visual_token_indices,
                dtype=torch.long,
                device=probs.device,
            )
            visual_scores = probs.index_select(0, local_indices)
            previous = self._score_sums.get(spec.seq_id)
            self._score_sums[spec.seq_id] = (
                visual_scores if previous is None else previous + visual_scores
            )
        self._observed_layers.add(layer_id)

    def finalize(self) -> dict[int, dict[int, float]]:
        """完成跨 layer/TP 聚合并返回 sequence-local token score maps。"""

        missing = sorted(set(self.layer_ids) - self._observed_layers)
        if missing:
            raise RuntimeError(f"runtime visual scorer missing layers: {missing}")
        observation_count = len(self._observed_layers)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        result: dict[int, dict[int, float]] = {}
        for spec in self.sequences:
            if not spec.visual_token_indices:
                result[spec.seq_id] = {}
                continue
            scores = self._score_sums.get(spec.seq_id)
            if scores is None:
                raise RuntimeError(
                    f"runtime visual scorer has no scores for seq_id={spec.seq_id}"
                )
            scores = scores / observation_count
            if world_size > 1:
                dist.all_reduce(scores, op=dist.ReduceOp.SUM)
                scores = scores / world_size
            values = scores.cpu().tolist()
            result[spec.seq_id] = {
                token_index: float(score)
                for token_index, score in zip(spec.visual_token_indices, values)
            }
        return result


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

        kept_set = set(self.kept_token_indices)
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
            "kept_visual_tokens_by_span": [
                {
                    "modality": span.modality,
                    "span_index": span.index,
                    "kept_tokens": sum(
                        token_index in kept_set
                        for token_index in range(span.start, span.end)
                    ),
                }
                for span in self.visual_token_spans
            ],
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
        raise ValueError(
            "score-based pruning requires token_scores keyed by sequence token index"
        )

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
    elif config.strategy == "attention":
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


def build_runtime_visual_token_scorer(
    seqs: Sequence[object],
    *,
    num_hidden_layers: int,
    attention_last_n_layers: int,
) -> RuntimeVisualTokenScorer:
    """构造最后 N 个 decoder layers 的 runtime attention scorer。"""

    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    if not 1 <= attention_last_n_layers <= num_hidden_layers:
        raise ValueError(
            "attention_last_n_layers must be within model depth: "
            f"last_n={attention_last_n_layers}, layers={num_hidden_layers}"
        )
    first_layer = num_hidden_layers - attention_last_n_layers
    return RuntimeVisualTokenScorer(
        seqs,
        layer_ids=range(first_layer, num_hidden_layers),
    )


def finalize_attention_pruning_decisions(
    seqs: Sequence[object],
    config: VisualPruningConfig,
    scorer: RuntimeVisualTokenScorer,
) -> tuple[dict[str, object] | None, ...]:
    """用 runtime attention scores 生成并持久化 batch-aligned decisions。"""

    if config.strategy != "attention":
        raise ValueError(
            "runtime attention finalization requires strategy='attention', got "
            f"{config.strategy!r}"
        )
    score_maps = scorer.finalize()
    records: list[dict[str, object] | None] = []
    for batch_index, seq in enumerate(seqs):
        token_scores = score_maps.get(int(seq.seq_id))
        if token_scores is None:
            raise RuntimeError(f"missing runtime score map for seq_id={seq.seq_id}")
        decision = compute_pruning_decision(
            seq,
            config,
            token_scores=token_scores,
        )
        if decision is None:
            records.append(None)
            continue
        values = tuple(token_scores.values())
        record = decision.to_record()
        record.update(
            {
                "batch_index": batch_index,
                "score_source": "prefill_last_query_attention",
                "score_layers": list(scorer.layer_ids),
                "score_min": min(values),
                "score_max": max(values),
                "score_mean": sum(values) / len(values),
            }
        )
        seq.visual_pruning_decision_record = record
        records.append(record)
    return tuple(records)


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


def build_retained_slot_mapping(
    decision_record: dict[str, object] | None,
    context_len: int,
    block_table: Sequence[int],
    block_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """把 retained logical indices 映射为可跨 attention 层复用的 physical slots。

    返回 shape 为 ``[retained_len]`` 的 int64 tensor。该 mapping 只改变 logical
    pruning 的读取方式，不移动 KV、不释放 block，因此不是 physical compaction。
    """

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    retained_indices = build_retained_context_indices(decision_record, context_len)
    if not retained_indices:
        raise RuntimeError("visual pruning retained zero decode context tokens")

    required_blocks = (context_len + block_size - 1) // block_size
    if len(block_table) < required_blocks:
        raise RuntimeError(
            "visual pruning block table is shorter than decode context: "
            f"context_len={context_len}, block_size={block_size}, "
            f"required_blocks={required_blocks}, actual_blocks={len(block_table)}"
        )

    physical_slots: list[int] = []
    for logical_index in retained_indices:
        block_ordinal = logical_index // block_size
        block_id = int(block_table[block_ordinal])
        if block_id < 0:
            raise RuntimeError(
                "visual pruning logical index maps to an invalid block: "
                f"logical_index={logical_index}, block_ordinal={block_ordinal}, "
                f"block_id={block_id}"
            )
        block_offset = logical_index % block_size
        physical_slots.append(block_id * block_size + block_offset)

    target_device = torch.device("cpu" if device is None else device)
    # physical_slots: [retained_len]
    cpu_slots = torch.tensor(
        physical_slots,
        dtype=torch.long,
        pin_memory=target_device.type == "cuda",
    )
    return cpu_slots.to(target_device, non_blocking=target_device.type == "cuda")
