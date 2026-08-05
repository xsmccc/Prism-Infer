"""KV cache compression metadata and runtime checks.

The floating-point ``off`` mode provides the uncompressed baseline.
``visual_prune`` controls logical retention, while the FP8 modes select
physical KV storage without changing the logical context contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence as TypingSequence
from dataclasses import asdict, dataclass

from prism_infer.engine.visual_pruning import (
    DEFAULT_VISUAL_PRUNING_ATTENTION_LAST_N_LAYERS,
    DEFAULT_VISUAL_PRUNING_KEEP_RATIO,
    DEFAULT_VISUAL_PRUNING_MIN_KEEP_TOKENS,
    DEFAULT_VISUAL_PRUNING_STRATEGY,
    DEFAULT_VISUAL_PRUNING_VIDEO_MIN_KEEP_TOKENS,
    VisualPruningConfig,
    compute_pruning_decision,
    find_visual_token_spans,
)

COMPRESSION_OFF = "off"
COMPRESSION_VISUAL_PRUNE = "visual_prune"
COMPRESSION_VISUAL_COMPACT = "visual_compact"
COMPRESSION_FP8_KV = "fp8_kv"
COMPRESSION_VISUAL_COMPACT_FP8 = "visual_compact_fp8"
COMPRESSION_SCALED_FP8_KV = "scaled_fp8_kv"
COMPRESSION_VISUAL_COMPACT_SCALED_FP8 = "visual_compact_scaled_fp8"
SUPPORTED_COMPRESSION_MODES = {
    COMPRESSION_OFF,
    COMPRESSION_VISUAL_PRUNE,
    COMPRESSION_VISUAL_COMPACT,
    COMPRESSION_FP8_KV,
    COMPRESSION_VISUAL_COMPACT_FP8,
    COMPRESSION_SCALED_FP8_KV,
    COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
}
CUDA_GRAPH_SAFE_COMPRESSION_MODES = frozenset(
    {
        COMPRESSION_OFF,
        COMPRESSION_FP8_KV,
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
        COMPRESSION_SCALED_FP8_KV,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    }
)


@dataclass(frozen=True)
class CompressionMetadata:
    """Per-step compression state carried through Context."""

    mode: str
    is_prefill: bool
    num_sequences: int
    total_prompt_tokens: int
    total_image_tokens: int
    total_video_tokens: int
    block_size: int
    visual_pruning_shadow_enabled: bool = False
    visual_pruning_config: dict[str, object] | None = None
    visual_pruning_decision_records: tuple[dict[str, object], ...] = ()
    visual_pruning_records_by_batch: tuple[dict[str, object] | None, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.mode != COMPRESSION_OFF

    @property
    def total_visual_tokens(self) -> int:
        return self.total_image_tokens + self.total_video_tokens

    @property
    def visual_pruning_active(self) -> bool:
        return self.mode == COMPRESSION_VISUAL_PRUNE

    @property
    def visual_pruning_effective(self) -> bool:
        """Return whether logical pruning removed visual tokens in this batch."""

        return self.visual_pruning_active and any(
            record is not None and int(record.get("dropped_visual_tokens", 0)) > 0
            for record in self.visual_pruning_records_by_batch
        )

    @property
    def fp8_kv_active(self) -> bool:
        return compression_mode_uses_fp8_payload(self.mode)

    @property
    def scaled_fp8_kv_active(self) -> bool:
        return compression_mode_uses_token_head_scales(self.mode)

    @property
    def unit_scale_fp8_kv_active(self) -> bool:
        return self.mode in (COMPRESSION_FP8_KV, COMPRESSION_VISUAL_COMPACT_FP8)

    @property
    def visual_compact_active(self) -> bool:
        return self.mode in (
            COMPRESSION_VISUAL_COMPACT,
            COMPRESSION_VISUAL_COMPACT_FP8,
            COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
        )


def compression_mode_uses_fp8_payload(mode: str) -> bool:
    """Return whether a mode stores K/V payload elements as E4M3FN."""

    return mode in (
        COMPRESSION_FP8_KV,
        COMPRESSION_VISUAL_COMPACT_FP8,
        COMPRESSION_SCALED_FP8_KV,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    )


def compression_mode_uses_token_head_scales(mode: str) -> bool:
    """Return whether a mode owns independent K/V token-head scale caches."""

    return mode in (
        COMPRESSION_SCALED_FP8_KV,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    )


def normalize_compression_mode(mode: str | None) -> str:
    """Normalize and validate the engine compression mode."""

    normalized = (mode or COMPRESSION_OFF).strip().lower()
    if normalized not in SUPPORTED_COMPRESSION_MODES:
        supported = ", ".join(repr(value) for value in sorted(SUPPORTED_COMPRESSION_MODES))
        raise ValueError(f"supported compression_mode values are {supported}; got {mode!r}")
    return normalized


def compression_mode_supports_cuda_graph(mode: str) -> bool:
    """返回压缩模式是否能完全通过静态 replay tensor 表达。"""

    return mode in CUDA_GRAPH_SAFE_COMPRESSION_MODES


def compression_supports_cuda_graph(
    metadata: CompressionMetadata | None,
) -> bool:
    """返回当前 decode 压缩状态能否复用静态 CUDA Graph。

    FP8 和 physical compaction 只改变 KV dtype、physical context length 与
    block table；这些状态均通过 capture 时绑定、replay 前更新的 tensor 表达。
    logical ``visual_prune`` 仍依赖动态 retained-slot gather，因此显式拒绝。
    """

    if metadata is None:
        return True
    return compression_mode_supports_cuda_graph(metadata.mode)


def build_visual_pruning_config(config) -> VisualPruningConfig:
    """Build a validated visual-pruning decision config."""

    return VisualPruningConfig(
        keep_ratio=float(
            getattr(
                config,
                "visual_pruning_keep_ratio",
                DEFAULT_VISUAL_PRUNING_KEEP_RATIO,
            )
        ),
        min_keep_tokens=int(
            getattr(
                config,
                "visual_pruning_min_keep_tokens",
                DEFAULT_VISUAL_PRUNING_MIN_KEEP_TOKENS,
            )
        ),
        video_min_keep_tokens=(
            None
            if getattr(
                config,
                "visual_pruning_video_min_keep_tokens",
                DEFAULT_VISUAL_PRUNING_VIDEO_MIN_KEEP_TOKENS,
            )
            is None
            else int(config.visual_pruning_video_min_keep_tokens)
        ),
        strategy=str(
            getattr(
                config,
                "visual_pruning_strategy",
                DEFAULT_VISUAL_PRUNING_STRATEGY,
            )
        ),
        attention_last_n_layers=int(
            getattr(
                config,
                "visual_pruning_attention_last_n_layers",
                DEFAULT_VISUAL_PRUNING_ATTENTION_LAST_N_LAYERS,
            )
        ),
    )


def _sequence_visual_token_count(seq) -> int:
    """Return the number of visual placeholder tokens recorded on a sequence."""

    return int(getattr(seq, "image_token_count", 0)) + int(getattr(seq, "video_token_count", 0))


def _with_batch_index(record: dict[str, object], batch_index: int) -> dict[str, object]:
    """Return an audit record copy annotated with current batch position."""

    annotated = dict(record)
    annotated["batch_index"] = batch_index
    return annotated


def _replay_index_tuple(value: object, *, name: str) -> tuple[int, ...]:
    """Validate one sorted, unique sequence-local token-index list."""

    if not isinstance(value, list | tuple):
        raise ValueError(f"visual pruning replay {name} must be a list or tuple")
    indices = tuple(value)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"visual pruning replay {name} must contain integers")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError(f"visual pruning replay {name} must be sorted and unique")
    return indices


def _normalize_visual_pruning_replay_record(
    seq,
    pruning_config: VisualPruningConfig,
    replay_record: object,
) -> dict[str, object]:
    """Validate and normalize a locked visual-token selection for fresh prefill."""

    if not isinstance(replay_record, Mapping):
        raise ValueError("visual_pruning_replay_record must be a mapping")
    source_strategy = replay_record.get("strategy")
    if source_strategy != pruning_config.strategy:
        raise ValueError(
            "visual pruning replay strategy differs from runtime strategy: "
            f"source={source_strategy!r}, runtime={pruning_config.strategy!r}"
        )

    source_prompt_token_count = replay_record.get("prompt_token_count")
    if (
        isinstance(source_prompt_token_count, bool)
        or not isinstance(source_prompt_token_count, int)
        or source_prompt_token_count != int(seq.num_prompt_tokens)
    ):
        raise ValueError(
            "visual pruning replay prompt length differs from the current prompt: "
            f"source={source_prompt_token_count!r}, current={seq.num_prompt_tokens}"
        )

    expected_spans = [span.to_record() for span in find_visual_token_spans(seq)]
    if replay_record.get("visual_token_spans") != expected_spans:
        raise ValueError("visual pruning replay spans differ from the current prompt")
    visual_indices = {
        token_index
        for span in expected_spans
        for token_index in range(int(span["start"]), int(span["end"]))
    }
    kept_indices = _replay_index_tuple(
        replay_record.get("kept_token_indices"),
        name="kept_token_indices",
    )
    dropped_indices = _replay_index_tuple(
        replay_record.get("dropped_token_indices"),
        name="dropped_token_indices",
    )
    kept_set = set(kept_indices)
    dropped_set = set(dropped_indices)
    if kept_set & dropped_set:
        raise ValueError("visual pruning replay kept and dropped indices overlap")
    if kept_set | dropped_set != visual_indices:
        raise ValueError("visual pruning replay must partition exactly the visual tokens")
    count_fields = {
        "total_visual_tokens": len(visual_indices),
        "kept_visual_tokens": len(kept_indices),
        "dropped_visual_tokens": len(dropped_indices),
    }
    for name, expected in count_fields.items():
        value = replay_record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"visual pruning replay {name} is inconsistent: "
                f"source={value!r}, expected={expected}"
            )

    score_config = VisualPruningConfig(
        keep_ratio=pruning_config.keep_ratio,
        min_keep_tokens=pruning_config.min_keep_tokens,
        video_min_keep_tokens=pruning_config.video_min_keep_tokens,
        strategy="score",
        attention_last_n_layers=pruning_config.attention_last_n_layers,
    )
    token_scores = {
        token_index: (1.0 if token_index in kept_set else 0.0) for token_index in visual_indices
    }
    validated = compute_pruning_decision(
        seq,
        score_config,
        token_scores=token_scores,
    )
    if validated is None or validated.kept_token_indices != kept_indices:
        raise ValueError("visual pruning replay kept count differs from the runtime policy")

    normalized = validated.to_record()
    normalized.update(
        {
            "strategy": pruning_config.strategy,
            "selection_replay_locked": True,
            "selection_source_seq_id": replay_record.get("seq_id"),
            "selection_source_prompt_token_count": source_prompt_token_count,
        }
    )
    for name in (
        "score_source",
        "score_layers",
        "score_min",
        "score_max",
        "score_mean",
        "selection_source_sample_id",
    ):
        if name in replay_record:
            normalized[name] = replay_record[name]
    return normalized


def _build_visual_pruning_records_by_batch(
    config,
    seqs: TypingSequence,
    *,
    mode: str,
    is_prefill: bool,
) -> tuple[dict[str, object] | None, ...]:
    """Build batch-aligned visual-pruning records for shadow or active mode."""

    shadow_enabled = bool(getattr(config, "enable_visual_pruning_shadow", False))
    active = mode in (
        COMPRESSION_VISUAL_PRUNE,
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    )
    if not shadow_enabled and not active:
        return ()
    if not is_prefill and not active:
        return ()

    if is_prefill:
        pruning_config = build_visual_pruning_config(config)
        records: list[dict[str, object] | None] = []
        for batch_index, seq in enumerate(seqs):
            replay_record = getattr(seq, "visual_pruning_replay_record", None)
            if replay_record is not None:
                compact_record = getattr(seq, "visual_pruning_decision_record", None)
                if getattr(seq, "kv_layout", None) is not None:
                    if not isinstance(compact_record, dict) or not bool(
                        compact_record.get("selection_replay_locked")
                    ):
                        raise RuntimeError(
                            "locked visual pruning replay lost its compacted decision"
                        )
                    records.append(_with_batch_index(compact_record, batch_index))
                    continue
                record = _with_batch_index(
                    _normalize_visual_pruning_replay_record(
                        seq,
                        pruning_config,
                        replay_record,
                    ),
                    batch_index,
                )
                if not active:
                    raise ValueError(
                        "visual pruning replay requires an active visual compaction mode"
                    )
                seq.visual_pruning_decision_record = record
                records.append(record)
                continue
            cached_record = getattr(
                seq,
                "visual_pruning_decision_record",
                None,
            )
            if (
                getattr(seq, "multimodal_prefix_cache_hit", False)
                and getattr(seq, "kv_layout", None) is not None
                and cached_record is not None
            ):
                records.append(_with_batch_index(cached_record, batch_index))
                continue
            if (active or shadow_enabled) and pruning_config.strategy == "attention":
                # Runtime scores are collected in selected decoder layers.
                # The decision is finalized only after a complete cold prefill.
                # Shadow mode keeps the dense KV layout and is used only to
                # produce a selection record for a later locked replay.
                records.append(None)
                continue
            decision = compute_pruning_decision(seq, pruning_config)
            record = (
                _with_batch_index(decision.to_record(), batch_index)
                if decision is not None
                else None
            )
            if active:
                seq.visual_pruning_decision_record = record
            records.append(record)
        return tuple(records)

    records = []
    for batch_index, seq in enumerate(seqs):
        record = getattr(seq, "visual_pruning_decision_record", None)
        if record is None:
            if _sequence_visual_token_count(seq) > 0:
                raise RuntimeError(
                    "visual_prune decode requires a prefill pruning decision; "
                    f"missing record for seq_id={seq.seq_id}"
                )
            records.append(None)
            continue
        records.append(_with_batch_index(record, batch_index))
    return tuple(records)


def build_compression_metadata(
    config,
    seqs: TypingSequence,
    *,
    is_prefill: bool,
) -> CompressionMetadata:
    """Build compression metadata for one prefill/decode step."""

    mode = normalize_compression_mode(getattr(config, "compression_mode", None))
    shadow_enabled = bool(getattr(config, "enable_visual_pruning_shadow", False))
    pruning_metadata_enabled = shadow_enabled or mode in (
        COMPRESSION_VISUAL_PRUNE,
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    )
    visual_pruning_config = (
        asdict(build_visual_pruning_config(config)) if pruning_metadata_enabled else None
    )
    visual_pruning_records_by_batch = _build_visual_pruning_records_by_batch(
        config,
        seqs,
        mode=mode,
        is_prefill=is_prefill,
    )
    visual_pruning_decision_records = tuple(
        record for record in visual_pruning_records_by_batch if record is not None
    )
    return CompressionMetadata(
        mode=mode,
        is_prefill=is_prefill,
        num_sequences=len(seqs),
        total_prompt_tokens=sum(int(getattr(seq, "num_prompt_tokens", 0)) for seq in seqs),
        total_image_tokens=sum(int(getattr(seq, "image_token_count", 0)) for seq in seqs),
        total_video_tokens=sum(int(getattr(seq, "video_token_count", 0)) for seq in seqs),
        block_size=int(getattr(config, "kvcache_block_size", 0)),
        visual_pruning_shadow_enabled=shadow_enabled,
        visual_pruning_config=visual_pruning_config,
        visual_pruning_decision_records=visual_pruning_decision_records,
        visual_pruning_records_by_batch=visual_pruning_records_by_batch,
    )


def ensure_compression_off(metadata: CompressionMetadata | None) -> None:
    """Guard paths that intentionally require the exact compression-off baseline."""

    if metadata is not None and metadata.enabled:
        raise NotImplementedError(
            f"compression_mode={metadata.mode!r} is not allowed on an off-only path"
        )


def ensure_supported_compression_metadata(
    metadata: CompressionMetadata | None,
) -> None:
    """Reject compression metadata states that have no runtime implementation."""

    if metadata is None:
        return
    if metadata.mode == COMPRESSION_OFF:
        return
    if metadata.mode == COMPRESSION_VISUAL_PRUNE:
        if (
            not metadata.is_prefill
            and metadata.total_visual_tokens > 0
            and not metadata.visual_pruning_records_by_batch
        ):
            raise RuntimeError("visual_prune decode requires batch-aligned pruning records")
        return
    if metadata.mode in (
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
        COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    ):
        if (
            not metadata.is_prefill
            and metadata.total_visual_tokens > 0
            and not metadata.visual_pruning_records_by_batch
        ):
            raise RuntimeError("visual_compact decode requires batch-aligned pruning records")
        return
    if metadata.mode in (COMPRESSION_FP8_KV, COMPRESSION_SCALED_FP8_KV):
        return
    raise NotImplementedError(f"compression_mode={metadata.mode!r} is not implemented")


def get_visual_pruning_record_for_batch(
    metadata: CompressionMetadata,
    batch_index: int,
) -> dict[str, object] | None:
    """Return the active visual-pruning decision for one decode batch row."""

    if not metadata.visual_pruning_active:
        return None
    records = metadata.visual_pruning_records_by_batch
    if not records:
        if metadata.total_visual_tokens == 0:
            return None
        raise RuntimeError("visual_prune metadata has no batch-aligned records")
    if batch_index < 0 or batch_index >= len(records):
        raise RuntimeError(
            "visual_prune batch index outside records: "
            f"batch_index={batch_index}, records={len(records)}"
        )
    return records[batch_index]
