"""KV cache compression metadata and gates.

P5 keeps the floating-point `off` baseline intact and introduces
`visual_prune` as a logical retention strategy.  P5.3 adds `fp8_kv` as a
physical KV storage baseline that keeps logical context unchanged but stores
cache elements in float8.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Sequence as TypingSequence

from prism_infer.engine.visual_pruning import (
    VisualPruningConfig,
    compute_pruning_decision,
)


COMPRESSION_OFF = "off"
COMPRESSION_VISUAL_PRUNE = "visual_prune"
COMPRESSION_VISUAL_COMPACT = "visual_compact"
COMPRESSION_FP8_KV = "fp8_kv"
COMPRESSION_VISUAL_COMPACT_FP8 = "visual_compact_fp8"
SUPPORTED_COMPRESSION_MODES = {
    COMPRESSION_OFF,
    COMPRESSION_VISUAL_PRUNE,
    COMPRESSION_VISUAL_COMPACT,
    COMPRESSION_FP8_KV,
    COMPRESSION_VISUAL_COMPACT_FP8,
}
CUDA_GRAPH_SAFE_COMPRESSION_MODES = frozenset(
    {
        COMPRESSION_OFF,
        COMPRESSION_FP8_KV,
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
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
        """logical pruning mode 是否在当前 batch 真正删除了 visual token。"""

        return self.visual_pruning_active and any(
            record is not None
            and int(record.get("dropped_visual_tokens", 0)) > 0
            for record in self.visual_pruning_records_by_batch
        )

    @property
    def fp8_kv_active(self) -> bool:
        return self.mode in (COMPRESSION_FP8_KV, COMPRESSION_VISUAL_COMPACT_FP8)

    @property
    def visual_compact_active(self) -> bool:
        return self.mode in (
            COMPRESSION_VISUAL_COMPACT,
            COMPRESSION_VISUAL_COMPACT_FP8,
        )


def normalize_compression_mode(mode: str | None) -> str:
    """Normalize and validate the engine compression mode."""

    normalized = (mode or COMPRESSION_OFF).strip().lower()
    if normalized not in SUPPORTED_COMPRESSION_MODES:
        raise ValueError(
            "supported compression_mode values are 'off', 'visual_prune', "
            "'visual_compact', 'fp8_kv', and 'visual_compact_fp8'; "
            f"got {mode!r}"
        )
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
    """Build the config for P5.2 visual-pruning decisions."""

    return VisualPruningConfig(
        keep_ratio=float(getattr(config, "visual_pruning_keep_ratio", 0.6)),
        min_keep_tokens=int(getattr(config, "visual_pruning_min_keep_tokens", 32)),
        strategy=str(getattr(config, "visual_pruning_strategy", "uniform")),
        attention_last_n_layers=int(
            getattr(config, "visual_pruning_attention_last_n_layers", 1)
        ),
    )


def _sequence_visual_token_count(seq) -> int:
    """Return the number of visual placeholder tokens recorded on a sequence."""

    return int(getattr(seq, "image_token_count", 0)) + int(
        getattr(seq, "video_token_count", 0)
    )


def _with_batch_index(record: dict[str, object], batch_index: int) -> dict[str, object]:
    """Return an audit record copy annotated with current batch position."""

    annotated = dict(record)
    annotated["batch_index"] = batch_index
    return annotated


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
    )
    if not shadow_enabled and not active:
        return ()
    if not is_prefill and not active:
        return ()

    if is_prefill:
        pruning_config = build_visual_pruning_config(config)
        if active and pruning_config.strategy == "attention":
            # Runtime score 在选定 decoder layers 内收集；完整 prefill 结束后
            # 才生成 decision，供 logical pruning 或 physical compaction 复用。
            return tuple(None for _ in seqs)
        records: list[dict[str, object] | None] = []
        for batch_index, seq in enumerate(seqs):
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
            raise RuntimeError(
                "visual_prune decode requires batch-aligned pruning records"
            )
        return
    if metadata.mode in (
        COMPRESSION_VISUAL_COMPACT,
        COMPRESSION_VISUAL_COMPACT_FP8,
    ):
        if (
            not metadata.is_prefill
            and metadata.total_visual_tokens > 0
            and not metadata.visual_pruning_records_by_batch
        ):
            raise RuntimeError(
                "visual_compact decode requires batch-aligned pruning records"
            )
        return
    if metadata.mode == COMPRESSION_FP8_KV:
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
