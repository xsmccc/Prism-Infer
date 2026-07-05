"""KV cache compression metadata and gates.

P5.0 only establishes the compression-off baseline.  The execution path carries
metadata so later pruning/quantization strategies have a stable integration
point, but any non-off mode must fail loudly until it is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence as TypingSequence


COMPRESSION_OFF = "off"
SUPPORTED_COMPRESSION_MODES = {COMPRESSION_OFF}


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

    @property
    def enabled(self) -> bool:
        return self.mode != COMPRESSION_OFF

    @property
    def total_visual_tokens(self) -> int:
        return self.total_image_tokens + self.total_video_tokens


def normalize_compression_mode(mode: str | None) -> str:
    """Normalize and validate the engine compression mode."""

    normalized = (mode or COMPRESSION_OFF).strip().lower()
    if normalized not in SUPPORTED_COMPRESSION_MODES:
        raise ValueError(
            "P5.0 only supports compression_mode='off'; "
            f"got {mode!r}"
        )
    return normalized


def build_compression_metadata(
    config,
    seqs: TypingSequence,
    *,
    is_prefill: bool,
) -> CompressionMetadata:
    """Build no-op compression metadata for one prefill/decode step."""

    mode = normalize_compression_mode(getattr(config, "compression_mode", None))
    return CompressionMetadata(
        mode=mode,
        is_prefill=is_prefill,
        num_sequences=len(seqs),
        total_prompt_tokens=sum(int(getattr(seq, "num_prompt_tokens", 0)) for seq in seqs),
        total_image_tokens=sum(int(getattr(seq, "image_token_count", 0)) for seq in seqs),
        total_video_tokens=sum(int(getattr(seq, "video_token_count", 0)) for seq in seqs),
        block_size=int(getattr(config, "kvcache_block_size", 0)),
    )


def ensure_compression_off(metadata: CompressionMetadata | None) -> None:
    """Guard attention paths until a real compression strategy is implemented."""

    if metadata is not None and metadata.enabled:
        raise NotImplementedError(
            f"compression_mode={metadata.mode!r} is not implemented"
        )
