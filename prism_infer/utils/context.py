"""Task-local attention context shared by preparation and model execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Context:
    """Immutable attention metadata for one model execution step."""

    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    logical_context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    decode_max_context_len: torch.Tensor | None = None
    packed_decode_metadata: torch.Tensor | None = None
    paged_decode_block_n: int = 32
    paged_decode_num_splits: int = 1
    trace_metadata: Any | None = None
    compression_metadata: Any | None = None
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = ()
    visual_pruning_scorer: Any | None = None

    def __post_init__(self) -> None:
        for name in ("max_seqlen_q", "max_seqlen_k"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"Context.{name} must be non-negative, got {value}")
        if self.paged_decode_block_n <= 0 or self.paged_decode_block_n & (
            self.paged_decode_block_n - 1
        ):
            raise ValueError("Context.paged_decode_block_n must be a positive power of two")
        if self.decode_max_context_len is not None:
            if self.decode_max_context_len.numel() != 1:
                raise ValueError("Context.decode_max_context_len must contain one scalar")
            if self.decode_max_context_len.dtype != torch.int32:
                raise ValueError("Context.decode_max_context_len must use torch.int32")


# Task-local forward bridge; ownership remains in DeviceBatch.  ContextVar
# isolates concurrent threads/async tasks and supports exact nested restoration.
_CONTEXT: ContextVar[Context] = ContextVar(
    "prism_infer_execution_context",
    default=Context(),  # noqa: B039 - Context is frozen and has immutable defaults.
)


def get_context() -> Context:
    """Return the context installed for the current task."""

    return _CONTEXT.get()


def install_context(context: Context) -> None:
    """Install an immutable context carried by a prepared device batch."""

    _CONTEXT.set(context)


@contextmanager
def use_context(context: Context) -> Iterator[Context]:
    """Install one execution context and restore the exact previous value."""

    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    decode_max_context_len: torch.Tensor | None = None,
    packed_decode_metadata: torch.Tensor | None = None,
    paged_decode_block_n: int = 32,
    paged_decode_num_splits: int = 1,
    trace_metadata: Any | None = None,
    compression_metadata: Any | None = None,
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = (),
    visual_pruning_scorer: Any | None = None,
    logical_context_lens: torch.Tensor | None = None,
) -> None:
    """Build and install an execution context for compatibility callers."""

    install_context(
        Context(
            is_prefill=is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            logical_context_lens=logical_context_lens,
            block_tables=block_tables,
            decode_max_context_len=decode_max_context_len,
            packed_decode_metadata=packed_decode_metadata,
            paged_decode_block_n=paged_decode_block_n,
            paged_decode_num_splits=paged_decode_num_splits,
            trace_metadata=trace_metadata,
            compression_metadata=compression_metadata,
            visual_pruning_slot_mappings=visual_pruning_slot_mappings,
            visual_pruning_scorer=visual_pruning_scorer,
        )
    )


def reset_context() -> None:
    """Release tensor references held by the current task context."""

    install_context(Context())
