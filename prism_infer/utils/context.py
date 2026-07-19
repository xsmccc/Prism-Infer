# ═══════════════════════════════════════════════════════════════
# context.py —— 单次执行上下文的 attention bridge
#
# prepare 阶段生成 immutable Context 并装入 DeviceBatch；execute 阶段在 task-local
# scope 内安装，forward 结束后恢复此前值。该 bridge 不拥有跨 batch 状态，也不能
# 替代 scheduler/request contract。
# ═══════════════════════════════════════════════════════════════

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator
import torch


@dataclass(frozen=True, slots=True)
class Context:
    """一次推理步骤的上下文信息"""

    is_prefill: bool = False  # True=Prefill, False=Decode
    cu_seqlens_q: torch.Tensor | None = None  # Q 的累积序列长度 (Prefill 用)
    cu_seqlens_k: torch.Tensor | None = None  # K 的累积序列长度 (Prefill 用)
    max_seqlen_q: int = 0  # 最长 Q 序列 (Flash Attention 需要)
    max_seqlen_k: int = 0  # 最长 K 序列
    slot_mapping: torch.Tensor | None = None  # 每个 token 在 KV Cache 中的全局槽位
    context_lens: torch.Tensor | None = None  # 每条序列的上下文长度 (Decode 用)
    logical_context_lens: torch.Tensor | None = None  # M-RoPE/审计使用的未压缩长度
    block_tables: torch.Tensor | None = None  # 每条序列的 block 页表 (Decode/PrefixCache 用)
    decode_max_context_len: torch.Tensor | None = None  # decode batch 动态物理 K 上界
    trace_metadata: Any | None = None  # KV trace 元数据; 默认关闭时为 None
    compression_metadata: Any | None = None  # KV 压缩元数据; P5.0 off baseline 为 no-op
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = ()
    visual_pruning_scorer: Any | None = None  # prefill runtime attention score collector

    def __post_init__(self) -> None:
        if not isinstance(self.is_prefill, bool):
            raise TypeError(f"Context.is_prefill must be a boolean, got {self.is_prefill!r}")
        for name in ("max_seqlen_q", "max_seqlen_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Context.{name} must be a non-negative integer, got {value!r}")
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_k",
            "slot_mapping",
            "context_lens",
            "logical_context_lens",
            "block_tables",
            "decode_max_context_len",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"Context.{name} must be a tensor or None")
        if self.decode_max_context_len is not None:
            if self.decode_max_context_len.numel() != 1:
                raise ValueError("Context.decode_max_context_len must contain one scalar")
            if self.decode_max_context_len.dtype != torch.int32:
                raise ValueError("Context.decode_max_context_len must use torch.int32")
        if not isinstance(self.visual_pruning_slot_mappings, tuple) or any(
            not isinstance(mapping, torch.Tensor) for mapping in self.visual_pruning_slot_mappings
        ):
            raise TypeError("Context.visual_pruning_slot_mappings must be a tuple of tensors")


# Task-local forward bridge; ownership remains in DeviceBatch.  ContextVar
# isolates concurrent threads/async tasks and supports exact nested restoration.
_CONTEXT: ContextVar[Context] = ContextVar(
    "prism_infer_execution_context",
    default=Context(),
)


def get_context() -> Context:
    """attention 层调用: 获取当前步骤的上下文"""
    return _CONTEXT.get()


def install_context(context: Context) -> None:
    """Install an immutable context carried by a prepared device batch."""

    if not isinstance(context, Context):
        raise TypeError(f"context must be Context, got {type(context).__name__}")
    _CONTEXT.set(context)


@contextmanager
def use_context(context: Context) -> Iterator[Context]:
    """Install one execution context and restore the exact previous value."""

    if not isinstance(context, Context):
        raise TypeError(f"context must be Context, got {type(context).__name__}")
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
    trace_metadata: Any | None = None,
    compression_metadata: Any | None = None,
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = (),
    visual_pruning_scorer: Any | None = None,
    logical_context_lens: torch.Tensor | None = None,
) -> None:
    """model_runner 调用: 设置当前步骤的上下文"""
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
            trace_metadata=trace_metadata,
            compression_metadata=compression_metadata,
            visual_pruning_slot_mappings=visual_pruning_slot_mappings,
            visual_pruning_scorer=visual_pruning_scorer,
        )
    )


def reset_context() -> None:
    """推理完成后调用: 清除上下文, 释放 tensor 引用"""
    install_context(Context())
