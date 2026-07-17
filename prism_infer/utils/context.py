# ═══════════════════════════════════════════════════════════════
# context.py —— 单次执行上下文的 attention bridge
#
# prepare 阶段生成 immutable Context 并装入 DeviceBatch；execute 阶段显式安装，
# model/attention forward 完成后立即 reset。该 process-local bridge 不拥有跨 batch
# 状态，也不能替代 scheduler/request contract。
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True, slots=True)
class Context:
    """一次推理步骤的上下文信息"""
    is_prefill: bool = False                       # True=Prefill, False=Decode
    cu_seqlens_q: torch.Tensor | None = None       # Q 的累积序列长度 (Prefill 用)
    cu_seqlens_k: torch.Tensor | None = None       # K 的累积序列长度 (Prefill 用)
    max_seqlen_q: int = 0                          # 最长 Q 序列 (Flash Attention 需要)
    max_seqlen_k: int = 0                          # 最长 K 序列
    slot_mapping: torch.Tensor | None = None       # 每个 token 在 KV Cache 中的全局槽位
    context_lens: torch.Tensor | None = None       # 每条序列的上下文长度 (Decode 用)
    logical_context_lens: torch.Tensor | None = None  # M-RoPE/审计使用的未压缩长度
    block_tables: torch.Tensor | None = None       # 每条序列的 block 页表 (Decode/PrefixCache 用)
    trace_metadata: Any | None = None              # KV trace 元数据; 默认关闭时为 None
    compression_metadata: Any | None = None        # KV 压缩元数据; P5.0 off baseline 为 no-op
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = ()
    visual_pruning_scorer: Any | None = None        # prefill runtime attention score collector

    def __post_init__(self) -> None:
        if not isinstance(self.is_prefill, bool):
            raise TypeError(
                f"Context.is_prefill must be a boolean, got {self.is_prefill!r}"
            )
        for name in ("max_seqlen_q", "max_seqlen_k"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"Context.{name} must be a non-negative integer, got {value!r}"
                )
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_k",
            "slot_mapping",
            "context_lens",
            "logical_context_lens",
            "block_tables",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"Context.{name} must be a tensor or None")
        if not isinstance(self.visual_pruning_slot_mappings, tuple) or any(
            not isinstance(mapping, torch.Tensor)
            for mapping in self.visual_pruning_slot_mappings
        ):
            raise TypeError(
                "Context.visual_pruning_slot_mappings must be a tuple of tensors"
            )


# ── Process-local forward bridge; ownership remains in DeviceBatch. ──
_CONTEXT = Context()


def get_context() -> Context:
    """attention 层调用: 获取当前步骤的上下文"""
    return _CONTEXT


def install_context(context: Context) -> None:
    """Install an immutable context carried by a prepared device batch."""

    if not isinstance(context, Context):
        raise TypeError(
            f"context must be Context, got {type(context).__name__}"
        )
    global _CONTEXT
    _CONTEXT = context


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    trace_metadata: Any | None = None,
    compression_metadata: Any | None = None,
    visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = (),
    visual_pruning_scorer: Any | None = None,
    logical_context_lens: torch.Tensor | None = None,
) -> None:
    """model_runner 调用: 设置当前步骤的上下文"""
    global _CONTEXT                                # 声明要修改模块级变量
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        logical_context_lens=logical_context_lens,
        block_tables=block_tables,
        trace_metadata=trace_metadata,
        compression_metadata=compression_metadata,
        visual_pruning_slot_mappings=visual_pruning_slot_mappings,
        visual_pruning_scorer=visual_pruning_scorer,
    )


def reset_context() -> None:
    """推理完成后调用: 清除上下文, 释放 tensor 引用"""
    global _CONTEXT
    _CONTEXT = Context()                           # 重置为默认值 (全部 None/0)
