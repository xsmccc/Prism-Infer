# ═══════════════════════════════════════════════════════════════
# context.py —— 全局上下文 (model_runner 和 attention 层之间的"传话筒")
#
# 职责: 存储一次推理步骤中 attention 层需要的元数据
#       model_runner 在 prepare_prefill/decode 里 set_context()
#       attention 层在 forward() 里 get_context() 读取
#
# 为什么要用全局变量?
#   因为 model.forward(input_ids, positions) 的接口只接受这两个参数,
#   而 attention 层需要 slot_mapping, block_tables, cu_seqlens 等信息。
#   不想改模型接口 → 用全局变量"偷偷"传递。
#
# C++ 类比: 类似 thread_local 全局状态, 或者一个 Context 单例
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Context:
    """一次推理步骤的上下文信息"""
    is_prefill: bool = False                       # True=Prefill, False=Decode
    cu_seqlens_q: torch.Tensor | None = None       # Q 的累积序列长度 (Prefill 用)
    cu_seqlens_k: torch.Tensor | None = None       # K 的累积序列长度 (Prefill 用)
    max_seqlen_q: int = 0                          # 最长 Q 序列 (Flash Attention 需要)
    max_seqlen_k: int = 0                          # 最长 K 序列
    slot_mapping: torch.Tensor | None = None       # 每个 token 在 KV Cache 中的全局槽位
    context_lens: torch.Tensor | None = None       # 每条序列的上下文长度 (Decode 用)
    block_tables: torch.Tensor | None = None       # 每条序列的 block 页表 (Decode/PrefixCache 用)
    trace_metadata: Any | None = None              # KV trace 元数据; 默认关闭时为 None
    compression_metadata: Any | None = None        # KV 压缩元数据; P5.0 off baseline 为 no-op

# ── 模块级全局变量: 单例 Context ──
_CONTEXT = Context()
# 只有一个实例, 整个进程共享 (单线程推理, 不需要锁)

def get_context():
    """attention 层调用: 获取当前步骤的上下文"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, trace_metadata=None, compression_metadata=None):
    """model_runner 调用: 设置当前步骤的上下文"""
    global _CONTEXT                                # 声明要修改模块级变量
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, trace_metadata, compression_metadata)
    # 每次创建新的 Context 对象 (不是修改旧的)
    # dataclass 的 __init__ 按字段顺序接收参数

def reset_context():
    """推理完成后调用: 清除上下文, 释放 tensor 引用"""
    global _CONTEXT
    _CONTEXT = Context()                           # 重置为默认值 (全部 None/0)
