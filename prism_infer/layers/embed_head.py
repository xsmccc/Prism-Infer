import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from prism_infer.utils.context import get_context


# ============================================================
# VocabParallelEmbedding — 词表并行的 Embedding 层
# ============================================================
# 将词表按 TP 切分，每个 GPU 只存一部分词的嵌入向量
# 例如 vocab_size=152064, tp_size=2:
#   GPU 0 存 token 0~76031 的嵌入
#   GPU 1 存 token 76032~152063 的嵌入
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # 总词表大小 (如 152064)
        embedding_dim: int,  # 嵌入维度 (如 3584)
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()  # 当前 GPU 编号
        self.tp_size = dist.get_world_size()  # GPU 总数
        if (
            isinstance(num_embeddings, bool)
            or not isinstance(num_embeddings, int)
            or num_embeddings <= 0
        ):
            raise ValueError(f"num_embeddings must be a positive integer, got {num_embeddings!r}")
        if num_embeddings % self.tp_size != 0:
            raise ValueError(
                "vocabulary size must be divisible by tensor parallel size: "
                f"num_embeddings={num_embeddings}, tp_size={self.tp_size}"
            )
        self.num_embeddings = num_embeddings
        # 每个 GPU 负责的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本 GPU 负责的词表范围 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 只分配本 GPU 负责的那部分嵌入矩阵
        # 形状: [num_embeddings_per_partition, embedding_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 挂载 weight_loader，用于从完整权重切出本 GPU 的部分
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整词表权重中切出本 GPU 的部分"""
        param_data = param.data
        shard_size = param_data.size(0)  # = num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        # narrow(0, start, len): 在 dim=0 上从 start 开始取 len 行
        # C++ 类比: loaded_weight[start_idx : start_idx + shard_size]
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        x: token id 序列, 形状 [num_tokens]

        TP 时的处理:
        1. mask 标记哪些 token 属于本 GPU    (在范围内=True)
        2. 偏移 id 到本地索引               (减去 start_idx)
        3. 查表得到嵌入                     (F.embedding)
        4. 不属于本 GPU 的位置置零           (mask * y)
        5. AllReduce 求和                   (零+嵌入=嵌入，完整结果)
        """
        if self.tp_size > 1:
            # 哪些 token id 落在本 GPU 负责的范围内
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 不在范围内的 id 被 mask 乘以 0 → 变成 id=0 (安全的合法索引)
            # 在范围内的 id 减去偏移 → 变成本地索引
            x = mask * (x - self.vocab_start_idx)
        # 查嵌入表 (本地的小表)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 不属于本 GPU 的位置嵌入被置零
            # mask 形状 [N], y 形状 [N, D], unsqueeze(1) 广播
            y = mask.unsqueeze(1) * y
            # AllReduce: 所有 GPU 的 y 相加
            # 每个位置只有一个 GPU 有非零值 → 加起来就是完整结果
            dist.all_reduce(y)
        return y


# ============================================================
# ParallelLMHead — 并行的语言模型输出头
# ============================================================
# 继承 VocabParallelEmbedding，复用其权重和 weight_loader
# 但 forward 完全不同: 做线性映射而不是查表
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,  # vocab_size
        embedding_dim: int,  # hidden_size
        bias: bool = False,
    ):
        if bias:
            raise ValueError("ParallelLMHead does not support bias")
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        x: 隐藏状态, 形状 [num_tokens, hidden_size]
        输出: logits, 形状 [num_seqs, vocab_size] (仅 rank=0)

        步骤:
        1. Prefill 时只取每个序列的最后一个 token
        2. 用 embedding 权重做线性映射 → logits (本地词表部分)
        3. Gather 到 rank 0 拼接出完整 logits
        """
        context = get_context()
        if context.is_prefill:
            # Prefill 阶段: 只需要每个序列最后一个 token 的 logits
            # cu_seqlens_q[1:] 是每个序列的结束位置, 减1得到最后一个token的索引
            # 例如 cu_seqlens_q = [0, 5, 12] → last_indices = [4, 11]
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()  # [num_seqs, hidden_size]
        # F.linear(x, W) = x @ W.T
        # W 形状 [vocab_per_partition, hidden_size]
        # 输出: [num_seqs, vocab_per_partition]  ← 只有本 GPU 负责的词表部分
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # Gather: 所有 GPU 把各自的 logits 片段发给 rank 0
            # rank 0 收集到 all_logits 列表, 其他 rank 传 None
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            # rank 0 沿最后一维拼接 → [num_seqs, vocab_size] 完整 logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
