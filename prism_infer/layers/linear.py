# ═══════════════════════════════════════════════════════════════
# linear.py —— TP (Tensor Parallelism) 感知的线性层
#
# 核心思想: 把 nn.Linear 包装成支持多 GPU 切分的版本
#           5 个类形成继承链:
#           LinearBase (基类)
#             ├── ReplicatedLinear      (不切, 每 GPU 有完整权重)
#             ├── ColumnParallelLinear   (按列切, 每 GPU 输出一部分)
#             │     ├── MergedColumnParallelLinear  (gate+up 合并)
#             │     └── QKVParallelLinear           (Q+K+V 合并)
#             └── RowParallelLinear      (按行切, 需要 AllReduce)
#
# C++ 类比: 类似你写的 GEMM 算子, 但每个 GPU 只负责矩阵的一部分
# ═══════════════════════════════════════════════════════════════

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


# ─── 工具函数: 检查整除 ──────────────────────────────────────
def divide(numerator: int, denominator: int) -> int:
    if isinstance(numerator, bool) or not isinstance(numerator, int) or numerator < 0:
        raise ValueError(f"numerator must be a non-negative integer, got {numerator!r}")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError(f"denominator must be a positive integer, got {denominator!r}")
    if numerator % denominator != 0:
        raise ValueError(f"tensor-parallel dimension {numerator} is not divisible by {denominator}")
    return numerator // denominator  # 整数除法


# ─── LinearBase: 所有 TP 线性层的基类 ─────────────────────────
# 负责: 创建权重张量, 绑定 weight_loader
class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度 (已经是 TP 切分后的大小)
        bias: bool = False,  # 是否有偏置
        tp_dim: int | None = None,  # 权重在哪个维度上切分 (0=列, 1=行)
    ):
        super().__init__()
        self.tp_dim = tp_dim  # 切分维度 (列并行=0, 行并行=1)
        self.tp_rank = dist.get_rank()  # 当前 GPU 的编号 (0, 1, ...)
        self.tp_size = dist.get_world_size()  # 总 GPU 数
        # 创建空权重 (output_size × input_size)
        # nn.Parameter: 标记为可训练参数, PyTorch 会自动管理
        # torch.empty: 只分配内存, 不初始化 (之后 weight_loader 会填入真正的值)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 给 weight 绑定加载函数 — loader.py 遍历参数时会调用 param.weight_loader(...)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # register_parameter("bias", None): 告诉 PyTorch "这个参数存在但为 None"
            # 而不是简单的 self.bias = None
            # 这样 F.linear(x, w, self.bias) 可以安全传 None
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # 子类必须实现


# ─── ReplicatedLinear: 不切分, 每个 GPU 有完整权重 ─────────────
# 用在: 不需要 TP 的层 (如 q_norm, k_norm 的权重)
class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)  # tp_dim=None, 不切

    # 权重加载: 直接拷贝, 不切分
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)  # 每个 GPU 拿到完整权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
        # F.linear(x, w, b) = x @ w^T + b
        # 注意 PyTorch 的 Linear 权重是 [out, in], 内部做 x @ w.T


# ─── ColumnParallelLinear: 列并行 ─────────────────────────────
# 每个 GPU 只存 output_size/tp_size 列的权重
# 输出是完整输出的一段, 不需要 AllReduce
# 用在: qkv_proj (每 GPU 负责一部分 head), gate_up_proj
class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,  # 这里传入的是完整的 output_size
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # divide(output_size, tp_size): 把输出维度均分给每个 GPU
        # tp_dim=0: 权重的第 0 维是 output_size, 在这个维度切
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    # 权重加载: 从完整权重中切出自己负责的那部分
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # 每个 GPU 负责的大小
        start_idx = self.tp_rank * shard_size  # 本 GPU 从哪里开始切
        # narrow(dim, start, length): 在 dim 维度上取 [start, start+length) 的切片
        # 类似 C++: loaded_weight[start_idx : start_idx + shard_size] (在某个维度)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)  # 跟普通 Linear 一样


# ─── MergedColumnParallelLinear: 合并多个列并行 ───────────────
# 用在: gate_up_proj (把 gate 和 up 两个独立权重合并成一个大矩阵)
# 继承 ColumnParallelLinear, output_size = sum(output_sizes)
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # [gate_size, up_size] 如 [18944, 18944]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        tp_size = dist.get_world_size()
        if not output_sizes:
            raise ValueError("MergedColumnParallelLinear requires at least one output shard")
        for output_size in output_sizes:
            divide(output_size, tp_size)
        # sum(output_sizes) = 37888, 传给父类做列并行切分
        super().__init__(input_size, sum(output_sizes), bias)

    # 权重加载: loaded_shard_id = 0 表示 gate, 1 表示 up
    # 要把两个独立权重放到合并矩阵的正确位置
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        if (
            isinstance(loaded_shard_id, bool)
            or not isinstance(loaded_shard_id, int)
            or not 0 <= loaded_shard_id < len(self.output_sizes)
        ):
            raise ValueError(f"invalid merged projection shard id: {loaded_shard_id!r}")
        param_data = param.data
        # 算这个 shard 在合并矩阵中的偏移量
        # 例: output_sizes = [18944, 18944], tp_size=2
        #   shard_id=0 (gate): shard_offset = sum([])/2 = 0, shard_size = 18944/2 = 9472
        #   shard_id=1 (up):   shard_offset = sum([18944])/2 = 9472, shard_size = 9472
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 在合并矩阵中定位到这个 shard 的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中切出本 GPU 负责的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # chunk(n, dim): 把张量在 dim 维度等分成 n 份, 取第 tp_rank 份
        param_data.copy_(loaded_weight)


# ─── QKVParallelLinear: Q/K/V 合并的列并行 ──────────────────
# 用在: Attention 的 qkv_proj
# 特殊之处: Q 有 28 个 head, K/V 只有 4 个 (GQA), 切分比例不同
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,  # 输入维度 (3584)
        head_size: int,  # 每个 head 的维度 (128)
        total_num_heads: int,  # Q 的总 head 数 (28)
        total_num_kv_heads: int | None = None,  # KV 的总 head 数 (4)
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads  # None → 用 Q 的头数
        self.head_size = head_size  # 128
        self.num_heads = divide(total_num_heads, tp_size)  # 每 GPU 的 Q head 数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 每 GPU 的 KV head 数
        # 总输出 = (Q_heads + K_heads + V_heads) × head_size
        # 例: (28 + 4 + 4) × 128 = 4608, TP=2 时每 GPU: 2304
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    # 权重加载: loaded_shard_id = "q" / "k" / "v"
    # 要把 HuggingFace 的独立 q/k/v 权重放到合并矩阵的正确偏移
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        if loaded_shard_id not in ("q", "k", "v"):
            raise ValueError(f"invalid QKV projection shard id: {loaded_shard_id!r}")
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size  # 14 × 128 = 1792 (TP=2)
            shard_offset = 0  # Q 在最前面
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size  # 2 × 128 = 256
            shard_offset = self.num_heads * self.head_size  # K 在 Q 后面
        else:  # v
            shard_size = self.num_kv_heads * self.head_size  # 2 × 128 = 256
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            # V 在 K 后面
        # 在合并矩阵中定位
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取本 GPU 的份额
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# ─── RowParallelLinear: 行并行 ────────────────────────────────
# 每个 GPU 只存 input_size/tp_size 行的权重
# 输出是部分和, 需要 AllReduce
# 用在: o_proj, down_proj
class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # divide(input_size, tp_size): 把输入维度均分
        # tp_dim=1: 权重的第 1 维是 input_size, 在这个维度切
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # 每 GPU 的行数
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 只在 rank 0 加, 避免 AllReduce 后 bias 被加了 tp_size 次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # AllReduce: 所有 GPU 的 y 相加
            # 这一行就是 TP 的核心通信!
        return y
