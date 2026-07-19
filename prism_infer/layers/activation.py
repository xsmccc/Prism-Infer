import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# SiluAndMul — SwiGLU 激活函数的核心操作
# ============================================================
# SwiGLU = SiLU(gate) * up
# 输入是 gate 和 up 拼在一起的张量 (来自 MergedColumnParallelLinear)
# 切成两半, 前半做 SiLU, 后半直接乘 → 实现 gate 机制
#
# SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile  # 融合 chunk + silu + mul 为一个 kernel
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: [N, 2 * intermediate_size]  (gate 和 up 拼在一起)
        # chunk(2, -1): 沿最后一维切成两等份
        #   x → [N, intermediate_size]  (gate 部分)
        #   y → [N, intermediate_size]  (up 部分)
        x, y = x.chunk(2, -1)
        # SiLU(gate) * up
        return F.silu(x) * y
