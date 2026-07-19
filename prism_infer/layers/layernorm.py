import torch
from torch import nn


# ============================================================
# RMSNorm — Root Mean Square Layer Normalization
# ============================================================
# LayerNorm 的简化版: 去掉了均值中心化, 只做方差归一化
# 公式: y = x / sqrt(mean(x^2) + eps) * weight
# 比标准 LayerNorm 少算一次 mean (减均值), 更快且效果相近
# Qwen3, LLaMA 等现代 LLM 都用 RMSNorm
class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,  # 隐藏维度 (如 3584)
        eps: float = 1e-6,  # 防除零的小常数
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数, 初始化为全1, 形状 [hidden_size]
        # 每个维度一个缩放因子
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile  # JIT 编译, 融合多个小算子为一个 kernel
    def rms_forward(
        self,
        x: torch.Tensor,  # [N, hidden_size]
    ) -> torch.Tensor:
        orig_dtype = x.dtype  # 记录原始精度 (如 bfloat16)
        x = x.float()  # 转 float32 提高精度
        # x.pow(2): 每个元素平方
        # .mean(dim=-1, keepdim=True): 沿最后一维求均值, 保持维度
        #   [N, 3584] → [N, 1]  (keepdim=True 保留该维度为1)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1/sqrt, 即 1/sqrt(mean(x^2) + eps)
        # mul_: 原地乘法 (in-place)
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原始精度, 再乘以可学习权重 weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,  # 当前层的输出
        residual: torch.Tensor,  # 残差 (上一层传下来的)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """先做残差加法, 再做 RMSNorm — 融合版"""
        orig_dtype = x.dtype
        # 残差加法: x = x + residual (在 float32 精度下)
        # add_: 原地加法
        x = x.float().add_(residual.float())
        # 保存加完后的结果作为新的 residual (传给下一层)
        residual = x.to(orig_dtype)
        # 以下和 rms_forward 一样
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual  # 返回归一化结果 + 新的 residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 根据是否有 residual 选择对应的方法
        if residual is None:
            return self.rms_forward(x)  # 第一层, 无残差
        else:
            return self.add_rms_forward(x, residual)  # 其他层, 有残差
