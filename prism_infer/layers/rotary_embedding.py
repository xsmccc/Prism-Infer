from functools import lru_cache
import torch
from torch import nn


# ============================================================
# apply_rotary_emb — 对 q/k 施加旋转位置编码 (RoPE)
# ============================================================
# RoPE 的核心公式: 将向量分成两半, 做旋转变换
#   [y1, y2] = [x1*cos - x2*sin, x2*cos + x1*sin]
# 数学上等价于二维旋转矩阵:
#   [cos θ, -sin θ] [x1]   [x1*cos - x2*sin]
#   [sin θ,  cos θ] [x2] = [x1*sin + x2*cos]
def apply_rotary_emb(
    x: torch.Tensor,  # [N, num_heads, head_dim]
    cos: torch.Tensor,  # [N, 1, head_dim//2] 或广播兼容
    sin: torch.Tensor,
) -> torch.Tensor:
    # chunk(2, dim=-1): 沿最后一维切成两半
    # head_dim=128 → x1=[..., :64], x2=[..., 64:]
    # .float(): 转 float32 提高精度 (原始可能是 bfloat16)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 旋转公式
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼回原来的维度, 转回原始精度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# ============================================================
# RotaryEmbedding — 预计算并缓存 cos/sin 表
# ============================================================
# 初始化时预计算所有位置的 cos/sin 值 (查表法)
# forward 时只需按 position 索引查表
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,  # 每个 head 的维度 (如 128)
        rotary_dim: int,  # 旋转维度 (= head_size)
        max_position_embeddings: int,  # 最大位置 (如 40960)
        base: float,  # RoPE 基数 (如 1000000.0)
    ) -> None:
        super().__init__()
        if isinstance(head_size, bool) or not isinstance(head_size, int) or head_size <= 0:
            raise ValueError(f"head_size must be a positive integer, got {head_size!r}")
        if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int) or rotary_dim <= 0:
            raise ValueError(f"rotary_dim must be a positive integer, got {rotary_dim!r}")
        if rotary_dim != head_size:
            raise ValueError(
                "partial rotary dimensions are unsupported: "
                f"rotary_dim={rotary_dim}, head_size={head_size}"
            )
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if base <= 0:
            raise ValueError(f"RoPE base must be positive, got {base}")
        self.head_size = head_size
        # 计算频率: 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        # arange(0, 128, 2) = [0, 2, 4, ..., 126] → 64 个值
        # / rotary_dim → [0/128, 2/128, ..., 126/128]
        # base** → 指数衰减的频率
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 位置序列: [0, 1, 2, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积: [max_pos] × [rotary_dim/2] → [max_pos, rotary_dim/2]
        # 每个位置 p 和每个频率 f 的乘积 p*f
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()  # [max_pos, 64]
        sin = freqs.sin()  # [max_pos, 64]
        # 拼接 cos 和 sin → [max_pos, 128]
        # unsqueeze_(1): [max_pos, 128] → [max_pos, 1, 128]
        #   中间的 1 是 num_heads 维, 广播用
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer: 注册为模块属性, 随模型移动到 GPU, 但不是参数 (不训练)
        # persistent=False: 不保存到 state_dict (因为可以重新计算)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile  # torch.compile 优化: JIT 编译, 融合小算子
    def forward(
        self,
        positions: torch.Tensor,  # [N] 每个 token 的位置编号
        query: torch.Tensor,  # [N, num_heads, head_dim]
        key: torch.Tensor,  # [N, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按位置索引查表: [N] → [N, 1, 128]
        cos_sin = self.cos_sin_cache[positions]
        # 切分 cos 和 sin: [N, 1, 128] → [N, 1, 64] + [N, 1, 64]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 对 q 和 k 分别施加旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# ============================================================
# get_rope — 工厂函数, 带缓存 (单例模式)
# ============================================================
# @lru_cache(1): 只缓存一个实例, 相同参数返回同一个对象
# 所有 Transformer 层共享同一个 RoPE (因为参数一样)
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is not None:
        raise NotImplementedError("legacy RotaryEmbedding does not support rope_scaling")
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
