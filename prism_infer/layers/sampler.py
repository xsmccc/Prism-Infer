import torch
from torch import nn


SAMPLING_NUMERICAL_EPSILON = 1.0e-10
SAMPLING_MODES = frozenset({"greedy", "random", "mixed"})


# ============================================================
# Sampler — 从 logits 采样下一个 token
# ============================================================
# 输入: logits [num_seqs, vocab_size] + temperatures [num_seqs]
# 输出: sample_tokens [num_seqs] (每个序列采样一个 token id)
#
# 采样策略: Gumbel-Max trick (等价于 multinomial 采样, 但更适合 GPU 并行)
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        *,
        sampling_mode: str | None = None,
    ):
        """根据每条请求的 temperature 采样下一个 token。

        logits: [num_seqs, vocab_size]
        temperatures: [num_seqs]，temperature <= 1e-10 时走 greedy argmax。
        """

        if sampling_mode is not None and sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"unsupported sampling mode: {sampling_mode!r}")
        if sampling_mode == "greedy":
            return logits.argmax(dim=-1)
        if temperatures is None:
            raise ValueError("temperatures are required for non-greedy sampling")
        if sampling_mode == "random":
            return self._sample_random(logits, temperatures)

        greedy_mask = temperatures <= SAMPLING_NUMERICAL_EPSILON
        if sampling_mode is None and bool(greedy_mask.all().item()):
            return logits.argmax(dim=-1)
        if sampling_mode is None and bool((~greedy_mask).all().item()):
            return self._sample_random(logits, temperatures)

        sample_tokens = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        sample_tokens[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        sample_tokens[~greedy_mask] = self._sample_random(
            logits[~greedy_mask],
            temperatures[~greedy_mask],
        )
        return sample_tokens

    @torch.compile  # 融合随机采样流程为一个 kernel
    def _sample_random(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. Temperature scaling: logits / temperature
        #    temperature 越高 → 分布越平 (更随机)
        #    temperature 越低 → 分布越尖 (越确定)
        #    temperature = 0 已在 forward 中走 greedy
        #    unsqueeze(dim=1): [num_seqs] → [num_seqs, 1] 广播到 [num_seqs, vocab]
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        # 2. Softmax: logits → 概率分布
        probs = torch.softmax(logits, dim=-1)
        # 3. Gumbel-Max trick 采样:
        #    等价于 torch.multinomial(probs, 1) 但更快
        #    原理: probs / Exp(1) 的 argmax 等价于按 probs 概率采样
        #    - exponential_(1): 原地生成指数分布随机数 (Exp(1))
        #    - clamp_min_(1e-10): 防止除零
        #    - probs / exp_random: 概率高的位置更可能是最大值
        #    - argmax: 取最大值的索引 = 采样结果
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(SAMPLING_NUMERICAL_EPSILON)
        ).argmax(dim=-1)
        return sample_tokens
