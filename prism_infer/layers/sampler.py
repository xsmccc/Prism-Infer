"""Greedy and temperature-based token sampling."""

import torch
from torch import nn

SAMPLING_NUMERICAL_EPSILON = 1.0e-10
SAMPLING_MODES = frozenset({"greedy", "random", "mixed"})


class Sampler(nn.Module):
    """Choose greedy tokens or sample with the exponential-race method."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        *,
        sampling_mode: str | None = None,
    ):
        """Select one token per row according to the requested sampling mode."""

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

    @torch.compile
    def _sample_random(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(SAMPLING_NUMERICAL_EPSILON)
        ).argmax(dim=-1)
        return sample_tokens
