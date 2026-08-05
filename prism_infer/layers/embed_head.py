"""Tensor-parallel token embeddings and language-model output heads."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from prism_infer.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """Shard vocabulary rows across tensor-parallel ranks."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
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
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Copy this rank's vocabulary shard from a full checkpoint tensor."""

        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """Embed local vocabulary rows and all-reduce the complete result."""

        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """Compute vocabulary-parallel logits or exact distributed greedy tokens."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        select_prefill_tokens: bool = True,
    ) -> None:
        if bias:
            raise ValueError("ParallelLMHead does not support bias")
        if not isinstance(select_prefill_tokens, bool):
            raise TypeError("select_prefill_tokens must be a boolean")
        super().__init__(num_embeddings, embedding_dim)
        self.select_prefill_tokens = select_prefill_tokens

    def _select_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Select one prefill state per sequence when the caller has not."""

        context = get_context()
        if context.is_prefill and self.select_prefill_tokens:
            last_indices = context.cu_seqlens_q[1:] - 1
            return x[last_indices].contiguous()
        return x

    def forward(self, x: torch.Tensor):
        """Return full logits on rank zero and ``None`` on other TP ranks."""

        x = self._select_hidden_states(x)
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

    def compute_greedy_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Select global greedy tokens while communicating two scalars per row."""

        x = self._select_hidden_states(x)
        local_logits = F.linear(x, self.weight)
        local_values, local_offsets = local_logits.max(dim=-1)
        local_ids = local_offsets + self.vocab_start_idx
        local_candidates = torch.stack(
            (local_values.float(), local_ids.float()),
            dim=0,
        )
        gathered_candidates = [torch.empty_like(local_candidates) for _ in range(self.tp_size)]
        dist.all_gather(gathered_candidates, local_candidates)
        candidate_values = torch.stack(
            [candidate[0] for candidate in gathered_candidates],
            dim=0,
        )
        candidate_ids = torch.stack(
            [candidate[1] for candidate in gathered_candidates],
            dim=0,
        )
        winning_ranks = candidate_values.argmax(dim=0, keepdim=True)
        return candidate_ids.gather(0, winning_ranks).squeeze(0).long()
