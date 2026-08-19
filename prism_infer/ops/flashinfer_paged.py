"""FlashInfer paged-attention adapter for the Prism paged KV cache.

步骤 1: bf16 KV (dense / compression off) 直接消费引擎的 paged cache:
k_cache 每层布局 [blocks, block_size, kv_heads, head_dim] (NHD) 与
flashinfer 期望一致, 零数据搬移。fp8 per-token-per-head scale 路径留待
步骤 2 (经 *args 的 scale tensor, 形状需在 GPU 上实测)。

导入是可选依赖: flashinfer 未安装时 HAS_FLASHINFER=False, 调用方回退。
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - 导入路径依赖环境
    import flashinfer

    HAS_FLASHINFER = True
except Exception:  # noqa: BLE001
    flashinfer = None
    HAS_FLASHINFER = False


class FlashInferPagedPrefill:
    """One persistent wrapper per (layer, shape) for batch paged prefill.

    plan() 每批调用 (CPU 侧开销与 batch 同阶); run() 一次覆盖整批 suffix
    prefill, 替代现有的逐 seq Python 循环 + 全上下文 gather + masked SDPA。
    """

    def __init__(
        self,
        *,
        block_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        workspace_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not HAS_FLASHINFER:
            raise RuntimeError("flashinfer is not available in this environment")
        self.block_size = block_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        self._wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
        )

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        """Plan one batch; inputs are int32 CUDA tensors (flashinfer 契约)."""

        self._wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.block_size,
            causal=True,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """q: [total_q_tokens, num_qo_heads, head_dim]; 输出同形。"""

        return self._wrapper.run(q, (k_cache, v_cache))


def build_flashinfer_plan_inputs(
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert engine scheduling metadata to flashinfer paged-plan inputs.

    block_tables: [batch, max_blocks] int32 GPU; 返回 (qo_indptr, kv_indptr,
    paged_kv_indices, last_page_len) 全部为 int32 CUDA。
    """

    num_seqs = int(cu_seqlens_q.numel()) - 1
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    kv_indptr = torch.empty(num_seqs + 1, dtype=torch.int32, device=block_tables.device)
    kv_indptr[0] = 0
    torch.cumsum(k_lens.to(torch.int32), dim=0, out=kv_indptr[1:])

    num_pages = (kv_indptr[-1].item() + block_size - 1) // block_size
    max_blocks = int(block_tables.shape[1])
    flat = block_tables.flatten()
    valid = flat >= 0
    # 每 seq 页数 = ceil(k_len / block_size); 逐 seq 收集非负页 id
    indices_list: list[torch.Tensor] = []
    last_page_lens: list[int] = []
    for seq in range(num_seqs):
        k_len = int(k_lens[seq].item())
        pages = (k_len + block_size - 1) // block_size
        seq_blocks = block_tables[seq, :pages]
        if bool((seq_blocks < 0).any()):
            raise RuntimeError("paged prefill block table has holes")
        indices_list.append(seq_blocks)
        last_page_lens.append(k_len % block_size or block_size)
    paged_kv_indices = torch.cat(indices_list).to(torch.int32)
    last_page_len = torch.tensor(
        last_page_lens, dtype=torch.int32, device=block_tables.device
    )
    return cu_seqlens_q.to(torch.int32), kv_indptr, paged_kv_indices, last_page_len
