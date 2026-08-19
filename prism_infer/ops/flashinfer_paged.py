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


class FlashInferPagedDecode:
    """One persistent wrapper per layer for batch paged decode.

    ``use_cuda_graph`` 模式下元数据走静态 buffer: stage 由调用方在 graph
    外执行, 捕获期只 run()。
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
        use_cuda_graph: bool = False,
        max_batch: int = 0,
        max_blocks: int = 0,
    ) -> None:
        if not HAS_FLASHINFER:
            raise RuntimeError("flashinfer is not available in this environment")
        self.block_size = block_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.use_cuda_graph = use_cuda_graph
        self.max_batch = max_batch
        self.max_blocks = max_blocks
        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        self._planned_key: tuple[int, int] | None = None
        if max_batch <= 0 or max_blocks <= 0:
            raise ValueError(
                "FlashInferPagedDecode requires max_batch and max_blocks"
            )
        self.indptr_buf = torch.zeros(max_batch + 1, dtype=torch.int32, device="cuda")
        self.indices_buf = torch.full(
            (max_batch * max_blocks,), -1, dtype=torch.int32, device="cuda"
        )
        self.last_page_len_buf = torch.zeros(max_batch, dtype=torch.int32, device="cuda")
        # 0.6.13 父类签名 (vLLM 同款调用): (workspace, kv_layout,
        # use_cuda_graph, paged_kv_*_buffer, use_tensor_cores)
        self._wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.indptr_buf,
            paged_kv_indices_buffer=self.indices_buf,
            paged_kv_last_page_len_buffer=self.last_page_len_buf,
            use_tensor_cores=True,
        )

    def plan(
        self,
        *,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
    ) -> None:
        self._wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def stage_plan_inputs(
        self,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> None:
        """Fill static buffers with current metadata, then plan (graph 外调用)。"""

        batch = int(context_lens.numel())
        if batch > self.max_batch:
            raise RuntimeError(
                f"flashinfer decode batch {batch} exceeds static capacity {self.max_batch}"
            )
        width = int(block_tables.shape[1])
        if width > self.max_blocks:
            raise RuntimeError(
                f"flashinfer decode block width {width} exceeds static capacity {self.max_blocks}"
            )
        pages_per_seq = (context_lens + self.block_size - 1) // self.block_size
        self.indptr_buf[0] = 0
        torch.cumsum(
            pages_per_seq.to(torch.int32),
            dim=0,
            out=self.indptr_buf[1 : batch + 1],
        )
        self.indices_buf.fill_(-1)
        self.indices_buf[: batch * width] = block_tables.flatten()[: batch * width]
        self.last_page_len_buf[:batch] = torch.where(
            context_lens % self.block_size == 0,
            torch.full_like(context_lens, self.block_size),
            context_lens % self.block_size,
        ).to(torch.int32)
        key = (batch, width)
        if key != self._planned_key:
            # 形状不变时跳过 replan (重放期 staging 是热路径)
            self.plan(
                indptr=self.indptr_buf,
                indices=self.indices_buf,
                last_page_len=self.last_page_len_buf,
            )
            self._planned_key = key

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """q: [batch, num_qo_heads, head_dim]; 输出同形。"""

        return self._wrapper.run(q, (k_cache, v_cache))


def build_flashinfer_decode_plan_inputs(
    *,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert decode scheduling metadata to flashinfer decode-plan inputs.

    context_lens: [batch] int32 GPU; block_tables: [batch, max_blocks] int32。
    返回 (indptr=页数累加, indices=扁平页表, last_page_len)。
    """

    batch = int(context_lens.numel())
    pages_per_seq = (context_lens + block_size - 1) // block_size
    indptr = torch.empty(batch + 1, dtype=torch.int32, device=context_lens.device)
    indptr[0] = 0
    torch.cumsum(pages_per_seq.to(torch.int32), dim=0, out=indptr[1:])
    # 全宽展平: 有效页数由 indptr 界定, 免逐步 GPU->CPU 同步
    # (CUDA Graph 捕获期不允许 .item())
    indices = block_tables.flatten().to(torch.int32)
    last_page_len = torch.where(
        context_lens % block_size == 0,
        torch.full_like(context_lens, block_size),
        context_lens % block_size,
    ).to(torch.int32)
    return indptr, indices, last_page_len


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

    # paged_kv_indptr 是每请求页数的累加 (不是 token 数)
    pages_per_seq = (k_lens + block_size - 1) // block_size
    kv_indptr = torch.empty(num_seqs + 1, dtype=torch.int32, device=block_tables.device)
    kv_indptr[0] = 0
    torch.cumsum(pages_per_seq.to(torch.int32), dim=0, out=kv_indptr[1:])

    # 一次小同步取整批长度 (与 cu_seqlens 元素数同阶, 不逐 seq sync)
    pages_cpu = pages_per_seq.cpu().tolist()
    k_lens_cpu = k_lens.cpu().tolist()
    indices_list: list[torch.Tensor] = []
    last_page_lens: list[int] = []
    for seq in range(num_seqs):
        k_len = k_lens_cpu[seq]
        pages = pages_cpu[seq]
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
