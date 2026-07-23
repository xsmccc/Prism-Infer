"""Attention semantic routing across optimized and reference backends."""

import torch
import torch.nn.functional as F
from torch import nn

from prism_infer.engine.compression import (
    CompressionMetadata,
    ensure_supported_compression_metadata,
)
from prism_infer.observability import (
    is_trace_enabled,
    profile_region,
    record_attention_layer,
)
from prism_infer.ops.kv_cache_store import (
    FP8_CACHE_DTYPES,
    HAS_TRITON,
    _store_kvcache_eager,
    is_fp8_cache_tensor as _is_fp8_cache_tensor,
    store_kvcache,
)
from prism_infer.ops.paged_attention_reference import (
    paged_decode_attention_reference,
    paged_prefill_attention_reference,
    visual_pruned_decode_attention_reference,
)
from prism_infer.ops.paged_decode import (
    HAS_TRITON as HAS_PAGED_DECODE_TRITON,
    paged_decode_attention,
)
from prism_infer.utils.context import Context, get_context


# FlashAttention is optional and selected only for compatible CUDA paths.
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as vllm_paged_flash_attn,
    )

    HAS_VLLM_PAGED_FLASH_ATTN = True
except (ImportError, RuntimeError):
    HAS_VLLM_PAGED_FLASH_ATTN = False


# Compatibility exports: existing callers historically imported the storage
# implementation from this module. New code should use ops.kv_cache_store.
__all__ = [
    "Attention",
    "FP8_CACHE_DTYPES",
    "HAS_FLASH_ATTN",
    "HAS_TRITON",
    "_is_fp8_cache_tensor",
    "_store_kvcache_eager",
    "store_kvcache",
]


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Dynamically scaled FP8 keeps one independent FP32 scale for each
        # (physical token, KV head), separately for K and V. ModelRunner binds
        # these views once after allocation so CUDA Graph sees stable addresses.
        self.k_scale_cache: torch.Tensor | None = None
        self.v_scale_cache: torch.Tensor | None = None
        self.layer_idx: int | None = None
        self._paged_flash_cu_seqlens_q: dict[int, torch.Tensor] = {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_already_stored: bool = False,
    ) -> torch.Tensor:
        """Route one attention call while keeping storage and semantics explicit."""

        context = get_context()
        ensure_supported_compression_metadata(context.compression_metadata)
        payload_cache_bound = self._validate_cache_bindings(context.compression_metadata)
        if kv_already_stored:
            if context.is_prefill or not payload_cache_bound:
                raise RuntimeError("pre-stored KV is valid only for bound decode caches")
            if self.k_scale_cache is not None or self.v_scale_cache is not None:
                raise RuntimeError("pre-stored KV does not support scaled caches")
        elif payload_cache_bound:
            self._store_current_kv(k, v, context)
        output = (
            self._forward_prefill(q, k, v, context)
            if context.is_prefill
            else self._forward_decode(q, k, v, context)
        )
        self._record_observability(q, k, v, output, context)
        return output

    def _validate_cache_bindings(
        self,
        compression_metadata: CompressionMetadata | None,
    ) -> bool:
        """Validate payload/scale binding and return payload availability."""

        if (self.k_scale_cache is None) != (self.v_scale_cache is None):
            raise RuntimeError("attention K/V scale caches must be bound together")
        scaled_cache_bound = self.k_scale_cache is not None
        k_payload_bound = bool(self.k_cache.numel())
        v_payload_bound = bool(self.v_cache.numel())
        if k_payload_bound != v_payload_bound:
            raise RuntimeError("attention K/V payload caches must be bound together")
        payload_cache_bound = k_payload_bound and v_payload_bound
        if scaled_cache_bound and not payload_cache_bound:
            raise RuntimeError("scale caches cannot be bound without K/V payload caches")
        if scaled_cache_bound and (
            not self.k_scale_cache.numel() or not self.v_scale_cache.numel()
        ):
            raise RuntimeError("bound K/V scale caches must not be empty")
        self._validate_compression_cache_mode(
            compression_metadata,
            payload_cache_bound=payload_cache_bound,
            scaled_cache_bound=scaled_cache_bound,
        )
        return payload_cache_bound

    @staticmethod
    def _validate_compression_cache_mode(
        compression_metadata: CompressionMetadata | None,
        *,
        payload_cache_bound: bool,
        scaled_cache_bound: bool,
    ) -> None:
        if not payload_cache_bound or compression_metadata is None:
            return
        expected_scaled_cache = compression_metadata.scaled_fp8_kv_active
        if expected_scaled_cache == scaled_cache_bound:
            return
        raise RuntimeError(
            "compression metadata/scale cache mismatch: "
            f"mode={compression_metadata.mode!r}, "
            f"scale_cache_bound={scaled_cache_bound}"
        )

    def _store_current_kv(self, k: torch.Tensor, v: torch.Tensor, context: Context) -> None:
        if context.slot_mapping is None:
            raise RuntimeError("bound KV cache requires slot_mapping")
        store_kvcache(
            k,
            v,
            self.k_cache,
            self.v_cache,
            context.slot_mapping,
            self.k_scale_cache,
            self.v_scale_cache,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        if context.block_tables is not None:
            return self._forward_prefill_paged(q, context)
        if HAS_FLASH_ATTN and q.is_cuda:
            with profile_region("attention.prefill.flash_attn_varlen"):
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    deterministic=True,
                )
        if (
            HAS_VLLM_PAGED_FLASH_ATTN
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            with profile_region("attention.prefill.vllm_flash_attn_varlen"):
                return vllm_paged_flash_attn(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    deterministic=True,
                    fa_version=2,
                )
        return self._forward_prefill_sdpa(q, k, v, context)

    def _forward_prefill_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Run variable-length flattened prefill through shape-correct SDPA."""

        if context.cu_seqlens_q is None or context.cu_seqlens_k is None:
            raise RuntimeError("SDPA prefill requires cu_seqlens_q/cu_seqlens_k")
        if context.cu_seqlens_q.numel() != context.cu_seqlens_k.numel():
            raise RuntimeError("SDPA prefill Q/K sequence counts must match")

        outputs: list[torch.Tensor] = []
        for sequence_index in range(context.cu_seqlens_q.numel() - 1):
            query_start = int(context.cu_seqlens_q[sequence_index].item())
            query_end = int(context.cu_seqlens_q[sequence_index + 1].item())
            key_start = int(context.cu_seqlens_k[sequence_index].item())
            key_end = int(context.cu_seqlens_k[sequence_index + 1].item())
            if query_end - query_start != key_end - key_start:
                raise RuntimeError(
                    "contiguous SDPA prefill requires equal Q/K lengths; "
                    "prefix and chunked prefill must use paged history"
                )

            query = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
            key = k[key_start:key_end].transpose(0, 1).unsqueeze(0)
            value = v[key_start:key_end].transpose(0, 1).unsqueeze(0)
            kwargs = {"is_causal": True, "scale": self.scale}
            if query.is_cuda:
                kwargs["enable_gqa"] = True
            elif self.num_heads != self.num_kv_heads:
                groups = self.num_heads // self.num_kv_heads
                key = key.repeat_interleave(groups, dim=1)
                value = value.repeat_interleave(groups, dim=1)
            with profile_region("attention.prefill.sdpa_varlen"):
                output = F.scaled_dot_product_attention(query, key, value, **kwargs)
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        metadata = context.compression_metadata
        if metadata is not None and metadata.visual_pruning_effective:
            return self._forward_decode_visual_prune_eager(q, context)
        if metadata is not None and metadata.fp8_kv_active:
            return self._forward_decode_fp8(q, context)
        if HAS_FLASH_ATTN and q.is_cuda and context.block_tables is None:
            return self._forward_decode_flash(q, context)
        if context.block_tables is not None:
            return self._forward_decode_paged(q, context)
        with profile_region("attention.decode.sdpa"):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)

    def _forward_decode_fp8(self, q: torch.Tensor, context: Context) -> torch.Tensor:
        if context.block_tables is None:
            raise RuntimeError("fp8_kv decode requires paged block_tables")
        if HAS_PAGED_DECODE_TRITON and q.is_cuda:
            return self._forward_decode_paged_triton(
                q,
                context,
                region_name="attention.decode.fp8_paged_triton",
            )
        return self._forward_decode_eager(
            q,
            context,
            profile_prefix="attention.decode.fp8_reference",
        )

    def _forward_decode_flash(self, q: torch.Tensor, context: Context) -> torch.Tensor:
        with profile_region("attention.decode.flash_attn_kvcache"):
            return flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache,
                self.v_cache,
                cache_seqlens=context.context_lens,
                softmax_scale=self.scale,
                causal=True,
            )

    def _forward_decode_paged(self, q: torch.Tensor, context: Context) -> torch.Tensor:
        if (
            HAS_VLLM_PAGED_FLASH_ATTN
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and self.k_cache.dtype == torch.bfloat16
            and self.v_cache.dtype == torch.bfloat16
        ):
            return self._forward_decode_paged_flash(q, context)
        if HAS_PAGED_DECODE_TRITON and q.is_cuda:
            return self._forward_decode_paged_triton(
                q,
                context,
                region_name="attention.decode.paged_triton",
            )
        return self._forward_decode_eager(
            q,
            context,
            profile_prefix="attention.decode.bf16_eager",
        )

    def _forward_decode_paged_flash(
        self,
        q: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        if context.block_tables is None or context.context_lens is None:
            raise RuntimeError("paged FlashAttention requires block tables and context lengths")
        batch_size = q.shape[0]
        cu_seqlens_q = self._paged_flash_cu_seqlens_q.get(batch_size)
        if cu_seqlens_q is None or cu_seqlens_q.device != q.device:
            cu_seqlens_q = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=q.device,
            )
            self._paged_flash_cu_seqlens_q[batch_size] = cu_seqlens_q
        max_seqlen_k = context.block_tables.shape[1] * self.k_cache.shape[1]
        with profile_region("attention.decode.vllm_flash_attn_paged"):
            return vllm_paged_flash_attn(
                q,
                self.k_cache,
                self.v_cache,
                max_seqlen_q=1,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                seqused_k=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
                fa_version=2,
            )

    def _forward_decode_paged_triton(
        self,
        q: torch.Tensor,
        context: Context,
        *,
        region_name: str,
    ) -> torch.Tensor:
        if context.block_tables is None or context.context_lens is None:
            raise RuntimeError("paged decode requires block_tables and context_lens")
        with profile_region(region_name):
            return paged_decode_attention(
                q,
                self.k_cache,
                self.v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                k_scale_cache=self.k_scale_cache,
                v_scale_cache=self.v_scale_cache,
                max_context_len=context.decode_max_context_len,
                block_n=context.paged_decode_block_n,
            )

    def _record_observability(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        context: Context,
    ) -> None:
        visual_pruning_scorer = context.visual_pruning_scorer
        if context.is_prefill and visual_pruning_scorer is not None:
            if self.layer_idx is None:
                raise RuntimeError("runtime visual scorer requires attention layer_idx")
            visual_pruning_scorer.observe(
                layer_id=self.layer_idx,
                q=q,
                k=k,
                scale=self.scale,
            )
        if not is_trace_enabled():
            return
        record_attention_layer(
            layer_id=self.layer_idx,
            q=q,
            k=k,
            v=v,
            output=output,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            context=context,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
        )

    def forward_decode_explicit(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Commit dense BF16 K/V and execute paged decode outside Dynamo.

        AOT functionalization clones mutated cache views while preserving their
        non-zero storage offsets. That is prohibitively expensive and unsafe
        for per-layer views into the monolithic cache allocation, so only the
        surrounding pure QKV region is fullgraph compiled.
        """

        if not self._validate_cache_bindings(context.compression_metadata):
            raise RuntimeError("decode torch.compile requires bound K/V caches")
        if self.k_scale_cache is not None or self.v_scale_cache is not None:
            raise RuntimeError("decode torch.compile does not support scaled KV caches")
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, self.k_cache, self.v_cache)):
            raise RuntimeError("decode torch.compile supports only dense BF16 Q/K/V tensors")
        if context.is_prefill:
            raise RuntimeError("decode torch.compile received a prefill context")
        if context.block_tables is None or context.context_lens is None:
            raise RuntimeError("decode torch.compile requires block_tables and context_lens")
        if context.decode_max_context_len is None:
            raise RuntimeError("decode torch.compile requires decode_max_context_len")
        self._store_current_kv(k, v, context)
        with profile_region("attention.decode.compile_split_paged_triton"):
            return paged_decode_attention(
                q,
                self.k_cache,
                self.v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                max_context_len=context.decode_max_context_len,
                block_n=context.paged_decode_block_n,
            )

    def _forward_prefill_paged(self, q: torch.Tensor, context: Context) -> torch.Tensor:
        """Dispatch the explicit correctness path for paged prefill."""

        return paged_prefill_attention_reference(
            q,
            self.k_cache,
            self.v_cache,
            context,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            k_scale_cache=self.k_scale_cache,
            v_scale_cache=self.v_scale_cache,
        )

    def _forward_decode_eager(
        self,
        q: torch.Tensor,
        context: Context,
        *,
        profile_prefix: str,
    ) -> torch.Tensor:
        """Dispatch the explicit PyTorch paged-decode reference."""

        return paged_decode_attention_reference(
            q,
            self.k_cache,
            self.v_cache,
            context,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            profile_prefix=profile_prefix,
            k_scale_cache=self.k_scale_cache,
            v_scale_cache=self.v_scale_cache,
        )

    def _forward_decode_visual_prune_eager(
        self,
        q: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Dispatch retained-slot visual-pruning decode reference."""

        return visual_pruned_decode_attention_reference(
            q,
            self.k_cache,
            self.v_cache,
            context,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            k_scale_cache=self.k_scale_cache,
            v_scale_cache=self.v_scale_cache,
        )
