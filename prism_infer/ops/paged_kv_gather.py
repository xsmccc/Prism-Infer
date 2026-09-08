"""One-kernel paged K/V gather with the reference dequantization rounding order.

Outputs are dense [context, kv_heads, head_dim] tensors for the existing SDPA
path, not a new Attention implementation. Page IDs are scheduler-owned; padding
after ceil(context_len / page_size) is ignored. The caller selects the CUDA device.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None
    HAS_TRITON = False

_BLOCK_ELEMENTS = 1024


if HAS_TRITON:

    @triton.jit(do_not_specialize=["context_len"])
    def _gather_paged_kv_kernel(
        k_ptr,
        v_ptr,
        k_scale_ptr,
        v_scale_ptr,
        block_ids_ptr,
        output_k_ptr,
        output_v_ptr,
        context_len,
        PAGE_SIZE: tl.constexpr,
        KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        K_STRIDE_P: tl.constexpr,
        K_STRIDE_T: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_D: tl.constexpr,
        V_STRIDE_P: tl.constexpr,
        V_STRIDE_T: tl.constexpr,
        V_STRIDE_H: tl.constexpr,
        V_STRIDE_D: tl.constexpr,
        KS_STRIDE_P: tl.constexpr,
        KS_STRIDE_T: tl.constexpr,
        KS_STRIDE_H: tl.constexpr,
        VS_STRIDE_P: tl.constexpr,
        VS_STRIDE_T: tl.constexpr,
        VS_STRIDE_H: tl.constexpr,
        BLOCK_IDS_STRIDE: tl.constexpr,
        HAS_SCALES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Promote before products, including large page strides and output offsets.
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        token = offsets // (KV_HEADS * HEAD_DIM)
        head = (offsets // HEAD_DIM) % KV_HEADS
        dim = offsets % HEAD_DIM
        valid = token < context_len.to(tl.int64)
        page = tl.load(
            block_ids_ptr + (token // PAGE_SIZE) * BLOCK_IDS_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.int64)
        row = token % PAGE_SIZE
        k = tl.load(
            k_ptr + page * K_STRIDE_P + row * K_STRIDE_T + head * K_STRIDE_H + dim * K_STRIDE_D,
            mask=valid,
            other=0.0,
        )
        v = tl.load(
            v_ptr + page * V_STRIDE_P + row * V_STRIDE_T + head * V_STRIDE_H + dim * V_STRIDE_D,
            mask=valid,
            other=0.0,
        )
        target_dtype = output_k_ptr.dtype.element_ty
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        if HAS_SCALES:
            k_scale = tl.load(
                k_scale_ptr + page * KS_STRIDE_P + row * KS_STRIDE_T + head * KS_STRIDE_H,
                mask=valid,
                other=0.0,
            ).to(target_dtype)
            v_scale = tl.load(
                v_scale_ptr + page * VS_STRIDE_P + row * VS_STRIDE_T + head * VS_STRIDE_H,
                mask=valid,
                other=0.0,
            ).to(target_dtype)
            # Match PyTorch: round BOTH operands to target dtype before multiply,
            # then multiply in opmath precision and round the output to target.
            k = (k.to(tl.float32) * k_scale.to(tl.float32)).to(target_dtype)
            v = (v.to(tl.float32) * v_scale.to(tl.float32)).to(target_dtype)
        tl.store(output_k_ptr + offsets, k, mask=valid)
        tl.store(output_v_ptr + offsets, v, mask=valid)


def gather_paged_kv_fused(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_ids: torch.Tensor,
    context_len: int,
    *,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather CUDA NHD pages, optionally applying per-token/head FP32 scales.

    Positive tensor strides, distinct K/V strides, and int32/int64 page tables
    are supported. The context length is a runtime scalar, so changing it does
    not generate a separate specialization. CPU/other dtype routing is owned
    by the existing Attention caller; this function has no fallback or flags.
    """

    # Attention validates cache bindings; the scheduler owns the page table.
    # Do not repeat that work for every layer's gather.
    _, page_size, heads, head_dim = k_cache.shape
    scaled = k_scale_cache is not None
    k_scale_strides = v_scale_strides = (0, 0, 0)
    if scaled:
        k_scale_strides = k_scale_cache.stride()
        v_scale_strides = v_scale_cache.stride()

    keys = torch.empty((context_len, heads, head_dim), device=k_cache.device, dtype=target_dtype)
    values = torch.empty_like(keys)
    grid = (triton.cdiv(context_len * heads * head_dim, _BLOCK_ELEMENTS),)
    _gather_paged_kv_kernel[grid](
        k_cache,
        v_cache,
        k_scale_cache if scaled else k_cache,
        v_scale_cache if scaled else v_cache,
        block_ids,
        keys,
        values,
        context_len,
        page_size,
        heads,
        head_dim,
        *k_cache.stride(),
        *v_cache.stride(),
        *k_scale_strides,
        *v_scale_strides,
        block_ids.stride(0),
        scaled,
        _BLOCK_ELEMENTS,
        num_warps=4,
    )
    return keys, values


__all__ = ["HAS_TRITON", "gather_paged_kv_fused"]
