"""Physical writes into canonical paged KV-cache storage.

This module owns the storage contract and the optional Triton implementation.
Attention layers select *when* to write; this module defines *how* payloads and
per-token/per-head scales are validated and written.
"""

from __future__ import annotations

import torch

from prism_infer.engine.kv_quantization import (
    FP8_E4M3FN_MAX,
    FP8_E4M3FN_MIN,
    KV_SCALE_DTYPE,
    PER_TOKEN_HEAD_SCALE_FLOOR,
    scale_cache_shape,
)
from prism_infer.observability import profile_region


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


TRITON_WARP_WIDTH = 32
MAX_STORE_WARPS = 16
KV_TOKEN_HEAD_TENSOR_RANK = 3
SUPPORTED_KV_CACHE_RANKS = (3, 4)
SLOT_MAPPING_TENSOR_RANK = 1


def _float8_cache_dtypes() -> tuple[torch.dtype, ...]:
    """Return the FP8 cache dtypes supported by the active PyTorch runtime."""

    dtypes = []
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    return tuple(dtypes)


FP8_CACHE_DTYPES = _float8_cache_dtypes()


def is_fp8_cache_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` uses a supported FP8 KV payload dtype."""

    return tensor.dtype in FP8_CACHE_DTYPES


if HAS_TRITON:

    @triton.jit
    def _store_kvcache_triton(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    @triton.jit
    def _store_scaled_kvcache_triton(
        key_ptr,
        value_ptr,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_cache_ptr,
        v_scale_cache_ptr,
        slot_mapping_ptr,
        key_stride_token: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_dim: tl.constexpr,
        k_cache_stride_slot: tl.constexpr,
        k_cache_stride_head: tl.constexpr,
        k_cache_stride_dim: tl.constexpr,
        v_cache_stride_slot: tl.constexpr,
        v_cache_stride_head: tl.constexpr,
        v_cache_stride_dim: tl.constexpr,
        k_scale_stride_slot: tl.constexpr,
        k_scale_stride_head: tl.constexpr,
        v_scale_stride_slot: tl.constexpr,
        v_scale_stride_head: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        QUANT_MIN: tl.constexpr,
        QUANT_MAX: tl.constexpr,
        SCALE_FLOOR: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        slot = tl.load(slot_mapping_ptr + token)
        if slot < 0:
            return

        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < HEAD_DIM
        key = tl.load(
            key_ptr + token * key_stride_token + head * key_stride_head + offsets * key_stride_dim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            value_ptr
            + token * value_stride_token
            + head * value_stride_head
            + offsets * value_stride_dim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        k_scale = tl.maximum(
            tl.max(tl.abs(key), axis=0) / QUANT_MAX,
            SCALE_FLOOR,
        )
        v_scale = tl.maximum(
            tl.max(tl.abs(value), axis=0) / QUANT_MAX,
            SCALE_FLOOR,
        )
        quantized_key = tl.clamp(key / k_scale, QUANT_MIN, QUANT_MAX)
        quantized_value = tl.clamp(value / v_scale, QUANT_MIN, QUANT_MAX)

        tl.store(
            k_cache_ptr
            + slot * k_cache_stride_slot
            + head * k_cache_stride_head
            + offsets * k_cache_stride_dim,
            quantized_key,
            mask=mask,
        )
        tl.store(
            v_cache_ptr
            + slot * v_cache_stride_slot
            + head * v_cache_stride_head
            + offsets * v_cache_stride_dim,
            quantized_value,
            mask=mask,
        )
        tl.store(
            k_scale_cache_ptr + slot * k_scale_stride_slot + head * k_scale_stride_head,
            k_scale,
        )
        tl.store(
            v_scale_cache_ptr + slot * v_scale_stride_slot + head * v_scale_stride_head,
            v_scale,
        )


def _store_kvcache_eager(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> None:
    """Write flat slots using the explicit PyTorch correctness path."""

    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache/v_cache shape mismatch: {list(k_cache.shape)} vs {list(v_cache.shape)}"
        )
    if k_cache.ndim not in (3, 4):
        raise ValueError(
            "k_cache/v_cache must be [slots, heads, dim] or "
            f"[num_blocks, block_size, heads, dim], got {list(k_cache.shape)}"
        )

    scaled = k_scale_cache is not None
    flat_k_scale = (
        None if k_scale_cache is None else k_scale_cache.reshape(-1, k_scale_cache.shape[-1])
    )
    flat_v_scale = (
        None if v_scale_cache is None else v_scale_cache.reshape(-1, v_scale_cache.shape[-1])
    )
    flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

    for token_index in range(key.shape[0]):
        slot = int(slot_mapping[token_index].item())
        if slot < 0:
            continue
        if scaled:
            key_float = key[token_index].float()
            value_float = value[token_index].float()
            k_scale = torch.clamp(
                key_float.abs().amax(dim=-1) / FP8_E4M3FN_MAX,
                min=PER_TOKEN_HEAD_SCALE_FLOOR,
            )
            v_scale = torch.clamp(
                value_float.abs().amax(dim=-1) / FP8_E4M3FN_MAX,
                min=PER_TOKEN_HEAD_SCALE_FLOOR,
            )
            flat_k_cache[slot] = torch.clamp(
                key_float / k_scale.unsqueeze(-1),
                min=FP8_E4M3FN_MIN,
                max=FP8_E4M3FN_MAX,
            ).to(k_cache.dtype)
            flat_v_cache[slot] = torch.clamp(
                value_float / v_scale.unsqueeze(-1),
                min=FP8_E4M3FN_MIN,
                max=FP8_E4M3FN_MAX,
            ).to(v_cache.dtype)
            flat_k_scale[slot] = k_scale
            flat_v_scale[slot] = v_scale
        else:
            flat_k_cache[slot] = key[token_index].to(k_cache.dtype)
            flat_v_cache[slot] = value[token_index].to(v_cache.dtype)


def _validate_store_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> bool:
    """Validate the public storage contract and return whether scales are used."""

    _validate_payload_store_inputs(key, value, k_cache, v_cache, slot_mapping)
    return _validate_scale_store_inputs(k_cache, k_scale_cache, v_scale_cache)


def _validate_payload_store_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if (
        key.ndim != KV_TOKEN_HEAD_TENSOR_RANK
        or value.ndim != KV_TOKEN_HEAD_TENSOR_RANK
        or key.shape != value.shape
    ):
        raise ValueError(
            "key/value must have the same [tokens, KV heads, head dim] shape, "
            f"got {list(key.shape)} and {list(value.shape)}"
        )
    if k_cache.shape != v_cache.shape or k_cache.dtype != v_cache.dtype:
        raise ValueError("K/V payload caches must have matching shape and dtype")
    if k_cache.ndim not in SUPPORTED_KV_CACHE_RANKS:
        raise ValueError(
            "K/V payload caches must be [slots, heads, dim] or [blocks, page, heads, dim]"
        )
    if tuple(k_cache.shape[-2:]) != tuple(key.shape[-2:]):
        raise ValueError(
            "input/cache KV head shape mismatch: "
            f"input={list(key.shape[-2:])}, cache={list(k_cache.shape[-2:])}"
        )
    if slot_mapping.ndim != SLOT_MAPPING_TENSOR_RANK or slot_mapping.numel() != key.shape[0]:
        raise ValueError("slot_mapping must contain one flat slot per input token")


def _validate_scale_store_inputs(
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> bool:
    if (k_scale_cache is None) != (v_scale_cache is None):
        raise ValueError("K/V scale caches must be provided together")
    scaled = k_scale_cache is not None
    if not scaled:
        return False
    if not is_fp8_cache_tensor(k_cache):
        raise ValueError("token-head scales require FP8 K/V payload caches")
    expected_scale_shape = scale_cache_shape(k_cache.shape)
    if (
        k_scale_cache.shape != v_scale_cache.shape
        or tuple(k_scale_cache.shape) != expected_scale_shape
    ):
        raise ValueError(
            "K/V scale cache shape must equal payload shape without head_dim: "
            f"expected={list(expected_scale_shape)}"
        )
    if k_scale_cache.dtype != KV_SCALE_DTYPE or v_scale_cache.dtype != KV_SCALE_DTYPE:
        raise ValueError("K/V scale caches must use torch.float32")
    return scaled


def _validate_triton_store_layout(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
) -> None:
    optional_scales = tuple(
        tensor for tensor in (k_scale_cache, v_scale_cache) if tensor is not None
    )
    tensors = (key, value, k_cache, v_cache, slot_mapping, *optional_scales)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("Triton KV store requires all tensors on CUDA")
    if any(tensor.device != key.device for tensor in tensors[1:]):
        raise RuntimeError("Triton KV store requires all tensors on the same device")
    if slot_mapping.dtype != torch.int32:
        raise ValueError("Triton KV store requires torch.int32 slot_mapping")
    head_dim = key.shape[-1]
    if (
        key.stride(-1) != 1
        or value.stride(-1) != 1
        or key.stride(-2) != head_dim
        or value.stride(-2) != head_dim
    ):
        raise RuntimeError("Triton KV store requires contiguous head/dim inputs")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise RuntimeError("Triton KV store requires contiguous KV caches")
    if k_scale_cache is not None and (
        not k_scale_cache.is_contiguous() or not v_scale_cache.is_contiguous()
    ):
        raise RuntimeError("Triton scaled KV store requires contiguous scale caches")


def _store_profile_region(use_triton: bool, scaled: bool, k_cache: torch.Tensor) -> str:
    if not use_triton:
        return "attention.kv_store.eager"
    if scaled:
        return "attention.kv_store.triton_scaled_fp8"
    if is_fp8_cache_tensor(k_cache):
        return "attention.kv_store.triton_fp8"
    return "attention.kv_store.triton"


def _launch_scaled_triton_store(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = key.shape
    block_d = triton.next_power_of_2(head_dim)
    num_warps = min(MAX_STORE_WARPS, max(1, block_d // TRITON_WARP_WIDTH))
    flat_k_cache = k_cache.reshape(-1, num_heads, head_dim)
    flat_v_cache = v_cache.reshape(-1, num_heads, head_dim)
    flat_k_scale = k_scale_cache.reshape(-1, num_heads)
    flat_v_scale = v_scale_cache.reshape(-1, num_heads)
    _store_scaled_kvcache_triton[(num_tokens, num_heads)](
        key,
        value,
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        flat_k_cache.stride(0),
        flat_k_cache.stride(1),
        flat_k_cache.stride(2),
        flat_v_cache.stride(0),
        flat_v_cache.stride(1),
        flat_v_cache.stride(2),
        flat_k_scale.stride(0),
        flat_k_scale.stride(1),
        flat_v_scale.stride(0),
        flat_v_scale.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        QUANT_MIN=FP8_E4M3FN_MIN,
        QUANT_MAX=FP8_E4M3FN_MAX,
        SCALE_FLOOR=PER_TOKEN_HEAD_SCALE_FLOOR,
        num_warps=num_warps,
    )


def _launch_dense_triton_store(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = key.shape
    _store_kvcache_triton[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        num_heads * head_dim,
    )


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> None:
    """Write current K/V values into canonical paged KV-cache slots."""

    scaled = _validate_store_inputs(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale_cache,
        v_scale_cache,
    )
    use_triton = HAS_TRITON and key.is_cuda
    if use_triton:
        _validate_triton_store_layout(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale_cache,
            v_scale_cache,
        )
    num_tokens = key.shape[0]
    with profile_region(
        _store_profile_region(use_triton, scaled, k_cache),
        metadata={"cache_dtype": str(k_cache.dtype), "tokens": num_tokens},
    ):
        if use_triton and scaled:
            _launch_scaled_triton_store(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                k_scale_cache,
                v_scale_cache,
            )
        elif use_triton:
            _launch_dense_triton_store(key, value, k_cache, v_cache, slot_mapping)
        else:
            _store_kvcache_eager(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                k_scale_cache,
                v_scale_cache,
            )


__all__ = [
    "FP8_CACHE_DTYPES",
    "HAS_TRITON",
    "_store_kvcache_eager",
    "is_fp8_cache_tensor",
    "store_kvcache",
]
