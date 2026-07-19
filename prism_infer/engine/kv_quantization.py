"""Shared contracts for dynamically scaled FP8 KV storage."""

from __future__ import annotations

from typing import NamedTuple

import torch


FP8_E4M3FN_MAX = 448.0
FP8_E4M3FN_MIN = -FP8_E4M3FN_MAX
PER_TOKEN_HEAD_SCALE_FLOOR = 1.0e-6
KV_SCALE_DTYPE = torch.float32
KV_COMPONENT_COUNT = 2  # independent K and V storage


class KVStorageBytes(NamedTuple):
    """Physical bytes owned by one paged KV block across all layers."""

    payload: int
    scales: int

    @property
    def total(self) -> int:
        return self.payload + self.scales


def _positive_dimension(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def scale_cache_shape(payload_shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    """Return the token-head scale shape for one paged KV payload tensor."""

    shape = tuple(payload_shape)
    if len(shape) not in (3, 4):
        raise ValueError(
            "KV payload must be [slots, heads, dim] or "
            f"[blocks, page, heads, dim], got {list(shape)}"
        )
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in shape
    ):
        raise ValueError(f"KV payload dimensions must be positive, got {list(shape)}")
    return shape[:-1]


def kv_block_storage_bytes(
    *,
    num_layers: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    payload_dtype: torch.dtype,
    token_head_scales: bool,
) -> KVStorageBytes:
    """Return payload/scale bytes for one physical block across all layers."""

    dimensions = {
        "num_layers": num_layers,
        "page_size": page_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
    for name, value in dimensions.items():
        _positive_dimension(value, name=name)
    if not isinstance(payload_dtype, torch.dtype):
        raise TypeError(f"payload_dtype must be torch.dtype, got {payload_dtype!r}")
    if not isinstance(token_head_scales, bool):
        raise TypeError("token_head_scales must be bool")

    payload_elements = KV_COMPONENT_COUNT * num_layers * page_size * num_kv_heads * head_dim
    payload_bytes = payload_elements * torch.empty((), dtype=payload_dtype).element_size()
    scale_elements = (
        KV_COMPONENT_COUNT * num_layers * page_size * num_kv_heads if token_head_scales else 0
    )
    scale_bytes = scale_elements * torch.empty((), dtype=KV_SCALE_DTYPE).element_size()
    return KVStorageBytes(payload=payload_bytes, scales=scale_bytes)


def tensor_storage_bytes(tensor: torch.Tensor | None) -> int:
    """Return physical payload bytes without treating a missing tensor as data."""

    if tensor is None:
        return 0
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"storage object must be a tensor or None, got {type(tensor).__name__}")
    return tensor.numel() * tensor.element_size()


def kv_cache_storage_bytes(
    payload_cache: torch.Tensor,
    scale_cache: torch.Tensor | None,
) -> KVStorageBytes:
    """Return total physical payload/scale bytes for an allocated KV cache."""

    if not isinstance(payload_cache, torch.Tensor):
        raise TypeError("payload_cache must be a tensor")
    return KVStorageBytes(
        payload=tensor_storage_bytes(payload_cache),
        scales=tensor_storage_bytes(scale_cache),
    )
