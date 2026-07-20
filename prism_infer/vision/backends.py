"""Explicit vision-attention backend policy shared by config and model code."""

from enum import Enum


class VisionAttentionBackendName(str, Enum):
    """Startup-selected vision attention backend."""

    SDPA = "sdpa"
    FLASH_ATTN = "flash_attn"


def normalize_vision_attention_backend(
    value: VisionAttentionBackendName | str,
) -> VisionAttentionBackendName:
    """Normalize a public backend value and reject implicit fallback modes."""

    try:
        return VisionAttentionBackendName(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(item.value for item in VisionAttentionBackendName)
        raise ValueError(
            f"vision attention backend must be one of {supported}; got {value!r}"
        ) from exc
