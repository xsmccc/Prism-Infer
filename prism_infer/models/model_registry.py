"""Fail-closed mapping from checkpoint metadata to implemented model families."""

from __future__ import annotations

from enum import Enum

from prism_infer.models.qwen3_vl_architecture import Qwen3VLArchitecture


class ModelFamily(str, Enum):
    QWEN3 = "qwen3"
    QWEN3_VL = "qwen3_vl"


def resolve_model_family(hf_config: object) -> ModelFamily:
    model_type = getattr(hf_config, "model_type", None)
    try:
        return ModelFamily(model_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported model_type {model_type!r}; "
            f"supported={[family.value for family in ModelFamily]}"
        ) from exc


def validate_model_architecture(hf_config: object) -> ModelFamily:
    """Validate implemented architectural assumptions before GPU side effects."""

    family = resolve_model_family(hf_config)
    if family is ModelFamily.QWEN3_VL:
        Qwen3VLArchitecture.from_config(hf_config)
    return family


__all__ = ["ModelFamily", "resolve_model_family", "validate_model_architecture"]
