import os
import sys
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    import pytest
except ImportError:  # Allows running lightweight tests as plain python scripts.
    pytest = None


def _skip(message: str):
    is_pytest = (
        os.environ.get("PYTEST_CURRENT_TEST")
        or "pytest" in Path(sys.argv[0]).name
    )
    if pytest is not None and is_pytest:
        pytest.skip(message, allow_module_level=True)
    print(f"SKIP: {message}")
    raise SystemExit(0)


def get_model_path() -> str:
    """Return the local Qwen3-VL model path or skip heavyweight HF tests."""
    candidates = []
    env_path = os.environ.get("PRISM_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "/data/models/Qwen3-VL-8B-Instruct",
    ])

    for candidate in candidates:
        path = Path(candidate)
        if (path / "config.json").is_file():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return str(path)

    message = (
        "Qwen3-VL model files not found. Set PRISM_MODEL_PATH to the local "
        "snapshot directory containing config.json."
    )
    _skip(message)


def require_transformers():
    try:
        import transformers
    except ImportError:
        _skip("transformers is not installed")
    return transformers


def build_mm_token_type_ids(input_ids, *, image_token_id, video_token_id):
    """Build Qwen3-VL HF reference modality ids from expanded pad tokens."""

    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    if image_token_id is not None:
        token_type_ids[input_ids == int(image_token_id)] = 1
    if video_token_id is not None:
        token_type_ids[input_ids == int(video_token_id)] = 2
    return token_type_ids


def hf_qwen3_vl_rope_index(
    transformers,
    config,
    *,
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    """Call HF Qwen3-VL get_rope_index across old/new transformers signatures."""

    hf_model_cls = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel
    dummy_model = SimpleNamespace(config=config)
    if hasattr(hf_model_cls, "get_vision_position_ids"):
        dummy_model.get_vision_position_ids = (
            lambda *args, **kwargs: hf_model_cls.get_vision_position_ids(
                dummy_model,
                *args,
                **kwargs,
            )
        )
    kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "attention_mask": attention_mask,
    }
    if "mm_token_type_ids" in signature(hf_model_cls.get_rope_index).parameters:
        kwargs["mm_token_type_ids"] = build_mm_token_type_ids(
            input_ids,
            image_token_id=getattr(config, "image_token_id", None),
            video_token_id=getattr(config, "video_token_id", None),
        )
    return hf_model_cls.get_rope_index(dummy_model, **kwargs)


def hf_qwen3_vl_visual(model):
    """Return HF Qwen3-VL vision module across transformers layouts."""

    if hasattr(model, "visual"):
        return model.visual
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return inner.visual
    raise AttributeError(f"{type(model).__name__} does not expose a visual module")
