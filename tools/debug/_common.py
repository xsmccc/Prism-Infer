"""Shared bootstrap helpers for manually executed diagnostics."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_model_path() -> str:
    """Resolve a local Qwen3-VL checkpoint without initiating a download."""

    configured = os.environ.get("PRISM_MODEL_PATH")
    candidates = [Path(configured)] if configured else []
    model_root = Path("/data/models/Qwen3-VL-8B-Instruct")
    candidates.append(model_root)
    if model_root.is_dir():
        candidates.extend(sorted(path for path in model_root.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return str(candidate)
    raise FileNotFoundError(
        "Qwen3-VL checkpoint not found; set PRISM_MODEL_PATH to a local "
        "directory containing config.json"
    )


def require_transformers():
    """Import Transformers with an actionable error for manual runs."""

    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required; install Prism-Infer before running diagnostics"
        ) from exc
    return transformers
