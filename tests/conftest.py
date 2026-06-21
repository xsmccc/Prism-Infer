import os
import sys
from pathlib import Path

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
