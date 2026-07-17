#!/usr/bin/env python3
"""Prism-Infer 安装、模型快照与 CUDA 环境的无权重加载检查。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


GIB = 1024 ** 3
CORE_DISTRIBUTIONS = (
    "prism-infer",
    "torch",
    "transformers",
    "numpy",
    "safetensors",
    "tqdm",
    "xxhash",
    "Pillow",
)
OPTIONAL_BACKENDS = ("triton", "flash_attn")


def _distribution_version(name: str) -> str | None:
    """返回已安装 distribution 版本；缺失 metadata 时返回 None。"""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(name: str) -> dict[str, Any]:
    """检查可选 backend 是否可导入，不把缺失视为核心安装失败。"""

    try:
        module = importlib.import_module(name)
    except Exception as exc:  # Import-time ABI errors are part of the diagnosis.
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "version": None,
        }
    return {
        "available": True,
        "error": None,
        "version": getattr(module, "__version__", None),
    }


def _sha256(path: Path) -> str:
    """流式计算小型身份文件的 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model(path_value: str | None) -> tuple[dict[str, Any], list[str]]:
    """验证 Qwen3-VL 本地 snapshot 的最小文件集合。"""

    if not path_value:
        return {"status": "NOT_CHECKED", "path": None}, []

    path = Path(path_value).expanduser().resolve()
    result: dict[str, Any] = {"status": "FAIL", "path": str(path)}
    errors: list[str] = []
    if not path.is_dir():
        return result, [f"model directory does not exist: {path}"]

    required = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    weights = sorted(path.glob("*.safetensors"))
    if missing:
        errors.append(f"model snapshot is missing: {', '.join(missing)}")
    if not weights:
        errors.append("model snapshot contains no *.safetensors weights")

    config_path = path / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse config.json: {exc}")

    architectures = config.get("architectures", [])
    model_type = config.get("model_type")
    if model_type != "qwen3_vl" and not any("Qwen3VL" in item for item in architectures):
        errors.append(
            "model config is not identified as Qwen3-VL: "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )

    result.update(
        {
            "status": "PASS" if not errors else "FAIL",
            "model_type": model_type,
            "architectures": architectures,
            "config_sha256": _sha256(config_path) if config_path.is_file() else None,
            "weight_files": len(weights),
            "weight_gib": round(sum(item.stat().st_size for item in weights) / GIB, 3),
            "missing_files": missing,
        }
    )
    return result, errors


def inspect_cuda(torch_module: Any) -> dict[str, Any]:
    """查询 CUDA 设备与当前进程可见空闲显存，不加载模型。"""

    available = bool(torch_module.cuda.is_available())
    result: dict[str, Any] = {
        "available": available,
        "torch_cuda_version": torch_module.version.cuda,
        "device_count": 0,
        "devices": [],
    }
    if not available:
        return result

    result["device_count"] = torch_module.cuda.device_count()
    for index in range(result["device_count"]):
        with torch_module.cuda.device(index):
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        properties = torch_module.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": properties.name,
                "capability": list(torch_module.cuda.get_device_capability(index)),
                "free_gib": round(free_bytes / GIB, 3),
                "total_gib": round(total_bytes / GIB, 3),
            }
        )
    return result


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    """构造机器可读报告，并分离失败与非阻断警告。"""

    errors: list[str] = []
    warnings: list[str] = []
    distributions = {
        name: _distribution_version(name) for name in CORE_DISTRIBUTIONS
    }
    missing_distributions = [name for name, version in distributions.items() if version is None]
    if missing_distributions:
        errors.append(f"missing core distributions: {', '.join(missing_distributions)}")

    try:
        import prism_infer
    except Exception as exc:
        prism_module = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"cannot import prism_infer: {type(exc).__name__}: {exc}")
    else:
        prism_module = {
            "importable": True,
            "error": None,
            "path": str(Path(prism_infer.__file__).resolve()),
        }

    try:
        import torch
    except Exception as exc:
        cuda = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"cannot import torch: {type(exc).__name__}: {exc}")
    else:
        cuda = inspect_cuda(torch)

    if args.require_cuda and not cuda.get("available", False):
        errors.append("CUDA is required but torch.cuda.is_available() is false")
    devices = cuda.get("devices", [])
    if args.min_free_gib is not None:
        if not devices:
            errors.append("--min-free-gib requires at least one visible CUDA device")
        elif devices[0]["free_gib"] < args.min_free_gib:
            errors.append(
                f"GPU 0 has {devices[0]['free_gib']:.3f} GiB free; "
                f"{args.min_free_gib:.3f} GiB required"
            )

    backends = {name: _module_version(name) for name in OPTIONAL_BACKENDS}
    for name, backend in backends.items():
        if not backend["available"]:
            warnings.append(
                f"optional backend {name} is unavailable: {backend['error']}"
            )

    model, model_errors = inspect_model(args.model)
    errors.extend(model_errors)
    if model["status"] == "NOT_CHECKED":
        warnings.append("model snapshot not checked; pass --model or set PRISM_MODEL_PATH")

    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "distributions": distributions,
        "prism_module": prism_module,
        "optional_backends": backends,
        "cuda": cuda,
        "model": model,
        "errors": errors,
        "warnings": warnings,
    }
    return report, errors, warnings


def print_human(report: dict[str, Any]) -> None:
    """输出适合 README smoke 的紧凑文本。"""

    print("Prism-Infer environment check")
    print(f"status: {report['status']}")
    print(
        "runtime: "
        f"Python {report['python']['version']}, "
        f"Prism {report['distributions']['prism-infer']}, "
        f"Torch {report['distributions']['torch']}, "
        f"Transformers {report['distributions']['transformers']}"
    )
    backend_text = ", ".join(
        f"{name}={'yes' if value['available'] else 'no'}"
        for name, value in report["optional_backends"].items()
    )
    print(f"optional backends: {backend_text}")
    cuda = report["cuda"]
    print(
        f"cuda: available={cuda.get('available', False)}, "
        f"version={cuda.get('torch_cuda_version')}, "
        f"devices={cuda.get('device_count', 0)}"
    )
    for device in cuda.get("devices", []):
        print(
            f"gpu[{device['index']}]: {device['name']}, "
            f"cc={device['capability'][0]}.{device['capability'][1]}, "
            f"free/total={device['free_gib']:.3f}/{device['total_gib']:.3f} GiB"
        )
    model = report["model"]
    if model["status"] == "NOT_CHECKED":
        print("model: NOT_CHECKED")
    else:
        print(
            f"model: {model['status']}, type={model.get('model_type')}, "
            f"weights={model.get('weight_files', 0)} files/{model.get('weight_gib', 0):.3f} GiB"
        )
        print(f"model path: {model['path']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the Prism-Infer installation, local model snapshot, optional "
            "CUDA backends, and visible free memory without loading model weights."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help="local Qwen3-VL snapshot; defaults to PRISM_MODEL_PATH",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when PyTorch cannot see a CUDA device",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=None,
        help="fail when visible free memory on GPU 0 is below this threshold",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()
    if args.model is None:
        import os

        args.model = os.environ.get("PRISM_MODEL_PATH")
    if args.min_free_gib is not None and args.min_free_gib <= 0:
        parser.error("--min-free-gib must be positive")
    return args


def main() -> int:
    args = parse_args()
    report, errors, _warnings = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
