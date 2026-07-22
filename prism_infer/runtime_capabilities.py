"""Fail-closed runtime capability contract for optimized execution paths."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from packaging.version import InvalidVersion, Version

from prism_infer.engine.compression import compression_mode_uses_fp8_payload


MIN_TORCH_VERSION = Version("2.5")
MAX_TORCH_VERSION_EXCLUSIVE = Version("2.9")


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Observed APIs; availability does not imply a benchmark claim."""

    torch_version: str
    torch_version_supported: bool
    cuda_available: bool
    distributed_available: bool
    sdpa_available: bool
    default_device_api_available: bool
    compile_available: bool
    cuda_graph_available: bool
    fp8_e4m3fn_available: bool
    triton_available: bool
    triton_error: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _supported_torch_version(raw_version: str) -> bool:
    try:
        version = Version(raw_version)
    except InvalidVersion:
        return False
    return MIN_TORCH_VERSION <= version < MAX_TORCH_VERSION_EXCLUSIVE


def detect_runtime_capabilities() -> RuntimeCapabilities:
    """Inspect required APIs without allocating model or KV-cache tensors."""

    triton_error = None
    try:
        importlib.import_module("triton")
    except Exception as exc:  # Import-time ABI failures are capabilities too.
        triton_available = False
        triton_error = f"{type(exc).__name__}: {exc}"
    else:
        triton_available = True
    torch_version = str(torch.__version__)
    return RuntimeCapabilities(
        torch_version=torch_version,
        torch_version_supported=_supported_torch_version(torch_version),
        cuda_available=bool(torch.cuda.is_available()),
        distributed_available=bool(torch.distributed.is_available()),
        sdpa_available=callable(getattr(F, "scaled_dot_product_attention", None)),
        default_device_api_available=(
            callable(getattr(torch, "get_default_device", None))
            and callable(getattr(torch, "set_default_device", None))
        ),
        compile_available=callable(getattr(torch, "compile", None)),
        cuda_graph_available=(
            hasattr(torch.cuda, "CUDAGraph") and callable(getattr(torch.cuda, "graph", None))
        ),
        fp8_e4m3fn_available=hasattr(torch, "float8_e4m3fn"),
        triton_available=triton_available,
        triton_error=triton_error,
    )


def runtime_capability_errors(
    capabilities: RuntimeCapabilities,
    *,
    execution_backend: str,
    compression_mode: str,
    require_cuda: bool = True,
) -> tuple[str, ...]:
    """Return every missing requirement for one startup-selected path."""

    errors = _core_capability_errors(capabilities, require_cuda=require_cuda)
    errors.extend(_backend_capability_errors(capabilities, execution_backend))
    errors.extend(_compression_capability_errors(capabilities, compression_mode))
    return tuple(errors)


def _core_capability_errors(
    capabilities: RuntimeCapabilities,
    *,
    require_cuda: bool,
) -> list[str]:
    """Validate APIs shared by every engine execution backend."""

    errors: list[str] = []
    if not capabilities.torch_version_supported:
        errors.append(
            "unsupported PyTorch version "
            f"{capabilities.torch_version!r}; supported range is "
            f">={MIN_TORCH_VERSION},<{MAX_TORCH_VERSION_EXCLUSIVE}"
        )
    if not capabilities.distributed_available:
        errors.append("torch.distributed is unavailable")
    if not capabilities.sdpa_available:
        errors.append("torch.nn.functional.scaled_dot_product_attention is unavailable")
    if not capabilities.default_device_api_available:
        errors.append("torch default-device APIs are unavailable")
    if require_cuda and not capabilities.cuda_available:
        errors.append("CUDA is unavailable")
    return errors


def _backend_capability_errors(
    capabilities: RuntimeCapabilities,
    execution_backend: str,
) -> list[str]:
    """Validate requirements owned by one explicitly selected backend."""

    errors: list[str] = []
    if execution_backend == "cuda_graph":
        if not capabilities.cuda_graph_available:
            errors.append("CUDA Graph APIs are unavailable")
        if not capabilities.triton_available:
            errors.append("CUDA Graph decode requires the Triton paged-attention backend")
    elif execution_backend == "compile":
        if not capabilities.compile_available:
            errors.append("torch.compile is unavailable")
        if not capabilities.triton_available:
            errors.append("torch.compile execution requires Triton")
    elif execution_backend != "eager":
        errors.append(f"unknown execution backend {execution_backend!r}")
    return errors


def _compression_capability_errors(
    capabilities: RuntimeCapabilities,
    compression_mode: str,
) -> list[str]:
    """Validate dtype and kernel requirements of the selected KV payload."""

    errors: list[str] = []
    if compression_mode_uses_fp8_payload(compression_mode):
        if not capabilities.fp8_e4m3fn_available:
            errors.append("FP8 KV cache requires torch.float8_e4m3fn")
        if not capabilities.triton_available:
            errors.append("FP8 KV cache requires Triton store/decode kernels")
    return errors


def validate_runtime_capabilities(
    *,
    execution_backend: str,
    compression_mode: str,
    require_cuda: bool = True,
    capabilities: RuntimeCapabilities | None = None,
) -> RuntimeCapabilities:
    """Validate one runtime path before tokenizer, process, or model side effects."""

    observed = capabilities or detect_runtime_capabilities()
    errors = runtime_capability_errors(
        observed,
        execution_backend=execution_backend,
        compression_mode=compression_mode,
        require_cuda=require_cuda,
    )
    if errors:
        details = "; ".join(errors)
        if observed.triton_error and not observed.triton_available:
            details += f"; Triton import: {observed.triton_error}"
        raise RuntimeError(f"runtime capability check failed: {details}")
    return observed
