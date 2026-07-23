"""Optional CUTLASS dual-GEMM SwiGLU path for large Blackwell prefills."""

from __future__ import annotations

import importlib.util
import os
import sysconfig
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


MIN_CUTLASS_DUAL_SWIGLU_ROWS = 1024
QWEN3_VL_HIDDEN_SIZE = 4096
QWEN3_VL_PACKED_INTERMEDIATE_SIZE = 24576
SUPPORTED_COMPUTE_CAPABILITY = (12, 0)


@lru_cache(maxsize=1)
def _cutlass_source_root() -> Path | None:
    spec = importlib.util.find_spec("cutlass_library")
    if spec is None or spec.origin is None:
        return None
    source_root = Path(spec.origin).resolve().parent / "source"
    if not (source_root / "include" / "cutlass" / "cutlass.h").is_file():
        return None
    if not (
        source_root / "examples" / "45_dual_gemm" / "device" / "dual_gemm.h"
    ).is_file():
        return None
    return source_root


def _ensure_ninja_on_path() -> None:
    scripts_dir = Path(sysconfig.get_path("scripts"))
    ninja_name = "ninja.exe" if os.name == "nt" else "ninja"
    if not (scripts_dir / ninja_name).is_file():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(scripts_dir) not in path_entries:
        os.environ["PATH"] = os.pathsep.join((str(scripts_dir), *path_entries))


@lru_cache(maxsize=1)
def _load_cutlass_dual_swiglu() -> ModuleType | None:
    source_root = _cutlass_source_root()
    if source_root is None:
        return None

    _ensure_ninja_on_path()
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "cutlass_dual_gemm_swiglu.cu"
    try:
        return load(
            name="prism_cutlass_dual_swiglu",
            sources=[str(source)],
            extra_include_paths=[
                str(source_root / "include"),
                str(source_root / "examples"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-gencode=arch=compute_120,code=sm_120",
            ],
            verbose=os.environ.get("PRISM_VERBOSE_KERNEL_BUILD") == "1",
        )
    except Exception as error:  # pragma: no cover - depends on the local CUDA toolchain
        warnings.warn(
            f"CUTLASS dual-GEMM SwiGLU is unavailable: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def can_use_cutlass_dual_swiglu(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
) -> bool:
    """Return whether the validated RTX 5090 large-prefill path applies."""

    if os.environ.get("PRISM_DISABLE_CUTLASS_DUAL_SWIGLU") == "1":
        return False
    if torch.compiler.is_compiling():
        return False
    if hidden_states.ndim != 2 or packed_weight.ndim != 2:
        return False
    if hidden_states.shape[0] < MIN_CUTLASS_DUAL_SWIGLU_ROWS:
        return False
    if hidden_states.shape[1] != QWEN3_VL_HIDDEN_SIZE:
        return False
    if packed_weight.shape != (
        QWEN3_VL_PACKED_INTERMEDIATE_SIZE,
        QWEN3_VL_HIDDEN_SIZE,
    ):
        return False
    if hidden_states.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        return False
    if not hidden_states.is_cuda or not packed_weight.is_cuda:
        return False
    if hidden_states.device != packed_weight.device:
        return False
    if not hidden_states.is_contiguous() or not packed_weight.is_contiguous():
        return False
    if torch.cuda.get_device_capability(hidden_states.device) != SUPPORTED_COMPUTE_CAPABILITY:
        return False
    return _cutlass_source_root() is not None


def maybe_cutlass_dual_swiglu(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor | None:
    """Run the fused projection when supported, otherwise request eager fallback."""

    if not can_use_cutlass_dual_swiglu(hidden_states, packed_weight):
        return None
    extension = _load_cutlass_dual_swiglu()
    if extension is None:
        return None
    return extension.dual_gemm_swiglu_bf16(hidden_states, packed_weight)


__all__ = [
    "MIN_CUTLASS_DUAL_SWIGLU_ROWS",
    "can_use_cutlass_dual_swiglu",
    "maybe_cutlass_dual_swiglu",
]
