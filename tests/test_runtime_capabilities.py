"""Runtime dependency and optimized-backend capability contracts."""

from dataclasses import replace

import pytest

from prism_infer.runtime_capabilities import (
    RuntimeCapabilities,
    detect_runtime_capabilities,
    runtime_capability_errors,
    validate_runtime_capabilities,
)


pytestmark = pytest.mark.unit


def _complete_capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        torch_version="2.6.0",
        torch_version_supported=True,
        cuda_available=True,
        distributed_available=True,
        sdpa_available=True,
        default_device_api_available=True,
        compile_available=True,
        cuda_graph_available=True,
        fp8_e4m3fn_available=True,
        triton_available=True,
    )


def test_eager_bf16_keeps_triton_as_optional_correctness_acceleration() -> None:
    capabilities = replace(
        _complete_capabilities(),
        triton_available=False,
        triton_error="ImportError: unavailable",
    )

    assert (
        runtime_capability_errors(
            capabilities,
            execution_backend="eager",
            compression_mode="off",
        )
        == ()
    )


def test_cuda_graph_reports_all_missing_optimized_requirements() -> None:
    capabilities = replace(
        _complete_capabilities(),
        cuda_graph_available=False,
        triton_available=False,
        triton_error="ImportError: incompatible ABI",
    )

    errors = runtime_capability_errors(
        capabilities,
        execution_backend="cuda_graph",
        compression_mode="off",
    )

    assert errors == (
        "CUDA Graph APIs are unavailable",
        "CUDA Graph decode requires the Triton paged-attention backend",
    )
    with pytest.raises(RuntimeError, match="incompatible ABI"):
        validate_runtime_capabilities(
            execution_backend="cuda_graph",
            compression_mode="off",
            capabilities=capabilities,
        )


def test_fp8_runtime_never_silently_falls_back_without_dtype_or_triton() -> None:
    capabilities = replace(
        _complete_capabilities(),
        fp8_e4m3fn_available=False,
        triton_available=False,
    )

    errors = runtime_capability_errors(
        capabilities,
        execution_backend="eager",
        compression_mode="scaled_fp8_kv",
    )

    assert errors == (
        "FP8 KV cache requires torch.float8_e4m3fn",
        "FP8 KV cache requires Triton store/decode kernels",
    )


def test_core_capability_errors_include_version_and_required_apis() -> None:
    capabilities = replace(
        _complete_capabilities(),
        torch_version="2.9.0",
        torch_version_supported=False,
        cuda_available=False,
        distributed_available=False,
        sdpa_available=False,
        default_device_api_available=False,
    )

    errors = runtime_capability_errors(
        capabilities,
        execution_backend="eager",
        compression_mode="off",
    )

    assert len(errors) == 5
    assert errors[0].startswith("unsupported PyTorch version")
    assert errors[-1] == "CUDA is unavailable"


def test_detected_capabilities_are_machine_readable() -> None:
    observed = detect_runtime_capabilities()
    payload = observed.as_dict()

    assert payload["torch_version"]
    assert isinstance(payload["torch_version_supported"], bool)
    assert isinstance(payload["triton_available"], bool)
