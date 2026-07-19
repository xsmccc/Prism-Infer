"""Fail-closed P9 quality and KV-pool contract for external vLLM baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_protocol import validate_p9_quality_protocol
from prism_infer.analysis.p9_quality_comparison import (
    compare_quality_sample_sets,
    validate_quality_artifact,
    validate_quality_samples,
)
from prism_infer.analysis.p9_quality_materialization import selected_ids_sha256
from prism_infer.analysis.schema_constants import (
    COMPUTE_CAPABILITY_COMPONENT_COUNT,
    GIT_SHA1_HEX_LENGTH,
    LOWERCASE_HEX_DIGITS,
    SHA256_HEX_LENGTH,
)

EXTERNAL_QUALITY_SCHEMA_VERSION = 1
EXTERNAL_QUALITY_COMPARISON_SCHEMA_VERSION = 1
EXTERNAL_QUALITY_RECORD_TYPE = "p9_external_quality_predictions"
EXTERNAL_QUALITY_COMPARISON_RECORD_TYPE = "p9_prism_external_quality_comparison"
VLLM_FRAMEWORK = "vllm"
VLLM_FRAMEWORK_VERSION = "0.24.0"
VLLM_DISTRIBUTION_COMMIT = "gee0da84ab"
VLLM_TRANSFORMERS_VERSION = "5.13.0"
VLLM_ATTENTION_BACKEND = "TRITON_ATTN"
VLLM_VISION_ATTENTION_BACKEND = "FLASH_ATTN"
VLLM_KV_BLOCK_SIZE = 16
VLLM_REQUIRED_ENVIRONMENT = (
    ("CUDA_VISIBLE_DEVICES", "0"),
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ("VLLM_USE_FLASHINFER_SAMPLER", "0"),
)
# Source files that define the frozen vLLM execution path.  The external
# runner records each file's bytes directly; a wheel version or RECORD alone
# is not enough to identify editable/site-local Python implementations.
VLLM_IMPLEMENTATION_FILES = (
    "config/cache.py",
    "model_executor/models/qwen2_vl.py",
    "model_executor/models/qwen3_vl.py",
    "model_executor/models/vision.py",
    "multimodal/inputs.py",
    "multimodal/parse.py",
    "multimodal/processing/processor.py",
    "v1/attention/backends/flash_attn.py",
    "v1/attention/backends/triton_attn.py",
    "v1/attention/ops/triton_reshape_and_cache_flash.py",
    "v1/core/kv_cache_utils.py",
    "v1/kv_cache_interface.py",
    "v1/worker/gpu_model_runner.py",
)
# Qwen3-VL uses Qwen2-VL's image processor and its own text/video processors.
# These files determine resize, placeholder expansion, and timestamp semantics.
TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES = (
    "models/qwen2_vl/image_processing_qwen2_vl.py",
    "models/qwen3_vl/processing_qwen3_vl.py",
    "models/qwen3_vl/video_processing_qwen3_vl.py",
)
# Effective minima inherited from the pinned Qwen3-VL processor revision.
# The runner verifies these values against AutoProcessor before engine startup;
# passing only ``max_pixels`` is invalid for Transformers 5.13 video processing.
FROZEN_QWEN3_VL_PROCESSOR_MIN_PIXELS = {
    "image": 65_536,
    "video": 4_096,
}
VLLM_PROMPT_ADAPTERS = {
    "image": "none",
    "video": "qwen3_vl_preserve_hf_outer_video_markers_v1",
}
VLLM_QUALITY_MODES = {
    "bf16": {
        "kv_cache_dtype": "bfloat16",
        "payload_dtype": "torch.bfloat16",
        "payload_element_bytes": 2,
        "scale_dtype": "none",
        "storage_scale_elements_per_token_head": 0,
    },
    "fp8_per_token_head": {
        "kv_cache_dtype": "fp8_per_token_head",
        "payload_dtype": "torch.float8_e4m3fn",
        "payload_element_bytes": 1,
        "scale_dtype": "torch.float32",
        # One FP32 scale for K and one for V at every (token, KV-head).
        "storage_scale_elements_per_token_head": 2,
    },
}
QUALITY_DATASETS = {
    "docvqa_validation",
    "muirbench_test",
    "mvbench_test",
}


def adapt_vllm_prompt_text(
    prompt_text: str,
    *,
    modality: str,
    vision_start_token: str,
    media_token: str,
    vision_end_token: str,
) -> str:
    """Apply the frozen vLLM-0.24 prompt compatibility adapter."""

    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("vLLM prompt text must be a non-empty string")
    adapter = VLLM_PROMPT_ADAPTERS.get(modality)
    if adapter is None:
        raise ValueError(f"unsupported vLLM prompt modality: {modality!r}")
    if adapter == "none":
        return prompt_text
    if adapter != "qwen3_vl_preserve_hf_outer_video_markers_v1":
        raise ValueError(f"unsupported vLLM prompt adapter: {adapter!r}")
    markers = (vision_start_token, media_token, vision_end_token)
    if not all(isinstance(marker, str) and marker for marker in markers):
        raise ValueError("vLLM video marker tokens must be non-empty strings")
    target = "".join(markers)
    if prompt_text.count(target) != 1:
        raise ValueError("frozen video prompt must contain exactly one placeholder")
    # vLLM 0.24 replaces the complete triplet with per-frame timestamp/vision
    # groups, whereas HF replaces only ``media_token`` and retains this outer
    # pair. The wrapper leaves one outer pair after vLLM's replacement.
    replacement = f"{vision_start_token}{target}{vision_end_token}"
    return prompt_text.replace(target, replacement, 1)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _string(value: object, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{path} must be {qualifier}")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if len(digest) != SHA256_HEX_LENGTH or any(
        character not in LOWERCASE_HEX_DIGITS for character in digest
    ):
        raise ValueError(f"{path} must be a lowercase SHA256 digest")
    return digest


def frozen_semantic_runtime(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    """Project only cross-framework semantic knobs from the Prism evaluator."""

    runtime = _mapping(evaluator.get("runtime"), "evaluator.runtime")
    return {
        "max_model_len": runtime["max_model_len"],
        "image_max_pixels": runtime["image_max_pixels"],
        "video_max_pixels": runtime["video_max_pixels"],
        "video_frames": runtime["video_frames"],
        "sampling": runtime["sampling"],
    }


def equal_kv_token_capacity(evaluator: Mapping[str, Any]) -> int:
    """Return the frozen logical KV slots shared by Prism and vLLM."""

    runtime = _mapping(evaluator.get("runtime"), "evaluator.runtime")
    page_size = _integer(
        runtime.get("kv_cache_page_size"),
        "evaluator.runtime.kv_cache_page_size",
        minimum=1,
    )
    blocks = _integer(
        runtime.get("num_kv_cache_blocks"),
        "evaluator.runtime.num_kv_cache_blocks",
        minimum=1,
    )
    return page_size * blocks


def expected_vllm_num_blocks(evaluator: Mapping[str, Any]) -> int:
    """Map Prism's frozen token capacity to vLLM's native 16-token blocks."""

    capacity = equal_kv_token_capacity(evaluator)
    if capacity % VLLM_KV_BLOCK_SIZE:
        raise ValueError("frozen KV token capacity is not divisible by vLLM block size")
    return capacity // VLLM_KV_BLOCK_SIZE


def _text_model_dimensions(model_config: Mapping[str, Any]) -> dict[str, int]:
    text = _mapping(model_config.get("text_config"), "model_config.text_config")
    dimensions = {
        "num_layers": _integer(
            text.get("num_hidden_layers"),
            "model_config.text_config.num_hidden_layers",
            minimum=1,
        ),
        "num_kv_heads": _integer(
            text.get("num_key_value_heads"),
            "model_config.text_config.num_key_value_heads",
            minimum=1,
        ),
        "head_size": _integer(
            text.get("head_dim"),
            "model_config.text_config.head_dim",
            minimum=1,
        ),
    }
    return dimensions


def _vision_model_depth(model_config: Mapping[str, Any]) -> int:
    vision = _mapping(model_config.get("vision_config"), "model_config.vision_config")
    return _integer(
        vision.get("depth"),
        "model_config.vision_config.depth",
        minimum=1,
    )


def expected_vllm_kv_cache(
    *,
    mode: str,
    evaluator: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute vLLM's exact allocated KV tensor layout from frozen inputs."""

    mode_spec = VLLM_QUALITY_MODES.get(mode)
    if mode_spec is None:
        raise ValueError(f"unsupported vLLM quality mode: {mode!r}")
    dimensions = _text_model_dimensions(model_config)
    num_blocks = expected_vllm_num_blocks(evaluator)
    block_size = VLLM_KV_BLOCK_SIZE
    slots = num_blocks * block_size
    payload_elements = (
        2 * dimensions["num_layers"] * slots * dimensions["num_kv_heads"] * dimensions["head_size"]
    )
    payload_bytes = payload_elements * mode_spec["payload_element_bytes"]
    scale_elements = (
        mode_spec["storage_scale_elements_per_token_head"]
        * dimensions["num_layers"]
        * slots
        * dimensions["num_kv_heads"]
    )
    scale_bytes = scale_elements * 4
    total_bytes = payload_bytes + scale_bytes
    scale_pad = (
        4 // mode_spec["payload_element_bytes"]
        if mode_spec["storage_scale_elements_per_token_head"]
        else 0
    )
    raw_shape = [
        num_blocks,
        2,
        block_size,
        dimensions["num_kv_heads"],
        dimensions["head_size"] + scale_pad,
    ]
    per_layer_bytes = math.prod(raw_shape) * mode_spec["payload_element_bytes"]
    if per_layer_bytes * dimensions["num_layers"] != total_bytes:
        raise AssertionError("vLLM inline scale layout accounting is inconsistent")
    return {
        "framework_mode": mode,
        "kv_cache_dtype": mode_spec["kv_cache_dtype"],
        "payload_dtype": mode_spec["payload_dtype"],
        "scale_dtype": mode_spec["scale_dtype"],
        **dimensions,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "logical_token_capacity": slots,
        "raw_tensor_shape_per_layer": raw_shape,
        "raw_tensor_bytes_per_layer": per_layer_bytes,
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
    }


def validate_vllm_kv_cache(
    cache: Mapping[str, Any],
    *,
    mode: str,
    evaluator: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    """Validate formulas against vLLM's observed allocation descriptors."""

    expected = expected_vllm_kv_cache(
        mode=mode,
        evaluator=evaluator,
        model_config=model_config,
    )
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(
                f"external KV cache {key} differs from frozen vLLM layout: "
                f"{cache.get(key)!r} != {value!r}"
            )
    _validate_reported_vllm_kv_tensors(cache, expected)
    _validate_vllm_kv_metadata_accounting(cache)


def _validate_reported_vllm_kv_tensors(
    cache: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if cache.get("accounting_scope") != (
        "allocated_gpu_kv_tensors_including_inline_per_token_head_scales"
    ):
        raise ValueError("external KV cache accounting scope is unsupported")
    tensor_sizes = _list(
        cache.get("reported_tensor_sizes"),
        "artifact.kv_cache.reported_tensor_sizes",
    )
    if tensor_sizes != [expected["raw_tensor_bytes_per_layer"]] * expected["num_layers"]:
        raise ValueError("external KV tensor sizes differ from the expected layout")
    if cache.get("reported_tensor_count") != len(tensor_sizes):
        raise ValueError("external KV tensor count is inconsistent")
    if cache.get("reported_total_bytes") != sum(tensor_sizes):
        raise ValueError("external reported KV bytes do not sum to tensor sizes")
    if cache.get("reported_total_bytes") != expected["total_bytes"]:
        raise ValueError("external reported KV bytes differ from formula accounting")


def _validate_vllm_kv_metadata_accounting(cache: Mapping[str, Any]) -> None:
    metadata = _mapping(
        cache.get("metadata_accounting"),
        "artifact.kv_cache.metadata_accounting",
    )
    for key in ("gpu_block_table_bytes", "cpu_block_table_bytes"):
        _integer(metadata.get(key), f"artifact.kv_cache.metadata_accounting.{key}")
    complete = metadata.get("complete_for_cross_framework_physical_claim")
    if not isinstance(complete, bool):
        raise ValueError("artifact.kv_cache.metadata_accounting completion flag must be bool")
    allocator_bytes = metadata.get("allocator_structural_bytes")
    if complete:
        _integer(
            allocator_bytes,
            "artifact.kv_cache.metadata_accounting.allocator_structural_bytes",
        )
    elif allocator_bytes is not None:
        raise ValueError(
            "incomplete external allocator accounting must be null, not a measured byte value"
        )
    _string(
        metadata.get("scope"),
        "artifact.kv_cache.metadata_accounting.scope",
    )


def vllm_framework_runtime(
    *,
    evaluator: Mapping[str, Any],
    dataset_id: str,
    mode: str,
    max_media_per_prompt: int,
) -> dict[str, Any]:
    if dataset_id not in QUALITY_DATASETS:
        raise ValueError(f"unsupported external quality dataset: {dataset_id!r}")
    mode_spec = VLLM_QUALITY_MODES.get(mode)
    if mode_spec is None:
        raise ValueError(f"unsupported vLLM quality mode: {mode!r}")
    if (
        isinstance(max_media_per_prompt, bool)
        or not isinstance(max_media_per_prompt, int)
        or max_media_per_prompt < 1
    ):
        raise ValueError("max_media_per_prompt must be an integer >= 1")
    runtime = _mapping(evaluator.get("runtime"), "evaluator.runtime")
    modality = "video" if dataset_id == "mvbench_test" else "image"
    pixel_budget_key = "video_max_pixels" if modality == "video" else "image_max_pixels"
    return {
        "execution_backend": "eager",
        "tensor_parallel_size": 1,
        "max_model_len": runtime["max_model_len"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "max_num_seqs": 1,
        "gpu_memory_utilization": runtime["gpu_memory_utilization"],
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "seed": 0,
        "block_size": VLLM_KV_BLOCK_SIZE,
        "num_gpu_blocks_override": expected_vllm_num_blocks(evaluator),
        "logical_kv_token_capacity": equal_kv_token_capacity(evaluator),
        "kv_cache_dtype": mode_spec["kv_cache_dtype"],
        "attention_backend": VLLM_ATTENTION_BACKEND,
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {modality: max_media_per_prompt},
        "mm_processor_kwargs": {
            "size": {
                "shortest_edge": FROZEN_QWEN3_VL_PROCESSOR_MIN_PIXELS[modality],
                "longest_edge": runtime[pixel_budget_key],
            }
        },
        "prompt_adapter": VLLM_PROMPT_ADAPTERS[modality],
    }


def _max_media_for_samples(
    samples: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
) -> int:
    references = {record["sample_id"]: record for record in reference_records}
    counts = []
    for sample in samples:
        record = references.get(sample["sample_id"])
        if record is None:
            raise ValueError(f"external sample {sample['sample_id']!r} has no reference record")
        media = record.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError("external reference record has no media list")
        counts.append(len(media))
    return max(counts)


def validate_external_quality_artifact(
    artifact: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model_config: Mapping[str, Any],
    reference_records: Sequence[Mapping[str, Any]],
    require_headline: bool = False,
) -> None:
    """Validate one completed vLLM artifact without trusting stored scores."""

    _validate_external_protocol_binding(evaluator, protocol)
    artifact = _mapping(artifact, "artifact")
    _validate_external_artifact_header(artifact)
    context = _validate_external_run_contract(
        artifact,
        evaluator=evaluator,
        model_config=model_config,
        require_headline=require_headline,
    )
    _validate_external_harness_identity(context.contract, context.headline)
    _validate_external_environment(artifact, context.contract)
    _validate_external_execution_evidence(
        artifact,
        contract=context.contract,
        model_config=model_config,
    )
    samples = _validate_external_framework_runtime(
        artifact,
        contract=context.contract,
        evaluator=evaluator,
        dataset_id=context.dataset_id,
        mode=context.mode,
        reference_records=reference_records,
    )

    validate_vllm_kv_cache(
        _mapping(artifact.get("kv_cache"), "artifact.kv_cache"),
        mode=context.mode,
        evaluator=evaluator,
        model_config=model_config,
    )
    sample_validation = validate_quality_samples(
        samples,
        dataset_id=context.dataset_id,
        runtime=evaluator["runtime"],
        dataset_evaluator=context.dataset_evaluator,
        reference_records=reference_records,
        path_prefix="external.samples",
    )
    _validate_external_sample_inputs(samples, context.dataset_id)
    _validate_external_selection(
        artifact,
        contract=context.contract,
        samples=samples,
        sample_validation=sample_validation,
    )
    _validate_external_materialization(artifact, context.contract)


@dataclass(frozen=True, slots=True)
class _ExternalRunContext:
    contract: Mapping[str, Any]
    dataset_id: str
    mode: str
    headline: bool
    dataset_evaluator: Mapping[str, Any]


def _validate_external_protocol_binding(
    evaluator: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    validate_p9_quality_protocol(protocol)
    if evaluator.get("quality_protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("evaluator references a different quality protocol")


def _validate_external_artifact_header(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != EXTERNAL_QUALITY_SCHEMA_VERSION:
        raise ValueError("external artifact has unsupported schema_version")
    if artifact.get("record_type") != EXTERNAL_QUALITY_RECORD_TYPE:
        raise ValueError("external artifact has unsupported record_type")
    if artifact.get("status") != "complete":
        raise ValueError("external quality artifact must be complete")


def _validate_external_run_contract(
    artifact: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    model_config: Mapping[str, Any],
    require_headline: bool,
) -> _ExternalRunContext:
    contract = _mapping(artifact.get("run_contract"), "artifact.run_contract")
    _validate_external_contract_hashes(artifact, contract, evaluator)
    dataset_id, headline = _validate_external_scope(
        artifact,
        contract,
        require_headline=require_headline,
    )
    mode = _string(contract.get("framework_mode"), "artifact.run_contract.framework_mode")
    if mode not in VLLM_QUALITY_MODES:
        raise ValueError(f"unsupported external framework mode: {mode!r}")
    dataset_evaluator = _validate_external_model_contract(
        contract,
        evaluator=evaluator,
        model_config=model_config,
        dataset_id=dataset_id,
    )
    return _ExternalRunContext(contract, dataset_id, mode, headline, dataset_evaluator)


def _validate_external_contract_hashes(
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> None:
    identity = _sha256(
        artifact.get("run_identity_sha256"),
        "artifact.run_identity_sha256",
    )
    if identity != canonical_json_sha256(contract):
        raise ValueError("external run identity does not match its contract")
    if contract.get("evaluator_sha256") != canonical_json_sha256(evaluator):
        raise ValueError("external artifact references a different evaluator")


def _validate_external_scope(
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    require_headline: bool,
) -> tuple[str, bool]:
    dataset_id = _string(contract.get("dataset"), "artifact.run_contract.dataset")
    if dataset_id not in QUALITY_DATASETS:
        raise ValueError(f"unsupported external quality dataset: {dataset_id!r}")
    subset = _string(contract.get("subset"), "artifact.run_contract.subset")
    if subset not in ("development", "final"):
        raise ValueError("external quality subset is unsupported")
    scope = _string(contract.get("scope"), "artifact.run_contract.scope")
    formal_scope = f"formal_{subset}"
    if scope not in ("smoke_not_quality_gate", formal_scope):
        raise ValueError("external artifact scope is inconsistent with subset")
    headline = artifact.get("headline_eligible")
    if not isinstance(headline, bool) or headline != (scope == formal_scope):
        raise ValueError("external headline eligibility is inconsistent with scope")
    if require_headline and not headline:
        raise ValueError("external comparison requires formal headline evidence")
    return dataset_id, headline


def _validate_external_model_contract(
    contract: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    model_config: Mapping[str, Any],
    dataset_id: str,
) -> Mapping[str, Any]:
    if contract.get("framework") != VLLM_FRAMEWORK:
        raise ValueError("external run contract framework must be vLLM")
    model = _string(contract.get("model"), "artifact.run_contract.model")
    if not model.startswith("/"):
        raise ValueError("external model path must be absolute")
    model_revision = evaluator["model"]["revision"]
    if contract.get("model_revision") != model_revision:
        raise ValueError("external model revision differs from evaluator")
    if contract.get("processor_revision") != model_revision:
        raise ValueError("external processor revision differs from evaluator")
    if contract.get("model_config_canonical_sha256") != canonical_json_sha256(model_config):
        raise ValueError("external model config identity is inconsistent")
    if contract.get("semantic_runtime") != frozen_semantic_runtime(evaluator):
        raise ValueError("external semantic runtime differs from evaluator")
    dataset_evaluator = _mapping(
        evaluator["datasets"][dataset_id],
        f"evaluator.datasets.{dataset_id}",
    )
    if contract.get("dataset_evaluator") != dataset_evaluator:
        raise ValueError("external dataset evaluator differs from frozen evaluator")
    return dataset_evaluator


def _validate_external_harness_identity(
    contract: Mapping[str, Any],
    headline: bool,
) -> None:
    harness_git = _mapping(
        contract.get("harness_git"),
        "artifact.run_contract.harness_git",
    )
    commit = _string(
        harness_git.get("commit"),
        "artifact.run_contract.harness_git.commit",
    )
    if len(commit) != GIT_SHA1_HEX_LENGTH or any(
        char not in LOWERCASE_HEX_DIGITS for char in commit
    ):
        raise ValueError("external harness commit must be a full Git revision")
    dirty = harness_git.get("dirty")
    if not isinstance(dirty, bool) or (headline and dirty):
        raise ValueError("formal external evidence requires a clean harness")


def _validate_external_environment(
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    environment = _mapping(artifact.get("environment"), "artifact.environment")
    _validate_external_software_identity(environment)
    _validate_external_hardware_identity(environment)
    _sha256(
        environment.get("framework_distribution_record_sha256"),
        "artifact.environment.framework_distribution_record_sha256",
    )
    _validate_external_implementation_files(environment)
    contract_environment = _mapping(
        contract.get("environment"),
        "artifact.run_contract.environment",
    )
    if environment != contract_environment:
        raise ValueError("external environment differs from its frozen run contract")


def _validate_external_software_identity(environment: Mapping[str, Any]) -> None:
    if environment.get("framework") != VLLM_FRAMEWORK:
        raise ValueError("external artifact framework must be vllm")
    if environment.get("framework_version") != VLLM_FRAMEWORK_VERSION:
        raise ValueError("external vLLM framework version differs from frozen cell")
    if environment.get("framework_distribution_commit") != VLLM_DISTRIBUTION_COMMIT:
        raise ValueError("external vLLM distribution commit differs from frozen cell")
    if environment.get("transformers") != VLLM_TRANSFORMERS_VERSION:
        raise ValueError("external Transformers version differs from frozen cell")
    for key in (
        "framework_version",
        "framework_distribution_commit",
        "python",
        "torch",
        "transformers",
        "cuda",
        "gpu",
    ):
        _string(environment.get(key), f"artifact.environment.{key}")


def _validate_external_hardware_identity(environment: Mapping[str, Any]) -> None:
    gpu_uuid = _string(environment.get("gpu_uuid"), "artifact.environment.gpu_uuid")
    if gpu_uuid == "unknown" or not gpu_uuid.startswith("GPU-"):
        raise ValueError("external GPU UUID is unavailable or malformed")
    driver = _string(environment.get("driver"), "artifact.environment.driver")
    if driver == "unknown":
        raise ValueError("external GPU driver identity is unavailable")
    compute_capability = _string(
        environment.get("compute_capability"),
        "artifact.environment.compute_capability",
    )
    capability_parts = compute_capability.split(".")
    if len(capability_parts) != COMPUTE_CAPABILITY_COMPONENT_COUNT or not all(
        part.isdigit() for part in capability_parts
    ):
        raise ValueError("external GPU compute capability is malformed")
    _integer(
        environment.get("total_memory_bytes"),
        "artifact.environment.total_memory_bytes",
        minimum=1,
    )


def _validate_external_implementation_files(environment: Mapping[str, Any]) -> None:
    implementation_files = _mapping(
        environment.get("framework_implementation_files"),
        "artifact.environment.framework_implementation_files",
    )
    if not implementation_files:
        raise ValueError("external framework implementation identity is empty")
    _validate_digest_file_set(
        implementation_files,
        expected_paths=VLLM_IMPLEMENTATION_FILES,
        path="artifact.environment.framework_implementation_files",
        mismatch_message="external vLLM implementation file set is incomplete",
    )
    processor_files = _mapping(
        environment.get("transformers_processor_implementation_files"),
        "artifact.environment.transformers_processor_implementation_files",
    )
    _validate_digest_file_set(
        processor_files,
        expected_paths=TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES,
        path="artifact.environment.transformers_processor_implementation_files",
        mismatch_message="external Transformers processor file set is incomplete",
    )


def _validate_digest_file_set(
    files: Mapping[str, Any],
    *,
    expected_paths: Sequence[str],
    path: str,
    mismatch_message: str,
) -> None:
    if set(files) != set(expected_paths):
        raise ValueError(mismatch_message)
    for file_path, digest in files.items():
        _string(file_path, f"{path}.path")
        _sha256(digest, f"{path}[{file_path}]")


def _validate_external_execution_evidence(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    execution = _mapping(
        artifact.get("execution_evidence"),
        "artifact.execution_evidence",
    )
    if execution.get("required_environment") != dict(VLLM_REQUIRED_ENVIRONMENT):
        raise ValueError("external required environment evidence is inconsistent")
    _validate_external_language_backends(execution, model_config)
    _validate_external_vision_backend(execution, model_config)
    framework_runtime = _mapping(
        contract.get("framework_runtime"),
        "artifact.run_contract.framework_runtime",
    )
    if framework_runtime.get("attention_backend") != VLLM_ATTENTION_BACKEND:
        raise ValueError("external requested and effective language backends differ")


def _validate_external_language_backends(
    execution: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    language_backends = _list(
        execution.get("language_attention_backends"),
        "artifact.execution_evidence.language_attention_backends",
    )
    if not language_backends:
        raise ValueError("external language attention backend evidence is empty")
    language_layers: list[str] = []
    for index, raw_backend in enumerate(language_backends):
        language_layers.extend(_validate_external_language_backend(raw_backend, index))
    expected_layers = _text_model_dimensions(model_config)["num_layers"]
    if len(language_layers) != expected_layers:
        raise ValueError("external language attention layer count is inconsistent")
    if len(language_layers) != len(set(language_layers)):
        raise ValueError("external language attention layer evidence contains duplicates")


def _validate_external_language_backend(raw_backend: object, index: int) -> list[str]:
    path = f"artifact.execution_evidence.language_attention_backends[{index}]"
    backend = _mapping(raw_backend, path)
    if backend.get("name") != VLLM_ATTENTION_BACKEND:
        raise ValueError("external effective language attention backend is unsupported")
    _string(backend.get("backend_class"), f"{path}.backend_class")
    _integer(backend.get("kv_cache_group_id"), f"{path}.kv_cache_group_id")
    layer_names = _list(backend.get("layer_names"), f"{path}.layer_names")
    if not layer_names:
        raise ValueError("external language attention group has no layers")
    return [_string(layer, f"{path}.layer_names") for layer in layer_names]


def _validate_external_vision_backend(
    execution: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    path = "artifact.execution_evidence.vision_attention_backend"
    vision_backend = _mapping(execution.get("vision_attention_backend"), path)
    if vision_backend.get("name") != VLLM_VISION_ATTENTION_BACKEND:
        raise ValueError("external effective vision attention backend is unsupported")
    _string(vision_backend.get("selector_class"), f"{path}.selector_class")
    if vision_backend.get("layer_count") != _vision_model_depth(model_config):
        raise ValueError("external vision attention layer count is inconsistent")


def _validate_external_framework_runtime(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    dataset_id: str,
    mode: str,
    reference_records: Sequence[Mapping[str, Any]],
) -> list[Any]:
    samples = _list(artifact.get("samples"), "artifact.samples")
    if not samples:
        raise ValueError("external artifact samples must not be empty")
    expected_runtime = vllm_framework_runtime(
        evaluator=evaluator,
        dataset_id=dataset_id,
        mode=mode,
        max_media_per_prompt=_max_media_for_samples(samples, reference_records),
    )
    if contract.get("framework_runtime") != expected_runtime:
        raise ValueError("external framework runtime differs from frozen vLLM cell")
    return samples


def _validate_external_sample_inputs(samples: list[Any], dataset_id: str) -> None:
    for index, sample in enumerate(samples):
        sample_mapping = _mapping(sample, f"external.samples[{index}]")
        _validate_external_sample_input(sample_mapping, index)
        if dataset_id == "mvbench_test":
            _validate_video_frame_bridge(sample_mapping, index)


def _validate_external_sample_input(sample: Mapping[str, Any], index: int) -> None:
    framework_input = _mapping(
        sample.get("framework_input"),
        f"external.samples[{index}].framework_input",
    )
    identity = _mapping(sample.get("input"), f"external.samples[{index}].input")
    if framework_input.get("prompt_token_count") != identity.get("prompt_token_count"):
        raise ValueError("vLLM request prompt token count differs from preflight")
    if framework_input.get("prompt_token_ids_sha256") != identity.get("prompt_token_ids_sha256"):
        raise ValueError("vLLM request prompt token IDs differ from preflight")
    if framework_input.get("matches_prepared_semantic_identity") is not True:
        raise ValueError("vLLM request did not prove exact prepared input identity")


def _validate_video_frame_bridge(sample: Mapping[str, Any], index: int) -> None:
    path = f"external.samples[{index}].video_frame_bridge"
    bridge = _mapping(sample.get("video_frame_bridge"), path)
    for key in ("helper_bundle_sha256", "bundle_content_sha256"):
        _sha256(bridge.get(key), f"{path}.{key}")


def _validate_external_selection(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    samples: list[Any],
    sample_validation: Mapping[str, Any],
) -> None:
    selection = _mapping(artifact.get("selection"), "artifact.selection")
    sample_ids = sample_validation["sample_ids"]
    if selection.get("eligible_run_samples") != len(samples):
        raise ValueError("external eligible sample count is inconsistent")
    eligible_sha256 = selected_ids_sha256(sample_ids)
    if selection.get("eligible_run_ids_sha256") != eligible_sha256:
        raise ValueError("external eligible sample ID hash is inconsistent")
    if contract.get("eligible_sample_ids_sha256") != eligible_sha256:
        raise ValueError("external contract sample ID hash is inconsistent")
    _integer(
        selection.get("selected_contract_samples"),
        "artifact.selection.selected_contract_samples",
        minimum=len(samples),
    )
    _sha256(
        selection.get("selected_contract_ids_sha256"),
        "artifact.selection.selected_contract_ids_sha256",
    )
    _list(selection.get("protocol_exclusions"), "artifact.selection.protocol_exclusions")
    if artifact.get("completed_samples") != len(samples):
        raise ValueError("external completed sample count is inconsistent")
    if artifact.get("aggregate") != sample_validation["aggregate"]:
        raise ValueError("external aggregate differs from independently recomputed scores")


def _validate_external_materialization(
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    materialization = _mapping(
        artifact.get("materialization_verification"),
        "artifact.materialization_verification",
    )
    if materialization.get("status") != "PASS":
        raise ValueError("external materialization verification did not pass")
    manifest_sha256 = _sha256(
        materialization.get("manifest_sha256"),
        "artifact.materialization_verification.manifest_sha256",
    )
    if manifest_sha256 != contract.get("materialization_manifest_sha256"):
        raise ValueError("external materialization identity is inconsistent")


def compare_prism_external_quality(
    prism: Mapping[str, Any],
    external: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model_config: Mapping[str, Any],
    reference_records: Sequence[Mapping[str, Any]],
    require_headline: bool = False,
) -> dict[str, Any]:
    """Build a semantic-exact Prism/vLLM quality and KV-pool comparison."""

    validate_quality_artifact(
        prism,
        evaluator=evaluator,
        require_headline=require_headline,
        reference_records=reference_records,
    )
    validate_external_quality_artifact(
        external,
        evaluator=evaluator,
        protocol=protocol,
        model_config=model_config,
        reference_records=reference_records,
        require_headline=require_headline,
    )
    prism_contract = prism["run_contract"]
    external_contract = external["run_contract"]
    for key in (
        "dataset",
        "subset",
        "scope",
        "model",
        "model_revision",
        "evaluator_sha256",
        "materialization_manifest_sha256",
        "eligible_sample_ids_sha256",
        "dataset_evaluator",
    ):
        if prism_contract.get(key) != external_contract.get(key):
            raise ValueError(f"Prism/vLLM semantic contract differs at {key}")
    if prism["selection"] != external["selection"]:
        raise ValueError("Prism/vLLM frozen selections differ")
    sample_comparison = compare_quality_sample_sets(
        prism_contract["dataset"],
        prism["samples"],
        external["samples"],
        protocol=protocol,
    )
    formal = bool(prism["headline_eligible"] and external["headline_eligible"])
    if require_headline and not formal:
        raise ValueError("Prism/vLLM comparison requires formal artifacts")

    prism_cache = prism["kv_cache"]
    external_cache = external["kv_cache"]
    prism_capacity = prism_cache["payload_shape"][2] * prism_cache["payload_shape"][3]
    if prism_capacity != external_cache["logical_token_capacity"]:
        raise ValueError("Prism/vLLM logical KV capacities differ")
    metadata_complete = bool(
        external_cache["metadata_accounting"]["complete_for_cross_framework_physical_claim"]
    )
    full_physical_comparable = False
    physical_limitations = []
    if not metadata_complete:
        physical_limitations.append("vllm_metadata_accounting_incomplete")
    # Schema-v1 Prism quality artifacts freeze the allocated KV pool only.
    physical_limitations.append("prism_page_table_allocator_accounting_not_recorded")

    all_required = sample_comparison["all_required_metrics_pass"]
    decision = ("PASS" if all_required else "FAIL") if formal else "SMOKE_ONLY"
    return {
        "schema_version": EXTERNAL_QUALITY_COMPARISON_SCHEMA_VERSION,
        "record_type": EXTERNAL_QUALITY_COMPARISON_RECORD_TYPE,
        "validation_status": "PASS",
        "dataset": prism_contract["dataset"],
        "subset": prism_contract["subset"],
        "scope": prism_contract["scope"],
        "prism_mode": prism_contract["mode"],
        "external_framework": VLLM_FRAMEWORK,
        "external_mode": external_contract["framework_mode"],
        "samples": len(prism["samples"]),
        "protocol_sha256": canonical_json_sha256(protocol),
        "evaluator_sha256": canonical_json_sha256(evaluator),
        "prism_artifact_sha256": canonical_json_sha256(prism),
        "external_artifact_sha256": canonical_json_sha256(external),
        "paired_input_identity_sha256": sample_comparison["paired_input_identity_sha256"],
        "semantic_input_exact": True,
        "reference_scores_recomputed": True,
        "metrics": sample_comparison["metrics"],
        "diagnostics": sample_comparison["diagnostics"],
        "kv_cache": {
            "accounting_scope": "allocated_kv_pool_payload_plus_scales",
            "logical_token_capacity": prism_capacity,
            "prism_payload_bytes": prism_cache["payload_bytes"],
            "prism_scale_bytes": prism_cache["scale_bytes"],
            "prism_total_bytes": prism_cache["total_bytes"],
            "external_payload_bytes": external_cache["payload_bytes"],
            "external_scale_bytes": external_cache["scale_bytes"],
            "external_total_bytes": external_cache["total_bytes"],
            "external_to_prism_total_ratio": (
                external_cache["total_bytes"] / prism_cache["total_bytes"]
            ),
            "full_physical_comparable": full_physical_comparable,
            "full_physical_claim_limitations": physical_limitations,
        },
        "all_required_metrics_pass": all_required,
        "formal_evidence": formal,
        "headline_eligible": formal and all_required,
        "decision": decision,
    }
