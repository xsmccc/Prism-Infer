"""P9 external vLLM quality/KV contract tests."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_external_quality import (
    EXTERNAL_QUALITY_RECORD_TYPE,
    EXTERNAL_QUALITY_SCHEMA_VERSION,
    TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES,
    VLLM_IMPLEMENTATION_FILES,
    VLLM_REQUIRED_ENVIRONMENT,
    adapt_vllm_prompt_text,
    compare_prism_external_quality,
    expected_vllm_kv_cache,
    validate_external_quality_artifact,
    vllm_framework_runtime,
)
from prism_infer.analysis.p9_quality_materialization import selected_ids_sha256
from prism_infer.analysis.p9_quality_metrics import (
    aggregate_quality_predictions,
    score_quality_prediction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = json.loads((REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json").read_text())
PROTOCOL = json.loads((REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json").read_text())
MODEL_CONFIG = {
    "model_type": "qwen3_vl",
    "text_config": {
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "vision_config": {"depth": 32},
}


def _sample(score: float = 1.0) -> dict[str, object]:
    return {
        "sample_id": "doc-1",
        "input": {
            "source_prompt_sha256": "1" * 64,
            "chat_prompt_sha256": "2" * 64,
            "prompt_token_count": 5,
            "prompt_token_ids_sha256": "3" * 64,
            "media_sha256": ["4" * 64],
            "modality": "image",
            "image_grid_thw": [[1, 2, 2]],
            "visual_placeholder_tokens": 1,
        },
        "framework_input": {
            "prompt_token_count": 5,
            "prompt_token_ids_sha256": "3" * 64,
            "matches_prepared_semantic_identity": True,
        },
        "raw_prediction": "answer",
        "decoded_with_special_tokens": "answer<|im_end|>",
        "output_token_ids": [42, 151645],
        "framework_output_text": "answer",
        "finish_reason": "stop",
        "stop_reason": 151645,
        "score": {"target": ["answer"], "anls": score},
    }


def _reference_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "doc-1",
            "answers": ["answer"],
            "media": [{"sha256": "4" * 64}],
        }
    ]


def _mvbench_artifact_and_records() -> tuple[dict[str, object], list[dict[str, object]]]:
    artifact = _external_artifact()
    sample = artifact["samples"][0]
    sample_id = "mv-1"
    sample["sample_id"] = sample_id
    sample["input"].pop("image_grid_thw")
    sample["input"].update(
        {
            "modality": "video",
            "video_grid_thw": [[8, 2, 2]],
            "visual_placeholder_tokens": 8,
        }
    )
    record = {
        "sample_id": sample_id,
        "question": "What happens?",
        "candidates": ["yes", "no"],
        "answer_index": 0,
        "task": "test_task",
        "media": [{"sha256": "4" * 64}],
    }
    sample.update(
        {
            "raw_prediction": "A",
            "decoded_with_special_tokens": "A<|im_end|>",
            "framework_output_text": "A",
            "score": score_quality_prediction("mvbench_test", record, "A"),
            "task": record["task"],
            "video_sampling": {
                "source_kind": "video_file",
                "decoder": deepcopy(
                    EVALUATOR["datasets"]["mvbench_test"]["video_sampling"]["video_file_decoder"]
                ),
                "fps": 3.0,
                "source_frame_count": 32,
                "frame_access": {
                    "method": "random_seek",
                    "reported_frame_count": 32,
                    "fallback_trigger": None,
                },
                "sampled_indices": list(range(EVALUATOR["runtime"]["video_frames"])),
                "sampled_rgb_identity_sha256": "9" * 64,
            },
            "video_frame_bridge": {
                "helper_bundle_sha256": "a" * 64,
                "bundle_content_sha256": "b" * 64,
            },
        }
    )
    sample_ids_sha256 = selected_ids_sha256([sample_id])
    contract = artifact["run_contract"]
    contract.update(
        {
            "dataset": "mvbench_test",
            "eligible_sample_ids_sha256": sample_ids_sha256,
            "framework_runtime": vllm_framework_runtime(
                evaluator=EVALUATOR,
                dataset_id="mvbench_test",
                mode=contract["framework_mode"],
                max_media_per_prompt=1,
            ),
            "dataset_evaluator": EVALUATOR["datasets"]["mvbench_test"],
        }
    )
    artifact["selection"].update(
        {
            "selected_contract_ids_sha256": sample_ids_sha256,
            "eligible_run_ids_sha256": sample_ids_sha256,
        }
    )
    artifact["aggregate"] = aggregate_quality_predictions("mvbench_test", [sample])
    _rehash(artifact)
    return artifact, [record]


def _external_artifact(mode: str = "fp8_per_token_head") -> dict[str, object]:
    sample = _sample()
    sample_ids_sha256 = selected_ids_sha256(["doc-1"])
    manifest_sha256 = "5" * 64
    cache = expected_vllm_kv_cache(
        mode=mode,
        evaluator=EVALUATOR,
        model_config=MODEL_CONFIG,
    )
    cache.update(
        {
            "accounting_scope": ("allocated_gpu_kv_tensors_including_inline_per_token_head_scales"),
            "reported_tensor_count": cache["num_layers"],
            "reported_tensor_sizes": [cache["raw_tensor_bytes_per_layer"]] * cache["num_layers"],
            "reported_total_bytes": cache["total_bytes"],
            "metadata_accounting": {
                "scope": "test block table tensors; allocator graph pending",
                "gpu_block_table_bytes": 2048,
                "cpu_block_table_bytes": 2048,
                "allocator_structural_bytes": None,
                "complete_for_cross_framework_physical_claim": False,
            },
        }
    )
    environment = {
        "framework": "vllm",
        "framework_version": "0.24.0",
        "framework_distribution_commit": "gee0da84ab",
        "framework_distribution_record_sha256": "6" * 64,
        "framework_implementation_files": {path: "7" * 64 for path in VLLM_IMPLEMENTATION_FILES},
        "transformers_processor_implementation_files": {
            path: "8" * 64 for path in TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES
        },
        "python": "3.12.0",
        "torch": "2.11.0+cu130",
        "transformers": "5.13.0",
        "cuda": "13.0",
        "gpu": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "driver": "610.43.02",
        "compute_capability": "12.0",
        "total_memory_bytes": 33_711_521_792,
    }
    contract = {
        "dataset": "docvqa_validation",
        "subset": "development",
        "scope": "smoke_not_quality_gate",
        "framework": "vllm",
        "framework_mode": mode,
        "model": "/model",
        "model_revision": EVALUATOR["model"]["revision"],
        "processor_revision": EVALUATOR["model"]["revision"],
        "model_config_canonical_sha256": canonical_json_sha256(MODEL_CONFIG),
        "harness_git": {"commit": "a" * 40, "dirty": True},
        "environment": deepcopy(environment),
        "evaluator_sha256": canonical_json_sha256(EVALUATOR),
        "materialization_manifest_sha256": manifest_sha256,
        "eligible_sample_ids_sha256": sample_ids_sha256,
        "semantic_runtime": {
            key: EVALUATOR["runtime"][key]
            for key in (
                "max_model_len",
                "image_max_pixels",
                "video_max_pixels",
                "video_frames",
                "sampling",
            )
        },
        "framework_runtime": vllm_framework_runtime(
            evaluator=EVALUATOR,
            dataset_id="docvqa_validation",
            mode=mode,
            max_media_per_prompt=1,
        ),
        "dataset_evaluator": EVALUATOR["datasets"]["docvqa_validation"],
    }
    return {
        "schema_version": EXTERNAL_QUALITY_SCHEMA_VERSION,
        "record_type": EXTERNAL_QUALITY_RECORD_TYPE,
        "status": "complete",
        "headline_eligible": False,
        "run_contract": contract,
        "run_identity_sha256": canonical_json_sha256(contract),
        "selection": {
            "selected_contract_samples": 1,
            "selected_contract_ids_sha256": sample_ids_sha256,
            "eligible_run_samples": 1,
            "eligible_run_ids_sha256": sample_ids_sha256,
            "protocol_exclusions": [],
        },
        "materialization_verification": {
            "status": "PASS",
            "manifest_sha256": manifest_sha256,
        },
        "environment": environment,
        "execution_evidence": {
            "required_environment": dict(VLLM_REQUIRED_ENVIRONMENT),
            "language_attention_backends": [
                {
                    "name": "TRITON_ATTN",
                    "backend_class": (
                        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
                    ),
                    "kv_cache_group_id": 0,
                    "layer_names": [f"model.layers.{index}.self_attn" for index in range(36)],
                }
            ],
            "vision_attention_backend": {
                "name": "FLASH_ATTN",
                "selector_class": ("vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"),
                "layer_count": 32,
            },
        },
        "kv_cache": cache,
        "samples": [sample],
        "completed_samples": 1,
        "aggregate": aggregate_quality_predictions(
            "docvqa_validation",
            [sample],
        ),
    }


def _prism_artifact() -> dict[str, object]:
    sample = deepcopy(_sample())
    sample.pop("framework_input")
    sample.pop("framework_output_text")
    sample.pop("finish_reason")
    sample.pop("stop_reason")
    sample_ids_sha256 = selected_ids_sha256(["doc-1"])
    payload_shape = [2, 1, 40, 256, 1, 128]
    payload_bytes = math.prod(payload_shape) * 2
    contract = {
        "dataset": "docvqa_validation",
        "subset": "development",
        "scope": "smoke_not_quality_gate",
        "mode": "off",
        "model": "/model",
        "model_revision": EVALUATOR["model"]["revision"],
        "git": {"commit": "a" * 40, "dirty": True},
        "evaluator_sha256": canonical_json_sha256(EVALUATOR),
        "materialization_manifest_sha256": "5" * 64,
        "eligible_sample_ids_sha256": sample_ids_sha256,
        "runtime": EVALUATOR["runtime"],
        "dataset_evaluator": EVALUATOR["datasets"]["docvqa_validation"],
    }
    return {
        "schema_version": 1,
        "record_type": "p9_quality_predictions",
        "status": "complete",
        "headline_eligible": False,
        "run_contract": contract,
        "run_identity_sha256": canonical_json_sha256(contract),
        "selection": {
            "selected_contract_samples": 1,
            "selected_contract_ids_sha256": sample_ids_sha256,
            "eligible_run_samples": 1,
            "eligible_run_ids_sha256": sample_ids_sha256,
            "protocol_exclusions": [],
        },
        "materialization_verification": {
            "status": "PASS",
            "manifest_sha256": "5" * 64,
        },
        "environment": {"gpu": "GPU", "torch": "2", "cuda": "1"},
        "kv_cache": {
            "payload_dtype": "torch.bfloat16",
            "payload_shape": payload_shape,
            "scale_dtype": "none",
            "scale_shape": [],
            "payload_bytes": payload_bytes,
            "scale_bytes": 0,
            "total_bytes": payload_bytes,
        },
        "samples": [sample],
        "completed_samples": 1,
        "aggregate": aggregate_quality_predictions(
            "docvqa_validation",
            [sample],
        ),
    }


def _rehash(artifact: dict[str, object]) -> None:
    artifact["run_identity_sha256"] = canonical_json_sha256(artifact["run_contract"])


def test_vllm_cache_formulas_match_equal_capacity_reference_bytes() -> None:
    bf16 = expected_vllm_kv_cache(mode="bf16", evaluator=EVALUATOR, model_config=MODEL_CONFIG)
    fp8 = expected_vllm_kv_cache(
        mode="fp8_per_token_head",
        evaluator=EVALUATOR,
        model_config=MODEL_CONFIG,
    )

    assert bf16["logical_token_capacity"] == fp8["logical_token_capacity"] == 10240
    assert bf16["total_bytes"] == 1_509_949_440
    assert fp8["payload_bytes"] == 754_974_720
    assert fp8["scale_bytes"] == 23_592_960
    assert fp8["total_bytes"] == 778_567_680
    assert fp8["raw_tensor_shape_per_layer"] == [640, 2, 16, 8, 132]


def test_vllm_framework_runtime_freezes_processor_size_and_prompt_adapter() -> None:
    image = vllm_framework_runtime(
        evaluator=EVALUATOR,
        dataset_id="docvqa_validation",
        mode="bf16",
        max_media_per_prompt=1,
    )
    video = vllm_framework_runtime(
        evaluator=EVALUATOR,
        dataset_id="mvbench_test",
        mode="fp8_per_token_head",
        max_media_per_prompt=1,
    )

    assert image["mm_processor_kwargs"] == {
        "size": {"shortest_edge": 65_536, "longest_edge": 602_112}
    }
    assert image["prompt_adapter"] == "none"
    assert video["mm_processor_kwargs"] == {
        "size": {"shortest_edge": 4_096, "longest_edge": 802_816}
    }
    assert video["prompt_adapter"] == ("qwen3_vl_preserve_hf_outer_video_markers_v1")


@pytest.mark.parametrize(
    ("dataset_id", "mode", "max_media", "message"),
    [
        ("unknown", "bf16", 1, "unsupported external quality dataset"),
        ("docvqa_validation", "unknown", 1, "unsupported vLLM quality mode"),
        ("docvqa_validation", "bf16", 0, "max_media_per_prompt"),
        ("docvqa_validation", "bf16", True, "max_media_per_prompt"),
    ],
)
def test_vllm_framework_runtime_rejects_open_ended_cells(
    dataset_id: str,
    mode: str,
    max_media: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        vllm_framework_runtime(
            evaluator=EVALUATOR,
            dataset_id=dataset_id,
            mode=mode,
            max_media_per_prompt=max_media,
        )


def test_vllm_video_prompt_adapter_preserves_hf_outer_markers() -> None:
    prompt = "prefix<VS><VIDEO><VE>suffix"
    adapted = adapt_vllm_prompt_text(
        prompt,
        modality="video",
        vision_start_token="<VS>",
        media_token="<VIDEO>",
        vision_end_token="<VE>",
    )

    assert adapted == "prefix<VS><VS><VIDEO><VE><VE>suffix"
    assert (
        adapt_vllm_prompt_text(
            prompt,
            modality="image",
            vision_start_token="",
            media_token="",
            vision_end_token="",
        )
        == prompt
    )


def test_vllm_video_prompt_adapter_rejects_ambiguous_placeholder_count() -> None:
    with pytest.raises(ValueError, match="exactly one placeholder"):
        adapt_vllm_prompt_text(
            "<VS><VIDEO><VE><VS><VIDEO><VE>",
            modality="video",
            vision_start_token="<VS>",
            media_token="<VIDEO>",
            vision_end_token="<VE>",
        )


def test_external_artifact_validates_scores_runtime_and_physical_kv() -> None:
    validate_external_quality_artifact(
        _external_artifact(),
        evaluator=EVALUATOR,
        protocol=PROTOCOL,
        model_config=MODEL_CONFIG,
        reference_records=_reference_records(),
    )


@pytest.mark.parametrize(
    "bridge_key",
    ["helper_bundle_sha256", "bundle_content_sha256"],
)
def test_mvbench_artifact_validates_both_video_bridge_identities(
    bridge_key: str,
) -> None:
    artifact, records = _mvbench_artifact_and_records()
    validate_external_quality_artifact(
        artifact,
        evaluator=EVALUATOR,
        protocol=PROTOCOL,
        model_config=MODEL_CONFIG,
        reference_records=records,
    )

    artifact["samples"][0]["video_frame_bridge"][bridge_key] = "invalid"
    with pytest.raises(ValueError, match=bridge_key):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=records,
        )


def test_incomplete_allocator_accounting_must_be_explicitly_unmeasured() -> None:
    artifact = _external_artifact()
    artifact["kv_cache"]["metadata_accounting"]["allocator_structural_bytes"] = 0

    with pytest.raises(ValueError, match="allocator accounting must be null"):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=_reference_records(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda artifact: artifact["environment"].update(framework_version="0.25.0"),
            "framework version",
        ),
        (
            lambda artifact: artifact["environment"].update(
                framework_distribution_commit="gdeadbeef"
            ),
            "distribution commit",
        ),
        (
            lambda artifact: artifact["environment"].update(transformers="5.14.0"),
            "Transformers version",
        ),
        (
            lambda artifact: artifact["environment"].pop("gpu_uuid"),
            "environment.gpu_uuid",
        ),
        (
            lambda artifact: artifact["environment"].update(compute_capability="sm_120"),
            "compute capability",
        ),
        (
            lambda artifact: artifact["environment"].update(total_memory_bytes=0),
            "total_memory_bytes",
        ),
        (
            lambda artifact: artifact["environment"]["framework_implementation_files"].pop(
                VLLM_IMPLEMENTATION_FILES[0]
            ),
            "vLLM implementation file set",
        ),
        (
            lambda artifact: artifact["environment"][
                "transformers_processor_implementation_files"
            ].pop(TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES[0]),
            "Transformers processor file set",
        ),
        (
            lambda artifact: artifact["environment"].update(driver="611.00"),
            "environment differs from its frozen run contract",
        ),
    ],
)
def test_external_artifact_requires_hardware_and_source_identity(
    mutation,
    message: str,
) -> None:
    artifact = _external_artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=_reference_records(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact["kv_cache"].update(total_bytes=1), "total_bytes"),
        (
            lambda artifact: artifact["samples"][0]["framework_input"].update(
                prompt_token_ids_sha256="8" * 64
            ),
            "prompt token IDs",
        ),
        (
            lambda artifact: artifact["samples"][0]["score"].update(anls=0.0),
            "independently recomputed score",
        ),
        (
            lambda artifact: artifact["execution_evidence"]["vision_attention_backend"].update(
                name="TORCH_SDPA"
            ),
            "vision attention backend",
        ),
    ],
)
def test_external_artifact_rejects_tampering(mutation, message: str) -> None:
    artifact = _external_artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=_reference_records(),
        )


def test_external_artifact_rejects_wrong_framework_mode_contract() -> None:
    artifact = _external_artifact()
    artifact["run_contract"]["framework_mode"] = "bf16"
    _rehash(artifact)
    with pytest.raises(ValueError, match="framework runtime"):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=_reference_records(),
        )


def test_external_artifact_rejects_wrong_framework_contract() -> None:
    artifact = _external_artifact()
    artifact["run_contract"]["framework"] = "sglang"
    _rehash(artifact)
    with pytest.raises(ValueError, match="run contract framework"):
        validate_external_quality_artifact(
            artifact,
            evaluator=EVALUATOR,
            protocol=PROTOCOL,
            model_config=MODEL_CONFIG,
            reference_records=_reference_records(),
        )


def test_prism_vllm_comparison_is_semantic_exact_but_limits_full_memory_claim() -> None:
    result = compare_prism_external_quality(
        _prism_artifact(),
        _external_artifact(),
        evaluator=EVALUATOR,
        protocol=PROTOCOL,
        model_config=MODEL_CONFIG,
        reference_records=_reference_records(),
    )

    assert result["decision"] == "SMOKE_ONLY"
    assert result["semantic_input_exact"] is True
    assert result["all_required_metrics_pass"] is True
    assert result["kv_cache"]["logical_token_capacity"] == 10240
    assert result["kv_cache"]["full_physical_comparable"] is False
    assert result["kv_cache"]["external_total_bytes"] == 778_567_680
