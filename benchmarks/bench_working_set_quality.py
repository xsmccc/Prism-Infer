#!/usr/bin/env python3
"""Evaluate quality for repeated visual contexts with independently runnable stages."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_system import MODE_SPECS
from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
from benchmarks.working_set_workload import verify_working_set_model
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_materialization import (
    selected_ids_sha256,
    sha256_file,
    write_json_atomic,
)
from prism_infer.analysis.p9_quality_metrics import (
    MUIRBENCH_RANDOM_FALLBACK_SEED,
    aggregate_quality_predictions,
    build_docvqa_prompt,
    build_muirbench_prompt,
    build_mvbench_prompt,
    score_quality_prediction,
)
from prism_infer.analysis.p9_quality_runtime import (
    close_images,
    load_record_images,
    materialization_artifact_by_id,
    prepare_dataset_records,
    quality_input_identity,
    read_json_object,
    safe_materialized_path,
    validate_resume_samples,
)
from prism_infer.analysis.p9_video_sampling import (
    sample_frame_manifest,
    sample_video_file,
)
from prism_infer.analysis.working_set_quality import (
    QUALITY_STAGE_SPECS,
    GroupedQualitySample,
    QualityStageSpec,
    build_muirbench_media_first_prompt,
    paired_deleted_subset,
    select_multi_question_groups,
    summarize_stage_samples,
)
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes
from prism_infer.engine.online import _cache_namespace, _visual_embedding_fingerprint
from prism_infer.engine.vl_inputs import (
    ImageInputs,
    VideoInputs,
    prepare_image_inputs,
    prepare_interleaved_image_inputs,
    prepare_video_inputs,
)
from scripts.verify_p9_quality_materialization import verify_materialization

QUALITY_WORKING_SET_SCHEMA_VERSION = 1
DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_SELECTION = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/p9_quality/raw"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"
DEFAULT_NUM_KV_CACHE_BLOCKS = 220
EXPECTED_KV_CACHE_BYTES = 4_282_122_240
DEFAULT_KEEP_RATIO = 0.6
DEFAULT_IMAGE_MIN_KEEP_TOKENS = 768
DEFAULT_VIDEO_MIN_KEEP_TOKENS = 256
DEFAULT_ATTENTION_LAST_N_LAYERS = 1


def _cache_record(llm: LLM) -> dict[str, Any]:
    payload = llm.model_runner.kv_cache
    scales = llm.model_runner.kv_scale_cache
    storage = kv_cache_storage_bytes(payload, scales)
    return {
        "payload_dtype": str(payload.dtype),
        "payload_shape": list(payload.shape),
        "scale_dtype": "none" if scales is None else str(scales.dtype),
        "scale_shape": [] if scales is None else list(scales.shape),
        "payload_bytes": storage.payload,
        "scale_bytes": storage.scales,
        "total_bytes": storage.total,
    }


def _build_llm(
    model: str,
    spec: QualityStageSpec,
    runtime: dict[str, Any],
    *,
    num_kv_cache_blocks: int,
    keep_ratio: float,
    image_min_keep_tokens: int,
    video_min_keep_tokens: int,
    attention_last_n_layers: int,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    selection_only: bool = False,
) -> LLM:
    if selection_only:
        mode_name = "scaled_fp8_kv"
    elif spec.uses_compaction:
        mode_name = "visual_compact_scaled_fp8_compile_graph"
    else:
        mode_name = "scaled_fp8_kv_compile_graph"
    mode = MODE_SPECS[mode_name]
    return LLM(
        model,
        enforce_eager=mode.enforce_eager,
        execution_backend=mode.execution,
        decode_compile_region=mode.decode_compile_region,
        decode_compile_mode="max-autotune-no-cudagraphs",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
        allow_unsafe_decode_compile=(mode.decode_compile_region != "none"),
        compression_mode=("scaled_fp8_kv" if selection_only else spec.compression_mode),
        tensor_parallel_size=1,
        max_model_len=runtime["max_model_len"],
        max_num_batched_tokens=runtime["max_model_len"],
        max_num_seqs=1,
        enable_chunked_prefill=enable_chunked_prefill,
        max_chunk_size=(
            runtime["max_model_len"] if enable_chunked_prefill else runtime["max_chunk_size"]
        ),
        kvcache_block_size=runtime["kv_cache_page_size"],
        num_kvcache_blocks=num_kv_cache_blocks,
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_prefix_caching=enable_prefix_caching,
        enable_visual_embedding_cache=True,
        enable_visual_pruning_shadow=selection_only,
        image_max_pixels=runtime["image_max_pixels"],
        video_max_pixels=runtime["video_max_pixels"],
        visual_pruning_keep_ratio=keep_ratio,
        visual_pruning_min_keep_tokens=image_min_keep_tokens,
        visual_pruning_video_min_keep_tokens=video_min_keep_tokens,
        visual_pruning_strategy=spec.pruning_strategy,
        visual_pruning_attention_last_n_layers=attention_last_n_layers,
        logits_precision=mode.logits_precision or "model",
        paged_decode_block_n=mode.paged_decode_block_n or 32,
        enable_fused_qk_rmsnorm=mode.fused_qk_rmsnorm,
        enable_fused_qk_mrope=mode.fused_qk_mrope,
        enable_fused_add_rmsnorm=mode.fused_add_rmsnorm,
        enable_packed_kv_projection=mode.packed_kv_projection,
        vision_attention_backend="sdpa",
    )


def _prepare_inputs(
    *,
    llm: LLM,
    spec: QualityStageSpec,
    grouped_sample: GroupedQualitySample,
    materialized_root: Path,
    evaluator_dataset: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[str, ImageInputs | VideoInputs, list[Image.Image], dict[str, Any] | None]:
    record = grouped_sample.record
    images: list[Image.Image] = []
    video_sampling = None
    try:
        if spec.dataset_id == "docvqa_validation":
            prompt = build_docvqa_prompt(record["question"])
            images = load_record_images(record, materialized_root=materialized_root)
            inputs = prepare_image_inputs(llm.vl_processor, prompt, images)
        elif spec.dataset_id == "muirbench_test":
            images = load_record_images(record, materialized_root=materialized_root)
            if spec.prompt_layout == "official_interleaved":
                prompt = build_muirbench_prompt(record["question"], record["options"])
                inputs = prepare_interleaved_image_inputs(
                    llm.vl_processor,
                    prompt,
                    images,
                    image_marker=evaluator_dataset["image_marker"],
                )
            else:
                prompt = build_muirbench_media_first_prompt(
                    record["question"],
                    record["options"],
                    expected_media_count=len(record["media"]),
                    image_marker=evaluator_dataset["image_marker"],
                )
                inputs = prepare_image_inputs(llm.vl_processor, prompt, images)
        elif spec.dataset_id == "mvbench_test":
            prompt = build_mvbench_prompt(record["question"], record["candidates"])
            media = record["media"][0]
            if media.get("identity_kind") == "canonical_frame_manifest_sha256":
                images, video_sampling = sample_frame_manifest(
                    media["frames"],
                    materialized_root=materialized_root,
                    frames=runtime["video_frames"],
                    fps=evaluator_dataset["video_sampling"]["frame_directory_fps"],
                    temporal_bound=record["temporal_bound"],
                )
            else:
                video_path = safe_materialized_path(
                    materialized_root,
                    media["materialized_path"],
                )
                images, video_sampling = sample_video_file(
                    video_path,
                    frames=runtime["video_frames"],
                    temporal_bound=record["temporal_bound"],
                    decoder_contract=evaluator_dataset["video_sampling"]["video_file_decoder"],
                )
            inputs = prepare_video_inputs(
                llm.vl_processor,
                prompt,
                images,
                video_metadata=video_sampling,
            )
        else:  # pragma: no cover - stage table is validated at import time.
            raise ValueError(f"unsupported dataset: {spec.dataset_id}")
        return prompt, inputs, images, video_sampling
    except BaseException:
        close_images(images)
        raise


def _retention_signature(decision: dict[str, Any] | None) -> tuple[int, ...] | None:
    if decision is None:
        return None
    values = decision.get("kept_token_indices")
    if not isinstance(values, (list, tuple)):
        raise ValueError("compression decision has no kept_token_indices")
    return tuple(int(value) for value in values)


def _generate_with_audit(
    llm: LLM,
    inputs: ImageInputs | VideoInputs,
    sampling: SamplingParams,
    *,
    replay_record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(inputs, ImageInputs):
        sequence = llm._prepare_image_sequence(inputs, sampling)
    else:
        sequence = llm._prepare_video_sequence(inputs, sampling)
    if replay_record is not None:
        sequence.visual_pruning_replay_record = copy.deepcopy(replay_record)
    request_type = "images" if isinstance(inputs, ImageInputs) else "video"
    visual_fingerprint = _visual_embedding_fingerprint(
        _cache_namespace(llm),
        request_type,
        inputs,
    )
    sequence.visual_embedding_cache_key = visual_fingerprint
    sequence.multimodal_prefix_cache_key = visual_fingerprint
    sequence_id = llm._submit_sequence(sequence)
    submission_audit = {
        "prefix_hit_on_submit": bool(sequence.multimodal_prefix_cache_hit),
        "pre_admission_hit": bool(getattr(sequence, "multimodal_prefix_pre_admission_hit", False)),
        "prefix_cache_candidate_tokens": int(sequence.prefix_cache_candidate_tokens),
        "multimodal_prefix_boundary": sequence.multimodal_prefix_boundary,
    }
    output = llm._finish_single_generation(sequence_id, None)
    submission_audit["compression_decision"] = copy.deepcopy(
        sequence.visual_pruning_decision_record
    )
    return output, submission_audit


def _run_sample(
    *,
    llm: LLM,
    spec: QualityStageSpec,
    grouped_sample: GroupedQualitySample,
    materialized_root: Path,
    evaluator_dataset: dict[str, Any],
    runtime: dict[str, Any],
    sampling: SamplingParams,
    muirbench_random: random.Random,
    first_decision: dict[str, Any] | None,
    first_question_sample_id: str,
    selection_record: dict[str, Any] | None,
) -> dict[str, Any]:
    images: list[Image.Image] = []
    try:
        prompt, inputs, images, video_sampling = _prepare_inputs(
            llm=llm,
            spec=spec,
            grouped_sample=grouped_sample,
            materialized_root=materialized_root,
            evaluator_dataset=evaluator_dataset,
            runtime=runtime,
        )
        if len(inputs.token_ids) + sampling.max_tokens > runtime["max_model_len"]:
            raise ValueError(
                f"sample {grouped_sample.record['sample_id']} prompt + output exceeds "
                f"max_model_len: {len(inputs.token_ids)} + {sampling.max_tokens} > "
                f"{runtime['max_model_len']}"
            )
        input_identity = quality_input_identity(
            inputs,
            source_prompt=prompt,
            media_sha256=[media["sha256"] for media in grouped_sample.record["media"]],
        )
        replay_record = None
        if selection_record is not None:
            if selection_record.get("input") != input_identity:
                raise RuntimeError(
                    "attention selection input differs from replay input: "
                    f"sample={grouped_sample.record['sample_id']}"
                )
            replay_record = selection_record.get("compression_decision")
            if not isinstance(replay_record, dict):
                raise RuntimeError("attention selection has no replayable decision")
        output, audit = _generate_with_audit(
            llm,
            inputs,
            sampling,
            replay_record=replay_record,
        )
        decision = audit["compression_decision"]
        used_prefix_boundary = audit["multimodal_prefix_boundary"] is not None
        if spec.requires_prefix_boundary and not used_prefix_boundary:
            raise RuntimeError(
                "quality output did not expose the required media-first prefix boundary"
            )
        used_physical_prefix_kv = isinstance(decision, dict) and bool(
            decision.get("physical_compaction")
        )
        if spec.requires_physical_prefix_kv and not used_physical_prefix_kv:
            raise RuntimeError(
                "quality output was not generated from the prefix-boundary compacted KV path"
            )
        if selection_record is not None and not bool(
            decision and decision.get("selection_replay_locked")
        ):
            raise RuntimeError("measured attention request did not use its locked selection")
        if selection_record is not None and audit["pre_admission_hit"]:
            raise RuntimeError("locked attention replay unexpectedly restored a cached prefix")
        reused_first_selection = False
        if grouped_sample.question_index > 0:
            reused_first_selection = bool(audit["pre_admission_hit"]) and (
                _retention_signature(decision) == _retention_signature(first_decision)
            )
        if (
            spec.reuse_scope == "media_group"
            and grouped_sample.question_index > 0
            and not reused_first_selection
        ):
            raise RuntimeError(
                "cross-question quality stage did not restore the first-question "
                f"selection: sample={grouped_sample.record['sample_id']}, "
                f"pre_admission_hit={audit['pre_admission_hit']}, "
                f"prefix_hit_on_submit={audit['prefix_hit_on_submit']}"
            )
        attention_selection_source_sample_id = None
        if selection_record is not None:
            attention_selection_source_sample_id = selection_record["sample_id"]
        elif isinstance(decision, dict):
            attention_selection_source_sample_id = decision.get(
                "selection_source_sample_id"
            )
        record = grouped_sample.record
        sample = {
            "sample_id": record["sample_id"],
            "media_group_id": grouped_sample.media_group_id,
            "question_index": grouped_sample.question_index,
            "group_size": grouped_sample.group_size,
            "first_question_sample_id": first_question_sample_id,
            "input": input_identity,
            "raw_prediction": output["text"],
            "decoded_with_special_tokens": output["raw_text"],
            "output_token_ids": list(output["token_ids"]),
            "score": score_quality_prediction(
                spec.dataset_id,
                record,
                output["text"],
                muirbench_random=muirbench_random,
            ),
            "prefix_hit_on_submit": audit["prefix_hit_on_submit"],
            "pre_admission_hit": audit["pre_admission_hit"],
            "prefix_cache_candidate_tokens": audit["prefix_cache_candidate_tokens"],
            "multimodal_prefix_boundary": audit["multimodal_prefix_boundary"],
            "compression_decision": decision,
            "quality_used_prefix_boundary": used_prefix_boundary,
            "quality_used_physical_prefix_kv": used_physical_prefix_kv,
            "prefix_retention_reused": reused_first_selection,
            "first_question_selection_reused": bool(
                reused_first_selection and spec.attention_selection_scope == "first_question"
            ),
            "attention_selection_source_sample_id": attention_selection_source_sample_id,
        }
        if spec.dataset_id == "mvbench_test":
            sample["task"] = record["task"]
            sample["video_sampling"] = video_sampling
        return sample
    finally:
        close_images(images)


def _stage_run_contract(
    *,
    args: argparse.Namespace,
    spec: QualityStageSpec,
    evaluator: dict[str, Any],
    manifest_path: Path,
    grouped_samples: list[GroupedQualitySample],
    media_sampling_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = evaluator["runtime"]
    effective_keep_ratio = spec.effective_keep_ratio(args.keep_ratio)
    sample_ids = [str(grouped.record["sample_id"]) for grouped in grouped_samples]
    group_ids = list(dict.fromkeys(grouped.media_group_id for grouped in grouped_samples))
    return {
        "stage": args.stage,
        "stage_spec": asdict(spec),
        "model": str(Path(args.model).resolve()),
        "model_revision": evaluator["model"]["revision"],
        "git": asdict(collect_git_metadata(REPO_ROOT, strict=True)),
        "harness_sha256": sha256_file(Path(__file__)),
        "quality_helper_sha256": sha256_file(
            REPO_ROOT / "prism_infer/analysis/working_set_quality.py"
        ),
        "evaluator_sha256": canonical_json_sha256(evaluator),
        "materialization_manifest_sha256": sha256_file(manifest_path),
        "selected_sample_ids_sha256": selected_ids_sha256(sample_ids),
        "selected_media_group_ids_sha256": canonical_json_sha256(group_ids),
        "selected_samples": len(sample_ids),
        "selected_media_groups": len(group_ids),
        "selection_order": "media_group_sha256_then_materialized_source_order",
        "quality_prefix_scope": spec.reuse_scope,
        "media_sampling_identity": media_sampling_identity,
        "runtime": {
            "execution_backend": "compile_graph",
            "decode_compile_region": "stateless",
            "tensor_parallel_size": 1,
            "max_model_len": runtime["max_model_len"],
            "max_num_seqs": 1,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "prefix_boundary_replay": True,
            "max_chunk_size": runtime["max_model_len"],
            "kv_cache_page_size": runtime["kv_cache_page_size"],
            "num_kv_cache_blocks": args.num_kv_cache_blocks,
            "fixed_comparison_kv_cache_bytes": (
                EXPECTED_KV_CACHE_BYTES
                if args.num_kv_cache_blocks == DEFAULT_NUM_KV_CACHE_BLOCKS
                else None
            ),
            "enable_visual_embedding_cache": True,
            "image_max_pixels": runtime["image_max_pixels"],
            "video_max_pixels": runtime["video_max_pixels"],
            "video_frames": runtime["video_frames"],
            "sampling": runtime["sampling"],
        },
        "compression": {
            "mode": spec.compression_mode,
            "keep_ratio": effective_keep_ratio,
            "image_min_keep_tokens": args.image_min_keep_tokens,
            "video_min_keep_tokens": args.video_min_keep_tokens,
            "strategy": spec.pruning_strategy,
            "attention_last_n_layers": args.attention_last_n_layers,
        },
        "attention_selection": (
            None
            if not spec.requires_attention_selection
            else {
                "scope": spec.attention_selection_scope,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "complete_dense_prefill": True,
                "dense_kv_layout": True,
                "visual_pruning_shadow": True,
                "physical_compaction": False,
                "execution_backend": "eager",
                "decode_compile_region": "none",
                "selection_output_used_for_quality": False,
                "max_output_tokens": 1,
            }
        ),
    }


def _prepare_selected_samples(
    *,
    args: argparse.Namespace,
    spec: QualityStageSpec,
    materialization: dict[str, Any],
    materialized_root: Path,
    evaluator_dataset: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[
    list[GroupedQualitySample],
    list[dict[str, str]],
    list[str],
    dict[str, Any] | None,
]:
    artifact = materialization_artifact_by_id(materialization, spec.dataset_id)
    records, exclusions, selected_contract_ids = prepare_dataset_records(
        artifact=artifact,
        materialized_root=materialized_root,
        subset=args.subset,
        max_samples=None,
    )
    media_sampling_identity = None
    if spec.dataset_id == "mvbench_test":
        video_sampling = evaluator_dataset["video_sampling"]
        if int(video_sampling["frames"]) != int(runtime["video_frames"]):
            raise ValueError("MVBench evaluator and runtime frame counts differ")
        media_sampling_identity = {
            "algorithm": video_sampling["algorithm"],
            "frames": int(runtime["video_frames"]),
            "frame_directory_fps": video_sampling["frame_directory_fps"],
            "video_file_decoder": video_sampling["video_file_decoder"],
        }
    grouped = select_multi_question_groups(
        records,
        max_groups=args.max_groups,
        sampling_identity=media_sampling_identity,
    )
    if not grouped:
        raise SystemExit(f"{spec.dataset_id} contains no eligible multi-question media groups")
    return grouped, exclusions, selected_contract_ids, media_sampling_identity


def _restore_first_question_prefix(
    *,
    llm: LLM,
    spec: QualityStageSpec,
    grouped_sample: GroupedQualitySample,
    grouped_by_id: dict[str, list[GroupedQualitySample]],
    completed_by_id: dict[str, dict[str, Any]],
    materialized_root: Path,
    evaluator_dataset: dict[str, Any],
    runtime: dict[str, Any],
    sampling: SamplingParams,
    selection_record: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    first = grouped_by_id[grouped_sample.media_group_id][0]
    images: list[Image.Image] = []
    try:
        prompt, inputs, images, _ = _prepare_inputs(
            llm=llm,
            spec=spec,
            grouped_sample=first,
            materialized_root=materialized_root,
            evaluator_dataset=evaluator_dataset,
            runtime=runtime,
        )
        input_identity = quality_input_identity(
            inputs,
            source_prompt=prompt,
            media_sha256=[media["sha256"] for media in first.record["media"]],
        )
        replay_record = None
        if selection_record is not None:
            if selection_record.get("input") != input_identity:
                raise RuntimeError("resume attention-selection input identity changed")
            replay_record = selection_record.get("compression_decision")
            if not isinstance(replay_record, dict):
                raise RuntimeError("resume attention selection has no replayable decision")
        output, audit = _generate_with_audit(
            llm,
            inputs,
            sampling,
            replay_record=replay_record,
        )
    finally:
        close_images(images)
    recorded = completed_by_id.get(str(first.record["sample_id"]))
    if recorded is None:
        raise RuntimeError("resume prefix restoration has no recorded first question")
    if output["token_ids"] != recorded["output_token_ids"]:
        raise RuntimeError("resume prefix restoration changed first-question output tokens")
    decision = audit["compression_decision"]
    used_prefix_boundary = audit["multimodal_prefix_boundary"] is not None
    if spec.requires_prefix_boundary and not used_prefix_boundary:
        raise RuntimeError("resume prefix restoration did not expose a prefix boundary")
    used_physical_prefix_kv = isinstance(decision, dict) and bool(
        decision.get("physical_compaction")
    )
    if spec.requires_physical_prefix_kv and not used_physical_prefix_kv:
        raise RuntimeError("resume prefix restoration did not use compacted prefix KV")
    if audit["pre_admission_hit"]:
        raise RuntimeError("resume prefix restoration was not a cold first-question replay")
    if _retention_signature(decision) != _retention_signature(recorded.get("compression_decision")):
        raise RuntimeError("resume prefix restoration changed first-question retention")
    if selection_record is not None and not bool(
        decision and decision.get("selection_replay_locked")
    ):
        raise RuntimeError("resume prefix restoration did not use locked attention selection")
    return decision, {
        "media_group_id": grouped_sample.media_group_id,
        "source_sample_id": first.record["sample_id"],
        "output_tokens_match": True,
        "retention_signature_match": True,
    }


def _attention_selection_samples(
    spec: QualityStageSpec,
    grouped_samples: list[GroupedQualitySample],
) -> list[GroupedQualitySample]:
    """Return the requests that need a complete dense attention selection."""

    if spec.attention_selection_scope is None:
        return []
    if spec.attention_selection_scope == "per_question":
        return list(grouped_samples)
    if spec.attention_selection_scope == "first_question":
        return [sample for sample in grouped_samples if sample.question_index == 0]
    raise ValueError(f"unsupported attention selection scope: {spec.attention_selection_scope!r}")


def _validate_selection_records(
    records: object,
    expected_samples: list[GroupedQualitySample],
) -> list[dict[str, Any]]:
    """Accept only an ordered prefix of the required attention selections."""

    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("attention selection checkpoint has no valid record list")
    if len(records) > len(expected_samples):
        raise ValueError("attention selection checkpoint has extra records")
    for record, expected in zip(records, expected_samples, strict=False):
        if str(record.get("sample_id")) != str(expected.record["sample_id"]):
            raise ValueError("attention selection records are not an ordered sample prefix")
        if record.get("media_group_id") != expected.media_group_id:
            raise ValueError("attention selection media-group identity changed")
        if int(record.get("question_index", -1)) != expected.question_index:
            raise ValueError("attention selection question index changed")
        if not isinstance(record.get("input"), dict):
            raise ValueError("attention selection record has no input identity")
        decision = record.get("compression_decision")
        if not isinstance(decision, dict) or decision.get("strategy") != "attention":
            raise ValueError("attention selection record has no attention decision")
        if bool(decision.get("selection_replay_locked")):
            raise ValueError("attention selection checkpoint contains a replay decision")
    return records


def _validate_fixed_kv_cache(cache_record: dict[str, Any], num_blocks: int) -> None:
    if (
        num_blocks == DEFAULT_NUM_KV_CACHE_BLOCKS
        and cache_record["total_bytes"] != EXPECTED_KV_CACHE_BYTES
    ):
        raise RuntimeError(
            "scaled-FP8 KV allocation differs from the fixed-byte comparison: "
            f"expected={EXPECTED_KV_CACHE_BYTES}, actual={cache_record['total_bytes']}"
        )


def _run_attention_selection_phase(
    *,
    args: argparse.Namespace,
    spec: QualityStageSpec,
    grouped_samples: list[GroupedQualitySample],
    artifact: dict[str, Any],
    materialized_root: Path,
    evaluator_dataset: dict[str, Any],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select attention tokens densely, persist them, then release the engine."""

    expected = _attention_selection_samples(spec, grouped_samples)
    records = _validate_selection_records(artifact.get("selection_records", []), expected)
    phase = artifact.setdefault(
        "attention_selection_phase",
        {
            "status": "not_required" if not expected else "in_progress",
            "required_samples": len(expected),
            "completed_samples": len(records),
        },
    )
    if not expected:
        phase.update({"status": "not_required", "completed_samples": 0})
        artifact["selection_records"] = []
        return []
    if len(records) == len(expected):
        phase.update({"status": "complete", "completed_samples": len(records)})
        write_json_atomic(args.output, artifact)
        return records

    selection_sampling = SamplingParams(
        temperature=runtime["sampling"]["temperature"],
        max_tokens=1,
        ignore_eos=runtime["sampling"]["ignore_eos"],
    )
    selection_llm: LLM | None = None
    try:
        selection_llm = _build_llm(
            args.model,
            spec,
            runtime,
            num_kv_cache_blocks=args.num_kv_cache_blocks,
            keep_ratio=spec.effective_keep_ratio(args.keep_ratio),
            image_min_keep_tokens=args.image_min_keep_tokens,
            video_min_keep_tokens=args.video_min_keep_tokens,
            attention_last_n_layers=args.attention_last_n_layers,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            selection_only=True,
        )
        selection_cache = _cache_record(selection_llm)
        _validate_fixed_kv_cache(selection_cache, args.num_kv_cache_blocks)
        if selection_llm.config.compression_mode != "scaled_fp8_kv":
            raise RuntimeError("attention selection must use a dense Scaled-FP8 KV layout")
        if not selection_llm.config.enable_visual_pruning_shadow:
            raise RuntimeError("attention selection must collect scores in shadow mode")
        if selection_llm.config.execution_backend != "eager":
            raise RuntimeError("attention selection must not create decode Graph resources")
        selection_cache.update(
            {
                "layout": "dense",
                "compression_mode": selection_llm.config.compression_mode,
                "visual_pruning_shadow": True,
                "execution_backend": selection_llm.config.execution_backend,
            }
        )
        artifact["attention_selection_kv_cache"] = selection_cache
        for grouped in expected[len(records) :]:
            images: list[Image.Image] = []
            try:
                prompt, inputs, images, video_sampling = _prepare_inputs(
                    llm=selection_llm,
                    spec=spec,
                    grouped_sample=grouped,
                    materialized_root=materialized_root,
                    evaluator_dataset=evaluator_dataset,
                    runtime=runtime,
                )
                if len(inputs.token_ids) + selection_sampling.max_tokens > runtime["max_model_len"]:
                    raise ValueError(
                        f"attention selection sample {grouped.record['sample_id']} exceeds "
                        "max_model_len"
                    )
                input_identity = quality_input_identity(
                    inputs,
                    source_prompt=prompt,
                    media_sha256=[media["sha256"] for media in grouped.record["media"]],
                )
                output, audit = _generate_with_audit(
                    selection_llm,
                    inputs,
                    selection_sampling,
                )
                decision = audit["compression_decision"]
                if not isinstance(decision, dict) or decision.get("strategy") != "attention":
                    raise RuntimeError("dense attention selection produced no decision")
                if bool(decision.get("physical_compaction")):
                    raise RuntimeError("dense attention selection unexpectedly compacted KV")
                if bool(decision.get("selection_replay_locked")):
                    raise RuntimeError("dense attention selection unexpectedly used replay")
                decision = copy.deepcopy(decision)
                decision["selection_source_sample_id"] = grouped.record["sample_id"]
                selection_record = {
                    "sample_id": grouped.record["sample_id"],
                    "media_group_id": grouped.media_group_id,
                    "question_index": grouped.question_index,
                    "input": input_identity,
                    "compression_decision": decision,
                    "selection_probe_token_ids": list(output["token_ids"]),
                    "video_sampling": video_sampling,
                }
                records.append(selection_record)
                artifact["selection_records"] = records
                phase.update(
                    {
                        "status": "in_progress",
                        "completed_samples": len(records),
                    }
                )
                write_json_atomic(args.output, artifact)
            finally:
                close_images(images)
    finally:
        if selection_llm is not None:
            torch.cuda.synchronize()
            selection_llm.exit()
            selection_llm = None
        gc.collect()
        torch.cuda.empty_cache()

    phase.update({"status": "complete", "completed_samples": len(records)})
    artifact["selection_records"] = records
    write_json_atomic(args.output, artifact)
    return records


def run_stage(args: argparse.Namespace) -> None:
    spec = QUALITY_STAGE_SPECS[args.stage]
    if spec.requires_attention_selection:
        if args.phase not in ("selection", "replay"):
            raise SystemExit(
                "attention stages require process-isolated phases: run --phase selection, "
                "then run --phase replay --resume"
            )
        if args.phase == "replay" and (not args.resume or not args.output.exists()):
            raise SystemExit("attention replay requires an existing selection output and --resume")
    elif args.phase != "all":
        raise SystemExit("--phase selection/replay is only valid for attention stages")
    if args.num_kv_cache_blocks <= 0:
        raise SystemExit("--num-kv-cache-blocks must be positive")
    if not 0.0 < args.keep_ratio <= 1.0:
        raise SystemExit("--keep-ratio must be in (0, 1]")
    if args.image_min_keep_tokens <= 0 or args.video_min_keep_tokens <= 0:
        raise SystemExit("minimum keep-token values must be positive")
    if args.attention_last_n_layers <= 0:
        raise SystemExit("--attention-last-n-layers must be positive")
    if torch.cuda.device_count() != 1:
        raise SystemExit("quality stages require exactly one visible CUDA device")

    materialized_root = args.materialized_root.resolve()
    verification = verify_materialization(
        protocol_path=args.protocol,
        selection_path=args.selection,
        raw_root=args.raw_root,
        materialized_root=materialized_root,
    )
    evaluator = read_json_object(args.evaluator)
    protocol = read_json_object(args.protocol)
    verify_working_set_model(
        {"model": {"revision": evaluator["model"]["revision"]}},
        args.model,
    )
    if evaluator["quality_protocol_sha256"] != canonical_json_sha256(protocol):
        raise SystemExit("evaluator references a different quality protocol")
    manifest_path = materialized_root / "p9_quality_materialization.json"
    materialization = read_json_object(manifest_path)
    runtime = evaluator["runtime"]
    evaluator_dataset = evaluator["datasets"][spec.dataset_id]
    (
        grouped_samples,
        exclusions,
        selected_contract_ids,
        media_sampling_identity,
    ) = _prepare_selected_samples(
        args=args,
        spec=spec,
        materialization=materialization,
        materialized_root=materialized_root,
        evaluator_dataset=evaluator_dataset,
        runtime=runtime,
    )
    expected_ids = [str(grouped.record["sample_id"]) for grouped in grouped_samples]
    run_contract = _stage_run_contract(
        args=args,
        spec=spec,
        evaluator=evaluator,
        manifest_path=manifest_path,
        grouped_samples=grouped_samples,
        media_sampling_identity=media_sampling_identity,
    )
    run_identity_sha256 = canonical_json_sha256(run_contract)
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists; pass --resume: {args.output}")
        artifact = read_json_object(args.output)
        samples = validate_resume_samples(
            artifact,
            run_identity_sha256=run_identity_sha256,
            expected_ids=expected_ids,
        )
        artifact["status"] = "in_progress"
        artifact.pop("failure", None)
        artifact.setdefault("resume_prefix_restores", [])
        artifact.setdefault("selection_records", [])
        artifact.setdefault(
            "attention_selection_phase",
            {
                "status": "in_progress" if spec.requires_attention_selection else "not_required",
                "required_samples": len(_attention_selection_samples(spec, grouped_samples)),
                "completed_samples": len(artifact["selection_records"]),
            },
        )
        artifact.setdefault(
            "quality_replay_phase",
            {"status": "in_progress", "completed_samples": len(samples)},
        )
        write_json_atomic(args.output, artifact)
    else:
        samples = []
        artifact = {
            "schema_version": QUALITY_WORKING_SET_SCHEMA_VERSION,
            "record_type": "repeated_visual_context_quality",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "run_identity_sha256": run_identity_sha256,
            "run_contract": run_contract,
            "selection": {
                "selected_contract_samples": len(selected_contract_ids),
                "selected_contract_ids_sha256": selected_ids_sha256(selected_contract_ids),
                "multi_question_samples": len(expected_ids),
                "multi_question_ids_sha256": selected_ids_sha256(expected_ids),
                "protocol_exclusions": exclusions,
            },
            "materialization_verification": verification,
            "resume_prefix_restores": [],
            "selection_records": [],
            "attention_selection_phase": {
                "status": "in_progress" if spec.requires_attention_selection else "not_required",
                "required_samples": len(_attention_selection_samples(spec, grouped_samples)),
                "completed_samples": 0,
            },
            "quality_replay_phase": {
                "status": "in_progress",
                "completed_samples": 0,
            },
            "samples": samples,
            "summary": summarize_stage_samples(spec.dataset_id, samples),
        }
        write_json_atomic(args.output, artifact)

    artifact["environment"] = {
        "gpu": asdict(collect_gpu_metadata(0, strict_identity=True)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json_atomic(args.output, artifact)

    required_selection_records = len(_attention_selection_samples(spec, grouped_samples))
    if args.phase == "replay" and len(artifact["selection_records"]) != required_selection_records:
        raise SystemExit("attention replay requires a complete selection checkpoint")

    try:
        selection_records = _run_attention_selection_phase(
            args=args,
            spec=spec,
            grouped_samples=grouped_samples,
            artifact=artifact,
            materialized_root=materialized_root,
            evaluator_dataset=evaluator_dataset,
            runtime=runtime,
        )

        if args.phase == "selection":
            artifact["status"] = "selection_complete"
            artifact["quality_replay_phase"].update(
                {"status": "not_started", "completed_samples": len(samples)}
            )
            artifact["selection_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            output_sha256 = write_json_atomic(args.output, artifact)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "output_sha256": output_sha256,
                        "stage": args.stage,
                        "phase": "selection",
                        "selection_records": len(selection_records),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        muirbench_random = random.Random(MUIRBENCH_RANDOM_FALLBACK_SEED)
        records_by_sample_id = {
            str(grouped.record["sample_id"]): grouped.record for grouped in grouped_samples
        }
        for sample in samples:
            if spec.dataset_id != "muirbench_test":
                continue
            replayed = score_quality_prediction(
                spec.dataset_id,
                records_by_sample_id[str(sample["sample_id"])],
                sample["raw_prediction"],
                muirbench_random=muirbench_random,
            )
            if replayed != sample["score"]:
                raise ValueError("resume MuirBench parser state differs from checkpoint")

        sampling = SamplingParams(
            temperature=runtime["sampling"]["temperature"],
            max_tokens=evaluator_dataset["max_output_tokens"],
            ignore_eos=runtime["sampling"]["ignore_eos"],
        )
        grouped_by_id: dict[str, list[GroupedQualitySample]] = {}
        for grouped in grouped_samples:
            grouped_by_id.setdefault(grouped.media_group_id, []).append(grouped)
        completed_by_id = {str(sample["sample_id"]): sample for sample in samples}
        selection_by_id = {str(record["sample_id"]): record for record in selection_records}
        first_decisions: dict[str, dict[str, Any] | None] = {}
        loaded_prefix_groups: set[str] = set()
        active_prefix_group: str | None = None

        llm: LLM | None = None
        try:
            llm = _build_llm(
                args.model,
                spec,
                runtime,
                num_kv_cache_blocks=args.num_kv_cache_blocks,
                keep_ratio=spec.effective_keep_ratio(args.keep_ratio),
                image_min_keep_tokens=args.image_min_keep_tokens,
                video_min_keep_tokens=args.video_min_keep_tokens,
                attention_last_n_layers=args.attention_last_n_layers,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
            )
            artifact["kv_cache"] = _cache_record(llm)
            _validate_fixed_kv_cache(artifact["kv_cache"], args.num_kv_cache_blocks)
            artifact["prefix_cache_before"] = (
                llm.scheduler.block_manager.multimodal_prefix_cache_metadata()
            )
            if llm.vl_processor.image_processor.size.longest_edge != runtime["image_max_pixels"]:
                raise RuntimeError("runtime image pixel budget differs from evaluator")
            if llm.vl_processor.video_processor.size.longest_edge != runtime["video_max_pixels"]:
                raise RuntimeError("runtime video pixel budget differs from evaluator")

            for grouped in grouped_samples[len(samples) :]:
                group_id = grouped.media_group_id
                if spec.reuse_scope == "per_request":
                    llm.scheduler.block_manager.clear_multimodal_prefix_cache()
                    loaded_prefix_groups.clear()
                    active_prefix_group = None
                elif spec.reuse_scope == "media_group" and group_id != active_prefix_group:
                    llm.scheduler.block_manager.clear_multimodal_prefix_cache()
                    loaded_prefix_groups.clear()
                    active_prefix_group = group_id
                if (
                    spec.reuse_scope == "media_group"
                    and grouped.question_index > 0
                    and group_id not in loaded_prefix_groups
                ):
                    first_grouped = grouped_by_id[group_id][0]
                    first_sample_id = str(first_grouped.record["sample_id"])
                    first_decision, restore = _restore_first_question_prefix(
                        llm=llm,
                        spec=spec,
                        grouped_sample=grouped,
                        grouped_by_id=grouped_by_id,
                        completed_by_id=completed_by_id,
                        materialized_root=materialized_root,
                        evaluator_dataset=evaluator_dataset,
                        runtime=runtime,
                        sampling=sampling,
                        selection_record=selection_by_id.get(first_sample_id),
                    )
                    first_decisions[group_id] = first_decision
                    loaded_prefix_groups.add(group_id)
                    artifact["resume_prefix_restores"].append(restore)

                sample_id = str(grouped.record["sample_id"])
                selection_record = None
                if spec.attention_selection_scope == "per_question":
                    selection_record = selection_by_id.get(sample_id)
                elif (
                    spec.attention_selection_scope == "first_question"
                    and grouped.question_index == 0
                ):
                    selection_record = selection_by_id.get(sample_id)
                if (
                    spec.requires_attention_selection
                    and (
                        grouped.question_index == 0
                        or spec.attention_selection_scope == "per_question"
                    )
                    and selection_record is None
                ):
                    raise RuntimeError(f"missing locked attention selection for sample {sample_id}")

                sample = _run_sample(
                    llm=llm,
                    spec=spec,
                    grouped_sample=grouped,
                    materialized_root=materialized_root,
                    evaluator_dataset=evaluator_dataset,
                    runtime=runtime,
                    sampling=sampling,
                    muirbench_random=muirbench_random,
                    first_decision=first_decisions.get(group_id),
                    first_question_sample_id=str(grouped_by_id[group_id][0].record["sample_id"]),
                    selection_record=selection_record,
                )
                samples.append(sample)
                completed_by_id[str(sample["sample_id"])] = sample
                if grouped.question_index == 0:
                    first_decisions[group_id] = sample.get("compression_decision")
                    if spec.reuse_scope == "media_group":
                        loaded_prefix_groups.add(group_id)
                artifact["samples"] = samples
                artifact["completed_samples"] = len(samples)
                artifact["quality_replay_phase"].update(
                    {"status": "in_progress", "completed_samples": len(samples)}
                )
                artifact["summary"] = summarize_stage_samples(spec.dataset_id, samples)
                write_json_atomic(args.output, artifact)
        finally:
            if llm is not None:
                artifact["prefix_cache_after"] = (
                    llm.scheduler.block_manager.multimodal_prefix_cache_metadata()
                )
                artifact["visual_embedding_cache_after"] = (
                    llm.model_runner.visual_embedding_cache_metadata()
                )
                llm.exit()
                llm = None
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        artifact["status"] = "failed"
        artifact["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json_atomic(args.output, artifact)
        raise

    artifact["status"] = "complete"
    artifact["quality_replay_phase"].update(
        {"status": "complete", "completed_samples": len(samples)}
    )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    artifact["completed_samples"] = len(samples)
    artifact["summary"] = summarize_stage_samples(spec.dataset_id, samples)
    output_sha256 = write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "stage": args.stage,
                "samples": len(samples),
                "summary": artifact["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _artifact_samples_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["sample_id"]): sample for sample in artifact["samples"]}


def _artifact_comparison_identity(artifact: dict[str, Any]) -> dict[str, str]:
    try:
        git = artifact["run_contract"]["git"]
        environment = artifact["environment"]
        gpu = environment["gpu"]
        identity = {
            "git.commit": git["commit"],
            "environment.torch": environment["torch"],
            "environment.cuda": environment["cuda"],
            "environment.transformers": environment["transformers"],
            "environment.gpu.name": gpu["name"],
        }
        dirty = git["dirty"]
    except (KeyError, TypeError) as exc:
        raise ValueError("quality artifact is missing comparison identity metadata") from exc
    if not isinstance(dirty, bool):
        raise ValueError("quality artifact Git dirty metadata must be boolean")
    if dirty:
        raise ValueError("quality artifact was produced from a dirty Git worktree")
    for key, value in identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"quality artifact has invalid comparison identity field {key!r}")
    return identity


def _assert_comparable_artifacts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    reference_contract = reference["run_contract"]
    candidate_contract = candidate["run_contract"]
    for key in (
        "model_revision",
        "harness_sha256",
        "quality_helper_sha256",
        "evaluator_sha256",
        "materialization_manifest_sha256",
        "selected_sample_ids_sha256",
        "runtime",
    ):
        if reference_contract[key] != candidate_contract[key]:
            raise ValueError(f"quality artifacts differ at comparison identity field {key!r}")
    reference_identity = _artifact_comparison_identity(reference)
    candidate_identity = _artifact_comparison_identity(candidate)
    for key, reference_value in reference_identity.items():
        if reference_value != candidate_identity[key]:
            raise ValueError(f"quality artifacts differ at comparison identity field {key!r}")


def _paired_all_samples(
    dataset_id: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    _assert_comparable_artifacts(reference, candidate)
    reference_by_id = _artifact_samples_by_id(reference)
    candidate_by_id = _artifact_samples_by_id(candidate)
    if list(reference_by_id) != list(candidate_by_id):
        raise ValueError("paired quality artifacts have different sample order or identity")
    sample_ids = list(reference_by_id)
    return {
        "samples": len(sample_ids),
        "sample_ids_sha256": canonical_json_sha256(sample_ids),
        "reference_quality": aggregate_quality_predictions(
            dataset_id,
            [reference_by_id[sample_id] for sample_id in sample_ids],
        ),
        "candidate_quality": aggregate_quality_predictions(
            dataset_id,
            [candidate_by_id[sample_id] for sample_id in sample_ids],
        ),
    }


def summarize_artifacts(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"summary output already exists: {args.output}")
    by_stage: dict[str, dict[str, Any]] = {}
    input_records = []
    for path in args.input:
        artifact = read_json_object(path)
        if artifact.get("record_type") != "repeated_visual_context_quality":
            raise ValueError(f"not a repeated-context quality artifact: {path}")
        if artifact.get("status") != "complete":
            raise ValueError(f"quality artifact is not complete: {path}")
        run_contract = artifact["run_contract"]
        if artifact.get("run_identity_sha256") != canonical_json_sha256(run_contract):
            raise ValueError(f"quality artifact run identity is invalid: {path}")
        stage = str(run_contract["stage"])
        if stage not in QUALITY_STAGE_SPECS:
            raise ValueError(f"quality artifact has unsupported stage {stage!r}")
        if stage in by_stage:
            raise ValueError(f"duplicate quality stage: {stage}")
        by_stage[stage] = artifact
        input_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    expected_stages = set(QUALITY_STAGE_SPECS)
    missing_stages = sorted(expected_stages - by_stage.keys())
    complete_matrix = not missing_stages and len(by_stage) == len(expected_stages)
    if not complete_matrix and not args.allow_partial:
        raise ValueError(
            "quality summary requires all QUALITY_STAGE_SPECS; missing stages: "
            + ", ".join(missing_stages)
            + "; pass --allow-partial to summarize an explicit subset"
        )

    comparison_identity: dict[str, str] | None = None
    for stage, artifact in sorted(by_stage.items()):
        stage_identity = _artifact_comparison_identity(artifact)
        if comparison_identity is None:
            comparison_identity = stage_identity
            continue
        for key, reference_value in comparison_identity.items():
            if reference_value != stage_identity[key]:
                raise ValueError(
                    f"quality stage {stage!r} differs at comparison identity field {key!r}"
                )

    comparisons: dict[str, Any] = {}
    if {"muir_dense_official", "muir_dense_media_first"} <= by_stage.keys():
        comparisons["muir_dense_prompt_layout"] = _paired_all_samples(
            "muirbench_test",
            by_stage["muir_dense_official"],
            by_stage["muir_dense_media_first"],
        )
    if "muir_dense_media_first" in by_stage:
        dense = by_stage["muir_dense_media_first"]
        for candidate_stage in (
            "muir_attention_per_question",
            "muir_attention_first_reuse",
            "muir_uniform_reuse",
        ):
            if candidate_stage not in by_stage:
                continue
            _assert_comparable_artifacts(dense, by_stage[candidate_stage])
            comparisons[candidate_stage] = paired_deleted_subset(
                dataset_id="muirbench_test",
                reference_samples=dense["samples"],
                candidate_samples=by_stage[candidate_stage]["samples"],
            )
    for dataset_prefix, dataset_id in (
        ("docvqa", "docvqa_validation"),
        ("mvbench", "mvbench_test"),
    ):
        dense_stage = f"{dataset_prefix}_dense"
        uniform_stage = f"{dataset_prefix}_uniform"
        if {dense_stage, uniform_stage} <= by_stage.keys():
            _assert_comparable_artifacts(by_stage[dense_stage], by_stage[uniform_stage])
            comparisons[f"{dataset_prefix}_uniform_actual_deletion"] = paired_deleted_subset(
                dataset_id=dataset_id,
                reference_samples=by_stage[dense_stage]["samples"],
                candidate_samples=by_stage[uniform_stage]["samples"],
            )

    summary = {
        "schema_version": QUALITY_WORKING_SET_SCHEMA_VERSION,
        "record_type": "repeated_visual_context_quality_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_matrix": complete_matrix,
        "allow_partial": bool(args.allow_partial),
        "missing_stages": missing_stages,
        "comparison_identity": comparison_identity,
        "inputs": input_records,
        "stage_summaries": {
            stage: artifact["summary"] for stage, artifact in sorted(by_stage.items())
        },
        "comparisons": comparisons,
    }
    output_sha256 = write_json_atomic(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "stages": sorted(by_stage),
                "complete_matrix": complete_matrix,
                "missing_stages": missing_stages,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run one quality stage; attention selection and replay use separate processes",
    )
    run.add_argument("--model", required=True)
    run.add_argument("--stage", choices=sorted(QUALITY_STAGE_SPECS), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--subset", choices=("development", "final"), default="final")
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    run.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    run.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    run.add_argument("--max-groups", type=int)
    run.add_argument("--num-kv-cache-blocks", type=int, default=DEFAULT_NUM_KV_CACHE_BLOCKS)
    run.add_argument("--keep-ratio", type=float, default=DEFAULT_KEEP_RATIO)
    run.add_argument(
        "--image-min-keep-tokens",
        type=int,
        default=DEFAULT_IMAGE_MIN_KEEP_TOKENS,
    )
    run.add_argument(
        "--video-min-keep-tokens",
        type=int,
        default=DEFAULT_VIDEO_MIN_KEEP_TOKENS,
    )
    run.add_argument(
        "--attention-last-n-layers",
        type=int,
        default=DEFAULT_ATTENTION_LAST_N_LAYERS,
    )
    run.add_argument(
        "--phase",
        choices=("all", "selection", "replay"),
        default="all",
        help=(
            "attention stages use selection then replay in separate processes; other stages use all"
        ),
    )
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=run_stage)

    summarize = subparsers.add_parser(
        "summarize",
        help="build paired tables from completed stage artifacts",
    )
    summarize.add_argument("--input", type=Path, nargs="+", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow a named subset instead of requiring all nine completed quality stages",
    )
    summarize.set_defaults(handler=summarize_artifacts)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
