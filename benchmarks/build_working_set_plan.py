#!/usr/bin/env python3
"""Measure dense MuirBench prefix pages and build a shared working-set plan.

The command runs one cold, media-first request per exact ordered-media group.
Each successful measurement is written atomically, so an interrupted run can
continue from the existing page artifact without repeating completed groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_system import MODE_SPECS
from benchmarks.working_set_workload import (
    verify_working_set_model,
    verify_working_set_processor,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.p9_quality_runtime import safe_materialized_path
from prism_infer.analysis.working_set_plan import (
    DEFAULT_IMAGE_MARKER,
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_IMAGE_MIN_PIXELS,
    DEFAULT_KV_BUDGET_BYTES,
    DEFAULT_KV_BUDGET_PAGES,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    DEFAULT_PAGE_SIZE_TOKENS,
    DENSE_PREFIX_PAGES_RECORD_TYPE,
    MUIRBENCH_DATASET_ID,
    WORKING_SET_PLAN_SCHEMA_VERSION,
    build_media_first_groups,
    build_working_set_plan,
    load_muirbench_records,
    write_working_set_plan,
)
from prism_infer.engine.online import OnlineRequest, OnlineServingSession

DEFAULT_SELECTION = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
DEFAULT_SUBSET = "final"
_MEASUREMENT_MODE = "visual_compact_scaled_fp8_compile_graph"
_MAX_NEW_TOKENS = 1
_MAX_NUM_BATCHED_TOKENS = DEFAULT_MAX_MODEL_LEN
_MAX_CHUNK_SIZE = DEFAULT_MAX_CHUNK_SIZE


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def _measurement_contract(*, model_name: str, model_revision: str) -> dict[str, Any]:
    return {
        "model": {
            "name": model_name,
            "revision": model_revision,
            "max_model_len": DEFAULT_MAX_MODEL_LEN,
        },
        "runtime": {
            "engine": "prism",
            "tensor_parallel_size": 1,
            "compression_mode": "visual_compact_scaled_fp8",
            "execution_mode": _MEASUREMENT_MODE,
            "kv_budget_bytes": DEFAULT_KV_BUDGET_BYTES,
            "kv_budget_pages": DEFAULT_KV_BUDGET_PAGES,
            "page_size_tokens": DEFAULT_PAGE_SIZE_TOKENS,
            "max_num_batched_tokens": _MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": 1,
            "max_chunk_size": _MAX_CHUNK_SIZE,
            "enable_prefix_caching": True,
            "visual_pruning_strategy": "uniform",
            "visual_pruning_keep_ratio": 1.0,
            "visual_pruning_min_keep_tokens": 768,
            "image_max_pixels": DEFAULT_IMAGE_MAX_PIXELS,
        },
        "processor": {
            "image_min_pixels": DEFAULT_IMAGE_MIN_PIXELS,
            "image_max_pixels": DEFAULT_IMAGE_MAX_PIXELS,
            "image_size": {
                "shortest_edge": DEFAULT_IMAGE_MIN_PIXELS,
                "longest_edge": DEFAULT_IMAGE_MAX_PIXELS,
            },
        },
        "sampling": {
            "temperature": 0.0,
            "greedy": True,
            "ignore_eos": True,
            "max_new_tokens": _MAX_NEW_TOKENS,
        },
        "page_measurement": (
            "multimodal_prefix_cache.resident_blocks_minus_resident_tail_clone_blocks"
        ),
    }


def _new_page_artifact(
    *,
    model_name: str,
    model_revision: str,
    source_identity: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": WORKING_SET_PLAN_SCHEMA_VERSION,
        "record_type": DENSE_PREFIX_PAGES_RECORD_TYPE,
        "dataset": {
            "id": MUIRBENCH_DATASET_ID,
            "source_identity": dict(source_identity),
            "media_groups": len(groups),
        },
        **_measurement_contract(
            model_name=model_name,
            model_revision=model_revision,
        ),
        "groups": [],
    }


def _resume_measurements(
    artifact: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate a partial artifact and return completed groups by identity."""

    for field in (
        "schema_version",
        "record_type",
        "dataset",
        "model",
        "runtime",
        "sampling",
        "page_measurement",
    ):
        if artifact.get(field) != expected.get(field):
            raise ValueError(f"page artifact {field!r} differs from this measurement run")

    expected_groups = {str(group["group_id"]): group for group in groups}
    raw_measurements = artifact.get("groups")
    if not isinstance(raw_measurements, list):
        raise ValueError("page artifact groups must be a list")
    completed: dict[str, dict[str, Any]] = {}
    for index, raw_measurement in enumerate(raw_measurements):
        if not isinstance(raw_measurement, dict):
            raise ValueError(f"page artifact groups[{index}] must be an object")
        group_id = raw_measurement.get("group_id")
        if not isinstance(group_id, str) or group_id not in expected_groups:
            raise ValueError(f"page artifact groups[{index}] has an unknown group_id")
        if group_id in completed:
            raise ValueError(f"page artifact contains duplicate group {group_id!r}")
        group = expected_groups[group_id]
        if raw_measurement.get("ordered_media_sha256") != group["ordered_media_sha256"]:
            raise ValueError(f"page artifact media identity differs for group {group_id!r}")
        dense_prefix_pages = raw_measurement.get("dense_prefix_pages")
        if (
            isinstance(dense_prefix_pages, bool)
            or not isinstance(dense_prefix_pages, int)
            or dense_prefix_pages <= 0
        ):
            raise ValueError(f"page artifact has invalid pages for group {group_id!r}")
        completed[group_id] = dict(raw_measurement)
    return completed


def _load_page_artifact(
    path: Path,
    *,
    expected: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        artifact = dict(expected)
        _write_json_atomic(path, artifact)
        return artifact, {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("page artifact must be a JSON object")
    completed = _resume_measurements(value, expected=expected, groups=groups)
    return value, completed


def _verified_group_media_paths(
    group: Mapping[str, Any],
    *,
    materialized_root: Path,
) -> list[Path]:
    paths = []
    for index, media in enumerate(group["media"]):
        path = safe_materialized_path(
            materialized_root,
            str(media["materialized_path"]),
        )
        actual_sha256 = _sha256_file(path)
        expected_sha256 = str(group["ordered_media_sha256"][index])
        if actual_sha256 != expected_sha256 or actual_sha256 != media["sha256"]:
            raise ValueError(
                "materialized media SHA256 mismatch for "
                f"group {group['group_id']!r}, media {index}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        paths.append(path)
    return paths


def _verified_group_images(
    group: Mapping[str, Any],
    *,
    materialized_root: Path,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        paths = _verified_group_media_paths(
            group,
            materialized_root=materialized_root,
        )
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


def _build_engine(model: str) -> LLM:
    mode = MODE_SPECS[_MEASUREMENT_MODE]
    return LLM(
        model,
        enforce_eager=mode.enforce_eager,
        execution_backend=mode.execution,
        decode_compile_region=mode.decode_compile_region,
        decode_compile_mode="max-autotune-no-cudagraphs",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
        allow_unsafe_decode_compile=True,
        compression_mode=mode.compression,
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        num_kvcache_blocks=DEFAULT_KV_BUDGET_PAGES,
        kvcache_block_size=DEFAULT_PAGE_SIZE_TOKENS,
        enable_chunked_prefill=True,
        max_chunk_size=_MAX_CHUNK_SIZE,
        enable_prefix_caching=True,
        scheduler_policy="fcfs",
        visual_pruning_keep_ratio=1.0,
        visual_pruning_min_keep_tokens=768,
        visual_pruning_video_min_keep_tokens=256,
        visual_pruning_strategy="uniform",
        visual_pruning_attention_last_n_layers=1,
        logits_precision=mode.logits_precision,
        mlp_projection_mode="packed",
        paged_decode_block_n=mode.paged_decode_block_n,
        enable_fused_qk_rmsnorm=mode.fused_qk_rmsnorm,
        enable_fused_qk_mrope=mode.fused_qk_mrope,
        enable_fused_add_rmsnorm=mode.fused_add_rmsnorm,
        enable_packed_kv_projection=mode.packed_kv_projection,
        enable_visual_embedding_cache=False,
        vision_attention_backend="sdpa",
        image_max_pixels=DEFAULT_IMAGE_MAX_PIXELS,
    )


def _measure_group(
    llm: LLM,
    group: Mapping[str, Any],
    *,
    materialized_root: Path,
) -> dict[str, Any]:
    images = _verified_group_images(group, materialized_root=materialized_root)
    block_manager = llm.scheduler.block_manager
    session: OnlineServingSession | None = None
    try:
        sample = group["samples"][0]
        payload = {
            "type": "interleaved_images",
            "prompt": sample["source_prompt"],
            "images": images,
            "image_marker": DEFAULT_IMAGE_MARKER,
        }
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        session = OnlineServingSession(llm)
        result = session.run(
            (
                OnlineRequest(
                    request_key=f"measure:{group['group_id']}",
                    arrival_offset_s=0.0,
                    payload=payload,
                    sampling_params=sampling,
                ),
            )
        )
        request = result.requests[0]
        if request.state != "finished" or len(request.token_ids) != _MAX_NEW_TOKENS:
            raise RuntimeError(
                f"dense-prefix measurement did not finish for group {group['group_id']!r}: "
                f"state={request.state!r}, output_tokens={len(request.token_ids)}"
            )
        torch.cuda.synchronize()
        metadata = llm.multimodal_prefix_cache_metadata()
        resident_blocks = int(metadata["resident_blocks"])
        resident_tail_blocks = int(metadata["resident_tail_clone_blocks"])
        dense_prefix_pages = resident_blocks - resident_tail_blocks
        if int(metadata["entries"]) != 1 or dense_prefix_pages <= 0:
            raise RuntimeError(
                "cold dense-prefix measurement produced no unique resident entry: "
                f"entries={metadata['entries']}, resident={resident_blocks}, "
                f"tail_clones={resident_tail_blocks}"
            )
        total_pool_blocks = int(metadata["total_pool_blocks"])
        max_cache_blocks = int(metadata["max_blocks"])
        if (
            total_pool_blocks != DEFAULT_KV_BUDGET_PAGES
            or max_cache_blocks != DEFAULT_KV_BUDGET_PAGES
        ):
            raise RuntimeError(
                "Prism full-pool prefix capacity differs from the measurement contract: "
                f"total={total_pool_blocks}, cache={max_cache_blocks}, "
                f"expected={DEFAULT_KV_BUDGET_PAGES}"
            )
        bytes_per_block = int(metadata["bytes_per_block_all_ranks"])
        total_pool_bytes = total_pool_blocks * bytes_per_block
        if total_pool_bytes != DEFAULT_KV_BUDGET_BYTES:
            raise RuntimeError(
                "Prism KV pool byte size differs from the measurement contract: "
                f"{total_pool_bytes} != {DEFAULT_KV_BUDGET_BYTES}"
            )
        return {
            "group_id": group["group_id"],
            "ordered_media_sha256": list(group["ordered_media_sha256"]),
            "sample_id": sample["sample_id"],
            "source_prompt_sha256": sample["source_prompt_sha256"],
            "dense_prefix_pages": dense_prefix_pages,
            "resident_blocks": resident_blocks,
            "resident_tail_clone_blocks": resident_tail_blocks,
            "bytes_per_block_all_ranks": bytes_per_block,
            "dense_prefix_bytes": dense_prefix_pages * bytes_per_block,
            "media_sha256_verified": True,
        }
    finally:
        if llm.is_finished():
            block_manager.clear_multimodal_prefix_cache()
            llm.reset_metrics()
        for image in images:
            image.close()
        del session


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen3-VL model path")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="portable model name recorded in the output plan",
    )
    parser.add_argument(
        "--model-revision",
        required=True,
        help="exact model snapshot or revision recorded in both artifacts",
    )
    parser.add_argument(
        "--materialized-root",
        required=True,
        help="P9 quality materialization root containing MuirBench records and media",
    )
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument(
        "--page-artifact",
        required=True,
        help="atomic, resumable muirbench_dense_prefix_pages JSON output",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="final fit/knee/pressure working-set plan JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model_name.strip() or not args.model_revision.strip():
        raise ValueError("model name and revision must be non-empty")
    verify_working_set_model(
        {"model": {"revision": args.model_revision}},
        args.model,
    )
    materialized_root = Path(args.materialized_root).resolve()
    selection_path = Path(args.selection).resolve()
    records, source_identity = load_muirbench_records(
        materialized_root,
        selection_path=selection_path,
        subset=args.subset,
    )
    groups = build_media_first_groups(records)
    expected_artifact = _new_page_artifact(
        model_name=args.model_name,
        model_revision=args.model_revision,
        source_identity=source_identity,
        groups=groups,
    )
    page_artifact_path = Path(args.page_artifact).resolve()
    plan_path = Path(args.output).resolve()
    if page_artifact_path == plan_path:
        raise ValueError("page artifact and final plan must use different paths")
    artifact, completed = _load_page_artifact(
        page_artifact_path,
        expected=expected_artifact,
        groups=groups,
    )
    for group in groups:
        if str(group["group_id"]) in completed:
            _verified_group_media_paths(
                group,
                materialized_root=materialized_root,
            )
    missing = [group for group in groups if group["group_id"] not in completed]

    llm: LLM | None = None
    try:
        if missing:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "dense-prefix measurement requires exactly one visible CUDA device"
                )
            llm = _build_engine(args.model)
            verify_working_set_processor(llm.vl_processor, artifact)
            for index, group in enumerate(missing, start=len(completed) + 1):
                measurement = _measure_group(
                    llm,
                    group,
                    materialized_root=materialized_root,
                )
                completed[str(group["group_id"])] = measurement
                artifact["groups"] = [
                    completed[str(candidate["group_id"])]
                    for candidate in groups
                    if str(candidate["group_id"]) in completed
                ]
                artifact_sha256 = _write_json_atomic(page_artifact_path, artifact)
                print(
                    f"[{index}/{len(groups)}] {group['group_id']} "
                    f"dense_prefix_pages={measurement['dense_prefix_pages']} "
                    f"artifact_sha256={artifact_sha256}"
                )
    finally:
        if llm is not None:
            llm.exit()

    dense_prefix_pages = {
        group_id: int(measurement["dense_prefix_pages"])
        for group_id, measurement in completed.items()
    }
    plan = build_working_set_plan(
        records,
        dense_prefix_pages=dense_prefix_pages,
        model_name=args.model_name,
        model_revision=args.model_revision,
        materialization_identity=source_identity,
    )
    plan_sha256 = write_working_set_plan(plan_path, plan)
    print(
        json.dumps(
            {
                "page_artifact": str(page_artifact_path),
                "page_artifact_sha256": _sha256_file(page_artifact_path),
                "plan": str(plan_path),
                "plan_sha256": plan_sha256,
                "measured_groups": len(completed),
                "worksets": {
                    workset["id"]: {
                        "groups": workset["groups"],
                        "dense_prefix_pages": workset["dense_prefix_pages"],
                    }
                    for workset in plan["worksets"]
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
