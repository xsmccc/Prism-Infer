#!/usr/bin/env python3
"""运行冻结 P9 quality 子集并增量保存 raw prediction 与逐样本分数。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_materialization import (
    selected_ids_sha256,
    sha256_file,
    write_json_atomic,
)
from prism_infer.analysis.p9_quality_metrics import (
    MUIRBENCH_RANDOM_FALLBACK_SEED,
    aggregate_quality_predictions as aggregate_predictions,
    build_docvqa_prompt,
    build_muirbench_prompt,
    build_mvbench_prompt,
    score_quality_prediction as score_prediction,
)
from prism_infer.analysis.p9_video_sampling import (
    sample_frame_manifest,
    sample_video_file,
)
from prism_infer.engine.compression import SUPPORTED_COMPRESSION_MODES
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes
from prism_infer.engine.vl_inputs import (
    ImageInputs,
    VideoInputs,
    prepare_image_inputs,
    prepare_interleaved_image_inputs,
    prepare_video_inputs,
)
from scripts.verify_p9_quality_materialization import verify_materialization


QUALITY_RECORD_SCHEMA_VERSION = 1
DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_SELECTION = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/p9_quality/raw"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"
DATASET_IDS = ("docvqa_validation", "muirbench_test", "mvbench_test")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL records: {path}")
    return records


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def _artifact_by_id(
    manifest: Mapping[str, Any],
    dataset_id: str,
) -> Mapping[str, Any]:
    for artifact in manifest["datasets"]:
        if artifact["id"] == dataset_id:
            return artifact
    raise ValueError(f"materialization has no dataset {dataset_id!r}")


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid materialized path: {relative!r}")
    return path


def _load_images(
    record: Mapping[str, Any],
    *,
    materialized_root: Path,
) -> list[Image.Image]:
    images = []
    for media in record["media"]:
        path = _safe_path(materialized_root, media["materialized_path"])
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _close_images(images: Sequence[Image.Image]) -> None:
    for image in images:
        image.close()


def _input_identity(
    inputs: ImageInputs | VideoInputs,
    *,
    source_prompt: str,
    media_sha256: Sequence[str],
) -> dict[str, Any]:
    record = {
        "source_prompt_sha256": hashlib.sha256(
            source_prompt.encode("utf-8")
        ).hexdigest(),
        "chat_prompt_sha256": hashlib.sha256(
            inputs.prompt_text.encode("utf-8")
        ).hexdigest(),
        "prompt_token_count": len(inputs.token_ids),
        "prompt_token_ids_sha256": canonical_json_sha256(inputs.token_ids),
        "media_sha256": list(media_sha256),
    }
    if isinstance(inputs, ImageInputs):
        record.update(
            {
                "modality": "image",
                "image_grid_thw": inputs.image_grid_thw.tolist(),
                "visual_placeholder_tokens": inputs.image_token_count,
            }
        )
    else:
        record.update(
            {
                "modality": "video",
                "video_grid_thw": inputs.video_grid_thw.tolist(),
                "visual_placeholder_tokens": inputs.video_token_count,
            }
        )
    return record


def _prepare_dataset_records(
    *,
    artifact: Mapping[str, Any],
    materialized_root: Path,
    subset: str,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    records_path = materialized_root / artifact["selected_records"]["path"]
    records = _read_jsonl(records_path)
    by_id = {record["sample_id"]: record for record in records}
    selected_ids = artifact["selection"][subset]["sample_ids"]
    selected_records = [by_id[sample_id] for sample_id in selected_ids]
    eligible = []
    exclusions = []
    for record in selected_records:
        unresolved = [media for media in record["media"] if media.get("sha256") is None]
        if unresolved:
            reasons = sorted(
                str(media.get("materialization_status")) for media in unresolved
            )
            exclusions.append(
                {"sample_id": record["sample_id"], "reason": ",".join(reasons)}
            )
        else:
            eligible.append(record)
    if max_samples is not None:
        eligible = eligible[:max_samples]
    return eligible, exclusions, list(selected_ids)


def _build_llm(
    model: str,
    mode: str,
    runtime: Mapping[str, Any],
) -> LLM:
    return LLM(
        model,
        compression_mode=mode,
        enforce_eager=True,
        tensor_parallel_size=runtime["tensor_parallel_size"],
        max_model_len=runtime["max_model_len"],
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        max_num_seqs=runtime["max_num_seqs"],
        enable_chunked_prefill=runtime["enable_chunked_prefill"],
        max_chunk_size=runtime["max_chunk_size"],
        kvcache_block_size=runtime["kv_cache_page_size"],
        num_kvcache_blocks=runtime["num_kv_cache_blocks"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_prefix_caching=runtime["enable_prefix_caching"],
        image_max_pixels=runtime["image_max_pixels"],
        video_max_pixels=runtime["video_max_pixels"],
    )


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


def _run_sample(
    *,
    llm: LLM,
    dataset_id: str,
    record: Mapping[str, Any],
    materialized_root: Path,
    evaluator_dataset: Mapping[str, Any],
    runtime: Mapping[str, Any],
    sampling: SamplingParams,
    muirbench_random: random.Random,
) -> dict[str, Any]:
    images: list[Image.Image] = []
    video_sampling = None
    try:
        if dataset_id == "docvqa_validation":
            prompt = build_docvqa_prompt(record["question"])
            images = _load_images(record, materialized_root=materialized_root)
            inputs = prepare_image_inputs(llm.vl_processor, prompt, images)
        elif dataset_id == "muirbench_test":
            prompt = build_muirbench_prompt(record["question"], record["options"])
            images = _load_images(record, materialized_root=materialized_root)
            inputs = prepare_interleaved_image_inputs(
                llm.vl_processor,
                prompt,
                images,
                image_marker=evaluator_dataset["image_marker"],
            )
        elif dataset_id == "mvbench_test":
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
                path = _safe_path(materialized_root, media["materialized_path"])
                images, video_sampling = sample_video_file(
                    path,
                    frames=runtime["video_frames"],
                    temporal_bound=record["temporal_bound"],
                    decoder_contract=evaluator_dataset["video_sampling"][
                        "video_file_decoder"
                    ],
                )
            inputs = prepare_video_inputs(
                llm.vl_processor,
                prompt,
                images,
                video_metadata=video_sampling,
            )
        else:
            raise ValueError(f"unsupported dataset: {dataset_id}")
        if len(inputs.token_ids) + sampling.max_tokens > runtime["max_model_len"]:
            raise ValueError(
                f"sample {record['sample_id']} prompt + output budget exceeds "
                f"frozen model length: {len(inputs.token_ids)} + "
                f"{sampling.max_tokens} > {runtime['max_model_len']}"
            )
        if isinstance(inputs, ImageInputs):
            output = llm.generate_prepared_image_inputs(
                inputs,
                sampling,
                use_tqdm=False,
            )
        else:
            output = llm.generate_prepared_video_inputs(
                inputs,
                sampling,
                use_tqdm=False,
            )
        raw_prediction = output["text"]
        sample = {
            "sample_id": record["sample_id"],
            "input": _input_identity(
                inputs,
                source_prompt=prompt,
                media_sha256=[media["sha256"] for media in record["media"]],
            ),
            "raw_prediction": raw_prediction,
            "decoded_with_special_tokens": output["raw_text"],
            "output_token_ids": list(output["token_ids"]),
            "score": score_prediction(
                dataset_id,
                record,
                raw_prediction,
                muirbench_random=muirbench_random,
            ),
        }
        if dataset_id == "mvbench_test":
            sample["task"] = record["task"]
            sample["video_sampling"] = video_sampling
        return sample
    finally:
        _close_images(images)


def _validate_resume(
    artifact: Mapping[str, Any],
    *,
    run_identity_sha256: str,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if artifact.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("resume artifact run identity differs from requested run")
    samples = artifact.get("samples")
    if not isinstance(samples, list):
        raise ValueError("resume artifact has no sample list")
    completed_ids = [sample["sample_id"] for sample in samples]
    if completed_ids != list(expected_ids[: len(completed_ids)]):
        raise ValueError("resume samples are not a prefix of frozen eligible IDs")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=DATASET_IDS, required=True)
    parser.add_argument("--subset", choices=("development", "final"), default="development")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_COMPRESSION_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("P9 quality runs require CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("P9 quality runs require exactly one visible CUDA device")

    materialized_root = args.materialized_root.resolve()
    verification = verify_materialization(
        protocol_path=args.protocol,
        selection_path=args.selection,
        raw_root=args.raw_root,
        materialized_root=materialized_root,
    )
    evaluator = _read_json(args.evaluator)
    protocol = _read_json(args.protocol)
    if evaluator["quality_protocol_sha256"] != canonical_json_sha256(protocol):
        raise SystemExit("evaluator references a different quality protocol")
    manifest_path = materialized_root / "p9_quality_materialization.json"
    materialization = _read_json(manifest_path)
    artifact = _artifact_by_id(materialization, args.dataset)
    records, exclusions, selected_contract_ids = _prepare_dataset_records(
        artifact=artifact,
        materialized_root=materialized_root,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    expected_ids = [record["sample_id"] for record in records]
    if not expected_ids:
        raise SystemExit("quality run selected no eligible samples")

    git = _git_metadata()
    scope = (
        "smoke_not_quality_gate"
        if args.max_samples is not None
        else f"formal_{args.subset}"
    )
    if scope.startswith("formal_") and git["dirty"]:
        raise SystemExit("formal quality runs require a clean evaluator commit")
    runtime = evaluator["runtime"]
    evaluator_dataset = evaluator["datasets"][args.dataset]
    run_contract = {
        "dataset": args.dataset,
        "subset": args.subset,
        "scope": scope,
        "mode": args.mode,
        "model": str(Path(args.model).resolve()),
        "model_revision": evaluator["model"]["revision"],
        "git": git,
        "evaluator_sha256": canonical_json_sha256(evaluator),
        "materialization_manifest_sha256": sha256_file(manifest_path),
        "eligible_sample_ids_sha256": selected_ids_sha256(expected_ids),
        "runtime": runtime,
        "dataset_evaluator": evaluator_dataset,
    }
    run_identity_sha256 = canonical_json_sha256(run_contract)
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists; pass --resume: {args.output}")
        output_artifact = _read_json(args.output)
        samples = _validate_resume(
            output_artifact,
            run_identity_sha256=run_identity_sha256,
            expected_ids=expected_ids,
        )
    else:
        samples = []
        output_artifact = {
            "schema_version": QUALITY_RECORD_SCHEMA_VERSION,
            "record_type": "p9_quality_predictions",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "headline_eligible": False,
            "run_identity_sha256": run_identity_sha256,
            "run_contract": run_contract,
            "selection": {
                "selected_contract_samples": len(selected_contract_ids),
                "selected_contract_ids_sha256": selected_ids_sha256(
                    selected_contract_ids
                ),
                "eligible_run_samples": len(expected_ids),
                "eligible_run_ids_sha256": selected_ids_sha256(expected_ids),
                "protocol_exclusions": exclusions,
            },
            "materialization_verification": verification,
            "samples": samples,
            "aggregate": {"samples": 0},
        }
        write_json_atomic(args.output, output_artifact)

    muirbench_random = random.Random(MUIRBENCH_RANDOM_FALLBACK_SEED)
    for index, sample in enumerate(samples):
        if args.dataset == "muirbench_test":
            replayed = score_prediction(
                args.dataset,
                records[index],
                sample["raw_prediction"],
                muirbench_random=muirbench_random,
            )
            if replayed != sample["score"]:
                raise ValueError("resume MuirBench parser state differs from checkpoint")

    llm: LLM | None = None
    try:
        llm = _build_llm(args.model, args.mode, runtime)
        output_artifact["environment"] = {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        output_artifact["kv_cache"] = _cache_record(llm)
        if llm.vl_processor.image_processor.size.longest_edge != runtime[
            "image_max_pixels"
        ]:
            raise RuntimeError("runtime image pixel budget differs from evaluator")
        if llm.vl_processor.video_processor.size.longest_edge != runtime[
            "video_max_pixels"
        ]:
            raise RuntimeError("runtime video pixel budget differs from evaluator")
        sampling = SamplingParams(
            temperature=runtime["sampling"]["temperature"],
            max_tokens=evaluator_dataset["max_output_tokens"],
            ignore_eos=runtime["sampling"]["ignore_eos"],
        )
        for record in records[len(samples) :]:
            sample = _run_sample(
                llm=llm,
                dataset_id=args.dataset,
                record=record,
                materialized_root=materialized_root,
                evaluator_dataset=evaluator_dataset,
                runtime=runtime,
                sampling=sampling,
                muirbench_random=muirbench_random,
            )
            samples.append(sample)
            output_artifact["samples"] = samples
            output_artifact["aggregate"] = aggregate_predictions(
                args.dataset,
                samples,
            )
            output_artifact["completed_samples"] = len(samples)
            write_json_atomic(args.output, output_artifact)
    except BaseException as exc:
        output_artifact["status"] = "failed"
        output_artifact["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json_atomic(args.output, output_artifact)
        raise
    finally:
        if llm is not None:
            llm.exit()
            del llm
        gc.collect()
        torch.cuda.empty_cache()

    output_artifact["status"] = "complete"
    output_artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_artifact["completed_samples"] = len(samples)
    output_artifact["aggregate"] = aggregate_predictions(args.dataset, samples)
    output_artifact["headline_eligible"] = scope.startswith("formal_")
    output_sha256 = write_json_atomic(args.output, output_artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "status": output_artifact["status"],
                "scope": scope,
                "samples": len(samples),
                "aggregate": output_artifact["aggregate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
