#!/usr/bin/env python3
"""Run frozen P9 quality samples with vLLM BF16 or per-token-head FP8 KV.

Use the isolated vLLM environment and one in-process engine so the runner can
audit the concrete KV allocation and block-table tensors.  The generated
artifact is intentionally distinct from Prism's internal artifact: semantic
inputs must match exactly, while framework-native page/layout metadata remains
explicit instead of being forced into Prism's six-dimensional tensor schema.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_external_quality import (
    EXTERNAL_QUALITY_RECORD_TYPE,
    EXTERNAL_QUALITY_SCHEMA_VERSION,
    FROZEN_QWEN3_VL_PROCESSOR_MIN_PIXELS,
    TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES,
    VLLM_ATTENTION_BACKEND,
    VLLM_DISTRIBUTION_COMMIT,
    VLLM_FRAMEWORK_VERSION,
    VLLM_IMPLEMENTATION_FILES,
    VLLM_QUALITY_MODES,
    VLLM_REQUIRED_ENVIRONMENT,
    VLLM_TRANSFORMERS_VERSION,
    VLLM_VISION_ATTENTION_BACKEND,
    adapt_vllm_prompt_text,
    expected_vllm_kv_cache,
    vllm_framework_runtime,
)
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
    git_metadata,
    load_record_images,
    materialization_artifact_by_id,
    prepare_dataset_records,
    quality_input_identity,
    read_json_object,
    validate_resume_samples,
)
from prism_infer.analysis.p9_video_bundle import load_video_bundle
from prism_infer.engine.vl_inputs import (
    ImageInputs,
    VideoInputs,
    load_vl_processor,
    prepare_image_inputs,
    prepare_interleaved_image_inputs,
    prepare_video_inputs,
)
from scripts.verify_p9_quality_materialization import verify_materialization

DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_SELECTION = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/p9_quality/raw"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"
DEFAULT_PRISM_PYTHON = REPO_ROOT / ".venv-local/bin/python"
DEFAULT_VIDEO_HELPER = REPO_ROOT / "scripts/prepare_p9_video_sample.py"
DATASET_IDS = ("docvqa_validation", "muirbench_test", "mvbench_test")


def _source_file_hashes(
    package_root: Path,
    relative_paths: tuple[str, ...],
    *,
    distribution_name: str,
) -> dict[str, str]:
    identities = {}
    for relative in relative_paths:
        source = package_root / relative
        if not source.is_file():
            raise RuntimeError(
                f"installed {distribution_name} implementation file is missing: "
                f"{source}"
            )
        identities[relative] = sha256_file(source)
    return identities


def _distribution_identity() -> dict[str, Any]:
    """Hash the installed wheel record and the implementation-critical files."""

    import transformers
    import vllm
    from vllm._version import __commit_id__

    distribution = importlib.metadata.distribution("vllm")
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError("installed vLLM distribution has no RECORD identity")
    package_root = Path(vllm.__file__).resolve().parent
    transformers_root = Path(transformers.__file__).resolve().parent
    framework_version = importlib.metadata.version("vllm")
    distribution_commit = str(__commit_id__)
    transformers_version = transformers.__version__
    if framework_version != VLLM_FRAMEWORK_VERSION:
        raise RuntimeError("installed vLLM version differs from frozen P9 cell")
    if distribution_commit != VLLM_DISTRIBUTION_COMMIT:
        raise RuntimeError("installed vLLM commit differs from frozen P9 cell")
    if transformers_version != VLLM_TRANSFORMERS_VERSION:
        raise RuntimeError("installed Transformers version differs from frozen P9 cell")
    return {
        "framework": "vllm",
        "framework_version": framework_version,
        "framework_distribution_commit": distribution_commit,
        "framework_distribution_record_sha256": hashlib.sha256(
            record.encode("utf-8")
        ).hexdigest(),
        "framework_implementation_files": _source_file_hashes(
            package_root,
            VLLM_IMPLEMENTATION_FILES,
            distribution_name="vLLM",
        ),
        "transformers_processor_implementation_files": _source_file_hashes(
            transformers_root,
            TRANSFORMERS_PROCESSOR_IMPLEMENTATION_FILES,
            distribution_name="Transformers",
        ),
        "transformers": transformers_version,
    }


def _gpu_metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    metadata: dict[str, Any] = {
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }
    try:
        lines = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        if len(lines) != 1:
            raise RuntimeError("nvidia-smi did not identify exactly physical GPU0")
        uuid, driver = (part.strip() for part in lines[0].split(",", maxsplit=1))
        if not uuid.startswith("GPU-") or not driver:
            raise RuntimeError("nvidia-smi returned incomplete GPU identity")
        metadata.update({"gpu_uuid": uuid, "driver": driver})
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        raise RuntimeError("cannot establish the frozen GPU0 identity") from exc
    return metadata


def _load_model_config(model: str | Path) -> dict[str, Any]:
    path = Path(model).resolve() / "config.json"
    return read_json_object(path)


def _build_llm(
    model: str,
    *,
    framework_runtime: Mapping[str, Any],
) -> Any:
    from vllm import LLM

    return LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=framework_runtime["tensor_parallel_size"],
        max_model_len=framework_runtime["max_model_len"],
        max_num_seqs=framework_runtime["max_num_seqs"],
        max_num_batched_tokens=framework_runtime["max_num_batched_tokens"],
        gpu_memory_utilization=framework_runtime["gpu_memory_utilization"],
        block_size=framework_runtime["block_size"],
        num_gpu_blocks_override=framework_runtime["num_gpu_blocks_override"],
        kv_cache_dtype=framework_runtime["kv_cache_dtype"],
        enforce_eager=framework_runtime["enforce_eager"],
        enable_prefix_caching=framework_runtime["enable_prefix_caching"],
        enable_chunked_prefill=framework_runtime["enable_chunked_prefill"],
        async_scheduling=framework_runtime["async_scheduling"],
        mm_processor_cache_gb=framework_runtime["mm_processor_cache_gb"],
        limit_mm_per_prompt=framework_runtime["limit_mm_per_prompt"],
        mm_processor_kwargs=framework_runtime["mm_processor_kwargs"],
        attention_config={"backend": framework_runtime["attention_backend"]},
        disable_log_stats=True,
        seed=framework_runtime["seed"],
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _block_table_bytes(llm: Any) -> tuple[int, int]:
    """Read actual worker CPU/GPU block-table allocations in inproc mode."""

    executor = llm.llm_engine.model_executor
    worker = executor.driver_worker.worker
    block_tables = worker.model_runner.input_batch.block_table.block_tables
    if not block_tables:
        raise RuntimeError("vLLM model runner exposes no block tables")
    gpu_bytes = 0
    cpu_bytes = 0
    for table in block_tables:
        buffer = table.block_table
        gpu_bytes += _tensor_bytes(buffer.gpu)
        cpu_bytes += _tensor_bytes(buffer.cpu)
    return gpu_bytes, cpu_bytes


def _class_identity(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _backend_enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"vLLM backend enum has no stable name: {value!r}")
    return name


def _prompt_identity_mismatch(
    expected: list[int],
    actual: list[int],
) -> str:
    common = min(len(expected), len(actual))
    first_difference = next(
        (index for index in range(common) if expected[index] != actual[index]),
        common,
    )
    window_start = max(0, first_difference - 8)
    window_end = first_difference + 9
    return (
        f"expected_count={len(expected)}, actual_count={len(actual)}, "
        f"expected_sha256={canonical_json_sha256(expected)}, "
        f"actual_sha256={canonical_json_sha256(actual)}, "
        f"first_difference={first_difference}, "
        f"expected_window={expected[window_start:window_end]}, "
        f"actual_window={actual[window_start:window_end]}"
    )


def _execution_evidence(llm: Any) -> dict[str, Any]:
    """Read effective language/vision backends from the in-process worker."""

    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    language_backends = []
    for groups in runner.attn_groups:
        for group in groups:
            language_backends.append(
                {
                    "name": group.backend.get_name(),
                    "backend_class": _class_identity(group.backend),
                    "kv_cache_group_id": group.kv_cache_group_id,
                    "layer_names": sorted(group.layer_names),
                }
            )
    if not language_backends:
        raise RuntimeError("vLLM worker exposes no effective language backend")
    if {backend["name"] for backend in language_backends} != {VLLM_ATTENTION_BACKEND}:
        raise RuntimeError("vLLM effective language backend differs from frozen cell")

    visual = runner.model.visual
    vision_backend = visual.attn_backend
    vision_name = _backend_enum_name(vision_backend)
    layer_backend_names = {
        _backend_enum_name(block.attn.attn.attn_backend) for block in visual.blocks
    }
    if layer_backend_names != {vision_name}:
        raise RuntimeError("vLLM vision layers do not share the selected backend")
    if vision_name != VLLM_VISION_ATTENTION_BACKEND:
        raise RuntimeError("vLLM effective vision backend differs from frozen cell")
    selector_class = vision_backend.get_class()
    return {
        "required_environment": {
            name: os.environ[name] for name, _ in VLLM_REQUIRED_ENVIRONMENT
        },
        "language_attention_backends": language_backends,
        "vision_attention_backend": {
            "name": vision_name,
            "selector_class": _class_identity(selector_class),
            "layer_count": len(visual.blocks),
        },
    }


def _verify_processor_runtime(
    processor: Any,
    *,
    dataset_id: str,
    framework_runtime: Mapping[str, Any],
) -> None:
    modality = "video" if dataset_id == "mvbench_test" else "image"
    component = getattr(processor, f"{modality}_processor", None)
    size = getattr(component, "size", None)
    if size is None:
        raise RuntimeError(f"Qwen3-VL {modality} processor exposes no size contract")
    actual = {
        "shortest_edge": getattr(size, "shortest_edge", None),
        "longest_edge": getattr(size, "longest_edge", None),
    }
    expected = framework_runtime["mm_processor_kwargs"]["size"]
    if actual != expected:
        raise RuntimeError(
            f"Qwen3-VL {modality} processor size differs from frozen cell: "
            f"{actual!r} != {expected!r}"
        )
    if actual["shortest_edge"] != FROZEN_QWEN3_VL_PROCESSOR_MIN_PIXELS[modality]:
        raise RuntimeError(f"Qwen3-VL {modality} processor minimum changed")


def _vllm_prompt_text(
    processor: Any,
    *,
    dataset_id: str,
    prepared_prompt_text: str,
) -> str:
    modality = "video" if dataset_id == "mvbench_test" else "image"
    token_names = ("vision_start_token", "video_token", "vision_end_token")
    tokens = [getattr(processor, name, None) for name in token_names]
    if modality == "video" and not all(
        isinstance(token, str) and token for token in tokens
    ):
        raise RuntimeError("Qwen3-VL processor has no stable video marker tokens")
    return adapt_vllm_prompt_text(
        prepared_prompt_text,
        modality=modality,
        vision_start_token=tokens[0] or "",
        media_token=tokens[1] or "",
        vision_end_token=tokens[2] or "",
    )


def _kv_cache_record(
    llm: Any,
    *,
    mode: str,
    evaluator: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = expected_vllm_kv_cache(
        mode=mode,
        evaluator=evaluator,
        model_config=model_config,
    )
    engine_core = llm.llm_engine.engine_core.engine_core
    cache_config = engine_core.scheduler.kv_cache_config
    if cache_config.num_blocks != expected["num_blocks"]:
        raise RuntimeError(
            "vLLM effective block count differs from frozen override: "
            f"{cache_config.num_blocks} != {expected['num_blocks']}"
        )
    tensor_sizes = [int(tensor.size) for tensor in cache_config.kv_cache_tensors]
    gpu_table_bytes, cpu_table_bytes = _block_table_bytes(llm)
    return {
        **expected,
        "accounting_scope": (
            "allocated_gpu_kv_tensors_including_inline_per_token_head_scales"
        ),
        "reported_tensor_count": len(tensor_sizes),
        "reported_tensor_sizes": tensor_sizes,
        "reported_total_bytes": sum(tensor_sizes),
        "metadata_accounting": {
            "scope": (
                "actual_worker_block_table_tensors; Python allocator object graph "
                "not yet assigned a cross-framework byte contract"
            ),
            "gpu_block_table_bytes": gpu_table_bytes,
            "cpu_block_table_bytes": cpu_table_bytes,
            "allocator_structural_bytes": None,
            "complete_for_cross_framework_physical_claim": False,
        },
    }


def _prepare_video_bridge(
    *,
    record: Mapping[str, Any],
    materialized_root: Path,
    records_path: Path,
    evaluator_path: Path,
    prism_python: Path,
    video_helper: Path,
) -> tuple[list[Image.Image], dict[str, Any], dict[str, Any]]:
    if not prism_python.is_file():
        raise FileNotFoundError(f"frozen decoder Python is missing: {prism_python}")
    if not video_helper.is_file():
        raise FileNotFoundError(f"video bridge helper is missing: {video_helper}")
    with tempfile.TemporaryDirectory(prefix="p9-vllm-video-") as temporary:
        bundle_path = Path(temporary) / "frames.npz"
        completed = subprocess.run(
            [
                str(prism_python),
                str(video_helper),
                "--sample-id",
                record["sample_id"],
                "--output",
                str(bundle_path),
                "--evaluator",
                str(evaluator_path),
                "--records",
                str(records_path),
                "--materialized-root",
                str(materialized_root),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "no helper diagnostics"
            raise RuntimeError(
                "video bridge helper failed with exit code "
                f"{completed.returncode}: {details}"
            )
        helper_evidence = json.loads(completed.stdout)
        images, metadata, load_evidence = load_video_bundle(
            bundle_path,
            expected_sample_id=record["sample_id"],
            expected_source_media_sha256=record["media"][0]["sha256"],
        )
        if helper_evidence["sha256"] != load_evidence["sha256"]:
            close_images(images)
            raise RuntimeError("video helper and consumer bundle identities differ")
        if (
            helper_evidence["bundle_content_sha256"]
            != metadata["bundle_content_sha256"]
        ):
            close_images(images)
            raise RuntimeError("video helper and consumer content identities differ")
        bridge_evidence = {
            "helper_bundle_sha256": helper_evidence["sha256"],
            "bundle_content_sha256": metadata["bundle_content_sha256"],
        }
        return images, dict(metadata["video_sampling"]), bridge_evidence


def _prepare_sample(
    *,
    processor: Any,
    dataset_id: str,
    record: Mapping[str, Any],
    materialized_root: Path,
    records_path: Path,
    evaluator_path: Path,
    evaluator_dataset: Mapping[str, Any],
    prism_python: Path,
    video_helper: Path,
) -> tuple[
    ImageInputs | VideoInputs,
    dict[str, Any],
    list[Image.Image],
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    images: list[Image.Image] = []
    video_sampling = None
    bridge_evidence = None
    if dataset_id == "docvqa_validation":
        source_prompt = build_docvqa_prompt(record["question"])
        images = load_record_images(record, materialized_root=materialized_root)
        prepared = prepare_image_inputs(processor, source_prompt, images)
        multi_modal_data = {
            "image": images[0] if len(images) == 1 else images,
        }
    elif dataset_id == "muirbench_test":
        source_prompt = build_muirbench_prompt(record["question"], record["options"])
        images = load_record_images(record, materialized_root=materialized_root)
        prepared = prepare_interleaved_image_inputs(
            processor,
            source_prompt,
            images,
            image_marker=evaluator_dataset["image_marker"],
        )
        multi_modal_data = {"image": images}
    elif dataset_id == "mvbench_test":
        source_prompt = build_mvbench_prompt(record["question"], record["candidates"])
        images, video_sampling, bridge_evidence = _prepare_video_bridge(
            record=record,
            materialized_root=materialized_root,
            records_path=records_path,
            evaluator_path=evaluator_path,
            prism_python=prism_python,
            video_helper=video_helper,
        )
        prepared = prepare_video_inputs(
            processor,
            source_prompt,
            images,
            video_metadata=video_sampling,
        )
        frames = np.stack([np.asarray(image) for image in images])
        fps = float(video_sampling["fps"])
        source_frame_count = int(video_sampling["source_frame_count"])
        multi_modal_data = {
            "video": (
                frames,
                {
                    "total_num_frames": source_frame_count,
                    "fps": fps,
                    "duration": source_frame_count / fps,
                    "video_backend": "prism_p9_lossless_rgb_bridge",
                    "frames_indices": list(video_sampling["sampled_indices"]),
                    "do_sample_frames": False,
                },
            )
        }
    else:
        raise ValueError(f"unsupported quality dataset: {dataset_id!r}")
    prompt = {
        "prompt": _vllm_prompt_text(
            processor,
            dataset_id=dataset_id,
            prepared_prompt_text=prepared.prompt_text,
        ),
        "multi_modal_data": multi_modal_data,
    }
    return (
        prepared,
        prompt,
        images,
        source_prompt,
        video_sampling,
        bridge_evidence,
    )


def _run_sample(
    *,
    llm: Any,
    processor: Any,
    tokenizer: Any,
    sampling: Any,
    dataset_id: str,
    record: Mapping[str, Any],
    materialized_root: Path,
    records_path: Path,
    evaluator_path: Path,
    evaluator_dataset: Mapping[str, Any],
    runtime: Mapping[str, Any],
    prism_python: Path,
    video_helper: Path,
    muirbench_random: random.Random,
) -> dict[str, Any]:
    images: list[Image.Image] = []
    try:
        (
            prepared,
            prompt,
            images,
            source_prompt,
            video_sampling,
            bridge_evidence,
        ) = _prepare_sample(
            processor=processor,
            dataset_id=dataset_id,
            record=record,
            materialized_root=materialized_root,
            records_path=records_path,
            evaluator_path=evaluator_path,
            evaluator_dataset=evaluator_dataset,
            prism_python=prism_python,
            video_helper=video_helper,
        )
        if len(prepared.token_ids) + sampling.max_tokens > runtime["max_model_len"]:
            raise ValueError(
                f"sample {record['sample_id']} exceeds frozen model length"
            )
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("vLLM quality runner expected one completion")
        output = outputs[0]
        request_prompt_ids = list(output.prompt_token_ids or [])
        if request_prompt_ids != prepared.token_ids:
            raise RuntimeError(
                f"vLLM prompt token IDs differ from frozen preflight for "
                f"sample {record['sample_id']}: "
                f"{_prompt_identity_mismatch(prepared.token_ids, request_prompt_ids)}"
            )
        completion = output.outputs[0]
        output_token_ids = list(completion.token_ids)
        raw_prediction = tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True,
        )
        decoded_with_special_tokens = tokenizer.decode(
            output_token_ids,
            skip_special_tokens=False,
        )
        sample: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "input": quality_input_identity(
                prepared,
                source_prompt=source_prompt,
                media_sha256=[media["sha256"] for media in record["media"]],
            ),
            "framework_input": {
                "prompt_token_count": len(request_prompt_ids),
                "prompt_token_ids_sha256": canonical_json_sha256(request_prompt_ids),
                "matches_prepared_semantic_identity": True,
            },
            "raw_prediction": raw_prediction,
            "decoded_with_special_tokens": decoded_with_special_tokens,
            "output_token_ids": output_token_ids,
            "framework_output_text": completion.text,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "score": score_quality_prediction(
                dataset_id,
                record,
                raw_prediction,
                muirbench_random=muirbench_random,
            ),
        }
        if dataset_id == "mvbench_test":
            sample.update(
                {
                    "task": record["task"],
                    "video_sampling": video_sampling,
                    "video_frame_bridge": bridge_evidence,
                }
            )
        return sample
    finally:
        close_images(images)


def _shutdown_llm(llm: Any | None) -> None:
    if llm is not None:
        try:
            llm.llm_engine.engine_core.shutdown()
        finally:
            del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=DATASET_IDS, required=True)
    parser.add_argument(
        "--subset",
        choices=("development", "final"),
        default="development",
    )
    parser.add_argument("--mode", choices=sorted(VLLM_QUALITY_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    parser.add_argument("--prism-python", type=Path, default=DEFAULT_PRISM_PYTHON)
    parser.add_argument("--video-helper", type=Path, default=DEFAULT_VIDEO_HELPER)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    for name, expected in VLLM_REQUIRED_ENVIRONMENT:
        if os.environ.get(name) != expected:
            raise SystemExit(f"external P9 quality runs require {name}={expected}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("external P9 quality runs require exactly one visible GPU")

    from vllm import SamplingParams

    runtime_environment = {
        **_distribution_identity(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        **_gpu_metadata(),
    }
    materialized_root = args.materialized_root.resolve()
    verification = verify_materialization(
        protocol_path=args.protocol,
        selection_path=args.selection,
        raw_root=args.raw_root,
        materialized_root=materialized_root,
    )
    evaluator = read_json_object(args.evaluator)
    protocol = read_json_object(args.protocol)
    if evaluator["quality_protocol_sha256"] != canonical_json_sha256(protocol):
        raise SystemExit("evaluator references a different quality protocol")
    manifest_path = materialized_root / "p9_quality_materialization.json"
    materialization = read_json_object(manifest_path)
    dataset_artifact = materialization_artifact_by_id(
        materialization,
        args.dataset,
    )
    records, exclusions, selected_contract_ids = prepare_dataset_records(
        artifact=dataset_artifact,
        materialized_root=materialized_root,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    expected_ids = [record["sample_id"] for record in records]
    if not expected_ids:
        raise SystemExit("external quality run selected no eligible samples")
    records_path = materialized_root / dataset_artifact["selected_records"]["path"]
    max_media = max(len(record["media"]) for record in records)

    harness_git = git_metadata(REPO_ROOT)
    scope = (
        "smoke_not_quality_gate"
        if args.max_samples is not None
        else f"formal_{args.subset}"
    )
    if scope.startswith("formal_") and harness_git["dirty"]:
        raise SystemExit("formal external quality runs require a clean harness commit")
    runtime = evaluator["runtime"]
    evaluator_dataset = evaluator["datasets"][args.dataset]
    framework_runtime = vllm_framework_runtime(
        evaluator=evaluator,
        dataset_id=args.dataset,
        mode=args.mode,
        max_media_per_prompt=max_media,
    )
    model_config = _load_model_config(args.model)
    run_contract = {
        "dataset": args.dataset,
        "subset": args.subset,
        "scope": scope,
        "framework": "vllm",
        "framework_mode": args.mode,
        "model": str(Path(args.model).resolve()),
        "model_revision": evaluator["model"]["revision"],
        "processor_revision": evaluator["model"]["revision"],
        "model_config_canonical_sha256": canonical_json_sha256(model_config),
        "harness_git": harness_git,
        "environment": runtime_environment,
        "evaluator_sha256": canonical_json_sha256(evaluator),
        "materialization_manifest_sha256": sha256_file(manifest_path),
        "eligible_sample_ids_sha256": selected_ids_sha256(expected_ids),
        "semantic_runtime": {
            "max_model_len": runtime["max_model_len"],
            "image_max_pixels": runtime["image_max_pixels"],
            "video_max_pixels": runtime["video_max_pixels"],
            "video_frames": runtime["video_frames"],
            "sampling": runtime["sampling"],
        },
        "framework_runtime": framework_runtime,
        "dataset_evaluator": evaluator_dataset,
    }
    run_identity_sha256 = canonical_json_sha256(run_contract)
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists; pass --resume: {args.output}")
        output_artifact = read_json_object(args.output)
        samples = validate_resume_samples(
            output_artifact,
            run_identity_sha256=run_identity_sha256,
            expected_ids=expected_ids,
        )
        output_artifact.update({"status": "in_progress", "headline_eligible": False})
        output_artifact.pop("failure", None)
        write_json_atomic(args.output, output_artifact)
    else:
        samples: list[dict[str, Any]] = []
        output_artifact = {
            "schema_version": EXTERNAL_QUALITY_SCHEMA_VERSION,
            "record_type": EXTERNAL_QUALITY_RECORD_TYPE,
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
            replayed = score_quality_prediction(
                args.dataset,
                records[index],
                sample["raw_prediction"],
                muirbench_random=muirbench_random,
            )
            if replayed != sample["score"]:
                raise ValueError(
                    "resume MuirBench parser state differs from checkpoint"
                )

    processor = load_vl_processor(
        args.model,
        image_max_pixels=runtime["image_max_pixels"],
        video_max_pixels=runtime["video_max_pixels"],
    )
    _verify_processor_runtime(
        processor,
        dataset_id=args.dataset,
        framework_runtime=framework_runtime,
    )
    llm = None
    try:
        llm = _build_llm(args.model, framework_runtime=framework_runtime)
        output_artifact["environment"] = runtime_environment
        output_artifact["kv_cache"] = _kv_cache_record(
            llm,
            mode=args.mode,
            evaluator=evaluator,
            model_config=model_config,
        )
        output_artifact["execution_evidence"] = _execution_evidence(llm)
        if framework_runtime["attention_backend"] != VLLM_ATTENTION_BACKEND:
            raise RuntimeError("external quality cell did not retain Triton attention")
        sampling = SamplingParams(
            temperature=runtime["sampling"]["temperature"],
            max_tokens=evaluator_dataset["max_output_tokens"],
            ignore_eos=runtime["sampling"]["ignore_eos"],
        )
        tokenizer = llm.get_tokenizer()
        for record in records[len(samples) :]:
            sample = _run_sample(
                llm=llm,
                processor=processor,
                tokenizer=tokenizer,
                sampling=sampling,
                dataset_id=args.dataset,
                record=record,
                materialized_root=materialized_root,
                records_path=records_path,
                evaluator_path=args.evaluator.resolve(),
                evaluator_dataset=evaluator_dataset,
                runtime=runtime,
                # Keep the virtual-environment entrypoint intact. Resolving this
                # symlink would launch the system interpreter without the frozen
                # OpenCV/FFmpeg dependencies installed in .venv-local.
                prism_python=args.prism_python.absolute(),
                video_helper=args.video_helper.resolve(),
                muirbench_random=muirbench_random,
            )
            samples.append(sample)
            output_artifact["samples"] = samples
            output_artifact["completed_samples"] = len(samples)
            output_artifact["aggregate"] = aggregate_quality_predictions(
                args.dataset,
                samples,
            )
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
        _shutdown_llm(llm)

    output_artifact["status"] = "complete"
    output_artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_artifact["completed_samples"] = len(samples)
    output_artifact["aggregate"] = aggregate_quality_predictions(args.dataset, samples)
    output_artifact["headline_eligible"] = scope.startswith("formal_")
    output_sha256 = write_json_atomic(args.output, output_artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "status": output_artifact["status"],
                "scope": scope,
                "framework_mode": args.mode,
                "samples": len(samples),
                "aggregate": output_artifact["aggregate"],
                "kv_cache": output_artifact["kv_cache"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
