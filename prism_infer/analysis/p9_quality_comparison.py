"""Validate P9 quality artifacts and run paired non-inferiority gates."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from prism_infer.analysis.benchmark_schema import (
    TORCH_DTYPE_ELEMENT_BYTES,
    canonical_json_sha256,
    percentile,
)
from prism_infer.analysis.p9_protocol import validate_p9_quality_protocol
from prism_infer.analysis.p9_quality_materialization import selected_ids_sha256
from prism_infer.analysis.p9_quality_metrics import (
    MUIRBENCH_RANDOM_FALLBACK_SEED,
    aggregate_quality_predictions,
    score_quality_prediction,
)
from prism_infer.engine.compression import (
    COMPRESSION_OFF,
    compression_mode_uses_fp8_payload,
    compression_mode_uses_token_head_scales,
    normalize_compression_mode,
)


QUALITY_ARTIFACT_SCHEMA_VERSION = 1
QUALITY_COMPARISON_SCHEMA_VERSION = 1
QUALITY_DATASETS = {
    "docvqa_validation",
    "muirbench_test",
    "mvbench_test",
}


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


def _number(
    value: object,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{path} must be finite and >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return numeric


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{path} must be a lowercase SHA256 digest")
    return digest


def _binary_score(value: object, path: str) -> int:
    score = _integer(value, path)
    if score not in (0, 1):
        raise ValueError(f"{path} must be 0 or 1")
    return score


def _tensor_bytes(dtype: str, shape: Sequence[int], path: str) -> int:
    element_bytes = TORCH_DTYPE_ELEMENT_BYTES.get(dtype)
    if element_bytes is None:
        raise ValueError(f"{path}.dtype has unknown element size: {dtype!r}")
    return math.prod(shape) * element_bytes


def _validate_kv_cache(cache: Mapping[str, Any], *, mode: str) -> None:
    payload_dtype = _string(
        cache.get("payload_dtype"), "artifact.kv_cache.payload_dtype"
    )
    payload_shape = _list(cache.get("payload_shape"), "artifact.kv_cache.payload_shape")
    if len(payload_shape) != 6 or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in payload_shape
    ):
        raise ValueError(
            "artifact.kv_cache.payload_shape must have 6 positive dimensions"
        )
    if payload_shape[0] != 2:
        raise ValueError("artifact.kv_cache.payload_shape[0] must represent K and V")

    expected_payload_dtype = (
        "torch.float8_e4m3fn"
        if compression_mode_uses_fp8_payload(mode)
        else "torch.bfloat16"
    )
    if payload_dtype != expected_payload_dtype:
        raise ValueError(
            "artifact.kv_cache payload dtype does not match compression mode: "
            f"{payload_dtype!r} != {expected_payload_dtype!r}"
        )
    payload_bytes = _integer(
        cache.get("payload_bytes"),
        "artifact.kv_cache.payload_bytes",
        minimum=1,
    )
    if payload_bytes != _tensor_bytes(
        payload_dtype,
        payload_shape,
        "artifact.kv_cache.payload",
    ):
        raise ValueError(
            "artifact.kv_cache.payload_bytes does not match dtype and shape"
        )

    scale_dtype = _string(cache.get("scale_dtype"), "artifact.kv_cache.scale_dtype")
    scale_shape = _list(cache.get("scale_shape"), "artifact.kv_cache.scale_shape")
    scale_bytes = _integer(cache.get("scale_bytes"), "artifact.kv_cache.scale_bytes")
    if compression_mode_uses_token_head_scales(mode):
        if scale_dtype != "torch.float32" or scale_shape != payload_shape[:-1]:
            raise ValueError(
                "scaled FP8 artifact requires FP32 token/head scales matching "
                "payload_shape[:-1]"
            )
        if scale_bytes != _tensor_bytes(
            scale_dtype,
            scale_shape,
            "artifact.kv_cache.scale",
        ):
            raise ValueError(
                "artifact.kv_cache.scale_bytes does not match dtype and shape"
            )
    elif scale_dtype != "none" or scale_shape or scale_bytes != 0:
        raise ValueError("non-scaled KV mode must not report a scale cache")

    total_bytes = _integer(
        cache.get("total_bytes"),
        "artifact.kv_cache.total_bytes",
        minimum=1,
    )
    if total_bytes != payload_bytes + scale_bytes:
        raise ValueError("artifact.kv_cache.total_bytes must equal payload + scales")


def _validate_visual_input(identity: Mapping[str, Any], path: str) -> None:
    modality = _string(identity.get("modality"), f"{path}.modality")
    if modality not in ("image", "video"):
        raise ValueError(f"{path}.modality must be image or video")
    grid_name = "image_grid_thw" if modality == "image" else "video_grid_thw"
    grid = _list(identity.get(grid_name), f"{path}.{grid_name}")
    if not grid:
        raise ValueError(f"{path}.{grid_name} must not be empty")
    expected_visual_tokens = 0
    for row_index, row in enumerate(grid):
        row = _list(row, f"{path}.{grid_name}[{row_index}]")
        if len(row) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in row
        ):
            raise ValueError(f"{path}.{grid_name}[{row_index}] must be [T, H, W]")
        raw_patches = math.prod(row)
        if raw_patches % 4:
            raise ValueError(
                f"{path}.{grid_name}[{row_index}] is not merge-size aligned"
            )
        expected_visual_tokens += raw_patches // 4
    visual_tokens = _integer(
        identity.get("visual_placeholder_tokens"),
        f"{path}.visual_placeholder_tokens",
        minimum=1,
    )
    if visual_tokens != expected_visual_tokens:
        raise ValueError(f"{path}.visual_placeholder_tokens does not match {grid_name}")


def _index_reference_records(
    reference_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(reference_records, (str, bytes)):
        raise ValueError("reference_records must be a sequence of objects")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(reference_records):
        record = _mapping(value, f"reference_records[{index}]")
        sample_id = _string(
            record.get("sample_id"),
            f"reference_records[{index}].sample_id",
        )
        if sample_id in indexed:
            raise ValueError(f"reference_records contains duplicate ID {sample_id!r}")
        indexed[sample_id] = record
    return indexed


def _validate_reference_score(
    sample: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    dataset_id: str,
    muirbench_random: random.Random | None,
    path: str,
) -> None:
    media = _list(reference.get("media"), f"{path}.reference.media")
    if not media:
        raise ValueError(f"{path}.reference.media must not be empty")
    reference_media_sha256 = []
    for media_index, value in enumerate(media):
        item = _mapping(value, f"{path}.reference.media[{media_index}]")
        reference_media_sha256.append(
            _sha256(
                item.get("sha256"),
                f"{path}.reference.media[{media_index}].sha256",
            )
        )
    if sample["input"]["media_sha256"] != reference_media_sha256:
        raise ValueError(f"{path}.input media SHA256 differs from reference record")

    if dataset_id == "mvbench_test":
        reference_task = _string(reference.get("task"), f"{path}.reference.task")
        if sample.get("task") != reference_task:
            raise ValueError(f"{path}.task differs from reference record")
    try:
        recomputed = score_quality_prediction(
            dataset_id,
            reference,
            sample["raw_prediction"],
            muirbench_random=muirbench_random,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} could not be scored from its reference record"
        ) from exc
    if sample["score"] != recomputed:
        raise ValueError(f"{path}.score differs from independently recomputed score")


def _validate_sample(
    sample: Mapping[str, Any],
    *,
    dataset_id: str,
    runtime: Mapping[str, Any],
    dataset_evaluator: Mapping[str, Any],
    path: str,
) -> str:
    sample_id = _string(sample.get("sample_id"), f"{path}.sample_id")
    identity = _mapping(sample.get("input"), f"{path}.input")
    for key in (
        "source_prompt_sha256",
        "chat_prompt_sha256",
        "prompt_token_ids_sha256",
    ):
        _sha256(identity.get(key), f"{path}.input.{key}")
    prompt_tokens = _integer(
        identity.get("prompt_token_count"),
        f"{path}.input.prompt_token_count",
        minimum=1,
    )
    media_sha256 = _list(identity.get("media_sha256"), f"{path}.input.media_sha256")
    if not media_sha256:
        raise ValueError(f"{path}.input.media_sha256 must not be empty")
    for media_index, digest in enumerate(media_sha256):
        _sha256(digest, f"{path}.input.media_sha256[{media_index}]")
    _validate_visual_input(identity, f"{path}.input")

    output_token_ids = _list(sample.get("output_token_ids"), f"{path}.output_token_ids")
    for token_index, token_id in enumerate(output_token_ids):
        _integer(token_id, f"{path}.output_token_ids[{token_index}]")
    raw_prediction = _string(
        sample.get("raw_prediction"),
        f"{path}.raw_prediction",
        allow_empty=True,
    )
    _string(
        sample.get("decoded_with_special_tokens"),
        f"{path}.decoded_with_special_tokens",
        allow_empty=True,
    )
    max_output_tokens = _integer(
        dataset_evaluator.get("max_output_tokens"),
        "evaluator.datasets[].max_output_tokens",
        minimum=1,
    )
    if len(output_token_ids) > max_output_tokens:
        raise ValueError(f"{path}.output_token_ids exceeds the evaluator output budget")
    max_model_len = _integer(
        runtime.get("max_model_len"),
        "evaluator.runtime.max_model_len",
        minimum=1,
    )
    if prompt_tokens + max_output_tokens > max_model_len:
        raise ValueError(f"{path} exceeds the frozen model length")

    score = _mapping(sample.get("score"), f"{path}.score")
    if dataset_id == "docvqa_validation":
        target = _list(score.get("target"), f"{path}.score.target")
        if not target or any(not isinstance(answer, str) for answer in target):
            raise ValueError(f"{path}.score.target must contain reference strings")
        _number(score.get("anls"), f"{path}.score.anls", maximum=1.0)
    elif dataset_id == "muirbench_test":
        target = _string(score.get("target"), f"{path}.score.target")
        for prefix in ("strict", "official"):
            prediction = score.get(f"{prefix}_prediction")
            if prediction is not None:
                _string(prediction, f"{path}.score.{prefix}_prediction")
            _string(
                score.get(f"{prefix}_parse_method"),
                f"{path}.score.{prefix}_parse_method",
            )
            observed = _binary_score(
                score.get(f"{prefix}_score"),
                f"{path}.score.{prefix}_score",
            )
            if observed != int(prediction == target):
                raise ValueError(f"{path}.score.{prefix}_score is inconsistent")
    else:
        target = _string(score.get("target"), f"{path}.score.target")
        prediction = score.get("prediction")
        if prediction is not None:
            _string(prediction, f"{path}.score.prediction")
        _string(score.get("parse_method"), f"{path}.score.parse_method")
        observed = _binary_score(score.get("score"), f"{path}.score.score")
        if observed != int(prediction == target):
            raise ValueError(f"{path}.score.score is inconsistent")
        answered = score.get("answered")
        if not isinstance(answered, bool) or answered != bool(raw_prediction.strip()):
            raise ValueError(f"{path}.score.answered is inconsistent")
        _string(sample.get("task"), f"{path}.task")
        video_sampling = _mapping(
            sample.get("video_sampling"), f"{path}.video_sampling"
        )
        indices = _list(
            video_sampling.get("sampled_indices"),
            f"{path}.video_sampling.sampled_indices",
        )
        expected_frames = _integer(
            runtime.get("video_frames"),
            "evaluator.runtime.video_frames",
            minimum=1,
        )
        if len(indices) != expected_frames:
            raise ValueError(
                f"{path}.video_sampling does not contain frozen frame count"
            )
        for frame_index, index in enumerate(indices):
            _integer(index, f"{path}.video_sampling.sampled_indices[{frame_index}]")
        _sha256(
            video_sampling.get("sampled_rgb_identity_sha256"),
            f"{path}.video_sampling.sampled_rgb_identity_sha256",
        )
        _number(video_sampling.get("fps"), f"{path}.video_sampling.fps", minimum=1e-12)
        _integer(
            video_sampling.get("source_frame_count"),
            f"{path}.video_sampling.source_frame_count",
            minimum=1,
        )
        source_kind = _string(
            video_sampling.get("source_kind"),
            f"{path}.video_sampling.source_kind",
        )
        decoder = video_sampling.get("decoder")
        if source_kind == "video_file":
            expected_decoder = _mapping(
                _mapping(
                    dataset_evaluator.get("video_sampling"),
                    "evaluator.datasets.mvbench_test.video_sampling",
                ).get("video_file_decoder"),
                "evaluator.datasets.mvbench_test.video_sampling.video_file_decoder",
            )
            if decoder != expected_decoder:
                raise ValueError(
                    f"{path}.video_sampling.decoder drifted from evaluator"
                )
        elif source_kind == "frame_directory":
            if decoder != "Pillow":
                raise ValueError(f"{path}.video_sampling.decoder must be Pillow")
        else:
            raise ValueError(f"{path}.video_sampling.source_kind is unsupported")
    return sample_id


def validate_quality_artifact(
    artifact: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    require_headline: bool = False,
    reference_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate one completed artifact without trusting stored score aggregates."""

    artifact = _mapping(artifact, "artifact")
    if artifact.get("schema_version") != QUALITY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact has unsupported schema_version")
    if artifact.get("record_type") != "p9_quality_predictions":
        raise ValueError("artifact has unsupported record_type")
    if artifact.get("status") != "complete":
        raise ValueError("quality artifact must be complete")

    evaluator = _mapping(evaluator, "evaluator")
    if evaluator.get("schema_version") != 1:
        raise ValueError("evaluator has unsupported schema_version")
    evaluator_sha256 = canonical_json_sha256(evaluator)
    contract = _mapping(artifact.get("run_contract"), "artifact.run_contract")
    if _sha256(
        artifact.get("run_identity_sha256"),
        "artifact.run_identity_sha256",
    ) != canonical_json_sha256(contract):
        raise ValueError("artifact.run_identity_sha256 does not match run_contract")
    if (
        _sha256(
            contract.get("evaluator_sha256"),
            "artifact.run_contract.evaluator_sha256",
        )
        != evaluator_sha256
    ):
        raise ValueError("artifact references a different evaluator")

    dataset_id = _string(contract.get("dataset"), "artifact.run_contract.dataset")
    if dataset_id not in QUALITY_DATASETS:
        raise ValueError(f"unsupported quality dataset: {dataset_id!r}")
    subset = _string(contract.get("subset"), "artifact.run_contract.subset")
    if subset not in ("development", "final"):
        raise ValueError("artifact.run_contract.subset is unsupported")
    scope = _string(contract.get("scope"), "artifact.run_contract.scope")
    formal_scope = f"formal_{subset}"
    if scope not in ("smoke_not_quality_gate", formal_scope):
        raise ValueError("artifact.run_contract.scope is inconsistent with subset")
    headline_eligible = artifact.get("headline_eligible")
    if not isinstance(headline_eligible, bool):
        raise ValueError("artifact.headline_eligible must be a bool")
    if headline_eligible != (scope == formal_scope):
        raise ValueError("artifact.headline_eligible is inconsistent with scope")
    if require_headline and not headline_eligible:
        raise ValueError("headline evidence requires a formal quality artifact")
    if headline_eligible and reference_records is None:
        raise ValueError(
            "formal quality artifacts require reference records to recompute scores"
        )

    mode = normalize_compression_mode(
        _string(contract.get("mode"), "artifact.run_contract.mode")
    )
    model = _string(contract.get("model"), "artifact.run_contract.model")
    if not model.startswith("/"):
        raise ValueError("artifact.run_contract.model must be an absolute path")
    if contract.get("model_revision") != _mapping(
        evaluator.get("model"), "evaluator.model"
    ).get("revision"):
        raise ValueError("artifact model revision differs from evaluator")
    runtime = _mapping(contract.get("runtime"), "artifact.run_contract.runtime")
    if runtime != _mapping(evaluator.get("runtime"), "evaluator.runtime"):
        raise ValueError("artifact runtime differs from evaluator")
    evaluator_datasets = _mapping(evaluator.get("datasets"), "evaluator.datasets")
    dataset_evaluator = _mapping(
        evaluator_datasets.get(dataset_id),
        f"evaluator.datasets.{dataset_id}",
    )
    if contract.get("dataset_evaluator") != dataset_evaluator:
        raise ValueError("artifact dataset evaluator differs from frozen evaluator")

    git = _mapping(contract.get("git"), "artifact.run_contract.git")
    commit = _string(git.get("commit"), "artifact.run_contract.git.commit")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("artifact.run_contract.git.commit must be a full Git revision")
    dirty = git.get("dirty")
    if not isinstance(dirty, bool):
        raise ValueError("artifact.run_contract.git.dirty must be a bool")
    if headline_eligible and dirty:
        raise ValueError("formal quality artifact cannot come from a dirty tree")

    environment = _mapping(artifact.get("environment"), "artifact.environment")
    for key in ("gpu", "torch", "cuda"):
        _string(environment.get(key), f"artifact.environment.{key}")
    _validate_kv_cache(
        _mapping(artifact.get("kv_cache"), "artifact.kv_cache"),
        mode=mode,
    )
    cache_shape = artifact["kv_cache"]["payload_shape"]
    if cache_shape[2] != runtime["num_kv_cache_blocks"]:
        raise ValueError("artifact KV block count differs from evaluator runtime")
    if cache_shape[3] != runtime["kv_cache_page_size"]:
        raise ValueError("artifact KV page size differs from evaluator runtime")

    samples = _list(artifact.get("samples"), "artifact.samples")
    if not samples:
        raise ValueError("artifact.samples must not be empty")
    references_by_id = (
        None
        if reference_records is None
        else _index_reference_records(reference_records)
    )
    muirbench_random = (
        random.Random(MUIRBENCH_RANDOM_FALLBACK_SEED)
        if references_by_id is not None and dataset_id == "muirbench_test"
        else None
    )
    sample_ids = []
    for index, sample in enumerate(samples):
        path = f"artifact.samples[{index}]"
        sample = _mapping(sample, path)
        sample_id = _validate_sample(
            sample,
            dataset_id=dataset_id,
            runtime=runtime,
            dataset_evaluator=dataset_evaluator,
            path=path,
        )
        sample_ids.append(sample_id)
        if references_by_id is not None:
            reference = references_by_id.get(sample_id)
            if reference is None:
                raise ValueError(f"{path} has no matching reference record")
            _validate_reference_score(
                sample,
                reference,
                dataset_id=dataset_id,
                muirbench_random=muirbench_random,
                path=path,
            )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("artifact.samples contains duplicate sample IDs")

    selection = _mapping(artifact.get("selection"), "artifact.selection")
    eligible_samples = _integer(
        selection.get("eligible_run_samples"),
        "artifact.selection.eligible_run_samples",
        minimum=1,
    )
    if eligible_samples != len(samples):
        raise ValueError("artifact sample count differs from eligible_run_samples")
    eligible_sha256 = selected_ids_sha256(sample_ids)
    if (
        _sha256(
            selection.get("eligible_run_ids_sha256"),
            "artifact.selection.eligible_run_ids_sha256",
        )
        != eligible_sha256
    ):
        raise ValueError("artifact eligible sample ID hash is inconsistent")
    if (
        _sha256(
            contract.get("eligible_sample_ids_sha256"),
            "artifact.run_contract.eligible_sample_ids_sha256",
        )
        != eligible_sha256
    ):
        raise ValueError("run contract eligible sample ID hash is inconsistent")
    _integer(
        selection.get("selected_contract_samples"),
        "artifact.selection.selected_contract_samples",
        minimum=eligible_samples,
    )
    _sha256(
        selection.get("selected_contract_ids_sha256"),
        "artifact.selection.selected_contract_ids_sha256",
    )
    _list(
        selection.get("protocol_exclusions"), "artifact.selection.protocol_exclusions"
    )

    completed = _integer(
        artifact.get("completed_samples"),
        "artifact.completed_samples",
        minimum=1,
    )
    if completed != len(samples):
        raise ValueError("artifact.completed_samples is inconsistent")
    recomputed = aggregate_quality_predictions(dataset_id, samples)
    if artifact.get("aggregate") != recomputed:
        raise ValueError("artifact.aggregate does not match per-sample scores")

    materialization = _mapping(
        artifact.get("materialization_verification"),
        "artifact.materialization_verification",
    )
    if materialization.get("status") != "PASS":
        raise ValueError("artifact materialization verification did not pass")
    manifest_sha256 = _sha256(
        materialization.get("manifest_sha256"),
        "artifact.materialization_verification.manifest_sha256",
    )
    if manifest_sha256 != _sha256(
        contract.get("materialization_manifest_sha256"),
        "artifact.run_contract.materialization_manifest_sha256",
    ):
        raise ValueError("artifact materialization manifest identity is inconsistent")


def paired_bootstrap_non_inferiority(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    margin: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Bootstrap paired sample deltas and test lower CI against ``-margin``."""

    if len(baseline_scores) != len(candidate_scores) or not baseline_scores:
        raise ValueError("paired bootstrap requires equal non-empty score vectors")
    margin = _number(margin, "margin", maximum=1.0)
    seed = _integer(seed, "seed", minimum=1)
    resamples = _integer(resamples, "resamples", minimum=1)
    baseline = [
        _number(value, f"baseline_scores[{index}]", maximum=1.0)
        for index, value in enumerate(baseline_scores)
    ]
    candidate = [
        _number(value, f"candidate_scores[{index}]", maximum=1.0)
        for index, value in enumerate(candidate_scores)
    ]
    differences = [right - left for left, right in zip(baseline, candidate)]
    rng = random.Random(seed)
    sample_count = len(differences)
    bootstrap_means = [
        sum(differences[rng.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(resamples)
    ]
    lower = percentile(bootstrap_means, 0.025)
    upper = percentile(bootstrap_means, 0.975)
    observed_delta = sum(differences) / sample_count
    return {
        "samples": sample_count,
        "baseline_mean": sum(baseline) / sample_count,
        "candidate_mean": sum(candidate) / sample_count,
        "delta_candidate_minus_baseline": observed_delta,
        "confidence_interval_95": {"lower": lower, "upper": upper},
        "interval_method": "paired_bootstrap_percentile_nearest_rank",
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "non_inferiority_margin": margin,
        "minimum_acceptable_delta": -margin,
        "pass": lower >= -margin,
    }


def _score_vectors(
    dataset_id: str,
    baseline_samples: Sequence[Mapping[str, Any]],
    candidate_samples: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[float], list[float], bool]]:
    if dataset_id == "docvqa_validation":
        specs = (("anls", "anls", True),)
    elif dataset_id == "muirbench_test":
        specs = (
            ("official_compatible_accuracy", "official_score", True),
            ("strict_accuracy_guardrail", "strict_score", True),
        )
    else:
        specs = (("selected_denominator_accuracy", "score", True),)
    return [
        (
            metric_name,
            [float(sample["score"][score_key]) for sample in baseline_samples],
            [float(sample["score"][score_key]) for sample in candidate_samples],
            required,
        )
        for metric_name, score_key, required in specs
    ]


def compare_quality_artifacts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    protocol: Mapping[str, Any],
    require_headline: bool = False,
    reference_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a baseline/candidate pair and emit deterministic gate evidence."""

    validate_p9_quality_protocol(protocol)
    protocol_sha256 = canonical_json_sha256(protocol)
    evaluator = _mapping(evaluator, "evaluator")
    if (
        _sha256(
            evaluator.get("quality_protocol_sha256"),
            "evaluator.quality_protocol_sha256",
        )
        != protocol_sha256
    ):
        raise ValueError("evaluator references a different quality protocol")
    validate_quality_artifact(
        baseline,
        evaluator=evaluator,
        require_headline=require_headline,
        reference_records=reference_records,
    )
    validate_quality_artifact(
        candidate,
        evaluator=evaluator,
        require_headline=require_headline,
        reference_records=reference_records,
    )

    baseline_contract = _mapping(baseline["run_contract"], "baseline.run_contract")
    candidate_contract = _mapping(candidate["run_contract"], "candidate.run_contract")
    baseline_mode = normalize_compression_mode(str(baseline_contract["mode"]))
    candidate_mode = normalize_compression_mode(str(candidate_contract["mode"]))
    if baseline_mode != COMPRESSION_OFF:
        raise ValueError("quality comparison baseline mode must be off")
    if candidate_mode == COMPRESSION_OFF:
        raise ValueError("quality comparison candidate mode must differ from off")
    baseline_common = {
        key: value for key, value in baseline_contract.items() if key != "mode"
    }
    candidate_common = {
        key: value for key, value in candidate_contract.items() if key != "mode"
    }
    if baseline_common != candidate_common:
        raise ValueError("quality pair run contracts differ beyond compression mode")
    if baseline["environment"] != candidate["environment"]:
        raise ValueError("quality pair environments differ")
    if baseline["selection"] != candidate["selection"]:
        raise ValueError("quality pair selections differ")
    if (
        baseline["materialization_verification"]
        != candidate["materialization_verification"]
    ):
        raise ValueError("quality pair materialization evidence differs")

    baseline_samples = baseline["samples"]
    candidate_samples = candidate["samples"]
    for index, (left, right) in enumerate(zip(baseline_samples, candidate_samples)):
        if left["sample_id"] != right["sample_id"]:
            raise ValueError(f"quality pair sample order differs at index {index}")
        if left["input"] != right["input"]:
            raise ValueError(f"quality pair input identity differs at index {index}")
        if left["score"]["target"] != right["score"]["target"]:
            raise ValueError(f"quality pair target differs at index {index}")

    non_inferiority = _mapping(
        protocol.get("non_inferiority"),
        "protocol.non_inferiority",
    )
    seed = _integer(
        non_inferiority.get("bootstrap_seed"),
        "protocol.non_inferiority.bootstrap_seed",
        minimum=1,
    )
    resamples = _integer(
        non_inferiority.get("bootstrap_resamples"),
        "protocol.non_inferiority.bootstrap_resamples",
        minimum=1,
    )
    dataset_id = str(baseline_contract["dataset"])
    margin = (
        _number(
            non_inferiority.get("normalized_generation_metric_margin"),
            "protocol.non_inferiority.normalized_generation_metric_margin",
        )
        if dataset_id == "docvqa_validation"
        else _number(
            non_inferiority.get("bounded_accuracy_margin_percentage_points"),
            "protocol.non_inferiority.bounded_accuracy_margin_percentage_points",
        )
        / 100.0
    )
    metric_results = {}
    required_passes = []
    for metric_name, left_scores, right_scores, required in _score_vectors(
        dataset_id,
        baseline_samples,
        candidate_samples,
    ):
        result = paired_bootstrap_non_inferiority(
            left_scores,
            right_scores,
            margin=margin,
            seed=seed,
            resamples=resamples,
        )
        result["required_for_gate"] = required
        metric_results[metric_name] = result
        if required:
            required_passes.append(bool(result["pass"]))

    all_required_pass = all(required_passes)
    formal_evidence = bool(
        baseline["headline_eligible"] and candidate["headline_eligible"]
    )
    if require_headline and not formal_evidence:
        raise ValueError("comparison requires formal headline artifacts")
    exact_tokens = sum(
        left["output_token_ids"] == right["output_token_ids"]
        for left, right in zip(baseline_samples, candidate_samples)
    )
    diagnostics: dict[str, Any] = {
        "exact_output_token_matches": exact_tokens,
        "exact_output_token_match_rate": exact_tokens / len(baseline_samples),
    }
    if dataset_id == "muirbench_test":
        diagnostics["official_random_fallback_samples"] = {
            "baseline": baseline["aggregate"]["official_random_fallback_samples"],
            "candidate": candidate["aggregate"]["official_random_fallback_samples"],
        }
    if dataset_id == "mvbench_test":
        diagnostics["answered_samples"] = {
            "baseline": baseline["aggregate"]["answered_samples"],
            "candidate": candidate["aggregate"]["answered_samples"],
        }

    baseline_cache = baseline["kv_cache"]
    candidate_cache = candidate["kv_cache"]
    decision = (
        ("PASS" if all_required_pass else "FAIL") if formal_evidence else "SMOKE_ONLY"
    )
    paired_input_identity = [
        {"sample_id": sample["sample_id"], "input": sample["input"]}
        for sample in baseline_samples
    ]
    return {
        "schema_version": QUALITY_COMPARISON_SCHEMA_VERSION,
        "record_type": "p9_quality_non_inferiority",
        "validation_status": "PASS",
        "dataset": dataset_id,
        "subset": baseline_contract["subset"],
        "scope": baseline_contract["scope"],
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "samples": len(baseline_samples),
        "protocol_sha256": protocol_sha256,
        "evaluator_sha256": canonical_json_sha256(evaluator),
        "baseline_artifact_sha256": canonical_json_sha256(baseline),
        "candidate_artifact_sha256": canonical_json_sha256(candidate),
        "paired_input_identity_sha256": canonical_json_sha256(paired_input_identity),
        "reference_scores_recomputed": reference_records is not None,
        "metrics": metric_results,
        "diagnostics": diagnostics,
        "kv_cache": {
            "baseline_total_bytes": baseline_cache["total_bytes"],
            "candidate_payload_bytes": candidate_cache["payload_bytes"],
            "candidate_scale_bytes": candidate_cache["scale_bytes"],
            "candidate_total_bytes": candidate_cache["total_bytes"],
            "candidate_to_baseline_total_ratio": (
                candidate_cache["total_bytes"] / baseline_cache["total_bytes"]
            ),
            "total_savings_fraction": 1.0
            - candidate_cache["total_bytes"] / baseline_cache["total_bytes"],
        },
        "all_required_metrics_pass": all_required_pass,
        "formal_evidence": formal_evidence,
        "headline_eligible": formal_evidence and all_required_pass,
        "decision": decision,
    }
