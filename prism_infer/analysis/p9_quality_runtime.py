"""Shared runtime helpers for Prism and external P9 quality evaluators.

This module deliberately contains no inference-engine imports.  Both the Prism
and vLLM runners use it for materialized-record selection, media loading, input
identity construction, Git identity, and resumable artifact validation.  A
single implementation prevents the external baseline from silently drifting
from the frozen Prism evaluator at these non-framework boundaries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_materialization import (
    selected_ids_sha256,
    sha256_file,
)
from prism_infer.engine.vl_inputs import ImageInputs, VideoInputs

PreparedVisualInputs = ImageInputs | VideoInputs


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and reject arrays/scalars at the file boundary."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file whose every row is an object."""

    source = Path(path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL records: {source}")
    return records


def git_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Return the exact commit and dirty state of a benchmark harness."""

    root = Path(repo_root)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def materialization_artifact_by_id(
    manifest: Mapping[str, Any],
    dataset_id: str,
) -> Mapping[str, Any]:
    """Find one dataset artifact in a materialization manifest."""

    for artifact in manifest["datasets"]:
        if artifact["id"] == dataset_id:
            return artifact
    raise ValueError(f"materialization has no dataset {dataset_id!r}")


def safe_materialized_path(root: str | Path, relative: str) -> Path:
    """Resolve one regular file without allowing materialization-root escape."""

    materialized_root = Path(root).resolve()
    path = (materialized_root / relative).resolve()
    if not path.is_relative_to(materialized_root) or not path.is_file():
        raise ValueError(f"invalid materialized path: {relative!r}")
    return path


def load_record_images(
    record: Mapping[str, Any],
    *,
    materialized_root: str | Path,
) -> list[Image.Image]:
    """Load every record image as an owned RGB PIL object."""

    images = []
    for media in record["media"]:
        path = safe_materialized_path(
            materialized_root,
            media["materialized_path"],
        )
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def close_images(images: Sequence[Image.Image]) -> None:
    """Release all owned PIL images, including partially prepared samples."""

    for image in images:
        image.close()


def prepare_dataset_records(
    *,
    artifact: Mapping[str, Any],
    materialized_root: str | Path,
    subset: str,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Select eligible rows in frozen order and preserve explicit exclusions."""

    root = Path(materialized_root)
    records_path = root / artifact["selected_records"]["path"]
    records = read_jsonl_objects(records_path)
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


def quality_input_identity(
    inputs: PreparedVisualInputs,
    *,
    source_prompt: str,
    media_sha256: Sequence[str],
) -> dict[str, Any]:
    """Build the framework-neutral semantic identity of one visual request."""

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


def validate_resume_samples(
    artifact: Mapping[str, Any],
    *,
    run_identity_sha256: str,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Accept only an exact prefix checkpoint from the same frozen run."""

    if artifact.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("resume artifact run identity differs from requested run")
    samples = artifact.get("samples")
    if not isinstance(samples, list) or not all(
        isinstance(sample, dict) for sample in samples
    ):
        raise ValueError("resume artifact has no valid sample list")
    completed_ids = [sample["sample_id"] for sample in samples]
    if completed_ids != list(expected_ids[: len(completed_ids)]):
        raise ValueError("resume samples are not a prefix of frozen eligible IDs")
    return samples


def load_reference_records_for_artifacts(
    materialized_root: str | Path,
    artifacts: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve and verify the common reference JSONL bound to artifacts."""

    if not artifacts:
        raise ValueError("at least one quality artifact is required")
    root = Path(materialized_root).resolve()
    manifest_path = root / "p9_quality_materialization.json"
    manifest_sha256 = sha256_file(manifest_path)
    datasets = []
    for index, artifact in enumerate(artifacts):
        contract = artifact.get("run_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"artifact[{index}] has no run contract")
        if contract.get("materialization_manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"artifact[{index}] references a different materialization manifest"
            )
        dataset_id = contract.get("dataset")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"artifact[{index}] has no dataset identity")
        datasets.append(dataset_id)
    if len(set(datasets)) != 1:
        raise ValueError("quality artifacts do not reference the same dataset")

    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("materialization manifest has unsupported schema_version")
    if manifest.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("materialization manifest references a different protocol")
    manifest_datasets = manifest.get("datasets")
    if not isinstance(manifest_datasets, list):
        raise ValueError("materialization manifest has no dataset list")
    matches = [
        artifact
        for artifact in manifest_datasets
        if isinstance(artifact, dict) and artifact.get("id") == datasets[0]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"materialization manifest must contain dataset {datasets[0]!r} once"
        )
    selected = matches[0].get("selected_records")
    if not isinstance(selected, Mapping):
        raise ValueError("materialization dataset has no selected_records identity")
    relative_path = selected.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("materialization selected_records.path is invalid")
    records_path = safe_materialized_path(root, relative_path)
    if sha256_file(records_path) != selected.get("sha256"):
        raise ValueError("materialized selected records SHA256 mismatch")
    return read_jsonl_objects(records_path)


def selected_sample_identity(sample_ids: Sequence[str]) -> str:
    """Alias the frozen sample-ID hash for runner-facing code."""

    return selected_ids_sha256(sample_ids)
