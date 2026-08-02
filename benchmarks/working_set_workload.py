"""Materialize one shared multimodal working-set plan for online engines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from prism_infer.analysis.working_set_plan import (
    load_working_set_plan,
    validate_working_set_plan,
)


@dataclass(slots=True)
class MaterializedWorkingSet:
    """Owned image objects plus the exact population and measured schedules."""

    plan: dict[str, Any]
    workset: dict[str, Any]
    population_payloads: list[dict[str, Any]]
    population_request_ids: list[str]
    population_group_ids: list[str]
    population_sample_ids: list[str]
    population_offsets_s: list[float]
    measured_payloads: list[dict[str, Any]]
    measured_request_ids: list[str]
    measured_group_ids: list[str]
    measured_sample_ids: list[str]
    measured_offsets_s: list[float]
    _owned_images: list[Image.Image]

    def close(self) -> None:
        """Release every image opened while materializing the plan."""

        for image in self._owned_images:
            image.close()
        self._owned_images.clear()


def _resolve_media_path(materialized_root: Path, media: Mapping[str, Any]) -> Path:
    relative_path = media.get("materialized_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("working-set media requires materialized_path")
    path = (materialized_root / relative_path).resolve()
    if not path.is_relative_to(materialized_root) or not path.is_file():
        raise ValueError(f"invalid working-set media path: {relative_path!r}")
    expected_sha256 = media.get("sha256")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "working-set media SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} for {relative_path}"
        )
    return path


def _load_group_images(
    group: Mapping[str, Any],
    *,
    materialized_root: Path,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        for media in group["media"]:
            path = _resolve_media_path(materialized_root, media)
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


def _request_payload(
    request: Mapping[str, Any],
    *,
    groups: Mapping[str, Mapping[str, Any]],
    images_by_group: Mapping[str, Sequence[Image.Image]],
) -> tuple[dict[str, Any], str, float]:
    group_id = str(request["group_id"])
    sample_id = str(request["sample_id"])
    group = groups[group_id]
    samples = {str(sample["sample_id"]): sample for sample in group["samples"]}
    sample = samples.get(sample_id)
    if sample is None:
        raise ValueError(f"working-set request references unknown sample {sample_id!r}")
    images = images_by_group[group_id]
    payload = {
        "type": "images",
        "prompt": sample["source_prompt"],
        "images": list(images),
    }
    return payload, sample_id, float(request["arrival_offset_s"])


def materialize_working_set(
    plan_path: str | Path,
    *,
    workset_id: str,
    materialized_root: str | Path,
) -> MaterializedWorkingSet:
    """Load one workset while preserving its framework-neutral request order."""

    plan = load_working_set_plan(plan_path)
    validate_working_set_plan(plan)
    worksets = {str(item["id"]): item for item in plan["worksets"]}
    workset = worksets.get(workset_id)
    if workset is None:
        raise ValueError(f"working-set plan has no workset {workset_id!r}")
    groups = {str(group["group_id"]): group for group in plan["groups"]}
    group_ids = [str(group_id) for group_id in workset["group_ids"]]
    unknown_groups = [group_id for group_id in group_ids if group_id not in groups]
    if unknown_groups:
        raise ValueError(f"working set references unknown groups: {unknown_groups}")

    root = Path(materialized_root).resolve()
    images_by_group: dict[str, list[Image.Image]] = {}
    owned_images: list[Image.Image] = []
    try:
        for group_id in group_ids:
            images = _load_group_images(groups[group_id], materialized_root=root)
            images_by_group[group_id] = images
            owned_images.extend(images)

        def materialize_phase(
            phase: str,
        ) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[float]]:
            payloads: list[dict[str, Any]] = []
            request_ids: list[str] = []
            request_group_ids: list[str] = []
            sample_ids: list[str] = []
            offsets_s: list[float] = []
            for request in workset[f"{phase}_requests"]:
                payload, sample_id, offset_s = _request_payload(
                    request,
                    groups=groups,
                    images_by_group=images_by_group,
                )
                payloads.append(payload)
                request_ids.append(str(request["request_id"]))
                request_group_ids.append(str(request["group_id"]))
                sample_ids.append(sample_id)
                offsets_s.append(offset_s)
            return payloads, request_ids, request_group_ids, sample_ids, offsets_s

        population = materialize_phase("population")
        measured = materialize_phase("measured")
        return MaterializedWorkingSet(
            plan=plan,
            workset=workset,
            population_payloads=population[0],
            population_request_ids=population[1],
            population_group_ids=population[2],
            population_sample_ids=population[3],
            population_offsets_s=population[4],
            measured_payloads=measured[0],
            measured_request_ids=measured[1],
            measured_group_ids=measured[2],
            measured_sample_ids=measured[3],
            measured_offsets_s=measured[4],
            _owned_images=owned_images,
        )
    except BaseException:
        for image in owned_images:
            image.close()
        raise


def verify_working_set_model(
    plan: Mapping[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    """Bind a run to the model revision named by the shared plan."""

    revision = str(plan["model"]["revision"])
    resolved = Path(model_path).resolve()
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise ValueError(f"working-set model has no config.json: {resolved}")
    path_matches = revision in resolved.parts
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_revision = config.get("_commit_hash")
    if not path_matches and config_revision != revision:
        raise ValueError(
            "working-set model revision mismatch: "
            f"plan={revision!r}, path={resolved}, config={config_revision!r}"
        )
    return {
        "expected_revision": revision,
        "resolved_path": str(resolved),
        "config_commit_hash": config_revision,
        "matched_by": (
            "path_component_and_config_commit_hash"
            if path_matches and config_revision == revision
            else "path_component"
            if path_matches
            else "config_commit_hash"
        ),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def working_set_processor_kwargs(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Qwen3-VL image processor kwargs named by the plan."""

    processor = plan.get("processor")
    if not isinstance(processor, Mapping):
        raise ValueError("working-set plan has no processor contract")
    image_size = processor.get("image_size")
    if not isinstance(image_size, Mapping):
        raise ValueError("working-set plan has no image processor size")
    return {
        "min_pixels": int(image_size["shortest_edge"]),
        "max_pixels": int(image_size["longest_edge"]),
    }


def verify_working_set_processor(
    processor: Any,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the processor used to construct prompts against the shared plan."""

    processor_kwargs = working_set_processor_kwargs(plan)
    expected = {
        "shortest_edge": processor_kwargs["min_pixels"],
        "longest_edge": processor_kwargs["max_pixels"],
    }
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if size is None:
        raise ValueError("working-set processor exposes no image size contract")

    def value(name: str) -> Any:
        if isinstance(size, Mapping):
            return size.get(name)
        return getattr(size, name, None)

    actual = {
        "shortest_edge": value("shortest_edge"),
        "longest_edge": value("longest_edge"),
    }
    if actual != expected:
        raise ValueError(
            f"working-set image processor contract mismatch: expected {expected!r}, got {actual!r}"
        )
    return {
        "image_size": actual,
        "image_max_pixels": actual["longest_edge"],
    }


def source_prompt_schedule_sha256(payloads: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact source-prompt order before framework chat expansion."""

    prompts = []
    for index, payload in enumerate(payloads):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"working-set payload {index} has no source prompt")
        prompts.append(prompt)
    encoded = json.dumps(
        prompts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
