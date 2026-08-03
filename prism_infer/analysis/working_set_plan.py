"""Build deterministic repeated-media working sets for MuirBench.

The generated plan is inference-engine independent. Prism-Infer, vLLM, and
SGLang consume the same media groups, questions, and arrival schedule.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism_infer.analysis.p9_quality_metrics import build_muirbench_prompt

WORKING_SET_PLAN_SCHEMA_VERSION = 3
WORKING_SET_PLAN_RECORD_TYPE = "multimodal_working_set_plan"
DENSE_PREFIX_PAGES_SCHEMA_VERSION = 2
DENSE_PREFIX_PAGES_RECORD_TYPE = "muirbench_dense_prefix_pages"
MUIRBENCH_DATASET_ID = "muirbench_test"

DEFAULT_MODEL_NAME = "Qwen3-VL-8B-Instruct"
DEFAULT_KV_BUDGET_BYTES = 4_282_122_240
DEFAULT_KV_BUDGET_PAGES = 220
DEFAULT_PAGE_SIZE_TOKENS = 256
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MEASURED_REQUESTS = 600
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_REQUEST_RATE = 4.0
DEFAULT_ZIPF_ALPHA = 1.0
DEFAULT_SEED = 20260801
DEFAULT_IMAGE_MARKER = "<image>"
DEFAULT_IMAGE_MIN_PIXELS = 65_536
DEFAULT_IMAGE_MAX_PIXELS = 602_112
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_MAX_CHUNK_SIZE = DEFAULT_MAX_MODEL_LEN

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_WORKSET_IDS = ("fit", "knee", "pressure")
_GROUP_SELECTION_RNG_SALT = 0x47524F5550
_ARRIVAL_RNG_SALT = 0x4152524956414C


def load_muirbench_records(
    materialized_root: str | Path,
    *,
    selection_path: str | Path | None = None,
    subset: str = "final",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify a materialized MuirBench selection.

    Args:
        materialized_root: Directory containing the materialization manifest.
        selection_path: Optional selection manifest to bind to the same IDs.
        subset: Selection subset, normally ``final``.

    Returns:
        Selected records in source order and their source identity.
    """

    root = Path(materialized_root).resolve()
    manifest_path = root / "p9_quality_materialization.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("materialization manifest has unsupported schema_version")
    artifact = _dataset_by_id(manifest, MUIRBENCH_DATASET_ID, "materialization")
    selected = artifact.get("selected_records")
    if not isinstance(selected, Mapping):
        raise ValueError("MuirBench materialization has no selected_records identity")
    relative_records_path = selected.get("path")
    if not isinstance(relative_records_path, str) or not relative_records_path:
        raise ValueError("MuirBench selected_records.path is invalid")
    records_path = _safe_materialized_path(root, relative_records_path)
    records_sha256 = _sha256_file(records_path)
    if records_sha256 != selected.get("sha256"):
        raise ValueError("materialized MuirBench records SHA256 mismatch")

    all_records = _read_jsonl_objects(records_path)
    by_id = {record.get("sample_id"): record for record in all_records}
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get(subset), Mapping):
        raise ValueError(f"MuirBench materialization has no {subset!r} selection")
    selected_ids = selection[subset].get("sample_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or not all(isinstance(sample_id, str) and sample_id for sample_id in selected_ids)
    ):
        raise ValueError("MuirBench materialization has invalid selected sample IDs")
    if len(by_id) != len(all_records) or any(sample_id not in by_id for sample_id in selected_ids):
        raise ValueError("MuirBench selected records do not cover the selection")
    records = [by_id[sample_id] for sample_id in selected_ids]
    for record in records:
        media = record.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError(f"MuirBench sample {record.get('sample_id')!r} has no media")
        if any(not isinstance(item, Mapping) or item.get("sha256") is None for item in media):
            raise ValueError("MuirBench working-set records must have fully materialized media")
    record_ids = [str(record["sample_id"]) for record in records]

    source_identity: dict[str, Any] = {
        "materialization_manifest_sha256": _sha256_file(manifest_path),
        "selected_records_sha256": records_sha256,
        "selected_sample_ids_sha256": _selected_ids_sha256(record_ids),
        "subset": subset,
    }
    if selection_path is not None:
        source = Path(selection_path)
        selection = _read_json_object(source)
        selection_dataset = _dataset_by_id(selection, MUIRBENCH_DATASET_ID, "selection")
        selection_contract = selection_dataset.get("selection")
        if not isinstance(selection_contract, Mapping):
            raise ValueError("MuirBench selection manifest has no selection contract")
        selection_subset = selection_contract.get(subset)
        if not isinstance(selection_subset, Mapping):
            raise ValueError(f"MuirBench selection manifest has no {subset!r} subset")
        external_ids = selection_subset.get("sample_ids")
        if external_ids != selected_ids:
            raise ValueError("selection manifest IDs differ from materialized records")
        source_identity["selection_manifest_sha256"] = _sha256_file(source)
    return records, source_identity


def build_media_first_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    image_marker: str = DEFAULT_IMAGE_MARKER,
) -> list[dict[str, Any]]:
    """Group MuirBench records by their exact ordered media content."""

    if not records:
        raise ValueError("MuirBench working-set construction requires records")
    if not image_marker:
        raise ValueError("image_marker must be non-empty")

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    seen_sample_ids: set[str] = set()
    for record in records:
        sample_id = _nonempty_string(record.get("sample_id"), "record.sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate MuirBench sample_id: {sample_id!r}")
        seen_sample_ids.add(sample_id)
        media = record.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError(f"MuirBench sample {sample_id!r} has no media")
        ordered_digests = tuple(
            _sha256(item.get("sha256"), f"sample {sample_id!r} media[{index}].sha256")
            if isinstance(item, Mapping)
            else _raise_invalid_media(sample_id, index)
            for index, item in enumerate(media)
        )
        grouped.setdefault(ordered_digests, []).append(record)

    groups = []
    for ordered_digests, group_records in grouped.items():
        group_id = _canonical_json_sha256(list(ordered_digests))
        first_media = group_records[0]["media"]
        samples = []
        for sample_offset, record in enumerate(group_records):
            prompt = _media_first_source_prompt(record, image_marker=image_marker)
            samples.append(
                {
                    "sample_id": record["sample_id"],
                    "sample_offset": sample_offset,
                    "source_prompt": prompt,
                    "source_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
        groups.append(
            {
                "group_id": group_id,
                "ordered_media_sha256": list(ordered_digests),
                "media": [
                    _portable_media_record(item, path=f"group {group_id!r} media[{index}]")
                    for index, item in enumerate(first_media)
                ],
                "sample_ids": [sample["sample_id"] for sample in samples],
                "samples": samples,
            }
        )
    groups.sort(key=lambda group: group["group_id"])
    return groups


def load_dense_prefix_pages(path: str | Path) -> dict[str, int]:
    """Load per-media-group dense FP8 prefix page measurements.

    The accepted artifact is an object with a ``groups`` list.  This allows
    either a dedicated ``muirbench_dense_prefix_pages`` record or a previously
    generated working-set plan to serve as the measurement source.
    """

    artifact = _read_json_object(path)
    record_type = artifact.get("record_type")
    if record_type not in (DENSE_PREFIX_PAGES_RECORD_TYPE, WORKING_SET_PLAN_RECORD_TYPE):
        raise ValueError(f"unsupported dense-prefix page artifact: {record_type!r}")
    raw_groups = artifact.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("dense-prefix page artifact must contain groups")
    pages: dict[str, int] = {}
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"groups[{index}] must be an object")
        group_id = _sha256(group.get("group_id"), f"groups[{index}].group_id")
        dense_pages = _positive_int(
            group.get("dense_prefix_pages"),
            f"groups[{index}].dense_prefix_pages",
        )
        if group_id in pages:
            raise ValueError(f"duplicate dense-prefix page group: {group_id}")
        pages[group_id] = dense_pages
    return pages


def build_working_set_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    dense_prefix_pages: Mapping[str, int],
    model_revision: str,
    materialization_identity: Mapping[str, Any] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    kv_budget_bytes: int = DEFAULT_KV_BUDGET_BYTES,
    kv_budget_pages: int = DEFAULT_KV_BUDGET_PAGES,
    page_size_tokens: int = DEFAULT_PAGE_SIZE_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    measured_requests: int = DEFAULT_MEASURED_REQUESTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    request_rate: float = DEFAULT_REQUEST_RATE,
    zipf_alpha: float = DEFAULT_ZIPF_ALPHA,
    seed: int = DEFAULT_SEED,
    image_marker: str = DEFAULT_IMAGE_MARKER,
) -> dict[str, Any]:
    """Build a complete three-engine MuirBench working-set plan."""

    _nonempty_string(model_name, "model_name")
    _nonempty_string(model_revision, "model_revision")
    _positive_int(kv_budget_bytes, "kv_budget_bytes")
    _positive_int(kv_budget_pages, "kv_budget_pages")
    _positive_int(page_size_tokens, "page_size_tokens")
    _positive_int(max_model_len, "max_model_len")
    _positive_int(measured_requests, "measured_requests")
    _positive_int(max_new_tokens, "max_new_tokens")
    _positive_number(request_rate, "request_rate")
    _positive_number(zipf_alpha, "zipf_alpha")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")

    groups = build_media_first_groups(records, image_marker=image_marker)
    expected_group_ids = {group["group_id"] for group in groups}
    measured_group_ids = set(dense_prefix_pages)
    if expected_group_ids != measured_group_ids:
        missing = sorted(expected_group_ids - measured_group_ids)
        extra = sorted(measured_group_ids - expected_group_ids)
        raise ValueError(f"dense-prefix page coverage mismatch; missing={missing}, extra={extra}")
    for group in groups:
        group["dense_prefix_pages"] = _positive_int(
            dense_prefix_pages[group["group_id"]],
            f"dense_prefix_pages[{group['group_id']!r}]",
        )

    repeated_groups = [group for group in groups if len(group["samples"]) >= 2]
    if not repeated_groups:
        raise ValueError("MuirBench selection has no repeated-media question groups")
    selected_groups = _select_worksets(repeated_groups, kv_budget_pages=kv_budget_pages)
    worksets = [
        _build_workset(
            workset_id,
            workset_groups,
            kv_budget_pages=kv_budget_pages,
            measured_requests=measured_requests,
            request_rate=request_rate,
            zipf_alpha=zipf_alpha,
            seed=seed,
        )
        for workset_id, workset_groups in selected_groups.items()
    ]
    dataset = {
        "id": MUIRBENCH_DATASET_ID,
        "selected_samples": len(records),
        "media_groups": len(groups),
        "repeated_media_groups": len(repeated_groups),
        "repeated_media_questions": sum(len(group["samples"]) for group in repeated_groups),
        "selected_sample_ids_sha256": _selected_ids_sha256(
            [str(record["sample_id"]) for record in records]
        ),
    }
    if materialization_identity is not None:
        dataset["source_identity"] = dict(materialization_identity)
    plan = {
        "schema_version": WORKING_SET_PLAN_SCHEMA_VERSION,
        "record_type": WORKING_SET_PLAN_RECORD_TYPE,
        "dataset": dataset,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "max_model_len": max_model_len,
        },
        "kv_budget": {
            "bytes": kv_budget_bytes,
            "pages": kv_budget_pages,
            "page_size_tokens": page_size_tokens,
        },
        "processor": {
            "image_min_pixels": DEFAULT_IMAGE_MIN_PIXELS,
            "image_max_pixels": DEFAULT_IMAGE_MAX_PIXELS,
            "image_size": {
                "shortest_edge": DEFAULT_IMAGE_MIN_PIXELS,
                "longest_edge": DEFAULT_IMAGE_MAX_PIXELS,
            },
        },
        "serving": {
            "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
            "max_chunk_size": DEFAULT_MAX_CHUNK_SIZE,
        },
        "prompt_layout": {
            "name": "media_first",
            "image_marker": image_marker,
            "source_prompt_transform": "labeled_ordered_media_prefix_v1",
            "chat_content_order": "image_label_then_image_repeated_before_question",
        },
        "traffic": {
            "seed": seed,
            "population_policy": "one_request_per_group_closed_loop",
            "measured_requests": measured_requests,
            "max_new_tokens": max_new_tokens,
            "group_distribution": "zipf",
            "group_eligibility": "at_least_two_questions_per_ordered_media",
            "zipf_alpha": float(zipf_alpha),
            "zipf_rank_order": "group_id_ascending",
            "arrival_process": "poisson",
            "request_rate_per_s": float(request_rate),
            "arrival_offset_unit": "seconds_from_measured_phase_start",
            "rng_streams": _rng_stream_contract(seed),
        },
        "groups": groups,
        "worksets": worksets,
    }
    validate_working_set_plan(plan)
    return plan


def load_working_set_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate one shared working-set plan."""

    plan = _read_json_object(path)
    validate_working_set_plan(plan)
    return plan


def write_working_set_plan(path: str | Path, plan: Mapping[str, Any]) -> str:
    """Validate and atomically write one shared working-set plan."""

    validate_working_set_plan(plan)
    return _write_json_atomic(path, plan)


def validate_working_set_plan(plan: Mapping[str, Any]) -> None:
    """Validate plan identities, workset boundaries, and request schedules."""

    if plan.get("schema_version") != WORKING_SET_PLAN_SCHEMA_VERSION:
        raise ValueError("working-set plan has unsupported schema_version")
    if plan.get("record_type") != WORKING_SET_PLAN_RECORD_TYPE:
        raise ValueError("working-set plan has invalid record_type")
    dataset = _mapping(plan.get("dataset"), "plan.dataset")
    if dataset.get("id") != MUIRBENCH_DATASET_ID:
        raise ValueError("working-set plan dataset must be muirbench_test")
    model = _mapping(plan.get("model"), "plan.model")
    _nonempty_string(model.get("name"), "plan.model.name")
    _nonempty_string(model.get("revision"), "plan.model.revision")
    _positive_int(model.get("max_model_len"), "plan.model.max_model_len")
    budget = _mapping(plan.get("kv_budget"), "plan.kv_budget")
    _positive_int(budget.get("bytes"), "plan.kv_budget.bytes")
    budget_pages = _positive_int(budget.get("pages"), "plan.kv_budget.pages")
    _positive_int(budget.get("page_size_tokens"), "plan.kv_budget.page_size_tokens")
    processor = _mapping(plan.get("processor"), "plan.processor")
    if (
        processor.get("image_min_pixels") != DEFAULT_IMAGE_MIN_PIXELS
        or processor.get("image_max_pixels") != DEFAULT_IMAGE_MAX_PIXELS
        or processor.get("image_size")
        != {
            "shortest_edge": DEFAULT_IMAGE_MIN_PIXELS,
            "longest_edge": DEFAULT_IMAGE_MAX_PIXELS,
        }
    ):
        raise ValueError("working-set plan has an unsupported image processor contract")
    serving = _mapping(plan.get("serving"), "plan.serving")
    if (
        serving.get("max_num_seqs") != DEFAULT_MAX_NUM_SEQS
        or serving.get("max_chunk_size") != DEFAULT_MAX_CHUNK_SIZE
    ):
        raise ValueError("working-set plan requires max_num_seqs=8 and max_chunk_size=8192")
    prompt_layout = _mapping(plan.get("prompt_layout"), "plan.prompt_layout")
    if prompt_layout.get("name") != "media_first":
        raise ValueError("working-set plan requires media_first prompt layout")
    image_marker = _nonempty_string(
        prompt_layout.get("image_marker"), "plan.prompt_layout.image_marker"
    )
    if (
        prompt_layout.get("source_prompt_transform")
        != "labeled_ordered_media_prefix_v1"
    ):
        raise ValueError("working-set plan has unsupported source prompt transform")
    if (
        prompt_layout.get("chat_content_order")
        != "image_label_then_image_repeated_before_question"
    ):
        raise ValueError("working-set plan has unsupported chat content order")

    traffic = _mapping(plan.get("traffic"), "plan.traffic")
    seed = traffic.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("plan.traffic.seed must be a non-negative int")
    measured_requests = _positive_int(
        traffic.get("measured_requests"), "plan.traffic.measured_requests"
    )
    _positive_int(traffic.get("max_new_tokens"), "plan.traffic.max_new_tokens")
    request_rate = _positive_number(
        traffic.get("request_rate_per_s"), "plan.traffic.request_rate_per_s"
    )
    zipf_alpha = _positive_number(traffic.get("zipf_alpha"), "plan.traffic.zipf_alpha")
    if traffic.get("group_distribution") != "zipf" or traffic.get("arrival_process") != "poisson":
        raise ValueError("working-set plan has unsupported traffic distributions")
    if traffic.get("rng_streams") != _rng_stream_contract(seed):
        raise ValueError("working-set plan has invalid RNG stream identities")

    raw_groups = plan.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("working-set plan must contain groups")
    groups = [_validate_group(group, index, image_marker) for index, group in enumerate(raw_groups)]
    group_ids = [group["group_id"] for group in groups]
    if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
        raise ValueError("working-set groups must have unique ascending group IDs")
    all_sample_ids = [sample_id for group in groups for sample_id in group["sample_ids"]]
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise ValueError("working-set sample IDs must be globally unique")
    if dataset.get("selected_samples") != len(all_sample_ids):
        raise ValueError("dataset.selected_samples does not match grouped samples")
    if dataset.get("media_groups") != len(groups):
        raise ValueError("dataset.media_groups does not match groups")
    repeated_groups = [group for group in groups if len(group["samples"]) >= 2]
    if dataset.get("repeated_media_groups") != len(repeated_groups):
        raise ValueError("dataset.repeated_media_groups does not match groups")
    repeated_questions = sum(len(group["samples"]) for group in repeated_groups)
    if dataset.get("repeated_media_questions") != repeated_questions:
        raise ValueError("dataset.repeated_media_questions does not match groups")
    if traffic.get("group_eligibility") != "at_least_two_questions_per_ordered_media":
        raise ValueError("working-set traffic must contain repeated-media question groups")
    # Group sorting intentionally changes the source sample order. The
    # dataset identity binds the source order, so only validate its digest form.
    _sha256(dataset.get("selected_sample_ids_sha256"), "plan.dataset.selected_sample_ids_sha256")

    expected_worksets = _select_worksets(repeated_groups, kv_budget_pages=budget_pages)
    raw_worksets = plan.get("worksets")
    if not isinstance(raw_worksets, list) or len(raw_worksets) != len(_WORKSET_IDS):
        raise ValueError("working-set plan must contain fit, knee, and pressure")
    by_id = {}
    for workset in raw_worksets:
        if not isinstance(workset, Mapping):
            raise ValueError("plan.worksets[] must be an object")
        workset_id = workset.get("id")
        if workset_id in by_id:
            raise ValueError(f"duplicate workset: {workset_id!r}")
        by_id[workset_id] = workset
    if set(by_id) != set(_WORKSET_IDS):
        raise ValueError("working-set plan must contain fit, knee, and pressure")
    for workset_id in _WORKSET_IDS:
        workset_groups = expected_worksets[workset_id]
        expected = _build_workset(
            workset_id,
            workset_groups,
            kv_budget_pages=budget_pages,
            measured_requests=measured_requests,
            request_rate=request_rate,
            zipf_alpha=zipf_alpha,
            seed=seed,
        )
        if by_id[workset_id] != expected:
            raise ValueError(f"workset {workset_id!r} differs from deterministic contract")


def _select_worksets(
    groups: Sequence[Mapping[str, Any]],
    *,
    kv_budget_pages: int,
) -> dict[str, list[Mapping[str, Any]]]:
    if any(len(group["samples"]) < 2 for group in groups):
        raise ValueError("working sets require at least two questions per media group")
    fit_limit = math.floor(kv_budget_pages * 0.70)
    fit = []
    fit_pages = 0
    for group in groups:
        pages = int(group["dense_prefix_pages"])
        if fit_pages + pages > fit_limit:
            break
        fit.append(group)
        fit_pages += pages
    if not fit:
        raise ValueError("no media group fits within 70% of the KV page budget")

    knee = []
    knee_pages = 0
    for group in groups:
        knee.append(group)
        knee_pages += int(group["dense_prefix_pages"])
        if knee_pages > kv_budget_pages:
            break
    if knee_pages <= kv_budget_pages:
        raise ValueError("dense MuirBench groups do not cross the KV page budget")

    pressure_target = math.ceil(kv_budget_pages * 1.50)
    pressure = []
    pressure_pages = 0
    for group in groups:
        pressure.append(group)
        pressure_pages += int(group["dense_prefix_pages"])
        if pressure_pages >= pressure_target:
            break
    return {"fit": fit, "knee": knee, "pressure": pressure}


def _build_workset(
    workset_id: str,
    groups: Sequence[Mapping[str, Any]],
    *,
    kv_budget_pages: int,
    measured_requests: int,
    request_rate: float,
    zipf_alpha: float,
    seed: int,
) -> dict[str, Any]:
    dense_pages = sum(int(group["dense_prefix_pages"]) for group in groups)
    target_pages = {
        "fit": math.floor(kv_budget_pages * 0.70),
        "knee": kv_budget_pages,
        "pressure": math.ceil(kv_budget_pages * 1.50),
    }[workset_id]
    target_reached = {
        "fit": dense_pages <= target_pages,
        "knee": dense_pages > target_pages,
        "pressure": dense_pages >= target_pages,
    }[workset_id]
    population, measured = _build_requests(
        workset_id,
        groups,
        measured_requests=measured_requests,
        request_rate=request_rate,
        zipf_alpha=zipf_alpha,
        seed=seed,
    )
    previous_sample = {
        request["group_id"]: request["sample_id"] for request in population
    }
    question_switches = 0
    for request in measured:
        group_id = request["group_id"]
        if previous_sample[group_id] != request["sample_id"]:
            question_switches += 1
        previous_sample[group_id] = request["sample_id"]
    observed_questions = {
        request["sample_id"] for request in (*population, *measured)
    }
    return {
        "id": workset_id,
        "selection_rule": {
            "fit": "largest_hash_sorted_repeated_media_prefix_not_above_70pct_budget",
            "knee": "first_hash_sorted_repeated_media_prefix_above_budget",
            "pressure": (
                "first_hash_sorted_repeated_media_prefix_at_or_above_150pct_budget_or_all"
            ),
        }[workset_id],
        "target_dense_prefix_pages": target_pages,
        "target_reached": target_reached,
        "group_ids": [group["group_id"] for group in groups],
        "groups": len(groups),
        "available_questions": sum(len(group["samples"]) for group in groups),
        "observed_questions": len(observed_questions),
        "measured_question_switches": question_switches,
        "dense_prefix_pages": dense_pages,
        "population_requests": population,
        "measured_requests": measured,
    }


def _build_requests(
    workset_id: str,
    groups: Sequence[Mapping[str, Any]],
    *,
    measured_requests: int,
    request_rate: float,
    zipf_alpha: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population = [
        _request_record(
            request_id=f"{workset_id}:population:{index:04d}",
            phase="population",
            phase_index=index,
            group=group,
            group_rank=index + 1,
            sample_offset=0,
            arrival_offset_s=0.0,
        )
        for index, group in enumerate(groups)
    ]
    sample_cursors = {group["group_id"]: 1 % len(group["samples"]) for group in groups}
    weights = [1.0 / ((index + 1) ** zipf_alpha) for index in range(len(groups))]
    cumulative_weights = []
    total_weight = 0.0
    for weight in weights:
        total_weight += weight
        cumulative_weights.append(total_weight)
    group_rng = random.Random(seed ^ _GROUP_SELECTION_RNG_SALT)
    arrival_rng = random.Random(seed ^ _ARRIVAL_RNG_SALT)
    arrival_offset_s = 0.0
    measured = []
    for index in range(measured_requests):
        draw = group_rng.random() * total_weight
        group_index = min(
            bisect.bisect_left(cumulative_weights, draw),
            len(groups) - 1,
        )
        group = groups[group_index]
        group_id = group["group_id"]
        sample_offset = sample_cursors[group_id]
        sample_cursors[group_id] = (sample_offset + 1) % len(group["samples"])
        arrival_offset_s += arrival_rng.expovariate(request_rate)
        measured.append(
            _request_record(
                request_id=f"{workset_id}:measured:{index:06d}",
                phase="measured",
                phase_index=index,
                group=group,
                group_rank=group_index + 1,
                sample_offset=sample_offset,
                arrival_offset_s=arrival_offset_s,
            )
        )
    return population, measured


def _request_record(
    *,
    request_id: str,
    phase: str,
    phase_index: int,
    group: Mapping[str, Any],
    group_rank: int,
    sample_offset: int,
    arrival_offset_s: float,
) -> dict[str, Any]:
    sample = group["samples"][sample_offset]
    return {
        "request_id": request_id,
        "phase": phase,
        "phase_index": phase_index,
        "group_id": group["group_id"],
        "group_rank": group_rank,
        "sample_id": sample["sample_id"],
        "sample_offset": sample_offset,
        "arrival_offset_s": arrival_offset_s,
    }


def _validate_group(
    raw_group: object,
    index: int,
    image_marker: str,
) -> Mapping[str, Any]:
    group = _mapping(raw_group, f"plan.groups[{index}]")
    group_id = _sha256(group.get("group_id"), f"plan.groups[{index}].group_id")
    digests = group.get("ordered_media_sha256")
    if not isinstance(digests, list) or not digests:
        raise ValueError(f"plan.groups[{index}].ordered_media_sha256 must be non-empty")
    ordered_digests = [
        _sha256(digest, f"plan.groups[{index}].ordered_media_sha256[{media_index}]")
        for media_index, digest in enumerate(digests)
    ]
    if group_id != _canonical_json_sha256(ordered_digests):
        raise ValueError(f"plan.groups[{index}].group_id does not match ordered media")
    media = group.get("media")
    if not isinstance(media, list) or len(media) != len(ordered_digests):
        raise ValueError(f"plan.groups[{index}].media does not match ordered media")
    for media_index, (item, digest) in enumerate(zip(media, ordered_digests, strict=True)):
        media_item = _mapping(item, f"plan.groups[{index}].media[{media_index}]")
        if _sha256(media_item.get("sha256"), "media.sha256") != digest:
            raise ValueError(f"plan.groups[{index}].media[{media_index}] SHA256 mismatch")
        _nonempty_string(media_item.get("materialized_path"), "media.materialized_path")
    samples = group.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"plan.groups[{index}].samples must be non-empty")
    sample_ids = []
    for sample_index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"plan.groups[{index}].samples[{sample_index}]")
        sample_id = _nonempty_string(sample.get("sample_id"), "sample.sample_id")
        if sample.get("sample_offset") != sample_index:
            raise ValueError("group sample offsets must be contiguous")
        prompt = _nonempty_string(sample.get("source_prompt"), "sample.source_prompt")
        expected_prefix = "\n".join(
            f"Image {media_index}: {image_marker}"
            for media_index in range(1, len(media) + 1)
        )
        if not prompt.startswith(f"{expected_prefix}\n"):
            raise ValueError("media-first source prompt has no labeled media prefix")
        if prompt.count(image_marker) != len(media):
            raise ValueError("media-first source prompt marker count differs from media")
        digest = _sha256(sample.get("source_prompt_sha256"), "sample.source_prompt_sha256")
        if digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise ValueError("sample source prompt SHA256 mismatch")
        sample_ids.append(sample_id)
    if group.get("sample_ids") != sample_ids:
        raise ValueError(f"plan.groups[{index}].sample_ids does not match samples")
    _positive_int(group.get("dense_prefix_pages"), "group.dense_prefix_pages")
    return group


def _media_first_source_prompt(
    record: Mapping[str, Any],
    *,
    image_marker: str,
) -> str:
    question = _nonempty_string(record.get("question"), "record.question")
    options = record.get("options")
    if not isinstance(options, list):
        raise ValueError("record.options must be a list")
    media = record.get("media")
    if not isinstance(media, list) or not media:
        raise ValueError("record.media must be a non-empty list")
    return build_muirbench_labeled_media_first_prompt(
        question,
        options,
        expected_media_count=len(media),
        image_marker=image_marker,
    )


def build_muirbench_media_first_source_prompt(
    question: str,
    options: Sequence[str],
    *,
    expected_media_count: int,
    image_marker: str = "<image>",
) -> str:
    """Replace interleaved markers with references to the ordered image prefix."""

    if not image_marker:
        raise ValueError("image_marker must be non-empty")
    if expected_media_count <= 0:
        raise ValueError("expected_media_count must be positive")
    question = _nonempty_string(question, "question")
    option_text = [
        _nonempty_string(option, f"options[{index}]") for index, option in enumerate(options)
    ]
    observed_markers = question.count(image_marker) + sum(
        option.count(image_marker) for option in option_text
    )
    if observed_markers != expected_media_count:
        raise ValueError(
            "MuirBench marker count differs from ordered media: "
            f"markers={observed_markers}, media={expected_media_count}"
        )

    next_image = 0

    def replace_markers(text: str) -> str:
        nonlocal next_image
        parts = text.split(image_marker)
        rewritten = parts[0]
        for suffix in parts[1:]:
            next_image += 1
            rewritten += f"Image {next_image}{suffix}"
        return rewritten.strip()

    rewritten_question = replace_markers(question)
    rewritten_options = [replace_markers(option) for option in option_text]
    if next_image != expected_media_count:
        raise RuntimeError("ordered image references were not assigned exactly once")
    if not rewritten_question or any(not option for option in rewritten_options):
        raise ValueError("MuirBench media-first text contains an empty question or option")
    image_range = (
        "Image 1" if expected_media_count == 1 else f"Image 1 through Image {expected_media_count}"
    )
    reference_note = f"The images above are numbered {image_range} in order."
    return build_muirbench_prompt(
        f"{reference_note}\n{rewritten_question}",
        rewritten_options,
    )


def build_muirbench_labeled_media_first_prompt(
    question: str,
    options: Sequence[str],
    *,
    expected_media_count: int,
    image_marker: str = DEFAULT_IMAGE_MARKER,
) -> str:
    """Place explicitly numbered image blocks before the rewritten question."""

    source_prompt = build_muirbench_media_first_source_prompt(
        question,
        options,
        expected_media_count=expected_media_count,
        image_marker=image_marker,
    )
    media_prefix = "\n".join(
        f"Image {index}: {image_marker}" for index in range(1, expected_media_count + 1)
    )
    return f"{media_prefix}\n{source_prompt}"


def _rng_stream_contract(seed: int) -> dict[str, int | str]:
    return {
        "algorithm": "python_random_mt19937",
        "group_selection_seed": seed ^ _GROUP_SELECTION_RNG_SALT,
        "arrival_seed": seed ^ _ARRIVAL_RNG_SALT,
    }


def _portable_media_record(item: object, *, path: str) -> dict[str, Any]:
    media = _mapping(item, path)
    result = {
        "sha256": _sha256(media.get("sha256"), f"{path}.sha256"),
        "materialized_path": _nonempty_string(
            media.get("materialized_path"), f"{path}.materialized_path"
        ),
    }
    for key in ("width", "height", "format", "mode"):
        if key in media:
            result[key] = media[key]
    return result


def _dataset_by_id(
    manifest: Mapping[str, Any],
    dataset_id: str,
    label: str,
) -> Mapping[str, Any]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"{label} manifest has no datasets")
    matches = [
        dataset
        for dataset in datasets
        if isinstance(dataset, Mapping) and dataset.get("id") == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} manifest must contain {dataset_id!r} once")
    return matches[0]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _nonempty_string(value, path)
    if len(digest) != _SHA256_LENGTH or any(character not in _HEX_DIGITS for character in digest):
        raise ValueError(f"{path} must be a lowercase SHA256 digest")
    return digest


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive int")
    return value


def _positive_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{path} must be finite and positive")
    return numeric


def _raise_invalid_media(sample_id: str, index: int) -> str:
    raise ValueError(f"MuirBench sample {sample_id!r} media[{index}] must be an object")


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_ids_sha256(sample_ids: Sequence[str]) -> str:
    normalized = [_nonempty_string(sample_id, "sample_id") for sample_id in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("selected sample IDs must be unique")
    return _canonical_json_sha256(normalized)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def _read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL records: {source}")
    return records


def _safe_materialized_path(root: str | Path, relative: str) -> Path:
    materialized_root = Path(root).resolve()
    path = (materialized_root / relative).resolve()
    if not path.is_relative_to(materialized_root) or not path.is_file():
        raise ValueError(f"invalid materialized path: {relative!r}")
    return path


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
