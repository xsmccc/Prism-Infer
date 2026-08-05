#!/usr/bin/env python3
"""Build a deterministic MVBench subset with exact repeated-media questions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.identity import canonical_json_sha256
from prism_infer.analysis.quality_materialization import (
    MATERIALIZATION_SCHEMA_VERSION,
    QUALITY_MATERIALIZATION_RECORD_TYPE,
    SampleSelection,
    selection_manifest_from_materialization,
    write_json_atomic,
)
from prism_infer.analysis.quality_protocol import load_quality_protocol
from scripts.materialize_quality_data import (
    DEFAULT_MVBENCH_MAP,
    DEFAULT_PROTOCOL,
    DEFAULT_RAW_ROOT,
    _artifact_record,
    _dataset_by_id,
    _load_json,
    _load_mvbench_population,
    _source_file_records,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/quality/mvbench_repeated"


def mvbench_source_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact pre-materialization source identity for one MVBench row."""

    media = record.get("media")
    if not isinstance(media, list) or len(media) != 1 or not isinstance(media[0], Mapping):
        raise ValueError("MVBench repeated-media selection requires exactly one media source")
    item = media[0]
    archive = item.get("archive")
    archive_name = None
    archive_path = None
    if archive is not None:
        if not isinstance(archive, Mapping):
            raise ValueError("MVBench media archive contract must be an object")
        archive_name = archive.get("name")
        archive_path = archive.get("repository_path")
    contract = {
        "archive_name": archive_name,
        "archive_repository_path": archive_path,
        "archive_member_path": item.get("archive_member_path"),
        "media_type": item.get("media_type"),
        "temporal_bound": record.get("temporal_bound"),
    }
    if not isinstance(contract["archive_member_path"], str) or not contract["archive_member_path"]:
        raise ValueError("MVBench media source has no archive member path")
    if contract["media_type"] not in ("frames", "video"):
        raise ValueError("MVBench media source has an unsupported media type")
    return contract


def select_repeated_mvbench_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select every exact source group containing at least two questions."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    contracts: dict[str, dict[str, Any]] = {}
    for record in records:
        contract = mvbench_source_contract(record)
        group_id = canonical_json_sha256(contract)
        groups[group_id].append(record)
        contracts[group_id] = contract

    selected_records: list[dict[str, Any]] = []
    selected_groups: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        rows = groups[group_id]
        if len(rows) < 2:
            continue
        ordered = sorted(
            rows,
            key=lambda row: (
                str(row.get("task", "")),
                int(row.get("question_index", -1)),
                str(row.get("sample_id", "")),
            ),
        )
        sample_ids = [str(row["sample_id"]) for row in ordered]
        selected_records.extend(dict(row) for row in ordered)
        selected_groups.append(
            {
                "source_contract_sha256": group_id,
                "source_contract": contracts[group_id],
                "samples": len(sample_ids),
                "sample_ids": sample_ids,
                "sample_ids_sha256": canonical_json_sha256(sample_ids),
            }
        )
    if not selected_groups:
        raise ValueError("MVBench population contains no exact repeated-media groups")
    return selected_records, selected_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mvbench-map", type=Path, default=DEFAULT_MVBENCH_MAP)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    selection_output = (
        args.selection_output.resolve()
        if args.selection_output is not None
        else output_root / "mvbench_repeated_selection.json"
    )

    protocol = load_quality_protocol(args.protocol)
    dataset = _dataset_by_id(protocol, "mvbench_test")
    media_map = _load_json(args.mvbench_map)
    if media_map.get("dataset_revision") != dataset["revision"]:
        raise ValueError("MVBench media map revision differs from quality protocol")
    _, rows_by_id, source_paths = _load_mvbench_population(
        args.raw_root / "mvbench",
        media_map,
    )
    records, groups = select_repeated_mvbench_records(list(rows_by_id.values()))
    sample_ids = tuple(str(record["sample_id"]) for record in records)
    selection = SampleSelection(
        population_samples=len(rows_by_id),
        development_ids=sample_ids,
        final_ids=sample_ids,
    )
    artifact = _artifact_record(
        dataset=dataset,
        selection=selection,
        source_files=_source_file_records(source_paths, repository_prefix="json"),
        records=records,
        records_path=output_root / "records/mvbench_test.final.jsonl",
        output_root=output_root,
        require_media_sha256=False,
    )
    archive_samples: dict[str, int] = defaultdict(int)
    archive_bytes: dict[str, int] = {}
    for record in records:
        archive = record["media"][0].get("archive")
        if archive is None:
            archive_samples["manual"] += 1
            continue
        archive_samples[str(archive["name"])] += 1
        archive_bytes[str(archive["name"])] = int(archive["bytes"])
    artifact.update(
        {
            "media_map_sha256": canonical_json_sha256(media_map),
            "repeated_media_selection": {
                "algorithm": "all_exact_mvbench_source_contract_groups",
                "groups": len(groups),
                "samples": len(records),
                "group_records": groups,
                "group_ids_sha256": canonical_json_sha256(
                    [group["source_contract_sha256"] for group in groups]
                ),
            },
            "pending_media_plan": {
                "selected_samples_by_archive": dict(sorted(archive_samples.items())),
                "full_archive_download_bytes": sum(archive_bytes.values()),
            },
        }
    )
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "record_type": QUALITY_MATERIALIZATION_RECORD_TYPE,
        "protocol_sha256": canonical_json_sha256(protocol),
        "selection_contract": {
            "algorithm": "all_exact_mvbench_source_contract_groups",
            "group_preimage": (
                "canonical JSON of archive name/path, member path, media type, temporal bound"
            ),
            "group_order": "source_contract_sha256_ascending",
            "within_group_order": "task_then_question_index_then_sample_id",
            "development_is_prefix_of_final": True,
        },
        "datasets": [artifact],
    }
    manifest_path = output_root / "quality_materialization.json"
    manifest_sha256 = write_json_atomic(manifest_path, manifest)
    selection_sha256 = write_json_atomic(
        selection_output,
        selection_manifest_from_materialization(manifest),
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "selection": str(selection_output),
                "selection_sha256": selection_sha256,
                "population_samples": len(rows_by_id),
                "repeated_groups": len(groups),
                "repeated_samples": len(records),
                "archives": dict(sorted(archive_samples.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
