#!/usr/bin/env python3
"""独立复核 P9 quality source、selection、records 与每个媒体内容哈希。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_protocol import load_p9_quality_protocol
from prism_infer.analysis.p9_quality_materialization import (
    evaluation_subset_record,
    media_identity_record,
    selected_ids_sha256,
    selection_manifest_from_materialization,
    sha256_file,
)


DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_SELECTION = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/p9_quality/raw"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"


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


def _safe_materialized_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("materialized path must be a non-empty relative string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"materialized path escapes output root: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(f"materialized media does not exist: {path}")
    return path


def _verify_file_identity(
    path: Path,
    *,
    expected_bytes: object,
    expected_sha256: object,
    cache: dict[Path, tuple[int, str]],
) -> tuple[int, str]:
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise ValueError(f"invalid expected byte count for {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"invalid expected SHA256 for {path}")
    identity = cache.get(path)
    if identity is None:
        identity = (path.stat().st_size, sha256_file(path))
        cache[path] = identity
    if identity != (expected_bytes, expected_sha256):
        raise ValueError(
            f"file identity mismatch for {path}: "
            f"expected={(expected_bytes, expected_sha256)}, actual={identity}"
        )
    return identity


def _verify_media(
    media: Mapping[str, Any],
    *,
    root: Path,
    cache: dict[Path, tuple[int, str]],
) -> None:
    digest = media.get("sha256")
    status = media.get("materialization_status")
    if digest is None:
        if not (isinstance(status, str) and status.startswith("excluded_")):
            raise ValueError(f"unresolved media is not a protocol exclusion: {media}")
        if media.get("materialized_path") is not None:
            raise ValueError("excluded media must not point at a materialized file")
        return

    if media.get("identity_kind") == "canonical_frame_manifest_sha256":
        frames = media.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("frame-manifest media has no frames")
        frame_identity = []
        total_bytes = 0
        previous_member = ""
        for frame in frames:
            member = frame.get("archive_member_path")
            if not isinstance(member, str) or member <= previous_member:
                raise ValueError("frame members must be strictly sorted by source path")
            previous_member = member
            path = _safe_materialized_path(root, frame.get("materialized_path"))
            _verify_file_identity(
                path,
                expected_bytes=frame.get("bytes"),
                expected_sha256=frame.get("sha256"),
                cache=cache,
            )
            total_bytes += frame["bytes"]
            frame_identity.append(
                {"archive_member_path": member, "sha256": frame["sha256"]}
            )
        if media.get("bytes") != total_bytes:
            raise ValueError("frame-manifest aggregate bytes do not match frame files")
        if digest != canonical_json_sha256(frame_identity):
            raise ValueError("frame-manifest aggregate SHA256 does not match frame files")
        return

    path = _safe_materialized_path(root, media.get("materialized_path"))
    _verify_file_identity(
        path,
        expected_bytes=media.get("bytes"),
        expected_sha256=digest,
        cache=cache,
    )


def _source_path(
    *,
    dataset_id: str,
    repository_path: str,
    raw_root: Path,
) -> Path:
    if dataset_id == "docvqa_validation":
        return raw_root / "docvqa" / Path(repository_path).name
    if dataset_id == "muirbench_test":
        return raw_root / "muirbench" / Path(repository_path).name
    if dataset_id == "mvbench_test":
        return raw_root / "mvbench" / repository_path
    raise ValueError(f"unsupported quality dataset id: {dataset_id}")


def verify_materialization(
    *,
    protocol_path: Path,
    selection_path: Path,
    raw_root: Path,
    materialized_root: Path,
) -> dict[str, Any]:
    """执行完整只读验证；任一身份不一致立即抛错。"""

    protocol = load_p9_quality_protocol(protocol_path)
    manifest_path = materialized_root / "p9_quality_materialization.json"
    manifest = _read_json(manifest_path)
    if manifest.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("materialization protocol SHA256 mismatch")
    selection = _read_json(selection_path)
    expected_selection = selection_manifest_from_materialization(manifest)
    if selection != expected_selection:
        raise ValueError("tracked selection manifest differs from materialization")

    file_cache: dict[Path, tuple[int, str]] = {}
    source_cache: dict[Path, tuple[int, str]] = {}
    dataset_summaries = {}
    for artifact in manifest["datasets"]:
        dataset_id = artifact["id"]
        for source in artifact["source_files"]:
            path = _source_path(
                dataset_id=dataset_id,
                repository_path=source["repository_path"],
                raw_root=raw_root,
            )
            if not path.is_file():
                raise FileNotFoundError(f"raw quality source is missing: {path}")
            _verify_file_identity(
                path,
                expected_bytes=source["bytes"],
                expected_sha256=source["sha256"],
                cache=source_cache,
            )

        records_path = _safe_materialized_path(
            materialized_root,
            artifact["selected_records"]["path"],
        )
        if sha256_file(records_path) != artifact["selected_records"]["sha256"]:
            raise ValueError(f"selected records SHA256 mismatch for {dataset_id}")
        records = _read_jsonl(records_path)
        final_selection = artifact["selection"]["final"]
        record_ids = [record["sample_id"] for record in records]
        if record_ids != final_selection["sample_ids"]:
            raise ValueError(f"selected record order differs for {dataset_id}")
        if len(record_ids) != final_selection["samples"]:
            raise ValueError(f"selected record count differs for {dataset_id}")
        if selected_ids_sha256(record_ids) != final_selection[
            "selected_sample_ids_sha256"
        ]:
            raise ValueError(f"selected ID aggregate differs for {dataset_id}")

        for record in records:
            media = record.get("media")
            if not isinstance(media, list) or not media:
                raise ValueError(f"sample {record['sample_id']} has no media")
            for item in media:
                if not isinstance(item, Mapping):
                    raise ValueError(f"sample {record['sample_id']} has invalid media")
                _verify_media(item, root=materialized_root, cache=file_cache)
        actual_media_identity = media_identity_record(records)
        if actual_media_identity != artifact["media_identity"]:
            raise ValueError(f"media identity aggregate differs for {dataset_id}")
        actual_subset = evaluation_subset_record(records)
        if actual_subset != artifact["evaluation_subset"]:
            raise ValueError(f"evaluation subset differs for {dataset_id}")
        if actual_subset["status"] == "pending":
            raise ValueError(f"quality dataset still has pending media: {dataset_id}")
        dataset_summaries[dataset_id] = {
            "selected_samples": len(records),
            "eligible_samples": actual_subset["eligible_samples"],
            "excluded_samples": len(actual_subset["excluded_samples"]),
            "media_references": artifact["media_identity"][
                "sample_media_references"
            ],
        }

    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "source_files_verified": len(source_cache),
        "unique_media_files_verified": len(file_cache),
        "unique_media_bytes_verified": sum(size for size, _ in file_cache.values()),
        "datasets": dataset_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    args = parser.parse_args()
    result = verify_materialization(
        protocol_path=args.protocol,
        selection_path=args.selection,
        raw_root=args.raw_root,
        materialized_root=args.materialized_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
