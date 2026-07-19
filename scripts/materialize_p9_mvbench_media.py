#!/usr/bin/env python3
"""用严格 HTTP Range 从冻结 MVBench ZIP 中只物化 selected media。

脚本不会下载完整 archive，也不会使用近似文件名。每个远端请求必须返回可验证的
HTTP 206；每个 ZIP member 由标准库校验 CRC，再按内容 SHA256 落盘。官方 archive
确实缺失或需要 NTU 手工许可的样本保留在 frozen selection 中，并单独记录协议排除。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.http_range_reader import HTTPRangeReader
from prism_infer.analysis.p9_quality_materialization import (
    evaluation_subset_record,
    media_identity_record,
    selection_manifest_from_materialization,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/p9_quality/materialized"
DEFAULT_SELECTION_OUTPUT = REPO_ROOT / "benchmarks/workloads/p9_quality_selection.json"
COPY_BUFFER_BYTES = 1024 * 1024
VIDEO_SUFFIXES = {".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"}
FRAME_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return records


def _mvbench_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    for artifact in manifest["datasets"]:
        if artifact["id"] == "mvbench_test":
            return artifact
    raise ValueError("materialization manifest has no mvbench_test artifact")


def _archive_groups(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for media in record["media"]:
            archive = media.get("archive")
            if archive is not None:
                groups[archive["name"]].append(media)
    return dict(groups)


def _archive_url(
    artifact: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> str:
    repository = artifact["repository"]
    revision = artifact["revision"]
    repository_path = archive["repository_path"]
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{revision}/{repository_path}?download=true"
    )


def _selected_infos(
    archive: zipfile.ZipFile,
    media: Mapping[str, Any],
) -> list[zipfile.ZipInfo]:
    member_path = media["archive_member_path"]
    media_type = media["media_type"]
    if media_type == "frames":
        prefix = member_path.rstrip("/") + "/"
        return sorted(
            (
                info
                for info in archive.infolist()
                if info.filename.startswith(prefix) and not info.is_dir()
            ),
            key=lambda info: info.filename,
        )
    try:
        info = archive.getinfo(member_path)
    except KeyError:
        return []
    return [] if info.is_dir() else [info]


def _inventory_archive(
    archive: zipfile.ZipFile,
    selected_media: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique_infos: dict[str, zipfile.ZipInfo] = {}
    missing = []
    for media in selected_media:
        infos = _selected_infos(archive, media)
        if not infos:
            missing.append(media["archive_member_path"])
        for info in infos:
            unique_infos[info.filename] = info
    return {
        "selected_samples": len(selected_media),
        "unique_members": len(unique_infos),
        "missing_members": sorted(missing),
        "selected_compressed_bytes": sum(info.compress_size for info in unique_infos.values()),
        "selected_uncompressed_bytes": sum(info.file_size for info in unique_infos.values()),
    }


def _safe_suffix(info: zipfile.ZipInfo, *, media_type: str) -> str:
    suffix = PurePosixPath(info.filename).suffix.lower()
    allowed = FRAME_SUFFIXES if media_type == "frames" else VIDEO_SUFFIXES
    if suffix not in allowed:
        raise ValueError(f"unsupported {media_type} suffix in ZIP member {info.filename!r}")
    return suffix


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    archive_name: str,
    media_type: str,
    output_root: Path,
) -> dict[str, Any]:
    suffix = _safe_suffix(info, media_type=media_type)
    staging_identity = hashlib.sha256(
        f"{archive_name}\0{info.filename}".encode("utf-8")
    ).hexdigest()
    staging = output_root / ".staging" / f"{staging_identity}{suffix}.tmp"
    staging.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with archive.open(info, "r") as source, staging.open("wb") as target:
            while True:
                chunk = source.read(COPY_BUFFER_BYTES)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                copied_bytes += len(chunk)
        if copied_bytes != info.file_size:
            raise OSError(
                f"ZIP member size mismatch for {info.filename}: "
                f"expected={info.file_size}, actual={copied_bytes}"
            )
        media_sha256 = digest.hexdigest()
        relative_path = (
            PurePosixPath("media") / "mvbench_test" / media_sha256[:2] / f"{media_sha256}{suffix}"
        )
        target_path = output_root / Path(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            if (
                target_path.stat().st_size != copied_bytes
                or sha256_file(target_path) != media_sha256
            ):
                raise ValueError(f"existing MVBench media has wrong identity: {target_path}")
            staging.unlink()
        else:
            os.replace(staging, target_path)
        return {
            "archive_member_path": info.filename,
            "materialized_path": relative_path.as_posix(),
            "sha256": media_sha256,
            "bytes": copied_bytes,
        }
    finally:
        staging.unlink(missing_ok=True)


def _materialize_media(
    archive: zipfile.ZipFile,
    media: dict[str, Any],
    *,
    archive_name: str,
    output_root: Path,
    extracted_cache: dict[str, dict[str, Any]],
) -> None:
    infos = _selected_infos(archive, media)
    if not infos:
        media.update(
            {
                "materialization_status": "excluded_missing_frozen_archive_member",
                "exclusion_reason": (
                    "exact archive member is absent from the frozen dataset revision"
                ),
                "materialized_path": None,
                "sha256": None,
            }
        )
        return
    artifacts = []
    for info in infos:
        artifact = extracted_cache.get(info.filename)
        if artifact is None:
            artifact = _extract_member(
                archive,
                info,
                archive_name=archive_name,
                media_type=media["media_type"],
                output_root=output_root,
            )
            extracted_cache[info.filename] = artifact
        artifacts.append(artifact)
    if media["media_type"] == "frames":
        frame_identity = [
            {
                "archive_member_path": artifact["archive_member_path"],
                "sha256": artifact["sha256"],
            }
            for artifact in artifacts
        ]
        media.update(
            {
                "materialization_status": "complete",
                "identity_kind": "canonical_frame_manifest_sha256",
                "materialized_path": None,
                "sha256": canonical_json_sha256(frame_identity),
                "bytes": sum(artifact["bytes"] for artifact in artifacts),
                "frames": artifacts,
            }
        )
    else:
        if len(artifacts) != 1:
            raise ValueError("video media must map to exactly one ZIP member")
        artifact = artifacts[0]
        media.update(
            {
                "materialization_status": "complete",
                "identity_kind": "file_sha256",
                "materialized_path": artifact["materialized_path"],
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
            }
        )


def _exclude_manual_sources(records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        for media in record["media"]:
            if (
                media.get("archive") is None
                and media.get("materialization_status") == "pending_manual_source"
            ):
                media.update(
                    {
                        "materialization_status": ("excluded_manual_ntu_rgbd_license_required"),
                        "exclusion_reason": (
                            "frozen MVBench source requires separately licensed NTU RGB+D media"
                        ),
                    }
                )


def _refresh_artifacts(
    *,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    records_path: Path,
    manifest_path: Path,
    selection_output: Path | None,
) -> str:
    mvbench_artifact = _mvbench_artifact(manifest)
    mvbench_artifact["selected_records"]["sha256"] = write_jsonl_atomic(
        records_path,
        records,
    )
    for artifact in manifest["datasets"]:
        artifact_records_path = manifest_path.parent / artifact["selected_records"]["path"]
        artifact_records = (
            records if artifact["id"] == "mvbench_test" else _read_jsonl(artifact_records_path)
        )
        artifact["selected_records"]["sha256"] = sha256_file(artifact_records_path)
        artifact["media_identity"] = media_identity_record(artifact_records)
        subset = evaluation_subset_record(artifact_records)
        artifact["evaluation_subset"] = subset
        if subset["status"] == "pending":
            artifact["materialization_status"] = "media_materialization_in_progress"
        elif subset["excluded_samples"]:
            artifact["materialization_status"] = "complete_with_protocol_exclusions"
        else:
            artifact["materialization_status"] = "complete"

    pending_by_archive: dict[str, int] = {}
    pending_archive_bytes: dict[str, int] = {}
    pending_manual = 0
    for record in records:
        for media in record["media"]:
            if media.get("sha256") is not None:
                continue
            status = media.get("materialization_status")
            if isinstance(status, str) and status.startswith("excluded_"):
                continue
            archive = media.get("archive")
            if archive is None:
                pending_manual += 1
            else:
                name = archive["name"]
                pending_by_archive[name] = pending_by_archive.get(name, 0) + 1
                pending_archive_bytes[name] = archive["bytes"]
    mvbench_artifact["pending_media_plan"] = {
        "selected_samples_by_archive": dict(sorted(pending_by_archive.items())),
        "manual_source_samples": pending_manual,
        "full_archive_download_bytes": sum(pending_archive_bytes.values()),
    }
    manifest_sha256 = write_json_atomic(manifest_path, manifest)
    if selection_output is not None:
        write_json_atomic(
            selection_output,
            selection_manifest_from_materialization(manifest),
        )
    return manifest_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT,
    )
    parser.add_argument(
        "--archive",
        action="append",
        help="archive allowlist; repeat flag or omit to process every selected archive",
    )
    parser.add_argument("--range-chunk-mib", type=int, default=8)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="refresh deterministic manifests without opening remote archives",
    )
    parser.add_argument(
        "--exclude-unavailable-manual",
        action="store_true",
        help="record protocol exclusion for separately licensed NTU RGB+D media",
    )
    args = parser.parse_args()
    if args.range_chunk_mib <= 0:
        raise SystemExit("--range-chunk-mib must be positive")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "p9_quality_materialization.json"
    manifest = _read_json(manifest_path)
    artifact = _mvbench_artifact(manifest)
    records_path = output_root / artifact["selected_records"]["path"]
    records = _read_jsonl(records_path)
    if args.inventory_only and args.finalize_only:
        raise SystemExit("--inventory-only and --finalize-only are mutually exclusive")
    if args.finalize_only:
        if args.exclude_unavailable_manual:
            _exclude_manual_sources(records)
        manifest_sha256 = _refresh_artifacts(
            manifest=manifest,
            records=records,
            records_path=records_path,
            manifest_path=manifest_path,
            selection_output=args.selection_output,
        )
        subset = _mvbench_artifact(manifest)["evaluation_subset"]
        print(
            json.dumps(
                {
                    "manifest_sha256": manifest_sha256,
                    "evaluation_status": subset["status"],
                    "eligible_samples": subset["eligible_samples"],
                    "excluded_samples": len(subset["excluded_samples"]),
                    "pending_samples": len(subset["pending_sample_ids"]),
                },
                sort_keys=True,
            )
        )
        return
    groups = _archive_groups(records)
    selected_archives = sorted(set(args.archive or groups))
    unknown = set(selected_archives) - set(groups)
    if unknown:
        raise SystemExit(f"archives are not referenced by selected samples: {sorted(unknown)}")

    inventory: dict[str, Any] = {}
    for archive_name in selected_archives:
        selected_media = groups[archive_name]
        archive_spec = selected_media[0]["archive"]
        url = _archive_url(artifact, archive_spec)
        with HTTPRangeReader(
            url,
            expected_size=archive_spec["bytes"],
            chunk_bytes=args.range_chunk_mib * 1024 * 1024,
        ) as reader:
            with zipfile.ZipFile(reader) as archive:
                archive_inventory = _inventory_archive(archive, selected_media)
                archive_inventory.update(
                    {
                        "range_requests_before_extraction": reader.range_requests,
                        "range_bytes_before_extraction": reader.range_response_bytes,
                    }
                )
                inventory[archive_name] = archive_inventory
                if args.inventory_only:
                    continue
                extracted_cache: dict[str, dict[str, Any]] = {}
                for media in selected_media:
                    if media.get("sha256") is None and not str(
                        media.get("materialization_status", "")
                    ).startswith("excluded_"):
                        _materialize_media(
                            archive,
                            media,
                            archive_name=archive_name,
                            output_root=output_root,
                            extracted_cache=extracted_cache,
                        )
                archive_inventory.update(
                    {
                        "range_requests_total": reader.range_requests,
                        "range_response_bytes_total": reader.range_response_bytes,
                        "range_cache_hits": reader.cache_hits,
                    }
                )
        if not args.inventory_only:
            _refresh_artifacts(
                manifest=manifest,
                records=records,
                records_path=records_path,
                manifest_path=manifest_path,
                selection_output=args.selection_output,
            )
        print(json.dumps({archive_name: inventory[archive_name]}, sort_keys=True))

    if args.inventory_only:
        summary = {
            "archives": len(inventory),
            "selected_samples": sum(row["selected_samples"] for row in inventory.values()),
            "unique_members": sum(row["unique_members"] for row in inventory.values()),
            "missing_members": sum(len(row["missing_members"]) for row in inventory.values()),
            "selected_compressed_bytes": sum(
                row["selected_compressed_bytes"] for row in inventory.values()
            ),
            "selected_uncompressed_bytes": sum(
                row["selected_uncompressed_bytes"] for row in inventory.values()
            ),
        }
        print(json.dumps({"summary": summary}, sort_keys=True))
        return

    if args.exclude_unavailable_manual:
        _exclude_manual_sources(records)
    manifest_sha256 = _refresh_artifacts(
        manifest=manifest,
        records=records,
        records_path=records_path,
        manifest_path=manifest_path,
        selection_output=args.selection_output,
    )
    subset = _mvbench_artifact(manifest)["evaluation_subset"]
    print(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "evaluation_status": subset["status"],
                "eligible_samples": subset["eligible_samples"],
                "excluded_samples": len(subset["excluded_samples"]),
                "pending_samples": len(subset["pending_sample_ids"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
