#!/usr/bin/env python3
"""从冻结 revision 的本地原始文件物化 P9 标准质量子集。

DocVQA 与 MuirBench 的图片嵌在 parquet 中，本脚本只提取 SHA256 选中的 final
子集。MVBench 先冻结 4,000 条 metadata 上的选样和 archive member 路径；在视频内容
逐项取回并哈希前，它会明确保持 ``pending``，不能进入 Gate A。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_protocol import load_p9_quality_protocol
from prism_infer.analysis.p9_quality_materialization import (
    MATERIALIZATION_SCHEMA_VERSION,
    SELECTION_PREIMAGE_ENCODING,
    SampleSelection,
    build_mvbench_row,
    evaluation_subset_record,
    materialize_docvqa_row,
    materialize_muirbench_row,
    media_identity_record,
    mvbench_sample_id,
    select_sample_ids,
    selection_manifest_from_materialization,
    sha256_file,
    validate_selected_records,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_MVBENCH_MAP = REPO_ROOT / "benchmarks/workloads/p9_mvbench_media_map.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/p9_quality/raw"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/p9_quality/materialized"
SHARD_PATTERN = re.compile(r"^(?P<split>[a-z]+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.parquet$")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _dataset_by_id(protocol: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    for dataset in protocol["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    raise ValueError(f"quality protocol has no dataset {dataset_id!r}")


def _discover_complete_shards(directory: Path, *, split: str) -> list[Path]:
    """发现 parquet shard，并从文件名证明 index 连续且数量完整。"""

    matches: list[tuple[int, int, Path]] = []
    for path in sorted(directory.glob(f"{split}-*-of-*.parquet")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None or match.group("split") != split:
            raise ValueError(f"invalid parquet shard name: {path}")
        matches.append((int(match.group("index")), int(match.group("total")), path))
    if not matches:
        raise FileNotFoundError(f"no {split} parquet shards in {directory}")
    totals = {total for _, total, _ in matches}
    if len(totals) != 1:
        raise ValueError(f"inconsistent shard totals in {directory}: {sorted(totals)}")
    total = totals.pop()
    indices = [index for index, _, _ in matches]
    if indices != list(range(total)):
        raise ValueError(
            f"incomplete {split} shards in {directory}: "
            f"expected={list(range(total))}, actual={indices}"
        )
    return [path for _, _, path in matches]


def _parquet_batches(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """延迟导入可选 pyarrow，按 batch 读取避免一次加载完整媒体集。"""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "P9 quality materialization requires optional dependency pyarrow"
        ) from exc
    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
        ):
            yield batch.to_pylist()


def _parquet_ids(
    paths: Sequence[Path],
    *,
    id_column: str,
    batch_size: int,
) -> list[object]:
    sample_ids: list[object] = []
    for rows in _parquet_batches(
        paths,
        columns=[id_column],
        batch_size=batch_size,
    ):
        sample_ids.extend(row[id_column] for row in rows)
    return sample_ids


def _source_file_records(
    paths: Sequence[Path],
    *,
    repository_prefix: str,
) -> list[dict[str, Any]]:
    return [
        {
            "repository_path": f"{repository_prefix}/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _selection(
    sample_ids: Sequence[object],
    *,
    dataset: Mapping[str, Any],
    seed: int,
) -> SampleSelection:
    return select_sample_ids(
        sample_ids,
        dataset_id=dataset["id"],
        revision=dataset["revision"],
        seed=seed,
        development_samples=dataset["development_samples"],
        final_samples=dataset["final_samples"],
    )


def _selected_parquet_records(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    id_column: str,
    selection: SampleSelection,
    batch_size: int,
    materialize_row: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = set(selection.final_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for rows in _parquet_batches(
        paths,
        columns=columns,
        batch_size=batch_size,
    ):
        for row in rows:
            sample_id = str(row[id_column])
            if sample_id not in selected_ids:
                continue
            if sample_id in by_id:
                raise ValueError(f"selected parquet id occurs twice: {sample_id}")
            by_id[sample_id] = materialize_row(row)
    missing = selected_ids - set(by_id)
    if missing:
        raise ValueError(f"selected parquet rows are missing: {sorted(missing)}")
    return [by_id[sample_id] for sample_id in selection.final_ids]


def _artifact_record(
    *,
    dataset: Mapping[str, Any],
    selection: SampleSelection,
    source_files: list[dict[str, Any]],
    records: list[dict[str, Any]],
    records_path: Path,
    output_root: Path,
    require_media_sha256: bool,
) -> dict[str, Any]:
    validate_selected_records(
        records,
        selection,
        require_media_sha256=require_media_sha256,
    )
    records_sha256 = write_jsonl_atomic(records_path, records)
    media_identity = media_identity_record(records)
    relative_records_path = records_path.relative_to(output_root).as_posix()
    return {
        "id": dataset["id"],
        "category": dataset["category"],
        "repository": dataset["repository"],
        "revision": dataset["revision"],
        "config": dataset.get("config"),
        "split": dataset["split"],
        "sample_id_field": dataset["sample_id_field"],
        "source_files": source_files,
        "selection": selection.to_record(),
        "selected_records": {
            "path": relative_records_path,
            "sha256": records_sha256,
        },
        "media_identity": media_identity,
        "evaluation_subset": evaluation_subset_record(records),
        "materialization_status": (
            "complete"
            if media_identity["status"] == "complete"
            else "metadata_selected_media_pending"
        ),
    }


def _materialize_docvqa(
    *,
    protocol: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset = _dataset_by_id(protocol, "docvqa_validation")
    paths = _discover_complete_shards(raw_root / "docvqa", split="validation")
    selection = _selection(
        _parquet_ids(paths, id_column="questionId", batch_size=batch_size),
        dataset=dataset,
        seed=seed,
    )
    records = _selected_parquet_records(
        paths,
        columns=[
            "questionId",
            "question",
            "question_types",
            "image",
            "docId",
            "ucsf_document_id",
            "ucsf_document_page_no",
            "answers",
        ],
        id_column="questionId",
        selection=selection,
        batch_size=batch_size,
        materialize_row=lambda row: materialize_docvqa_row(
            row,
            output_root=output_root,
            dataset_id=dataset["id"],
        ),
    )
    return _artifact_record(
        dataset=dataset,
        selection=selection,
        source_files=_source_file_records(paths, repository_prefix="DocVQA"),
        records=records,
        records_path=output_root / "records/docvqa_validation.final.jsonl",
        output_root=output_root,
        require_media_sha256=True,
    )


def _materialize_muirbench(
    *,
    protocol: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset = _dataset_by_id(protocol, "muirbench_test")
    paths = _discover_complete_shards(raw_root / "muirbench", split="test")
    selection = _selection(
        _parquet_ids(paths, id_column="idx", batch_size=batch_size),
        dataset=dataset,
        seed=seed,
    )
    records = _selected_parquet_records(
        paths,
        columns=[
            "idx",
            "task",
            "image_relation",
            "image_type",
            "question",
            "options",
            "answer",
            "image_list",
            "counterpart_idx",
        ],
        id_column="idx",
        selection=selection,
        batch_size=batch_size,
        materialize_row=lambda row: materialize_muirbench_row(
            row,
            output_root=output_root,
            dataset_id=dataset["id"],
        ),
    )
    return _artifact_record(
        dataset=dataset,
        selection=selection,
        source_files=_source_file_records(paths, repository_prefix="data"),
        records=records,
        records_path=output_root / "records/muirbench_test.final.jsonl",
        output_root=output_root,
        require_media_sha256=True,
    )


def _load_mvbench_population(
    raw_directory: Path,
    media_map: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], list[Path]]:
    task_specs = media_map.get("tasks")
    if not isinstance(task_specs, Mapping) or not task_specs:
        raise ValueError("MVBench media map has no tasks")
    json_paths = sorted((raw_directory / "json").glob("*.json"))
    task_names = {path.stem for path in json_paths}
    if task_names != set(task_specs):
        raise ValueError(
            "MVBench JSON task set differs from frozen media map: "
            f"missing={sorted(set(task_specs) - task_names)}, "
            f"extra={sorted(task_names - set(task_specs))}"
        )
    sample_ids: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    archives = media_map.get("archives")
    if not isinstance(archives, Mapping):
        raise ValueError("MVBench media map has no archives")
    for path in json_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"MVBench task JSON must be a non-empty list: {path}")
        task = path.stem
        task_media = task_specs[task]
        for question_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"MVBench {task}[{question_index}] is not an object")
            video = row.get("video")
            if not isinstance(video, str):
                raise ValueError(f"MVBench {task}[{question_index}] has no video")
            sample_id = mvbench_sample_id(task, video, question_index)
            sample_ids.append(sample_id)
            rows_by_id[sample_id] = build_mvbench_row(
                row,
                task=task,
                question_index=question_index,
                task_media=task_media,
                archives=archives,
            )
    return sample_ids, rows_by_id, json_paths


def _materialize_mvbench(
    *,
    protocol: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    seed: int,
    media_map: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _dataset_by_id(protocol, "mvbench_test")
    if media_map.get("dataset_revision") != dataset["revision"]:
        raise ValueError("MVBench media map revision differs from quality protocol")
    sample_ids, rows_by_id, json_paths = _load_mvbench_population(
        raw_root / "mvbench",
        media_map,
    )
    selection = _selection(sample_ids, dataset=dataset, seed=seed)
    records = [rows_by_id[sample_id] for sample_id in selection.final_ids]
    artifact = _artifact_record(
        dataset=dataset,
        selection=selection,
        source_files=_source_file_records(json_paths, repository_prefix="json"),
        records=records,
        records_path=output_root / "records/mvbench_test.final.jsonl",
        output_root=output_root,
        require_media_sha256=False,
    )
    artifact["media_map_sha256"] = canonical_json_sha256(media_map)
    archive_counts: dict[str, int] = {}
    manual_samples = 0
    for record in records:
        media = record["media"][0]
        archive = media.get("archive")
        if archive is None:
            manual_samples += 1
        else:
            name = archive["name"]
            archive_counts[name] = archive_counts.get(name, 0) + 1
    artifact["pending_media_plan"] = {
        "selected_samples_by_archive": dict(sorted(archive_counts.items())),
        "manual_source_samples": manual_samples,
        "full_archive_download_bytes": sum(
            media_map["archives"][name]["bytes"] for name in archive_counts
        ),
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mvbench-map", type=Path, default=DEFAULT_MVBENCH_MAP)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--selection-output",
        type=Path,
        help="optional tracked summary containing selected IDs and per-media hashes",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.parquet_batch_size <= 0:
        raise SystemExit("--parquet-batch-size must be positive")

    protocol = load_p9_quality_protocol(args.protocol)
    media_map = _load_json(args.mvbench_map)
    seed = protocol["selection"]["seed"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _materialize_docvqa(
            protocol=protocol,
            raw_root=args.raw_root,
            output_root=output_root,
            seed=seed,
            batch_size=args.parquet_batch_size,
        ),
        _materialize_muirbench(
            protocol=protocol,
            raw_root=args.raw_root,
            output_root=output_root,
            seed=seed,
            batch_size=args.parquet_batch_size,
        ),
        _materialize_mvbench(
            protocol=protocol,
            raw_root=args.raw_root,
            output_root=output_root,
            seed=seed,
            media_map=media_map,
        ),
    ]
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "record_type": "p9_quality_materialization",
        "protocol_sha256": canonical_json_sha256(protocol),
        "selection_contract": {
            "algorithm": protocol["selection"]["algorithm"],
            "preimage_encoding": SELECTION_PREIMAGE_ENCODING,
            "tie_breaker": "sample_id_ascending",
            "seed": seed,
            "development_is_prefix_of_final": True,
        },
        "datasets": artifacts,
    }
    manifest_path = output_root / "p9_quality_materialization.json"
    manifest_sha256 = write_json_atomic(manifest_path, manifest)
    if args.selection_output is not None:
        write_json_atomic(
            args.selection_output,
            selection_manifest_from_materialization(manifest),
        )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "datasets": {
                    artifact["id"]: artifact["materialization_status"] for artifact in artifacts
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
