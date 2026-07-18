"""P9 标准质量集的确定性选样、媒体身份与物化契约。

该模块不下载数据，也不运行模型。数据集适配器只向这里传入公开数据行；本模块负责
把冻结协议中的 SHA256 选样规则变成可测试的纯函数，并确保每个已物化媒体都有内容
哈希。这样 evaluator 不需要信任文件名或下载缓存即可复核输入身份。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image

from prism_infer.analysis.benchmark_schema import canonical_json_sha256


MATERIALIZATION_SCHEMA_VERSION = 1
MV_SAMPLE_ID_SEPARATOR = "|"
SELECTION_PREIMAGE_ENCODING = (
    "utf8(dataset_id + revision + sample_id + decimal_seed)"
)
IMAGE_FORMAT_SUFFIXES = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def sha256_bytes(payload: bytes) -> str:
    """返回 bytes 的小写 SHA256。"""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA256，避免把 parquet 或视频整体读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sample_id(value: object) -> str:
    """将公开数据 ID 规范成稳定字符串；拒绝隐式 bool/float 转换。"""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"sample id must be a string or integer, got {value!r}")
    sample_id = str(value)
    if not sample_id:
        raise ValueError("sample id must not be empty")
    return sample_id


def selection_sha256(
    *,
    dataset_id: str,
    revision: str,
    sample_id: str,
    seed: int,
) -> str:
    """实现冻结协议的字面拼接 SHA256 排序键。"""

    if not dataset_id or not revision or not sample_id:
        raise ValueError("selection identity fields must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("selection seed must be a positive integer")
    preimage = f"{dataset_id}{revision}{sample_id}{seed}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def selected_ids_sha256(sample_ids: Sequence[str]) -> str:
    """对按 selection rank 排序的 ID 列表计算 canonical JSON SHA256。"""

    normalized = [canonical_sample_id(sample_id) for sample_id in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("selected sample ids must be unique")
    return canonical_json_sha256(normalized)


@dataclass(frozen=True)
class SampleSelection:
    """一个数据集的嵌套 development/final 固定子集。"""

    population_samples: int
    development_ids: tuple[str, ...]
    final_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """生成不含时间戳、可逐字复现的选择记录。"""

        return {
            "population_samples": self.population_samples,
            "development": {
                "samples": len(self.development_ids),
                "sample_ids": list(self.development_ids),
                "selected_sample_ids_sha256": selected_ids_sha256(
                    self.development_ids
                ),
            },
            "final": {
                "samples": len(self.final_ids),
                "sample_ids": list(self.final_ids),
                "selected_sample_ids_sha256": selected_ids_sha256(
                    self.final_ids
                ),
            },
        }


def select_sample_ids(
    sample_ids: Sequence[object],
    *,
    dataset_id: str,
    revision: str,
    seed: int,
    development_samples: int,
    final_samples: int,
) -> SampleSelection:
    """按冻结 SHA256 算法选样，并令 development 严格嵌套于 final。"""

    normalized = [canonical_sample_id(sample_id) for sample_id in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{dataset_id} contains duplicate sample ids")
    for name, count in (
        ("development_samples", development_samples),
        ("final_samples", final_samples),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if development_samples > final_samples:
        raise ValueError("development_samples cannot exceed final_samples")
    if final_samples > len(normalized):
        raise ValueError(
            f"{dataset_id} final selection requests {final_samples} of "
            f"{len(normalized)} samples"
        )
    ranked = sorted(
        normalized,
        key=lambda sample_id: (
            selection_sha256(
                dataset_id=dataset_id,
                revision=revision,
                sample_id=sample_id,
                seed=seed,
            ),
            sample_id,
        ),
    )
    final_ids = tuple(ranked[:final_samples])
    return SampleSelection(
        population_samples=len(normalized),
        development_ids=final_ids[:development_samples],
        final_ids=final_ids,
    )


def mvbench_sample_id(task: str, video: str, question_index: int) -> str:
    """编码协议中的 ``task + video + question_index`` 复合 ID。"""

    if not task or not video:
        raise ValueError("MVBench task and video must be non-empty")
    if MV_SAMPLE_ID_SEPARATOR in task or MV_SAMPLE_ID_SEPARATOR in video:
        raise ValueError("MVBench identity fields contain the reserved separator")
    if (
        isinstance(question_index, bool)
        or not isinstance(question_index, int)
        or question_index < 0
    ):
        raise ValueError("MVBench question_index must be a non-negative integer")
    return MV_SAMPLE_ID_SEPARATOR.join((task, video, str(question_index)))


def _image_metadata(payload: bytes) -> tuple[str, int, int]:
    """验证嵌入图片，并返回规范后缀与原始宽高。"""

    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise ValueError("embedded media is not a valid image") from exc
    suffix = IMAGE_FORMAT_SUFFIXES.get(str(image_format).upper())
    if suffix is None:
        raise ValueError(f"unsupported embedded image format: {image_format!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid embedded image dimensions: {width}x{height}")
    return suffix, width, height


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """同目录原子落盘；已有同名内容由调用者先校验。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_embedded_image(
    image: Mapping[str, Any],
    *,
    output_root: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    """按内容哈希去重落盘一张 parquet 内嵌图片。"""

    payload = image.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("embedded image.bytes must be non-empty bytes")
    source_path = image.get("path")
    if source_path is not None and not isinstance(source_path, str):
        raise ValueError("embedded image.path must be a string or null")
    suffix, width, height = _image_metadata(payload)
    digest = sha256_bytes(payload)
    relative_path = (
        PurePosixPath("media")
        / dataset_id
        / digest[:2]
        / f"{digest}{suffix}"
    )
    target = Path(output_root) / Path(relative_path)
    if target.exists():
        if target.stat().st_size != len(payload) or sha256_file(target) != digest:
            raise ValueError(f"existing materialized image has wrong identity: {target}")
    else:
        _atomic_write_bytes(target, payload)
    return {
        "source_path": source_path,
        "materialized_path": relative_path.as_posix(),
        "sha256": digest,
        "bytes": len(payload),
        "width": width,
        "height": height,
    }


def materialize_docvqa_row(
    row: Mapping[str, Any],
    *,
    output_root: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    """把一个选中的 DocVQA row 转成 evaluator 可消费记录。"""

    sample_id = canonical_sample_id(row.get("questionId"))
    question = row.get("question")
    answers = row.get("answers")
    image = row.get("image")
    if not isinstance(question, str) or not question:
        raise ValueError(f"DocVQA {sample_id} has no question")
    if not isinstance(answers, list) or not answers or not all(
        isinstance(answer, str) and answer for answer in answers
    ):
        raise ValueError(f"DocVQA {sample_id} has invalid answers")
    if not isinstance(image, Mapping):
        raise ValueError(f"DocVQA {sample_id} has no embedded image")
    return {
        "sample_id": sample_id,
        "question": question,
        "answers": list(answers),
        "question_types": list(row.get("question_types") or []),
        "document": {
            "doc_id": row.get("docId"),
            "ucsf_document_id": row.get("ucsf_document_id"),
            "ucsf_document_page_no": row.get("ucsf_document_page_no"),
        },
        "media": [
            materialize_embedded_image(
                image,
                output_root=output_root,
                dataset_id=dataset_id,
            )
        ],
    }


def materialize_muirbench_row(
    row: Mapping[str, Any],
    *,
    output_root: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    """把一个选中的 MuirBench row 转成多图选择题记录。"""

    sample_id = canonical_sample_id(row.get("idx"))
    question = row.get("question")
    options = row.get("options")
    answer = row.get("answer")
    image_list = row.get("image_list")
    if not isinstance(question, str) or not question:
        raise ValueError(f"MuirBench {sample_id} has no question")
    if not isinstance(options, list) or not options or not all(
        isinstance(option, str) for option in options
    ):
        raise ValueError(f"MuirBench {sample_id} has invalid options")
    valid_answers = {chr(ord("A") + index) for index in range(len(options))}
    if answer not in valid_answers:
        raise ValueError(
            f"MuirBench {sample_id} answer {answer!r} is outside its options"
        )
    if not isinstance(image_list, list) or not image_list or not all(
        isinstance(image, Mapping) for image in image_list
    ):
        raise ValueError(f"MuirBench {sample_id} has invalid image_list")
    return {
        "sample_id": sample_id,
        "task": row.get("task"),
        "image_relation": row.get("image_relation"),
        "image_type": row.get("image_type"),
        "question": question,
        "options": list(options),
        "answer": answer,
        "counterpart_idx": row.get("counterpart_idx"),
        "media": [
            materialize_embedded_image(
                image,
                output_root=output_root,
                dataset_id=dataset_id,
            )
            for image in image_list
        ],
    }


def build_mvbench_row(
    row: Mapping[str, Any],
    *,
    task: str,
    question_index: int,
    task_media: Mapping[str, Any],
    archives: Mapping[str, Any],
) -> dict[str, Any]:
    """生成 MVBench metadata 记录；媒体未取到时必须显式标成 pending。"""

    video = row.get("video")
    question = row.get("question")
    candidates = row.get("candidates")
    answer = row.get("answer")
    if not isinstance(video, str) or not video:
        raise ValueError(f"MVBench {task}[{question_index}] has no video")
    if not isinstance(question, str) or not question:
        raise ValueError(f"MVBench {task}[{question_index}] has no question")
    if not isinstance(candidates, list) or not candidates or not all(
        isinstance(candidate, str) and candidate for candidate in candidates
    ):
        raise ValueError(f"MVBench {task}[{question_index}] has invalid candidates")
    if answer not in candidates:
        raise ValueError(f"MVBench {task}[{question_index}] answer is not a candidate")

    temporal_bound = None
    if task_media.get("uses_temporal_bound"):
        start = row.get("start")
        end = row.get("end")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0.0
            or float(end) <= float(start)
        ):
            raise ValueError(
                f"MVBench {task}[{question_index}] has invalid temporal bound"
            )
        temporal_bound = {"start": float(start), "end": float(end)}

    archive_name = task_media.get("archive")
    availability = task_media.get("availability", "archive_available")
    archive = None
    if archive_name is not None:
        archive = archives.get(archive_name)
        if not isinstance(archive, Mapping):
            raise ValueError(f"MVBench task {task!r} references unknown archive")
    media_prefix = task_media.get("media_prefix")
    if not isinstance(media_prefix, str) or not media_prefix:
        raise ValueError(f"MVBench task {task!r} has no media_prefix")
    media_reference: dict[str, Any] = {
        "source_path": video,
        "archive_member_path": (PurePosixPath(media_prefix) / video).as_posix(),
        "media_type": task_media.get("media_type"),
        "materialization_status": (
            "pending_manual_source" if archive is None else "pending_archive_extraction"
        ),
        "materialized_path": None,
        "sha256": None,
    }
    if archive is not None:
        media_reference["archive"] = {
            "name": archive_name,
            "repository_path": archive["repository_path"],
            "bytes": archive["bytes"],
            "sha256": archive["sha256"],
        }
    return {
        "sample_id": mvbench_sample_id(task, video, question_index),
        "task": task,
        "question_index": question_index,
        "question": question,
        "candidates": list(candidates),
        "answer": answer,
        "answer_index": candidates.index(answer),
        "temporal_bound": temporal_bound,
        "subtitle": row.get("subtitle"),
        "source_availability": availability,
        "media": [media_reference],
    }


def validate_selected_records(
    records: Sequence[Mapping[str, Any]],
    selection: SampleSelection,
    *,
    require_media_sha256: bool,
) -> None:
    """确保 records 与 final selection 一一对应，且无静默媒体缺失。"""

    record_ids = [canonical_sample_id(record.get("sample_id")) for record in records]
    if record_ids != list(selection.final_ids):
        raise ValueError("materialized record order/identity differs from final selection")
    for record in records:
        media = record.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError(f"sample {record['sample_id']} has no media identity")
        for item in media:
            if not isinstance(item, Mapping):
                raise ValueError(f"sample {record['sample_id']} has invalid media")
            digest = item.get("sha256")
            if require_media_sha256 and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"sample {record['sample_id']} is missing media SHA256"
                )


def media_identity_record(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总逐样本媒体 hash；pending 媒体不会被伪装成完整身份。"""

    per_sample = []
    complete = True
    all_resolved = True
    unique_digests: set[str] = set()
    total_references = 0
    eligible_per_sample = []
    for record in records:
        identities = []
        sample_complete = True
        for media in record["media"]:
            digest = media.get("sha256")
            total_references += 1
            if digest is None:
                complete = False
                sample_complete = False
                status = media.get("materialization_status")
                if not (
                    isinstance(status, str) and status.startswith("excluded_")
                ):
                    all_resolved = False
            else:
                unique_digests.add(digest)
            identities.append(digest)
        identity = {
            "sample_id": record["sample_id"],
            "media_sha256": identities,
        }
        per_sample.append(identity)
        if sample_complete:
            eligible_per_sample.append(identity)
    return {
        "status": (
            "complete"
            if complete
            else (
                "complete_for_eligible_subset" if all_resolved else "pending"
            )
        ),
        "sample_media_references": total_references,
        "unique_materialized_media": len(unique_digests),
        "per_sample": per_sample,
        "per_sample_media_identity_sha256": (
            canonical_json_sha256(per_sample) if complete else None
        ),
        "eligible_per_sample_media_identity_sha256": (
            canonical_json_sha256(eligible_per_sample)
            if eligible_per_sample
            else None
        ),
    }


def evaluation_subset_record(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """区分可评估、协议排除和仍 pending 的样本，禁止静默缩小分母。"""

    eligible_ids: list[str] = []
    pending_ids: list[str] = []
    exclusions: list[dict[str, str]] = []
    for record in records:
        sample_id = canonical_sample_id(record.get("sample_id"))
        media = record.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError(f"sample {sample_id} has no media")
        statuses = [item.get("materialization_status") for item in media]
        if all(isinstance(item.get("sha256"), str) for item in media):
            eligible_ids.append(sample_id)
            continue
        excluded = [
            status
            for status in statuses
            if isinstance(status, str) and status.startswith("excluded_")
        ]
        if excluded:
            unresolved = [
                status
                for item, status in zip(media, statuses)
                if item.get("sha256") is None
                and not (isinstance(status, str) and status.startswith("excluded_"))
            ]
            if unresolved:
                raise ValueError(
                    f"sample {sample_id} mixes excluded and pending media: {unresolved}"
                )
            exclusions.append(
                {
                    "sample_id": sample_id,
                    "reason": ",".join(sorted(set(excluded))),
                }
            )
        else:
            pending_ids.append(sample_id)
    return {
        "status": (
            "pending"
            if pending_ids
            else (
                "ready_with_protocol_exclusions" if exclusions else "ready"
            )
        ),
        "selected_samples": len(records),
        "eligible_samples": len(eligible_ids),
        "eligible_sample_ids": eligible_ids,
        "eligible_sample_ids_sha256": (
            selected_ids_sha256(eligible_ids) if eligible_ids else None
        ),
        "excluded_samples": exclusions,
        "pending_sample_ids": pending_ids,
    }


def selection_manifest_from_materialization(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """移除问题/答案，仅保留可提交的 source、ID、媒体与排除证据。"""

    datasets = []
    for artifact in manifest["datasets"]:
        dataset = {
            "id": artifact["id"],
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "split": artifact["split"],
            "sample_id_field": artifact["sample_id_field"],
            "source_files": artifact["source_files"],
            "selection": artifact["selection"],
            "media_identity": artifact["media_identity"],
            "materialization_status": artifact["materialization_status"],
        }
        for key in ("evaluation_subset", "pending_media_plan"):
            if key in artifact:
                dataset[key] = artifact[key]
        datasets.append(dataset)
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "record_type": "p9_quality_selection",
        "protocol_sha256": manifest["protocol_sha256"],
        "selection_contract": manifest["selection_contract"],
        "datasets": datasets,
    }


def write_json_atomic(path: str | Path, value: object) -> str:
    """稳定序列化 JSON 并原子写入，返回落盘文件 SHA256。"""

    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    target = Path(path)
    _atomic_write_bytes(target, payload)
    return sha256_bytes(payload)


def write_jsonl_atomic(path: str | Path, records: Sequence[Mapping[str, Any]]) -> str:
    """稳定写入 selected records JSONL，并返回文件 SHA256。"""

    payload = b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for record in records
    )
    target = Path(path)
    _atomic_write_bytes(target, payload)
    return sha256_bytes(payload)
