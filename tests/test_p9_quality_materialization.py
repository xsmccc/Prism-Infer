"""P9 标准质量集选样与媒体身份的 CPU 契约测试。"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from prism_infer.analysis.p9_quality_materialization import (
    SampleSelection,
    build_mvbench_row,
    materialize_docvqa_row,
    materialize_embedded_image,
    materialize_muirbench_row,
    media_identity_record,
    mvbench_sample_id,
    select_sample_ids,
    validate_selected_records,
)
from scripts.materialize_p9_quality import _discover_complete_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
MVBENCH_MAP = REPO_ROOT / "benchmarks/workloads/p9_mvbench_media_map.json"


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_sha256_selection_is_order_independent_and_frozen() -> None:
    kwargs = {
        "dataset_id": "fixture",
        "revision": "a" * 40,
        "seed": 20260717,
        "development_samples": 2,
        "final_samples": 3,
    }
    first = select_sample_ids(["10", "2", "7", "1"], **kwargs)
    second = select_sample_ids(["1", "7", "2", "10"], **kwargs)

    assert first == second
    assert first.development_ids == ("10", "7")
    assert first.final_ids == ("10", "7", "2")
    assert first.to_record()["development"]["selected_sample_ids_sha256"] == (
        "3fb5ce51265b5aade557b18f5859024cbc28e7cf7eb5f60c0d8ea6a2702ab6e4"
    )
    assert first.to_record()["final"]["selected_sample_ids_sha256"] == (
        "9e86fe1c76109e698773cad239660556f141a5cc1eccffc5fbd96fdb9eb26b85"
    )


def test_selection_rejects_duplicates_or_oversized_final_set() -> None:
    kwargs = {
        "dataset_id": "fixture",
        "revision": "b" * 40,
        "seed": 1,
        "development_samples": 1,
        "final_samples": 2,
    }
    with pytest.raises(ValueError, match="duplicate sample ids"):
        select_sample_ids(["same", "same"], **kwargs)
    with pytest.raises(ValueError, match="requests 2 of 1"):
        select_sample_ids(["only"], **kwargs)


def test_embedded_image_materialization_is_content_addressed(tmp_path: Path) -> None:
    payload = _png_bytes()
    image = {"bytes": payload, "path": "source/document.png"}

    first = materialize_embedded_image(
        image,
        output_root=tmp_path,
        dataset_id="fixture",
    )
    second = materialize_embedded_image(
        image,
        output_root=tmp_path,
        dataset_id="fixture",
    )

    assert first == second
    assert first["sha256"] == hashlib.sha256(payload).hexdigest()
    assert first["width"] == 8
    assert first["height"] == 6
    assert (tmp_path / first["materialized_path"]).read_bytes() == payload


def test_docvqa_and_muirbench_rows_keep_labels_and_media_hashes(
    tmp_path: Path,
) -> None:
    doc = materialize_docvqa_row(
        {
            "questionId": "doc-1",
            "question": "What is shown?",
            "answers": ["invoice", "an invoice"],
            "question_types": ["document"],
            "image": {"bytes": _png_bytes(), "path": "doc.png"},
            "docId": 1,
            "ucsf_document_id": "u1",
            "ucsf_document_page_no": "1",
        },
        output_root=tmp_path,
        dataset_id="docvqa_validation",
    )
    muir = materialize_muirbench_row(
        {
            "idx": "muir-1",
            "task": "spatial",
            "image_relation": "same",
            "image_type": "natural",
            "question": "Choose one.",
            "options": ["first", "second"],
            "answer": "B",
            "counterpart_idx": "muir-2",
            "image_list": [
                {"bytes": _png_bytes(), "path": "a.png"},
                {"bytes": _png_bytes((30, 20, 10)), "path": "b.png"},
            ],
        },
        output_root=tmp_path,
        dataset_id="muirbench_test",
    )

    assert doc["answers"] == ["invoice", "an invoice"]
    assert len(doc["media"][0]["sha256"]) == 64
    assert muir["answer"] == "B"
    assert len(muir["media"]) == 2


def test_mvbench_metadata_stays_pending_until_media_has_content_hash() -> None:
    task_media = {
        "archive": "fixture.zip",
        "media_prefix": "fixture/videos",
        "media_type": "video",
        "uses_temporal_bound": True,
    }
    archives = {
        "fixture.zip": {
            "repository_path": "video/fixture.zip",
            "bytes": 123,
            "sha256": "c" * 64,
        }
    }
    row = build_mvbench_row(
        {
            "video": "clip.mp4",
            "question": "What happened?",
            "candidates": ["A thing", "Nothing"],
            "answer": "A thing",
            "start": 1.0,
            "end": 2.0,
        },
        task="fixture_task",
        question_index=3,
        task_media=task_media,
        archives=archives,
    )

    assert row["sample_id"] == "fixture_task|clip.mp4|3"
    assert row["media"][0]["archive_member_path"] == (
        "fixture/videos/clip.mp4"
    )
    assert row["media"][0]["sha256"] is None
    assert media_identity_record([row])["status"] == "pending"


def test_complete_media_contract_rejects_pending_mvbench_reference() -> None:
    selection = SampleSelection(
        population_samples=1,
        development_ids=("task|video.mp4|0",),
        final_ids=("task|video.mp4|0",),
    )
    record = {
        "sample_id": mvbench_sample_id("task", "video.mp4", 0),
        "media": [{"sha256": None}],
    }

    with pytest.raises(ValueError, match="missing media SHA256"):
        validate_selected_records(
            [record],
            selection,
            require_media_sha256=True,
        )


def test_parquet_shard_discovery_fails_closed_on_missing_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "validation-00000-of-00003.parquet").touch()
    (tmp_path / "validation-00002-of-00003.parquet").touch()

    with pytest.raises(ValueError, match="incomplete validation shards"):
        _discover_complete_shards(tmp_path, split="validation")


def test_frozen_mvbench_map_covers_all_official_tasks_and_archives() -> None:
    media_map = json.loads(MVBENCH_MAP.read_text(encoding="utf-8"))

    assert len(media_map["tasks"]) == 20
    assert media_map["dataset_revision"] == (
        "230a2d4fac8900333c61754641c7a13e069ac9c6"
    )
    assert media_map["tasks"]["fine_grained_pose"]["archive"] is None
    referenced_archives = {
        task["archive"]
        for task in media_map["tasks"].values()
        if task["archive"] is not None
    }
    assert referenced_archives == set(media_map["archives"])
