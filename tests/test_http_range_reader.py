"""HTTP Range 与 selective ZIP 物化的离线契约测试。"""

from __future__ import annotations

import io
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import MethodType

import pytest

from prism_infer.analysis.http_range_reader import (
    HTTPRangeReader,
    parse_content_range,
)
from prism_infer.analysis.p9_quality_materialization import sha256_bytes
from scripts.materialize_p9_mvbench_media import (
    _extract_member,
    _inventory_archive,
    _materialize_media,
)


class _FakeSession:
    def close(self) -> None:
        pass


def _memory_range_reader(payload: bytes, chunk_bytes: int) -> HTTPRangeReader:
    reader = HTTPRangeReader.__new__(HTTPRangeReader)
    io.RawIOBase.__init__(reader)
    reader._expected_size = len(payload)
    reader._chunk_bytes = chunk_bytes
    reader._position = 0
    reader._cache = OrderedDict()
    reader._session = _FakeSession()

    def fetch(self: HTTPRangeReader, index: int) -> bytes:
        start = index * chunk_bytes
        return payload[start : start + chunk_bytes]

    reader._fetch_chunk = MethodType(fetch, reader)
    return reader


def test_content_range_parser_requires_exact_bounded_interval() -> None:
    assert parse_content_range("bytes 10-19/100") == (10, 19, 100)
    for invalid in (None, "10-19/100", "bytes 20-10/100", "bytes 0-100/100"):
        with pytest.raises(ValueError):
            parse_content_range(invalid)


def test_range_reader_seek_and_cross_chunk_read() -> None:
    payload = b"0123456789abcdef"
    with _memory_range_reader(payload, chunk_bytes=5) as reader:
        reader.seek(3)
        assert reader.read(8) == b"3456789a"
        assert reader.tell() == 11
        reader.seek(-4, io.SEEK_END)
        assert reader.read() == b"cdef"


def test_selective_zip_inventory_reports_exact_missing_members(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/a.mp4", b"video-a")
        archive.writestr("frames/show/00001.jpg", b"frame-1")
        archive.writestr("frames/show/00002.jpg", b"frame-2")
    selected = [
        {
            "archive_member_path": "videos/a.mp4",
            "media_type": "video",
        },
        {
            "archive_member_path": "frames/show",
            "media_type": "frames",
        },
        {
            "archive_member_path": "videos/missing.mp4",
            "media_type": "video",
        },
    ]

    with zipfile.ZipFile(archive_path) as archive:
        inventory = _inventory_archive(archive, selected)

    assert inventory["selected_samples"] == 3
    assert inventory["unique_members"] == 3
    assert inventory["missing_members"] == ["videos/missing.mp4"]
    assert inventory["selected_uncompressed_bytes"] == 21


def test_selective_zip_materializes_content_addressed_video_and_frames(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/a.mp4", b"video-a")
        archive.writestr("frames/show/00001.jpg", b"frame-1")
        archive.writestr("frames/show/00002.jpg", b"frame-2")
    video = {
        "archive_member_path": "videos/a.mp4",
        "media_type": "video",
        "sha256": None,
    }
    frames = {
        "archive_member_path": "frames/show",
        "media_type": "frames",
        "sha256": None,
    }

    with zipfile.ZipFile(archive_path) as archive:
        cache = {}
        _materialize_media(
            archive,
            video,
            archive_name="fixture.zip",
            output_root=tmp_path,
            extracted_cache=cache,
        )
        _materialize_media(
            archive,
            frames,
            archive_name="fixture.zip",
            output_root=tmp_path,
            extracted_cache=cache,
        )

    assert video["sha256"] == sha256_bytes(b"video-a")
    assert (tmp_path / video["materialized_path"]).read_bytes() == b"video-a"
    assert frames["identity_kind"] == "canonical_frame_manifest_sha256"
    assert len(frames["frames"]) == 2
    assert sum(frame["bytes"] for frame in frames["frames"]) == 14


def test_zip_member_extractor_rejects_unknown_media_suffix(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video.bin", b"not-a-video")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsupported video suffix"):
            _extract_member(
                archive,
                archive.getinfo("video.bin"),
                archive_name="fixture.zip",
                media_type="video",
                output_root=tmp_path,
            )
