"""Lossless P9 video bridge identity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_video_bundle import (
    load_video_bundle,
    rgb_frame_records,
    write_video_bundle,
)


def _frames() -> list[Image.Image]:
    return [
        Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8), mode="RGB")
        for value in (17, 203)
    ]


def _sampling(frames: list[Image.Image]) -> dict[str, object]:
    indices = [3, 9]
    _, records = rgb_frame_records(frames, indices)
    return {
        "source_kind": "video_file",
        "decoder": {"name": "frozen-test-decoder"},
        "fps": 3.0,
        "source_frame_count": 12,
        "frame_access": {
            "method": "random_seek",
            "reported_frame_count": 12,
            "fallback_trigger": None,
        },
        "sampled_indices": indices,
        "sampled_rgb_identity_sha256": canonical_json_sha256(records),
    }


def test_lossless_video_bundle_round_trip_preserves_rgb_identity(
    tmp_path: Path,
) -> None:
    frames = _frames()
    path = tmp_path / "sample.npz"
    sampling = _sampling(frames)
    try:
        written = write_video_bundle(
            path,
            frames,
            sample_id="mv-1",
            source_media_sha256="1" * 64,
            video_sampling=sampling,
        )
        loaded, metadata, evidence = load_video_bundle(
            path,
            expected_sample_id="mv-1",
            expected_source_media_sha256="1" * 64,
        )
        try:
            assert [np.asarray(frame).tolist() for frame in loaded] == [
                np.asarray(frame).tolist() for frame in frames
            ]
            assert metadata["video_sampling"] == sampling
            assert evidence["sha256"] == written["sha256"]
        finally:
            for frame in loaded:
                frame.close()
    finally:
        for frame in frames:
            frame.close()


def test_lossless_video_bundle_rejects_tampered_frame_bytes(tmp_path: Path) -> None:
    frames = _frames()
    path = tmp_path / "sample.npz"
    try:
        write_video_bundle(
            path,
            frames,
            sample_id="mv-1",
            source_media_sha256="1" * 64,
            video_sampling=_sampling(frames),
        )
        with np.load(path, allow_pickle=False) as bundle:
            metadata = bundle["metadata"].copy()
            arrays = bundle["frames"].copy()
        arrays[0, 0, 0, 0] ^= 1
        with path.open("wb") as output:
            np.savez_compressed(output, metadata=metadata, frames=arrays)

        with pytest.raises(ValueError, match="RGB identity"):
            load_video_bundle(path)
    finally:
        for frame in frames:
            frame.close()


def test_lossless_video_bundle_rejects_wrong_source_identity(tmp_path: Path) -> None:
    frames = _frames()
    path = tmp_path / "sample.npz"
    try:
        write_video_bundle(
            path,
            frames,
            sample_id="mv-1",
            source_media_sha256="1" * 64,
            video_sampling=_sampling(frames),
        )
        with pytest.raises(ValueError, match="source media SHA256"):
            load_video_bundle(
                path,
                expected_source_media_sha256="2" * 64,
            )
    finally:
        for frame in frames:
            frame.close()
