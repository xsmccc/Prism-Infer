"""MVBench 冻结帧采样的 CPU 契约。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import prism_infer.analysis.p9_video_sampling as video_sampling
from prism_infer.analysis.p9_video_sampling import (
    VIDEO_FRAME_ACCESS_POLICY,
    _validate_decoder_contract,
    sample_frame_manifest,
    sample_video_file,
    uniform_segment_center_indices,
)


def test_uniform_segment_centers_match_official_rounding() -> None:
    assert uniform_segment_center_indices(
        first_index=0,
        max_index=9,
        frames=4,
        fps=2.0,
        temporal_bound=None,
    ) == [1, 3, 5, 8]
    assert uniform_segment_center_indices(
        first_index=0,
        max_index=9,
        frames=4,
        fps=2.0,
        temporal_bound={"start": 1.0, "end": 3.0},
    ) == [2, 3, 4, 5]


def test_video_decoder_contract_rejects_environment_drift() -> None:
    actual = {
        "distribution": "opencv-python-headless",
        "distribution_version": "4.10.0.84",
        "api_version": "4.10.0",
        "backend": "FFMPEG",
        "color_conversion": "BGR_to_RGB",
        "frame_access_policy": VIDEO_FRAME_ACCESS_POLICY,
    }

    _validate_decoder_contract(actual, dict(actual))
    drifted = dict(actual, backend="GSTREAMER")
    try:
        _validate_decoder_contract(actual, drifted)
    except RuntimeError as exc:
        assert "video decoder contract mismatch" in str(exc)
        assert "GSTREAMER" in str(exc)
    else:
        raise AssertionError("expected decoder backend drift to fail closed")


def test_frame_manifest_sampling_preserves_duplicate_free_source_indices(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(1, 9):
        path = tmp_path / f"{index:05d}.jpg"
        Image.new("RGB", (8, 6), color=(index, 0, 0)).save(path)
        records.append(
            {
                "archive_member_path": f"show/{index:05d}.jpg",
                "materialized_path": path.name,
            }
        )

    images, metadata = sample_frame_manifest(
        records,
        materialized_root=tmp_path,
        frames=4,
        fps=3.0,
        temporal_bound=None,
    )

    assert metadata["sampled_indices"] == [1, 3, 5, 6]
    assert metadata["frame_access"] == {
        "method": "frame_manifest",
        "reported_frame_count": 8,
        "fallback_trigger": None,
    }
    assert len(images) == 4
    assert len(metadata["sampled_rgb_identity_sha256"]) == 64


def test_video_sampling_falls_back_when_reported_frame_count_is_too_large(
    monkeypatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(5)]

    class FakeCapture:
        def __init__(self) -> None:
            self.position = 0

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

        def getBackendName(self) -> str:
            return "FFMPEG"

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 9.0
            if prop == cv2.CAP_PROP_FPS:
                return 2.0
            return float(self.position)

        def set(self, prop: int, value: float) -> bool:
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.position = int(value)
            return True

        def read(self):
            if self.position >= len(frames):
                return False, None
            frame = frames[self.position]
            self.position += 1
            return True, frame.copy()

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr(video_sampling, "version", lambda _: "4.10.0.84")
    contract = {
        "distribution": "opencv-python-headless",
        "distribution_version": "4.10.0.84",
        "api_version": cv2.__version__,
        "backend": "FFMPEG",
        "color_conversion": "BGR_to_RGB",
        "frame_access_policy": VIDEO_FRAME_ACCESS_POLICY,
    }

    images, metadata = sample_video_file(
        "truncated.webm",
        frames=4,
        temporal_bound=None,
        decoder_contract=contract,
    )

    assert metadata["source_frame_count"] == 5
    assert metadata["sampled_indices"] == [0, 1, 2, 3]
    assert metadata["frame_access"] == {
        "method": "sequential_fallback",
        "reported_frame_count": 9,
        "fallback_trigger": {"operation": "decode", "frame_index": 5},
    }
    assert [int(np.asarray(image)[0, 0, 0]) for image in images] == [0, 1, 2, 3]
    for image in images:
        image.close()
