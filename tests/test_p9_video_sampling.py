"""MVBench 冻结帧采样的 CPU 契约。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from prism_infer.analysis.p9_video_sampling import (
    _validate_decoder_contract,
    sample_frame_manifest,
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
    assert len(images) == 4
    assert len(metadata["sampled_rgb_identity_sha256"]) == 64
