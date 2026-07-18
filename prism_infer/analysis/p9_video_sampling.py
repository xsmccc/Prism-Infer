"""MVBench 冻结的 16-segment center 视频/帧目录采样。"""

from __future__ import annotations

import hashlib
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from prism_infer.analysis.benchmark_schema import canonical_json_sha256


VIDEO_DECODER_DISTRIBUTION = "opencv-python-headless"
VIDEO_COLOR_CONVERSION = "BGR_to_RGB"


def _validate_decoder_contract(
    actual: Mapping[str, str],
    expected: Mapping[str, Any] | None,
) -> None:
    """Fail closed when a formal run's concrete video decoder drifts."""

    if expected is None:
        return
    required = {
        "distribution",
        "distribution_version",
        "api_version",
        "backend",
        "color_conversion",
    }
    if set(expected) != required:
        raise ValueError(
            "video decoder contract must contain exactly "
            f"{sorted(required)}, got {sorted(expected)}"
        )
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in sorted(required)
        if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(f"video decoder contract mismatch: {mismatches}")


def uniform_segment_center_indices(
    *,
    first_index: int,
    max_index: int,
    frames: int,
    fps: float,
    temporal_bound: Mapping[str, Any] | None,
) -> list[int]:
    """复现 MVBench notebook ``get_index`` 的 segment-center 索引。"""

    for name, value in (
        ("first_index", first_index),
        ("max_index", max_index),
        ("frames", frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if first_index < 0 or max_index < first_index or frames <= 0:
        raise ValueError("invalid frame index range or sample count")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("fps must be numeric")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if temporal_bound is None:
        start_index = first_index
        end_index = max_index
    else:
        start = temporal_bound.get("start")
        end = temporal_bound.get("end")
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
            raise ValueError("temporal bound must contain finite 0 <= start < end")
        start_index = max(first_index, round(float(start) * fps))
        end_index = min(round(float(end) * fps), max_index)
    if end_index < start_index:
        raise ValueError(
            "temporal bound does not overlap available frames: "
            f"start={start_index}, end={end_index}"
        )
    segment_size = float(end_index - start_index) / frames
    return [
        int(
            start_index
            + segment_size / 2.0
            + round(segment_size * index)
        )
        for index in range(frames)
    ]


def _rgb_frame_identity(image: Image.Image, *, index: int) -> dict[str, Any]:
    rgb = image.convert("RGB")
    return {
        "index": index,
        "width": rgb.width,
        "height": rgb.height,
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }


def sample_video_file(
    path: str | Path,
    *,
    frames: int,
    temporal_bound: Mapping[str, Any] | None,
    decoder_contract: Mapping[str, Any] | None = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """用冻结 OpenCV/FFmpeg 解码器采样 segment-center RGB 帧。"""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MVBench video sampling requires optional dependency cv2") from exc
    try:
        distribution_version = version(VIDEO_DECODER_DISTRIBUTION)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "MVBench video sampling requires the opencv-python-headless "
            "quality dependency; a different cv2 distribution is not accepted"
        ) from exc
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open MVBench video: {video_path}")
    try:
        decoder = {
            "distribution": VIDEO_DECODER_DISTRIBUTION,
            "distribution_version": distribution_version,
            "api_version": str(cv2.__version__),
            "backend": str(capture.getBackendName()),
            "color_conversion": VIDEO_COLOR_CONVERSION,
        }
        _validate_decoder_contract(decoder, decoder_contract)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise ValueError(f"video reports no frames: {video_path}")
        indices = uniform_segment_center_indices(
            first_index=0,
            max_index=frame_count - 1,
            frames=frames,
            fps=fps,
            temporal_bound=temporal_bound,
        )
        decoded: dict[int, Image.Image] = {}
        for index in sorted(set(indices)):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"cannot seek video {video_path} to frame {index}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode video {video_path} frame {index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = Image.fromarray(rgb)
        sampled = [decoded[index].copy() for index in indices]
    finally:
        capture.release()
    frame_identity = [
        _rgb_frame_identity(image, index=index)
        for image, index in zip(sampled, indices)
    ]
    return sampled, {
        "source_kind": "video_file",
        "decoder": decoder,
        "fps": fps,
        "source_frame_count": frame_count,
        "sampled_indices": indices,
        "sampled_rgb_identity_sha256": canonical_json_sha256(frame_identity),
    }


def sample_frame_manifest(
    frame_records: Sequence[Mapping[str, Any]],
    *,
    materialized_root: str | Path,
    frames: int,
    fps: float,
    temporal_bound: Mapping[str, Any] | None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """从 TVQA content-addressed frame manifest 复现 first_idx=1 采样。"""

    if not frame_records:
        raise ValueError("frame manifest must be non-empty")
    by_index: dict[int, Mapping[str, Any]] = {}
    for record in frame_records:
        member_path = record.get("archive_member_path")
        if not isinstance(member_path, str):
            raise ValueError("frame manifest has no archive member path")
        stem = Path(member_path).stem
        if not stem.isdigit():
            raise ValueError(f"MVBench frame filename is not numeric: {member_path}")
        index = int(stem)
        if index in by_index:
            raise ValueError(f"duplicate MVBench frame index: {index}")
        by_index[index] = record
    max_index = max(by_index)
    expected = set(range(1, max_index + 1))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        raise ValueError(f"MVBench frame directory is not contiguous: missing={missing}")
    indices = uniform_segment_center_indices(
        first_index=1,
        max_index=max_index,
        frames=frames,
        fps=fps,
        temporal_bound=temporal_bound,
    )
    root = Path(materialized_root)
    sampled = []
    for index in indices:
        path = root / by_index[index]["materialized_path"]
        with Image.open(path) as image:
            sampled.append(image.convert("RGB").copy())
    frame_identity = [
        _rgb_frame_identity(image, index=index)
        for image, index in zip(sampled, indices)
    ]
    return sampled, {
        "source_kind": "frame_directory",
        "decoder": "Pillow",
        "fps": float(fps),
        "source_frame_count": max_index,
        "sampled_indices": indices,
        "sampled_rgb_identity_sha256": canonical_json_sha256(frame_identity),
    }
