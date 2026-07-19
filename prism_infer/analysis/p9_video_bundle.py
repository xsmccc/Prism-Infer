"""Lossless one-sample bridge for frozen P9 MVBench RGB frames.

The formal MVBench decoder is pinned to OpenCV 4.10/FFmpeg, while the isolated
vLLM environment currently contains OpenCV 5.  Decoding the source again in
that environment would therefore violate the frozen semantic contract.  This
module serializes one already-selected RGB frame sequence into a temporary,
lossless NPZ bundle and validates every byte when the external runner loads it.

Bundles are intentionally per-sample and disposable.  Persisting all 190 × 16
raw frame sequences would consume several GiB on the benchmark volume without
adding semantic evidence: the source SHA, selected indices, decoder identity,
and per-frame RGB hashes are already retained in the prediction artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prism_infer.analysis.schema_constants import (
    RGB_CHANNEL_COUNT,
    SHA256_HEX_LENGTH,
)
from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_materialization import sha256_file


RGB_FRAME_TENSOR_RANK = 3
RGB_VIDEO_TENSOR_RANK = 4
VIDEO_BUNDLE_SCHEMA_VERSION = 1
VIDEO_BUNDLE_RECORD_TYPE = "p9_lossless_sampled_rgb_bundle"


def rgb_frame_records(
    frames: Sequence[Image.Image | np.ndarray],
    sampled_indices: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Normalize frames to packed uint8 RGB and compute frozen identities."""

    if not frames or len(frames) != len(sampled_indices):
        raise ValueError("frames and sampled_indices must be equal and non-empty")
    arrays = []
    records = []
    expected_shape: tuple[int, int, int] | None = None
    for position, (frame, source_index) in enumerate(zip(frames, sampled_indices, strict=True)):
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"sampled_indices[{position}] must be an integer")
        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        else:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                raise ValueError(f"frame {position} must have uint8 dtype")
            if array.ndim != RGB_FRAME_TENSOR_RANK or array.shape[-1] != RGB_CHANNEL_COUNT:
                raise ValueError(f"frame {position} must have shape [H, W, 3]")
            array = np.ascontiguousarray(array)
        shape = tuple(int(value) for value in array.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"sampled video frames must share one shape: {shape} != {expected_shape}"
            )
        arrays.append(array)
        records.append(
            {
                "index": source_index,
                "width": shape[1],
                "height": shape[0],
                "rgb_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    return np.stack(arrays), records


def _validate_bundle_metadata(
    metadata: Mapping[str, Any],
    frames: np.ndarray,
    *,
    expected_sample_id: str | None,
    expected_source_media_sha256: str | None,
) -> None:
    _validate_bundle_header(
        metadata,
        expected_sample_id=expected_sample_id,
        expected_source_media_sha256=expected_source_media_sha256,
    )
    _validate_bundle_frames(frames)
    _validate_bundle_sampling(metadata, frames)
    metadata_without_identity = dict(metadata)
    stored_identity = metadata_without_identity.pop("bundle_content_sha256", None)
    if stored_identity != canonical_json_sha256(metadata_without_identity):
        raise ValueError("video bundle content identity is inconsistent")


def _validate_bundle_header(
    metadata: Mapping[str, Any],
    *,
    expected_sample_id: str | None,
    expected_source_media_sha256: str | None,
) -> None:
    if metadata.get("schema_version") != VIDEO_BUNDLE_SCHEMA_VERSION:
        raise ValueError("video bundle has unsupported schema_version")
    if metadata.get("record_type") != VIDEO_BUNDLE_RECORD_TYPE:
        raise ValueError("video bundle has unsupported record_type")
    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("video bundle sample_id must be a non-empty string")
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise ValueError("video bundle sample identity differs from request")
    source_sha256 = metadata.get("source_media_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != SHA256_HEX_LENGTH:
        raise ValueError("video bundle source media identity is invalid")
    if expected_source_media_sha256 is not None and source_sha256 != expected_source_media_sha256:
        raise ValueError("video bundle source media SHA256 differs from request")


def _validate_bundle_frames(frames: np.ndarray) -> None:
    if (
        frames.dtype != np.uint8
        or frames.ndim != RGB_VIDEO_TENSOR_RANK
        or frames.shape[-1] != RGB_CHANNEL_COUNT
    ):
        raise ValueError("video bundle frames must have uint8 [N, H, W, 3] shape")


def _validate_bundle_sampling(metadata: Mapping[str, Any], frames: np.ndarray) -> None:
    video_sampling = metadata.get("video_sampling")
    if not isinstance(video_sampling, Mapping):
        raise ValueError("video bundle has no video_sampling object")
    indices = video_sampling.get("sampled_indices")
    if not isinstance(indices, list) or len(indices) != frames.shape[0]:
        raise ValueError("video bundle sampled indices do not match frame count")
    normalized, frame_records = rgb_frame_records(list(frames), indices)
    if not np.array_equal(normalized, frames):
        raise ValueError("video bundle frame normalization changed content")
    expected_rgb_identity = canonical_json_sha256(frame_records)
    if video_sampling.get("sampled_rgb_identity_sha256") != expected_rgb_identity:
        raise ValueError("video bundle RGB identity differs from frame bytes")
    if metadata.get("frame_records") != frame_records:
        raise ValueError("video bundle frame records differ from frame bytes")


def write_video_bundle(
    path: str | Path,
    frames: Sequence[Image.Image | np.ndarray],
    *,
    sample_id: str,
    source_media_sha256: str,
    video_sampling: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write and describe one compressed, lossless RGB bundle."""

    indices = video_sampling.get("sampled_indices")
    if not isinstance(indices, list):
        raise ValueError("video_sampling.sampled_indices must be a list")
    arrays, frame_records = rgb_frame_records(frames, indices)
    if video_sampling.get("sampled_rgb_identity_sha256") != canonical_json_sha256(frame_records):
        raise ValueError("video_sampling RGB identity differs from provided frames")
    metadata: dict[str, Any] = {
        "schema_version": VIDEO_BUNDLE_SCHEMA_VERSION,
        "record_type": VIDEO_BUNDLE_RECORD_TYPE,
        "sample_id": sample_id,
        "source_media_sha256": source_media_sha256,
        "video_sampling": dict(video_sampling),
        "frame_records": frame_records,
    }
    metadata["bundle_content_sha256"] = canonical_json_sha256(metadata)
    metadata_payload = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                metadata=np.frombuffer(metadata_payload, dtype=np.uint8),
                frames=arrays,
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "bundle_content_sha256": metadata["bundle_content_sha256"],
        "sampled_rgb_identity_sha256": video_sampling["sampled_rgb_identity_sha256"],
    }


def load_video_bundle(
    path: str | Path,
    *,
    expected_sample_id: str | None = None,
    expected_source_media_sha256: str | None = None,
) -> tuple[list[Image.Image], dict[str, Any], dict[str, Any]]:
    """Load a bundle after validating metadata and every RGB frame byte."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as bundle:
            if set(bundle.files) != {"metadata", "frames"}:
                raise ValueError("video bundle contains unexpected arrays")
            metadata_array = bundle["metadata"]
            frames = bundle["frames"]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read lossless video bundle: {source}") from exc
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
        raise ValueError("video bundle metadata must be a uint8 vector")
    try:
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("video bundle metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("video bundle metadata must be an object")
    _validate_bundle_metadata(
        metadata,
        frames,
        expected_sample_id=expected_sample_id,
        expected_source_media_sha256=expected_source_media_sha256,
    )
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    evidence = {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "bundle_content_sha256": metadata["bundle_content_sha256"],
    }
    return images, metadata, evidence
