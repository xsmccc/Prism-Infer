"""Shared, framework-neutral benchmark harness primitives.

The helpers in this module own source identity, GPU identity, deterministic
workload expansion, and media materialization.  Keeping these boundaries in a
single module prevents internal, online, kernel, and external runners from
quietly recording incompatible evidence.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class GitMetadata:
    """Exact benchmark harness source state."""

    commit: str
    dirty: bool

    def as_dict(
        self,
        *,
        commit_key: str = "commit",
        dirty_key: str = "dirty",
    ) -> dict[str, str | bool]:
        return {commit_key: self.commit, dirty_key: self.dirty}


def collect_git_metadata(
    repo_root: str | Path = REPO_ROOT,
    *,
    strict: bool = False,
) -> GitMetadata:
    """Collect commit and dirty state, optionally failing on missing Git data."""

    root = Path(repo_root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if strict:
            raise RuntimeError(f"cannot establish Git identity for {root}") from exc
        return GitMetadata(commit="unknown", dirty=True)
    return GitMetadata(commit=commit, dirty=dirty)


def _safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _normalize_gpu_uuid(raw: object) -> str:
    value = str(raw)
    if not value or value.lower() in {"none", "unknown"}:
        return "unknown"
    return value if value.startswith("GPU-") else f"GPU-{value}"


def _query_nvidia_smi(gpu_uuid: str) -> dict[str, Any]:
    """Match NVSMI state by UUID, never by remapped CUDA logical index."""

    query_fields = (
        "uuid",
        "driver_version",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "clocks.sm",
        "power.draw",
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}

    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == len(query_fields):
            rows.append(fields)
    normalized_target = gpu_uuid.removeprefix("GPU-").lower()
    matching = [
        fields for fields in rows if fields[0].removeprefix("GPU-").lower() == normalized_target
    ]
    if gpu_uuid == "unknown" and len(rows) == 1 and torch.cuda.device_count() == 1:
        matching = rows
    if len(matching) != 1:
        return {"available": False}
    fields = matching[0]
    return {
        "available": True,
        "gpu_uuid": fields[0],
        "driver": fields[1],
        "memory_used_mib": _safe_float(fields[2]),
        "memory_free_mib": _safe_float(fields[3]),
        "utilization_gpu_percent": _safe_float(fields[4]),
        "sm_clock_mhz": _safe_float(fields[5]),
        "power_draw_w": _safe_float(fields[6]),
    }


@dataclass(frozen=True, slots=True)
class GpuMetadata:
    """Stable CUDA identity plus a same-UUID NVSMI snapshot."""

    logical_device_index: int
    name: str
    gpu_uuid: str
    driver: str
    compute_capability: str
    multiprocessor_count: int
    total_memory_bytes: int
    cuda_visible_devices: str | None
    nvidia_visible_devices: str | None
    nvidia_smi: dict[str, Any]

    def environment_dict(self) -> dict[str, Any]:
        """Return the compact identity used by system benchmark schemas."""

        return {
            "gpu": self.name,
            "compute_capability": self.compute_capability,
            "total_memory_bytes": self.total_memory_bytes,
            "gpu_uuid": self.gpu_uuid,
            "driver": self.driver,
        }

    def detailed_dict(self) -> dict[str, Any]:
        """Return logical mapping and live state used by kernel preflight."""

        return {
            "logical_device_index": self.logical_device_index,
            "name": self.name,
            "gpu_uuid": self.gpu_uuid,
            "compute_capability": self.compute_capability,
            "multiprocessor_count": self.multiprocessor_count,
            "total_memory_bytes": self.total_memory_bytes,
            "cuda_visible_devices": self.cuda_visible_devices,
            "nvidia_visible_devices": self.nvidia_visible_devices,
            "nvidia_smi": dict(self.nvidia_smi),
        }


def collect_gpu_metadata(
    device_index: int = 0,
    *,
    strict_identity: bool = False,
) -> GpuMetadata:
    """Collect CUDA properties and match NVSMI metadata by physical UUID."""

    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError(f"device_index must be a non-negative integer, got {device_index!r}")
    properties = torch.cuda.get_device_properties(device_index)
    property_uuid = _normalize_gpu_uuid(getattr(properties, "uuid", "unknown"))
    nvidia_smi = _query_nvidia_smi(property_uuid)
    gpu_uuid = str(nvidia_smi.get("gpu_uuid", property_uuid))
    driver = str(nvidia_smi.get("driver", "unknown"))
    if strict_identity and (gpu_uuid == "unknown" or driver == "unknown"):
        raise RuntimeError(f"cannot establish GPU identity for CUDA logical device {device_index}")
    return GpuMetadata(
        logical_device_index=device_index,
        name=properties.name,
        gpu_uuid=gpu_uuid,
        driver=driver,
        compute_capability=f"{properties.major}.{properties.minor}",
        multiprocessor_count=properties.multi_processor_count,
        total_memory_bytes=properties.total_memory,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nvidia_visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        nvidia_smi=nvidia_smi,
    )


def _make_image(spec: Mapping[str, Any]) -> Image.Image:
    size = (int(spec["width"]), int(spec["height"]))
    color = tuple(int(channel) for channel in spec["color"])
    return Image.new("RGB", size, color=color)


def _load_file_image(
    spec: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> Image.Image:
    configured_path = Path(spec["path"])
    image_path = (
        configured_path if configured_path.is_absolute() else Path(repo_root) / configured_path
    )
    if not image_path.is_file():
        raise FileNotFoundError(
            f"benchmark image is missing: {image_path}; run scripts/download_p6_real_samples.sh"
        )
    actual_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    expected_sha256 = str(spec["sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"benchmark image SHA256 mismatch for {image_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    expected_size = (int(spec["width"]), int(spec["height"]))
    if image.size != expected_size:
        image.close()
        raise ValueError(
            f"benchmark image size mismatch for {image_path}: "
            f"expected {expected_size}, got {image.size}"
        )
    return image


def materialize_requests(
    case: Mapping[str, Any],
    *,
    repo_root: str | Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Convert deterministic JSON media specs into public LLM payloads."""

    requests: list[dict[str, Any]] = []
    for request in case["requests"]:
        request_type = request["type"]
        materialized: dict[str, Any] = {
            "type": request_type,
            "prompt": request["prompt"],
        }
        if request_type == "image":
            materialized["image"] = _make_image(request["image"])
        elif request_type == "image_file":
            materialized["type"] = "image"
            materialized["image"] = _load_file_image(
                request["image"],
                repo_root=repo_root,
            )
        elif request_type == "images":
            materialized["images"] = [_make_image(image) for image in request["images"]]
        elif request_type == "video":
            materialized["video"] = [_make_image(frame) for frame in request["frames"]]
        requests.append(materialized)
    return requests


def describe_case_inputs(
    case: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Describe visual shapes and image/video counts without materialization."""

    input_shapes: list[dict[str, Any]] = []
    image_count = 0
    video_count = 0
    video_frame_count = 0
    for request in case["requests"]:
        request_type = request["type"]
        visual_specs: list[Mapping[str, Any]] = []
        if request_type in ("image", "image_file"):
            visual_specs = [request["image"]]
            image_count += 1
        elif request_type == "images":
            visual_specs = request["images"]
            image_count += len(visual_specs)
        elif request_type == "video":
            visual_specs = request["frames"]
            video_count += 1
            video_frame_count += len(visual_specs)
        input_shapes.append(
            {
                "type": request_type,
                "visual_shapes": [
                    [int(spec["height"]), int(spec["width"]), 3] for spec in visual_specs
                ],
            }
        )
    return input_shapes, image_count, video_count, video_frame_count


def find_workload_case(
    manifest: Mapping[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Resolve one named case and fail with the complete available set."""

    for case in manifest["cases"]:
        if case["id"] == case_id:
            return case
    available = [case["id"] for case in manifest["cases"]]
    raise ValueError(f"unknown case {case_id!r}; available cases: {available}")


def expand_case_batch(
    case: Mapping[str, Any],
    batch_size: int,
) -> tuple[dict[str, Any], int, int]:
    """Replicate complete request groups into an explicit offline batch."""

    source_num_requests = len(case["requests"])
    if batch_size < source_num_requests or batch_size % source_num_requests != 0:
        raise ValueError(
            "batch size must be a positive multiple of the source case size: "
            f"case={case['id']}, source={source_num_requests}, requested={batch_size}"
        )
    replication_factor = batch_size // source_num_requests
    expanded = deepcopy(case)
    expanded["requests"] = [
        deepcopy(request) for _ in range(replication_factor) for request in case["requests"]
    ]
    return expanded, source_num_requests, replication_factor
