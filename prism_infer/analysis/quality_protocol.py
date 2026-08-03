"""Validation for the deterministic quality-data protocol.

The protocol records model, dataset, and sample-selection identities. Quality
results are reported directly; this module does not encode pass thresholds or
performance policy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

QUALITY_PROTOCOL_SCHEMA_VERSION = 2
GIT_REVISION_HEX_LENGTH = 40
QUALITY_MATERIALIZATION_STATUSES = {
    "pending",
    "materialized",
    "conditional_manual_media",
    "excluded",
}


def _mapping(container: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _list(container: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty list")
    return value


def _string(container: Mapping[str, Any], key: str, path: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _bool(container: Mapping[str, Any], key: str, path: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a bool")
    return value


def _positive_int(container: Mapping[str, Any], key: str, path: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive int")
    return value


def _git_revision(container: Mapping[str, Any], key: str, path: str) -> str:
    revision = _string(container, key, path)
    if len(revision) != GIT_REVISION_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{path}.{key} must be a full lowercase Git revision")
    return revision


def validate_quality_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate dataset revisions and deterministic sample selection."""

    if protocol.get("schema_version") != QUALITY_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported quality protocol schema_version")
    _string(protocol, "name", "protocol")
    model = _mapping(protocol, "model", "protocol")
    _git_revision(model, "revision", "protocol.model")
    _git_revision(model, "processor_revision", "protocol.model")
    _validate_selection(protocol)
    _validate_datasets(protocol)


def _validate_selection(protocol: Mapping[str, Any]) -> None:
    selection = _mapping(protocol, "selection", "protocol")
    _string(selection, "algorithm", "protocol.selection")
    _positive_int(selection, "seed", "protocol.selection")
    _bool(selection, "materialization_requires_media_sha256", "protocol.selection")


def _validate_datasets(protocol: Mapping[str, Any]) -> None:
    dataset_ids: set[str] = set()
    for index, dataset in enumerate(_list(protocol, "datasets", "protocol")):
        path = f"protocol.datasets[{index}]"
        if not isinstance(dataset, Mapping):
            raise ValueError(f"{path} must be an object")
        dataset_id = _string(dataset, "id", path)
        if dataset_id in dataset_ids:
            raise ValueError(f"duplicate quality dataset id {dataset_id!r}")
        dataset_ids.add(dataset_id)
        for key in ("category", "source", "repository", "split", "sample_id_field", "metric"):
            _string(dataset, key, path)
        _git_revision(dataset, "revision", path)
        development_samples = _positive_int(dataset, "development_samples", path)
        final_samples = _positive_int(dataset, "final_samples", path)
        if development_samples > final_samples:
            raise ValueError(f"{path}.development_samples exceeds final_samples")
        status = _string(dataset, "materialization_status", path)
        if status not in QUALITY_MATERIALIZATION_STATUSES:
            raise ValueError(f"{path}.materialization_status is unsupported: {status!r}")


def load_quality_protocol(path: str | Path) -> dict[str, Any]:
    """Read and validate a quality protocol JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_quality_protocol(protocol)
    return protocol
