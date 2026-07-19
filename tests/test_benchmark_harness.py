"""Shared benchmark harness identity and workload contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import harness


pytestmark = pytest.mark.unit


def test_git_metadata_has_one_fail_closed_implementation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_check_output(command, **kwargs):
        calls.append(tuple(command))
        assert kwargs["text"] is True
        if command[-2:] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[-2:] == ["status", "--porcelain"]:
            return " M changed.py\n"
        raise AssertionError(command)

    monkeypatch.setattr(harness.subprocess, "check_output", fake_check_output)
    metadata = harness.collect_git_metadata(tmp_path, strict=True)

    assert metadata.commit == "a" * 40
    assert metadata.dirty is True
    assert metadata.as_dict(
        commit_key="git_commit",
        dirty_key="git_dirty",
    ) == {"git_commit": "a" * 40, "git_dirty": True}
    assert len(calls) == 2


def test_git_metadata_fallback_is_explicit_and_strict_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(harness.subprocess, "check_output", fail)

    assert harness.collect_git_metadata(tmp_path) == harness.GitMetadata(
        commit="unknown",
        dirty=True,
    )
    with pytest.raises(RuntimeError, match="cannot establish Git identity"):
        harness.collect_git_metadata(tmp_path, strict=True)


def test_gpu_metadata_matches_nvidia_smi_by_uuid_not_logical_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Synthetic GPU",
        uuid="bbbb",
        major=12,
        minor=0,
        multi_processor_count=192,
        total_memory=32 * 1024**3,
    )
    monkeypatch.setattr(
        harness.torch.cuda,
        "get_device_properties",
        lambda index: properties,
    )
    monkeypatch.setattr(harness.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    monkeypatch.setattr(
        harness.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "GPU-aaaa, 590.1, 11, 21, 1, 1001, 31\nGPU-bbbb, 590.2, 12, 22, 2, 1002, 32\n"
        ),
    )

    metadata = harness.collect_gpu_metadata(1, strict_identity=True)

    assert metadata.gpu_uuid == "GPU-bbbb"
    assert metadata.driver == "590.2"
    assert metadata.cuda_visible_devices == "7,3"
    assert metadata.nvidia_smi["memory_used_mib"] == 12.0
    assert metadata.environment_dict() == {
        "gpu": "Synthetic GPU",
        "compute_capability": "12.0",
        "total_memory_bytes": 32 * 1024**3,
        "gpu_uuid": "GPU-bbbb",
        "driver": "590.2",
    }
    assert metadata.detailed_dict()["logical_device_index"] == 1


def test_gpu_metadata_rejects_ambiguous_identity_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Synthetic GPU",
        uuid="unknown",
        major=9,
        minor=0,
        multi_processor_count=1,
        total_memory=1024,
    )
    monkeypatch.setattr(
        harness.torch.cuda,
        "get_device_properties",
        lambda index: properties,
    )
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *args, **kwargs: "")

    with pytest.raises(RuntimeError, match="cannot establish GPU identity"):
        harness.collect_gpu_metadata(strict_identity=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        harness.collect_gpu_metadata(True)
