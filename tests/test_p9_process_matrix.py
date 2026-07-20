"""P9 fresh-process benchmark orchestration contracts."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.run_p9_process_matrix import (
    REPO_ROOT,
    GpuState,
    balanced_pair_order,
    bootstrap_median_ratio,
    comparability_checks,
    parse_gpu_state,
    validate_artifact_destination,
    validate_idle_gpu,
)

EXPECTED_MEMORY_USED_MIB = 2.0
EXPECTED_HALF_RATIO = 0.5
TEST_BOOTSTRAP_RESAMPLES = 100


@pytest.mark.parametrize("repeats", [1, 2, 5, 8])
def test_balanced_pair_order_has_exact_counts(repeats: int) -> None:
    order = balanced_pair_order(repeats)
    assert len(order) == 2 * repeats
    assert order.count("A") == repeats
    assert order.count("B") == repeats
    assert order[:8] == tuple("ABBABAAB")[: len(order[:8])]


def test_gpu_state_matches_physical_uuid_and_idle_gate() -> None:
    state = parse_gpu_state(
        "GPU-other, NVIDIA GeForce RTX 5090, 1, 32149, 0\n"
        "GPU-target, NVIDIA GeForce RTX 5090, 2, 32148, 1\n",
        expected_uuid="GPU-target",
    )
    assert state.uuid == "GPU-target"
    assert state.memory_used_mib == EXPECTED_MEMORY_USED_MIB
    validate_idle_gpu(
        state,
        max_memory_used_mib=64.0,
        max_utilization_percent=5.0,
    )


@pytest.mark.parametrize(
    "state, message",
    [
        (GpuState("GPU-x", "gpu", 65.0, 0.0, 0.0), "memory"),
        (GpuState("GPU-x", "gpu", 1.0, 0.0, 6.0), "utilization"),
    ],
)
def test_gpu_idle_gate_rejects_contaminated_runs(state: GpuState, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_idle_gpu(
            state,
            max_memory_used_mib=64.0,
            max_utilization_percent=5.0,
        )


def test_bootstrap_median_ratio_is_deterministic_for_constant_samples() -> None:
    result = bootstrap_median_ratio(
        [10.0] * 5,
        [5.0] * 5,
        seed=7,
        resamples=TEST_BOOTSTRAP_RESAMPLES,
    )
    assert result["point_estimate"] == EXPECTED_HALF_RATIO
    assert result["confidence_interval_95"] == [EXPECTED_HALF_RATIO, EXPECTED_HALF_RATIO]


def _comparable_record(mode: str, execution: str) -> dict[str, object]:
    graph_enabled = execution == "cuda_graph"
    return {
        "environment": {"gpu_uuid": "GPU-x", "git_commit": "a" * 40},
        "model": {"path": "/model", "dtype": "torch.bfloat16"},
        "workload": {"case_id": "h1", "prompt_tokens": 10},
        "traffic": {"batch_size": 1},
        "sampling": {"max_tokens": 4},
        "measurement": {"warmup": 2},
        "mode": {
            "name": mode,
            "execution": execution,
            "attention": "paged",
            "compression": "off",
        },
        "execution_backend": {
            "decode_backend": execution,
            "cuda_graph_enabled": graph_enabled,
            "cuda_graph_capture_scope": "decode" if graph_enabled else "none",
            "cuda_graph_capture_ms": 1.0 if graph_enabled else 0.0,
            "cuda_graph_batch_sizes": [1] if graph_enabled else [],
            "cuda_graph_replay_counts": (
                [{"actual_batch_size": 1, "captured_batch_size": 1, "count": 3}]
                if graph_enabled
                else []
            ),
            "decode_batch_size_counts": [{"actual_batch_size": 1, "count": 3}],
            "vision_attention_backend": "sdpa",
            "requested_decode_batch_size": 1,
        },
        "kv_cache": {"bytes": 1024, "blocks": 1},
        "correctness": {
            "token_ids": [[1, 2, 3, 4]],
            "decoded_texts": ["same"],
            "output_tokens": 4,
        },
    }


def test_comparability_allows_only_cuda_graph_execution_differences() -> None:
    eager = _comparable_record("off_eager", "eager")
    graph = _comparable_record("off_graph", "cuda_graph")
    checks = comparability_checks(
        [eager, graph],
        mode_a="off_eager",
        mode_b="off_graph",
        repeats_per_mode=1,
        expected_order=("off_eager", "off_graph"),
        expected_output_tokens=4,
    )
    assert all(checks.values())

    changed_workload = copy.deepcopy(graph)
    changed_workload["workload"]["prompt_tokens"] = 11
    failed = comparability_checks(
        [eager, changed_workload],
        mode_a="off_eager",
        mode_b="off_graph",
        repeats_per_mode=1,
        expected_order=("off_eager", "off_graph"),
        expected_output_tokens=4,
    )
    assert not failed["workload_exact"]


def test_artifact_destination_must_not_dirty_the_repository(tmp_path: Path) -> None:
    validate_artifact_destination(tmp_path / "outside-repository.jsonl")
    validate_artifact_destination(REPO_ROOT / "data" / "ignored.jsonl")
    with pytest.raises(RuntimeError, match="gitignored"):
        validate_artifact_destination(REPO_ROOT / "benchmarks" / "tracked-output.jsonl")


def test_importing_orchestrator_does_not_initialize_torch() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import benchmarks.run_p9_process_matrix; "
            "raise SystemExit('torch' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    assert completed.returncode == 0
