from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from prism_infer.analysis.compile_preflight import (
    COMPILE_PREFLIGHT_SCHEMA_VERSION,
    compare_tensor_outputs,
    run_recompile_probe,
    summarize_explain_output,
    validate_compile_preflight_record,
)


def _complete_record() -> dict:
    return {
        "schema_version": COMPILE_PREFLIGHT_SCHEMA_VERSION,
        "record_type": "compile_preflight",
        "region": "decoder_layer",
        "environment": {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu": "test-gpu",
            "git_commit": "0" * 40,
            "git_dirty": True,
        },
        "inputs": [[[1, 4], [1, 4]]],
        "dynamo": {
            "graph_count": 1,
            "graph_break_count": 0,
            "op_count": 2,
            "guard_count": 3,
            "break_reasons": [],
        },
        "recompile": {
            "dynamic": False,
            "invocation_shapes": [[[1, 4]], [[2, 4]]],
            "compile_event_count": 2,
            "compile_events": [{}, {}],
            "guard_failure_count": 1,
            "guard_failures": [{}],
        },
        "benchmark": {
            "attempted": True,
            "status": "pass",
            "warmup": 2,
            "repeat": 5,
            "first_call_ms": 20.0,
            "compile_overhead_ms": 15.0,
            "allocated_memory_mb": 8.0,
            "reserved_memory_mb": 12.0,
            "peak_memory_mb": 10.0,
            "eager_ms": {
                "count": 5,
                "median": 2.0,
                "p90": 2.2,
                "p99": 2.2,
                "min": 1.9,
                "max": 2.2,
            },
            "compiled_ms": {
                "count": 5,
                "median": 1.5,
                "p90": 1.7,
                "p99": 1.7,
                "min": 1.4,
                "max": 1.7,
            },
            "correctness": {"max_abs_diff": 0.0},
        },
    }


def test_summarize_explain_output_records_graphs_guards_and_ops() -> None:
    def function(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 1

    default_device = torch.get_default_device()
    torch.set_default_device(None)
    try:
        torch._dynamo.reset()
        output = torch._dynamo.explain(function)(torch.ones(2))
        torch._dynamo.reset()
    finally:
        torch.set_default_device(default_device)
    summary = summarize_explain_output(output)
    assert summary["graph_count"] == 1
    assert summary["graph_break_count"] == 0
    assert summary["op_count"] == 2
    assert summary["ops_per_graph"] == [2]
    assert summary["guard_count"] > 0
    assert sum(summary["guard_sources"].values()) == summary["guard_count"]
    print(f"compile explain summary: {summary}")


def test_recompile_probe_records_static_shape_guard_failure() -> None:
    def function(x: torch.Tensor) -> torch.Tensor:
        return x.sin() + 1

    outputs, report = run_recompile_probe(
        function,
        [(torch.ones(1, 4),), (torch.ones(2, 4),)],
        dynamic=False,
    )
    assert len(outputs) == 2
    assert report["compile_event_count"] == 2
    assert report["guard_failure_count"] >= 1
    assert report["invocation_shapes"] == [[[1, 4]], [[2, 4]]]
    print(f"compile recompile report: {report}")


def test_compare_tensor_outputs_reports_numerical_evidence() -> None:
    reference = (torch.tensor([1.0, 2.0]), [torch.tensor([3.0])])
    candidate = (torch.tensor([1.0, 2.001]), [torch.tensor([3.0])])
    comparison = compare_tensor_outputs(reference, candidate)
    assert comparison["tensor_count"] == 2
    assert comparison["max_abs_diff"] == pytest.approx(0.001, abs=1e-6)
    assert comparison["tensors"][0]["shape"] == [2]
    print(f"compile correctness comparison: {comparison}")


def test_compile_preflight_record_accepts_complete_pass() -> None:
    record = _complete_record()
    validate_compile_preflight_record(record)
    print("compile preflight complete record: PASS")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("region",), "scheduler"),
        (("dynamo", "graph_break_count"), 1),
        (("recompile", "compile_event_count"), 1),
        (("benchmark", "warmup"), 0),
    ],
)
def test_compile_preflight_record_rejects_invalid_contract(
    path: tuple[str, ...],
    value: object,
) -> None:
    record = deepcopy(_complete_record())
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_compile_preflight_record(record)


def test_compile_preflight_record_accepts_explicit_skip() -> None:
    record = _complete_record()
    record["benchmark"] = {
        "attempted": False,
        "status": "skipped",
        "reason": "graph breaks exceed the preflight gate",
    }
    validate_compile_preflight_record(record)
    print("compile preflight explicit skip: PASS")
