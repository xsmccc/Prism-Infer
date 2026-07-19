"""P6 ``torch.compile`` preflight 的结构化诊断工具。

该模块只封装 PyTorch Dynamo 提供的诊断接口，不参与模型执行语义。Dynamo
本身是成熟的编译基础设施，重写其 graph capture/guard 机制不会给 Prism-Infer
带来项目收益；这里使用安装环境中的 ``torch._dynamo.explain`` 和
``torch._dynamo.optimize`` 作为实验工具，并把结果转换成可审计 JSON record。

参考:
``torch/_dynamo/eval_frame.py:916-966`` 和
``torch/_dynamo/backends/debugging.py:291-333``（当前 PyTorch 安装）。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from prism_infer.analysis.benchmark_schema import summarize_values


MAX_RECORDED_INPUT_SHAPES = 32
COMPILE_PREFLIGHT_SCHEMA_VERSION = 1
COMPILE_PREFLIGHT_REGIONS = (
    "decoder_layer",
    "language_model_decode",
    "vision_encoder",
)
COMPILE_STATUSES = ("pass", "failed", "skipped")


def _tensor_shapes(values: Sequence[Any]) -> list[list[int]]:
    """提取一次调用中所有 tensor 参数的 shape。"""

    flat_values, _ = tree_flatten(values)
    return [list(value.shape) for value in flat_values if isinstance(value, torch.Tensor)]


def summarize_explain_output(output: Any) -> dict[str, Any]:
    """将 ``torch._dynamo.explain`` 输出转换成稳定的 JSON 数据。"""

    break_reasons = []
    for reason in output.break_reasons:
        user_stack = []
        for frame in reason.user_stack:
            user_stack.append(
                {
                    "file": Path(frame.filename).name,
                    "line": int(frame.lineno),
                    "function": frame.name,
                }
            )
        break_reasons.append(
            {
                "reason": str(reason.reason),
                "user_stack": user_stack,
            }
        )

    guards = list(output.out_guards or [])
    guard_sources = Counter(type(guard.originating_source).__name__ for guard in guards)
    guard_create_functions = Counter(
        getattr(guard.create_fn, "__name__", type(guard.create_fn).__name__) for guard in guards
    )
    return {
        "graph_count": int(output.graph_count),
        "graph_break_count": int(output.graph_break_count),
        "op_count": int(output.op_count),
        "ops_per_graph": [len(ops) for ops in (output.ops_per_graph or [])],
        "guard_count": len(guards),
        "guard_sources": dict(sorted(guard_sources.items())),
        "guard_create_functions": dict(sorted(guard_create_functions.items())),
        "break_reasons": break_reasons,
        "compile_times": str(output.compile_times or ""),
    }


@dataclass
class DynamoCompileRecorder:
    """记录 Dynamo eager backend 的编译和 guard failure 事件。"""

    compile_events: list[dict[str, Any]] = field(default_factory=list)
    guard_failures: list[dict[str, Any]] = field(default_factory=list)

    def backend(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[Any],
    ) -> Callable[..., Any]:
        """Dynamo backend：记录 graph 输入后直接执行 FX graph。"""

        tensor_shapes = _tensor_shapes(example_inputs)
        self.compile_events.append(
            {
                "graph_index": len(self.compile_events),
                "tensor_input_count": len(tensor_shapes),
                "input_shapes": tensor_shapes[:MAX_RECORDED_INPUT_SHAPES],
                "input_shapes_truncated": (len(tensor_shapes) > MAX_RECORDED_INPUT_SHAPES),
                "node_count": len(list(graph_module.graph.nodes)),
            }
        )
        return graph_module.forward

    def guard_fail(self, failure: Any) -> None:
        """记录导致 cache miss/recompile 的 guard。"""

        self.guard_failures.append(
            {
                "reason": str(failure.reason),
                "function": str(failure.orig_code.co_name),
                "file": Path(failure.orig_code.co_filename).name,
                "line": int(failure.orig_code.co_firstlineno),
            }
        )


def run_recompile_probe(
    function: Callable[..., Any],
    invocations: Sequence[tuple[Any, ...]],
    *,
    dynamic: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """用 eager FX backend 执行 shape matrix 并记录 recompile 证据。"""

    if not invocations:
        raise ValueError("compile recompile probe requires at least one invocation")
    torch._dynamo.reset()
    recorder = DynamoCompileRecorder()
    optimized = torch._dynamo.optimize(
        recorder.backend,
        dynamic=dynamic,
        guard_fail_fn=recorder.guard_fail,
    )(function)
    outputs = []
    invocation_shapes = []
    try:
        for args in invocations:
            invocation_shapes.append(_tensor_shapes(args))
            outputs.append(optimized(*args))
    finally:
        torch._dynamo.reset()
    return outputs, {
        "dynamic": dynamic,
        "invocation_shapes": invocation_shapes,
        "compile_event_count": len(recorder.compile_events),
        "compile_events": recorder.compile_events,
        "guard_failure_count": len(recorder.guard_failures),
        "guard_failures": recorder.guard_failures,
    }


def compare_tensor_outputs(reference: Any, candidate: Any) -> dict[str, Any]:
    """比较任意 pytree tensor 输出，并记录 shape、误差与统计量。"""

    reference_values, reference_spec = tree_flatten(reference)
    candidate_values, candidate_spec = tree_flatten(candidate)
    if reference_spec != candidate_spec:
        raise ValueError("compiled output pytree structure differs from eager reference")
    if len(reference_values) != len(candidate_values):
        raise ValueError("compiled output leaf count differs from eager reference")

    tensor_records = []
    max_abs_diff = 0.0
    for index, (expected, actual) in enumerate(zip(reference_values, candidate_values)):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            if expected != actual:
                raise ValueError(
                    f"compiled non-tensor output differs at leaf {index}: "
                    f"{expected!r} != {actual!r}"
                )
            continue
        if expected.shape != actual.shape:
            raise ValueError(
                f"compiled tensor shape differs at leaf {index}: "
                f"{list(expected.shape)} != {list(actual.shape)}"
            )
        expected_fp32 = expected.detach().float()
        actual_fp32 = actual.detach().float()
        difference = (expected_fp32 - actual_fp32).abs()
        leaf_max_diff = float(difference.max().item()) if difference.numel() else 0.0
        leaf_mean_diff = float(difference.mean().item()) if difference.numel() else 0.0
        max_abs_diff = max(max_abs_diff, leaf_max_diff)
        tensor_records.append(
            {
                "leaf_index": index,
                "shape": list(expected.shape),
                "dtype": str(expected.dtype),
                "max_abs_diff": leaf_max_diff,
                "mean_abs_diff": leaf_mean_diff,
                "reference_mean": float(expected_fp32.mean().item()),
                "reference_std": float(expected_fp32.std(unbiased=False).item()),
                "candidate_mean": float(actual_fp32.mean().item()),
                "candidate_std": float(actual_fp32.std(unbiased=False).item()),
            }
        )
    if not tensor_records:
        raise ValueError("compile output comparison requires at least one tensor leaf")
    return {
        "tensor_count": len(tensor_records),
        "max_abs_diff": max_abs_diff,
        "tensors": tensor_records,
    }


def validate_compile_preflight_record(record: Mapping[str, Any]) -> None:
    """校验一条 P6.3-B compile preflight record。"""

    if record.get("schema_version") != COMPILE_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported compile preflight schema_version")
    if record.get("record_type") != "compile_preflight":
        raise ValueError("compile preflight record_type is invalid")
    if record.get("region") not in COMPILE_PREFLIGHT_REGIONS:
        raise ValueError("compile preflight region is invalid")
    _validate_compile_environment(record.get("environment"))
    _validate_compile_inputs(record.get("inputs"))
    _validate_dynamo_evidence(record.get("dynamo"))
    _validate_recompile_evidence(record.get("recompile"))
    _validate_compile_benchmark(record.get("benchmark"))


def _validate_compile_environment(raw_environment: object) -> None:
    if not isinstance(raw_environment, Mapping):
        raise ValueError("compile preflight environment must be an object")
    environment = raw_environment
    for key in ("torch_version", "cuda_version", "gpu", "git_commit"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"compile preflight environment.{key} is required")
    if not isinstance(environment.get("git_dirty"), bool):
        raise ValueError("compile preflight environment.git_dirty must be bool")


def _validate_compile_inputs(raw_inputs: object) -> None:
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("compile preflight inputs must be a non-empty list")
    for shape_group in raw_inputs:
        if not isinstance(shape_group, list):
            raise ValueError("compile preflight input shape group must be a list")


def _validate_dynamo_evidence(raw_dynamo: object) -> None:
    if not isinstance(raw_dynamo, Mapping):
        raise ValueError("compile preflight dynamo must be an object")
    dynamo = raw_dynamo
    for key in ("graph_count", "graph_break_count", "op_count", "guard_count"):
        value = dynamo.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"compile preflight dynamo.{key} must be non-negative int")
    if dynamo["graph_break_count"] != max(dynamo["graph_count"] - 1, 0):
        raise ValueError("compile preflight graph count/break count are inconsistent")
    if not isinstance(dynamo.get("break_reasons"), list):
        raise ValueError("compile preflight break_reasons must be a list")


def _validate_recompile_evidence(raw_recompile: object) -> None:
    if not isinstance(raw_recompile, Mapping):
        raise ValueError("compile preflight recompile must be an object")
    recompile = raw_recompile
    if not isinstance(recompile.get("dynamic"), bool):
        raise ValueError("compile preflight recompile.dynamic must be bool")
    for count_key, list_key in (
        ("compile_event_count", "compile_events"),
        ("guard_failure_count", "guard_failures"),
    ):
        count = recompile.get(count_key)
        values = recompile.get(list_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"compile preflight recompile.{count_key} is invalid")
        if not isinstance(values, list) or count != len(values):
            raise ValueError(f"compile preflight recompile.{count_key}/{list_key} mismatch")


def _validate_compile_benchmark(raw_benchmark: object) -> None:
    if not isinstance(raw_benchmark, Mapping):
        raise ValueError("compile preflight benchmark must be an object")
    benchmark = raw_benchmark
    if benchmark.get("status") not in COMPILE_STATUSES:
        raise ValueError("compile preflight benchmark.status is invalid")
    if not isinstance(benchmark.get("attempted"), bool):
        raise ValueError("compile preflight benchmark.attempted must be bool")
    if benchmark["status"] == "skipped":
        if benchmark["attempted"]:
            raise ValueError("skipped compile benchmark cannot be attempted")
        if not isinstance(benchmark.get("reason"), str) or not benchmark["reason"]:
            raise ValueError("skipped compile benchmark requires a reason")
        return
    if not benchmark["attempted"]:
        raise ValueError("pass/failed compile benchmark must be attempted")
    if benchmark["status"] == "failed":
        if not isinstance(benchmark.get("error"), str) or not benchmark["error"]:
            raise ValueError("failed compile benchmark requires an error")
        return
    _validate_successful_compile_benchmark(benchmark)


def _validate_successful_compile_benchmark(benchmark: Mapping[str, Any]) -> None:
    for key in ("warmup", "repeat"):
        value = benchmark.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"compile benchmark {key} must be a positive int")
    for key in (
        "first_call_ms",
        "compile_overhead_ms",
        "allocated_memory_mb",
        "reserved_memory_mb",
        "peak_memory_mb",
    ):
        value = benchmark.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"compile benchmark {key} must be finite and non-negative")
    for key in ("eager_ms", "compiled_ms"):
        _validate_compile_latency_stats(benchmark.get(key), key)
    correctness = benchmark.get("correctness")
    if not isinstance(correctness, Mapping):
        raise ValueError("compile benchmark correctness must be an object")
    max_abs_diff = correctness.get("max_abs_diff")
    if not isinstance(max_abs_diff, (int, float)) or not math.isfinite(max_abs_diff):
        raise ValueError("compile benchmark correctness max_abs_diff is invalid")


def _validate_compile_latency_stats(raw_stats: object, name: str) -> None:
    if not isinstance(raw_stats, Mapping):
        raise ValueError(f"compile benchmark {name} must be stats")
    expected_keys = {"count", "median", "p90", "p99", "min", "max"}
    if set(raw_stats) != expected_keys:
        raise ValueError(f"compile benchmark {name} stats keys are invalid")


def build_latency_stats(values: Sequence[float]) -> dict[str, int | float]:
    """构建 compile benchmark 使用的统一 latency 统计。"""

    return summarize_values(values)
