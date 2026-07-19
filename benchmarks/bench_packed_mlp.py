"""P7.5 Qwen gate/up packed projection microbenchmark。

该 benchmark只比较同一组权重的 legacy gate/up 与 packed gate_up执行，不加载完整
模型。正式性能记录要求 clean commit、低启动显存和 idle GPU；受其他容器污染时仍可
使用 ``--correctness-only`` 验证数值，但输出不得形成性能 claim。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import collect_git_metadata
from prism_infer.models.qwen3_vl import Qwen3VLTextMLP


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _summarize(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize empty timings")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _gpu_baseline() -> dict[str, Any]:
    fields = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi baseline: {fields}")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "memory_used_mb": int(fields[2]),
        "memory_free_mb": int(fields[3]),
        "utilization_gpu_percent": int(fields[4]),
        "power_draw_w": float(fields[5]),
    }


def _formal_environment_issues(
    *,
    git_dirty: bool,
    gpu_baseline: dict[str, Any],
    max_baseline_memory_mb: int,
    max_baseline_utilization_percent: int,
) -> list[str]:
    issues = []
    if git_dirty:
        issues.append("git worktree is dirty")
    if gpu_baseline["memory_used_mb"] > max_baseline_memory_mb:
        issues.append(
            "baseline GPU memory exceeds limit: "
            f"{gpu_baseline['memory_used_mb']} > {max_baseline_memory_mb} MiB"
        )
    if gpu_baseline["utilization_gpu_percent"] > max_baseline_utilization_percent:
        issues.append(
            "baseline GPU utilization exceeds limit: "
            f"{gpu_baseline['utilization_gpu_percent']} > "
            f"{max_baseline_utilization_percent}%"
        )
    return issues


def _legacy_forward(
    mlp: Qwen3VLTextMLP,
    x: torch.Tensor,
) -> torch.Tensor:
    return F.linear(
        F.silu(F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight),
        mlp.down_proj.weight,
    )


def _measure_alternating(
    packed: Callable[[], torch.Tensor],
    legacy: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        packed()
        legacy()
    torch.cuda.synchronize()
    values = {"packed": [], "legacy": []}
    functions = {"packed": packed, "legacy": legacy}
    for iteration in range(repeat):
        order = ("packed", "legacy") if iteration % 2 == 0 else ("legacy", "packed")
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end)))
    return values["packed"], values["legacy"]


def _capture(function: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        output = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def _case(
    mlp: Qwen3VLTextMLP,
    *,
    batch_size: int,
    hidden_size: int,
    warmup: int,
    repeat: int,
    correctness_only: bool,
) -> dict[str, Any]:
    x = torch.randn(
        batch_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def packed() -> torch.Tensor:
        return mlp(x)

    def legacy() -> torch.Tensor:
        return _legacy_forward(mlp, x)

    with torch.inference_mode():
        packed_output = packed()
        legacy_output = legacy()
    difference = (packed_output - legacy_output).abs()
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "packed_output_exact": torch.equal(packed_output, legacy_output),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.float().mean().item()),
        "timing_ms": None,
    }
    if correctness_only:
        return result

    with torch.inference_mode():
        packed_eager, legacy_eager = _measure_alternating(
            packed,
            legacy,
            warmup=warmup,
            repeat=repeat,
        )
        packed_graph, packed_graph_output = _capture(packed)
        legacy_graph, legacy_graph_output = _capture(legacy)
        packed_replay, legacy_replay = _measure_alternating(
            packed_graph.replay,
            legacy_graph.replay,
            warmup=warmup,
            repeat=repeat,
        )
    if not torch.equal(packed_graph_output, legacy_graph_output):
        raise RuntimeError(f"captured outputs diverged for batch={batch_size}")
    result["timing_ms"] = {
        "packed_eager": _summarize(packed_eager),
        "legacy_eager": _summarize(legacy_eager),
        "packed_graph": _summarize(packed_replay),
        "legacy_graph": _summarize(legacy_replay),
    }
    result["packed_over_legacy_eager"] = (
        result["timing_ms"]["packed_eager"]["median"]
        / result["timing_ms"]["legacy_eager"]["median"]
    )
    result["packed_over_legacy_graph"] = (
        result["timing_ms"]["packed_graph"]["median"]
        / result["timing_ms"]["legacy_graph"]["median"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=12288)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--require-formal-environment", action="store_true")
    parser.add_argument("--max-baseline-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--max-baseline-utilization-percent",
        type=int,
        default=5,
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.repeat < 1:
        raise SystemExit("warmup and repeat must be positive")
    batches = [int(value) for value in args.batch_sizes.split(",") if value]
    if not batches or any(batch < 1 for batch in batches):
        raise SystemExit("batch sizes must be positive")

    git = collect_git_metadata(REPO_ROOT, strict=True)
    baseline = _gpu_baseline()
    environment_issues = _formal_environment_issues(
        git_dirty=git.dirty,
        gpu_baseline=baseline,
        max_baseline_memory_mb=args.max_baseline_memory_mb,
        max_baseline_utilization_percent=(args.max_baseline_utilization_percent),
    )
    if args.require_formal_environment and environment_issues:
        raise SystemExit("formal environment gate failed: " + "; ".join(environment_issues))

    torch.manual_seed(20260717)
    mlp = (
        Qwen3VLTextMLP(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    cases = []
    try:
        for batch in batches:
            cases.append(
                _case(
                    mlp,
                    batch_size=batch,
                    hidden_size=args.hidden_size,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    correctness_only=args.correctness_only,
                )
            )
    finally:
        del mlp
        torch.cuda.empty_cache()

    all_exact = all(case["packed_output_exact"] for case in cases)
    formal_eligible = not args.correctness_only and not environment_issues and all_exact
    record = {
        "schema_version": 1,
        "record_type": "p75_packed_mlp_microbenchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_baseline": baseline,
            "formal_environment_issues": environment_issues,
        },
        "configuration": {
            "dtype": "torch.bfloat16",
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "batch_sizes": batches,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "correctness_only": args.correctness_only,
            "alternating_measurement_order": True,
        },
        "correctness": {"all_exact": all_exact},
        "formal_eligible": formal_eligible,
        "cases": cases,
        "claim_boundaries": [
            "This is an isolated MLP microbenchmark, not full-engine TPOT or online goodput.",
            "Timing is claim-eligible only on a clean commit and an idle, low-memory GPU baseline.",
            "Full-model HF logits, quality, E2E, and online regressions remain separate gates.",
        ],
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
