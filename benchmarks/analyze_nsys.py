"""分析 Prism-Infer P6 Nsight Systems SQLite capture。

输入 SQLite 由 ``nsys export --type sqlite`` 生成。脚本读取结构化 CUPTI/NVTX
表，按 engine NVTX range 汇总 kernel、runtime API、同步和 graph execution；
不解析终端文本报告，也不推断未记录的 GPU utilization。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _percentile(values: Sequence[float], fraction: float) -> float:
    """计算非空序列的 nearest-rank 分位数。"""

    if not values:
        raise ValueError("percentile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _summary(values: Sequence[float]) -> dict[str, int | float]:
    """生成 count/median/p90/p99/min/max 汇总。"""

    if not values:
        raise ValueError("summary requires non-empty values")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "count": len(ordered),
        "median": median,
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _require_tables(connection: sqlite3.Connection) -> None:
    """确认输入包含本分析依赖的 Nsight 表。"""

    required = {
        "NVTX_EVENTS",
        "StringIds",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
    }
    actual = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"Nsight SQLite missing required tables: {missing}")


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    """判断可选 Nsight 表是否存在。"""

    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _nvtx_ranges(
    connection: sqlite3.Connection,
    name: str,
) -> list[tuple[int, int]]:
    """读取指定 PushPop NVTX range 的时间区间。"""

    return list(
        connection.execute(
            """
            SELECT n.start, n.end
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON n.textId = s.id
            WHERE coalesce(n.text, s.value) = ? AND n.end IS NOT NULL
            ORDER BY n.start
            """,
            (name,),
        )
    )


def _runtime_counts(
    connection: sqlite3.Connection,
    start: int,
    end: int,
) -> dict[str, int]:
    """统计一个时间区间内的 CUDA runtime/driver API 次数。"""

    return dict(
        connection.execute(
            """
            SELECT s.value, count(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON r.nameId = s.id
            WHERE r.start >= ? AND r.end <= ?
            GROUP BY s.value
            """,
            (start, end),
        )
    )


def _matching_count(counts: dict[str, int], fragment: str) -> int:
    """汇总名称包含指定片段的 API 次数。"""

    return sum(count for name, count in counts.items() if fragment in name)


def _step_metrics(
    connection: sqlite3.Connection,
    start: int,
    end: int,
) -> dict[str, float | int]:
    """提取一个 engine NVTX range 的 kernel 与 launch 指标。"""

    kernels = list(
        connection.execute(
            """
            SELECT start, end, correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start >= ? AND end <= ?
            """,
            (start, end),
        )
    )
    launches = list(
        connection.execute(
            """
            SELECT r.start, r.end, r.correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON r.nameId = s.id
            WHERE s.value LIKE '%LaunchKernel%'
              AND r.start >= ? AND r.end <= ?
            """,
            (start, end),
        )
    )
    graph_traces = []
    if _has_table(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        graph_traces = list(
            connection.execute(
                """
                SELECT start, end
                FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                WHERE start >= ? AND end <= ?
                """,
                (start, end),
            )
        )
    runtime_counts = _runtime_counts(connection, start, end)
    kernel_start_by_correlation = {
        correlation_id: kernel_start
        for kernel_start, _, correlation_id in kernels
    }
    launch_to_kernel_us = [
        (kernel_start_by_correlation[correlation_id] - launch_end) / 1000.0
        for _, launch_end, correlation_id in launches
        if correlation_id in kernel_start_by_correlation
        and kernel_start_by_correlation[correlation_id] >= launch_end
    ]
    return {
        "kernel_count": len(kernels),
        "kernel_launch_api_count": _matching_count(
            runtime_counts,
            "LaunchKernel",
        ),
        "graph_launch_api_count": _matching_count(runtime_counts, "GraphLaunch"),
        "memcpy_async_count": _matching_count(runtime_counts, "MemcpyAsync"),
        "stream_synchronize_count": _matching_count(
            runtime_counts,
            "StreamSynchronize",
        ),
        "kernel_busy_ms": sum(kernel_end - kernel_start for kernel_start, kernel_end, _ in kernels) / 1e6,
        "graph_execution_ms": sum(trace_end - trace_start for trace_start, trace_end in graph_traces) / 1e6,
        "launch_to_kernel_median_us": (
            _summary(launch_to_kernel_us)["median"] if launch_to_kernel_us else 0.0
        ),
        "launch_to_kernel_p90_us": (
            _summary(launch_to_kernel_us)["p90"] if launch_to_kernel_us else 0.0
        ),
    }


def analyze_nsys_sqlite(
    path: str | Path,
    *,
    engine_range: str = "prism::engine.model_runner",
    prefill_steps: int = 1,
    target_ranges: Sequence[str] = (),
) -> dict[str, Any]:
    """分析一个 capture，并返回 JSON-compatible 结构化结果。"""

    sqlite_path = Path(path)
    if prefill_steps < 0:
        raise ValueError("prefill_steps must be >= 0")
    connection = sqlite3.connect(sqlite_path)
    try:
        _require_tables(connection)
        engine_ranges = _nvtx_ranges(connection, engine_range)
        if len(engine_ranges) <= prefill_steps:
            raise ValueError(
                "capture does not contain enough engine ranges for prefill/decode"
            )
        raw_steps = [
            {
                "phase": "prefill" if index < prefill_steps else "decode",
                **_step_metrics(connection, start, end),
            }
            for index, (start, end) in enumerate(engine_ranges)
        ]
        phase_summary: dict[str, dict[str, dict[str, int | float]]] = {}
        for phase in ("prefill", "decode"):
            matching = [step for step in raw_steps if step["phase"] == phase]
            if not matching:
                continue
            phase_summary[phase] = {
                key: _summary([float(step[key]) for step in matching])
                for key in raw_steps[0]
                if key != "phase"
            }

        targets: dict[str, dict[str, int | float]] = {}
        for name in target_ranges:
            ranges = _nvtx_ranges(connection, name)
            if not ranges:
                raise ValueError(f"target NVTX range not found: {name}")
            kernel_counts = []
            kernel_times_ms = []
            runtime_api_counts = []
            memcpy_async_counts = []
            stream_synchronize_counts = []
            graph_times_ms = []
            for start, end in ranges:
                runtime_rows = list(
                    connection.execute(
                        """
                        SELECT r.correlationId, s.value
                        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
                        JOIN StringIds AS s ON r.nameId = s.id
                        WHERE r.start >= ? AND r.end <= ?
                        """,
                        (start, end),
                    )
                )
                correlation_ids = sorted(
                    {
                        correlation_id
                        for correlation_id, _ in runtime_rows
                        if correlation_id is not None
                    }
                )
                kernels = []
                graph_traces = []
                if correlation_ids:
                    placeholders = ",".join("?" for _ in correlation_ids)
                    kernels = list(
                        connection.execute(
                            f"""
                            SELECT start, end
                            FROM CUPTI_ACTIVITY_KIND_KERNEL
                            WHERE correlationId IN ({placeholders})
                            """,
                            correlation_ids,
                        )
                    )
                    if _has_table(
                        connection,
                        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                    ):
                        graph_traces = list(
                            connection.execute(
                                f"""
                                SELECT start, end
                                FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                                WHERE correlationId IN ({placeholders})
                                """,
                                correlation_ids,
                            )
                        )
                kernel_counts.append(float(len(kernels)))
                kernel_times_ms.append(
                    sum(kernel_end - kernel_start for kernel_start, kernel_end in kernels)
                    / 1e6
                )
                runtime_api_counts.append(float(len(runtime_rows)))
                memcpy_async_counts.append(
                    float(
                        sum(
                            "MemcpyAsync" in runtime_name
                            for _, runtime_name in runtime_rows
                        )
                    )
                )
                stream_synchronize_counts.append(
                    float(
                        sum(
                            "StreamSynchronize" in runtime_name
                            for _, runtime_name in runtime_rows
                        )
                    )
                )
                graph_times_ms.append(
                    sum(
                        graph_end - graph_start
                        for graph_start, graph_end in graph_traces
                    )
                    / 1e6
                )
            targets[name] = {
                "range_count": len(ranges),
                "kernel_count_total": int(sum(kernel_counts)),
                "kernel_time_ms_total": sum(kernel_times_ms),
                "runtime_api_count_total": int(sum(runtime_api_counts)),
                "memcpy_async_count_total": int(sum(memcpy_async_counts)),
                "stream_synchronize_count_total": int(
                    sum(stream_synchronize_counts)
                ),
                "graph_execution_ms_total": sum(graph_times_ms),
                "kernels_per_range": _summary(kernel_counts),
                "kernel_time_ms_per_range": _summary(kernel_times_ms),
                "runtime_apis_per_range": _summary(runtime_api_counts),
                "memcpy_async_per_range": _summary(memcpy_async_counts),
                "stream_synchronize_per_range": _summary(
                    stream_synchronize_counts
                ),
                "graph_execution_ms_per_range": _summary(graph_times_ms),
            }

        return {
            "schema_version": 1,
            "record_type": "nsys_profile_summary",
            "source": str(sqlite_path.resolve()),
            "engine_range": engine_range,
            "prefill_steps": prefill_steps,
            "raw_steps": raw_steps,
            "phase_summary": phase_summary,
            "target_ranges": targets,
        }
    finally:
        connection.close()


def main() -> None:
    """Nsight SQLite analyzer CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--engine-range", default="prism::engine.model_runner")
    parser.add_argument("--prefill-steps", type=int, default=1)
    parser.add_argument("--target-range", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()

    result = analyze_nsys_sqlite(
        args.sqlite,
        engine_range=args.engine_range,
        prefill_steps=args.prefill_steps,
        target_ranges=args.target_range,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
