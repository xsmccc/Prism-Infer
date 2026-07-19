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
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
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
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
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


def _table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    """返回 SQLite 表字段；表不存在时返回空集合。"""

    if not _has_table(connection, name):
        return set()
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({name})")}


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    """计算一组半开时间区间的 union 长度。"""

    valid = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not valid:
        return 0
    total = 0
    current_start, current_end = valid[0]
    for start, end in valid[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _clipped_interval_union_ns(
    intervals: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> int:
    """计算 GPU activity 与一个 CPU range 的相交 union 长度。"""

    return _interval_union_ns(
        [
            (max(start, interval_start), min(end, interval_end))
            for interval_start, interval_end in intervals
        ]
    )


def _kernel_category(name: str) -> str:
    """按可审计的 kernel-name 规则生成 operation category。"""

    lowered = name.lower()
    if "gemv" in lowered or "gemm" in lowered:
        return "linear_gemv"
    if "_paged_decode_attention_kernel" in lowered:
        return "paged_decode_attention"
    if "_store_kvcache" in lowered:
        return "kv_store"
    if "reduce_kernel" in lowered:
        return "reduction"
    if "copy_kernel" in lowered or "bfloat16_copy" in lowered:
        return "copy_cast"
    if "catarray" in lowered or "indexselect" in lowered or "fillfunctor" in lowered:
        return "layout_index"
    if "sin_kernel" in lowered or "cos_kernel" in lowered:
        return "trigonometric"
    if "elementwise_kernel" in lowered or "silu_kernel" in lowered:
        return "elementwise"
    return "other"


def _runtime_rows(
    connection: sqlite3.Connection,
    start: int,
    end: int,
) -> list[tuple[int | None, str]]:
    """读取完全落在一个 CPU NVTX range 内的 CUDA runtime 调用。"""

    return list(
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


def _kernel_rows_for_correlations(
    connection: sqlite3.Connection,
    correlation_ids: Sequence[int],
) -> list[tuple[int, int, str]]:
    """按 runtime correlation 解析 kernel interval 与可读名称。"""

    if not correlation_ids:
        return []
    columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_columns = [
        column for column in ("demangledName", "shortName", "mangledName") if column in columns
    ]
    joins = []
    names = []
    for index, column in enumerate(name_columns):
        alias = f"kernel_name_{index}"
        joins.append(f"LEFT JOIN StringIds AS {alias} ON k.{column} = {alias}.id")
        names.append(f"{alias}.value")
    name_expression = f"coalesce({', '.join(names)}, '<unknown>')" if names else "'<unknown>'"
    placeholders = ",".join("?" for _ in correlation_ids)
    return [
        (int(start), int(end), str(name))
        for start, end, name in connection.execute(
            f"""
            SELECT k.start, k.end, {name_expression}
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            {" ".join(joins)}
            WHERE k.correlationId IN ({placeholders})
            """,
            tuple(correlation_ids),
        )
    ]


def _memory_rows_for_correlations(
    connection: sqlite3.Connection,
    table: str,
    correlation_ids: Sequence[int],
) -> list[tuple[int, int, int]]:
    """读取 memcpy/memset interval 与 bytes；旧 trace 缺表时返回空。"""

    columns = _table_columns(connection, table)
    if not correlation_ids or not {"start", "end", "correlationId"} <= columns:
        return []
    bytes_expression = "bytes" if "bytes" in columns else "0"
    placeholders = ",".join("?" for _ in correlation_ids)
    return [
        (int(start), int(end), int(num_bytes))
        for start, end, num_bytes in connection.execute(
            f"""
            SELECT start, end, {bytes_expression}
            FROM {table}
            WHERE correlationId IN ({placeholders})
            """,
            tuple(correlation_ids),
        )
    ]


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
        correlation_id: kernel_start for kernel_start, _, correlation_id in kernels
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
        "kernel_busy_ms": sum(kernel_end - kernel_start for kernel_start, kernel_end, _ in kernels)
        / 1e6,
        "graph_execution_ms": sum(
            trace_end - trace_start for trace_start, trace_end in graph_traces
        )
        / 1e6,
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
            raise ValueError("capture does not contain enough engine ranges for prefill/decode")
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

        targets: dict[str, dict[str, Any]] = {}
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
            cpu_range_times_ms = []
            memcpy_times_ms = []
            memcpy_bytes = []
            memset_times_ms = []
            memset_bytes = []
            gpu_busy_times_ms = []
            gpu_spans_ms = []
            cpu_gpu_overlap_times_ms = []
            cpu_gpu_overlap_fractions = []
            gpu_tail_after_cpu_ms = []
            gpu_start_offsets_us = []
            category_counts_by_range: list[dict[str, float]] = []
            category_times_by_range: list[dict[str, float]] = []
            kernel_name_totals: dict[str, list[float]] = {}
            for start, end in ranges:
                runtime_rows = _runtime_rows(connection, start, end)
                correlation_ids = sorted(
                    {
                        correlation_id
                        for correlation_id, _ in runtime_rows
                        if correlation_id is not None
                    }
                )
                kernel_rows = _kernel_rows_for_correlations(
                    connection,
                    correlation_ids,
                )
                kernels = [
                    (kernel_start, kernel_end) for kernel_start, kernel_end, _ in kernel_rows
                ]
                memcpy_rows = _memory_rows_for_correlations(
                    connection,
                    "CUPTI_ACTIVITY_KIND_MEMCPY",
                    correlation_ids,
                )
                memset_rows = _memory_rows_for_correlations(
                    connection,
                    "CUPTI_ACTIVITY_KIND_MEMSET",
                    correlation_ids,
                )
                graph_traces = []
                if correlation_ids:
                    placeholders = ",".join("?" for _ in correlation_ids)
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
                    sum(kernel_end - kernel_start for kernel_start, kernel_end in kernels) / 1e6
                )
                runtime_api_counts.append(float(len(runtime_rows)))
                memcpy_async_counts.append(
                    float(sum("MemcpyAsync" in runtime_name for _, runtime_name in runtime_rows))
                )
                stream_synchronize_counts.append(
                    float(
                        sum("StreamSynchronize" in runtime_name for _, runtime_name in runtime_rows)
                    )
                )
                graph_times_ms.append(
                    sum(graph_end - graph_start for graph_start, graph_end in graph_traces) / 1e6
                )
                cpu_range_ns = end - start
                cpu_range_times_ms.append(cpu_range_ns / 1e6)
                memcpy_times_ms.append(
                    sum(row_end - row_start for row_start, row_end, _ in memcpy_rows) / 1e6
                )
                memcpy_bytes.append(float(sum(row[2] for row in memcpy_rows)))
                memset_times_ms.append(
                    sum(row_end - row_start for row_start, row_end, _ in memset_rows) / 1e6
                )
                memset_bytes.append(float(sum(row[2] for row in memset_rows)))
                gpu_intervals = [
                    *kernels,
                    *[(row[0], row[1]) for row in memcpy_rows],
                    *[(row[0], row[1]) for row in memset_rows],
                ]
                gpu_busy_times_ms.append(_interval_union_ns(gpu_intervals) / 1e6)
                if gpu_intervals:
                    gpu_start = min(interval[0] for interval in gpu_intervals)
                    gpu_end = max(interval[1] for interval in gpu_intervals)
                    gpu_spans_ms.append((gpu_end - gpu_start) / 1e6)
                    overlap_ns = _clipped_interval_union_ns(
                        gpu_intervals,
                        start,
                        end,
                    )
                    cpu_gpu_overlap_times_ms.append(overlap_ns / 1e6)
                    cpu_gpu_overlap_fractions.append(
                        overlap_ns / cpu_range_ns if cpu_range_ns else 0.0
                    )
                    gpu_tail_after_cpu_ms.append(max(0, gpu_end - end) / 1e6)
                    gpu_start_offsets_us.append((gpu_start - start) / 1000.0)
                else:
                    gpu_spans_ms.append(0.0)
                    cpu_gpu_overlap_times_ms.append(0.0)
                    cpu_gpu_overlap_fractions.append(0.0)
                    gpu_tail_after_cpu_ms.append(0.0)
                    gpu_start_offsets_us.append(0.0)

                category_counts: dict[str, float] = {}
                category_times: dict[str, float] = {}
                for kernel_start, kernel_end, kernel_name in kernel_rows:
                    duration_ms = (kernel_end - kernel_start) / 1e6
                    category = _kernel_category(kernel_name)
                    category_counts[category] = category_counts.get(category, 0.0) + 1
                    category_times[category] = category_times.get(category, 0.0) + duration_ms
                    totals = kernel_name_totals.setdefault(kernel_name, [0.0, 0.0])
                    totals[0] += 1
                    totals[1] += duration_ms
                category_counts_by_range.append(category_counts)
                category_times_by_range.append(category_times)

            all_categories = sorted(
                {category for per_range in category_times_by_range for category in per_range}
            )
            total_kernel_time_ms = sum(kernel_times_ms)
            kernel_categories = {
                category: {
                    "kernel_count_total": int(
                        sum(values.get(category, 0.0) for values in category_counts_by_range)
                    ),
                    "kernel_time_ms_total": sum(
                        values.get(category, 0.0) for values in category_times_by_range
                    ),
                    "kernel_time_fraction": (
                        min(
                            1.0,
                            sum(values.get(category, 0.0) for values in category_times_by_range)
                            / total_kernel_time_ms,
                        )
                        if total_kernel_time_ms
                        else 0.0
                    ),
                    "kernels_per_range": _summary(
                        [values.get(category, 0.0) for values in category_counts_by_range]
                    ),
                    "kernel_time_ms_per_range": _summary(
                        [values.get(category, 0.0) for values in category_times_by_range]
                    ),
                }
                for category in all_categories
            }
            top_kernels = [
                {
                    "name": kernel_name,
                    "category": _kernel_category(kernel_name),
                    "kernel_count_total": int(values[0]),
                    "kernel_time_ms_total": values[1],
                    "kernel_time_fraction": (
                        min(1.0, values[1] / total_kernel_time_ms) if total_kernel_time_ms else 0.0
                    ),
                }
                for kernel_name, values in sorted(
                    kernel_name_totals.items(),
                    key=lambda item: (-item[1][1], item[0]),
                )[:20]
            ]
            targets[name] = {
                "range_count": len(ranges),
                "kernel_count_total": int(sum(kernel_counts)),
                "kernel_time_ms_total": sum(kernel_times_ms),
                "runtime_api_count_total": int(sum(runtime_api_counts)),
                "memcpy_async_count_total": int(sum(memcpy_async_counts)),
                "stream_synchronize_count_total": int(sum(stream_synchronize_counts)),
                "graph_execution_ms_total": sum(graph_times_ms),
                "memcpy_time_ms_total": sum(memcpy_times_ms),
                "memcpy_bytes_total": int(sum(memcpy_bytes)),
                "memset_time_ms_total": sum(memset_times_ms),
                "memset_bytes_total": int(sum(memset_bytes)),
                "kernels_per_range": _summary(kernel_counts),
                "kernel_time_ms_per_range": _summary(kernel_times_ms),
                "runtime_apis_per_range": _summary(runtime_api_counts),
                "memcpy_async_per_range": _summary(memcpy_async_counts),
                "stream_synchronize_per_range": _summary(stream_synchronize_counts),
                "graph_execution_ms_per_range": _summary(graph_times_ms),
                "cpu_range_ms_per_range": _summary(cpu_range_times_ms),
                "memcpy_time_ms_per_range": _summary(memcpy_times_ms),
                "memcpy_bytes_per_range": _summary(memcpy_bytes),
                "memset_time_ms_per_range": _summary(memset_times_ms),
                "memset_bytes_per_range": _summary(memset_bytes),
                "gpu_busy_ms_per_range": _summary(gpu_busy_times_ms),
                "gpu_span_ms_per_range": _summary(gpu_spans_ms),
                "cpu_gpu_busy_overlap_ms_per_range": _summary(cpu_gpu_overlap_times_ms),
                "cpu_gpu_busy_overlap_fraction_per_range": _summary(cpu_gpu_overlap_fractions),
                "gpu_tail_after_cpu_ms_per_range": _summary(gpu_tail_after_cpu_ms),
                "gpu_start_offset_us_per_range": _summary(gpu_start_offsets_us),
                "kernel_category_rules_version": 1,
                "kernel_categories": kernel_categories,
                "top_kernels": top_kernels,
            }

        return {
            "schema_version": 2,
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
