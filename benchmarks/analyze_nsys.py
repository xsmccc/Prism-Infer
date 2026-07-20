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
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


_NANOSECONDS_PER_MILLISECOND = 1_000_000.0
_NANOSECONDS_PER_MICROSECOND = 1_000.0
_TOP_KERNEL_LIMIT = 20
_KERNEL_CATEGORY_RULES_VERSION = 1
_KERNEL_CATEGORY_RULES = (
    ("linear_gemv", ("gemv", "gemm")),
    ("paged_decode_attention", ("_paged_decode_attention_kernel",)),
    ("kv_store", ("_store_kvcache",)),
    ("reduction", ("reduce_kernel",)),
    ("copy_cast", ("copy_kernel", "bfloat16_copy")),
    ("layout_index", ("catarray", "indexselect", "fillfunctor")),
    ("trigonometric", ("sin_kernel", "cos_kernel")),
    ("elementwise", ("elementwise_kernel", "silu_kernel")),
)
_TARGET_TOTAL_FIELDS = {
    "kernel_count": "kernel_count_total",
    "kernel_time_ms": "kernel_time_ms_total",
    "runtime_api_count": "runtime_api_count_total",
    "memcpy_async_count": "memcpy_async_count_total",
    "stream_synchronize_count": "stream_synchronize_count_total",
    "graph_execution_ms": "graph_execution_ms_total",
    "memcpy_time_ms": "memcpy_time_ms_total",
    "memcpy_bytes": "memcpy_bytes_total",
    "memset_time_ms": "memset_time_ms_total",
    "memset_bytes": "memset_bytes_total",
}
_TARGET_SUMMARY_FIELDS = {
    "kernel_count": "kernels_per_range",
    "kernel_time_ms": "kernel_time_ms_per_range",
    "runtime_api_count": "runtime_apis_per_range",
    "memcpy_async_count": "memcpy_async_per_range",
    "stream_synchronize_count": "stream_synchronize_per_range",
    "graph_execution_ms": "graph_execution_ms_per_range",
    "cpu_range_ms": "cpu_range_ms_per_range",
    "memcpy_time_ms": "memcpy_time_ms_per_range",
    "memcpy_bytes": "memcpy_bytes_per_range",
    "memset_time_ms": "memset_time_ms_per_range",
    "memset_bytes": "memset_bytes_per_range",
    "gpu_busy_ms": "gpu_busy_ms_per_range",
    "gpu_span_ms": "gpu_span_ms_per_range",
    "cpu_gpu_busy_overlap_ms": "cpu_gpu_busy_overlap_ms_per_range",
    "cpu_gpu_busy_overlap_fraction": "cpu_gpu_busy_overlap_fraction_per_range",
    "gpu_tail_after_cpu_ms": "gpu_tail_after_cpu_ms_per_range",
    "gpu_start_offset_us": "gpu_start_offset_us_per_range",
}


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


class _ActivityIndex:
    """一次加载 CUPTI activity，并按时间与 correlationId 在内存关联。"""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.runtime = [
            (int(start), int(end), correlation_id, str(name))
            for start, end, correlation_id, name in connection.execute(
                """
                SELECT r.start, r.end, r.correlationId, s.value
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
                JOIN StringIds AS s ON r.nameId = s.id
                ORDER BY r.start
                """
            )
        ]
        self.runtime_starts = [row[0] for row in self.runtime]

        kernel_columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_columns = [
            column
            for column in ("demangledName", "shortName", "mangledName")
            if column in kernel_columns
        ]
        joins = []
        names = []
        for index, column in enumerate(name_columns):
            alias = f"kernel_name_{index}"
            joins.append(f"LEFT JOIN StringIds AS {alias} ON k.{column} = {alias}.id")
            names.append(f"{alias}.value")
        name_expression = f"coalesce({', '.join(names)}, '<unknown>')" if names else "'<unknown>'"
        self.kernels = [
            (int(start), int(end), correlation_id, str(name))
            for start, end, correlation_id, name in connection.execute(
                f"""
                SELECT k.start, k.end, k.correlationId, {name_expression}
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                {" ".join(joins)}
                ORDER BY k.start
                """
            )
        ]
        self.kernel_starts = [row[0] for row in self.kernels]
        self.kernels_by_correlation = self._group_by_correlation(self.kernels)

        self.memory_by_correlation = {
            table: self._load_memory_rows(connection, table)
            for table in (
                "CUPTI_ACTIVITY_KIND_MEMCPY",
                "CUPTI_ACTIVITY_KIND_MEMSET",
            )
        }
        self.graphs = self._load_graph_rows(connection)
        self.graph_starts = [row[0] for row in self.graphs]
        self.graphs_by_correlation = self._group_by_correlation(self.graphs)

    @staticmethod
    def _group_by_correlation(
        rows: Sequence[tuple[int, int, int | None, Any]],
    ) -> dict[int, list[tuple[int, int, int | None, Any]]]:
        grouped: dict[int, list[tuple[int, int, int | None, Any]]] = defaultdict(list)
        for row in rows:
            correlation_id = row[2]
            if correlation_id is not None:
                grouped[int(correlation_id)].append(row)
        return dict(grouped)

    @staticmethod
    def _load_memory_rows(
        connection: sqlite3.Connection,
        table: str,
    ) -> dict[int, list[tuple[int, int, int | None, int]]]:
        columns = _table_columns(connection, table)
        if not {"start", "end", "correlationId"} <= columns:
            return {}
        bytes_expression = "bytes" if "bytes" in columns else "0"
        rows = [
            (int(start), int(end), correlation_id, int(num_bytes))
            for start, end, correlation_id, num_bytes in connection.execute(
                f"""
                SELECT start, end, correlationId, {bytes_expression}
                FROM {table}
                """
            )
        ]
        return _ActivityIndex._group_by_correlation(rows)

    @staticmethod
    def _load_graph_rows(
        connection: sqlite3.Connection,
    ) -> list[tuple[int, int, int | None, None]]:
        columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        if not {"start", "end"} <= columns:
            return []
        correlation_expression = "correlationId" if "correlationId" in columns else "NULL"
        return [
            (int(start), int(end), correlation_id, None)
            for start, end, correlation_id in connection.execute(
                f"""
                SELECT start, end, {correlation_expression}
                FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                ORDER BY start
                """
            )
        ]

    @staticmethod
    def _within_range(
        rows: Sequence[tuple[int, int, Any, Any]],
        starts: Sequence[int],
        start: int,
        end: int,
    ) -> list[tuple[int, int, Any, Any]]:
        first = bisect_left(starts, start)
        last = bisect_right(starts, end)
        return [row for row in rows[first:last] if row[1] <= end]

    def runtime_within(self, start: int, end: int) -> list[tuple[int, int, int | None, str]]:
        return self._within_range(self.runtime, self.runtime_starts, start, end)

    def kernels_within(self, start: int, end: int) -> list[tuple[int, int, int | None, str]]:
        return self._within_range(self.kernels, self.kernel_starts, start, end)

    def graphs_within(
        self,
        start: int,
        end: int,
    ) -> list[tuple[int, int, int | None, None]]:
        return self._within_range(self.graphs, self.graph_starts, start, end)

    def kernels_for_correlations(
        self,
        correlation_ids: Sequence[int],
    ) -> list[tuple[int, int, str]]:
        return [
            (row[0], row[1], str(row[3]))
            for correlation_id in correlation_ids
            for row in self.kernels_by_correlation.get(correlation_id, ())
        ]

    def memory_for_correlations(
        self,
        table: str,
        correlation_ids: Sequence[int],
    ) -> list[tuple[int, int, int]]:
        grouped = self.memory_by_correlation[table]
        return [
            (row[0], row[1], int(row[3]))
            for correlation_id in correlation_ids
            for row in grouped.get(correlation_id, ())
        ]

    def graphs_for_correlations(
        self,
        correlation_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        return [
            (row[0], row[1])
            for correlation_id in correlation_ids
            for row in self.graphs_by_correlation.get(correlation_id, ())
        ]


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
    for category, fragments in _KERNEL_CATEGORY_RULES:
        if any(fragment in lowered for fragment in fragments):
            return category
    return "other"


def _runtime_rows(
    activities: _ActivityIndex,
    start: int,
    end: int,
) -> list[tuple[int | None, str]]:
    """读取完全落在一个 CPU NVTX range 内的 CUDA runtime 调用。"""

    return [(row[2], row[3]) for row in activities.runtime_within(start, end)]


def _kernel_rows_for_correlations(
    activities: _ActivityIndex,
    correlation_ids: Sequence[int],
) -> list[tuple[int, int, str]]:
    """按 runtime correlation 解析 kernel interval 与可读名称。"""

    return activities.kernels_for_correlations(correlation_ids)


def _memory_rows_for_correlations(
    activities: _ActivityIndex,
    table: str,
    correlation_ids: Sequence[int],
) -> list[tuple[int, int, int]]:
    """读取 memcpy/memset interval 与 bytes；旧 trace 缺表时返回空。"""

    return activities.memory_for_correlations(table, correlation_ids)


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
    activities: _ActivityIndex,
    start: int,
    end: int,
) -> dict[str, int]:
    """统计一个时间区间内的 CUDA runtime/driver API 次数。"""

    counts: dict[str, int] = {}
    for _, _, _, name in activities.runtime_within(start, end):
        counts[name] = counts.get(name, 0) + 1
    return counts


def _matching_count(counts: dict[str, int], fragment: str) -> int:
    """汇总名称包含指定片段的 API 次数。"""

    return sum(count for name, count in counts.items() if fragment in name)


def _step_metrics(
    activities: _ActivityIndex,
    start: int,
    end: int,
) -> dict[str, float | int]:
    """提取一个 engine NVTX range 的 kernel 与 launch 指标。"""

    kernel_rows = activities.kernels_within(start, end)
    kernels = [(row[0], row[1], row[2]) for row in kernel_rows]
    launches = [
        (row[0], row[1], row[2])
        for row in activities.runtime_within(start, end)
        if "LaunchKernel" in row[3]
    ]
    graph_traces = [(row[0], row[1]) for row in activities.graphs_within(start, end)]
    runtime_counts = _runtime_counts(activities, start, end)
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
        "cpu_range_ms": (end - start) / 1e6,
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


def _phase_metrics(
    activities: _ActivityIndex,
    engine_ranges: Sequence[tuple[int, int]],
    prefill_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, int | float]]]]:
    """按 engine range 顺序生成 prefill/decode raw 与分位数。"""

    raw_steps = [
        {
            "phase": "prefill" if index < prefill_steps else "decode",
            **_step_metrics(activities, start, end),
        }
        for index, (start, end) in enumerate(engine_ranges)
    ]
    phase_summary = {}
    for phase in ("prefill", "decode"):
        matching = [step for step in raw_steps if step["phase"] == phase]
        if matching:
            phase_summary[phase] = {
                key: _summary([float(step[key]) for step in matching])
                for key in raw_steps[0]
                if key != "phase"
            }
    return raw_steps, phase_summary


def _gpu_timeline_observation(
    gpu_intervals: Sequence[tuple[int, int]],
    *,
    cpu_start: int,
    cpu_end: int,
) -> dict[str, float]:
    """计算一个 CPU range 直接关联 GPU activity 的 span、overlap 与 tail。"""

    busy_ms = _interval_union_ns(gpu_intervals) / _NANOSECONDS_PER_MILLISECOND
    if not gpu_intervals:
        return {
            "gpu_busy_ms": busy_ms,
            "gpu_span_ms": 0.0,
            "cpu_gpu_busy_overlap_ms": 0.0,
            "cpu_gpu_busy_overlap_fraction": 0.0,
            "gpu_tail_after_cpu_ms": 0.0,
            "gpu_start_offset_us": 0.0,
        }
    gpu_start = min(interval[0] for interval in gpu_intervals)
    gpu_end = max(interval[1] for interval in gpu_intervals)
    cpu_range_ns = cpu_end - cpu_start
    overlap_ns = _clipped_interval_union_ns(gpu_intervals, cpu_start, cpu_end)
    return {
        "gpu_busy_ms": busy_ms,
        "gpu_span_ms": (gpu_end - gpu_start) / _NANOSECONDS_PER_MILLISECOND,
        "cpu_gpu_busy_overlap_ms": overlap_ns / _NANOSECONDS_PER_MILLISECOND,
        "cpu_gpu_busy_overlap_fraction": (overlap_ns / cpu_range_ns if cpu_range_ns else 0.0),
        "gpu_tail_after_cpu_ms": (max(0, gpu_end - cpu_end) / _NANOSECONDS_PER_MILLISECOND),
        "gpu_start_offset_us": ((gpu_start - cpu_start) / _NANOSECONDS_PER_MICROSECOND),
    }


def _target_observation(
    activities: _ActivityIndex,
    start: int,
    end: int,
) -> dict[str, Any]:
    """提取一个 target NVTX range 的 direct activity，不重复计算嵌套 CPU 时间。"""

    runtime_rows = _runtime_rows(activities, start, end)
    correlation_ids = sorted(
        {correlation_id for correlation_id, _ in runtime_rows if correlation_id is not None}
    )
    kernel_rows = _kernel_rows_for_correlations(activities, correlation_ids)
    kernels = [(kernel_start, kernel_end) for kernel_start, kernel_end, _ in kernel_rows]
    memcpy_rows = _memory_rows_for_correlations(
        activities,
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        correlation_ids,
    )
    memset_rows = _memory_rows_for_correlations(
        activities,
        "CUPTI_ACTIVITY_KIND_MEMSET",
        correlation_ids,
    )
    graph_traces = activities.graphs_for_correlations(correlation_ids)
    category_counts: dict[str, float] = {}
    category_times: dict[str, float] = {}
    for kernel_start, kernel_end, kernel_name in kernel_rows:
        duration_ms = (kernel_end - kernel_start) / _NANOSECONDS_PER_MILLISECOND
        category = _kernel_category(kernel_name)
        category_counts[category] = category_counts.get(category, 0.0) + 1
        category_times[category] = category_times.get(category, 0.0) + duration_ms
    gpu_intervals = [
        *kernels,
        *[(row[0], row[1]) for row in memcpy_rows],
        *[(row[0], row[1]) for row in memset_rows],
    ]
    return {
        "kernel_count": float(len(kernels)),
        "kernel_time_ms": sum(item[1] - item[0] for item in kernels) / _NANOSECONDS_PER_MILLISECOND,
        "runtime_api_count": float(len(runtime_rows)),
        "memcpy_async_count": float(
            sum("MemcpyAsync" in runtime_name for _, runtime_name in runtime_rows)
        ),
        "stream_synchronize_count": float(
            sum("StreamSynchronize" in runtime_name for _, runtime_name in runtime_rows)
        ),
        "graph_execution_ms": sum(item[1] - item[0] for item in graph_traces)
        / _NANOSECONDS_PER_MILLISECOND,
        "cpu_range_ms": (end - start) / _NANOSECONDS_PER_MILLISECOND,
        "memcpy_time_ms": sum(item[1] - item[0] for item in memcpy_rows)
        / _NANOSECONDS_PER_MILLISECOND,
        "memcpy_bytes": float(sum(row[2] for row in memcpy_rows)),
        "memset_time_ms": sum(item[1] - item[0] for item in memset_rows)
        / _NANOSECONDS_PER_MILLISECOND,
        "memset_bytes": float(sum(row[2] for row in memset_rows)),
        "category_counts": category_counts,
        "category_times": category_times,
        "kernel_rows": kernel_rows,
        **_gpu_timeline_observation(
            gpu_intervals,
            cpu_start=start,
            cpu_end=end,
        ),
    }


def _kernel_breakdown(
    observations: Sequence[dict[str, Any]],
    *,
    total_kernel_time_ms: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """汇总 category 与 top kernel，同时保持历史逐 range 累加顺序。"""

    category_counts_by_range = [item["category_counts"] for item in observations]
    category_times_by_range = [item["category_times"] for item in observations]
    all_categories = sorted(
        {category for per_range in category_times_by_range for category in per_range}
    )
    categories = {}
    for category in all_categories:
        category_count_total = sum(values.get(category, 0.0) for values in category_counts_by_range)
        category_time_total = sum(values.get(category, 0.0) for values in category_times_by_range)
        categories[category] = {
            "kernel_count_total": int(category_count_total),
            "kernel_time_ms_total": category_time_total,
            "kernel_time_fraction": (
                min(1.0, category_time_total / total_kernel_time_ms)
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
    kernel_name_totals: dict[str, list[float]] = {}
    for observation in observations:
        for kernel_start, kernel_end, kernel_name in observation["kernel_rows"]:
            duration_ms = (kernel_end - kernel_start) / _NANOSECONDS_PER_MILLISECOND
            totals = kernel_name_totals.setdefault(kernel_name, [0.0, 0.0])
            totals[0] += 1
            totals[1] += duration_ms
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
        )[:_TOP_KERNEL_LIMIT]
    ]
    return categories, top_kernels


def _summarize_target(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把 target range observations 汇总为稳定的 schema-v2 字段。"""

    values_by_field = {
        field: [float(observation[field]) for observation in observations]
        for field in _TARGET_SUMMARY_FIELDS
    }
    totals = {
        output_name: (
            int(sum(values_by_field[field]))
            if field.endswith(("count", "bytes"))
            else sum(values_by_field[field])
        )
        for field, output_name in _TARGET_TOTAL_FIELDS.items()
    }
    summaries = {
        output_name: _summary(values_by_field[field])
        for field, output_name in _TARGET_SUMMARY_FIELDS.items()
    }
    total_kernel_time_ms = float(totals["kernel_time_ms_total"])
    categories, top_kernels = _kernel_breakdown(
        observations,
        total_kernel_time_ms=total_kernel_time_ms,
    )
    return {
        "range_count": len(observations),
        **totals,
        **summaries,
        "kernel_category_rules_version": _KERNEL_CATEGORY_RULES_VERSION,
        "kernel_categories": categories,
        "top_kernels": top_kernels,
    }


def _analyze_targets(
    connection: sqlite3.Connection,
    activities: _ActivityIndex,
    target_ranges: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """按名称解析全部 target ranges，并显式拒绝缺失标签。"""

    targets = {}
    for name in target_ranges:
        ranges = _nvtx_ranges(connection, name)
        if not ranges:
            raise ValueError(f"target NVTX range not found: {name}")
        targets[name] = _summarize_target(
            [_target_observation(activities, start, end) for start, end in ranges]
        )
    return targets


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
    connection = sqlite3.connect(
        sqlite_path.resolve().as_uri() + "?mode=ro",
        uri=True,
    )
    try:
        _require_tables(connection)
        activities = _ActivityIndex(connection)
        engine_ranges = _nvtx_ranges(connection, engine_range)
        if len(engine_ranges) <= prefill_steps:
            raise ValueError("capture does not contain enough engine ranges for prefill/decode")
        raw_steps, phase_summary = _phase_metrics(
            activities,
            engine_ranges,
            prefill_steps,
        )
        targets = _analyze_targets(connection, activities, target_ranges)
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write --output without printing the full JSON summary",
    )
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
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite Nsight summary: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        print(payload)


if __name__ == "__main__":
    main()
