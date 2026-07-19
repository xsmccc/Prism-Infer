"""P6 分层性能 profiling collector。

collector 只在显式进入 :func:`performance_profile` 时生效。CPU 时间用于观察
Python、调度和同步等待，CUDA Event 时间用于观察同一语义区域在当前 stream 上的
GPU 时间；默认关闭路径不创建 CUDA Event，也不执行同步。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator

import torch
from torch.autograd.profiler import record_function

from prism_infer.analysis.benchmark_schema import summarize_values
from prism_infer.observability.performance import install_performance_provider


PERFORMANCE_PROFILE_SCHEMA_VERSION = 1


def _summarize_profile_regions(
    regions: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """按 region name 聚合 CPU/CUDA inclusive 时间。"""

    summary: dict[str, dict[str, Any]] = {}
    for name in sorted({str(region["name"]) for region in regions}):
        matching = [region for region in regions if region["name"] == name]
        cpu_values = [float(region["cpu_ms"]) for region in matching]
        cuda_values = [
            float(region["cuda_ms"]) for region in matching if region["cuda_ms"] is not None
        ]
        summary[name] = {
            "calls": len(matching),
            "cpu_ms": summarize_values(cpu_values),
            "cuda_ms": summarize_values(cuda_values) if cuda_values else None,
        }
    return summary


def validate_performance_profile_record(record: Mapping[str, Any]) -> None:
    """校验一条 P6 performance profile record 及其聚合自洽性。"""

    if record.get("schema_version") != PERFORMANCE_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported performance profile schema_version")
    if record.get("record_type") != "performance_profile":
        raise ValueError("performance profile record_type is invalid")
    if not isinstance(record.get("metadata"), Mapping):
        raise ValueError("performance profile metadata must be an object")
    if not isinstance(record.get("cuda_timing"), bool):
        raise ValueError("performance profile cuda_timing must be a bool")
    steps = _validate_profile_steps(record.get("steps"))
    regions = _validate_profile_regions(record.get("regions"), steps)
    _validate_profile_summaries(record, regions)


def _validate_profile_steps(raw_steps: object) -> set[int]:
    if not isinstance(raw_steps, list):
        raise ValueError("performance profile steps must be a list")
    step_ids: set[int] = set()
    for step in raw_steps:
        if not isinstance(step, Mapping):
            raise ValueError("performance profile step must be an object")
        step_id = step.get("step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise ValueError("performance profile step_id must be a non-negative int")
        if step_id in step_ids:
            raise ValueError(f"duplicate performance profile step_id: {step_id}")
        step_ids.add(step_id)
        if step.get("status") not in ("ok", "error"):
            raise ValueError("performance profile step status must be ok or error")
    return step_ids


def _validate_profile_regions(
    raw_regions: object,
    step_ids: set[int],
) -> list[Mapping[str, Any]]:
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("performance profile regions must be a non-empty list")
    regions = []
    for region in raw_regions:
        if not isinstance(region, Mapping):
            raise ValueError("performance profile region must be an object")
        _validate_profile_region(region, step_ids)
        regions.append(region)
    return regions


def _validate_profile_region(region: Mapping[str, Any], step_ids: set[int]) -> None:
    name = region.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("performance profile region name must be non-empty")
    step_id = region.get("step_id")
    if step_id is not None and step_id not in step_ids:
        raise ValueError(f"performance profile region has unknown step_id: {step_id}")
    _validate_profile_duration(region.get("cpu_ms"), "cpu_ms", allow_none=False)
    _validate_profile_duration(region.get("cuda_ms"), "cuda_ms", allow_none=True)
    if not isinstance(region.get("metadata"), Mapping):
        raise ValueError("performance profile region metadata must be an object")
    phase = region.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError("performance profile region phase must be non-empty")


def _validate_profile_duration(value: object, name: str, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"performance profile region {name} must be non-negative")


def _validate_profile_summaries(
    record: Mapping[str, Any],
    regions: list[Mapping[str, Any]],
) -> None:
    region_names = {str(region["name"]) for region in regions}
    summary = record.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != region_names:
        raise ValueError("performance profile summary names must match regions")
    expected_summary = _summarize_profile_regions(regions)
    if summary != expected_summary:
        raise ValueError("performance profile summary does not match raw regions")

    summary_by_phase = record.get("summary_by_phase")
    phases = sorted({str(region["phase"]) for region in regions})
    if not isinstance(summary_by_phase, Mapping) or set(summary_by_phase) != set(phases):
        raise ValueError("performance profile phase summary names must match regions")
    expected_by_phase = {
        phase: _summarize_profile_regions(
            [region for region in regions if region["phase"] == phase]
        )
        for phase in phases
    }
    if summary_by_phase != expected_by_phase:
        raise ValueError("performance profile phase summary does not match raw regions")


@dataclass
class _PendingRegion:
    """一个尚未解析 CUDA Event 的语义区域。"""

    name: str
    step_id: int | None
    cpu_ms: float
    cuda_start: torch.cuda.Event | None
    cuda_end: torch.cuda.Event | None
    metadata: dict[str, Any]


@dataclass
class _StepRecord:
    """一次 engine step 的调度形态。"""

    step_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceProfileSession:
    """收集一次 profiling 会话中的语义区域与 step metadata。"""

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        cuda_timing: bool = True,
    ) -> None:
        self.metadata = dict(metadata or {})
        self.cuda_timing = bool(cuda_timing and torch.cuda.is_available())
        self._regions: list[_PendingRegion] = []
        self._steps: list[_StepRecord] = []
        self._active_step: _StepRecord | None = None
        self._next_step_id = 0
        self._closed = False

    @property
    def active_step_id(self) -> int | None:
        """返回当前 step ID；预处理区域没有 step，因此返回 ``None``。"""

        return None if self._active_step is None else self._active_step.step_id

    def begin_step(self) -> int:
        """开始一个 engine step，并返回单调递增的 step ID。"""

        if self._active_step is not None:
            raise RuntimeError(
                f"performance profile step {self._active_step.step_id} is still active"
            )
        step = _StepRecord(step_id=self._next_step_id)
        self._next_step_id += 1
        self._steps.append(step)
        self._active_step = step
        return step.step_id

    def annotate_step(self, **metadata: Any) -> None:
        """为当前 step 增加可 JSON 序列化的调度 metadata。"""

        if self._active_step is None:
            raise RuntimeError("cannot annotate performance profile without an active step")
        self._active_step.metadata.update(metadata)

    def end_step(self, *, status: str = "ok") -> None:
        """结束当前 step，并记录是否正常完成。"""

        if self._active_step is None:
            raise RuntimeError("cannot end performance profile without an active step")
        self._active_step.metadata["status"] = status
        self._active_step = None

    def add_region(
        self,
        *,
        name: str,
        cpu_ms: float,
        cuda_start: torch.cuda.Event | None,
        cuda_end: torch.cuda.Event | None,
        metadata: dict[str, Any],
    ) -> None:
        """保存一个已结束、但 CUDA 时间可能尚未解析的区域。"""

        if self._closed:
            raise RuntimeError("cannot add a region to a closed performance profile")
        self._regions.append(
            _PendingRegion(
                name=name,
                step_id=self.active_step_id,
                cpu_ms=cpu_ms,
                cuda_start=cuda_start,
                cuda_end=cuda_end,
                metadata=metadata,
            )
        )

    def close(self) -> None:
        """关闭会话；存在未结束 step 时显式失败。"""

        if self._active_step is not None:
            raise RuntimeError(
                f"performance profile step {self._active_step.step_id} was not ended"
            )
        self._closed = True

    def to_record(self) -> dict[str, Any]:
        """解析 CUDA Event，并生成带 summary 的 JSON-compatible record。"""

        if not self._closed:
            raise RuntimeError("performance profile must be closed before serialization")
        if self.cuda_timing and self._regions:
            torch.cuda.synchronize()

        step_phases = {
            step.step_id: str(step.metadata.get("phase", "unknown")) for step in self._steps
        }
        regions: list[dict[str, Any]] = []
        for pending in self._regions:
            cuda_ms = None
            if pending.cuda_start is not None and pending.cuda_end is not None:
                cuda_ms = float(pending.cuda_start.elapsed_time(pending.cuda_end))
            regions.append(
                {
                    "name": pending.name,
                    "step_id": pending.step_id,
                    "phase": (
                        "preprocess" if pending.step_id is None else step_phases[pending.step_id]
                    ),
                    "cpu_ms": pending.cpu_ms,
                    "cuda_ms": cuda_ms,
                    "metadata": pending.metadata,
                }
            )

        summary = _summarize_profile_regions(regions)
        phases = sorted({region["phase"] for region in regions})
        summary_by_phase = {
            phase: _summarize_profile_regions(
                [region for region in regions if region["phase"] == phase]
            )
            for phase in phases
        }

        record = {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "record_type": "performance_profile",
            "metadata": self.metadata,
            "cuda_timing": self.cuda_timing,
            "steps": [{"step_id": step.step_id, **step.metadata} for step in self._steps],
            "regions": regions,
            "summary": summary,
            "summary_by_phase": summary_by_phase,
        }
        validate_performance_profile_record(record)
        return record


_ACTIVE_SESSION: ContextVar[PerformanceProfileSession | None] = ContextVar(
    "prism_infer_performance_profile",
    default=None,
)


def get_performance_profile_session() -> PerformanceProfileSession | None:
    """返回当前 profiling session；默认关闭时返回 ``None``。"""

    return _ACTIVE_SESSION.get()


def is_performance_profile_enabled() -> bool:
    """判断当前进程是否显式开启了 performance profiling。"""

    return get_performance_profile_session() is not None


@contextmanager
def performance_profile(
    *,
    metadata: dict[str, Any] | None = None,
    cuda_timing: bool = True,
) -> Iterator[PerformanceProfileSession]:
    """开启一次不可嵌套的分层 profiling 会话。"""

    if get_performance_profile_session() is not None:
        raise RuntimeError("nested performance_profile sessions are not supported")
    session = PerformanceProfileSession(
        metadata=metadata,
        cuda_timing=cuda_timing,
    )
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)
        session.close()


class _DisabledRegion(AbstractContextManager[None]):
    """默认关闭路径复用的无状态 context manager。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None


_DISABLED_REGION = _DisabledRegion()


@contextmanager
def _enabled_region(
    session: PerformanceProfileSession,
    name: str,
    *,
    cuda: bool,
    metadata: dict[str, Any],
) -> Iterator[None]:
    """记录一个开启状态的 CPU/CUDA 语义区域。"""

    cuda_start = None
    cuda_end = None
    if cuda and session.cuda_timing:
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        torch.cuda.nvtx.range_push(f"prism::{name}")
    cpu_start = perf_counter()
    try:
        with record_function(f"prism::{name}"):
            yield
    finally:
        cpu_ms = (perf_counter() - cpu_start) * 1000.0
        if cuda_end is not None:
            torch.cuda.nvtx.range_pop()
            cuda_end.record()
        session.add_region(
            name=name,
            cpu_ms=cpu_ms,
            cuda_start=cuda_start,
            cuda_end=cuda_end,
            metadata=metadata,
        )


def profile_region(
    name: str,
    *,
    cuda: bool = True,
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    """返回语义计时区域；profiling 关闭或 Dynamo 捕获时返回 no-op。

    编译 region 的性能由外层 benchmark 计时。ContextVar、CUDA Event 和 NVTX
    都是 Python side effect，若在 Dynamo 捕获期间进入 collector，会把纯 tensor
    graph 人为切碎；因此编译路径不支持嵌套 semantic collector。
    """

    if torch.compiler.is_compiling():
        # Dynamo 对标准库 nullcontext 有专门处理；自定义 context manager 会让
        # enclosing frame被跳过或产生 graph break。
        return nullcontext()
    session = get_performance_profile_session()
    if session is None:
        return _DISABLED_REGION
    if not name:
        raise ValueError("performance profile region name must not be empty")
    return _enabled_region(
        session,
        name,
        cuda=cuda,
        metadata=dict(metadata or {}),
    )


install_performance_provider(
    profile_region_provider=profile_region,
    profile_session_provider=get_performance_profile_session,
)
