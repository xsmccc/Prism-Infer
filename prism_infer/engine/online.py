"""Single-process online arrival loop built on the P7 engine contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter_ns, sleep
from typing import Any, Callable, Iterable

from prism_infer.sampling_params import SamplingParams


NANOSECONDS_PER_SECOND = 1_000_000_000
_ONLINE_MEDIA_FIELD_BY_TYPE = {
    "image": "image",
    "images": "images",
    "video": "video",
}
_SUPPORTED_ONLINE_REQUEST_TYPES = frozenset({"text", *_ONLINE_MEDIA_FIELD_BY_TYPE})


def _non_negative_seconds(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return float(value)


def _validate_online_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("online request payload must be a dict")
    request_type = payload.get("type", "text")
    if not isinstance(request_type, str) or request_type not in _SUPPORTED_ONLINE_REQUEST_TYPES:
        raise ValueError(f"unsupported online request type: {request_type!r}")
    if "prompt" not in payload or payload["prompt"] is None:
        raise ValueError("online request payload requires prompt")
    prompt = payload["prompt"]
    if request_type == "text" and not isinstance(prompt, (str, list)):
        raise TypeError("online text prompt must be a string or token-id list")
    if request_type != "text" and not isinstance(prompt, str):
        raise TypeError(f"online {request_type} prompt must be a string")
    media_field = _ONLINE_MEDIA_FIELD_BY_TYPE.get(request_type)
    if media_field is not None and (media_field not in payload or payload[media_field] is None):
        raise ValueError(f"online {request_type} payload requires {media_field!r}")


def _normalize_cancel_offset(value: object, *, arrival_offset_s: float) -> float | None:
    if value is None:
        return None
    cancel_offset_s = _non_negative_seconds(value, name="cancel_offset_s")
    if cancel_offset_s < arrival_offset_s:
        raise ValueError("cancel_offset_s cannot precede arrival")
    return cancel_offset_s


@dataclass(frozen=True, slots=True)
class OnlineRequest:
    """One request arrival in a deterministic online workload."""

    request_key: str
    arrival_offset_s: float
    payload: dict[str, Any]
    sampling_params: SamplingParams
    cancel_offset_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, str) or not self.request_key:
            raise ValueError("request_key must not be empty")
        arrival_offset_s = _non_negative_seconds(
            self.arrival_offset_s,
            name="arrival_offset_s",
        )
        object.__setattr__(self, "arrival_offset_s", arrival_offset_s)
        _validate_online_payload(self.payload)
        if not isinstance(self.sampling_params, SamplingParams):
            raise TypeError("online request sampling_params must be SamplingParams")
        cancel_offset_s = _normalize_cancel_offset(
            self.cancel_offset_s,
            arrival_offset_s=arrival_offset_s,
        )
        object.__setattr__(self, "cancel_offset_s", cancel_offset_s)


@dataclass(frozen=True, slots=True)
class OnlineRequestResult:
    request_key: str
    request_id: int
    state: str
    token_ids: tuple[int, ...]
    finish_reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "request_key": self.request_key,
            "request_id": self.request_id,
            "state": self.state,
            "token_ids": list(self.token_ids),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True, slots=True)
class OnlineRunResult:
    started_ns: int
    finished_ns: int
    requests: tuple[OnlineRequestResult, ...]
    engine_metrics: dict[str, object]
    scheduler_metrics: dict[str, object]

    @property
    def duration_s(self) -> float:
        return (self.finished_ns - self.started_ns) / NANOSECONDS_PER_SECOND

    def to_record(self) -> dict[str, object]:
        return {
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "duration_s": self.duration_s,
            "requests": [request.to_record() for request in self.requests],
            "engine_metrics": self.engine_metrics,
            "scheduler_metrics": self.scheduler_metrics,
        }


@dataclass(slots=True)
class _OnlineRunState:
    """Mutable state for one online event-loop invocation."""

    submitted: tuple[OnlineRequest, ...]
    pending: deque[OnlineRequest]
    cancellations: deque[tuple[float, str]]
    started_ns: int
    internal_ids: dict[str, int] = field(default_factory=dict)
    outputs: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def elapsed_seconds(self, now_ns: int) -> float:
        return (now_ns - self.started_ns) / NANOSECONDS_PER_SECOND


class OnlineServingSession:
    """Drive arrivals and dynamic batches through one ``LLMEngine`` instance.

    Arrival processing and model execution intentionally share one control
    thread, matching the current engine architecture.  Arrivals that occur
    during a GPU step retain their intended arrival timestamp and enter the
    next scheduling decision when control returns.
    """

    def __init__(
        self,
        engine,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self.engine = engine
        self.clock_ns = clock_ns
        self.sleep_fn = sleep_fn

    def _submit(self, request: OnlineRequest, arrival_ns: int) -> int:
        payload = request.payload
        request_type = payload.get("type", "text")
        common = {
            "submitted_ns": arrival_ns,
            "raise_on_reject": False,
        }
        if request_type == "text":
            return self.engine.add_request(payload["prompt"], request.sampling_params, **common)
        if request_type == "image":
            return self.engine.add_vl_request(
                payload["prompt"],
                payload["image"],
                request.sampling_params,
                **common,
            )
        if request_type == "images":
            return self.engine.add_images_request(
                payload["prompt"],
                payload["images"],
                request.sampling_params,
                **common,
            )
        return self.engine.add_video_request(
            payload["prompt"],
            payload["video"],
            request.sampling_params,
            **common,
        )

    def _validate_requests(
        self,
        requests: Iterable[OnlineRequest],
    ) -> tuple[OnlineRequest, ...]:
        submitted = tuple(requests)
        if not submitted:
            raise ValueError("online serving session requires requests")
        invalid = [
            index
            for index, request in enumerate(submitted)
            if not isinstance(request, OnlineRequest)
        ]
        if invalid:
            raise TypeError(f"online session entries must be OnlineRequest: indices={invalid}")
        keys = [request.request_key for request in submitted]
        if len(set(keys)) != len(keys):
            raise ValueError("online request keys must be unique")
        if not self.engine.is_finished():
            raise RuntimeError("online session requires an idle engine")
        return submitted

    def _new_run_state(
        self,
        submitted: tuple[OnlineRequest, ...],
    ) -> _OnlineRunState:
        ordered_arrivals = sorted(
            submitted,
            key=lambda request: request.arrival_offset_s,
        )
        cancellations = sorted(
            (
                request.cancel_offset_s,
                request.request_key,
            )
            for request in submitted
            if request.cancel_offset_s is not None
        )
        return _OnlineRunState(
            submitted=submitted,
            pending=deque(ordered_arrivals),
            cancellations=deque(cancellations),
            started_ns=self.clock_ns(),
        )

    def _submit_ready_arrivals(
        self,
        state: _OnlineRunState,
    ) -> None:
        while state.pending:
            elapsed_s = state.elapsed_seconds(self.clock_ns())
            if state.pending[0].arrival_offset_s > elapsed_s:
                return
            request = state.pending.popleft()
            arrival_ns = state.started_ns + int(request.arrival_offset_s * NANOSECONDS_PER_SECOND)
            state.internal_ids[request.request_key] = self._submit(request, arrival_ns)

    def _apply_ready_cancellations(
        self,
        state: _OnlineRunState,
    ) -> None:
        while state.cancellations:
            elapsed_s = state.elapsed_seconds(self.clock_ns())
            if state.cancellations[0][0] > elapsed_s:
                return
            _, request_key = state.cancellations.popleft()
            request_id = state.internal_ids.get(request_key)
            if request_id is not None:
                self.engine.cancel_request(request_id)

    def _execute_step(self, state: _OnlineRunState) -> bool:
        if self.engine.is_finished():
            return False
        step = self.engine.step_result()
        for output in step.outputs:
            state.outputs[output.request_id] = output.token_ids
        return True

    def _wait_for_next_arrival(self, state: _OnlineRunState) -> None:
        if not state.pending:
            return
        wait_s = max(
            0.0,
            state.pending[0].arrival_offset_s - state.elapsed_seconds(self.clock_ns()),
        )
        if wait_s > 0:
            self.sleep_fn(wait_s)

    def _drive_event_loop(self, state: _OnlineRunState) -> None:
        while state.pending or not self.engine.is_finished():
            self._submit_ready_arrivals(state)
            self._apply_ready_cancellations(state)
            if self._execute_step(state):
                continue
            self._wait_for_next_arrival(state)

    def _request_results(
        self,
        state: _OnlineRunState,
        metrics: dict[str, object],
    ) -> tuple[OnlineRequestResult, ...]:
        metrics_by_id = {
            int(record["request_id"]): record for record in metrics.get("requests", [])
        }
        results: list[OnlineRequestResult] = []
        for request in state.submitted:
            request_id = state.internal_ids[request.request_key]
            request_state = self.engine.request_state(request_id)
            metric = metrics_by_id.get(request_id, {})
            results.append(
                OnlineRequestResult(
                    request_key=request.request_key,
                    request_id=request_id,
                    state=("unknown" if request_state is None else request_state.name.lower()),
                    token_ids=state.outputs.get(request_id, ()),
                    finish_reason=metric.get("finish_reason"),
                )
            )
        return tuple(results)

    def run(self, requests: Iterable[OnlineRequest]) -> OnlineRunResult:
        submitted = self._validate_requests(requests)
        state = self._new_run_state(submitted)
        self._drive_event_loop(state)
        finished_ns = self.clock_ns()
        metrics = self.engine.metrics_snapshot()
        return OnlineRunResult(
            started_ns=state.started_ns,
            finished_ns=finished_ns,
            requests=self._request_results(state, metrics),
            engine_metrics=metrics,
            scheduler_metrics=self.engine.scheduler.metrics_snapshot(),
        )
