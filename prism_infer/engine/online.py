"""Single-process online arrival loop built on the P7 engine contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns, sleep
from typing import Any, Callable, Iterable

from prism_infer.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class OnlineRequest:
    """One request arrival in a deterministic online workload."""

    request_key: str
    arrival_offset_s: float
    payload: dict[str, Any]
    sampling_params: SamplingParams
    cancel_offset_s: float | None = None

    def __post_init__(self) -> None:
        if not self.request_key:
            raise ValueError("request_key must not be empty")
        if self.arrival_offset_s < 0:
            raise ValueError("arrival_offset_s must be non-negative")
        request_type = self.payload.get("type", "text")
        if request_type not in {"text", "image", "images", "video"}:
            raise ValueError(f"unsupported online request type: {request_type!r}")
        if "prompt" not in self.payload:
            raise ValueError("online request payload requires prompt")
        if (
            self.cancel_offset_s is not None
            and self.cancel_offset_s < self.arrival_offset_s
        ):
            raise ValueError("cancel_offset_s cannot precede arrival")


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
        return (self.finished_ns - self.started_ns) / 1e9

    def to_record(self) -> dict[str, object]:
        return {
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "duration_s": self.duration_s,
            "requests": [request.to_record() for request in self.requests],
            "engine_metrics": self.engine_metrics,
            "scheduler_metrics": self.scheduler_metrics,
        }


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
            return self.engine.add_request(
                payload["prompt"], request.sampling_params, **common
            )
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

    def run(self, requests: Iterable[OnlineRequest]) -> OnlineRunResult:
        submitted = tuple(requests)
        ordered_arrivals = sorted(
            submitted,
            key=lambda request: request.arrival_offset_s,
        )
        keys = [request.request_key for request in submitted]
        if len(set(keys)) != len(keys):
            raise ValueError("online request keys must be unique")
        if not submitted:
            raise ValueError("online serving session requires requests")
        if not self.engine.is_finished():
            raise RuntimeError("online session requires an idle engine")

        pending = deque(ordered_arrivals)
        cancellations = deque(
            sorted(
                (
                    request.cancel_offset_s,
                    request.request_key,
                )
                for request in submitted
                if request.cancel_offset_s is not None
            )
        )
        internal_ids: dict[str, int] = {}
        outputs: dict[int, tuple[int, ...]] = {}
        started_ns = self.clock_ns()

        while pending or not self.engine.is_finished():
            now_ns = self.clock_ns()
            elapsed_s = (now_ns - started_ns) / 1e9
            while pending and pending[0].arrival_offset_s <= elapsed_s:
                request = pending.popleft()
                arrival_ns = started_ns + int(
                    request.arrival_offset_s * 1e9
                )
                internal_ids[request.request_key] = self._submit(
                    request, arrival_ns
                )

            while cancellations and cancellations[0][0] <= elapsed_s:
                _, request_key = cancellations.popleft()
                request_id = internal_ids.get(request_key)
                if request_id is not None:
                    self.engine.cancel_request(request_id)

            if not self.engine.is_finished():
                step = self.engine.step_result()
                for output in step.outputs:
                    outputs[output.request_id] = output.token_ids
                continue

            if pending:
                wait_s = max(
                    0.0,
                    pending[0].arrival_offset_s
                    - (self.clock_ns() - started_ns) / 1e9,
                )
                if wait_s > 0:
                    self.sleep_fn(wait_s)

        finished_ns = self.clock_ns()
        metrics = self.engine.metrics_snapshot()
        metrics_by_id = {
            int(record["request_id"]): record
            for record in metrics.get("requests", [])
        }
        request_results: list[OnlineRequestResult] = []
        for request in submitted:
            request_id = internal_ids[request.request_key]
            state = self.engine.request_state(request_id)
            metric = metrics_by_id.get(request_id, {})
            request_results.append(
                OnlineRequestResult(
                    request_key=request.request_key,
                    request_id=request_id,
                    state=(
                        "unknown"
                        if state is None
                        else state.name.lower()
                    ),
                    token_ids=outputs.get(request_id, ()),
                    finish_reason=metric.get("finish_reason"),
                )
            )
        return OnlineRunResult(
            started_ns=started_ns,
            finished_ns=finished_ns,
            requests=tuple(request_results),
            engine_metrics=metrics,
            scheduler_metrics=self.engine.scheduler.metrics_snapshot(),
        )
