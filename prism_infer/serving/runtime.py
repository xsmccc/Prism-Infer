"""单 GPU 引擎所有者线程与异步网络请求之间的并发桥接。"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from prism_infer.engine.contracts import StepResult
from prism_infer.engine.request import RequestState
from prism_infer.sampling_params import SamplingParams
from prism_infer.serving.protocol import (
    EventKind,
    GenerationRequest,
    Modality,
    ServingEvent,
)


class ServingOverloadedError(RuntimeError):
    """入口队列已满，调用方应向客户端返回明确的过载响应。"""


class ServingUnavailableError(RuntimeError):
    """服务尚未启动、已经停止或引擎线程已经失败。"""


class DuplicateRequestError(ValueError):
    """客户端请求标识与当前尚未终止的请求重复。"""


class _Tokenizer(Protocol):
    """运行时使用的最小 tokenizer 解码接口。"""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class ServingEngine(Protocol):
    """Serving 层依赖的最小 Prism-Infer 引擎接口。"""

    tokenizer: _Tokenizer

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int: ...

    def add_vl_request(
        self,
        prompt: str,
        image: Any,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int: ...

    def add_images_request(
        self,
        prompt: str,
        images: Any,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int: ...

    def add_video_request(
        self,
        prompt: str,
        video: Any,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int: ...

    def cancel_request(self, request_id: int) -> bool: ...

    def exit(self) -> None: ...

    def is_finished(self) -> bool: ...

    def request_state(self, request_id: int) -> RequestState | None: ...

    def step_result(self) -> StepResult: ...


EngineFactory = Callable[[], ServingEngine]


class RequestHandle:
    """一个请求在 ASGI event loop 中消费事件的句柄。"""

    def __init__(
        self,
        request_id: str,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.request_id = request_id
        self._event_loop = event_loop
        self._events: asyncio.Queue[ServingEvent] = asyncio.Queue()

    async def next_event(self) -> ServingEvent:
        """等待下一条 token、完成或错误事件。"""

        return await self._events.get()

    def publish(self, event: ServingEvent) -> None:
        """从引擎线程安全地投递一条事件。"""

        try:
            self._event_loop.call_soon_threadsafe(self._events.put_nowait, event)
        except RuntimeError:
            # 客户端 event loop 已关闭时，ASGI 层会同时提交取消；这里不能让
            # 网络连接的生命周期反向击穿唯一的 GPU 引擎所有者线程。
            return


@dataclass(frozen=True, slots=True)
class _Submission:
    request: GenerationRequest
    handle: RequestHandle


@dataclass(slots=True)
class _ActiveRequest:
    request: GenerationRequest
    handle: RequestHandle
    engine_request_id: int
    token_ids: list[int] = field(default_factory=list)


class ServingRuntime:
    """以一个专用线程独占并持续驱动一个 Prism-Infer 引擎。

    网络协程只负责将请求放入有界入口队列并异步消费事件。模型初始化、请求
    admission、每次 schedule/execute/commit、tokenizer 解码和资源释放都在
    同一个所有者线程完成，避免并发协程直接触碰 CUDA 引擎状态。

    Args:
        engine_factory: 在所有者线程内构造引擎的无参工厂。
        ingress_capacity: 尚未交给引擎 admission 的最大网络请求数。
        idle_poll_seconds: 引擎空闲时检查取消和关闭信号的周期。
    """

    def __init__(
        self,
        engine_factory: EngineFactory,
        *,
        ingress_capacity: int = 64,
        idle_poll_seconds: float = 0.01,
    ) -> None:
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")
        if (
            isinstance(ingress_capacity, bool)
            or not isinstance(ingress_capacity, int)
            or ingress_capacity <= 0
        ):
            raise ValueError("ingress_capacity must be a positive integer")
        if (
            isinstance(idle_poll_seconds, bool)
            or not isinstance(idle_poll_seconds, int | float)
            or idle_poll_seconds <= 0
        ):
            raise ValueError("idle_poll_seconds must be positive")

        self._engine_factory = engine_factory
        self._ingress: queue.Queue[_Submission] = queue.Queue(
            maxsize=ingress_capacity,
        )
        self._cancellations: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._idle_poll_seconds = float(idle_poll_seconds)
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._known_request_ids: set[str] = set()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._engine: ServingEngine | None = None
        self._active_by_external: dict[str, _ActiveRequest] = {}
        self._external_by_engine: dict[int, str] = {}
        self._cancelled_before_admission: set[str] = set()

    @property
    def is_healthy(self) -> bool:
        """返回引擎是否已经启动且所有者线程仍正常运行。"""

        thread = self._thread
        return (
            self._ready.is_set()
            and self._failure is None
            and thread is not None
            and thread.is_alive()
            and not self._stop_requested.is_set()
        )

    @property
    def failure(self) -> BaseException | None:
        """返回导致所有者线程退出的异常；健康运行时为 ``None``。"""

        return self._failure

    def start(self) -> None:
        """启动所有者线程，并等待模型初始化完成。

        Raises:
            ServingUnavailableError: 重复启动或引擎初始化失败。
        """

        with self._state_lock:
            if self._thread is not None:
                raise ServingUnavailableError("serving runtime has already been started")
            self._thread = threading.Thread(
                target=self._thread_main,
                name="prism-engine-owner",
                daemon=False,
            )
            self._thread.start()
        self._ready.wait()
        if self._failure is not None:
            raise ServingUnavailableError("serving engine failed to start") from self._failure

    def stop(self) -> None:
        """请求停止并等待所有请求取消、KV 释放和引擎退出。"""

        thread = self._thread
        if thread is None:
            return
        self._stop_requested.set()
        thread.join()

    def submit(
        self,
        request: GenerationRequest,
        event_loop: asyncio.AbstractEventLoop,
    ) -> RequestHandle:
        """非阻塞提交一条网络请求。

        Args:
            request: 已验证并完成媒体解码的请求。
            event_loop: 接收请求事件的 ASGI event loop。

        Returns:
            可异步消费事件的请求句柄。

        Raises:
            ServingOverloadedError: 入口队列已满。
            ServingUnavailableError: 运行时不可用。
            DuplicateRequestError: 请求标识当前正在使用。
        """

        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if not isinstance(event_loop, asyncio.AbstractEventLoop):
            raise TypeError("event_loop must be an asyncio event loop")
        if not self.is_healthy:
            raise ServingUnavailableError("serving runtime is not healthy")

        handle = RequestHandle(request.request_id, event_loop)
        with self._state_lock:
            if request.request_id in self._known_request_ids:
                raise DuplicateRequestError(f"duplicate active request_id: {request.request_id!r}")
            self._known_request_ids.add(request.request_id)
            try:
                self._ingress.put_nowait(_Submission(request=request, handle=handle))
            except queue.Full as exc:
                self._known_request_ids.remove(request.request_id)
                raise ServingOverloadedError("serving ingress queue is full") from exc
        return handle

    def cancel(self, request_id: str) -> None:
        """异步请求取消；真正的 KV 释放由引擎所有者线程执行。"""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._cancellations.put(request_id)

    def _forget_request_id(self, request_id: str) -> None:
        with self._state_lock:
            self._known_request_ids.discard(request_id)

    def _publish_error(
        self,
        submission: _Submission,
        *,
        message: str,
    ) -> None:
        submission.handle.publish(
            ServingEvent(
                request_id=submission.request.request_id,
                kind=EventKind.ERROR,
                error=message,
            )
        )
        self._forget_request_id(submission.request.request_id)

    def _submit_to_engine(
        self,
        engine: ServingEngine,
        request: GenerationRequest,
    ) -> int:
        common = {"raise_on_reject": False}
        if request.modality is Modality.TEXT:
            return engine.add_request(
                request.prompt,
                request.sampling_params,
                **common,
            )
        if request.modality is Modality.IMAGE:
            if isinstance(request.media, tuple):
                if len(request.media) == 1:
                    return engine.add_vl_request(
                        request.prompt,
                        request.media[0],
                        request.sampling_params,
                        **common,
                    )
                return engine.add_images_request(
                    request.prompt,
                    request.media,
                    request.sampling_params,
                    **common,
                )
            return engine.add_vl_request(
                request.prompt,
                request.media,
                request.sampling_params,
                **common,
            )
        return engine.add_video_request(
            request.prompt,
            request.media,
            request.sampling_params,
            **common,
        )

    def _admit(
        self,
        engine: ServingEngine,
        submission: _Submission,
    ) -> None:
        request = submission.request
        if request.request_id in self._cancelled_before_admission:
            self._cancelled_before_admission.remove(request.request_id)
            submission.handle.publish(
                ServingEvent(
                    request_id=request.request_id,
                    kind=EventKind.DONE,
                    finish_reason="cancelled",
                )
            )
            self._forget_request_id(request.request_id)
            return
        try:
            engine_request_id = self._submit_to_engine(engine, request)
        except (OSError, TypeError, ValueError) as exc:
            self._publish_error(
                submission,
                message=f"{type(exc).__name__}: {exc}",
            )
            return

        state = engine.request_state(engine_request_id)
        submission.handle.publish(
            ServingEvent(
                request_id=request.request_id,
                kind=EventKind.ACCEPTED,
                engine_request_id=engine_request_id,
            )
        )
        if state is RequestState.REJECTED:
            submission.handle.publish(
                ServingEvent(
                    request_id=request.request_id,
                    kind=EventKind.DONE,
                    engine_request_id=engine_request_id,
                    finish_reason="rejected",
                )
            )
            self._forget_request_id(request.request_id)
            return

        active = _ActiveRequest(
            request=request,
            handle=submission.handle,
            engine_request_id=engine_request_id,
        )
        self._active_by_external[request.request_id] = active
        self._external_by_engine[engine_request_id] = request.request_id

    def _drain_ingress(self, engine: ServingEngine) -> bool:
        admitted_any = False
        while True:
            try:
                submission = self._ingress.get_nowait()
            except queue.Empty:
                return admitted_any
            self._admit(engine, submission)
            admitted_any = True

    def _finish_active(
        self,
        engine: ServingEngine,
        active: _ActiveRequest,
        *,
        token_ids: tuple[int, ...],
        finish_reason: str,
    ) -> None:
        text = engine.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        active.handle.publish(
            ServingEvent(
                request_id=active.request.request_id,
                kind=EventKind.DONE,
                engine_request_id=active.engine_request_id,
                token_ids=token_ids,
                text=text,
                finish_reason=finish_reason,
            )
        )
        self._active_by_external.pop(active.request.request_id, None)
        self._external_by_engine.pop(active.engine_request_id, None)
        self._forget_request_id(active.request.request_id)

    def _apply_cancellations(self, engine: ServingEngine) -> None:
        while True:
            try:
                request_id = self._cancellations.get_nowait()
            except queue.Empty:
                return
            active = self._active_by_external.get(request_id)
            if active is None:
                with self._state_lock:
                    is_pending = request_id in self._known_request_ids
                if is_pending:
                    self._cancelled_before_admission.add(request_id)
                continue
            if not engine.cancel_request(active.engine_request_id):
                continue
            self._finish_active(
                engine,
                active,
                token_ids=tuple(active.token_ids),
                finish_reason="cancelled",
            )

    def _execute_step(self, engine: ServingEngine) -> None:
        step = engine.step_result()
        for engine_request_id, token_id in zip(
            step.plan.sequence_ids,
            step.execution.token_ids,
            strict=True,
        ):
            if token_id is None:
                continue
            external_id = self._external_by_engine.get(engine_request_id)
            if external_id is None:
                continue
            active = self._active_by_external[external_id]
            active.token_ids.append(token_id)
            token_text = engine.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            active.handle.publish(
                ServingEvent(
                    request_id=external_id,
                    kind=EventKind.TOKEN,
                    engine_request_id=engine_request_id,
                    token_id=token_id,
                    token_text=token_text,
                )
            )

        for output in step.outputs:
            external_id = self._external_by_engine.get(output.request_id)
            if external_id is None:
                continue
            active = self._active_by_external[external_id]
            self._finish_active(
                engine,
                active,
                token_ids=output.token_ids,
                finish_reason=output.finish_reason,
            )

    def _wait_for_submission(self, engine: ServingEngine) -> None:
        try:
            submission = self._ingress.get(timeout=self._idle_poll_seconds)
        except queue.Empty:
            return
        self._admit(engine, submission)

    def _cancel_all_active(self, engine: ServingEngine) -> None:
        for active in tuple(self._active_by_external.values()):
            engine.cancel_request(active.engine_request_id)
            self._finish_active(
                engine,
                active,
                token_ids=tuple(active.token_ids),
                finish_reason="shutdown",
            )
        while True:
            try:
                submission = self._ingress.get_nowait()
            except queue.Empty:
                return
            submission.handle.publish(
                ServingEvent(
                    request_id=submission.request.request_id,
                    kind=EventKind.DONE,
                    finish_reason="shutdown",
                )
            )
            self._forget_request_id(submission.request.request_id)

    def _fail_all(self, failure: BaseException) -> None:
        message = f"serving runtime failed: {type(failure).__name__}: {failure}"
        for active in tuple(self._active_by_external.values()):
            active.handle.publish(
                ServingEvent(
                    request_id=active.request.request_id,
                    kind=EventKind.ERROR,
                    engine_request_id=active.engine_request_id,
                    error=message,
                )
            )
            self._forget_request_id(active.request.request_id)
        self._active_by_external.clear()
        self._external_by_engine.clear()
        while True:
            try:
                submission = self._ingress.get_nowait()
            except queue.Empty:
                return
            self._publish_error(submission, message=message)

    def _run_engine(self, engine: ServingEngine) -> None:
        while not self._stop_requested.is_set():
            self._apply_cancellations(engine)
            admitted = self._drain_ingress(engine)
            self._apply_cancellations(engine)
            if not engine.is_finished():
                self._execute_step(engine)
            elif not admitted:
                self._wait_for_submission(engine)
        self._cancel_all_active(engine)

    def _thread_main(self) -> None:
        engine: ServingEngine | None = None
        try:
            engine = self._engine_factory()
            self._engine = engine
            self._ready.set()
            self._run_engine(engine)
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
            self._fail_all(exc)
        finally:
            if engine is not None:
                try:
                    engine.exit()
                except BaseException as exc:
                    if self._failure is None:
                        self._failure = exc
            self._engine = None
            self._ready.set()
