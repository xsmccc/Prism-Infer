"""Focused lifecycle checks for the network-to-engine concurrency bridge."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from prism_infer.engine.request import RequestState
from prism_infer.sampling_params import SamplingParams
from prism_infer.serving.protocol import EventKind, GenerationRequest, Modality
from prism_infer.serving.runtime import ServingOverloadedError, ServingRuntime


class _FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeEngine:
    """Small deterministic engine implementing only the Serving protocol."""

    def __init__(self, *, step_delay_s: float = 0.0) -> None:
        self.tokenizer = _FakeTokenizer()
        self.step_delay_s = step_delay_s
        self.next_request_id = 0
        self.active: dict[int, tuple[int, int]] = {}
        self.states: dict[int, RequestState] = {}
        self.exited = False
        self.owner_calls: list[tuple[str, int]] = []
        self.preparation_calls: list[tuple[str, int, int]] = []
        self.prepared_submissions: list[int] = []
        self.submitted_ns: dict[int, int] = {}
        self.prepare_started = threading.Event()
        self.prepare_release = threading.Event()
        self.prepare_release.set()
        self.prepare_finished = threading.Event()
        self.id_allocated = threading.Event()
        self.first_step_started = threading.Event()
        self.first_step_release = threading.Event()
        self.first_step_release.set()
        self.wait_for_prepare_on_second_step = False
        self.decode_during_preparation = threading.Event()
        self.prepare_errors: dict[str, Exception] = {}
        self.preparation_active = False
        self.exited_while_preparing = False
        self.step_count = 0

    def _record_owner(self, method: str) -> None:
        self.owner_calls.append((method, threading.get_ident()))

    def _allocate_request_id(self) -> int:
        self._record_owner("allocate")
        request_id = self.next_request_id
        self.next_request_id += 1
        self.id_allocated.set()
        return request_id

    def _add(self, sampling_params: SamplingParams, request_id: int | None = None) -> int:
        self._record_owner("add")
        if request_id is None:
            request_id = self._allocate_request_id()
        self.active[request_id] = (0, sampling_params.max_tokens)
        self.states[request_id] = RequestState.PREFILLING
        return request_id

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
        submitted_ns: int,
    ) -> int:
        del prompt, raise_on_reject
        request_id = self._add(sampling_params)
        self.submitted_ns[request_id] = submitted_ns
        return request_id

    def _prepare_media_request(
        self,
        request_type: str,
        prompt: str,
        media: object,
        sampling_params: SamplingParams,
        *,
        request_id: int,
        image_marker: str = "<image>",
    ) -> SimpleNamespace:
        del media, image_marker
        self.preparation_calls.append((request_type, request_id, threading.get_ident()))
        self.preparation_active = True
        self.prepare_started.set()
        try:
            assert self.prepare_release.wait(timeout=2.0), "test did not release CPU preparation"
            if prompt in self.prepare_errors:
                raise self.prepare_errors[prompt]
            return SimpleNamespace(seq_id=request_id, sampling_params=sampling_params)
        finally:
            self.preparation_active = False
            self.prepare_finished.set()

    def _submit_sequence(
        self,
        seq: SimpleNamespace,
        *,
        raise_on_reject: bool,
        submitted_ns: int,
    ) -> int:
        del raise_on_reject
        self._record_owner("submit_sequence")
        self.prepared_submissions.append(seq.seq_id)
        self.submitted_ns[seq.seq_id] = submitted_ns
        return self._add(seq.sampling_params, seq.seq_id)

    def cancel_request(self, request_id: int) -> bool:
        self._record_owner("cancel")
        if request_id not in self.active:
            return False
        self.active.pop(request_id)
        self.states[request_id] = RequestState.CANCELLED
        return True

    def exit(self) -> None:
        self._record_owner("exit")
        self.exited_while_preparing = self.preparation_active
        self.exited = True

    def is_finished(self) -> bool:
        self._record_owner("is_finished")
        return not self.active

    def request_state(self, request_id: int) -> RequestState | None:
        self._record_owner("request_state")
        return self.states.get(request_id)

    def step_result(self) -> SimpleNamespace:
        self._record_owner("step")
        self.step_count += 1
        if self.step_count == 1:
            self.first_step_started.set()
            assert self.first_step_release.wait(timeout=2.0)
        if self.step_count == 2 and self.wait_for_prepare_on_second_step:
            assert self.prepare_started.wait(timeout=2.0)
        if self.prepare_started.is_set() and not self.prepare_release.is_set():
            self.decode_during_preparation.set()
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        sequence_ids = tuple(self.active)
        token_ids: list[int] = []
        outputs: list[SimpleNamespace] = []
        for request_id in sequence_ids:
            generated, max_tokens = self.active[request_id]
            token_id = 100 + generated
            token_ids.append(token_id)
            generated += 1
            if generated == max_tokens:
                self.active.pop(request_id)
                self.states[request_id] = RequestState.FINISHED
                outputs.append(
                    SimpleNamespace(
                        request_id=request_id,
                        token_ids=tuple(range(100, 100 + generated)),
                        finish_reason="length",
                    )
                )
            else:
                self.active[request_id] = (generated, max_tokens)
                self.states[request_id] = RequestState.DECODING
        return SimpleNamespace(
            plan=SimpleNamespace(sequence_ids=sequence_ids),
            execution=SimpleNamespace(token_ids=tuple(token_ids)),
            outputs=tuple(outputs),
        )


def _request(
    request_id: str,
    *,
    max_tokens: int,
    modality: Modality = Modality.TEXT,
    media: object = None,
    prompt: str = "hello",
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        modality=modality,
        media=media,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def test_runtime_publishes_each_engine_token_before_done() -> None:
    engine = _FakeEngine()
    runtime = ServingRuntime(lambda: engine)
    runtime.start()

    async def exercise() -> list:
        handle = runtime.submit(_request("stream", max_tokens=2), asyncio.get_running_loop())
        events = []
        while not events or events[-1].kind is not EventKind.DONE:
            events.append(await asyncio.wait_for(handle.next_event(), timeout=1.0))
        return events

    try:
        events = asyncio.run(exercise())
    finally:
        runtime.stop()

    assert [event.kind for event in events] == [
        EventKind.ACCEPTED,
        EventKind.TOKEN,
        EventKind.TOKEN,
        EventKind.DONE,
    ]
    assert [event.token_id for event in events if event.kind is EventKind.TOKEN] == [100, 101]
    assert events[-1].token_ids == (100, 101)
    assert events[-1].text == "100 101"
    assert engine.exited


def test_runtime_cancellation_reaches_engine_owner_and_terminates_request() -> None:
    engine = _FakeEngine(step_delay_s=0.002)
    runtime = ServingRuntime(lambda: engine)
    runtime.start()

    async def exercise() -> list:
        handle = runtime.submit(_request("cancel", max_tokens=128), asyncio.get_running_loop())
        first = await asyncio.wait_for(handle.next_event(), timeout=1.0)
        runtime.cancel(handle.request_id)
        events = [first]
        while not events or events[-1].kind is not EventKind.DONE:
            events.append(await asyncio.wait_for(handle.next_event(), timeout=1.0))
        return events

    try:
        events = asyncio.run(exercise())
    finally:
        runtime.stop()

    assert events[0].kind is EventKind.ACCEPTED
    assert events[-1].kind is EventKind.DONE
    assert events[-1].finish_reason == "cancelled"
    assert engine.states[0] is RequestState.CANCELLED
    assert engine.exited


async def _wait_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 1.0), "expected thread event was not signalled"


async def _collect_events(handle) -> list:
    events = []
    while True:
        event = await asyncio.wait_for(handle.next_event(), timeout=1.0)
        events.append(event)
        if event.kind in (EventKind.DONE, EventKind.ERROR):
            return events


class _ObservedRuntime(ServingRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_released = threading.Event()

    def _release_admission_slot(self) -> None:
        super()._release_admission_slot()
        self.slot_released.set()


def test_media_preparation_runs_on_one_cpu_worker_and_admits_on_owner() -> None:
    engine = _FakeEngine()
    engine.prepare_release.clear()
    runtime = ServingRuntime(lambda: engine)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        before = time.perf_counter_ns()
        handle = runtime.submit(
            _request("image", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        after = time.perf_counter_ns()
        await _wait_event(engine.prepare_started)
        assert engine.states == {}
        assert engine.prepared_submissions == []
        engine.prepare_release.set()
        events = await _collect_events(handle)
        assert events[0].kind is EventKind.ACCEPTED
        assert before <= engine.submitted_ns[events[0].engine_request_id] <= after
        for request_id, modality, media in (
            ("images", Modality.IMAGE, (object(), object())),
            ("video", Modality.VIDEO, (object(), object())),
        ):
            events = await _collect_events(
                runtime.submit(
                    _request(request_id, max_tokens=2, modality=modality, media=media),
                    loop,
                )
            )
            assert events[-1].finish_reason == "length"

    try:
        asyncio.run(exercise())
    finally:
        engine.prepare_release.set()
        runtime.stop()
    assert [call[0] for call in engine.preparation_calls] == ["images", "images", "video"]
    owner_threads = {thread_id for _, thread_id in engine.owner_calls}
    preparation_threads = {thread_id for _, _, thread_id in engine.preparation_calls}
    assert owner_threads == {runtime._thread.ident}
    assert len(preparation_threads) == 1
    assert owner_threads.isdisjoint(preparation_threads)
    assert runtime._unadmitted_requests == 0


def test_decode_advances_while_media_cpu_preparation_is_blocked() -> None:
    engine = _FakeEngine()
    engine.prepare_release.clear()
    engine.first_step_release.clear()
    engine.wait_for_prepare_on_second_step = True
    runtime = ServingRuntime(lambda: engine)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        text = runtime.submit(_request("decoding", max_tokens=2), loop)
        await _wait_event(engine.first_step_started)
        image = runtime.submit(
            _request("cold", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        engine.first_step_release.set()
        await _wait_event(engine.prepare_started)
        events = await _collect_events(text)
        assert events[-1].token_ids == (100, 101)
        assert engine.decode_during_preparation.is_set()
        assert not engine.prepare_release.is_set()
        assert engine.prepared_submissions == []
        engine.prepare_release.set()
        image_events = await _collect_events(image)
        assert image_events[-1].finish_reason == "length"

    try:
        asyncio.run(exercise())
    finally:
        engine.first_step_release.set()
        engine.prepare_release.set()
        runtime.stop()


def test_cancelled_running_preparation_holds_ingress_capacity_until_finished() -> None:
    engine = _FakeEngine()
    engine.prepare_release.clear()
    runtime = _ObservedRuntime(lambda: engine, ingress_capacity=1)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        handle = runtime.submit(
            _request("cancel-cpu", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        await _wait_event(engine.prepare_started)
        assert runtime._ingress.empty()
        with unittest.TestCase().assertRaises(ServingOverloadedError):
            runtime.submit(_request("overflow", max_tokens=2), loop)
        runtime.cancel("cancel-cpu")
        runtime.cancel("cancel-cpu")
        events = await _collect_events(handle)
        assert [event.kind for event in events] == [EventKind.DONE]
        assert events[-1].finish_reason == "cancelled"
        assert not runtime.slot_released.is_set()
        with unittest.TestCase().assertRaises(ServingOverloadedError):
            runtime.submit(_request("still-full", max_tokens=2), loop)
        engine.prepare_release.set()
        await _wait_event(runtime.slot_released)
        later = await _collect_events(runtime.submit(_request("later", max_tokens=2), loop))
        assert later[-1].finish_reason == "length"
        assert engine.prepared_submissions == []
        assert handle._events.empty()

    try:
        asyncio.run(exercise())
    finally:
        engine.prepare_release.set()
        runtime.stop()
    assert runtime._unadmitted_requests == 0


def test_cancelled_queued_media_never_starts_a_second_preparation() -> None:
    engine = _FakeEngine()
    engine.prepare_release.clear()
    runtime = ServingRuntime(lambda: engine, ingress_capacity=2)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        first = runtime.submit(
            _request("first", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        await _wait_event(engine.prepare_started)
        engine.id_allocated.clear()
        queued = runtime.submit(
            _request("queued", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        await _wait_event(engine.id_allocated)
        with unittest.TestCase().assertRaises(ServingOverloadedError):
            runtime.submit(_request("overflow", max_tokens=2), loop)
        runtime.cancel("queued")
        events = await _collect_events(queued)
        assert [event.kind for event in events] == [EventKind.DONE]
        assert events[-1].finish_reason == "cancelled"
        assert len(engine.preparation_calls) == 1
        engine.prepare_release.set()
        await _collect_events(first)
        assert engine.prepared_submissions == [0]
        assert 1 not in engine.states
        assert queued._events.empty()

    try:
        asyncio.run(exercise())
    finally:
        engine.prepare_release.set()
        runtime.stop()
    assert runtime._unadmitted_requests == 0


def test_cpu_preparation_error_does_not_fail_other_requests_or_owner() -> None:
    engine = _FakeEngine()
    engine.prepare_errors["bad-media"] = RuntimeError("CPU processor failed")
    runtime = ServingRuntime(lambda: engine)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        failed = await _collect_events(
            runtime.submit(
                _request(
                    "bad",
                    max_tokens=2,
                    modality=Modality.IMAGE,
                    media=object(),
                    prompt="bad-media",
                ),
                loop,
            )
        )
        assert [event.kind for event in failed] == [EventKind.ERROR]
        assert "CPU processor failed" in failed[-1].error
        assert runtime.is_healthy
        later = await _collect_events(runtime.submit(_request("healthy", max_tokens=2), loop))
        assert later[-1].finish_reason == "length"
        assert engine.prepared_submissions == []

    try:
        asyncio.run(exercise())
    finally:
        runtime.stop()
    assert runtime.failure is None
    assert runtime._unadmitted_requests == 0


def test_shutdown_discards_running_and_queued_preparations_before_engine_exit() -> None:
    engine = _FakeEngine()
    engine.prepare_release.clear()
    runtime = ServingRuntime(lambda: engine, ingress_capacity=2)
    runtime.start()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        first = runtime.submit(
            _request("running", max_tokens=2, modality=Modality.IMAGE, media=object()),
            loop,
        )
        await _wait_event(engine.prepare_started)
        engine.id_allocated.clear()
        second = runtime.submit(
            _request("waiting", max_tokens=2, modality=Modality.VIDEO, media=object()),
            loop,
        )
        await _wait_event(engine.id_allocated)
        stopping = asyncio.create_task(asyncio.to_thread(runtime.stop))
        try:
            events = await asyncio.gather(_collect_events(first), _collect_events(second))
            assert all([event.kind for event in group] == [EventKind.DONE] for group in events)
            assert all(group[-1].finish_reason == "shutdown" for group in events)
            assert not stopping.done()
            assert engine.prepared_submissions == []
        finally:
            engine.prepare_release.set()
            await asyncio.wait_for(stopping, timeout=2.0)
        assert first._events.empty()
        assert second._events.empty()

    try:
        asyncio.run(exercise())
    finally:
        engine.prepare_release.set()
        runtime.stop()
    assert engine.exited
    assert not engine.exited_while_preparing
    assert runtime._unadmitted_requests == 0
    assert not runtime._pending_media
    assert not runtime._media_waiting
    assert runtime._preparing is None
