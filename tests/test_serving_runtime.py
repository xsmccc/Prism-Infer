"""Focused lifecycle checks for the network-to-engine concurrency bridge."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from prism_infer.engine.contracts import RequestOutput
from prism_infer.engine.request import RequestState
from prism_infer.sampling_params import SamplingParams
from prism_infer.serving.protocol import EventKind, GenerationRequest, Modality
from prism_infer.serving.runtime import ServingRuntime


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

    def _add(self, sampling_params: SamplingParams) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.active[request_id] = (0, sampling_params.max_tokens)
        self.states[request_id] = RequestState.PREFILLING
        return request_id

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int:
        del prompt, raise_on_reject
        return self._add(sampling_params)

    def add_vl_request(
        self,
        prompt: str,
        image: object,
        sampling_params: SamplingParams,
        *,
        raise_on_reject: bool,
    ) -> int:
        del prompt, image, raise_on_reject
        return self._add(sampling_params)

    add_images_request = add_vl_request
    add_video_request = add_vl_request

    def cancel_request(self, request_id: int) -> bool:
        if request_id not in self.active:
            return False
        self.active.pop(request_id)
        self.states[request_id] = RequestState.CANCELLED
        return True

    def exit(self) -> None:
        self.exited = True

    def is_finished(self) -> bool:
        return not self.active

    def request_state(self, request_id: int) -> RequestState | None:
        return self.states.get(request_id)

    def step_result(self) -> SimpleNamespace:
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        sequence_ids = tuple(self.active)
        token_ids: list[int] = []
        outputs: list[RequestOutput] = []
        for request_id in sequence_ids:
            generated, max_tokens = self.active[request_id]
            token_id = 100 + generated
            token_ids.append(token_id)
            generated += 1
            if generated == max_tokens:
                self.active.pop(request_id)
                self.states[request_id] = RequestState.FINISHED
                outputs.append(
                    RequestOutput(
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


def _request(request_id: str, *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt="hello",
        modality=Modality.TEXT,
        media=None,
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
