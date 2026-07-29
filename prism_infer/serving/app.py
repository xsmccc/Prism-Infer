"""FastAPI adapter for the Prism-Infer native generation protocol."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prism_infer.sampling_params import SamplingParams
from prism_infer.serving.protocol import (
    EventKind,
    GenerationRequest,
    Modality,
    ServingEvent,
)
from prism_infer.serving.runtime import (
    DuplicateRequestError,
    RequestHandle,
    ServingOverloadedError,
    ServingRuntime,
    ServingUnavailableError,
)


class SamplingBody(BaseModel):
    """网络协议暴露的最小采样参数集合。"""

    model_config = ConfigDict(extra="forbid")

    temperature: Annotated[float, Field(ge=0.0)] = 0.0
    max_tokens: Annotated[int, Field(gt=0)] = 64
    ignore_eos: bool = False


class GenerateBody(BaseModel):
    """``POST /v1/generate`` 请求体。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    prompt: Annotated[str, Field(min_length=1)]
    modality: Literal["text", "image", "video"] = "text"
    media: str | list[str] | None = None
    sampling: SamplingBody = Field(default_factory=SamplingBody)
    stream: bool = False

    @model_validator(mode="after")
    def validate_media(self) -> "GenerateBody":
        """拒绝模态和媒体字段不一致的歧义请求。"""

        if self.modality == "text" and self.media is not None:
            raise ValueError("text requests must not include media")
        if self.modality != "text" and self.media is None:
            raise ValueError(f"{self.modality} requests require media")
        if isinstance(self.media, list) and not self.media:
            raise ValueError("media lists must not be empty")
        if self.modality == "video":
            if not isinstance(self.media, list):
                raise ValueError(
                    "video requests require an explicit list of sampled data:image frames"
                )
            if any(not source.startswith("data:image/") for source in self.media):
                raise ValueError(
                    "video media lists must contain sampled data:image frame URLs"
                )
        return self


def _decode_image_data_url(source: str) -> Image.Image:
    """将一个 ``data:image/...;base64`` URL 解码为已加载的 RGB 图像。"""

    header, separator, encoded = source.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("media data URL must be base64-encoded image data")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise ValueError("media contains invalid base64 image data") from exc


def _decode_media_source(source: str) -> str | Image.Image:
    """解码内联图像；普通字符串保留为 processor 可读取的路径或 URL。"""

    if source.startswith("data:"):
        return _decode_image_data_url(source)
    return source


def _materialize_media(body: GenerateBody) -> object | tuple[object, ...] | None:
    """把 JSON 媒体字段转换成现有 VL processor 接受的对象。"""

    if body.media is None:
        return None
    if isinstance(body.media, list):
        return tuple(_decode_media_source(source) for source in body.media)
    return _decode_media_source(body.media)


def _to_generation_request(body: GenerateBody) -> GenerationRequest:
    """把 Pydantic 请求转换成框架无关的 Serving 请求。"""

    request_id = body.request_id or uuid4().hex
    return GenerationRequest(
        request_id=request_id,
        prompt=body.prompt,
        modality=Modality(body.modality),
        media=_materialize_media(body),
        sampling_params=SamplingParams(
            temperature=body.sampling.temperature,
            max_tokens=body.sampling.max_tokens,
            ignore_eos=body.sampling.ignore_eos,
        ),
    )


def _sse(event: ServingEvent) -> bytes:
    """将一条运行时事件编码成 UTF-8 Server-Sent Event。"""

    payload = json.dumps(
        event.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event.kind.value}\ndata: {payload}\n\n".encode()


async def _stream_events(
    runtime: ServingRuntime,
    handle: RequestHandle,
) -> AsyncIterator[bytes]:
    """逐条转发真实引擎 step 事件，并在连接中止时提交取消。"""

    terminal = False
    try:
        while True:
            event = await handle.next_event()
            yield _sse(event)
            if event.kind in {EventKind.DONE, EventKind.ERROR}:
                terminal = True
                return
    finally:
        if not terminal:
            runtime.cancel(handle.request_id)


async def _collect_response(
    handle: RequestHandle,
) -> ServingEvent:
    """收集非流式请求，直到运行时发送完成或错误。"""

    while True:
        event = await handle.next_event()
        if event.kind in {EventKind.DONE, EventKind.ERROR}:
            return event


async def _wait_for_disconnect(request: Request) -> None:
    """等待 ASGI 客户端断开，不使用轮询干扰请求延迟。"""

    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _non_stream_response(
    runtime: ServingRuntime,
    handle: RequestHandle,
    request: Request,
) -> JSONResponse:
    """在结果和客户端断开之间竞速，断开时释放请求资源。"""

    result_task = asyncio.create_task(_collect_response(handle))
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    completed, _ = await asyncio.wait(
        {result_task, disconnect_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if disconnect_task in completed:
        runtime.cancel(handle.request_id)
        result_task.cancel()
        await asyncio.gather(result_task, return_exceptions=True)
        raise HTTPException(status_code=499, detail="client disconnected")

    disconnect_task.cancel()
    await asyncio.gather(disconnect_task, return_exceptions=True)
    event = result_task.result()
    if event.kind is EventKind.ERROR:
        raise HTTPException(status_code=400, detail=event.error)
    return JSONResponse(event.to_dict())


def create_app(runtime: ServingRuntime) -> FastAPI:
    """构造绑定一个 ``ServingRuntime`` 的 ASGI 应用。

    Args:
        runtime: 唯一的 GPU 引擎所有者运行时。

    Returns:
        只包含健康检查和 Prism 原生生成端点的 FastAPI 应用。
    """

    if not isinstance(runtime, ServingRuntime):
        raise TypeError("runtime must be ServingRuntime")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(runtime.start)
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.stop)

    app = FastAPI(
        title="Prism-Infer",
        version="0.1",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        if runtime.is_healthy:
            return JSONResponse({"status": "ok"})
        failure = runtime.failure
        detail = None if failure is None else f"{type(failure).__name__}: {failure}"
        return JSONResponse(
            {"status": "unavailable", "detail": detail},
            status_code=503,
        )

    @app.post("/v1/generate", response_model=None)
    async def generate(body: GenerateBody, request: Request) -> JSONResponse | StreamingResponse:
        try:
            generation_request = _to_generation_request(body)
            handle = runtime.submit(
                generation_request,
                asyncio.get_running_loop(),
            )
        except ServingOverloadedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ServingUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except DuplicateRequestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.stream:
            return StreamingResponse(
                _stream_events(runtime, handle),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return await _non_stream_response(runtime, handle, request)

    return app
