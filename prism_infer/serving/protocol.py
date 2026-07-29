"""Prism-Infer 原生网络服务协议。

该模块只定义与 ASGI 框架无关的强类型请求和事件。HTTP/Pydantic 适配放在
``serving.app``，GPU 引擎所有权和调度驱动放在 ``serving.runtime``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from prism_infer.sampling_params import SamplingParams


class Modality(str, Enum):
    """网络请求支持的输入模态。"""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class EventKind(str, Enum):
    """请求生命周期中对网络层可见的事件类型。"""

    ACCEPTED = "accepted"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """一条已经过网络层解码的生成请求。

    Attributes:
        request_id: 客户端可见且在当前运行时内唯一的请求标识。
        prompt: 非空文本提示词。
        modality: 文本、图像或视频。
        media: 图像对象、图像对象元组、视频路径或视频帧元组。
        sampling_params: Prism-Infer 原生采样参数。
    """

    request_id: str
    prompt: str
    modality: Modality
    media: Any | tuple[Any, ...] | None
    sampling_params: SamplingParams

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality")
        if not isinstance(self.sampling_params, SamplingParams):
            raise TypeError("sampling_params must be SamplingParams")
        if self.modality is Modality.TEXT and self.media is not None:
            raise ValueError("text requests must not include media")
        if self.modality is not Modality.TEXT and self.media is None:
            raise ValueError(f"{self.modality.value} requests require media")
        if isinstance(self.media, tuple) and not self.media:
            raise ValueError("media tuples must not be empty")


@dataclass(frozen=True, slots=True)
class ServingEvent:
    """从引擎所有者线程发送给一个异步客户端的事件。"""

    request_id: str
    kind: EventKind
    engine_request_id: int | None = None
    token_id: int | None = None
    token_text: str | None = None
    token_ids: tuple[int, ...] = ()
    text: str | None = None
    finish_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """返回可直接 JSON 序列化的事件。"""

        payload: dict[str, object] = {
            "request_id": self.request_id,
            "event": self.kind.value,
        }
        optional_values = {
            "engine_request_id": self.engine_request_id,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        payload.update(
            {
                name: value
                for name, value in optional_values.items()
                if value is not None
            }
        )
        if self.kind is EventKind.DONE:
            payload["token_ids"] = list(self.token_ids)
        return payload
