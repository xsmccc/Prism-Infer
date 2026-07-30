"""Single-process online arrival loop built on the P7 engine contracts."""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field, replace
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter_ns, sleep
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image
import torch

from prism_infer.sampling_params import SamplingParams


NANOSECONDS_PER_SECOND = 1_000_000_000
_ONLINE_MEDIA_FIELD_BY_TYPE = {
    "image": "image",
    "images": "images",
    "video": "video",
}
_SUPPORTED_ONLINE_REQUEST_TYPES = frozenset({"text", *_ONLINE_MEDIA_FIELD_BY_TYPE})
_ONLINE_PREPROCESS_WORKERS = 1
_ONLINE_MEDIA_CACHE_MAX_ENTRIES = 128
_MEDIA_CACHE_KEY_SCHEMA = "prism_media_content_v1"


def _update_length_delimited(
    hasher: Any,
    value: bytes | bytearray | memoryview,
) -> None:
    view = memoryview(value).cast("B")
    hasher.update(len(view).to_bytes(8, byteorder="little", signed=False))
    hasher.update(view)


def _update_text(hasher: Any, value: str) -> None:
    _update_length_delimited(hasher, value.encode("utf-8"))


def _update_content_hash(hasher: Any, value: object) -> bool:
    """Add exact supported media content to ``hasher``.

    Unsupported opaque objects deliberately return ``False`` instead of using
    identity or ``repr``. That keeps the content-addressed cache fail-closed.
    """

    if isinstance(value, (bytes, bytearray, memoryview)):
        _update_text(hasher, "bytes")
        _update_length_delimited(hasher, value)
        return True
    if isinstance(value, Path) or isinstance(value, str):
        try:
            path = Path(value)
            is_file = path.is_file()
        except OSError:
            return False
        if not is_file:
            return False
        _update_text(hasher, "file")
        _update_text(hasher, path.suffix.lower())
        hasher.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as media_file:
            while chunk := media_file.read(1024 * 1024):
                hasher.update(chunk)
        return True
    if isinstance(value, Image.Image):
        _update_text(hasher, "pil")
        _update_text(hasher, value.mode)
        _update_text(hasher, json.dumps(value.size))
        _update_length_delimited(hasher, value.tobytes())
        return True
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _update_text(hasher, "numpy")
        _update_text(hasher, array.dtype.str)
        _update_text(hasher, json.dumps(array.shape))
        _update_length_delimited(hasher, memoryview(array).cast("B"))
        return True
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        byte_view = tensor.flatten().view(torch.uint8).numpy()
        _update_text(hasher, "torch")
        _update_text(hasher, str(tensor.dtype))
        _update_text(hasher, json.dumps(tuple(tensor.shape)))
        _update_length_delimited(hasher, memoryview(byte_view))
        return True
    if isinstance(value, (list, tuple)):
        _update_text(hasher, type(value).__name__)
        hasher.update(len(value).to_bytes(8, "little"))
        return all(_update_content_hash(hasher, item) for item in value)
    return False


def _content_fingerprint(*values: object) -> str | None:
    hasher = hashlib.sha256()
    _update_text(hasher, _MEDIA_CACHE_KEY_SCHEMA)
    if not all(_update_content_hash(hasher, value) for value in values):
        return None
    return hasher.hexdigest()


def _cache_namespace(engine: Any) -> str:
    """Fingerprint the model and processor semantics used by cached outputs."""

    config = getattr(engine, "config", None)
    processor = getattr(engine, "vl_processor", None)
    model_value = getattr(config, "model", "")
    model_path = Path(str(model_value)) if model_value else None
    model_files = []
    if model_path is not None and model_path.is_dir():
        for path in sorted(model_path.glob("*.json")):
            model_files.append(
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        for path in sorted(model_path.glob("*.safetensors")):
            stat = path.stat()
            model_files.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    def component_identity(component: object) -> dict[str, object] | None:
        if component is None:
            return None
        identity: dict[str, object] = {
            "class": (
                f"{type(component).__module__}.{type(component).__qualname__}"
            )
        }
        to_dict = getattr(component, "to_dict", None)
        if callable(to_dict):
            identity["config"] = to_dict()
        return identity

    namespace = {
        "schema": _MEDIA_CACHE_KEY_SCHEMA,
        "model_path": (
            str(model_path.resolve()) if model_path is not None else ""
        ),
        "model_files": model_files,
        "image_max_pixels": getattr(config, "image_max_pixels", None),
        "video_max_pixels": getattr(config, "video_max_pixels", None),
        "processor": component_identity(processor),
        "image_processor": component_identity(
            getattr(processor, "image_processor", None)
        ),
        "video_processor": component_identity(
            getattr(processor, "video_processor", None)
        ),
        "tokenizer": component_identity(
            getattr(processor, "tokenizer", None)
        ),
    }
    encoded = json.dumps(
        namespace,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visual_embedding_fingerprint(
    namespace: str,
    request_type: str,
    inputs: Any,
) -> str:
    """Hash the exact processor output consumed by the Vision Encoder."""

    if request_type in ("image", "images"):
        payload = inputs.pixel_values
        grid = inputs.image_grid_thw
        token_id = inputs.image_token_id
        token_count = inputs.image_token_count
    else:
        payload = inputs.pixel_values_videos
        grid = inputs.video_grid_thw
        token_id = inputs.video_token_id
        token_count = inputs.video_token_count
    fingerprint = _content_fingerprint(
        namespace.encode("ascii"),
        request_type.encode("ascii"),
        payload,
        grid,
        str(token_id).encode("ascii"),
        str(token_count).encode("ascii"),
    )
    if fingerprint is None:
        raise TypeError("processor output contains unsupported cache-key data")
    return fingerprint


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
    ttft_slo_ms: float | None = None

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
        if self.ttft_slo_ms is not None and (
            isinstance(self.ttft_slo_ms, bool)
            or not isinstance(self.ttft_slo_ms, (int, float))
            or not isfinite(float(self.ttft_slo_ms))
            or self.ttft_slo_ms <= 0
        ):
            raise ValueError(
                "ttft_slo_ms must be a finite positive number or None"
            )
        if self.ttft_slo_ms is not None:
            object.__setattr__(self, "ttft_slo_ms", float(self.ttft_slo_ms))


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
    media_preprocess_cache: dict[str, int] = field(default_factory=dict)

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
            "media_preprocess_cache": dict(self.media_preprocess_cache),
        }


@dataclass(slots=True)
class _PendingPreprocess:
    """One arrived media request being prepared off the engine thread."""

    request: OnlineRequest
    arrival_ns: int
    request_id: int
    future: Future


@dataclass(frozen=True, slots=True)
class _MediaPreprocessCacheEntry:
    """Reusable processor output for one exact media-content fingerprint."""

    inputs: Any
    visual_embedding_fingerprint: str
    prompt: str


def _tokenize_prompt_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = encoded["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("media prompt tokenizer returned a batched result")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _rebind_cached_media_prompt(
    cached: _MediaPreprocessCacheEntry,
    *,
    prompt: str,
    tokenizer: Any,
) -> Any | None:
    """Retokenize a changed question while retaining processed media tensors."""

    if tokenizer is None:
        return None
    if cached.prompt == prompt:
        return cached.inputs
    prompt_text = cached.inputs.prompt_text
    if prompt_text.count(cached.prompt) != 1:
        return None
    rebound_prompt_text = prompt_text.replace(cached.prompt, prompt, 1)
    old_template_ids = _tokenize_prompt_text(tokenizer, prompt_text)
    new_template_ids = _tokenize_prompt_text(tokenizer, rebound_prompt_text)

    common_prefix = 0
    for old_token, new_token in zip(
        old_template_ids,
        new_template_ids,
        strict=False,
    ):
        if old_token != new_token:
            break
        common_prefix += 1
    common_suffix = 0
    max_suffix = min(
        len(old_template_ids) - common_prefix,
        len(new_template_ids) - common_prefix,
    )
    while (
        common_suffix < max_suffix
        and old_template_ids[-1 - common_suffix]
        == new_template_ids[-1 - common_suffix]
    ):
        common_suffix += 1

    visual_token_id = getattr(cached.inputs, "image_token_id", None)
    if visual_token_id is None:
        visual_token_id = getattr(cached.inputs, "video_token_id", None)
    placeholder_positions = [
        index
        for index, token_id in enumerate(old_template_ids)
        if token_id == visual_token_id
    ]
    if not placeholder_positions or common_prefix <= placeholder_positions[-1]:
        return None

    expanded_ids = [int(token_id) for token_id in cached.inputs.token_ids]
    expansion_offset = len(expanded_ids) - len(old_template_ids)
    replace_start = common_prefix + expansion_offset
    replace_end = len(expanded_ids) - common_suffix
    if not 0 <= replace_start <= replace_end <= len(expanded_ids):
        return None
    if common_suffix and expanded_ids[replace_end:] != old_template_ids[-common_suffix:]:
        return None

    replacement_end = len(new_template_ids) - common_suffix
    rebound_ids = (
        expanded_ids[:replace_start]
        + new_template_ids[common_prefix:replacement_end]
        + expanded_ids[replace_end:]
    )
    input_ids = cached.inputs.input_ids.new_tensor(rebound_ids).unsqueeze(0)
    attention_mask = cached.inputs.attention_mask.new_ones(input_ids.shape)
    return replace(
        cached.inputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_text=rebound_prompt_text,
    )


@dataclass(frozen=True, slots=True)
class _MediaFingerprintMemoEntry:
    """Content fingerprint memoized behind a verified object-identity fast path."""

    media_objects: tuple[Any, ...]
    fingerprint: str


@dataclass(slots=True)
class _OnlineRunState:
    """Mutable state for one online event-loop invocation."""

    submitted: tuple[OnlineRequest, ...]
    pending: deque[OnlineRequest]
    cancellations: deque[tuple[float, str]]
    started_ns: int
    internal_ids: dict[str, int] = field(default_factory=dict)
    outputs: dict[int, tuple[int, ...]] = field(default_factory=dict)
    preprocessing: dict[str, _PendingPreprocess] = field(default_factory=dict)
    admitted_keys: set[str] = field(default_factory=set)
    deferred_cancellations: set[str] = field(default_factory=set)

    def elapsed_seconds(self, now_ns: int) -> float:
        return (now_ns - self.started_ns) / NANOSECONDS_PER_SECOND


class OnlineServingSession:
    """Drive arrivals and dynamic batches through one ``LLMEngine`` instance.

    Scheduler admission and model execution share one control thread. CPU
    media preprocessing runs on a bounded worker so it cannot stall decode.
    Arrivals retain their intended timestamp and enter scheduling after their
    host-side preprocessing completes.
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
        self._media_preprocess_cache: OrderedDict[
            tuple[str, str, str, str],
            _MediaPreprocessCacheEntry,
        ] = OrderedDict()
        self._media_layout_cache: OrderedDict[
            tuple[str, str, str],
            _MediaPreprocessCacheEntry,
        ] = OrderedDict()
        self._media_fingerprint_memo: OrderedDict[
            tuple[str, tuple[int, ...]],
            _MediaFingerprintMemoEntry,
        ] = OrderedDict()
        self._cache_namespace = _cache_namespace(engine)
        self._media_preprocess_cache_hits = 0
        self._media_preprocess_cache_misses = 0
        self._media_preprocess_cache_uncacheable = 0
        self._media_prompt_rebind_hits = 0
        self._media_prompt_rebind_misses = 0
        self._media_fingerprint_memo_hits = 0
        self._media_fingerprint_memo_misses = 0

    def _submit(self, request: OnlineRequest, arrival_ns: int) -> int:
        payload = request.payload
        request_type = payload.get("type", "text")
        common = {
            "submitted_ns": arrival_ns,
            "raise_on_reject": False,
        }
        if request_type == "text":
            return self.engine.add_request(
                payload["prompt"],
                request.sampling_params,
                ttft_slo_ms=request.ttft_slo_ms,
                **common,
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
        preprocess_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        while state.pending:
            elapsed_s = state.elapsed_seconds(self.clock_ns())
            if state.pending[0].arrival_offset_s > elapsed_s:
                return
            request = state.pending.popleft()
            arrival_ns = state.started_ns + int(request.arrival_offset_s * NANOSECONDS_PER_SECOND)
            if request.payload.get("type", "text") == "text":
                state.internal_ids[request.request_key] = self._submit(
                    request,
                    arrival_ns,
                )
                state.admitted_keys.add(request.request_key)
                continue
            request_id = self.engine._allocate_request_id()
            state.internal_ids[request.request_key] = request_id
            if preprocess_executor is None:
                raise RuntimeError(
                    "media arrivals require a preprocessing executor"
                )
            state.preprocessing[request.request_key] = _PendingPreprocess(
                request=request,
                arrival_ns=arrival_ns,
                request_id=request_id,
                future=preprocess_executor.submit(
                    self._prepare_media_sequence,
                    request,
                    request_id,
                ),
            )

    def _prepare_media_sequence(
        self,
        request: OnlineRequest,
        request_id: int,
    ) -> Any:
        """Prepare one media request without touching scheduler state."""

        payload = request.payload
        request_type = payload.get("type", "text")
        media_field = _ONLINE_MEDIA_FIELD_BY_TYPE.get(request_type)
        if media_field is None:
            raise RuntimeError(
                "background preprocessing received unsupported type "
                f"{request_type!r}"
            )
        media = payload[media_field]
        media_objects = (
            tuple(media)
            if isinstance(media, (list, tuple))
            else (media,)
        )
        fingerprint_memo_key = (
            request_type,
            tuple(id(item) for item in media_objects),
        )
        memoized = self._media_fingerprint_memo.get(
            fingerprint_memo_key
        )
        if (
            memoized is not None
            and len(memoized.media_objects) == len(media_objects)
            and all(
                memoized_item is request_item
                for memoized_item, request_item in zip(
                    memoized.media_objects,
                    media_objects,
                    strict=True,
                )
            )
        ):
            media_fingerprint = memoized.fingerprint
            self._media_fingerprint_memo.move_to_end(
                fingerprint_memo_key
            )
            self._media_fingerprint_memo_hits += 1
        else:
            self._media_fingerprint_memo_misses += 1
            if memoized is not None:
                del self._media_fingerprint_memo[
                    fingerprint_memo_key
                ]
            media_fingerprint = _content_fingerprint(*media_objects)
            if media_fingerprint is not None:
                self._media_fingerprint_memo[
                    fingerprint_memo_key
                ] = _MediaFingerprintMemoEntry(
                    media_objects=media_objects,
                    fingerprint=media_fingerprint,
                )
                self._media_fingerprint_memo.move_to_end(
                    fingerprint_memo_key
                )
                while (
                    len(self._media_fingerprint_memo)
                    > _ONLINE_MEDIA_CACHE_MAX_ENTRIES
                ):
                    self._media_fingerprint_memo.popitem(last=False)
        cache_key = (
            self._cache_namespace,
            request_type,
            payload["prompt"],
            media_fingerprint or "",
        )
        layout_key = (
            self._cache_namespace,
            request_type,
            media_fingerprint or "",
        )
        cached = (
            None
            if media_fingerprint is None
            else self._media_preprocess_cache.get(cache_key)
        )
        if cached is not None:
            self._media_preprocess_cache.move_to_end(cache_key)
            self._media_preprocess_cache_hits += 1
            inputs = cached.inputs
            visual_embedding_fingerprint = (
                cached.visual_embedding_fingerprint
            )
        else:
            layout_cached = (
                None
                if media_fingerprint is None
                else self._media_layout_cache.get(layout_key)
            )
            inputs = (
                None
                if layout_cached is None
                else _rebind_cached_media_prompt(
                    layout_cached,
                    prompt=payload["prompt"],
                    tokenizer=getattr(
                        getattr(self.engine, "vl_processor", None),
                        "tokenizer",
                        None,
                    ),
                )
            )
            if inputs is not None:
                self._media_preprocess_cache_hits += 1
                self._media_prompt_rebind_hits += 1
                self._media_layout_cache.move_to_end(layout_key)
                visual_embedding_fingerprint = (
                    layout_cached.visual_embedding_fingerprint
                )
            else:
                self._media_preprocess_cache_misses += 1
                if layout_cached is not None:
                    self._media_prompt_rebind_misses += 1
                if media_fingerprint is None:
                    self._media_preprocess_cache_uncacheable += 1
                if request_type in ("image", "images"):
                    inputs = self.engine._process_image_inputs(
                        payload["prompt"],
                        media,
                    )
                elif request_type == "video":
                    inputs = self.engine._process_video_inputs(
                        payload["prompt"],
                        media,
                    )
                else:
                    raise RuntimeError(
                        "background preprocessing received unsupported type "
                        f"{request_type!r}"
                    )
                visual_embedding_fingerprint = (
                    _visual_embedding_fingerprint(
                        self._cache_namespace,
                        request_type,
                        inputs=inputs,
                    )
                )
            if media_fingerprint is not None:
                cache_entry = _MediaPreprocessCacheEntry(
                    inputs=inputs,
                    visual_embedding_fingerprint=(
                        visual_embedding_fingerprint
                    ),
                    prompt=payload["prompt"],
                )
                self._media_preprocess_cache[cache_key] = cache_entry
                self._media_preprocess_cache.move_to_end(cache_key)
                while (
                    len(self._media_preprocess_cache)
                    > _ONLINE_MEDIA_CACHE_MAX_ENTRIES
                ):
                    self._media_preprocess_cache.popitem(last=False)
                self._media_layout_cache[layout_key] = cache_entry
                self._media_layout_cache.move_to_end(layout_key)
                while (
                    len(self._media_layout_cache)
                    > _ONLINE_MEDIA_CACHE_MAX_ENTRIES
                ):
                    self._media_layout_cache.popitem(last=False)

        if request_type in ("image", "images"):
            sequence = self.engine._prepare_image_sequence(
                inputs,
                request.sampling_params,
                request_id=request_id,
            )
        elif request_type == "video":
            sequence = self.engine._prepare_video_sequence(
                inputs,
                request.sampling_params,
                request_id=request_id,
            )
        else:
            raise AssertionError("unreachable media request type")
        engine_config = getattr(self.engine, "config", None)
        if getattr(
            engine_config,
            "enable_visual_embedding_cache",
            False,
        ):
            sequence.visual_embedding_cache_key = (
                visual_embedding_fingerprint
            )
        sequence.multimodal_prefix_cache_key = (
            visual_embedding_fingerprint
        )
        return sequence

    def _admit_ready_preprocessing(self, state: _OnlineRunState) -> None:
        """Publish completed preprocessing results from the engine thread."""

        for request_key, pending in tuple(state.preprocessing.items()):
            if not pending.future.done():
                continue
            sequence = pending.future.result()
            self.engine._submit_sequence(
                sequence,
                submitted_ns=pending.arrival_ns,
                ttft_slo_ms=pending.request.ttft_slo_ms,
                raise_on_reject=False,
            )
            state.admitted_keys.add(request_key)
            del state.preprocessing[request_key]
            if request_key in state.deferred_cancellations:
                self.engine.cancel_request(pending.request_id)
                state.deferred_cancellations.remove(request_key)

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
            if request_id is None:
                continue
            if request_key in state.admitted_keys:
                self.engine.cancel_request(request_id)
            else:
                state.deferred_cancellations.add(request_key)

    def _execute_step(self, state: _OnlineRunState) -> bool:
        if self.engine.is_finished():
            return False
        step = self.engine.step_result()
        for output in step.outputs:
            state.outputs[output.request_id] = output.token_ids
        return True

    def _wait_for_next_event(self, state: _OnlineRunState) -> None:
        wait_s = None
        if state.pending:
            wait_s = max(
                0.0,
                state.pending[0].arrival_offset_s
                - state.elapsed_seconds(self.clock_ns()),
            )
        futures = [
            pending.future for pending in state.preprocessing.values()
        ]
        if futures:
            wait(
                futures,
                timeout=wait_s,
                return_when=FIRST_COMPLETED,
            )
        elif wait_s is not None and wait_s > 0:
            self.sleep_fn(wait_s)

    def _drive_event_loop(
        self,
        state: _OnlineRunState,
        preprocess_executor: ThreadPoolExecutor,
    ) -> None:
        while (
            state.pending
            or state.preprocessing
            or not self.engine.is_finished()
        ):
            self._submit_ready_arrivals(state, preprocess_executor)
            self._admit_ready_preprocessing(state)
            self._apply_ready_cancellations(state)
            if self._execute_step(state):
                continue
            self._wait_for_next_event(state)

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
        with ThreadPoolExecutor(
            max_workers=_ONLINE_PREPROCESS_WORKERS,
            thread_name_prefix="prism-media-preprocess",
        ) as preprocess_executor:
            self._drive_event_loop(state, preprocess_executor)
        finished_ns = self.clock_ns()
        metrics = self.engine.metrics_snapshot()
        return OnlineRunResult(
            started_ns=state.started_ns,
            finished_ns=finished_ns,
            requests=self._request_results(state, metrics),
            engine_metrics=metrics,
            scheduler_metrics=self.engine.scheduler.metrics_snapshot(),
            media_preprocess_cache={
                "entries": len(self._media_preprocess_cache),
                "layout_entries": len(self._media_layout_cache),
                "hits": self._media_preprocess_cache_hits,
                "max_entries": _ONLINE_MEDIA_CACHE_MAX_ENTRIES,
                "misses": self._media_preprocess_cache_misses,
                "uncacheable": self._media_preprocess_cache_uncacheable,
                "prompt_rebind_hits": self._media_prompt_rebind_hits,
                "prompt_rebind_misses": self._media_prompt_rebind_misses,
                "fingerprint_memo_hits": (
                    self._media_fingerprint_memo_hits
                ),
                "fingerprint_memo_misses": (
                    self._media_fingerprint_memo_misses
                ),
            },
        )

    def reset_metrics(self) -> None:
        """Reset measured counters while retaining safe processor cache entries."""

        self._media_preprocess_cache_hits = 0
        self._media_preprocess_cache_misses = 0
        self._media_preprocess_cache_uncacheable = 0
        self._media_prompt_rebind_hits = 0
        self._media_prompt_rebind_misses = 0
        self._media_fingerprint_memo_hits = 0
        self._media_fingerprint_memo_misses = 0
