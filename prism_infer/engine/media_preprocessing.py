"""Shared CPU media preparation, content identity, and bounded processor-result reuse."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image

_MEDIA_CACHE_KEY_SCHEMA = "prism_media_content_v1"
MEDIA_CACHE_MAX_ENTRIES = 128


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

    if isinstance(value, bytes | bytearray | memoryview):
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
        if value.mode in ("P", "PA"):
            # Indexed bytes alone do not identify the colors of a palette image.
            _update_length_delimited(hasher, value.convert("RGBA").tobytes())
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
    if isinstance(value, list | tuple):
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
            "class": (f"{type(component).__module__}.{type(component).__qualname__}")
        }
        to_dict = getattr(component, "to_dict", None)
        if callable(to_dict):
            identity["config"] = to_dict()
        return identity

    namespace = {
        "schema": _MEDIA_CACHE_KEY_SCHEMA,
        "model_path": (str(model_path.resolve()) if model_path is not None else ""),
        "model_files": model_files,
        "image_max_pixels": getattr(config, "image_max_pixels", None),
        "video_max_pixels": getattr(config, "video_max_pixels", None),
        "processor": component_identity(processor),
        "image_processor": component_identity(getattr(processor, "image_processor", None)),
        "video_processor": component_identity(getattr(processor, "video_processor", None)),
        "tokenizer": component_identity(getattr(processor, "tokenizer", None)),
    }
    encoded = json.dumps(
        namespace,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _per_image_media_hashes(
    namespace: str,
    request_type: str,
    inputs: Any,
) -> tuple[bytes, ...] | None:
    """Compute one content SHA256 per image (payload order) from processor output.

    与整组 ``_visual_embedding_fingerprint`` 同域：像素张量分片 + grid 行 +
    token 身份。逐图 hash 供 block 级 mm-aware 前缀哈希把 pad 位置绑定到
    具体图片内容。视频或身份不可得时返回 None。
    """

    if request_type in ("image", "images", "interleaved_images"):
        request_type = "images"
    pixel_values = getattr(inputs, "pixel_values", None)
    grid = getattr(inputs, "image_grid_thw", None)
    token_id = getattr(inputs, "image_token_id", None)
    if pixel_values is None or grid is None or token_id is None:
        return None
    if grid.ndim != 2 or grid.shape[1] != 3:
        return None
    rows_per_image = [int(row.prod().item()) for row in grid]
    if sum(rows_per_image) != int(pixel_values.shape[0]):
        raise TypeError("per-image media hashing requires grid-aligned pixel payload")
    hashes: list[bytes] = []
    offset = 0
    for index, rows in enumerate(rows_per_image):
        slice_tensor = pixel_values[offset : offset + rows]
        fingerprint = _content_fingerprint(
            namespace.encode("ascii"),
            request_type.encode("ascii"),
            slice_tensor,
            grid[index : index + 1],
            str(token_id).encode("ascii"),
        )
        if fingerprint is None:
            return None
        hashes.append(bytes.fromhex(fingerprint))
        offset += rows
    return tuple(hashes)


def _visual_embedding_fingerprint(
    namespace: str,
    request_type: str,
    inputs: Any,
) -> str:
    """Hash the exact processor output consumed by the Vision Encoder."""

    if request_type in ("image", "images", "interleaved_images"):
        request_type = "images"
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


@dataclass(frozen=True, slots=True)
class _MediaPreprocessCacheEntry:
    """Reusable processor output for one exact media-content fingerprint."""

    inputs: Any
    visual_embedding_fingerprint: str
    prompt: str
    per_image_hashes: tuple[bytes, ...] | None = None


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
        and old_template_ids[-1 - common_suffix] == new_template_ids[-1 - common_suffix]
    ):
        common_suffix += 1

    visual_token_id = getattr(cached.inputs, "image_token_id", None)
    if visual_token_id is None:
        visual_token_id = getattr(cached.inputs, "video_token_id", None)
    placeholder_positions = [
        index for index, token_id in enumerate(old_template_ids) if token_id == visual_token_id
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
    if rebound_ids.count(visual_token_id) != expanded_ids.count(visual_token_id):
        return None
    input_ids = cached.inputs.input_ids.new_tensor(rebound_ids).unsqueeze(0)
    attention_mask = cached.inputs.attention_mask.new_ones(input_ids.shape)
    return replace(
        cached.inputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_text=rebound_prompt_text,
    )


class MediaPreprocessingCache:
    """One engine-owned LRU of processed media and its latest compatible prompt.

    Content reading, Processor calls, tokenization, and identity calculation run
    outside the state lock. Concurrent cold misses may compute twice; completed
    immutable entries are published under the lock. No object-ID content memo is
    used: mutable PIL/NumPy/tensor inputs are fingerprinted on every request.
    """

    def __init__(self, namespace: str, *, max_entries: int = MEDIA_CACHE_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("media preprocessing cache max_entries must be positive")
        self.namespace = namespace
        self.max_entries = max_entries
        self._lock = Lock()
        self._entries: OrderedDict[tuple[str, str, str], _MediaPreprocessCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._uncacheable = 0
        self._rebind_hits = 0
        self._rebind_misses = 0

    def prepare(
        self,
        request_type: str,
        prompt: str,
        media: Any,
        *,
        process_inputs: Callable[[], Any],
        tokenizer: Any,
        image_marker: str = "<image>",
    ) -> _MediaPreprocessCacheEntry:
        if request_type == "image":
            request_type = "images"
        media_objects = tuple(media) if isinstance(media, list | tuple) else (media,)
        content_key = _content_fingerprint(*media_objects)
        key = (request_type, image_marker, content_key) if content_key is not None else None
        with self._lock:
            cached = self._entries.get(key) if key is not None else None
            if cached is not None:
                self._entries.move_to_end(key)
                if cached.prompt == prompt:
                    self._hits += 1
                    return cached

        rebound = (
            None
            if cached is None
            else _rebind_cached_media_prompt(cached, prompt=prompt, tokenizer=tokenizer)
        )
        if rebound is not None:
            result = replace(cached, inputs=rebound, prompt=prompt)
        else:
            inputs = process_inputs()
            # Image submission aliases share the same processed-media identity.
            identity_type = "video" if request_type == "video" else "images"
            fingerprint = _visual_embedding_fingerprint(self.namespace, identity_type, inputs)
            per_image_hashes = (
                None
                if identity_type == "video"
                else _per_image_media_hashes(self.namespace, identity_type, inputs)
            )
            result = _MediaPreprocessCacheEntry(
                inputs=inputs,
                visual_embedding_fingerprint=fingerprint,
                prompt=prompt,
                per_image_hashes=per_image_hashes,
            )

        with self._lock:
            if rebound is not None:
                self._hits += 1
                self._rebind_hits += 1
            else:
                self._misses += 1
                self._rebind_misses += int(cached is not None)
                self._uncacheable += int(key is None)
            if key is not None:
                self._entries[key] = result
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        return result

    def metadata(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "scope": "engine_shared_cpu_media_results",
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "uncacheable": self._uncacheable,
                "prompt_rebind_hits": self._rebind_hits,
                "prompt_rebind_misses": self._rebind_misses,
            }

    def reset_metrics(self) -> None:
        with self._lock:
            self._hits = self._misses = self._uncacheable = 0
            self._rebind_hits = self._rebind_misses = 0

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
