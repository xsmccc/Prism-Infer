"""Deterministic content-reuse workloads for multimodal cache benchmarks."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

_MEDIA_KEYS = ("image", "images", "video")


def _copy_image(image: object, *, variant_id: int | None) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"multimodal cache workload requires decoded PIL images, got {type(image).__name__}"
        )
    copied = image.copy()
    if variant_id is None:
        return copied

    # Encode the variant ID as a small binary marker. This guarantees distinct
    # decoded content without changing the tensor shape or media layout.
    draw = ImageDraw.Draw(copied)
    draw.rectangle((0, 0, 35, 35), fill=(17, 34, 51))
    for bit in range(64):
        x = 2 + (bit % 8) * 4
        y = 2 + (bit // 8) * 4
        value = 255 if (variant_id >> bit) & 1 else 0
        draw.rectangle((x, y, x + 2, y + 2), fill=(value, value, value))
    return copied


def _copy_media_value(
    value: object,
    *,
    variant_id: int | None,
) -> object:
    if isinstance(value, list):
        if not value:
            raise ValueError("multimodal cache workload received empty media")
        return [
            _copy_image(
                image,
                variant_id=(variant_id if index == 0 else None),
            )
            for index, image in enumerate(value)
        ]
    return _copy_image(value, variant_id=variant_id)


def _is_multimodal(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in _MEDIA_KEYS)


def _repeat_selection(
    index: int,
    *,
    repeated_count: int,
    media_count: int,
) -> bool:
    """Spread an exact repeated-count target across the media sequence."""

    before = index * repeated_count // media_count
    after = (index + 1) * repeated_count // media_count
    return after > before


def build_multimodal_cache_workload(
    payloads: list[dict[str, Any]],
    *,
    repeat_rate: float,
    vary_questions: bool,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Clone media objects and materialize an exact content-repeat ratio.

    Repeated requests receive byte-identical media in new Python objects.
    Unique requests receive a deterministic content marker in the first image
    or video frame. Text-only requests remain unchanged.
    """

    if not 0.0 <= repeat_rate <= 1.0:
        raise ValueError(f"repeat_rate must be in [0, 1], got {repeat_rate}")
    media_count = sum(_is_multimodal(payload) for payload in payloads)
    repeated_count = round(media_count * repeat_rate)
    transformed: list[dict[str, Any]] = []
    media_index = 0
    for payload in payloads:
        copied = dict(payload)
        if not _is_multimodal(payload):
            transformed.append(copied)
            continue

        repeated = _repeat_selection(
            media_index,
            repeated_count=repeated_count,
            media_count=media_count,
        )
        variant_id = None if repeated else media_index + 1
        for key in _MEDIA_KEYS:
            if key in payload:
                copied[key] = _copy_media_value(
                    payload[key],
                    variant_id=variant_id,
                )
        if vary_questions:
            copied["prompt"] = (
                f"{str(payload['prompt']).rstrip()}\n"
                f"Answer request variant {media_index + 1} concisely."
            )
        transformed.append(copied)
        media_index += 1

    return transformed, {
        "media_requests": media_count,
        "requested_repeat_rate": repeat_rate,
        "repeated_media_requests": repeated_count,
        "unique_media_requests": media_count - repeated_count,
        "realized_repeat_rate": (repeated_count / media_count if media_count else None),
        "media_object_policy": "fresh_decoded_objects_per_request",
        "unique_media_policy": "deterministic_binary_content_marker",
        "question_policy": (
            "deterministic_different_suffix" if vary_questions else "original_question"
        ),
    }
