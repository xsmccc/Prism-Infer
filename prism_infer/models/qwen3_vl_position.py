"""Qwen3-VL 文本侧 M-RoPE position ids 构造。

本模块自实现 Qwen3-VL 图文 prefill 的 3D position ids 和 rope_delta。
参考 HF 4.57.1:
`transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033`。

P3.2 扩展为同时支持 image/video span。视频语义与 HF 保持一致:
先按帧展开 `video_grid_thw`，再把 T 置为 1；时间信息由
processor 展开的 timestamp 文本 token 承载。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from prism_infer.models.qwen3_vl_architecture import (
    MROPE_AXIS_COUNT,
    VISION_GRID_DIMENSIONS,
    VISION_GRID_TEMPORAL_AXIS,
)


BATCHED_TOKEN_IDS_RANK = 2
VISION_GRID_MATRIX_RANK = 2
VisualKind = Literal["image", "video"]


@dataclass(slots=True)
class _VisualGridCursor:
    """Consume image/video grid rows in prompt order across a batch."""

    image_grid_thw: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    image_index: int = 0
    video_index: int = 0

    def take(self, kind: VisualKind) -> torch.Tensor:
        grid = self.image_grid_thw if kind == "image" else self.video_grid_thw
        index = self.image_index if kind == "image" else self.video_index
        rows = 0 if grid is None else grid.shape[0]
        if grid is None or index >= rows:
            raise ValueError(
                f"{kind}_grid_thw has {rows} rows, "
                f"but input_ids contains at least {index + 1} {kind} spans"
            )
        if kind == "image":
            self.image_index += 1
        else:
            self.video_index += 1
        return grid[index]

    def ensure_exhausted(self) -> None:
        self._ensure_kind_exhausted("image", self.image_grid_thw, self.image_index)
        self._ensure_kind_exhausted("video", self.video_grid_thw, self.video_index)

    @staticmethod
    def _ensure_kind_exhausted(
        kind: VisualKind,
        grid: torch.Tensor | None,
        used_rows: int,
    ) -> None:
        if grid is not None and used_rows != grid.shape[0]:
            raise ValueError(
                f"{kind}_grid_thw has {grid.shape[0]} rows, "
                f"but only {used_rows} {kind} spans were used"
            )


@dataclass(slots=True)
class _RemainingVisualSpans:
    """Track which visual special token should be consumed next."""

    images: int
    videos: int

    def pop_next(
        self,
        input_tokens: list[int],
        *,
        start: int,
        image_token_id: int,
        video_token_id: int,
    ) -> tuple[VisualKind, int]:
        image_end = _next_token_end(input_tokens, image_token_id, start, self.images)
        video_end = _next_token_end(input_tokens, video_token_id, start, self.videos)
        if image_end < video_end:
            self.images -= 1
            return "image", image_end
        if video_end <= len(input_tokens):
            self.videos -= 1
            return "video", video_end
        raise ValueError("declared visual spans cannot be located after the previous visual span")


def _next_token_end(
    input_tokens: list[int],
    token_id: int,
    start: int,
    remaining: int,
) -> int:
    if remaining <= 0 or token_id not in input_tokens:
        return len(input_tokens) + 1
    return input_tokens.index(token_id, start)


def get_qwen3_vl_rope_index(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 Qwen3-VL 文本模型使用的 3D position ids 和 rope_delta。

    input_ids: [batch, seqlen]
    image_grid_thw: [num_images, 3] or None
    video_grid_thw: [num_videos, 3] or None
    attention_mask: [batch, seqlen] or None
    返回:
      position_ids: [3, batch, seqlen]
      rope_delta: [batch, 1]
    """

    if input_ids.ndim != BATCHED_TOKEN_IDS_RANK:
        raise ValueError(f"input_ids must have shape [batch, seqlen], got {list(input_ids.shape)}")
    if (
        isinstance(spatial_merge_size, bool)
        or not isinstance(spatial_merge_size, int)
        or spatial_merge_size <= 0
    ):
        raise ValueError(
            f"spatial_merge_size must be a positive integer, got {spatial_merge_size!r}"
        )
    special_token_ids = (
        image_token_id,
        video_token_id,
        vision_start_token_id,
    )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in special_token_ids
    ):
        raise ValueError(
            f"visual special token ids must be non-negative integers: {special_token_ids}"
        )
    if len(set(special_token_ids)) != len(special_token_ids):
        raise ValueError(f"visual special token ids must be distinct: {special_token_ids}")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask shape must match input_ids, "
            f"got {list(attention_mask.shape)} vs {list(input_ids.shape)}"
        )

    if image_grid_thw is None and video_grid_thw is None:
        return _get_text_rope_index(input_ids, attention_mask)

    if image_grid_thw is not None and (
        image_grid_thw.ndim != VISION_GRID_MATRIX_RANK
        or image_grid_thw.shape[1] != VISION_GRID_DIMENSIONS
    ):
        raise ValueError(
            f"image_grid_thw must have shape [num_images, 3], got {list(image_grid_thw.shape)}"
        )
    if video_grid_thw is not None and (
        video_grid_thw.ndim != VISION_GRID_MATRIX_RANK
        or video_grid_thw.shape[1] != VISION_GRID_DIMENSIONS
    ):
        raise ValueError(
            f"video_grid_thw must have shape [num_videos, 3], got {list(video_grid_thw.shape)}"
        )

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw,
            video_grid_thw[:, VISION_GRID_TEMPORAL_AXIS],
            dim=0,
        ).clone()
        video_grid_thw[:, VISION_GRID_TEMPORAL_AXIS] = 1

    return _get_visual_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        spatial_merge_size=spatial_merge_size,
    )


def _get_text_rope_index(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """纯文本 position ids，与 HF text-only 分支一致。"""

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = (
            position_ids.unsqueeze(0).expand(MROPE_AXIS_COUNT, -1, -1).to(attention_mask.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        rope_delta = max_position_ids + 1 - attention_mask.shape[-1]
        return position_ids.to(dtype=input_ids.dtype), rope_delta.to(dtype=input_ids.dtype)

    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .view(1, 1, -1)
        .expand(MROPE_AXIS_COUNT, input_ids.shape[0], -1)
    )
    rope_delta = torch.zeros(
        [input_ids.shape[0], 1],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    return position_ids.to(dtype=input_ids.dtype), rope_delta


def _get_visual_rope_index(
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """图像/视频输入 position ids，与 HF visual 分支逐步对齐。"""

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(input_ids.device)
    image_grid_thw = _to_optional_device(image_grid_thw, input_ids.device)
    video_grid_thw = _to_optional_device(video_grid_thw, input_ids.device)
    grid_cursor = _VisualGridCursor(image_grid_thw, video_grid_thw)

    position_ids = torch.ones(
        MROPE_AXIS_COUNT,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    rope_deltas: list[torch.Tensor] = []
    for batch_idx, seq_input_ids in enumerate(input_ids):
        active_mask = attention_mask[batch_idx] == 1
        llm_positions = _build_sequence_visual_positions(
            seq_input_ids[active_mask],
            grid_cursor=grid_cursor,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            spatial_merge_size=spatial_merge_size,
        )
        _validate_position_count(llm_positions, active_mask)
        position_ids[..., batch_idx, active_mask] = llm_positions.to(position_ids.device)
        rope_deltas.append(llm_positions.max() + 1 - len(seq_input_ids))

    grid_cursor.ensure_exhausted()

    rope_delta = (
        torch.stack(rope_deltas).to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(1)
    )
    return position_ids, rope_delta


def _to_optional_device(
    tensor: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


def _visual_span_counts(
    active_input_ids: torch.Tensor,
    *,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[int, int]:
    vision_starts = torch.argwhere(active_input_ids == vision_start_token_id).squeeze(1)
    if bool(torch.any(vision_starts + 1 >= active_input_ids.numel())):
        raise ValueError("vision_start_token_id cannot be the final active token")
    vision_tokens = active_input_ids[vision_starts + 1]
    image_count = int((vision_tokens == image_token_id).sum().item())
    video_count = int((vision_tokens == video_token_id).sum().item())
    return image_count, video_count


def _build_sequence_visual_positions(
    active_input_ids: torch.Tensor,
    *,
    grid_cursor: _VisualGridCursor,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    image_count, video_count = _visual_span_counts(
        active_input_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )
    remaining = _RemainingVisualSpans(image_count, video_count)
    input_tokens = active_input_ids.tolist()
    position_parts: list[torch.Tensor] = []
    start = 0
    for _ in range(image_count + video_count):
        kind, visual_end = remaining.pop_next(
            input_tokens,
            start=start,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        merged_grid = _merge_visual_grid(grid_cursor.take(kind), spatial_merge_size)
        _append_visual_span_positions(
            position_parts,
            text_length=visual_end - start,
            merged_grid=merged_grid,
            device=active_input_ids.device,
        )
        start = visual_end + _grid_token_count(merged_grid)

    _append_trailing_text_positions(
        position_parts,
        start=start,
        total_length=len(input_tokens),
        device=active_input_ids.device,
    )
    if not position_parts:
        raise ValueError(
            "visual grid was provided but no image/video spans were found in input_ids"
        )
    return torch.cat(position_parts, dim=1).reshape(MROPE_AXIS_COUNT, -1)


def _merge_visual_grid(
    raw_grid: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[int, int, int]:
    temporal, height, width = (int(value.item()) for value in raw_grid)
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError(
            "visual grid height/width must be divisible by spatial_merge_size: "
            f"grid={[temporal, height, width]}, merge={spatial_merge_size}"
        )
    merged_grid = (
        temporal,
        height // spatial_merge_size,
        width // spatial_merge_size,
    )
    if min(merged_grid) <= 0:
        raise ValueError(
            f"invalid merged visual grid: raw={[temporal, height, width]}, "
            f"merge={spatial_merge_size}"
        )
    return merged_grid


def _grid_token_count(grid: tuple[int, int, int]) -> int:
    temporal, height, width = grid
    return temporal * height * width


def _next_position_start(position_parts: list[torch.Tensor]) -> torch.Tensor | int:
    return position_parts[-1].max() + 1 if position_parts else 0


def _text_positions(
    length: int,
    *,
    start_position: torch.Tensor | int,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.arange(length, device=device).view(1, -1).expand(MROPE_AXIS_COUNT, -1)
        + start_position
    )


def _visual_positions(
    grid: tuple[int, int, int],
    *,
    start_position: torch.Tensor | int,
    device: torch.device,
) -> torch.Tensor:
    temporal, height, width = grid
    temporal_index = (
        torch.arange(temporal, device=device).view(-1, 1).expand(-1, height * width).flatten()
    )
    height_index = (
        torch.arange(height, device=device).view(1, -1, 1).expand(temporal, -1, width).flatten()
    )
    width_index = (
        torch.arange(width, device=device).view(1, 1, -1).expand(temporal, height, -1).flatten()
    )
    return torch.stack([temporal_index, height_index, width_index]) + start_position


def _append_visual_span_positions(
    position_parts: list[torch.Tensor],
    *,
    text_length: int,
    merged_grid: tuple[int, int, int],
    device: torch.device,
) -> None:
    start_position = _next_position_start(position_parts)
    position_parts.append(
        _text_positions(text_length, start_position=start_position, device=device)
    )
    position_parts.append(
        _visual_positions(
            merged_grid,
            start_position=start_position + text_length,
            device=device,
        )
    )


def _append_trailing_text_positions(
    position_parts: list[torch.Tensor],
    *,
    start: int,
    total_length: int,
    device: torch.device,
) -> None:
    if start >= total_length:
        return
    position_parts.append(
        _text_positions(
            total_length - start,
            start_position=_next_position_start(position_parts),
            device=device,
        )
    )


def _validate_position_count(
    positions: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    active_tokens = int(active_mask.sum().item())
    if positions.shape[1] != active_tokens:
        raise ValueError(
            f"position length mismatch: positions={positions.shape[1]}, "
            f"active_tokens={active_tokens}"
        )


def get_qwen3_vl_rope_index_from_config(
    input_ids: torch.Tensor,
    *,
    config,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 HF config-like 对象读取 token ids 和 merge size 后生成 rope index。"""

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise ValueError("config does not expose vision_config")
    return get_qwen3_vl_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        image_token_id=int(getattr(config, "image_token_id")),
        video_token_id=int(getattr(config, "video_token_id")),
        vision_start_token_id=int(getattr(config, "vision_start_token_id")),
        spatial_merge_size=int(getattr(vision_config, "spatial_merge_size")),
    )
