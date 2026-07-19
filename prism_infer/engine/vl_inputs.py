"""Qwen3-VL 视觉输入预处理边界。

本模块允许使用 Hugging Face processor 作为非核心工具，原因是图像
resize/patch packing/chat template/tokenizer 属于成熟预处理基础设施，
不是 Prism-Infer 的核心研究模块。核心模型、M-RoPE、attention、KV cache
和压缩逻辑仍由 Prism-Infer 自实现。

参考:
- HF Qwen3VLProcessor.__call__ 返回 input_ids/pixel_values/image_grid_thw:
  transformers/models/qwen3_vl/processing_qwen3_vl.py:146-194
- HF processor 对多图 image token 占位展开:
  transformers/models/qwen3_vl/processing_qwen3_vl.py:186-194
- HF processor 返回 pixel_values_videos/video_grid_thw 并展开视频 timestamp:
  transformers/models/qwen3_vl/processing_qwen3_vl.py:196-234
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from prism_infer.models.qwen3_vl_architecture import VISION_GRID_DIMENSIONS


PROCESSOR_TOKEN_MATRIX_RANK = 2
PROCESSOR_PATCH_MATRIX_RANK = 2
VISION_GRID_MATRIX_RANK = 2
SUPPORTED_PROCESSOR_BATCH_SIZE = 1


@dataclass(frozen=True)
class ImageInputs:
    """图像请求预处理结果。

    input_ids: [1, seqlen]
    attention_mask: [1, seqlen]
    pixel_values: [num_raw_vision_patches, patch_dim]
    image_grid_thw: [num_images, 3], 每行是 [T, H, W]
    image_token_id: `<|image_pad|>` 的 token id
    image_token_count: input_ids 中视觉占位 token 数
    expected_image_tokens: sum(image_grid_thw.prod(dim=1) // merge_size**2)
    prompt_text: processor chat template 展开后的文本
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_id: int
    image_token_count: int
    expected_image_tokens: int
    prompt_text: str

    @property
    def token_ids(self) -> list[int]:
        """返回单 batch token ids，供后续 Sequence 构造使用。"""
        return self.input_ids[0].tolist()


@dataclass(frozen=True)
class VideoInputs:
    """视频请求预处理结果。

    input_ids: [1, seqlen]
    attention_mask: [1, seqlen]
    pixel_values_videos: [num_raw_video_patches, patch_dim]
    video_grid_thw: [num_videos, 3], 每行是 [T, H, W]
    video_token_id: `<|video_pad|>` 的 token id
    video_token_count: input_ids 中视频占位 token 数
    expected_video_tokens: sum(video_grid_thw.prod(dim=1) // merge_size**2)
    prompt_text: processor chat template 展开后的文本
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    video_token_id: int
    video_token_count: int
    expected_video_tokens: int
    prompt_text: str

    @property
    def token_ids(self) -> list[int]:
        """返回单 batch token ids，供后续 Sequence 构造使用。"""
        return self.input_ids[0].tolist()


SingleImageInputs = ImageInputs


def load_vl_processor(
    model_path: str,
    *,
    image_max_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> Any:
    """从本地模型目录加载 Qwen3-VL processor。

    transformers 是可选运行时依赖，因此延迟导入，避免普通模型单元测试
    在不需要 processor 时被第三方依赖阻塞。
    """

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for Qwen3-VL processor preprocessing") from exc

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    for name, value, component_name in (
        ("image_max_pixels", image_max_pixels, "image_processor"),
        ("video_max_pixels", video_max_pixels, "video_processor"),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
        component = getattr(processor, component_name, None)
        size = getattr(component, "size", None)
        if size is None or not hasattr(size, "longest_edge"):
            raise ValueError(f"processor.{component_name}.size has no longest_edge budget")
        size.longest_edge = value
    return processor


def _normalize_images(images: Any | list[Any] | tuple[Any, ...]) -> list[Any]:
    """把单图或多图参数统一为非空 list。"""

    if isinstance(images, (list, tuple)):
        normalized = list(images)
    else:
        normalized = [images]
    if not normalized:
        raise ValueError("images must contain at least one image")
    return normalized


def _normalize_video(video: Any | list[Any] | tuple[Any, ...]) -> Any:
    """把视频参数规范为 processor 的单个 video 对象。

    P3.2 先支持一条请求包含一个视频。视频通常是帧列表
    `list[PIL.Image.Image]`，传给 HF processor 时再包装为 `[video]`。
    """

    if isinstance(video, (list, tuple)) and len(video) == 0:
        raise ValueError("video must contain at least one frame")
    return list(video) if isinstance(video, tuple) else video


def _processor_video_metadata(
    video: Any,
    metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Map audited source-frame identity to the HF processor metadata contract."""

    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise TypeError("video_metadata must be a mapping or None")
    required = ("fps", "source_frame_count", "sampled_indices")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"video_metadata missing required fields: {missing}")

    fps = metadata["fps"]
    source_frame_count = metadata["source_frame_count"]
    sampled_indices = metadata["sampled_indices"]
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("video_metadata.fps must be finite and positive")
    if (
        isinstance(source_frame_count, bool)
        or not isinstance(source_frame_count, int)
        or source_frame_count <= 0
    ):
        raise ValueError("video_metadata.source_frame_count must be positive")
    if not isinstance(sampled_indices, list) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= source_frame_count
        for index in sampled_indices
    ):
        raise ValueError("video_metadata.sampled_indices are outside the source video")
    try:
        provided_frames = len(video)
    except TypeError as exc:
        raise TypeError(
            "video_metadata requires an in-memory sequence of preselected frames"
        ) from exc
    if len(sampled_indices) != provided_frames:
        raise ValueError(
            "video_metadata sampled index count must match provided frames: "
            f"{len(sampled_indices)} != {provided_frames}"
        )
    return [
        {
            "total_num_frames": source_frame_count,
            "fps": float(fps),
            "frames_indices": list(sampled_indices),
        }
    ]


def build_image_prompt(
    processor: Any,
    prompt: str,
    images: Any | list[Any] | tuple[Any, ...],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """构造图像 chat prompt。

    images 参数只用于满足 processor 的 chat template 消息格式；实际图像
    像素处理在 prepare_image_inputs 中完成。
    """

    image_items = [{"type": "image", "image": image} for image in _normalize_images(images)]
    messages = [
        {
            "role": "user",
            "content": image_items + [{"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_interleaved_image_prompt(
    processor: Any,
    prompt: str,
    images: Any | list[Any] | tuple[Any, ...],
    *,
    image_marker: str = "<image>",
    add_generation_prompt: bool = True,
) -> str:
    """按文本 marker 的位置构造交错多图 chat prompt。

    该显式 API 用于 MuirBench 一类“图片出现在问题或选项中”的输入。普通
    ``build_image_prompt`` 仍保持 all-images-first 语义，避免改变既有调用方。
    """

    normalized_images = _normalize_images(images)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("interleaved image prompt must be a non-empty string")
    if not isinstance(image_marker, str) or not image_marker:
        raise ValueError("image_marker must be a non-empty string")
    segments = prompt.split(image_marker)
    marker_count = len(segments) - 1
    if marker_count != len(normalized_images):
        raise ValueError(
            "interleaved image marker count must equal image count: "
            f"markers={marker_count}, images={len(normalized_images)}"
        )
    content = []
    for index, image in enumerate(normalized_images):
        if segments[index]:
            content.append({"type": "text", "text": segments[index]})
        content.append({"type": "image", "image": image})
    if segments[-1]:
        content.append({"type": "text", "text": segments[-1]})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_video_prompt(
    processor: Any,
    prompt: str,
    video: Any | list[Any] | tuple[Any, ...],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """构造视频 chat prompt。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": _normalize_video(video)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_single_image_prompt(
    processor: Any,
    prompt: str,
    image: Any,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """构造单图 chat prompt，兼容 P2 API。"""

    return build_image_prompt(
        processor,
        prompt,
        image,
        add_generation_prompt=add_generation_prompt,
    )


def _prepare_image_inputs_from_prompt_text(
    processor: Any,
    prompt_text: str,
    normalized_images: list[Any],
) -> ImageInputs:
    """执行普通/交错 image prompt 共用的 processor 与 shape 校验。"""

    batch = processor(text=prompt_text, images=normalized_images, return_tensors="pt")

    required_keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise ValueError(f"processor output missing required keys: {missing}")

    image_token = getattr(processor, "image_token", None)
    if image_token is None:
        raise ValueError("processor does not expose image_token")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None or image_token_id < 0:
        raise ValueError(f"invalid image token id for token {image_token!r}")

    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", None)
    if merge_size is None:
        raise ValueError("processor.image_processor does not expose merge_size")

    result = ImageInputs(
        input_ids=batch["input_ids"].contiguous(),
        attention_mask=batch["attention_mask"].contiguous(),
        pixel_values=batch["pixel_values"].contiguous(),
        image_grid_thw=batch["image_grid_thw"].contiguous(),
        image_token_id=int(image_token_id),
        image_token_count=0,
        expected_image_tokens=0,
        prompt_text=prompt_text,
    )
    return validate_image_inputs(result, int(merge_size))


def prepare_image_inputs(
    processor: Any,
    prompt: str,
    images: Any | list[Any] | tuple[Any, ...],
    *,
    add_generation_prompt: bool = True,
) -> ImageInputs:
    """把 all-images-first prompt 预处理为 engine 可消费的数据。"""

    normalized_images = _normalize_images(images)
    prompt_text = build_image_prompt(
        processor,
        prompt,
        normalized_images,
        add_generation_prompt=add_generation_prompt,
    )
    return _prepare_image_inputs_from_prompt_text(
        processor,
        prompt_text,
        normalized_images,
    )


def prepare_interleaved_image_inputs(
    processor: Any,
    prompt: str,
    images: Any | list[Any] | tuple[Any, ...],
    *,
    image_marker: str = "<image>",
    add_generation_prompt: bool = True,
) -> ImageInputs:
    """把 marker-interleaved prompt 预处理为 engine 可消费的数据。"""

    normalized_images = _normalize_images(images)
    prompt_text = build_interleaved_image_prompt(
        processor,
        prompt,
        normalized_images,
        image_marker=image_marker,
        add_generation_prompt=add_generation_prompt,
    )
    return _prepare_image_inputs_from_prompt_text(
        processor,
        prompt_text,
        normalized_images,
    )


def prepare_single_image_inputs(
    processor: Any,
    prompt: str,
    image: Any,
    *,
    add_generation_prompt: bool = True,
) -> SingleImageInputs:
    """把单图 prompt 预处理为 Prism-Infer engine 可消费的数据。

    兼容 P2 API。多图请使用 prepare_image_inputs。
    """

    result = prepare_image_inputs(
        processor,
        prompt,
        image,
        add_generation_prompt=add_generation_prompt,
    )
    if result.image_grid_thw.shape[0] != 1:
        raise ValueError(
            "prepare_single_image_inputs expected exactly one image, "
            f"got image_grid_thw shape {list(result.image_grid_thw.shape)}"
        )
    return result


def prepare_video_inputs(
    processor: Any,
    prompt: str,
    video: Any | list[Any] | tuple[Any, ...],
    *,
    video_metadata: Mapping[str, Any] | None = None,
    add_generation_prompt: bool = True,
) -> VideoInputs:
    """把 prompt + 一个视频预处理为 engine 可消费的数据。

    视频 processor 仍只作为非核心预处理边界；视频 position ids、
    VisionEncoder、LLM forward 和 KV cache 由 Prism-Infer 自实现。
    """

    normalized_video = _normalize_video(video)
    prompt_text = build_video_prompt(
        processor,
        prompt,
        normalized_video,
        add_generation_prompt=add_generation_prompt,
    )
    processor_metadata = _processor_video_metadata(
        normalized_video,
        video_metadata,
    )
    videos_kwargs: dict[str, Any] = {}
    # A list/tuple is already a caller-selected frame sequence.  Preserve it
    # exactly; path/URL/array inputs without source metadata retain the HF
    # processor's configured sampling policy.
    if isinstance(normalized_video, list) or processor_metadata is not None:
        videos_kwargs["do_sample_frames"] = False
    if processor_metadata is not None:
        videos_kwargs["video_metadata"] = processor_metadata
    batch = processor(
        text=prompt_text,
        videos=[normalized_video],
        return_tensors="pt",
        videos_kwargs=videos_kwargs,
    )

    required_keys = (
        "input_ids",
        "attention_mask",
        "pixel_values_videos",
        "video_grid_thw",
    )
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise ValueError(f"processor output missing required video keys: {missing}")

    video_token = getattr(processor, "video_token", None)
    if video_token is None:
        raise ValueError("processor does not expose video_token")
    video_token_id = processor.tokenizer.convert_tokens_to_ids(video_token)
    if video_token_id is None or video_token_id < 0:
        raise ValueError(f"invalid video token id for token {video_token!r}")

    merge_size = getattr(getattr(processor, "video_processor", None), "merge_size", None)
    if merge_size is None:
        raise ValueError("processor.video_processor does not expose merge_size")

    result = VideoInputs(
        input_ids=batch["input_ids"].contiguous(),
        attention_mask=batch["attention_mask"].contiguous(),
        pixel_values_videos=batch["pixel_values_videos"].contiguous(),
        video_grid_thw=batch["video_grid_thw"].contiguous(),
        video_token_id=int(video_token_id),
        video_token_count=0,
        expected_video_tokens=0,
        prompt_text=prompt_text,
    )
    return validate_video_inputs(result, int(merge_size))


def validate_image_inputs(
    inputs: ImageInputs,
    merge_size: int,
) -> ImageInputs:
    """校验图像 processor 输出的 shape 和视觉 token 数量。

    返回新的 ImageInputs，补齐 image_token_count 和
    expected_image_tokens。校验失败时显式报错，禁止 silent fallback。
    """

    if merge_size <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    if (
        inputs.input_ids.ndim != PROCESSOR_TOKEN_MATRIX_RANK
        or inputs.input_ids.shape[0] != SUPPORTED_PROCESSOR_BATCH_SIZE
    ):
        raise ValueError(
            f"input_ids must have shape [1, seqlen], got {list(inputs.input_ids.shape)}"
        )
    if inputs.attention_mask.shape != inputs.input_ids.shape:
        raise ValueError(
            "attention_mask shape must match input_ids, "
            f"got {list(inputs.attention_mask.shape)} vs {list(inputs.input_ids.shape)}"
        )
    if inputs.pixel_values.ndim != PROCESSOR_PATCH_MATRIX_RANK:
        raise ValueError(
            f"pixel_values must have shape [num_patches, patch_dim], got {list(inputs.pixel_values.shape)}"
        )
    if (
        inputs.image_grid_thw.ndim != VISION_GRID_MATRIX_RANK
        or inputs.image_grid_thw.shape[1] != VISION_GRID_DIMENSIONS
    ):
        raise ValueError(
            "image_grid_thw must have shape [num_images, 3], "
            f"got {list(inputs.image_grid_thw.shape)}"
        )
    if inputs.image_grid_thw.shape[0] == 0:
        raise ValueError("image_grid_thw must contain at least one image row")

    per_image_raw_patches = inputs.image_grid_thw.prod(dim=1)
    raw_patch_count = int(per_image_raw_patches.sum().item())
    if raw_patch_count != inputs.pixel_values.shape[0]:
        raise ValueError(
            "pixel_values patch count does not match image_grid_thw: "
            f"{inputs.pixel_values.shape[0]} vs {raw_patch_count}"
        )

    merge_area = merge_size * merge_size
    if (per_image_raw_patches % merge_area != 0).any():
        raise ValueError(
            "some image_grid_thw products are not divisible by merge_size^2 "
            f"{merge_area}: {per_image_raw_patches.tolist()}"
        )

    expected_image_tokens = int((per_image_raw_patches // merge_area).sum().item())
    image_token_count = int((inputs.input_ids == inputs.image_token_id).sum().item())
    if image_token_count != expected_image_tokens:
        raise ValueError(
            "image token count mismatch: "
            f"input_ids has {image_token_count}, expected {expected_image_tokens} "
            f"from image_grid_thw={inputs.image_grid_thw.tolist()} and merge_size={merge_size}"
        )

    return ImageInputs(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        image_token_id=inputs.image_token_id,
        image_token_count=image_token_count,
        expected_image_tokens=expected_image_tokens,
        prompt_text=inputs.prompt_text,
    )


def validate_video_inputs(
    inputs: VideoInputs,
    merge_size: int,
) -> VideoInputs:
    """校验视频 processor 输出的 shape 和视频 token 数量。

    校验失败时显式报错，禁止把 video 当作 image-only fallback。
    """

    if merge_size <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    if (
        inputs.input_ids.ndim != PROCESSOR_TOKEN_MATRIX_RANK
        or inputs.input_ids.shape[0] != SUPPORTED_PROCESSOR_BATCH_SIZE
    ):
        raise ValueError(
            f"input_ids must have shape [1, seqlen], got {list(inputs.input_ids.shape)}"
        )
    if inputs.attention_mask.shape != inputs.input_ids.shape:
        raise ValueError(
            "attention_mask shape must match input_ids, "
            f"got {list(inputs.attention_mask.shape)} vs {list(inputs.input_ids.shape)}"
        )
    if inputs.pixel_values_videos.ndim != PROCESSOR_PATCH_MATRIX_RANK:
        raise ValueError(
            "pixel_values_videos must have shape [num_patches, patch_dim], "
            f"got {list(inputs.pixel_values_videos.shape)}"
        )
    if (
        inputs.video_grid_thw.ndim != VISION_GRID_MATRIX_RANK
        or inputs.video_grid_thw.shape[1] != VISION_GRID_DIMENSIONS
    ):
        raise ValueError(
            "video_grid_thw must have shape [num_videos, 3], "
            f"got {list(inputs.video_grid_thw.shape)}"
        )
    if inputs.video_grid_thw.shape[0] == 0:
        raise ValueError("video_grid_thw must contain at least one video row")

    per_video_raw_patches = inputs.video_grid_thw.prod(dim=1)
    raw_patch_count = int(per_video_raw_patches.sum().item())
    if raw_patch_count != inputs.pixel_values_videos.shape[0]:
        raise ValueError(
            "pixel_values_videos patch count does not match video_grid_thw: "
            f"{inputs.pixel_values_videos.shape[0]} vs {raw_patch_count}"
        )

    merge_area = merge_size * merge_size
    if (per_video_raw_patches % merge_area != 0).any():
        raise ValueError(
            "some video_grid_thw products are not divisible by merge_size^2 "
            f"{merge_area}: {per_video_raw_patches.tolist()}"
        )

    expected_video_tokens = int((per_video_raw_patches // merge_area).sum().item())
    video_token_count = int((inputs.input_ids == inputs.video_token_id).sum().item())
    if video_token_count != expected_video_tokens:
        raise ValueError(
            "video token count mismatch: "
            f"input_ids has {video_token_count}, expected {expected_video_tokens} "
            f"from video_grid_thw={inputs.video_grid_thw.tolist()} and merge_size={merge_size}"
        )

    return VideoInputs(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values_videos=inputs.pixel_values_videos,
        video_grid_thw=inputs.video_grid_thw,
        video_token_id=inputs.video_token_id,
        video_token_count=video_token_count,
        expected_video_tokens=expected_video_tokens,
        prompt_text=inputs.prompt_text,
    )


def validate_single_image_inputs(
    inputs: SingleImageInputs,
    merge_size: int,
) -> SingleImageInputs:
    """校验单图 processor 输出，兼容 P2 测试。"""

    result = validate_image_inputs(inputs, merge_size)
    if result.image_grid_thw.shape[0] != 1:
        raise ValueError(
            "single image inputs must have image_grid_thw shape [1, 3], "
            f"got {list(result.image_grid_thw.shape)}"
        )
    return result
