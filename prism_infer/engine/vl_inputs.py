"""Qwen3-VL 单图输入预处理边界。

本模块允许使用 Hugging Face processor 作为非核心工具，原因是图像
resize/patch packing/chat template/tokenizer 属于成熟预处理基础设施，
不是 Prism-Infer 的核心研究模块。核心模型、M-RoPE、attention、KV cache
和压缩逻辑仍由 Prism-Infer 自实现。

参考:
- HF Qwen3VLProcessor.__call__ 返回 input_ids/pixel_values/image_grid_thw:
  transformers/models/qwen3_vl/processing_qwen3_vl.py:146-194
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SingleImageInputs:
    """单图请求预处理结果。

    input_ids: [1, seqlen]
    attention_mask: [1, seqlen]
    pixel_values: [num_raw_vision_patches, patch_dim]
    image_grid_thw: [1, 3], 每行是 [T, H, W]
    image_token_id: `<|image_pad|>` 的 token id
    image_token_count: input_ids 中视觉占位 token 数
    expected_image_tokens: image_grid_thw.prod() // merge_size**2
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


def load_vl_processor(model_path: str) -> Any:
    """从本地模型目录加载 Qwen3-VL processor。

    transformers 是可选运行时依赖，因此延迟导入，避免普通模型单元测试
    在不需要 processor 时被第三方依赖阻塞。
    """

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for Qwen3-VL processor preprocessing"
        ) from exc

    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def build_single_image_prompt(
    processor: Any,
    prompt: str,
    image: Any,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """构造单图 chat prompt。

    image 参数只用于满足 processor 的 chat template 消息格式；实际图像
    像素处理在 prepare_single_image_inputs 中完成。
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def prepare_single_image_inputs(
    processor: Any,
    prompt: str,
    image: Any,
    *,
    add_generation_prompt: bool = True,
) -> SingleImageInputs:
    """把单图 prompt 预处理为 Prism-Infer engine 可消费的数据。

    当前仅支持单请求单图。多图、视频和 batch 混合输入不在 P2.1 范围内，
    后续阶段必须显式扩展，不允许在这里静默吞掉 unsupported state。
    """

    prompt_text = build_single_image_prompt(
        processor,
        prompt,
        image,
        add_generation_prompt=add_generation_prompt,
    )
    batch = processor(text=prompt_text, images=[image], return_tensors="pt")

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

    result = SingleImageInputs(
        input_ids=batch["input_ids"].contiguous(),
        attention_mask=batch["attention_mask"].contiguous(),
        pixel_values=batch["pixel_values"].contiguous(),
        image_grid_thw=batch["image_grid_thw"].contiguous(),
        image_token_id=int(image_token_id),
        image_token_count=0,
        expected_image_tokens=0,
        prompt_text=prompt_text,
    )
    return validate_single_image_inputs(result, int(merge_size))


def validate_single_image_inputs(
    inputs: SingleImageInputs,
    merge_size: int,
) -> SingleImageInputs:
    """校验单图 processor 输出的 shape 和视觉 token 数量。

    返回新的 SingleImageInputs，补齐 image_token_count 和
    expected_image_tokens。校验失败时显式报错，禁止 silent fallback。
    """

    if merge_size <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    if inputs.input_ids.ndim != 2 or inputs.input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, seqlen], got {list(inputs.input_ids.shape)}")
    if inputs.attention_mask.shape != inputs.input_ids.shape:
        raise ValueError(
            "attention_mask shape must match input_ids, "
            f"got {list(inputs.attention_mask.shape)} vs {list(inputs.input_ids.shape)}"
        )
    if inputs.pixel_values.ndim != 2:
        raise ValueError(
            f"pixel_values must have shape [num_patches, patch_dim], got {list(inputs.pixel_values.shape)}"
        )
    if inputs.image_grid_thw.shape != (1, 3):
        raise ValueError(
            f"image_grid_thw must have shape [1, 3], got {list(inputs.image_grid_thw.shape)}"
        )

    raw_patch_count = int(inputs.image_grid_thw.prod().item())
    if raw_patch_count != inputs.pixel_values.shape[0]:
        raise ValueError(
            "pixel_values patch count does not match image_grid_thw: "
            f"{inputs.pixel_values.shape[0]} vs {raw_patch_count}"
        )

    merge_area = merge_size * merge_size
    if raw_patch_count % merge_area != 0:
        raise ValueError(
            f"image_grid_thw product {raw_patch_count} is not divisible by merge_size^2 {merge_area}"
        )

    expected_image_tokens = raw_patch_count // merge_area
    image_token_count = int((inputs.input_ids == inputs.image_token_id).sum().item())
    if image_token_count != expected_image_tokens:
        raise ValueError(
            "image token count mismatch: "
            f"input_ids has {image_token_count}, expected {expected_image_tokens} "
            f"from image_grid_thw={inputs.image_grid_thw.tolist()} and merge_size={merge_size}"
        )

    return SingleImageInputs(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        image_token_id=inputs.image_token_id,
        image_token_count=image_token_count,
        expected_image_tokens=expected_image_tokens,
        prompt_text=inputs.prompt_text,
    )
