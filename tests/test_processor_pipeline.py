"""P2.1 Processor pipeline: 单图图文输入边界验证。"""

from dataclasses import replace

import pytest
import torch
from PIL import Image

from conftest import get_model_path, require_transformers
from prism_infer.engine.vl_inputs import (
    build_single_image_prompt,
    prepare_single_image_inputs,
    validate_single_image_inputs,
)


def _load_processor():
    transformers = require_transformers()
    return transformers.AutoProcessor.from_pretrained(
        get_model_path(),
        trust_remote_code=True,
        local_files_only=True,
    )


def _demo_image() -> Image.Image:
    return Image.new("RGB", (448, 448), color=(100, 150, 200))


def test_processor_pipeline_matches_hf_reference():
    """验证 Prism-Infer processor 边界不改变 HF 参考输出。"""

    processor = _load_processor()
    image = _demo_image()
    prompt = "Describe this image."

    ours = prepare_single_image_inputs(processor, prompt, image)
    reference_prompt = build_single_image_prompt(processor, prompt, image)
    reference = processor(text=reference_prompt, images=[image], return_tensors="pt")

    input_ids_equal = torch.equal(ours.input_ids, reference["input_ids"])
    attention_equal = torch.equal(ours.attention_mask, reference["attention_mask"])
    grid_equal = torch.equal(ours.image_grid_thw, reference["image_grid_thw"])
    pixel_max_diff = (ours.pixel_values - reference["pixel_values"]).abs().max().item()

    print(f"input_ids shape: {list(ours.input_ids.shape)}")
    print(f"pixel_values shape: {list(ours.pixel_values.shape)}")
    print(f"image_grid_thw shape: {list(ours.image_grid_thw.shape)}")
    print(f"image tokens: {ours.image_token_count} / expected {ours.expected_image_tokens}")
    print(f"pixel_values max diff: {pixel_max_diff:.6e}")

    assert ours.prompt_text == reference_prompt
    assert input_ids_equal
    assert attention_equal
    assert grid_equal
    assert pixel_max_diff == 0.0
    assert ours.image_token_count == ours.expected_image_tokens
    print("processor pipeline: PASS")


def test_processor_pipeline_token_ids_property():
    """验证 token_ids 属性可直接供后续 Sequence 构造使用。"""

    processor = _load_processor()
    inputs = prepare_single_image_inputs(processor, "What is in the image?", _demo_image())

    assert inputs.token_ids == inputs.input_ids[0].tolist()
    assert len(inputs.token_ids) == inputs.input_ids.shape[1]
    print(f"token_ids length: {len(inputs.token_ids)}")
    print("token_ids property: PASS")


def test_processor_pipeline_rejects_image_token_mismatch():
    """视觉占位 token 数和 grid 推导数量不一致时必须显式报错。"""

    processor = _load_processor()
    inputs = prepare_single_image_inputs(processor, "Describe this image.", _demo_image())
    corrupted = inputs.input_ids.clone()
    image_positions = torch.nonzero(corrupted == inputs.image_token_id, as_tuple=False)
    assert image_positions.numel() > 0
    corrupted[0, image_positions[0, 1]] = 0

    bad_inputs = replace(inputs, input_ids=corrupted)
    merge_size = int(processor.image_processor.merge_size)
    with pytest.raises(ValueError, match="image token count mismatch"):
        validate_single_image_inputs(bad_inputs, merge_size)
    print("mismatch rejection: PASS")
