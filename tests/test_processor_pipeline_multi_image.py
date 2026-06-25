"""P3.1 Processor pipeline: 多图图文输入边界验证。"""

from dataclasses import replace

import pytest
import torch
from PIL import Image

from conftest import get_model_path, require_transformers
from prism_infer.engine.vl_inputs import (
    build_image_prompt,
    prepare_image_inputs,
    validate_image_inputs,
)


def _load_processor():
    transformers = require_transformers()
    return transformers.AutoProcessor.from_pretrained(
        get_model_path(),
        trust_remote_code=True,
        local_files_only=True,
    )


def _demo_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (448, 448), color=(100, 150, 200)),
        Image.new("RGB", (448, 448), color=(200, 120, 80)),
    ]


def test_multi_image_processor_pipeline_matches_hf_reference():
    """多图 processor 边界必须保持 HF reference 输出。"""

    processor = _load_processor()
    images = _demo_images()
    prompt = "Compare these images."

    ours = prepare_image_inputs(processor, prompt, images)
    reference_prompt = build_image_prompt(processor, prompt, images)
    reference = processor(text=reference_prompt, images=images, return_tensors="pt")

    input_ids_equal = torch.equal(ours.input_ids, reference["input_ids"])
    attention_equal = torch.equal(ours.attention_mask, reference["attention_mask"])
    grid_equal = torch.equal(ours.image_grid_thw, reference["image_grid_thw"])
    pixel_max_diff = (ours.pixel_values - reference["pixel_values"]).abs().max().item()

    print(f"multi input_ids shape: {list(ours.input_ids.shape)}")
    print(f"multi pixel_values shape: {list(ours.pixel_values.shape)}")
    print(f"multi image_grid_thw shape: {list(ours.image_grid_thw.shape)}")
    print(f"multi image_grid_thw: {ours.image_grid_thw.tolist()}")
    print(f"multi image tokens: {ours.image_token_count} / expected {ours.expected_image_tokens}")
    print(f"multi pixel_values max diff: {pixel_max_diff:.6e}")

    assert ours.prompt_text == reference_prompt
    assert input_ids_equal
    assert attention_equal
    assert grid_equal
    assert pixel_max_diff == 0.0
    assert list(ours.image_grid_thw.shape) == [2, 3]
    assert ours.image_token_count == ours.expected_image_tokens
    print("multi image processor pipeline: PASS")


def test_multi_image_processor_rejects_image_token_mismatch():
    """多图视觉占位 token 数和 grid 推导数量不一致时必须显式报错。"""

    processor = _load_processor()
    inputs = prepare_image_inputs(processor, "Compare these images.", _demo_images())
    corrupted = inputs.input_ids.clone()
    image_positions = torch.nonzero(corrupted == inputs.image_token_id, as_tuple=False)
    assert image_positions.numel() > 0
    corrupted[0, image_positions[0, 1]] = 0

    bad_inputs = replace(inputs, input_ids=corrupted)
    merge_size = int(processor.image_processor.merge_size)
    with pytest.raises(ValueError, match="image token count mismatch"):
        validate_image_inputs(bad_inputs, merge_size)
    print("multi image mismatch rejection: PASS")
