"""P3.2 Processor pipeline: 视频输入边界验证。"""

from dataclasses import replace

import pytest
import torch
from PIL import Image

from conftest import get_model_path, require_transformers
from prism_infer.engine.vl_inputs import (
    _processor_video_metadata,
    build_video_prompt,
    prepare_video_inputs,
    validate_video_inputs,
)


pytestmark = [pytest.mark.model, pytest.mark.integration]


def _load_processor():
    transformers = require_transformers()
    return transformers.AutoProcessor.from_pretrained(
        get_model_path(),
        trust_remote_code=True,
        local_files_only=True,
    )


def demo_video_frames() -> list[Image.Image]:
    """构造本地可复现的 4 帧 synthetic video。"""

    return [Image.new("RGB", (448, 448), color=(80 + i * 30, 120, 180)) for i in range(4)]


def test_video_metadata_rejects_index_equal_to_source_frame_count() -> None:
    with pytest.raises(ValueError, match="outside the source video"):
        _processor_video_metadata(
            [object()],
            {
                "fps": 24.0,
                "source_frame_count": 4,
                "sampled_indices": [4],
            },
        )


def test_video_processor_pipeline_matches_hf_reference():
    """视频 processor 边界必须保持 HF reference 输出。"""

    processor = _load_processor()
    frames = demo_video_frames()
    prompt = "Describe this video."

    ours = prepare_video_inputs(processor, prompt, frames)
    reference_prompt = build_video_prompt(processor, prompt, frames)
    reference = processor(
        text=reference_prompt,
        videos=[frames],
        return_tensors="pt",
        videos_kwargs={"do_sample_frames": False},
    )

    input_ids_equal = torch.equal(ours.input_ids, reference["input_ids"])
    attention_equal = torch.equal(ours.attention_mask, reference["attention_mask"])
    grid_equal = torch.equal(ours.video_grid_thw, reference["video_grid_thw"])
    pixel_max_diff = (
        (ours.pixel_values_videos - reference["pixel_values_videos"]).abs().max().item()
    )

    print(f"video input_ids shape: {list(ours.input_ids.shape)}")
    print(f"video pixel_values_videos shape: {list(ours.pixel_values_videos.shape)}")
    print(f"video_grid_thw shape: {list(ours.video_grid_thw.shape)}")
    print(f"video_grid_thw: {ours.video_grid_thw.tolist()}")
    print(f"video tokens: {ours.video_token_count} / expected {ours.expected_video_tokens}")
    print(f"video pixel_values max diff: {pixel_max_diff:.6e}")

    assert ours.prompt_text == reference_prompt
    assert input_ids_equal
    assert attention_equal
    assert grid_equal
    assert pixel_max_diff == 0.0
    assert list(ours.video_grid_thw.shape) == [1, 3]
    assert ours.video_token_count == ours.expected_video_tokens
    print("video processor pipeline: PASS")


def test_video_processor_preserves_preselected_frames_and_source_timestamps():
    processor = _load_processor()
    frames = [Image.new("RGB", (480, 320), color=(index, 40, 80)) for index in range(16)]
    sampled_indices = [
        3,
        11,
        19,
        27,
        35,
        43,
        51,
        59,
        67,
        74,
        82,
        90,
        98,
        106,
        114,
        122,
    ]

    inputs = prepare_video_inputs(
        processor,
        "Describe this video.",
        frames,
        video_metadata={
            "fps": 25.0,
            "source_frame_count": 128,
            "sampled_indices": sampled_indices,
        },
    )
    decoded_prompt = processor.tokenizer.decode(inputs.token_ids)

    assert inputs.video_grid_thw[0, 0].item() == 8
    assert "<0.3 seconds>" in decoded_prompt
    assert "<4.7 seconds>" in decoded_prompt


def test_video_processor_rejects_video_token_mismatch():
    """视频视觉占位 token 数和 grid 推导数量不一致时必须显式报错。"""

    processor = _load_processor()
    inputs = prepare_video_inputs(processor, "Describe this video.", demo_video_frames())
    corrupted = inputs.input_ids.clone()
    video_positions = torch.nonzero(corrupted == inputs.video_token_id, as_tuple=False)
    assert video_positions.numel() > 0
    corrupted[0, video_positions[0, 1]] = 0

    bad_inputs = replace(inputs, input_ids=corrupted)
    merge_size = int(processor.video_processor.merge_size)
    with pytest.raises(ValueError, match="video token count mismatch"):
        validate_video_inputs(bad_inputs, merge_size)
    print("video mismatch rejection: PASS")
