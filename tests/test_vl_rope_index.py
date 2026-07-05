"""P2.3 Qwen3-VL 单图 3D position ids 验证。"""

import torch
from PIL import Image

from conftest import get_model_path, hf_qwen3_vl_rope_index, require_transformers
from prism_infer.engine.vl_inputs import prepare_single_image_inputs
from prism_infer.models.qwen3_vl_position import (
    get_qwen3_vl_rope_index,
    get_qwen3_vl_rope_index_from_config,
)


def _load_processor_and_config():
    transformers = require_transformers()
    model_path = get_model_path()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
    return transformers, processor, config

def test_single_image_rope_index_matches_hf():
    """单图 position_ids/rope_delta 必须与 HF get_rope_index 完全一致。"""

    transformers, processor, config = _load_processor_and_config()
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    inputs = prepare_single_image_inputs(processor, "Describe this image.", image)

    ours_pos, ours_delta = get_qwen3_vl_rope_index_from_config(
        inputs.input_ids,
        config=config,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    hf_pos, hf_delta = hf_qwen3_vl_rope_index(
        transformers,
        config,
        input_ids=inputs.input_ids,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )

    pos_diff = (ours_pos - hf_pos).abs().max().item()
    delta_diff = (ours_delta - hf_delta).abs().max().item()
    print(f"input_ids shape: {list(inputs.input_ids.shape)}")
    print(f"position_ids shape: {list(ours_pos.shape)}")
    print(f"rope_delta shape: {list(ours_delta.shape)}")
    print(f"position_ids max diff: {pos_diff:.6e}")
    print(f"rope_delta max diff: {delta_diff:.6e}")

    assert list(ours_pos.shape) == [3, 1, inputs.input_ids.shape[1]]
    assert list(ours_delta.shape) == [1, 1]
    assert pos_diff == 0
    assert delta_diff == 0
    print("single image rope index: PASS")


def test_text_rope_index_matches_hf():
    """纯文本分支也必须与 HF text-only 逻辑一致。"""

    transformers, _, config = _load_processor_and_config()
    input_ids = torch.tensor([[151644, 872, 198, 77091, 198]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    ours_pos, ours_delta = get_qwen3_vl_rope_index(
        input_ids,
        attention_mask=attention_mask,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    hf_pos, hf_delta = hf_qwen3_vl_rope_index(
        transformers,
        config,
        input_ids=input_ids,
        image_grid_thw=None,
        attention_mask=attention_mask,
    )

    pos_diff = (ours_pos - hf_pos).abs().max().item()
    delta_diff = (ours_delta - hf_delta).abs().max().item()
    print(f"text position_ids shape: {list(ours_pos.shape)}")
    print(f"text rope_delta shape: {list(ours_delta.shape)}")
    print(f"text position_ids max diff: {pos_diff:.6e}")
    print(f"text rope_delta max diff: {delta_diff:.6e}")

    assert pos_diff == 0
    assert delta_diff == 0
    print("text rope index: PASS")


def test_rope_index_rejects_grid_mismatch():
    """image_grid_thw 和 input_ids 中 image span 数量不一致时必须报错。"""

    _, processor, config = _load_processor_and_config()
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    inputs = prepare_single_image_inputs(processor, "Describe this image.", image)

    try:
        get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            image_grid_thw=inputs.image_grid_thw[:0],
            attention_mask=inputs.attention_mask,
        )
    except ValueError as exc:
        assert "image_grid_thw" in str(exc)
        print("grid mismatch rejection: PASS")
        return
    raise AssertionError("expected ValueError for image_grid_thw mismatch")
