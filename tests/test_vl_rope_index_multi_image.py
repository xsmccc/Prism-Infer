"""P3.1 Qwen3-VL 多图 3D position ids 验证。"""

import pytest
from PIL import Image

from conftest import get_model_path, hf_qwen3_vl_rope_index, require_transformers
from prism_infer.engine.vl_inputs import prepare_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config


pytestmark = [pytest.mark.model, pytest.mark.integration]


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


def _demo_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (448, 448), color=(100, 150, 200)),
        Image.new("RGB", (448, 448), color=(200, 120, 80)),
    ]


def test_multi_image_rope_index_matches_hf():
    """多图 position_ids/rope_delta 必须与 HF get_rope_index 完全一致。"""

    transformers, processor, config = _load_processor_and_config()
    inputs = prepare_image_inputs(processor, "Compare these images.", _demo_images())

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
    print(f"multi input_ids shape: {list(inputs.input_ids.shape)}")
    print(f"multi image_grid_thw shape: {list(inputs.image_grid_thw.shape)}")
    print(f"multi position_ids shape: {list(ours_pos.shape)}")
    print(f"multi rope_delta shape: {list(ours_delta.shape)}")
    print(f"multi position_ids max diff: {pos_diff:.6e}")
    print(f"multi rope_delta max diff: {delta_diff:.6e}")

    assert list(inputs.image_grid_thw.shape) == [2, 3]
    assert list(ours_pos.shape) == [3, 1, inputs.input_ids.shape[1]]
    assert list(ours_delta.shape) == [1, 1]
    assert pos_diff == 0
    assert delta_diff == 0
    print("multi image rope index: PASS")


def test_multi_image_rope_index_rejects_grid_mismatch():
    """多图 grid 行数少于 image span 数量时必须报错。"""

    _, processor, config = _load_processor_and_config()
    inputs = prepare_image_inputs(processor, "Compare these images.", _demo_images())

    try:
        get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            image_grid_thw=inputs.image_grid_thw[:1],
            attention_mask=inputs.attention_mask,
        )
    except ValueError as exc:
        assert "image_grid_thw" in str(exc)
        print("multi image grid mismatch rejection: PASS")
        return
    raise AssertionError("expected ValueError for multi image_grid_thw mismatch")
