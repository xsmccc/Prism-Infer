"""P3.2 Qwen3-VL 视频 3D position ids 验证。"""

import torch

from conftest import get_model_path, hf_qwen3_vl_rope_index, require_transformers
from prism_infer.engine.vl_inputs import prepare_video_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from test_processor_pipeline_video import demo_video_frames


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

def test_video_rope_index_matches_hf():
    """视频 position_ids/rope_delta 必须与 HF get_rope_index 完全一致。"""

    transformers, processor, config = _load_processor_and_config()
    inputs = prepare_video_inputs(processor, "Describe this video.", demo_video_frames())

    ours_pos, ours_delta = get_qwen3_vl_rope_index_from_config(
        inputs.input_ids,
        config=config,
        video_grid_thw=inputs.video_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    hf_pos, hf_delta = hf_qwen3_vl_rope_index(
        transformers,
        config,
        input_ids=inputs.input_ids,
        video_grid_thw=inputs.video_grid_thw,
        attention_mask=inputs.attention_mask,
    )

    pos_diff = (ours_pos - hf_pos).abs().max().item()
    delta_diff = (ours_delta - hf_delta).abs().max().item()
    print(f"video input_ids shape: {list(inputs.input_ids.shape)}")
    print(f"video_grid_thw shape: {list(inputs.video_grid_thw.shape)}")
    print(f"video position_ids shape: {list(ours_pos.shape)}")
    print(f"video rope_delta shape: {list(ours_delta.shape)}")
    print(f"video position_ids max diff: {pos_diff:.6e}")
    print(f"video rope_delta max diff: {delta_diff:.6e}")

    assert list(inputs.video_grid_thw.shape) == [1, 3]
    assert list(ours_pos.shape) == [3, 1, inputs.input_ids.shape[1]]
    assert list(ours_delta.shape) == [1, 1]
    assert pos_diff == 0
    assert delta_diff == 0
    print("video rope index: PASS")


def test_video_rope_index_rejects_grid_mismatch():
    """视频 grid 行数少于 video span 数量时必须报错。"""

    _, processor, config = _load_processor_and_config()
    inputs = prepare_video_inputs(processor, "Describe this video.", demo_video_frames())

    try:
        get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            video_grid_thw=inputs.video_grid_thw[:0],
            attention_mask=inputs.attention_mask,
        )
    except ValueError as exc:
        assert "video_grid_thw" in str(exc)
        print("video grid mismatch rejection: PASS")
        return
    raise AssertionError("expected ValueError for video_grid_thw mismatch")
