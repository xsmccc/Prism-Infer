"""P2.2 Sequence 多模态字段和序列化验证。"""

import pickle

import torch
from PIL import Image

from conftest import get_model_path, require_transformers
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.vl_inputs import prepare_single_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from prism_infer.sampling_params import SamplingParams


def _single_image_sequence() -> Sequence:
    transformers = require_transformers()
    model_path = get_model_path()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    inputs = prepare_single_image_inputs(processor, "Describe this image.", image)
    position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
        inputs.input_ids,
        config=config,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    return Sequence.from_single_image_inputs(
        inputs,
        SamplingParams(max_tokens=4),
        position_ids=position_ids,
        rope_delta=rope_delta,
    )


def test_text_sequence_behavior_unchanged():
    """纯文本 Sequence 的基础行为不应被 VL 字段破坏。"""

    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
    assert len(seq) == 3
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == []
    assert not seq.is_multimodal
    seq.append_token(4)
    assert seq.last_token == 4
    assert seq.completion_token_ids == [4]
    print("text sequence regression: PASS")


def test_single_image_sequence_prefill_state_roundtrip():
    """Prefill 序列化必须保留完整 token 和 VL payload。"""

    seq = _single_image_sequence()
    restored = pickle.loads(pickle.dumps(seq))

    assert restored.is_multimodal
    assert restored.token_ids == seq.token_ids
    assert torch.equal(restored.pixel_values, seq.pixel_values)
    assert torch.equal(restored.image_grid_thw, seq.image_grid_thw)
    assert torch.equal(restored.position_ids, seq.position_ids)
    assert torch.equal(restored.rope_delta, seq.rope_delta)
    assert restored.image_token_id == seq.image_token_id
    assert restored.image_token_count == seq.image_token_count
    print(f"prefill token_ids length: {len(restored.token_ids)}")
    print(f"prefill position_ids shape: {list(restored.position_ids.shape)}")
    print("single image prefill sequence roundtrip: PASS")


def test_single_image_sequence_decode_state_omits_pixels():
    """Decode 序列化不应重复发送 pixel_values，但必须保留 rope_delta。"""

    seq = _single_image_sequence()
    seq.block_table = [0]
    seq.append_token(42)
    restored = pickle.loads(pickle.dumps(seq))

    assert restored.is_multimodal
    assert restored.last_token == 42
    assert restored.pixel_values is None
    assert restored.image_grid_thw is None
    assert restored.position_ids is None
    assert torch.equal(restored.rope_delta, seq.rope_delta)
    print(f"decode rope_delta shape: {list(restored.rope_delta.shape)}")
    print("single image decode sequence roundtrip: PASS")
