"""P3.3 ModelRunner mixed text/VL batch 输入准备验证。"""

from types import SimpleNamespace

import torch
from PIL import Image

try:
    import pytest
except ImportError:
    pytest = None

from conftest import get_model_path, require_transformers
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.vl_inputs import (
    prepare_image_inputs,
    prepare_single_image_inputs,
    prepare_video_inputs,
)
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from prism_infer.sampling_params import SamplingParams
from prism_infer.utils.context import get_context, reset_context
from test_processor_pipeline_video import demo_video_frames


pytestmark = (
    []
    if pytest is None
    else [
        pytest.mark.model,
        pytest.mark.gpu,
        pytest.mark.integration,
    ]
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        message = "ModelRunner mixed VL prepare tests require CUDA"
        if pytest is not None:
            pytest.skip(message)
        raise SystemExit(f"SKIP: {message}")


def _make_runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.config = SimpleNamespace(enable_chunked_prefill=True, max_chunk_size=1024)
    return runner


def _processor_and_config():
    transformers = require_transformers()
    model_path = get_model_path()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
    return processor, config


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _vl_sequence(kind: str, block_start: int) -> Sequence:
    processor, config = _processor_and_config()
    sampling = SamplingParams(max_tokens=2, temperature=0.0)
    if kind == "single":
        inputs = prepare_single_image_inputs(
            processor,
            "Describe this image.",
            _image((100, 150, 200)),
        )
        position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=inputs.attention_mask,
        )
        seq = Sequence.from_single_image_inputs(
            inputs,
            sampling,
            block_size=256,
            request_id=block_start,
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
    elif kind == "multi":
        inputs = prepare_image_inputs(
            processor,
            "Compare these images.",
            [_image((100, 150, 200)), _image((200, 120, 80))],
        )
        position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=inputs.attention_mask,
        )
        seq = Sequence.from_image_inputs(
            inputs,
            sampling,
            block_size=256,
            request_id=block_start,
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
    elif kind == "video":
        inputs = prepare_video_inputs(
            processor,
            "Describe this video.",
            demo_video_frames(),
        )
        position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            video_grid_thw=inputs.video_grid_thw,
            attention_mask=inputs.attention_mask,
        )
        seq = Sequence.from_video_inputs(
            inputs,
            sampling,
            block_size=256,
            request_id=block_start,
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
    else:
        raise ValueError(kind)

    seq.block_table = list(range(block_start, block_start + seq.num_blocks))
    return seq


def _text_sequence(block_start: int) -> Sequence:
    seq = Sequence(
        [151644, 872, 198, 77091, 198],
        SamplingParams(max_tokens=2),
        block_size=256,
        request_id=block_start,
    )
    seq.block_table = list(range(block_start, block_start + seq.num_blocks))
    return seq


def test_prepare_prefill_mixed_text_image_video_batch():
    """mixed prefill 应统一 3D positions 并按请求顺序 concat visual payload。"""

    _require_cuda()
    runner = _make_runner()
    text_seq = _text_sequence(0)
    single_seq = _vl_sequence("single", 1)
    multi_seq = _vl_sequence("multi", 2)
    video_seq = _vl_sequence("video", 4)
    seqs = [text_seq, single_seq, multi_seq, video_seq]

    model_inputs = runner.prepare_prefill(seqs)
    context = get_context()
    expected_total = sum(len(seq) for seq in seqs)
    expected_cu = [0]
    for seq in seqs:
        expected_cu.append(expected_cu[-1] + len(seq))

    print(f"mixed prefill input_ids shape: {list(model_inputs.input_ids.shape)}")
    print(f"mixed prefill position_ids shape: {list(model_inputs.position_ids.shape)}")
    print(f"mixed pixel_values shape: {list(model_inputs.pixel_values.shape)}")
    print(f"mixed image_grid_thw shape: {list(model_inputs.image_grid_thw.shape)}")
    print(f"mixed pixel_values_videos shape: {list(model_inputs.pixel_values_videos.shape)}")
    print(f"mixed video_grid_thw shape: {list(model_inputs.video_grid_thw.shape)}")
    print(f"mixed cu_seqlens_q: {context.cu_seqlens_q.tolist()}")
    print(f"mixed slot_mapping shape: {list(context.slot_mapping.shape)}")

    assert list(model_inputs.input_ids.shape) == [expected_total]
    assert list(model_inputs.position_ids.shape) == [3, expected_total]
    assert model_inputs.position_ids[:, : len(text_seq)].tolist() == [
        list(range(len(text_seq))),
        list(range(len(text_seq))),
        list(range(len(text_seq))),
    ]
    assert list(model_inputs.pixel_values.shape) == [
        single_seq.pixel_values.shape[0] + multi_seq.pixel_values.shape[0],
        single_seq.pixel_values.shape[1],
    ]
    assert list(model_inputs.image_grid_thw.shape) == [3, 3]
    assert list(model_inputs.pixel_values_videos.shape) == list(video_seq.pixel_values_videos.shape)
    assert list(model_inputs.video_grid_thw.shape) == [1, 3]
    assert context.cu_seqlens_q.tolist() == expected_cu
    assert context.max_seqlen_q == max(len(seq) for seq in seqs)
    assert list(context.slot_mapping.shape) == [expected_total]
    reset_context()
    print("mixed text/image/video prefill inputs: PASS")


def test_prepare_decode_mixed_text_vl_batch_positions():
    """mixed decode 中 text-only 和 VL 请求应统一输出 [3, batch] position ids。"""

    _require_cuda()
    runner = _make_runner()
    text_seq = _text_sequence(0)
    image_seq = _vl_sequence("single", 1)
    video_seq = _vl_sequence("video", 2)
    for token, seq in zip([11, 22, 33], [text_seq, image_seq, video_seq]):
        seq.append_token(token)

    model_inputs = runner.prepare_decode([text_seq, image_seq, video_seq])
    context = get_context()
    expected_text_pos = len(text_seq) - 1
    expected_image_pos = len(image_seq) - 1 + int(image_seq.rope_delta.item())
    expected_video_pos = len(video_seq) - 1 + int(video_seq.rope_delta.item())

    print(f"mixed decode input_ids shape: {list(model_inputs.input_ids.shape)}")
    print(f"mixed decode position_ids shape: {list(model_inputs.position_ids.shape)}")
    print(f"mixed decode positions: {model_inputs.position_ids.tolist()}")
    print(f"mixed decode context_lens: {context.context_lens.tolist()}")

    assert model_inputs.input_ids.tolist() == [11, 22, 33]
    assert list(model_inputs.position_ids.shape) == [3, 3]
    assert model_inputs.position_ids[:, 0].tolist() == [expected_text_pos] * 3
    assert model_inputs.position_ids[:, 1].tolist() == [expected_image_pos] * 3
    assert model_inputs.position_ids[:, 2].tolist() == [expected_video_pos] * 3
    assert context.context_lens.tolist() == [len(text_seq), len(image_seq), len(video_seq)]
    reset_context()
    print("mixed text/VL decode positions: PASS")
