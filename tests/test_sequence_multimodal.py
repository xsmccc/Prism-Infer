"""Multimodal Sequence fields and serialization tests."""

import pickle

import pytest
import torch
from conftest import get_model_path, require_transformers
from PIL import Image

from prism_infer.engine.sequence import Sequence, split_image_payloads_for_range
from prism_infer.engine.vl_inputs import prepare_single_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from prism_infer.sampling_params import SamplingParams

pytestmark = [pytest.mark.model, pytest.mark.integration]


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
        block_size=256,
        request_id=0,
        position_ids=position_ids,
        rope_delta=rope_delta,
    )


def test_text_sequence_behavior_unchanged():
    """纯文本 Sequence 的基础行为不应被 VL 字段破坏。"""

    seq = Sequence(
        [1, 2, 3],
        SamplingParams(max_tokens=2),
        block_size=256,
        request_id=0,
    )
    assert len(seq) == 3
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == []
    assert not seq.is_multimodal
    seq.append_token(4)
    assert seq.last_token == 4
    assert seq.completion_token_ids == [4]
    print("text sequence regression: PASS")


def test_sequence_block_size_is_explicit_request_state():
    """Two engines/page contracts cannot mutate each other's Sequence state."""

    page4 = Sequence(
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=2),
        block_size=4,
        request_id=4,
    )
    page8 = Sequence(
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=2),
        block_size=8,
        request_id=8,
    )

    assert page4.block_size == 4
    assert page4.num_blocks == 2
    assert page8.block_size == 8
    assert page8.num_blocks == 1
    assert not hasattr(Sequence, "block_size")
    assert not hasattr(Sequence, "set_block_size")
    print("sequence explicit page contract isolation: PASS")


def test_decode_sequence_roundtrip_preserves_sampling_params():
    """Decode 序列化必须保留请求级采样参数。"""

    params = SamplingParams(temperature=0.25, max_tokens=7, ignore_eos=True)
    seq = Sequence(
        [1, 2, 3],
        params,
        block_size=256,
        request_id=0,
    )
    seq.block_table = [0]
    seq.append_token(42)

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.last_token == 42
    assert restored.temperature == params.temperature
    assert restored.max_tokens == params.max_tokens
    assert restored.ignore_eos == params.ignore_eos
    print(f"decode restored temperature: {restored.temperature}")
    print(f"decode restored max_tokens: {restored.max_tokens}")
    print(f"decode restored ignore_eos: {restored.ignore_eos}")
    print("decode sequence sampling params roundtrip: PASS")


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


@pytest.mark.parametrize("merge_size", [1, 2])
def test_partial_image_prefill_roundtrip_preserves_merge_size(merge_size):
    """TP workers must recover image spans after a partial prefix-cache hit."""

    image_token_id = 151655
    grid = torch.tensor([[1, 4, 4], [1, 4, 8]], dtype=torch.long)
    pixels = torch.arange(48 * 3, dtype=torch.float32).reshape(48, 3)
    first_tokens = 16 // (merge_size * merge_size)
    second_tokens = 32 // (merge_size * merge_size)
    tokens = (
        [7]
        + [image_token_id] * first_tokens
        + [9]
        + [image_token_id] * second_tokens
        + [10, 11]
    )
    seq = Sequence(
        tokens,
        block_size=4,
        request_id=1,
        pixel_values=pixels,
        image_grid_thw=grid,
        image_token_id=image_token_id,
        image_token_count=first_tokens + second_tokens,
        image_merge_size=merge_size,
    )
    seq.num_cached_tokens = 1 + first_tokens
    seq.num_computed_tokens = seq.num_cached_tokens

    restored = pickle.loads(pickle.dumps(seq))
    assert restored.image_merge_size == merge_size
    # A cached first image must not rebase the second image's original spans.
    assert torch.equal(restored.pixel_values, pixels)
    assert torch.equal(restored.image_grid_thw, grid)
    assert seq.pixel_values is pixels
    assert seq.image_grid_thw is grid
    spans = restored.image_token_spans()
    assert spans == seq.image_token_spans()
    payload, sliced_grid, covered_pads = split_image_payloads_for_range(
        pixel_values=restored.pixel_values,
        grid=restored.image_grid_thw,
        merge_size=restored.image_merge_size,
        image_spans=spans,
        range_start=restored.num_cached_tokens,
        range_end=restored.num_prompt_tokens,
    )
    assert torch.equal(payload, pixels[16:])
    assert torch.equal(sliced_grid, grid[1:])
    assert covered_pads == second_tokens


def _synthetic_image_video_sequence() -> Sequence:
    """CPU-only state: image spans [1, 5), video spans [6, 10), then text."""

    image_token_id = 151655
    video_token_id = 151656
    tokens = [7] + [image_token_id] * 4 + [8] + [video_token_id] * 4 + [9, 10]
    return Sequence(
        tokens,
        block_size=4,
        request_id=3,
        pixel_values=torch.arange(48, dtype=torch.float32).reshape(16, 3),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values_videos=torch.arange(48, dtype=torch.float32).reshape(16, 3) + 100,
        video_grid_thw=torch.tensor([[1, 4, 4]]),
        position_ids=torch.arange(len(tokens)).repeat(3, 1),
        rope_delta=torch.tensor([0]),
        image_token_id=image_token_id,
        image_token_count=4,
        video_token_id=video_token_id,
        video_token_count=4,
    )


@pytest.mark.parametrize("frontier", ["num_cached_tokens", "num_computed_tokens"])
def test_prefill_roundtrip_omits_fully_covered_visual_payload(frontier):
    """Full visual coverage omits image/video tensors, without changing rank 0."""

    seq = _synthetic_image_video_sequence()
    setattr(seq, frontier, 10)
    original_pixels = seq.pixel_values
    original_grid = seq.image_grid_thw
    original_video = seq.pixel_values_videos
    original_video_grid = seq.video_grid_thw

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.pixel_values is None
    assert restored.image_grid_thw is None
    assert restored.pixel_values_videos is None
    assert restored.video_grid_thw is None
    assert restored.token_ids == seq.token_ids
    assert torch.equal(restored.position_ids, seq.position_ids)
    assert torch.equal(restored.rope_delta, seq.rope_delta)
    assert restored.image_merge_size == seq.image_merge_size
    assert getattr(restored, frontier) == 10
    assert seq.pixel_values is original_pixels
    assert seq.image_grid_thw is original_grid
    assert seq.pixel_values_videos is original_video
    assert seq.video_grid_thw is original_video_grid


def test_prefill_roundtrip_keeps_uncovered_modality_only():
    """Covering the image does not suppress a later video's required payload."""

    seq = _synthetic_image_video_sequence()
    seq.num_cached_tokens = 5
    restored = pickle.loads(pickle.dumps(seq))

    assert restored.pixel_values is None
    assert restored.image_grid_thw is None
    assert torch.equal(restored.pixel_values_videos, seq.pixel_values_videos)
    assert torch.equal(restored.video_grid_thw, seq.video_grid_thw)
    assert restored.token_ids == seq.token_ids


def test_prefill_roundtrip_ignores_unconfirmed_prefix_candidate():
    """Probe flags cannot replace the actual cached/computed frontier."""

    seq = _synthetic_image_video_sequence()
    seq.prefix_cache_candidate_tokens = 10
    seq.multimodal_prefix_cache_hit = True
    restored = pickle.loads(pickle.dumps(seq))

    assert torch.equal(restored.pixel_values, seq.pixel_values)
    assert torch.equal(restored.image_grid_thw, seq.image_grid_thw)
    assert torch.equal(restored.pixel_values_videos, seq.pixel_values_videos)
    assert torch.equal(restored.video_grid_thw, seq.video_grid_thw)
