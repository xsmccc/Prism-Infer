"""P2.4/P2.5 ModelRunner VL prefill/decode 输入准备验证。"""

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
from prism_infer.engine.scheduler_policy import FCFSSchedulerPolicy
from prism_infer.engine.vl_inputs import prepare_single_image_inputs
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from prism_infer.sampling_params import SamplingParams
from prism_infer.utils.context import get_context, reset_context


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        message = "ModelRunner VL prepare tests require CUDA"
        if pytest is not None:
            pytest.skip(message)
        raise SystemExit(f"SKIP: {message}")


def _make_runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.config = SimpleNamespace(enable_chunked_prefill=True, max_chunk_size=512)
    return runner


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
    seq = Sequence.from_single_image_inputs(
        inputs,
        SamplingParams(max_tokens=4),
        position_ids=position_ids,
        rope_delta=rope_delta,
    )
    seq.block_table = [0]
    return seq


def test_prepare_prefill_carries_single_image_payload_and_context():
    """VL prefill 必须传递 3D position ids、图像 payload 和 attention context。"""

    _require_cuda()
    runner = _make_runner()
    seq = _single_image_sequence()
    model_inputs = runner.prepare_prefill([seq])
    context = get_context()

    expected_len = len(seq)
    print(f"prefill input_ids shape: {list(model_inputs.input_ids.shape)}")
    print(f"prefill position_ids shape: {list(model_inputs.position_ids.shape)}")
    print(f"prefill pixel_values shape: {list(model_inputs.pixel_values.shape)}")
    print(f"prefill image_grid_thw shape: {list(model_inputs.image_grid_thw.shape)}")
    print(f"prefill cu_seqlens_q: {context.cu_seqlens_q.tolist()}")
    print(f"prefill slot_mapping shape: {list(context.slot_mapping.shape)}")

    assert list(model_inputs.input_ids.shape) == [expected_len]
    assert list(model_inputs.position_ids.shape) == [3, expected_len]
    assert list(model_inputs.pixel_values.shape) == list(seq.pixel_values.shape)
    assert list(model_inputs.image_grid_thw.shape) == [1, 3]
    assert context.is_prefill
    assert context.cu_seqlens_q.tolist() == [0, expected_len]
    assert context.max_seqlen_q == expected_len
    assert list(context.slot_mapping.shape) == [expected_len]
    reset_context()
    print("model runner VL prefill inputs: PASS")


def test_prepare_decode_uses_rope_delta_without_pixels():
    """VL decode 只传 last token 和 rope_delta 延续的位置，不重复传图像。"""

    _require_cuda()
    runner = _make_runner()
    seq = _single_image_sequence()
    seq.append_token(42)

    model_inputs = runner.prepare_decode([seq])
    context = get_context()
    expected_pos = len(seq) - 1 + int(seq.rope_delta.item())

    print(f"decode input_ids shape: {list(model_inputs.input_ids.shape)}")
    print(f"decode position_ids shape: {list(model_inputs.position_ids.shape)}")
    print(f"decode expected position: {expected_pos}")
    print(f"decode actual positions: {model_inputs.position_ids[:, 0].tolist()}")
    print(f"decode context_lens: {context.context_lens.tolist()}")

    assert model_inputs.pixel_values is None
    assert model_inputs.image_grid_thw is None
    assert model_inputs.input_ids.tolist() == [42]
    assert list(model_inputs.position_ids.shape) == [3, 1]
    assert model_inputs.position_ids[:, 0].tolist() == [expected_pos] * 3
    assert not context.is_prefill
    assert context.context_lens.tolist() == [len(seq)]
    reset_context()
    print("model runner VL decode inputs: PASS")


def test_vl_chunk_policy_rejects_split_visual_span():
    """Visual placeholders must fit atomically in one prefill chunk."""

    seq = _single_image_sequence()
    policy = FCFSSchedulerPolicy(
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        max_chunk_size=16,
    )

    decision = policy.admit(seq, queued_requests=0)

    assert not decision.accepted
    assert "visual token span" in decision.reason
    print("VL visual-span atomic chunk admission: PASS")


def test_prepare_vl_followup_chunk_uses_paged_history_without_pixels():
    """After the visual span, a text-only tail chunk reuses paged KV safely."""

    _require_cuda()
    runner = _make_runner()
    seq = _single_image_sequence()
    visual_end = max(
        index
        for index, token_id in enumerate(seq.prompt_token_ids)
        if token_id == seq.image_token_id
    ) + 1
    assert visual_end < seq.num_prompt_tokens
    seq.num_cached_tokens = visual_end
    seq.num_computed_tokens = visual_end

    model_inputs = runner.prepare_prefill([seq])
    context = get_context()
    try:
        assert model_inputs.pixel_values is None
        assert model_inputs.image_grid_thw is None
        assert model_inputs.input_ids.tolist() == seq.token_ids[visual_end:]
        assert context.cu_seqlens_q.tolist() == [
            0,
            seq.num_prompt_tokens - visual_end,
        ]
        assert context.cu_seqlens_k.tolist() == [0, seq.num_prompt_tokens]
        assert context.context_lens.tolist() == [seq.num_prompt_tokens]
        assert context.block_tables.tolist() == [[0]]
        assert context.slot_mapping.tolist() == list(
            range(visual_end, seq.num_prompt_tokens)
        )
    finally:
        reset_context()
    print("VL follow-up paged prefill chunk: PASS")
