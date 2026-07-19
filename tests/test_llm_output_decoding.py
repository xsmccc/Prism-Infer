"""Public generation output must separate answer text from control tokens."""

from __future__ import annotations

import pytest

from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.sampling_params import SamplingParams


class _Tokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert token_ids == [2, 3]
        assert clean_up_tokenization_spaces is False
        return "answer" if skip_special_tokens else "answer<|im_end|>"


def test_generation_output_keeps_clean_and_lossless_decodes() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    engine.tokenizer = _Tokenizer()

    output = engine._format_generation_output([2, 3])

    assert output == {
        "text": "answer",
        "raw_text": "answer<|im_end|>",
        "token_ids": [2, 3],
    }


def test_generate_preserves_submission_order_for_non_monotonic_request_ids() -> None:
    request_ids = iter((41, 7))
    state = {"finished": False}

    def submit_sequence(_sequence):
        return next(request_ids)

    def step():
        state["finished"] = True
        return ([(7, [70]), (41, [410])], -2)

    engine = LLMEngine.__new__(LLMEngine)
    engine._prepare_text_sequence = lambda prompt, _params: prompt
    engine._submit_sequence = submit_sequence
    engine.is_finished = lambda: state["finished"]
    engine.step = step
    engine._format_generation_output = lambda token_ids: token_ids

    outputs = LLMEngine.generate(
        engine,
        ["first", "second"],
        SamplingParams(max_tokens=1),
        use_tqdm=False,
    )

    assert outputs == [[410], [70]]


def test_generate_rejects_sampling_parameter_cardinality_mismatch() -> None:
    engine = LLMEngine.__new__(LLMEngine)

    with pytest.raises(ValueError, match="one entry per prompt"):
        LLMEngine.generate(
            engine,
            ["first", "second"],
            [SamplingParams(max_tokens=1)],
            use_tqdm=False,
        )


def test_generate_mixed_validates_every_payload_before_preparing_or_submitting() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    prepared: list[str] = []
    submitted: list[object] = []
    engine._prepare_mixed_sequence = lambda request, _params: prepared.append(request.request_type)
    engine._submit_sequence = lambda sequence: submitted.append(sequence)

    with pytest.raises(ValueError, match="must provide 'image'"):
        engine.generate_mixed(
            [
                {"type": "text", "prompt": "first"},
                {"type": "image", "prompt": "second"},
            ],
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )

    assert prepared == []
    assert submitted == []


def test_generate_mixed_finishes_host_preparation_before_scheduler_submission() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    prepared: list[str] = []
    submitted: list[object] = []

    def prepare(request, _params):
        prepared.append(request.request_type)
        if request.request_type == "video":
            raise RuntimeError("synthetic preprocessing failure")
        return object()

    engine._prepare_mixed_sequence = prepare
    engine._submit_sequence = lambda sequence: submitted.append(sequence)

    with pytest.raises(RuntimeError, match="synthetic preprocessing failure"):
        engine.generate_mixed(
            [
                {"type": "text", "prompt": "first"},
                {"type": "video", "prompt": "second", "video": object()},
            ],
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )

    assert prepared == ["text", "video"]
    assert submitted == []


def test_generation_loop_closes_progress_bar_when_execution_fails() -> None:
    class ProgressBar:
        closed = False

        def close(self) -> None:
            self.closed = True

    def fail_step():
        raise RuntimeError("synthetic execution failure")

    engine = LLMEngine.__new__(LLMEngine)
    engine.is_finished = lambda: False
    engine.step = fail_step
    progress_bar = ProgressBar()

    with pytest.raises(RuntimeError, match="synthetic execution failure"):
        engine._run_generation([1], progress_bar=progress_bar)

    assert progress_bar.closed
