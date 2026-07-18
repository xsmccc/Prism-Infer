"""Public generation output must separate answer text from control tokens."""

from __future__ import annotations

from prism_infer.engine.llm_engine import LLMEngine


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
