"""P9 quality prompt 与官方兼容逐样本指标测试。"""

from __future__ import annotations

import random

import pytest

from prism_infer.analysis.p9_quality_metrics import (
    MUIRBENCH_RANDOM_FALLBACK_SEED,
    build_docvqa_prompt,
    build_muirbench_prompt,
    build_mvbench_prompt,
    docvqa_anls,
    levenshtein_distance,
    parse_muirbench_response,
    parse_mvbench_response,
)


def test_levenshtein_and_docvqa_anls_match_reference_contract() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert docvqa_anls("New   York", ["new york", "Boston"]) == 1.0
    assert docvqa_anls("invoice", ["invoices"]) == pytest.approx(0.875)
    assert docvqa_anls("completely different", ["42"]) == 0.0


def test_quality_prompts_are_frozen_before_candidate_results() -> None:
    assert build_docvqa_prompt("What is the total?") == (
        "What is the total?\nAnswer the question using a single word or phrase."
    )
    assert build_muirbench_prompt(
        "Compare <image> and <image>.",
        ["first", "second"],
    ) == (
        "Question: Compare <image> and <image>.\n"
        "Choices:\n"
        "(A) first\n"
        "(B) second\n"
        "Hint: Please provide the correct option letter, such as A, B, C, D, "
        "directly.\n"
        "Answer:"
    )
    assert build_mvbench_prompt("What happened?", ["Run", "Sit"]) == (
        "Question:What happened?\n"
        "Option:\n"
        "(A) Run\n"
        "(B) Sit\n"
        "Only give the best option.\n"
    )


def test_muirbench_parser_uses_last_mention_and_reports_random_fallback() -> None:
    options = ["red square", "blue circle", "green triangle"]
    parsed = parse_muirbench_response(
        "I first considered (A), but the answer is (B).",
        options,
    )
    assert parsed.label == "B"
    assert parsed.method == "bracketed_label_last_mention"

    strict = parse_muirbench_response("I cannot determine it", options)
    official = parse_muirbench_response(
        "I cannot determine it",
        options,
        random_generator=random.Random(MUIRBENCH_RANDOM_FALLBACK_SEED),
    )
    assert strict.label is None
    assert strict.method == "unparseable"
    assert official.label == "C"
    assert official.method == "official_random_fallback"


def test_muirbench_parser_can_recover_long_option_text_response() -> None:
    parsed = parse_muirbench_response(
        "After reviewing every image carefully, I choose the blue circle option",
        ["red square", "blue circle"],
    )

    assert parsed.label == "B"
    assert parsed.method == "option_text"


@pytest.mark.parametrize(
    ("response", "expected", "method"),
    [
        ("B. Sit", "B", "leading_label_period"),
        ("The best option is (C).", "C", "embedded_label"),
        ("", None, "unparseable"),
        ("No idea", None, "unparseable"),
    ],
)
def test_mvbench_parser_matches_lmms_eval_choice_extraction(
    response: str,
    expected: str | None,
    method: str,
) -> None:
    parsed = parse_mvbench_response(response)

    assert parsed.label == expected
    assert parsed.method == method
