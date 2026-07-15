"""P6.12 多参考 caption/free-text 质量指标测试。"""

import pytest

from prism_infer.analysis.reference_quality import (
    normalize_reference_text,
    rouge_l_f1,
    score_reference_batch,
    score_reference_text,
    token_f1,
)


def _task_reference(*texts: str) -> dict[str, object]:
    return {
        "task": "caption",
        "reference_source": "unit_test",
        "image_id": 1,
        "references": [
            {"annotation_id": index + 1, "text": text}
            for index, text in enumerate(texts)
        ],
    }


def test_reference_text_normalization_and_overlap_metrics() -> None:
    assert normalize_reference_text("  Two CATS—sleep! ") == [
        "two",
        "cats",
        "sleep",
    ]
    assert token_f1("cat cat sofa", "cat sofa sofa") == pytest.approx(2 / 3)
    assert rouge_l_f1(
        "two cats sleep on a sofa",
        "two cats sleeping on the sofa",
    ) == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
    assert rouge_l_f1("", "cat") == 0.0
    print("P6.12 reference text metrics: PASS")


def test_reference_quality_takes_independent_best_reference_per_metric() -> None:
    task_reference = _task_reference(
        "d c b a",
        "a b c",
    )

    score = score_reference_text("a b c d", task_reference)

    assert score["reference_count"] == 2
    assert score["token_f1"] == 1.0
    assert score["rouge_l_f1"] == pytest.approx(6 / 7)
    assert score["best_token_f1_annotation_id"] == 1
    assert score["best_rouge_l_annotation_id"] == 2
    print("P6.12 multi-reference best-match scoring: PASS")


def test_reference_quality_batch_fails_closed_when_reference_is_missing() -> None:
    unavailable = score_reference_batch(
        ["two cats", "a kitchen"],
        [_task_reference("two cats"), None],
    )

    assert unavailable["available"] is False
    assert unavailable["request_count"] == 2
    assert "requests lack task references: [1]" in unavailable["reason"]
    print("P6.12 incomplete reference batch guard: PASS")


def test_reference_quality_rejects_reference_without_normalized_tokens() -> None:
    with pytest.raises(ValueError, match="no normalized tokens"):
        score_reference_text("", _task_reference("!!!"))
    print("P6.12 normalized reference token guard: PASS")


def test_reference_quality_rejects_duplicate_annotation_ids() -> None:
    task_reference = _task_reference("one", "two")
    task_reference["references"][1]["annotation_id"] = 1

    with pytest.raises(ValueError, match="duplicate task reference"):
        score_reference_text("one", task_reference)
    print("P6.12 duplicate annotation guard: PASS")
