"""P9 标准质量集 prompt、答案解析与逐样本指标。

参考实现均固定到首次正式运行前的 revision：

- DocVQA ANLS: lmms-eval ``b485e662``，``lmms_eval/api/metrics.py``；
- MuirBench prompt/parser: MUIRBench ``840b85fe``，``eval/utils``；
- MVBench prompt/parser: lmms-eval ``b485e662``，``tasks/mvbench/utils.py``。

MuirBench 官方 parser 在无法解析时随机猜测。这里同时返回 frozen-order、seed=42 的
official-compatible 结果和“无法解析即错误”的 strict 结果，避免随机 fallback 被隐藏。
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DOCVQA_ANLS_THRESHOLD = 0.5
MUIRBENCH_RANDOM_FALLBACK_SEED = 42
MUIRBENCH_BARE_LABEL_MAX_WORDS = 5
DOCVQA_POST_PROMPT = "\nAnswer the question using a single word or phrase."
MUIRBENCH_HINT = "Hint: Please provide the correct option letter, such as A, B, C, D, directly."
MVBENCH_POST_PROMPT = "Only give the best option.\n"
MVBENCH_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
MVBENCH_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
MVBENCH_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def levenshtein_distance(left: str, right: str) -> int:
    """以 O(min(len(left), len(right))) 内存计算字符 Levenshtein 距离。"""

    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("Levenshtein inputs must be strings")
    if len(left) < len(right):
        left, right = right, left
    distances = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            substitution = distances[right_index - 1] + (left_character != right_character)
            current.append(
                min(
                    current[-1] + 1,
                    distances[right_index] + 1,
                    substitution,
                )
            )
        distances = current
    return distances[-1]


def _normalize_anls_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def docvqa_anls(
    prediction: str,
    references: Sequence[str],
    *,
    threshold: float = DOCVQA_ANLS_THRESHOLD,
) -> float:
    """复现 DocVQA/lmms-eval 的 max-over-references ANLS。"""

    if not isinstance(prediction, str):
        raise TypeError("DocVQA prediction must be a string")
    if not references or not all(isinstance(answer, str) for answer in references):
        raise ValueError("DocVQA references must be a non-empty string sequence")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("ANLS threshold must be in [0, 1]")
    normalized_prediction = _normalize_anls_text(prediction)
    normalized_distances = []
    for answer in references:
        normalized_answer = _normalize_anls_text(answer)
        distance = levenshtein_distance(
            normalized_answer,
            normalized_prediction,
        )
        # Keep the pinned evaluator's raw-string denominator for exact parity.
        length = max(len(answer.upper()), len(prediction.upper()))
        normalized_distances.append(0.0 if length == 0 else distance / length)
    score = 1.0 - min(normalized_distances)
    return 0.0 if score < threshold else score


def choice_labels(option_count: int) -> list[str]:
    if (
        isinstance(option_count, bool)
        or not isinstance(option_count, int)
        or option_count <= 0
        or option_count > len(string.ascii_uppercase)
    ):
        raise ValueError("option_count must be in [1, 26]")
    return list(string.ascii_uppercase[:option_count])


def build_docvqa_prompt(question: str) -> str:
    if not isinstance(question, str) or not question:
        raise ValueError("DocVQA question must be non-empty")
    return f"{question}{DOCVQA_POST_PROMPT}"


def build_muirbench_prompt(question: str, options: Sequence[str]) -> str:
    """复现 MuirBench ``eval/utils/preprocess.py:create_prompt``。"""

    if not isinstance(question, str) or not question:
        raise ValueError("MuirBench question must be non-empty")
    labels = choice_labels(len(options))
    if not all(isinstance(option, str) for option in options):
        raise ValueError("MuirBench options must be strings")
    choices = ["Choices:"] + [f"({label}) {option}" for label, option in zip(labels, options)]
    return "\n".join(
        (
            f"Question: {question}",
            *choices,
            MUIRBENCH_HINT,
            "Answer:",
        )
    ).strip()


@dataclass(frozen=True)
class ParsedChoice:
    label: str | None
    method: str
    candidates: tuple[str, ...]


def parse_muirbench_response(
    response: str,
    options: Sequence[str],
    *,
    random_generator: random.Random | None = None,
) -> ParsedChoice:
    """复现官方多阶段 parser；``random_generator=None`` 表示 strict。"""

    if not isinstance(response, str):
        raise TypeError("MuirBench response must be a string")
    labels = choice_labels(len(options))
    index_to_answer = dict(zip(labels, options))
    cleaned = response
    for character in (",", ".", "!", "?", ";", ":", "'"):
        cleaned = cleaned.strip(character)
    cleaned = f" {cleaned} "
    candidates = [label for label in labels if f"({label})" in cleaned]
    method = "bracketed_label"
    answer_is_label = True
    bracketed = bool(candidates)
    if not candidates:
        candidates = [label for label in labels if f" {label} " in cleaned]
        method = "bare_label"
    if not candidates and len(cleaned.split()) > MUIRBENCH_BARE_LABEL_MAX_WORDS:
        candidates = [
            label for label, answer in index_to_answer.items() if answer.lower() in cleaned.lower()
        ]
        method = "option_text"
        answer_is_label = False
    if not candidates:
        if random_generator is None:
            return ParsedChoice(None, "unparseable", ())
        return ParsedChoice(
            random_generator.choice(labels),
            "official_random_fallback",
            (),
        )
    if len(candidates) == 1:
        return ParsedChoice(candidates[0], method, tuple(candidates))
    if answer_is_label:
        indexes = [
            cleaned.rfind(f"({label})" if bracketed else f" {label} ") for label in candidates
        ]
    else:
        indexes = [cleaned.lower().rfind(index_to_answer[label].lower()) for label in candidates]
    winner = max(range(len(candidates)), key=indexes.__getitem__)
    return ParsedChoice(candidates[winner], f"{method}_last_mention", tuple(candidates))


def build_mvbench_prompt(question: str, candidates: Sequence[str]) -> str:
    """复现 lmms-eval MVBench prompt，包括其 ``Option`` 单数拼写。"""

    if not isinstance(question, str) or not question:
        raise ValueError("MVBench question must be non-empty")
    labels = choice_labels(len(candidates))
    if not all(isinstance(candidate, str) for candidate in candidates):
        raise ValueError("MVBench candidates must be strings")
    options = "".join(f"({label}) {candidate}\n" for label, candidate in zip(labels, candidates))
    return f"Question:{question}\nOption:\n{options}{MVBENCH_POST_PROMPT}"


def parse_mvbench_response(response: str) -> ParsedChoice:
    """复现 lmms-eval ``mcq_acc`` 的 A–E 提取与标点处理。"""

    if not isinstance(response, str):
        raise TypeError("MVBench response must be a string")
    direct = re.match(r"^([A-E])\.\s*(.+)$", response.strip(), re.IGNORECASE)
    if direct:
        return ParsedChoice(direct.group(1).upper(), "leading_label_period", ())
    cleaned = response.replace("\n", " ").replace("\t", " ").strip()
    original = cleaned
    for punctuation in MVBENCH_PUNCTUATION:
        if (
            f"{punctuation} " in original
            or f" {punctuation}" in original
            or MVBENCH_COMMA_STRIP.search(original) is not None
        ):
            cleaned = cleaned.replace(punctuation, "")
        else:
            cleaned = cleaned.replace(punctuation, " ")
    cleaned = MVBENCH_PERIOD_STRIP.sub("", cleaned)
    cleaned = cleaned.strip().strip("'").strip('"').strip(")").strip("(").lower()
    match = re.search(r"\b([A-E])\b", cleaned, re.IGNORECASE)
    if match:
        return ParsedChoice(match.group(1).upper(), "embedded_label", ())
    return ParsedChoice(None, "unparseable", ())


def score_quality_prediction(
    dataset_id: str,
    record: Mapping[str, Any],
    prediction: str,
    *,
    muirbench_random: random.Random | None = None,
) -> dict[str, Any]:
    """Return the frozen parser trace and per-sample score."""

    if dataset_id == "docvqa_validation":
        return {
            "target": list(record["answers"]),
            "anls": docvqa_anls(prediction, record["answers"]),
        }
    if dataset_id == "muirbench_test":
        if muirbench_random is None:
            raise ValueError("MuirBench official-compatible scoring requires an RNG")
        strict = parse_muirbench_response(prediction, record["options"])
        official = parse_muirbench_response(
            prediction,
            record["options"],
            random_generator=muirbench_random,
        )
        target = record["answer"]
        return {
            "target": target,
            "strict_prediction": strict.label,
            "strict_parse_method": strict.method,
            "strict_score": int(strict.label == target),
            "official_prediction": official.label,
            "official_parse_method": official.method,
            "official_score": int(official.label == target),
        }
    if dataset_id == "mvbench_test":
        parsed = parse_mvbench_response(prediction)
        target = choice_labels(len(record["candidates"]))[record["answer_index"]]
        return {
            "target": target,
            "prediction": parsed.label,
            "parse_method": parsed.method,
            "score": int(parsed.label == target),
            "answered": bool(prediction.strip()),
        }
    raise ValueError(f"unsupported quality dataset: {dataset_id}")


def aggregate_quality_predictions(
    dataset_id: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute one aggregate exclusively from auditable sample scores."""

    if not samples:
        return {"samples": 0}
    if dataset_id == "docvqa_validation":
        scores = [float(sample["score"]["anls"]) for sample in samples]
        return {"samples": len(scores), "mean_anls": sum(scores) / len(scores)}
    if dataset_id == "muirbench_test":
        strict = [int(sample["score"]["strict_score"]) for sample in samples]
        official = [int(sample["score"]["official_score"]) for sample in samples]
        fallback = sum(
            sample["score"]["official_parse_method"] == "official_random_fallback"
            for sample in samples
        )
        return {
            "samples": len(samples),
            "strict_accuracy": sum(strict) / len(strict),
            "official_compatible_accuracy": sum(official) / len(official),
            "official_random_fallback_samples": fallback,
        }
    if dataset_id == "mvbench_test":
        scores = [int(sample["score"]["score"]) for sample in samples]
        answered = [sample for sample in samples if sample["score"]["answered"]]
        by_task: dict[str, list[int]] = {}
        for sample in samples:
            by_task.setdefault(sample["task"], []).append(int(sample["score"]["score"]))
        return {
            "samples": len(samples),
            "selected_denominator_accuracy": sum(scores) / len(scores),
            "answered_samples": len(answered),
            "lmms_answered_denominator_accuracy": (
                sum(int(sample["score"]["score"]) for sample in answered) / len(answered)
                if answered
                else 0.0
            ),
            "task_accuracy": {
                task: sum(task_scores) / len(task_scores)
                for task, task_scores in sorted(by_task.items())
            },
        }
    raise ValueError(f"unsupported quality dataset: {dataset_id}")
