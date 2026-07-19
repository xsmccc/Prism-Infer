"""无外部依赖的多参考文本质量 preflight 指标。

该模块面向 P6.12 pruning 对比中的 caption/free-text QA 输出。它提供
Unicode 规范化、多重集 token-F1 和 ROUGE-L F1，并对每项指标分别取
多参考中的最高分。它不是 COCO 官方 CIDEr/SPICE evaluator。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


REFERENCE_QUALITY_SCHEMA_VERSION = 1
_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_reference_text(text: str) -> list[str]:
    """将文本规范化为稳定的 Unicode 字母/数字 token 序列。"""

    if not isinstance(text, str):
        raise TypeError(f"reference metric text must be str, got {type(text)!r}")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def token_f1(prediction: str, reference: str) -> float:
    """计算 prediction/reference 的多重集 token F1。"""

    prediction_tokens = normalize_reference_text(prediction)
    reference_tokens = normalize_reference_text(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(
    left: Sequence[str],
    right: Sequence[str],
) -> int:
    """用一维动态规划计算两个 token 序列的最长公共子序列长度。"""

    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """计算基于 token LCS 的 ROUGE-L F1。"""

    prediction_tokens = normalize_reference_text(prediction)
    reference_tokens = normalize_reference_text(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = _lcs_length(prediction_tokens, reference_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_reference_text(
    prediction: str,
    task_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """对一个输出计算多参考 token-F1/ROUGE-L 的独立 best-match 分数。"""

    task, reference_source, image_id, references = _validate_task_reference_header(task_reference)
    scored = _score_reference_entries(prediction, references)
    best_token_f1 = max(scored, key=lambda row: row["token_f1"])
    best_rouge_l = max(scored, key=lambda row: row["rouge_l_f1"])
    return {
        "schema_version": REFERENCE_QUALITY_SCHEMA_VERSION,
        "task": task,
        "reference_source": reference_source,
        "image_id": image_id,
        "reference_count": len(scored),
        "token_f1": best_token_f1["token_f1"],
        "rouge_l_f1": best_rouge_l["rouge_l_f1"],
        "best_token_f1_annotation_id": best_token_f1["annotation_id"],
        "best_rouge_l_annotation_id": best_rouge_l["annotation_id"],
    }


def _validate_task_reference_header(
    task_reference: Mapping[str, Any],
) -> tuple[str, str, int, list[Any]]:
    task = task_reference.get("task")
    reference_source = task_reference.get("reference_source")
    image_id = task_reference.get("image_id")
    if task not in ("caption", "free_text_qa"):
        raise ValueError("task reference has unsupported task")
    if not isinstance(reference_source, str) or not reference_source:
        raise ValueError("task reference has invalid reference_source")
    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 1:
        raise ValueError("task reference has invalid image_id")
    references = task_reference.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("task reference requires a non-empty references list")
    return task, reference_source, image_id, references


def _score_reference_entries(
    prediction: str,
    references: list[Any],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    seen_annotation_ids: set[int] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"task reference {index} must be an object")
        annotation_id = reference.get("annotation_id")
        text = reference.get("text")
        if (
            isinstance(annotation_id, bool)
            or not isinstance(annotation_id, int)
            or annotation_id < 1
        ):
            raise ValueError(f"task reference {index} has invalid annotation_id")
        if annotation_id in seen_annotation_ids:
            raise ValueError(f"duplicate task reference annotation_id {annotation_id}")
        seen_annotation_ids.add(annotation_id)
        if not isinstance(text, str) or not text:
            raise ValueError(f"task reference {index} has invalid text")
        if not normalize_reference_text(text):
            raise ValueError(f"task reference {index} text has no normalized tokens")
        scored.append(
            {
                "annotation_id": annotation_id,
                "token_f1": token_f1(prediction, text),
                "rouge_l_f1": rouge_l_f1(prediction, text),
            }
        )
    return scored


def score_reference_batch(
    decoded_texts: Sequence[str] | None,
    task_references: Sequence[Mapping[str, Any] | None] | None,
) -> dict[str, Any]:
    """评分一批输出；缺 decoded text/reference 时返回显式 unavailable。"""

    if decoded_texts is None or task_references is None:
        return {
            "available": False,
            "reason": "benchmark record lacks decoded_texts or task_references",
            "request_count": 0,
            "scores": [],
        }
    if len(decoded_texts) != len(task_references):
        raise ValueError("decoded_texts and task_references must have the same request count")
    if not decoded_texts:
        raise ValueError("reference quality batch must not be empty")
    missing = [
        index for index, task_reference in enumerate(task_references) if task_reference is None
    ]
    if missing:
        return {
            "available": False,
            "reason": f"requests lack task references: {missing}",
            "request_count": len(decoded_texts),
            "scores": [],
        }

    scores = [
        score_reference_text(text, task_reference)
        for text, task_reference in zip(
            decoded_texts,
            task_references,
            strict=True,
        )
        if task_reference is not None
    ]
    return {
        "available": True,
        "reason": None,
        "request_count": len(scores),
        "scores": scores,
    }
