"""P5.1 visual token importance scoring.

本模块只读取 P4 KV trace JSONL records，不运行模型、不修改 KV cache、
不改变推理输出。输入 trace records 的关键字段来自
`prism_infer.analysis.kv_trace.record_attention_layer`:

- `batch.sequences[].spans`: sequence 内 text/image/video token span。
- `attention.sequence_stats[].span_masses`: 当前 query 对 span 的 attention mass。
- `attention.sequence_stats[].top_visual_tokens`: 已记录的 top-k visual token。
- `span_stats`: prefill 阶段可用的 span-level K/V norm 统计。

P5.1 输出是 pruning 前的离线 ranking proxy，不是压缩策略本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


VISUAL_MODALITIES = {"image", "video"}
DEFAULT_KEEP_RATIOS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ImportanceWeights:
    """Visual token scoring weights.

    score = token_attention_mass * (
        attention_mass
        + entropy_focus * (1 - visual_entropy_norm)
        + k_norm * visual_text_k_norm_ratio
    )

    K norm is intentionally weak by default because P4 showed visual/text K norm
    ratio alone is not a deletion criterion.
    """

    attention_mass: float = 1.0
    entropy_focus: float = 0.5
    k_norm: float = 0.1

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    def to_dict(self) -> dict[str, float]:
        return {
            "attention_mass": float(self.attention_mass),
            "entropy_focus": float(self.entropy_focus),
            "k_norm": float(self.k_norm),
        }


@dataclass
class _TokenAggregate:
    seq_id: int
    modality: str
    span_index: int
    token_index: int
    score_sum: float = 0.0
    attention_mass_sum: float = 0.0
    entropy_focus_sum: float = 0.0
    observation_count: int = 0
    top_token_observation_count: int = 0
    k_norm_ratio_values: list[float] = field(default_factory=list)
    layers: set[int] = field(default_factory=set)
    phases: set[str] = field(default_factory=set)
    steps: set[int] = field(default_factory=set)

    def add(
        self,
        *,
        score: float,
        attention_mass: float,
        entropy_focus: float,
        k_norm_ratio: float | None,
        observed_top_token: bool,
        layer_id: int,
        phase: str,
        step_id: int,
    ) -> None:
        self.score_sum += score
        self.attention_mass_sum += attention_mass
        self.entropy_focus_sum += entropy_focus
        self.observation_count += 1
        if observed_top_token:
            self.top_token_observation_count += 1
        if k_norm_ratio is not None:
            self.k_norm_ratio_values.append(k_norm_ratio)
        self.layers.add(layer_id)
        self.phases.add(phase)
        self.steps.add(step_id)

    def to_dict(self) -> dict[str, Any]:
        k_norm_ratio_mean = None
        if self.k_norm_ratio_values:
            k_norm_ratio_mean = sum(self.k_norm_ratio_values) / len(self.k_norm_ratio_values)
        entropy_focus_mean = 0.0
        score_mean = 0.0
        attention_mass_mean = 0.0
        top_token_fraction = 0.0
        if self.observation_count:
            entropy_focus_mean = self.entropy_focus_sum / self.observation_count
            score_mean = self.score_sum / self.observation_count
            attention_mass_mean = self.attention_mass_sum / self.observation_count
            top_token_fraction = self.top_token_observation_count / self.observation_count
        return {
            "seq_id": self.seq_id,
            "modality": self.modality,
            "span_index": self.span_index,
            "token_index": self.token_index,
            "score_sum": self.score_sum,
            "score_mean": score_mean,
            "attention_mass_sum": self.attention_mass_sum,
            "attention_mass_mean": attention_mass_mean,
            "entropy_focus_mean": entropy_focus_mean,
            "k_norm_ratio_mean": k_norm_ratio_mean,
            "observation_count": self.observation_count,
            "top_token_observation_count": self.top_token_observation_count,
            "top_token_observation_fraction": top_token_fraction,
            "layers": sorted(self.layers),
            "phases": sorted(self.phases),
            "steps": sorted(self.steps),
        }


@dataclass
class _SpanAggregate:
    seq_id: int
    modality: str
    span_index: int
    token_start: int
    token_end: int
    score_sum: float = 0.0
    attention_mass_sum: float = 0.0
    entropy_focus_sum: float = 0.0
    observation_count: int = 0
    k_norm_ratio_values: list[float] = field(default_factory=list)
    layers: set[int] = field(default_factory=set)
    phases: set[str] = field(default_factory=set)
    steps: set[int] = field(default_factory=set)

    def add(
        self,
        *,
        score: float,
        attention_mass: float,
        entropy_focus: float,
        k_norm_ratio: float | None,
        layer_id: int,
        phase: str,
        step_id: int,
    ) -> None:
        self.score_sum += score
        self.attention_mass_sum += attention_mass
        self.entropy_focus_sum += entropy_focus
        self.observation_count += 1
        if k_norm_ratio is not None:
            self.k_norm_ratio_values.append(k_norm_ratio)
        self.layers.add(layer_id)
        self.phases.add(phase)
        self.steps.add(step_id)

    def to_dict(self) -> dict[str, Any]:
        k_norm_ratio_mean = None
        if self.k_norm_ratio_values:
            k_norm_ratio_mean = sum(self.k_norm_ratio_values) / len(self.k_norm_ratio_values)
        entropy_focus_mean = 0.0
        score_mean = 0.0
        attention_mass_mean = 0.0
        if self.observation_count:
            entropy_focus_mean = self.entropy_focus_sum / self.observation_count
            score_mean = self.score_sum / self.observation_count
            attention_mass_mean = self.attention_mass_sum / self.observation_count
        return {
            "seq_id": self.seq_id,
            "modality": self.modality,
            "span_index": self.span_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_end - self.token_start,
            "score_sum": self.score_sum,
            "score_mean": score_mean,
            "attention_mass_sum": self.attention_mass_sum,
            "attention_mass_mean": attention_mass_mean,
            "entropy_focus_mean": entropy_focus_mean,
            "k_norm_ratio_mean": k_norm_ratio_mean,
            "observation_count": self.observation_count,
            "layers": sorted(self.layers),
            "phases": sorted(self.phases),
            "steps": sorted(self.steps),
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _entropy_focus(seq_stat: dict[str, Any]) -> float:
    entropy_norm = seq_stat.get("visual_attention_entropy_normalized_mean")
    if entropy_norm is None:
        return 0.0
    return 1.0 - _clamp(float(entropy_norm), 0.0, 1.0)


def _top_visual_token_scores(seq_stat: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in seq_stat.get("top_visual_tokens", []) or []:
        token_index = int(item["token_index"])
        score = float(item["score"])
        result[token_index] = max(score, result.get(token_index, 0.0))
    return result


def _text_k_norm_by_seq(record: dict[str, Any]) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for stat in record.get("span_stats", []) or []:
        if stat.get("modality") != "text":
            continue
        seq_id = int(stat.get("seq_id", 0))
        values.setdefault(seq_id, []).append(float(stat["k_norm_mean"]))
    return {
        seq_id: sum(items) / len(items)
        for seq_id, items in values.items()
        if items
    }


def _span_k_norm_ratio(
    record: dict[str, Any],
    *,
    seq_id: int,
    modality: str,
    span_index: int,
    token_start: int,
    token_end: int,
) -> float | None:
    text_norms = _text_k_norm_by_seq(record)
    text_norm = text_norms.get(seq_id)
    if text_norm in (None, 0.0):
        return None
    for stat in record.get("span_stats", []) or []:
        if int(stat.get("seq_id", 0)) != seq_id:
            continue
        if stat.get("modality") != modality:
            continue
        if int(stat.get("span_index", -1)) != span_index:
            continue
        if int(stat.get("start", -1)) != token_start:
            continue
        if int(stat.get("end", -1)) != token_end:
            continue
        return float(stat["k_norm_mean"]) / text_norm
    return None


def _score_factor(
    *,
    weights: ImportanceWeights,
    entropy_focus: float,
    k_norm_ratio: float | None,
) -> float:
    factor = weights.attention_mass + weights.entropy_focus * entropy_focus
    if k_norm_ratio is not None:
        factor += weights.k_norm * k_norm_ratio
    return factor


def _span_score(
    *,
    attention_mass: float,
    entropy_focus: float,
    k_norm_ratio: float | None,
    weights: ImportanceWeights,
) -> float:
    return attention_mass * _score_factor(
        weights=weights,
        entropy_focus=entropy_focus,
        k_norm_ratio=k_norm_ratio,
    )


def _token_masses_for_span(
    *,
    token_start: int,
    token_end: int,
    span_attention_mass: float,
    top_visual_tokens: dict[int, float],
) -> dict[int, tuple[float, bool]]:
    token_count = token_end - token_start
    if token_count <= 0:
        return {}

    top_in_span = {
        token_index: score
        for token_index, score in top_visual_tokens.items()
        if token_start <= token_index < token_end
    }
    top_mass = sum(top_in_span.values())
    remaining_count = token_count - len(top_in_span)
    fallback_mass = 0.0
    if remaining_count > 0:
        fallback_mass = max(span_attention_mass - top_mass, 0.0) / remaining_count

    result: dict[int, tuple[float, bool]] = {}
    for token_index in range(token_start, token_end):
        if token_index in top_in_span:
            result[token_index] = (top_in_span[token_index], True)
        else:
            result[token_index] = (fallback_mass, False)
    return result


def _get_layer_id(record: dict[str, Any]) -> int:
    return int(record["layer_id"])


def _get_step_id(record: dict[str, Any]) -> int:
    return int(record.get("step_id", 0))


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    layer_records = [record for record in records if record.get("record_type") == "attention_layer"]
    layers = sorted({int(record["layer_id"]) for record in layer_records if record.get("layer_id") is not None})
    phases = sorted({str(record.get("phase")) for record in layer_records})
    return {
        "schema_versions": sorted(
            {
                int(record["schema_version"])
                for record in records
                if record.get("schema_version") is not None
            }
        ),
        "trace_headers": sum(1 for record in records if record.get("record_type") == "trace_header"),
        "layer_records": len(layer_records),
        "layers": layers,
        "phases": phases,
    }


def _sorted_token_scores(token_scores: dict[tuple[int, str, int, int], _TokenAggregate]) -> list[dict[str, Any]]:
    rows = [aggregate.to_dict() for aggregate in token_scores.values()]
    rows.sort(
        key=lambda row: (
            -float(row["score_sum"]),
            int(row["seq_id"]),
            str(row["modality"]),
            int(row["token_index"]),
        )
    )
    return rows


def _sorted_span_scores(span_scores: dict[tuple[int, str, int], _SpanAggregate]) -> list[dict[str, Any]]:
    rows = [aggregate.to_dict() for aggregate in span_scores.values()]
    rows.sort(
        key=lambda row: (
            -float(row["score_sum"]),
            int(row["seq_id"]),
            str(row["modality"]),
            int(row["span_index"]),
        )
    )
    return rows


def simulate_keep_ratios(
    token_scores: list[dict[str, Any]],
    keep_ratios: tuple[float, ...] = DEFAULT_KEEP_RATIOS,
) -> list[dict[str, Any]]:
    """Simulate keeping the top scored visual tokens for each keep ratio."""

    total_tokens = len(token_scores)
    total_score = sum(float(row["score_sum"]) for row in token_scores)
    simulations = []
    for ratio in keep_ratios:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"keep ratio must be in [0, 1], got {ratio}")
        keep_count = int(math.ceil(total_tokens * ratio)) if total_tokens else 0
        keep_count = min(keep_count, total_tokens)
        kept = token_scores[:keep_count]
        kept_score = sum(float(row["score_sum"]) for row in kept)
        simulations.append(
            {
                "keep_ratio": float(ratio),
                "total_visual_tokens": total_tokens,
                "keep_count": keep_count,
                "drop_count": total_tokens - keep_count,
                "kept_score_sum": kept_score,
                "total_score_sum": total_score,
                "kept_score_fraction": 0.0 if total_score == 0.0 else kept_score / total_score,
                "kept_tokens": [
                    {
                        "seq_id": row["seq_id"],
                        "modality": row["modality"],
                        "span_index": row["span_index"],
                        "token_index": row["token_index"],
                        "score_sum": row["score_sum"],
                    }
                    for row in kept
                ],
            }
        )
    return simulations


def score_visual_importance(
    records: list[dict[str, Any]],
    *,
    weights: ImportanceWeights | None = None,
    keep_ratios: tuple[float, ...] = DEFAULT_KEEP_RATIOS,
    top_k: int = 20,
) -> dict[str, Any]:
    """Score visual tokens from P4 KV trace records.

    The returned ranking is offline analysis evidence for P5.2. It does not
    mutate inference runtime or KV cache state.
    """

    active_weights = weights or ImportanceWeights()
    token_scores: dict[tuple[int, str, int, int], _TokenAggregate] = {}
    span_scores: dict[tuple[int, str, int], _SpanAggregate] = {}
    visual_observation_count = 0

    for record in records:
        if record.get("record_type") != "attention_layer":
            continue
        attention = record.get("attention", {})
        if not attention.get("available", False):
            continue
        layer_id = _get_layer_id(record)
        step_id = _get_step_id(record)
        phase = str(record.get("phase", "unknown"))
        for seq_stat in attention.get("sequence_stats", []) or []:
            seq_id = int(seq_stat.get("seq_id", 0))
            focus = _entropy_focus(seq_stat)
            top_visual_tokens = _top_visual_token_scores(seq_stat)
            for span_mass in seq_stat.get("span_masses", []) or []:
                modality = str(span_mass.get("modality"))
                if modality not in VISUAL_MODALITIES:
                    continue
                span_index = int(span_mass.get("span_index", 0))
                token_start = int(span_mass["start"])
                token_end = int(span_mass["end"])
                token_count = token_end - token_start
                if token_count <= 0:
                    continue
                attention_mass = _as_float(span_mass.get("mass_mean"))
                k_norm_ratio = _span_k_norm_ratio(
                    record,
                    seq_id=seq_id,
                    modality=modality,
                    span_index=span_index,
                    token_start=token_start,
                    token_end=token_end,
                )
                visual_observation_count += 1

                span_key = (seq_id, modality, span_index)
                if span_key not in span_scores:
                    span_scores[span_key] = _SpanAggregate(
                        seq_id=seq_id,
                        modality=modality,
                        span_index=span_index,
                        token_start=token_start,
                        token_end=token_end,
                    )
                span_scores[span_key].add(
                    score=_span_score(
                        attention_mass=attention_mass,
                        entropy_focus=focus,
                        k_norm_ratio=k_norm_ratio,
                        weights=active_weights,
                    ),
                    attention_mass=attention_mass,
                    entropy_focus=focus,
                    k_norm_ratio=k_norm_ratio,
                    layer_id=layer_id,
                    phase=phase,
                    step_id=step_id,
                )

                token_masses = _token_masses_for_span(
                    token_start=token_start,
                    token_end=token_end,
                    span_attention_mass=attention_mass,
                    top_visual_tokens=top_visual_tokens,
                )
                for token_index, (token_mass, observed_top_token) in token_masses.items():
                    token_key = (seq_id, modality, span_index, token_index)
                    if token_key not in token_scores:
                        token_scores[token_key] = _TokenAggregate(
                            seq_id=seq_id,
                            modality=modality,
                            span_index=span_index,
                            token_index=token_index,
                        )
                    token_scores[token_key].add(
                        score=_span_score(
                            attention_mass=token_mass,
                            entropy_focus=focus,
                            k_norm_ratio=k_norm_ratio,
                            weights=active_weights,
                        ),
                        attention_mass=token_mass,
                        entropy_focus=focus,
                        k_norm_ratio=k_norm_ratio,
                        observed_top_token=observed_top_token,
                        layer_id=layer_id,
                        phase=phase,
                        step_id=step_id,
                    )

    ranked_tokens = _sorted_token_scores(token_scores)
    ranked_spans = _sorted_span_scores(span_scores)
    token_count = len(ranked_tokens)
    return {
        "schema_version": 1,
        "record_type": "visual_importance_report",
        "weights": active_weights.to_dict(),
        "source": _source_summary(records),
        "total_visual_tokens": token_count,
        "visual_span_observations": visual_observation_count,
        "total_token_observations": sum(int(row["observation_count"]) for row in ranked_tokens),
        "top_tokens": ranked_tokens[: max(0, top_k)],
        "bottom_tokens": list(reversed(ranked_tokens[-max(0, top_k):])) if top_k > 0 else [],
        "token_scores": ranked_tokens,
        "span_scores": ranked_spans,
        "keep_ratio_simulations": simulate_keep_ratios(ranked_tokens, keep_ratios),
        "limitations": [
            "P5.1 is offline analysis only; it does not modify runtime KV cache.",
            "Token scores are attention-derived proxies, not a validated pruning mask.",
            "Full per-token attention distribution is not stored in P4 trace; top_visual_tokens "
            "refines mass allocation only for recorded top-k tokens.",
            "Compression ratio, memory benefit, latency, and quality degradation are not claimed "
            "until an active pruning strategy is implemented and benchmarked.",
        ],
    }


def format_importance_markdown(report: dict[str, Any], *, top_k: int = 20) -> str:
    """Render a visual importance report as Markdown."""

    source = report["source"]
    lines = [
        "# P5.1 Visual Token Importance Report",
        "",
        "## Summary",
        "",
        f"- layer records: `{source['layer_records']}`",
        f"- layers: `{source['layers']}`",
        f"- phases: `{source['phases']}`",
        f"- visual tokens ranked: `{report['total_visual_tokens']}`",
        f"- visual span observations: `{report['visual_span_observations']}`",
        f"- token observations: `{report['total_token_observations']}`",
        f"- weights: `{report['weights']}`",
        "",
        "## Top Visual Tokens",
        "",
        "| rank | seq | modality | span | token | score sum | attn mass sum | observations | top-hit fraction |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["top_tokens"][:top_k], start=1):
        lines.append(
            "| "
            f"{rank} | {row['seq_id']} | {row['modality']} | {row['span_index']} | "
            f"{row['token_index']} | {row['score_sum']:.6e} | "
            f"{row['attention_mass_sum']:.6e} | {row['observation_count']} | "
            f"{row['top_token_observation_fraction']:.3f} |"
        )
    if not report["top_tokens"]:
        lines.append("| 0 | - | - | - | - | 0.000000e+00 | 0.000000e+00 | 0 | 0.000 |")

    lines.extend(
        [
            "",
            "## Keep Ratio Simulation",
            "",
            "| keep ratio | keep | drop | kept score fraction |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in report["keep_ratio_simulations"]:
        lines.append(
            "| "
            f"{row['keep_ratio']:.2f} | {row['keep_count']} | {row['drop_count']} | "
            f"{row['kept_score_fraction']:.6f} |"
        )

    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
