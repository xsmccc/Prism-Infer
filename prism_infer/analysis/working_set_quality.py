"""Pure helpers for repeated-visual-context quality comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_metrics import aggregate_quality_predictions
from prism_infer.analysis.working_set_plan import (
    build_muirbench_media_first_source_prompt,
)


@dataclass(frozen=True, slots=True)
class QualityStageSpec:
    """One independently runnable quality comparison stage."""

    dataset_id: str
    prompt_layout: str
    compression_mode: str
    pruning_strategy: str
    enable_prefix_caching: bool
    reuse_scope: str
    keep_ratio_override: float | None
    attention_selection_scope: str | None
    comparison_role: str

    @property
    def uses_compaction(self) -> bool:
        return self.compression_mode == "visual_compact_scaled_fp8"

    def effective_keep_ratio(self, requested_keep_ratio: float) -> float:
        """Return the stage-specific retention ratio used by the runtime."""

        if self.keep_ratio_override is not None:
            return self.keep_ratio_override
        return requested_keep_ratio

    @property
    def requires_attention_selection(self) -> bool:
        return self.attention_selection_scope is not None


QUALITY_STAGE_SPECS: dict[str, QualityStageSpec] = {
    "muir_dense_official": QualityStageSpec(
        dataset_id="muirbench_test",
        prompt_layout="official_interleaved",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="per_request",
        keep_ratio_override=1.0,
        attention_selection_scope=None,
        comparison_role="dense_official_layout",
    ),
    "muir_dense_media_first": QualityStageSpec(
        dataset_id="muirbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=1.0,
        attention_selection_scope=None,
        comparison_role="dense_media_first",
    ),
    "muir_attention_per_question": QualityStageSpec(
        dataset_id="muirbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="attention",
        enable_prefix_caching=True,
        reuse_scope="per_request",
        keep_ratio_override=None,
        attention_selection_scope="per_question",
        comparison_role="question_conditioned_attention",
    ),
    "muir_attention_first_reuse": QualityStageSpec(
        dataset_id="muirbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="attention",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=None,
        attention_selection_scope="first_question",
        comparison_role="first_question_attention_reused",
    ),
    "muir_uniform_reuse": QualityStageSpec(
        dataset_id="muirbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=None,
        attention_selection_scope=None,
        comparison_role="query_agnostic_uniform_reused",
    ),
    "docvqa_dense": QualityStageSpec(
        dataset_id="docvqa_validation",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=1.0,
        attention_selection_scope=None,
        comparison_role="dense_reference",
    ),
    "docvqa_uniform": QualityStageSpec(
        dataset_id="docvqa_validation",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=None,
        attention_selection_scope=None,
        comparison_role="query_agnostic_uniform_reused",
    ),
    "mvbench_dense": QualityStageSpec(
        dataset_id="mvbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=1.0,
        attention_selection_scope=None,
        comparison_role="dense_reference",
    ),
    "mvbench_uniform": QualityStageSpec(
        dataset_id="mvbench_test",
        prompt_layout="media_first",
        compression_mode="visual_compact_scaled_fp8",
        pruning_strategy="uniform",
        enable_prefix_caching=True,
        reuse_scope="media_group",
        keep_ratio_override=None,
        attention_selection_scope=None,
        comparison_role="query_agnostic_uniform_reused",
    ),
}


@dataclass(frozen=True, slots=True)
class GroupedQualitySample:
    """A selected sample annotated with deterministic media-group position."""

    record: Mapping[str, Any]
    media_group_id: str
    question_index: int
    group_size: int


def ordered_media_sha256(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact ordered media identity used for cross-question reuse."""

    media = record.get("media")
    if not isinstance(media, list) or not media:
        raise ValueError("quality record must contain a non-empty media list")
    digests = tuple(str(item.get("sha256", "")) for item in media if isinstance(item, Mapping))
    if len(digests) != len(media) or any(
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise ValueError("quality record media SHA256 identities are invalid")
    return digests


def media_group_id(
    record: Mapping[str, Any],
    *,
    sampling_identity: Mapping[str, Any] | None = None,
) -> str:
    """Build the exact deterministic visual-context identity for grouping."""

    temporal_bound = record.get("temporal_bound")
    if temporal_bound is not None and not isinstance(temporal_bound, Mapping):
        raise ValueError("quality record temporal_bound must be a mapping or None")
    if sampling_identity is not None and not isinstance(sampling_identity, Mapping):
        raise ValueError("sampling_identity must be a mapping or None")
    return canonical_json_sha256(
        {
            "ordered_media_sha256": list(ordered_media_sha256(record)),
            "temporal_bound": (
                None
                if temporal_bound is None
                else {str(key): temporal_bound[key] for key in sorted(temporal_bound)}
            ),
            "sampling_identity": (None if sampling_identity is None else dict(sampling_identity)),
        }
    )


def select_multi_question_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    max_groups: int | None = None,
    sampling_identity: Mapping[str, Any] | None = None,
) -> list[GroupedQualitySample]:
    """Select exact-media groups with at least two questions.

    Groups are ordered by content digest. Records inside a group retain source
    order, which gives an explicit and reproducible meaning to "first question".
    """

    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive or None")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identities: dict[str, str] = {}
    for record in records:
        group_id = media_group_id(record, sampling_identity=sampling_identity)
        exact_identity = canonical_json_sha256(
            {
                "ordered_media_sha256": list(ordered_media_sha256(record)),
                "temporal_bound": record.get("temporal_bound"),
                "sampling_identity": sampling_identity,
            }
        )
        previous = identities.setdefault(group_id, exact_identity)
        if previous != exact_identity:
            raise RuntimeError("visual-context group digest collision")
        grouped[group_id].append(record)

    selected_group_ids = sorted(
        group_id for group_id, samples in grouped.items() if len(samples) >= 2
    )
    if max_groups is not None:
        selected_group_ids = selected_group_ids[:max_groups]
    selected: list[GroupedQualitySample] = []
    for group_id in selected_group_ids:
        samples = grouped[group_id]
        selected.extend(
            GroupedQualitySample(
                record=record,
                media_group_id=group_id,
                question_index=question_index,
                group_size=len(samples),
            )
            for question_index, record in enumerate(samples)
        )
    return selected


def build_muirbench_media_first_prompt(
    question: str,
    options: Sequence[str],
    *,
    expected_media_count: int | None = None,
    image_marker: str = "<image>",
) -> str:
    """Move ordered media first and preserve every interleaved image reference."""

    if expected_media_count is None:
        expected_media_count = question.count(image_marker) + sum(
            option.count(image_marker) for option in options
        )
    return build_muirbench_media_first_source_prompt(
        question,
        options,
        expected_media_count=expected_media_count,
        image_marker=image_marker,
    )


def sample_deleted_visual_tokens(sample: Mapping[str, Any]) -> int:
    """Return the measured token deletion count for one request."""

    decision = sample.get("compression_decision")
    if not isinstance(decision, Mapping):
        return 0
    return max(0, int(decision.get("dropped_visual_tokens", 0)))


def summarize_stage_samples(
    dataset_id: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize all requests and the subset that actually deleted tokens."""

    deleted = [sample for sample in samples if sample_deleted_visual_tokens(sample) > 0]
    followups = [sample for sample in samples if int(sample.get("question_index", 0)) > 0]
    prefix_hits = [sample for sample in followups if bool(sample.get("pre_admission_hit"))]
    selection_reuses = [
        sample for sample in followups if bool(sample.get("first_question_selection_reused"))
    ]
    return {
        "all_samples": aggregate_quality_predictions(dataset_id, samples),
        "actual_token_deletion": {
            "sample_ids_sha256": canonical_json_sha256(
                [str(sample["sample_id"]) for sample in deleted]
            ),
            "dropped_visual_tokens": sum(
                sample_deleted_visual_tokens(sample) for sample in deleted
            ),
            "quality": aggregate_quality_predictions(dataset_id, deleted),
        },
        "cross_question_reuse": {
            "followup_requests": len(followups),
            "pre_admission_hits": len(prefix_hits),
            "first_question_selection_reuses": len(selection_reuses),
        },
    }


def paired_deleted_subset(
    *,
    dataset_id: str,
    reference_samples: Sequence[Mapping[str, Any]],
    candidate_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one compact candidate only where it deleted visual tokens."""

    reference_by_id = {str(sample["sample_id"]): sample for sample in reference_samples}
    candidate_by_id = {str(sample["sample_id"]): sample for sample in candidate_samples}
    deleted_ids = [
        str(sample["sample_id"])
        for sample in candidate_samples
        if sample_deleted_visual_tokens(sample) > 0
    ]
    missing_reference = [sample_id for sample_id in deleted_ids if sample_id not in reference_by_id]
    if missing_reference:
        raise ValueError(
            "reference artifact is missing compacted sample IDs: " + ", ".join(missing_reference)
        )
    reference = [reference_by_id[sample_id] for sample_id in deleted_ids]
    candidate = [candidate_by_id[sample_id] for sample_id in deleted_ids]
    return {
        "actual_token_deletion_samples": len(deleted_ids),
        "sample_ids_sha256": canonical_json_sha256(deleted_ids),
        "dropped_visual_tokens": sum(sample_deleted_visual_tokens(sample) for sample in candidate),
        "reference_quality": aggregate_quality_predictions(dataset_id, reference),
        "candidate_quality": aggregate_quality_predictions(dataset_id, candidate),
    }
