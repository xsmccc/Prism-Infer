"""CPU contracts for the P9-C fixed-trajectory logit comparator."""

import pytest
import torch

from benchmarks.bench_kv_trajectory_quality import (
    RecordingTrajectorySampler,
    Trajectory,
    aggregate_comparisons,
    compare_trajectories,
)


def test_recording_sampler_separates_forced_tokens_from_natural_argmax() -> None:
    sampler = RecordingTrajectorySampler([2, 0])
    first = sampler(torch.tensor([[0.0, 4.0, 1.0]]), torch.tensor([0.0]))
    second = sampler(torch.tensor([[3.0, 1.0, 2.0]]), torch.tensor([0.0]))
    trajectory = sampler.finish([2, 0])

    assert first.tolist() == [2]
    assert second.tolist() == [0]
    assert trajectory.token_ids == [2, 0]
    assert trajectory.natural_argmax_ids == [1, 0]
    assert list(trajectory.logits.shape) == [2, 3]


def test_recording_sampler_rejects_non_greedy_baseline_output() -> None:
    sampler = RecordingTrajectorySampler()
    sampler(torch.tensor([[0.0, 4.0, 1.0]]), torch.tensor([0.0]))

    with pytest.raises(RuntimeError, match="natural greedy trajectory"):
        sampler.finish([2])


def test_trajectory_comparison_reports_distribution_and_ppl_drift() -> None:
    baseline = Trajectory(
        token_ids=[1, 2],
        natural_argmax_ids=[1, 2],
        logits=torch.tensor([[0.0, 3.0, 1.0], [0.0, 1.0, 3.0]]),
    )
    candidate = Trajectory(
        token_ids=[1, 2],
        natural_argmax_ids=[1, 0],
        logits=torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]),
    )

    result = compare_trajectories(baseline, candidate)

    assert result["steps"] == 2
    assert result["max_abs_logit_diff"] == 3.0
    assert result["mean_abs_logit_diff"] > 0.0
    assert result["ppl_ratio"] > 1.0
    assert result["mean_baseline_to_candidate_kl"] > 0.0
    assert result["natural_argmax_matches"] == 1
    assert result["natural_argmax_stable_prefix"] == 1


def test_trajectory_comparison_rejects_different_histories() -> None:
    logits = torch.zeros(2, 3)
    baseline = Trajectory([1, 2], [1, 2], logits)
    candidate = Trajectory([1, 0], [1, 0], logits)

    with pytest.raises(ValueError, match="baseline token trajectory"):
        compare_trajectories(baseline, candidate)


def test_trajectory_aggregate_uses_step_weighting() -> None:
    rows = [
        {
            "steps": 2,
            "max_abs_logit_diff": 1.0,
            "mean_abs_logit_diff": 0.5,
            "mean_baseline_to_candidate_kl": 0.1,
            "nll_delta": 0.2,
            "ppl_ratio": 1.2,
            "natural_argmax_matches": 1,
            "natural_argmax_stable_prefix": 1,
        },
        {
            "steps": 6,
            "max_abs_logit_diff": 2.0,
            "mean_abs_logit_diff": 1.5,
            "mean_baseline_to_candidate_kl": 0.3,
            "nll_delta": 0.4,
            "ppl_ratio": 1.4,
            "natural_argmax_matches": 5,
            "natural_argmax_stable_prefix": 4,
        },
    ]

    aggregate = aggregate_comparisons(rows)

    assert aggregate["steps"] == 8
    assert aggregate["max_abs_logit_diff"] == 2.0
    assert aggregate["step_weighted_mean_abs_logit_diff"] == 1.25
    assert aggregate["natural_argmax_match_ratio"] == 0.75
    assert aggregate["minimum_case_stable_prefix"] == 1
