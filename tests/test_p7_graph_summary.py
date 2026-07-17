"""P7.4-B CUDA Graph summary contract tests。"""

from copy import deepcopy

import pytest

from scripts.summarize_p7_graph import (
    EXPECTED_CATEGORIES,
    EXTERNAL_RANGES,
    REPLAY_RANGE,
    _require_padding_contract,
    _require_trace_contract,
)


def _stats(value: float) -> dict[str, float | int]:
    return {
        "count": 1,
        "median": value,
        "p90": value,
        "p99": value,
        "min": value,
        "max": value,
    }


def _trace() -> dict:
    categories = {
        name: {
            "kernel_time_fraction": 1.0 / len(EXPECTED_CATEGORIES),
            "kernels_per_range": _stats(1),
            "kernel_time_ms_per_range": _stats(1),
        }
        for name in EXPECTED_CATEGORIES
    }
    target = {
        "kernel_categories": categories,
        "cpu_range_ms_per_range": _stats(1),
        "kernel_time_ms_per_range": _stats(1),
        "kernels_per_range": _stats(1),
        "gpu_busy_ms_per_range": _stats(1),
        "gpu_span_ms_per_range": _stats(1),
        "cpu_gpu_busy_overlap_ms_per_range": _stats(1),
        "cpu_gpu_busy_overlap_fraction_per_range": _stats(1),
        "gpu_tail_after_cpu_ms_per_range": _stats(1),
        "range_count": 1,
    }
    return {
        "schema_version": 2,
        "record_type": "nsys_profile_summary",
        "target_ranges": {
            REPLAY_RANGE: target,
            **{name: deepcopy(target) for name in EXTERNAL_RANGES},
        },
    }


def _padding_records() -> list[dict]:
    baseline = [785, 2168]
    records = []
    for batch in range(1, 9):
        bucket = next(value for value in (1, 2, 4, 8) if value >= batch)
        records.append(
            {
                "environment": {"git_commit": "abc", "git_dirty": False},
                "model": {"max_num_seqs": 8},
                "traffic": {"batch_size": batch},
                "execution_backend": {
                    "cuda_graph_batch_sizes": [1, 2, 4, 8],
                    "selected_decode_batch_size": bucket,
                    "decode_batch_padding": bucket - batch,
                },
                "correctness": {
                    "outputs_identical_across_repeats": True,
                    "token_ids": [baseline] * batch,
                },
            }
        )
    return records


def test_trace_contract_requires_complete_category_partition() -> None:
    trace = _trace()
    _require_trace_contract(trace)

    invalid = deepcopy(trace)
    invalid["target_ranges"][REPLAY_RANGE]["kernel_categories"].pop("linear_gemv")
    with pytest.raises(ValueError, match="unexpected replay kernel categories"):
        _require_trace_contract(invalid)


def test_padding_contract_proves_fixed_bucket_mapping_and_output_isolation() -> None:
    records = _padding_records()
    _require_padding_contract(records)

    invalid = deepcopy(records)
    invalid[4]["execution_backend"]["decode_batch_padding"] = 0
    with pytest.raises(ValueError, match="padding mismatch"):
        _require_padding_contract(invalid)

    contaminated = deepcopy(records)
    contaminated[2]["correctness"]["token_ids"][-1] = [999]
    with pytest.raises(ValueError, match="replicated request output mismatch"):
        _require_padding_contract(contaminated)
