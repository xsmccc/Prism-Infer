"""P6.12 dataset-level pruning fidelity 汇总测试。"""

from copy import deepcopy

import pytest

from prism_infer.analysis.pruning_fidelity import (
    render_pruning_fidelity_markdown,
    summarize_pruning_fidelity_records,
)


def _record(
    *,
    case_id: str,
    mode_name: str,
    strategy: str,
    tokens: list[list[int]],
    physical_tokens: int,
    active_bytes: int,
    kept_by_span: list[int] | None = None,
) -> dict[str, object]:
    num_requests = len(tokens)
    layouts = []
    for request_index in range(num_requests):
        decision = None
        if mode_name != "off_graph":
            decision = {
                "kept_visual_tokens": 2,
                "visual_token_spans": [
                    {
                        "modality": "image",
                        "index": 0,
                        "token_count": 4,
                    }
                ],
            }
            if kept_by_span is not None:
                decision["kept_visual_tokens_by_span"] = [
                    {
                        "modality": "image",
                        "span_index": 0,
                        "kept_tokens": kept_by_span[request_index],
                    }
                ]
        layouts.append({"compression_record": decision})
    return {
        "environment": {"git_commit": "abc123", "git_dirty": False},
        "model": {
            "path": "/model",
            "dtype": "torch.bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": 1280,
            "max_num_batched_tokens": 2048,
            "kvcache_block_size": 256,
            "num_kvcache_blocks": 16,
            "prefix_caching_enabled": False,
        },
        "mode": {
            "name": mode_name,
            "execution": "cuda_graph",
            "compression": "off" if mode_name == "off_graph" else "visual_compact",
            "visual_pruning_strategy": strategy,
            "visual_pruning_keep_ratio": 0.5,
            "visual_pruning_min_keep_tokens": 1,
            "visual_pruning_attention_last_n_layers": 4,
        },
        "workload": {
            "manifest_name": "fidelity_test",
            "manifest_sha256": "0" * 64,
            "case_id": case_id,
            "num_requests": num_requests,
            "max_tokens": 4,
            "prompt_tokens": 20 * num_requests,
            "image_tokens": 4 * num_requests,
            "video_tokens": 0,
            "request_types": ["image_file"] * num_requests,
            "input_shapes": [
                {"type": "image_file", "visual_shapes": [[4, 4, 3]]}
                for _ in range(num_requests)
            ],
        },
        "traffic": {
            "kind": "offline_closed_loop",
            "batch_size": num_requests,
            "concurrency": num_requests,
        },
        "correctness": {"token_ids": tokens},
        "kv_cache": {
            "logical_prompt_tokens": 20 * num_requests,
            "physical_prompt_tokens": physical_tokens,
            "active_prompt_bytes": active_bytes,
            "layouts": layouts,
        },
    }


def test_pruning_fidelity_aggregates_cases_and_strategies() -> None:
    records = [
        _record(
            case_id="a",
            mode_name="off_graph",
            strategy="uniform",
            tokens=[[1, 2, 3, 4], [5, 6, 7, 8]],
            physical_tokens=40,
            active_bytes=400,
        ),
        _record(
            case_id="a",
            mode_name="visual_compact_graph",
            strategy="uniform",
            tokens=[[1, 2, 9, 4], [5, 6, 7, 8]],
            physical_tokens=24,
            active_bytes=200,
        ),
        _record(
            case_id="a",
            mode_name="visual_compact_graph",
            strategy="attention",
            tokens=[[1, 2, 3, 4], [5, 6, 0, 8]],
            physical_tokens=24,
            active_bytes=200,
            kept_by_span=[2, 2],
        ),
        _record(
            case_id="b",
            mode_name="off_graph",
            strategy="uniform",
            tokens=[[9, 8, 7, 6]],
            physical_tokens=20,
            active_bytes=200,
        ),
        _record(
            case_id="b",
            mode_name="visual_compact_graph",
            strategy="uniform",
            tokens=[[9, 0, 7, 6]],
            physical_tokens=12,
            active_bytes=100,
        ),
        _record(
            case_id="b",
            mode_name="visual_compact_graph",
            strategy="attention",
            tokens=[[9, 8, 7, 6]],
            physical_tokens=12,
            active_bytes=100,
            kept_by_span=[2],
        ),
    ]

    summary = summarize_pruning_fidelity_records(records)
    aggregates = {
        row["candidate"]["strategy"]: row for row in summary["aggregates"]
    }
    attention = aggregates["attention"]
    uniform = aggregates["uniform"]
    markdown = render_pruning_fidelity_markdown(summary)

    print(f"P6.12 fidelity attention aggregate: {attention}")
    assert attention["case_count"] == 2
    assert attention["request_count"] == 3
    assert attention["exact_request_rate"] == pytest.approx(2 / 3)
    assert attention["stable_prefix_ratio_micro"] == pytest.approx(10 / 12)
    assert attention["physical_token_ratio"] == pytest.approx(0.6)
    assert attention["active_prompt_bytes_ratio"] == pytest.approx(0.5)
    assert attention["span_audit"]["available"] is True
    assert attention["span_audit"]["zero_kept_visual_spans"] == 0
    assert uniform["stable_prefix_ratio_micro"] == pytest.approx(7 / 12)
    assert uniform["span_audit"]["available"] is False
    assert "visual_compact_graph/attention:last4" in markdown
    print("P6.12 dataset-level pruning fidelity aggregation: PASS")


def test_pruning_fidelity_rejects_incomplete_candidate_coverage() -> None:
    baseline_a = _record(
        case_id="a",
        mode_name="off_graph",
        strategy="uniform",
        tokens=[[1, 2]],
        physical_tokens=20,
        active_bytes=200,
    )
    baseline_b = deepcopy(baseline_a)
    baseline_b["workload"]["case_id"] = "b"
    candidate_a = _record(
        case_id="a",
        mode_name="visual_compact_graph",
        strategy="attention",
        tokens=[[1, 2]],
        physical_tokens=12,
        active_bytes=100,
        kept_by_span=[2],
    )

    with pytest.raises(ValueError, match="does not cover every selected baseline case"):
        summarize_pruning_fidelity_records([baseline_a, baseline_b, candidate_a])
    print("P6.12 fidelity incomplete-dataset guard: PASS")


def test_pruning_fidelity_rejects_invalid_span_audit() -> None:
    baseline = _record(
        case_id="a",
        mode_name="off_graph",
        strategy="uniform",
        tokens=[[1, 2]],
        physical_tokens=20,
        active_bytes=200,
    )
    candidate = _record(
        case_id="a",
        mode_name="visual_compact_graph",
        strategy="attention",
        tokens=[[1, 2]],
        physical_tokens=12,
        active_bytes=100,
        kept_by_span=[2],
    )
    candidate["kv_cache"]["layouts"][0]["compression_record"][
        "kept_visual_tokens_by_span"
    ][0]["kept_tokens"] = 1

    with pytest.raises(ValueError, match="audit sum does not match"):
        summarize_pruning_fidelity_records([baseline, candidate])
    print("P6.12 fidelity span-audit consistency guard: PASS")


def test_pruning_fidelity_rejects_incomparable_baseline() -> None:
    baseline = _record(
        case_id="a",
        mode_name="off_graph",
        strategy="uniform",
        tokens=[[1, 2]],
        physical_tokens=20,
        active_bytes=200,
    )
    candidate = _record(
        case_id="a",
        mode_name="visual_compact_graph",
        strategy="attention",
        tokens=[[1, 2]],
        physical_tokens=12,
        active_bytes=100,
        kept_by_span=[2],
    )
    candidate["model"]["dtype"] = "torch.float16"

    with pytest.raises(ValueError, match="model.dtype"):
        summarize_pruning_fidelity_records([baseline, candidate])
    print("P6.12 fidelity comparability guard: PASS")


def test_pruning_fidelity_rejects_duplicate_baseline() -> None:
    baseline = _record(
        case_id="a",
        mode_name="off_graph",
        strategy="uniform",
        tokens=[[1, 2]],
        physical_tokens=20,
        active_bytes=200,
    )
    candidate = _record(
        case_id="a",
        mode_name="visual_compact_graph",
        strategy="attention",
        tokens=[[1, 2]],
        physical_tokens=12,
        active_bytes=100,
        kept_by_span=[2],
    )

    with pytest.raises(ValueError, match="duplicate 'off_graph' baseline"):
        summarize_pruning_fidelity_records(
            [baseline, deepcopy(baseline), candidate]
        )
    print("P6.12 fidelity duplicate-baseline guard: PASS")
