"""CPU contracts for the P9 CUDA Graph fixed-trajectory diagnostic."""

from types import SimpleNamespace

import torch

from benchmarks.diagnose_graph_trajectory import (
    DynamicBatchTrajectorySampler,
    compare_logit_rows,
)
from prism_infer.engine.contracts import BatchPhase, DeviceBatch, DeviceModelInputs
from prism_infer.utils.context import Context


def _decode_batch(sequence_ids: tuple[int, ...]) -> DeviceBatch:
    batch_size = len(sequence_ids)
    return DeviceBatch(
        phase=BatchPhase.DECODE,
        sequence_ids=sequence_ids,
        scheduled_token_counts=(1,) * batch_size,
        model_inputs=DeviceModelInputs(
            input_ids=torch.tensor([101, 202], dtype=torch.long)[:batch_size],
            position_ids=torch.tensor(
                [[7, 9], [8, 10], [9, 11]],
                dtype=torch.long,
            )[:, :batch_size],
        ),
        attention_context=Context(
            is_prefill=False,
            slot_mapping=torch.tensor([3, 7], dtype=torch.int32)[:batch_size],
            context_lens=torch.tensor([8, 10], dtype=torch.int32)[:batch_size],
            decode_max_context_len=torch.tensor(10, dtype=torch.int32),
            block_tables=torch.tensor(
                [[0, 1], [2, 3]],
                dtype=torch.int32,
            )[:batch_size],
        ),
        temperatures=torch.zeros(batch_size),
        execution_bucket=batch_size,
    )


def test_compare_logit_rows_reports_low_margin_argmax_change() -> None:
    baseline = torch.tensor([0.0, 4.0, 3.5, -1.0])
    candidate = torch.tensor([0.0, 3.0, 3.5, -1.0])

    comparison = compare_logit_rows(
        baseline,
        candidate,
        selected_token_id=1,
    )

    assert comparison["logits_exact"] is False
    assert comparison["nonzero_logit_count"] == 1
    assert comparison["max_abs_logit_diff"] == 1.0
    assert comparison["baseline_top2"]["token_ids"] == [1, 2]
    assert comparison["candidate_top2"]["token_ids"] == [2, 1]
    assert comparison["candidate_selected_rank"] == 2


def test_dynamic_batch_sampler_maps_rows_and_forces_baseline_history() -> None:
    runner = SimpleNamespace(enforce_eager=True)
    device_batch = _decode_batch((20, 10))
    baseline_sampler = DynamicBatchTrajectorySampler(
        {10: 0, 20: 1},
        runner=runner,
    )
    baseline_sampler.begin_batch(device_batch)
    baseline_selected = baseline_sampler(
        torch.tensor([[0.0, 1.0, 4.0], [5.0, 1.0, 0.0]]),
        torch.zeros(2),
    )
    baseline_sampler.end_batch()
    baseline = baseline_sampler.finish({10: [0], 20: [2]})

    candidate_sampler = DynamicBatchTrajectorySampler(
        {10: 0, 20: 1},
        runner=runner,
        baseline=baseline,
    )
    candidate_sampler.begin_batch(device_batch)
    candidate_selected = candidate_sampler(
        torch.tensor([[0.0, 5.0, 4.0], [5.0, 1.0, 0.0]]),
        torch.zeros(2),
    )
    candidate_sampler.end_batch()
    candidate = candidate_sampler.finish({10: [0], 20: [2]})

    assert baseline_selected.tolist() == [2, 0]
    assert candidate_selected.tolist() == [2, 0]
    assert baseline[1].natural_argmax_ids == (2,)
    assert candidate[1].natural_argmax_ids == (1,)
    assert candidate_sampler.comparisons[0]["request_index"] == 1
    assert candidate_sampler.comparisons[0]["generation_index"] == 0
    assert candidate_sampler.comparisons[0]["candidate_selected_rank"] == 2
    assert candidate_sampler.batch_trace[0]["request_indices"] == [1, 0]
    assert candidate_sampler.batch_trace[0]["rows"][0] == {
        "input_token_id": 101,
        "position_ids": [7, 8, 9],
        "slot_mapping": 3,
        "context_len": 8,
        "block_table": [0, 1],
    }


def test_graph_static_buffer_audit_normalizes_text_positions_and_padding() -> None:
    graph_vars = {
        "input_ids": torch.tensor([101, 202, 99, 88], dtype=torch.long),
        "positions": torch.tensor(
            [[7, 9, 55, 66], [7, 9, 55, 66], [7, 9, 55, 66]],
            dtype=torch.long,
        ),
        "slot_mapping": torch.tensor([3, 7, -1, -1], dtype=torch.int32),
        "context_lens": torch.tensor([8, 10, 0, 0], dtype=torch.int32),
        "block_tables": torch.tensor(
            [
                [0, 1, -1, -1],
                [2, 3, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    }
    runner = SimpleNamespace(enforce_eager=False, graph_vars=graph_vars)
    device_batch = DeviceBatch(
        phase=BatchPhase.DECODE,
        sequence_ids=(10, 20),
        scheduled_token_counts=(1, 1),
        model_inputs=DeviceModelInputs(
            input_ids=torch.tensor([101, 202], dtype=torch.long),
            position_ids=torch.tensor([7, 9], dtype=torch.long),
        ),
        attention_context=Context(
            is_prefill=False,
            slot_mapping=torch.tensor([3, 7], dtype=torch.int32),
            context_lens=torch.tensor([8, 10], dtype=torch.int32),
            decode_max_context_len=torch.tensor(10, dtype=torch.int32),
            block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        ),
        temperatures=torch.zeros(2),
        execution_bucket=4,
    )
    sampler = DynamicBatchTrajectorySampler({10: 0, 20: 1}, runner=runner)

    sampler.begin_batch(device_batch)
    selected = sampler(
        torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]),
        torch.zeros(2),
    )
    sampler.end_batch()
    sampler.finish({10: [1], 20: [0]})
    audit = sampler.batch_trace[0]["graph_static_buffer_audit"]

    assert selected.tolist() == [1, 0]
    assert audit["active_position_ids_exact"] is True
    assert audit["active_block_table_tail_all_minus_one"] is True
    assert audit["padding_slot_mapping_all_minus_one"] is True
    assert audit["padding_context_lens_all_zero"] is True
    assert audit["padding_block_tables_all_minus_one"] is True
    assert audit["padding_input_ids"] == [99, 88]
