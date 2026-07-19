"""P6.4 logical/physical KV layout contract tests。"""

import pickle
from contextlib import contextmanager
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.kv_layout import (
    KVCacheLayoutDescriptor,
    KV_LAYOUT_VISUAL_COMPACT,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.utils.context import get_context, reset_context


@contextmanager
def _page_contract(block_size: int) -> Iterator[int]:
    """Assert a scenario cannot leak page size through Sequence class state."""

    assert block_size > 0
    assert not hasattr(Sequence, "block_size")
    assert not hasattr(Sequence, "set_block_size")
    try:
        yield block_size
    finally:
        reset_context()
        assert not hasattr(Sequence, "block_size")
        assert not hasattr(Sequence, "set_block_size")


def _compact_record() -> dict[str, object]:
    return {
        "prompt_token_count": 6,
        "total_visual_tokens": 4,
        "kept_visual_tokens": 2,
        "dropped_visual_tokens": 2,
        "kept_token_indices": [1, 4],
        "dropped_token_indices": [2, 3],
        "physical_compaction": True,
    }


def _layout() -> KVCacheLayoutDescriptor:
    return KVCacheLayoutDescriptor(
        mode=KV_LAYOUT_VISUAL_COMPACT,
        logical_context_len=6,
        physical_kv_len=4,
        prompt_logical_len=6,
        compressed_prompt_kv_len=4,
        retained_original_positions=(0, 1, 4, 5),
        kv_dtype="torch.bfloat16",
        compression_record=_compact_record(),
    )


def test_kv_layout_separates_logical_and_physical_lengths() -> None:
    layout = _layout()
    layout.validate(block_size=4, block_table=[7])

    print(f"logical context length: {layout.logical_context_len}")
    print(f"physical KV length: {layout.physical_kv_len}")
    print(f"retained original positions: {layout.retained_original_positions}")
    assert layout.logical_context_len == 6
    assert layout.physical_kv_len == 4
    print("P6.4 KV layout logical/physical split: PASS")


def test_compact_sequence_append_and_decode_pickle_preserve_layout() -> None:
    with _page_contract(4) as block_size:
        seq = Sequence(
            [10, 11, 12, 13, 14, 15],
            block_size=block_size,
            request_id=0,
        )
        seq.block_table = [7]
        seq.visual_pruning_decision_record = _compact_record()
        seq.install_kv_layout(_layout())
        seq.append_token(99)
        # Scheduler allocates the new physical page before sending decode state.
        seq.block_table.append(8)

        restored = pickle.loads(pickle.dumps(seq))

        print(f"restored logical length: {restored.num_tokens}")
        print(f"restored physical length: {restored.physical_kv_len}")
        print(f"restored block table: {restored.block_table}")
        assert restored.num_tokens == 7
        assert restored.physical_kv_len == 5
        assert restored.physical_num_blocks == 2
        assert restored.physical_last_block_num_tokens == 1
        assert restored.block_table == [7, 8]
        assert restored.kv_layout is not None
        print("P6.4 compact Sequence decode pickle: PASS")


def test_compact_sequence_swapped_pickle_uses_cpu_table() -> None:
    with _page_contract(4) as block_size:
        seq = Sequence(
            [10, 11, 12, 13, 14, 15],
            block_size=block_size,
            request_id=0,
        )
        seq.block_table = [7]
        seq.install_kv_layout(_layout())
        seq.cpu_block_table = [3]
        seq.block_table = []

        restored = pickle.loads(pickle.dumps(seq))

        assert restored.block_table == []
        assert restored.cpu_block_table == [3]
        assert restored.physical_kv_len == 4
        print("P6.4 compact Sequence swapped pickle: PASS")


def test_kv_layout_rejects_inconsistent_retained_positions() -> None:
    layout = _layout()
    layout.retained_original_positions = (0, 4, 1, 5)

    with pytest.raises(ValueError, match="sorted and unique"):
        layout.validate(block_size=4, block_table=[7])
    print("P6.4 KV layout retained-position guard: PASS")


def _manager_sequence(block_size: int = 4) -> Sequence:
    seq = Sequence(
        [10, 11, 99, 99, 99, 99, 99, 99, 12, 13],
        block_size=block_size,
        request_id=0,
        image_token_id=99,
        image_token_count=6,
    )
    seq.visual_pruning_decision_record = {
        "seq_id": seq.seq_id,
        "batch_index": 0,
        "prompt_token_count": 10,
        "total_visual_tokens": 6,
        "kept_visual_tokens": 2,
        "dropped_visual_tokens": 4,
        "keep_ratio_target": 1 / 3,
        "keep_ratio_actual": 1 / 3,
        "strategy": "uniform",
        "physical_compaction": False,
        "visual_token_spans": [
            {
                "modality": "image",
                "start": 2,
                "end": 8,
                "index": 0,
                "token_count": 6,
            }
        ],
        "kept_token_indices": [2, 7],
        "dropped_token_indices": [3, 4, 5, 6],
    }
    return seq


def test_block_manager_and_runner_commit_physical_compaction() -> None:
    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        old_table = list(seq.block_table)
        plan = manager.build_compaction_plan(seq, kv_dtype="torch.float32")
        assert plan is not None

        runner = object.__new__(ModelRunner)
        runner.block_size = block_size
        runner.world_size = 1
        # kv_cache: [2, layers, blocks, block_size, kv_heads, head_dim]
        runner.kv_cache = torch.arange(
            2 * 2 * 8 * 4 * 1 * 2,
            dtype=torch.float32,
        ).view(2, 2, 8, 4, 1, 2)
        flat_before = runner.kv_cache.reshape(2, 2, -1, 1, 2).clone()
        source = torch.tensor(plan.source_slots, dtype=torch.long)
        expected = flat_before.index_select(2, source)

        runner.compact_kv_cache([plan])
        manager.commit_compaction(seq, plan)

        flat_after = runner.kv_cache.reshape(2, 2, -1, 1, 2)
        destination = torch.tensor(plan.destination_slots, dtype=torch.long)
        actual = flat_after.index_select(2, destination)
        diff = (actual - expected).abs()
        print(f"old block table: {old_table}")
        print(f"new block table: {seq.block_table}")
        print(f"released blocks: {list(plan.released_block_ids)}")
        print(f"compact K/V shape: {list(actual.shape)}")
        print(f"compact reference max diff: {diff.max().item():.6e}")

        assert old_table == [0, 1, 2]
        assert seq.block_table == [0, 1]
        assert list(plan.released_block_ids) == [2]
        assert 2 in manager.free_block_id_set
        assert 2 not in manager.used_block_ids
        assert seq.num_tokens == 10
        assert seq.physical_kv_len == 6
        assert seq.visual_pruning_decision_record["physical_compaction"] is True
        assert diff.max().item() == 0.0
        print("P6.4 post-prefill KV compact/block release: PASS")


def test_compaction_commit_rejects_stale_decision_without_mutating_pages() -> None:
    """A plan cannot commit after its pruning decision has changed."""

    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        plan = manager.build_compaction_plan(seq, kv_dtype="torch.float32")
        assert plan is not None
        original_table = list(seq.block_table)
        original_used = set(manager.used_block_ids)
        seq.visual_pruning_decision_record = {
            **seq.visual_pruning_decision_record,
            "kept_token_indices": [2],
        }

        with pytest.raises(RuntimeError, match="decision changed"):
            manager.commit_compaction(seq, plan)

        assert seq.block_table == original_table
        assert seq.kv_layout is None
        assert manager.used_block_ids == original_used
        assert not (set(original_table) & manager.free_block_id_set)


def test_compact_decode_append_uses_physical_tail_and_clears_hashes() -> None:
    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        plan = manager.build_compaction_plan(seq, kv_dtype="torch.float32")
        assert plan is not None
        manager.commit_compaction(seq, plan)
        assert all(manager.blocks[block_id].hash == -1 for block_id in seq.block_table)

        seq.append_token(20)  # physical 7
        manager.may_append(seq)
        seq.append_token(21)  # physical 8, compact full page remains unhashed
        manager.may_append(seq)
        assert all(manager.blocks[block_id].hash == -1 for block_id in seq.block_table)
        seq.append_token(22)  # physical 9, allocate a new page
        assert manager.can_append(seq)
        manager.may_append(seq)

        print(f"append logical length: {len(seq)}")
        print(f"append physical length: {seq.physical_kv_len}")
        print(f"append block table: {seq.block_table}")
        assert len(seq) == 13
        assert seq.physical_kv_len == 9
        assert len(seq.block_table) == 3
        assert seq.physical_last_block_num_tokens == 1
        print("P6.4 compact decode physical-tail append: PASS")


def test_compaction_rejects_prefix_shared_blocks() -> None:
    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        manager.blocks[seq.block_table[0]].ref_count = 2

        with pytest.raises(RuntimeError, match="prefix-shared blocks"):
            manager.build_compaction_plan(seq, kv_dtype="torch.float32")
        print("P6.4 shared-block compaction guard: PASS")


def test_compact_swap_pickle_swap_in_preserves_layout_and_hash_state() -> None:
    """Compact KV 经 swap/pickle 后必须保留 physical layout 与禁用 hash。"""

    with _page_contract(4) as block_size:
        manager = BlockManager(
            num_blocks=8,
            block_size=block_size,
            num_cpu_blocks=4,
        )
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        plan = manager.build_compaction_plan(seq, kv_dtype="torch.bfloat16")
        assert plan is not None
        manager.commit_compaction(seq, plan)
        compact_gpu_table = list(seq.block_table)

        swap_out_map = manager.swap_out(seq)
        cpu_table = list(seq.cpu_block_table)
        restored = pickle.loads(pickle.dumps(seq))
        swap_in_map = manager.swap_in(restored)

        print(f"compact GPU table before swap: {compact_gpu_table}")
        print(f"compact swap-out map: {swap_out_map}")
        print(f"compact CPU table after pickle: {cpu_table}")
        print(f"compact swap-in map: {swap_in_map}")
        print(f"compact GPU table after swap-in: {restored.block_table}")
        print(f"logical/physical lengths: {len(restored)}/{restored.physical_kv_len}")

        assert restored.kv_layout is not None
        assert restored.kv_layout.mode == KV_LAYOUT_VISUAL_COMPACT
        assert len(restored) == 10
        assert restored.physical_kv_len == 6
        assert restored.physical_num_blocks == 2
        assert restored.cpu_block_table == []
        assert len(restored.block_table) == len(compact_gpu_table)
        assert [gpu_id for gpu_id, _ in swap_out_map] == compact_gpu_table
        assert [cpu_id for cpu_id, _ in swap_in_map] == cpu_table
        assert all(
            manager.blocks[block_id].hash == -1 and manager.blocks[block_id].token_ids == []
            for block_id in restored.block_table
        )
        assert all(
            block_id not in manager.hash_to_block_id.values() for block_id in restored.block_table
        )
        print("P6.4 compact swap/pickle/swap-in lifecycle: PASS")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_decode_uses_logical_mrope_and_physical_kv_tail() -> None:
    """Compact decode 必须分离 logical position 与 physical attention/write。"""

    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        seq.rope_delta = torch.tensor([[3]], dtype=torch.long)
        manager.allocate(seq)
        plan = manager.build_compaction_plan(seq, kv_dtype="torch.bfloat16")
        assert plan is not None
        manager.commit_compaction(seq, plan)
        seq.append_token(77)
        manager.may_append(seq)

        runner = object.__new__(ModelRunner)
        runner.block_size = block_size
        runner.config = SimpleNamespace(
            compression_mode="visual_compact",
            enable_visual_pruning_shadow=False,
            kvcache_block_size=block_size,
        )
        model_inputs = runner.prepare_decode([seq])
        context = get_context()

        expected_logical_position = len(seq) - 1 + int(seq.rope_delta.item())
        expected_physical_slot = (
            seq.block_table[-1] * seq.block_size + seq.physical_last_block_num_tokens - 1
        )
        print(f"decode logical length: {len(seq)}")
        print(f"decode physical KV length: {seq.physical_kv_len}")
        print(f"decode M-RoPE position: {model_inputs.position_ids[:, 0].tolist()}")
        print(f"decode physical slot: {context.slot_mapping.tolist()}")

        assert model_inputs.input_ids.tolist() == [77]
        assert model_inputs.position_ids[:, 0].tolist() == [expected_logical_position] * 3
        assert context.logical_context_lens.tolist() == [len(seq)]
        assert context.context_lens.tolist() == [seq.physical_kv_len]
        assert context.slot_mapping.tolist() == [expected_physical_slot]
        print("P6.4 compact decode logical/physical metadata: PASS")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_kv_compaction_matches_independent_retained_reference() -> None:
    """FP8 compact 必须绕过不支持的 CUDA index_select 并保持量化值 exact。"""

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required")
    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = _manager_sequence(block_size)
        manager.allocate(seq)
        plan = manager.build_compaction_plan(
            seq,
            kv_dtype="torch.float8_e4m3fn",
        )
        assert plan is not None

        runner = object.__new__(ModelRunner)
        runner.block_size = block_size
        runner.world_size = 1
        torch.manual_seed(20260711)
        source_cache = torch.randn(
            2,
            2,
            8,
            4,
            1,
            8,
            device="cuda",
            dtype=torch.bfloat16,
        )
        runner.kv_cache = source_cache.to(torch.float8_e4m3fn)
        flat_before = runner.kv_cache.to(torch.bfloat16).reshape(2, 2, -1, 1, 8)
        source_slots = torch.tensor(plan.source_slots, device="cuda", dtype=torch.long)
        expected = flat_before.index_select(2, source_slots)

        runner.compact_kv_cache([plan])
        destination_slots = torch.tensor(
            plan.destination_slots,
            device="cuda",
            dtype=torch.long,
        )
        flat_after = runner.kv_cache.to(torch.bfloat16).reshape(2, 2, -1, 1, 8)
        actual = flat_after.index_select(2, destination_slots)
        diff = (actual - expected).abs()
        torch.cuda.synchronize()

        print(f"FP8 compact cache shape: {list(runner.kv_cache.shape)}")
        print(f"FP8 compact retained shape: {list(actual.shape)}")
        print(
            "FP8 compact output/reference mean/std: "
            f"{actual.float().mean().item():.6e}/{actual.float().std().item():.6e} vs "
            f"{expected.float().mean().item():.6e}/{expected.float().std().item():.6e}"
        )
        print(f"FP8 compact max diff: {diff.max().item():.6e}")
        assert actual.shape == expected.shape == (2, 2, 6, 1, 8)
        assert diff.max().item() == 0.0
        print("P6.6 FP8 physical compaction: PASS")
