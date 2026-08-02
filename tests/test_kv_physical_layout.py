"""P6.4 logical/physical KV layout contract tests。"""

import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.kv_layout import (
    KV_LAYOUT_VISUAL_COMPACT,
    KVCacheLayoutDescriptor,
)
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.sequence import Sequence
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


def _store_one_page_multimodal_prefix(
    manager: BlockManager,
    *,
    cache_key: str,
) -> None:
    cold = _manager_sequence(manager.block_size)
    cold.multimodal_prefix_cache_key = cache_key
    cold.visual_pruning_decision_record.update(
        {
            "kept_visual_tokens": 1,
            "dropped_visual_tokens": 5,
            "keep_ratio_actual": 1 / 6,
            "kept_token_indices": [2],
            "dropped_token_indices": [3, 4, 5, 6, 7],
        }
    )
    manager.allocate(cold)
    plan = manager.build_compaction_plan(
        cold,
        kv_dtype="torch.float8_e4m3fn",
    )
    assert plan is not None
    cold.num_computed_tokens = cold.num_prompt_tokens
    manager.commit_compaction(cold, plan)
    assert manager.store_multimodal_prefix(cold)
    manager.deallocate(cold)


def test_submit_probes_prefix_before_visual_hydration() -> None:
    """A pre-admission prefix hit bypasses Vision cache hydration."""

    manager = BlockManager(num_blocks=32, block_size=4)
    _store_one_page_multimodal_prefix(manager, cache_key="same-media-layout")
    hydration_calls: list[int] = []

    class _Scheduler:
        block_manager = manager

        def add(self, seq: Sequence, *, raise_on_reject: bool):
            del raise_on_reject
            self.block_manager.cached_prefix_tokens(seq)
            return SimpleNamespace(accepted=True)

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(enable_visual_embedding_cache=True)
    engine.scheduler = _Scheduler()
    engine.model_runner = SimpleNamespace(
        hydrate_visual_embedding_cache=lambda seq: hydration_calls.append(seq.seq_id)
    )
    engine.metrics = EngineMetrics()
    engine.clock_ns = lambda: 123

    hit = _manager_sequence()
    hit.multimodal_prefix_cache_key = "same-media-layout"
    hit.visual_embedding_cache_key = "same-visual-output"
    original_pixels = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    hit.pixel_values = original_pixels

    assert engine._submit_sequence(hit) == hit.seq_id
    assert hydration_calls == []
    assert hit.pixel_values is original_pixels
    assert hit.prefix_cache_candidate_tokens == 8
    metadata = manager.multimodal_prefix_cache_metadata()
    assert metadata["pre_admission_hits"] == 1
    assert metadata["visual_hydration_skips"] == 1

    manager.clear_multimodal_prefix_cache()


def test_stale_pre_admission_probe_falls_back_to_dense_prefill() -> None:
    """An evicted early hit keeps raw media and becomes a normal cold prefill."""

    manager = BlockManager(num_blocks=32, block_size=4)
    _store_one_page_multimodal_prefix(manager, cache_key="evicted-media-layout")
    hit = _manager_sequence()
    hit.multimodal_prefix_cache_key = "evicted-media-layout"
    original_pixels = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    hit.pixel_values = original_pixels

    assert manager.probe_multimodal_prefix(hit, would_hydrate_visual=True) == 8
    manager.clear_multimodal_prefix_cache()
    assert manager.cached_prefix_tokens(hit) == 0
    assert not hit.multimodal_prefix_stale_fallback
    assert manager.multimodal_prefix_cache_metadata()["stale_probe_fallbacks"] == 0
    assert hit.pixel_values is original_pixels
    assert manager.can_allocate(hit)
    assert manager.allocate(hit) == ()
    assert hit.multimodal_prefix_stale_fallback
    assert not hit.multimodal_prefix_cache_hit
    assert len(hit.block_table) == hit.num_blocks == 3

    metadata = manager.multimodal_prefix_cache_metadata()
    assert metadata["stale_probe_fallbacks"] == 1
    manager.deallocate(hit)


def test_multimodal_prefix_identity_uses_media_boundary_and_prompt_digest() -> None:
    """Only the exact media namespace and public prompt prefix can hit."""

    manager = BlockManager(num_blocks=32, block_size=4)
    _store_one_page_multimodal_prefix(manager, cache_key="media-layout-a")

    same_prefix = _manager_sequence()
    same_prefix.token_ids[-2:] = [42, 43]
    same_prefix.multimodal_prefix_cache_key = "media-layout-a"
    assert manager.cached_prefix_tokens(same_prefix) == 8

    different_prompt_prefix = _manager_sequence()
    different_prompt_prefix.token_ids[0] = 77
    different_prompt_prefix.multimodal_prefix_cache_key = "media-layout-a"
    assert manager.cached_prefix_tokens(different_prompt_prefix) == 0

    different_processor_layout = _manager_sequence()
    different_processor_layout.multimodal_prefix_cache_key = "media-layout-b"
    assert manager.cached_prefix_tokens(different_processor_layout) == 0

    metadata = manager.multimodal_prefix_cache_metadata()
    assert metadata["lookup_strategy"] == "direct_boundary_prompt_sha256"
    manager.clear_multimodal_prefix_cache()


def test_multimodal_prefix_cache_uses_full_pool_and_yields_to_active_request() -> None:
    """A 32-page pool can retain more than four prefixes and reclaim on demand."""

    manager = BlockManager(num_blocks=32, block_size=4)
    for index in range(5):
        _store_one_page_multimodal_prefix(
            manager,
            cache_key=f"media-layout-{index}",
        )

    metadata = manager.multimodal_prefix_cache_metadata()
    assert metadata["total_pool_blocks"] == 32
    assert metadata["max_blocks"] == 32
    assert metadata["entries"] == 5
    assert metadata["resident_blocks"] == 5

    active = Sequence(
        list(range(112)),
        block_size=4,
        request_id=100,
    )
    assert active.num_blocks == 28
    assert manager.can_allocate(active)
    manager.allocate(active)
    assert len(active.block_table) == 28
    assert manager.multimodal_prefix_cache_metadata()["evictions"] >= 1

    manager.deallocate(active)
    manager.clear_multimodal_prefix_cache()
    assert not manager.used_block_ids


def test_decode_append_reclaims_idle_prefix_page() -> None:
    """Decode can grow after the idle Prefix Cache consumes the last page."""

    manager = BlockManager(num_blocks=5, block_size=4)
    for index in range(3):
        _store_one_page_multimodal_prefix(manager, cache_key=f"cached-{index}")

    active = Sequence(list(range(8)), block_size=4, request_id=200)
    manager.allocate(active)
    assert not manager.free_block_id_set

    active.append_token(8)
    assert manager.can_append(active)
    manager.may_append(active)

    assert len(active.block_table) == 3
    assert manager.multimodal_prefix_cache_metadata()["evictions"] >= 1
    manager.deallocate(active)
    manager.clear_multimodal_prefix_cache()


def test_decode_cow_reclaims_idle_prefix_page() -> None:
    """A shared decode tail can CoW by reclaiming an idle cached prefix."""

    manager = BlockManager(num_blocks=3, block_size=4)
    _store_one_page_multimodal_prefix(manager, cache_key="reclaim-for-cow")
    first = Sequence([1, 2, 3, 4], block_size=4, request_id=201)
    second = Sequence([1, 2, 3, 4], block_size=4, request_id=202)
    filler = Sequence([5, 6, 7, 8], block_size=4, request_id=203)
    manager.allocate(first)
    manager.allocate(second)
    manager.allocate(filler)
    assert first.block_table == second.block_table
    assert not manager.free_block_id_set

    pair = manager.copy_on_write(first)

    assert pair is not None
    assert first.block_table != second.block_table
    assert manager.multimodal_prefix_cache_metadata()["evictions"] == 1
    manager.deallocate(first)
    manager.deallocate(second)
    manager.deallocate(filler)


def test_swap_in_reclaims_idle_prefix_pages() -> None:
    """Atomic swap-in can reclaim cached pages from the shared GPU pool."""

    manager = BlockManager(
        num_blocks=8,
        block_size=4,
        num_cpu_blocks=4,
    )
    swapped = Sequence(list(range(8)), block_size=4, request_id=204)
    manager.allocate(swapped)
    manager.swap_out(swapped)
    for index in range(6):
        _store_one_page_multimodal_prefix(manager, cache_key=f"swap-cache-{index}")
    filler = Sequence(list(range(100, 108)), block_size=4, request_id=205)
    manager.allocate(filler)
    assert not manager.free_block_id_set

    assert manager.can_swap_in(swapped)
    swap_map = manager.swap_in(swapped)

    assert len(swap_map) == 2
    assert len(swapped.block_table) == 2
    assert manager.multimodal_prefix_cache_metadata()["evictions"] >= 2
    manager.deallocate(swapped)
    manager.deallocate(filler)
    manager.clear_multimodal_prefix_cache()


def test_terminal_sequence_releases_multimodal_runtime_tensors() -> None:
    """Completed requests do not pin evicted Vision/DeepStack cache tensors."""

    seq = _manager_sequence()
    seq.pixel_values = torch.ones(2, 4)
    seq.image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    seq.pixel_values_videos = torch.ones(2, 4)
    seq.video_grid_thw = torch.ones(1, 3, dtype=torch.long)
    seq.position_ids = torch.ones(3, 10, dtype=torch.long)
    seq.precomputed_visual_embeds = torch.ones(6, 8)
    seq.precomputed_deepstack_visual_embeds = (torch.ones(6, 8),)

    seq.release_multimodal_runtime_tensors()

    assert seq.pixel_values is None
    assert seq.image_grid_thw is None
    assert seq.pixel_values_videos is None
    assert seq.video_grid_thw is None
    assert seq.position_ids is None
    assert seq.precomputed_visual_embeds is None
    assert seq.precomputed_deepstack_visual_embeds == ()


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


def test_multimodal_prefix_cache_reuses_compacted_pages_with_tail_cow() -> None:
    """Same media/different question reuses only the safe compact prefix."""

    with _page_contract(4) as block_size:
        manager = BlockManager(num_blocks=32, block_size=block_size)
        cold = _manager_sequence(block_size)
        cold.multimodal_prefix_cache_key = "media-layout-sha256"
        cold.visual_pruning_decision_record.update(
            {
                "kept_visual_tokens": 1,
                "dropped_visual_tokens": 5,
                "keep_ratio_actual": 1 / 6,
                "kept_token_indices": [2],
                "dropped_token_indices": [3, 4, 5, 6, 7],
            }
        )
        manager.allocate(cold)
        plan = manager.build_compaction_plan(
            cold,
            kv_dtype="torch.float8_e4m3fn",
        )
        assert plan is not None
        cold.num_computed_tokens = cold.num_prompt_tokens
        manager.commit_compaction(cold, plan)
        assert manager.store_multimodal_prefix(cold)
        entry = manager.multimodal_prefix_entry_metadata(cold)
        assert entry is not None
        assert entry["logical_prefix_tokens"] == 8
        assert entry["physical_prefix_tokens"] == 3
        assert entry["canonical_blocks"] == 1
        assert entry["tail_clone_blocks"] == 0
        assert entry["resident_blocks"] == 1
        cached_block_id = cold.block_table[0]
        manager.deallocate(cold)

        hit = _manager_sequence(block_size)
        hit.token_ids[-2:] = [42, 43]
        hit.multimodal_prefix_cache_key = "media-layout-sha256"
        assert manager.cached_prefix_tokens(hit) == 8
        assert manager.can_allocate(hit)
        prefix_copies = manager.allocate(hit)

        assert prefix_copies == ((cached_block_id, hit.block_table[0], 3),)
        assert hit.block_table[0] != cached_block_id
        assert hit.multimodal_prefix_cache_hit
        assert hit.num_cached_tokens == 8
        assert hit.num_computed_tokens == 8
        assert hit.kv_layout is not None
        assert hit.kv_layout.logical_context_len == 8
        assert hit.kv_layout.physical_kv_len == 3
        assert hit.kv_layout.retained_original_positions == (0, 1, 2)
        assert manager.blocks[cached_block_id].ref_count == 1
        tail_clone_block_id = hit.block_table[0]
        assert manager.blocks[tail_clone_block_id].ref_count == 2

        hit.kv_layout.append_prefill_tokens(2)
        hit.num_computed_tokens = 10
        assert hit.kv_layout.logical_context_len == 10
        assert hit.kv_layout.physical_kv_len == 5
        manager.deallocate(hit)

        second_hit = _manager_sequence(block_size)
        second_hit.token_ids[-2:] = [44, 45]
        second_hit.multimodal_prefix_cache_key = "media-layout-sha256"
        assert manager.cached_prefix_tokens(second_hit) == 8
        assert manager.can_allocate(second_hit)
        assert manager.allocate(second_hit) == ()
        assert second_hit.block_table[0] == tail_clone_block_id
        manager.deallocate(second_hit)

        metadata = manager.multimodal_prefix_cache_metadata()
        assert metadata["hits"] == 2
        assert metadata["cow_copies"] == 1
        assert metadata["tail_clone_hits"] == 1
        assert metadata["tail_clone_admissions"] == 1
        assert metadata["tail_clone_reused_rows"] == 3
        assert metadata["copy_avoided_rows"] == 4
        assert metadata["resident_tail_clone_blocks"] == 1
        assert metadata["resident_blocks"] == 2
        manager.clear_multimodal_prefix_cache()
        assert not manager.used_block_ids


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
