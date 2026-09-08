"""CPU unit tests for the per-image host visual embedding cache (Event 6)."""

from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.contracts import BatchPhase, BatchPlan, PrefillSlice
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.model_runner import (
    ModelRunner,
    _assemble_per_image_visual_outputs,
    _image_row_ranges,
    _PerImageVisualEmbeddingHostCache,
)
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


def _entry_embeds(rows: int, value: float = 1.0, dim: int = 8) -> torch.Tensor:
    return torch.full((rows, dim), value, dtype=torch.float32)


class TestImageRowRanges:
    def test_two_images(self):
        grid = torch.tensor([[2, 2, 1], [3, 2, 1]], dtype=torch.int64)
        assert _image_row_ranges(grid) == [(0, 4), (4, 10)]

    def test_single_image(self):
        grid = torch.tensor([[2, 3, 1]], dtype=torch.int64)
        assert _image_row_ranges(grid) == [(0, 6)]

    def test_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            _image_row_ranges(
                torch.tensor([[2, 2, 1], [3, 2, 1], [4, 2, 1]], dtype=torch.int64).unsqueeze(0)
            )
        with pytest.raises(ValueError):
            _image_row_ranges(torch.tensor([[2, 2], [3, 2]], dtype=torch.int64))


class TestAssemblePerImageOutputs:
    def test_preserves_payload_order(self):
        parts = [
            (_entry_embeds(2, value=1.0), (_entry_embeds(2, value=10.0),)),
            (_entry_embeds(3, value=2.0), (_entry_embeds(3, value=20.0),)),
            (_entry_embeds(1, value=3.0), (_entry_embeds(1, value=30.0),)),
        ]
        vis, deep = _assemble_per_image_visual_outputs(parts)
        assert vis.shape == (6, 8)
        assert deep[0].shape == (6, 8)
        assert torch.allclose(vis[:, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 3.0]))
        assert torch.allclose(deep[0][:, 0], torch.tensor([10.0, 10.0, 20.0, 20.0, 20.0, 30.0]))

    def test_empty_parts(self):
        vis, deep = _assemble_per_image_visual_outputs([])
        assert vis.shape == (0, 0)
        assert deep == ()


class TestHostCacheLru:
    def test_lookup_hit_moves_to_end(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=1 << 20)
        for key in (b"a", b"b", b"c"):
            cache.store(key, _entry_embeds(1), ())
        assert list(cache._entries) == [b"a", b"b", b"c"]
        entry = cache.lookup(b"a")
        assert entry is not None
        assert list(cache._entries) == [b"b", b"c", b"a"]
        assert cache.hits == 1

    def test_lookup_miss_returns_none(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=1 << 20)
        assert cache.lookup(b"missing") is None
        assert cache.hits == 0

    def test_byte_budget_evicts_lru_first(self):
        # each entry: dim=4 float32 = 16 bytes
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=32)
        cache.store(b"a", _entry_embeds(1, dim=4), ())
        cache.store(b"b", _entry_embeds(1, dim=4), ())
        assert list(cache._entries) == [b"a", b"b"]
        cache.store(b"c", _entry_embeds(1, dim=4), ())
        assert list(cache._entries) == [b"b", b"c"]
        assert cache.evictions == 1
        assert cache.resident_bytes <= 32
        assert cache.lookup(b"a") is None

    def test_oversize_skip(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=16)
        cache.store(b"big", _entry_embeds(2, dim=8), ())  # 64 bytes > budget
        assert len(cache._entries) == 0
        assert cache.oversize_skips == 1
        assert cache.misses == 1

    def test_deepstack_counts_toward_budget(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=96)
        # dim=4: 16 bytes main + 16 bytes deepstack = 32 per entry
        cache.store(b"a", _entry_embeds(1, dim=4), (_entry_embeds(1, dim=4),))
        cache.store(b"b", _entry_embeds(1, dim=4), (_entry_embeds(1, dim=4),))
        assert list(cache._entries) == [b"a", b"b"]
        assert cache.resident_bytes == 64
        cache.store(b"c", _entry_embeds(1, dim=4), (_entry_embeds(1, dim=4),))
        assert list(cache._entries) == [b"a", b"b", b"c"]
        assert cache.evictions == 0
        cache.store(b"d", _entry_embeds(1, dim=4), (_entry_embeds(1, dim=4),))
        assert list(cache._entries) == [b"b", b"c", b"d"]
        assert cache.evictions == 1

    def test_store_keeps_cpu_copy_of_cuda_tensor_shape(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=1 << 20)
        cache.store(b"a", _entry_embeds(3), (_entry_embeds(3),))
        entry = cache.lookup(b"a")
        assert entry.visual_embeds.shape == (3, 8)
        assert entry.deepstack_visual_embeds[0].shape == (3, 8)
        assert not entry.visual_embeds.is_cuda
        assert entry.storage_bytes == 192

    def test_clear_and_reset_metrics(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=1 << 20)
        cache.store(b"a", _entry_embeds(1), ())
        cache.lookup(b"a")
        assert cache.hits == 1 and cache.misses == 1
        cache.reset_metrics()
        assert cache.hits == 0 and cache.misses == 0
        cache.clear()
        assert len(cache._entries) == 0 and cache.resident_bytes == 0

    def test_rejects_bad_budget(self):
        with pytest.raises(ValueError):
            _PerImageVisualEmbeddingHostCache(max_bytes=0)
        with pytest.raises(ValueError):
            _PerImageVisualEmbeddingHostCache(max_bytes=True)

    def test_metadata_shape(self):
        cache = _PerImageVisualEmbeddingHostCache(max_bytes=64)
        meta = cache.metadata()
        assert meta["scope"] == "per_image_pinned_lru"
        assert meta["max_bytes"] == 64
        assert set(meta) >= {
            "resident_bytes",
            "entries",
            "hits",
            "misses",
            "evictions",
            "oversize_skips",
        }


def _cache_sequence():
    return Sequence(
        [151655, 151655, 7, 151655, 151655, 8],
        SamplingParams(max_tokens=1),
        block_size=4,
        request_id=0,
        pixel_values=torch.zeros(4, 3),
        image_grid_thw=torch.tensor([[1, 2, 1], [1, 2, 1]]),
        image_merge_size=1,
        image_token_id=151655,
        image_token_count=4,
    )


def test_cache_enabled_submission_defers_vision_until_scheduled():
    seq = _cache_sequence()
    seq.visual_embedding_cache_key = "images"
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(enable_visual_embedding_cache=True)
    engine.scheduler = Scheduler(
        SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=32,
            max_model_len=64,
            enable_chunked_prefill=False,
            max_chunk_size=8,
            eos=-1,
            max_queue_size=None,
            max_consecutive_prefill_batches=1,
        ),
        kv_manager=BlockManager(8, 4),
    )
    engine.model_runner = SimpleNamespace(
        hydrate_visual_embedding_cache=lambda _: pytest.fail("submission must not run Vision")
    )
    engine._submit_sequence(seq)
    assert list(engine.scheduler.waiting) == [seq]
    hydrated = []
    runner = ModelRunner.__new__(ModelRunner)
    runner.hydrate_visual_embedding_cache = hydrated.append
    runner.prepare_prefill_visual_cache(engine.scheduler.schedule())
    assert hydrated == [seq]


def test_partial_or_full_visual_prefix_hit_does_not_hydrate_whole_group():
    seq = _cache_sequence()
    runner = ModelRunner.__new__(ModelRunner)
    runner.hydrate_visual_embedding_cache = lambda _: pytest.fail("covered images were restored")
    for start in (3, 5):
        seq.num_cached_tokens = start
        part = PrefillSlice(seq.seq_id, start, 6)
        plan = BatchPlan(BatchPhase.PREFILL, (seq,), (6 - start,), prefill_slices=(part,))
        runner.prepare_prefill_visual_cache(plan)


def test_all_image_cache_hits_do_not_stage_raw_pixel_tensor():
    seq = _cache_sequence()
    runner = ModelRunner.__new__(ModelRunner)
    runner._visual_embedding_host_cache = _PerImageVisualEmbeddingHostCache(1 << 20)
    for key in (b"first", b"second"):
        runner._visual_embedding_host_cache.store(key, _entry_embeds(2), (_entry_embeds(2),))

    def stage(value):
        assert value is not seq.pixel_values
        return value

    runner._visual_cache_device_tensor = stage
    assert runner._hydrate_visual_embedding_cache_per_image(seq, (b"first", b"second"))
    assert seq.precomputed_visual_embeds.shape == (4, 8)
