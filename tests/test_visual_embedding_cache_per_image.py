"""CPU unit tests for the per-image host visual embedding cache (Event 6)."""

import pytest
import torch

from prism_infer.engine.model_runner import (
    _PerImageVisualEmbeddingHostCache,
    _assemble_per_image_visual_outputs,
    _image_row_ranges,
)


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
            _image_row_ranges(torch.tensor([[2, 2, 1], [3, 2, 1], [4, 2, 1]], dtype=torch.int64).unsqueeze(0))
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
        assert torch.allclose(
            deep[0][:, 0], torch.tensor([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
        )

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
