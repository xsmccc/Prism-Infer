"""TP capacity selection uses one common limit before any rank can reject."""

from types import SimpleNamespace

import pytest

from prism_infer.engine import model_runner

BLOCK_BYTES = 256


class _CapacityScalar:
    def __init__(self, value, device):
        self.value = value
        self.device = device

    def item(self):
        return self.value


def _runner(rank, requested, *, world_size=2):
    runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    runner.rank = rank
    runner.world_size = world_size
    runner.config = SimpleNamespace(
        gpu_memory_utilization=1.0,
        num_kvcache_blocks=requested,
    )
    runner.kv_cache_dtype = "torch.bfloat16"
    runner.kv_payload_block_bytes = BLOCK_BYTES
    runner.kv_scale_block_bytes = 0
    return runner


def _mock_capacities(monkeypatch, capacities):
    rank = {"value": 0}
    calls = []
    monkeypatch.setattr(
        model_runner.torch.cuda,
        "mem_get_info",
        lambda: (capacities[rank["value"]] * BLOCK_BYTES, 64 * BLOCK_BYTES),
    )
    monkeypatch.setattr(
        model_runner.torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.peak": 128, "allocated_bytes.all.current": 128},
    )

    def tensor(value, *, dtype, device):
        assert dtype is model_runner.torch.int64
        return _CapacityScalar(value, device)

    def all_reduce(value, *, op):
        assert op == model_runner.dist.ReduceOp.MIN
        calls.append((value.device, value.value))
        value.value = min(capacities)

    monkeypatch.setattr(model_runner.torch, "tensor", tensor)
    monkeypatch.setattr(model_runner.dist, "all_reduce", all_reduce)
    return rank, calls


@pytest.mark.parametrize(("requested", "expected"), [(-1, 6), (4, 4)])
def test_tp_uses_common_capacity_for_auto_and_explicit_requests(monkeypatch, requested, expected):
    rank, calls = _mock_capacities(monkeypatch, (10, 6))
    selected = []
    for index in range(2):
        rank["value"] = index
        selected.append(_runner(index, requested)._select_num_kv_blocks(BLOCK_BYTES))
    assert selected == [expected, expected]
    assert calls == [("cuda:0", 10), ("cuda:1", 6)]


def test_explicit_capacity_exceeding_only_one_rank_is_rejected_by_both(monkeypatch):
    rank, calls = _mock_capacities(monkeypatch, (10, 4))
    for index in range(2):
        rank["value"] = index
        with pytest.raises(RuntimeError, match="requested=5, max=4"):
            _runner(index, 5)._select_num_kv_blocks(BLOCK_BYTES)
    assert calls == [("cuda:0", 10), ("cuda:1", 4)]


def test_rank_with_no_space_still_joins_collective_before_shared_failure(monkeypatch):
    rank, calls = _mock_capacities(monkeypatch, (10, 0))
    for index in range(2):
        rank["value"] = index
        with pytest.raises(RuntimeError, match="no KV cache blocks fit"):
            _runner(index, -1)._select_num_kv_blocks(BLOCK_BYTES)
    assert calls == [("cuda:0", 10), ("cuda:1", 0)]


def test_tp1_does_not_construct_a_capacity_tensor_or_call_collectives(monkeypatch):
    _, calls = _mock_capacities(monkeypatch, (8,))

    def unexpected_tensor(*args, **kwargs):
        raise AssertionError("TP1 must not create a collective capacity tensor")

    monkeypatch.setattr(model_runner.torch, "tensor", unexpected_tensor)
    assert _runner(0, -1, world_size=1)._select_num_kv_blocks(BLOCK_BYTES) == 8
    with pytest.raises(RuntimeError, match="requested=9, max=8"):
        _runner(0, 9, world_size=1)._select_num_kv_blocks(BLOCK_BYTES)
    assert calls == []
