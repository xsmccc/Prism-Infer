"""P3.5 ModelRunner CUDA Graph decode shape 验证。"""

from types import SimpleNamespace

import torch

from prism_infer.engine.model_runner import ModelRunner


def test_cudagraph_decode_positions_normalize_text_and_vl_shapes() -> None:
    """graph replay 前 text/VL decode positions 都应规范为 `[3,batch]`。"""

    text_positions = torch.tensor([5, 7], dtype=torch.long)
    normalized_text = ModelRunner._as_mrope_decode_positions(text_positions)
    vl_positions = torch.tensor(
        [[5, 28, 56], [5, 28, 56], [5, 28, 56]],
        dtype=torch.long,
    )
    normalized_vl = ModelRunner._as_mrope_decode_positions(vl_positions)

    print(f"text decode input positions shape: {list(text_positions.shape)}")
    print(f"text graph positions shape: {list(normalized_text.shape)}")
    print(f"text graph positions: {normalized_text.tolist()}")
    print(f"vl decode input positions shape: {list(vl_positions.shape)}")
    print(f"vl graph positions shape: {list(normalized_vl.shape)}")
    print(f"vl graph positions: {normalized_vl.tolist()}")

    assert list(normalized_text.shape) == [3, 2]
    assert normalized_text.tolist() == [[5, 7], [5, 7], [5, 7]]
    assert list(normalized_vl.shape) == [3, 3]
    assert torch.equal(normalized_vl, vl_positions)
    print("ModelRunner CUDA Graph decode position shape normalization: PASS")


def test_cudagraph_batch_sizes_cover_non_standard_max_bs() -> None:
    """非 1/2/4/8/16 档位的 max_num_seqs 也必须有可 replay graph。"""

    cases = {
        1: [1],
        3: [1, 2, 3],
        5: [1, 2, 4, 5],
        17: [1, 2, 4, 8, 16, 17],
    }
    for max_bs, expected in cases.items():
        got = ModelRunner._cudagraph_batch_sizes(max_bs)
        print(f"max_bs={max_bs}, graph_bs={got}")
        assert got == expected
    print("ModelRunner CUDA Graph batch size coverage: PASS")


def test_cudagraph_metadata_reports_capture_scope_and_selected_bucket() -> None:
    """Benchmark metadata 必须区分 actual batch、graph bucket 和 capture scope。"""

    runner = object.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.graph_bs = [1, 2, 4, 8]
    runner.cudagraph_capture_ms = 123.5

    metadata = runner.cudagraph_metadata(3)

    print(f"CUDA Graph execution metadata: {metadata}")
    assert metadata == {
        "enabled": True,
        "capture_scope": "decode_model_forward",
        "capture_ms": 123.5,
        "batch_sizes": [1, 2, 4, 8],
        "requested_batch_size": 3,
        "selected_batch_size": 4,
        "batch_padding": 1,
    }
    print("ModelRunner CUDA Graph execution metadata: PASS")


def test_eager_metadata_reports_no_graph_state() -> None:
    runner = object.__new__(ModelRunner)
    runner.enforce_eager = True

    metadata = runner.cudagraph_metadata(3)

    print(f"eager execution metadata: {metadata}")
    assert metadata == {
        "enabled": False,
        "capture_scope": "none",
        "capture_ms": 0.0,
        "batch_sizes": [],
        "requested_batch_size": 3,
        "selected_batch_size": 3,
        "batch_padding": 0,
    }
    print("ModelRunner eager execution metadata: PASS")


def test_compile_metadata_reports_attention_region_and_cold_time() -> None:
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        decode_compile_region="attention",
        decode_compile_mode="default",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
    )
    runner.decode_compile_first_call_ms = 2345.0

    metadata = runner.compile_metadata()

    assert metadata == {
        "enabled": True,
        "region": "decode_attention",
        "subgraph": "qkv_projection_qk_norm_mrope",
        "kv_cache_boundary": "validated_runtime_store_and_paged_decode",
        "backend": "inductor",
        "mode": "default",
        "emulate_precision_casts": True,
        "force_same_precision": True,
        "first_call_ms": 2345.0,
    }
    print(f"attention compile metadata: {metadata} PASS")


def test_compile_metadata_reports_disabled_state() -> None:
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        decode_compile_region="none",
        decode_compile_mode="default",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
    )
    runner.decode_compile_first_call_ms = 0.0

    metadata = runner.compile_metadata()

    assert metadata == {
        "enabled": False,
        "region": "none",
        "subgraph": "none",
        "kv_cache_boundary": "none",
        "backend": "none",
        "mode": "none",
        "emulate_precision_casts": False,
        "force_same_precision": False,
        "first_call_ms": 0.0,
    }
    print(f"disabled compile metadata: {metadata} PASS")
