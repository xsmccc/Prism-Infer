"""P3.5 ModelRunner CUDA Graph decode shape 验证。"""

from types import SimpleNamespace

import torch

from prism_infer.engine.contracts import BatchPhase
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


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
    """小 batch exact capture，非标准大上限也必须有可 replay graph。"""

    cases = {
        1: [1],
        3: [1, 2, 3],
        5: [1, 2, 3, 4, 5],
        8: [1, 2, 3, 4, 5, 6, 7, 8],
        17: [1, 2, 3, 4, 5, 6, 7, 8, 16, 17],
    }
    for max_bs, expected in cases.items():
        got = ModelRunner._cudagraph_batch_sizes(max_bs)
        print(f"max_bs={max_bs}, graph_bs={got}")
        assert got == expected
    print("ModelRunner CUDA Graph batch size coverage: PASS")


def test_scaled_fp8_batch_one_reuses_graph_host_staging() -> None:
    """Scaled KV has stable payload/scale addresses and can use the B1 fast path."""

    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.block_size = 4
    runner.uses_token_head_scales = True
    runner.kv_scale_cache = torch.zeros(2, 1, 1)
    runner.config = SimpleNamespace(
        compression_mode="scaled_fp8_kv",
        enable_visual_pruning_shadow=False,
        kvcache_block_size=4,
        paged_decode_block_n=256,
    )
    packed_model_inputs = torch.zeros(4, dtype=torch.int64)
    packed_decode_metadata = torch.full((6,), -1, dtype=torch.int32)
    runner.graph_vars = {
        1: {
            "host_packed_model_inputs_numpy": packed_model_inputs.numpy(),
            "host_packed_decode_metadata_numpy": packed_decode_metadata.numpy(),
            "host_packed_model_inputs": packed_model_inputs,
            "host_packed_decode_metadata": packed_decode_metadata,
            "host_input_ids": packed_model_inputs[:1],
            "host_positions": packed_model_inputs[1:].view(3, 1),
            "host_slot_mapping": packed_decode_metadata[:1],
            "host_context_lens": packed_decode_metadata[1:2],
            "host_decode_max_context_len": packed_decode_metadata[3:4],
            "host_block_tables": packed_decode_metadata[4:].view(1, 2),
        }
    }
    seq = Sequence(
        [10, 11],
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=4,
        request_id=0,
    )
    seq.block_table = [3]
    plan = SimpleNamespace(
        phase=BatchPhase.DECODE,
        batch_size=1,
        sequences=(seq,),
        sequence_ids=(seq.seq_id,),
        scheduled_token_counts=(1,),
    )

    batch = runner.prepare_single_greedy_decode_cudagraph(plan)

    assert batch is not None
    assert batch.attention_context.compression_metadata is not None
    assert batch.attention_context.compression_metadata.mode == "scaled_fp8_kv"
    assert batch.kv_scale_views[0].data_ptr() == runner.kv_scale_cache[0].data_ptr()
    assert batch.kv_scale_views[1].data_ptr() == runner.kv_scale_cache[1].data_ptr()
    assert packed_model_inputs.tolist() == [11, 1, 1, 1]
    assert packed_decode_metadata.tolist() == [13, 2, 2, 2, 3, -1]


def test_cudagraph_metadata_reports_capture_scope_and_selected_bucket() -> None:
    """Benchmark metadata 必须区分 actual batch、graph bucket 和 capture scope。"""

    runner = object.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.graph_bs = [1, 2, 3, 4, 5, 6, 7, 8]
    runner.cudagraph_capture_ms = 123.5

    metadata = runner.cudagraph_metadata(3)

    print(f"CUDA Graph execution metadata: {metadata}")
    assert metadata == {
        "enabled": True,
        "capture_scope": "decode_model_forward_logits_greedy",
        "capture_ms": 123.5,
        "batch_sizes": [1, 2, 3, 4, 5, 6, 7, 8],
        "requested_batch_size": 3,
        "selected_batch_size": 3,
        "batch_padding": 0,
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
        "fp8_lm_head_quantization_ms": 0.0,
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
        "fp8_lm_head_quantization_ms": 0.0,
    }
    print(f"disabled compile metadata: {metadata} PASS")


def test_compile_metadata_reports_stateless_fp8_candidate_path() -> None:
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        decode_compile_region="stateless",
        decode_compile_mode="default",
        decode_compile_emulate_precision_casts=True,
        decode_compile_force_same_precision=True,
    )
    runner.decode_compile_first_call_ms = 812.0
    runner.decode_fp8_lm_head_quantization_ms = 19.5

    metadata = runner.compile_metadata()

    assert metadata == {
        "enabled": True,
        "region": "decode_stateless",
        "subgraph": "batch1_o_proj_fp8_candidate_lm_head_exact_rerank",
        "kv_cache_boundary": "validated_runtime_store_and_paged_decode",
        "backend": "inductor",
        "mode": "default",
        "emulate_precision_casts": True,
        "force_same_precision": True,
        "first_call_ms": 812.0,
        "fp8_lm_head_quantization_ms": 19.5,
    }
