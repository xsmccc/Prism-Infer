"""P9-B configuration, identity, and execution-boundary contracts."""

from __future__ import annotations

import ast
import multiprocessing as mp
import pickle
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prism_infer.config import (
    CacheConfig,
    Config,
    ExecutionBackendName,
    ExecutionConfig,
    ModelConfig,
    MultimodalConfig,
    PrismConfig,
    SchedulerConfig,
)
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    DeviceBatch,
    DeviceModelInputs,
    ExecutionResult,
)
from prism_infer.engine.executor import ModelExecutor
from prism_infer.engine.execution_backend import (
    CudaGraphExecutionBackend,
    EagerExecutionBackend,
    create_execution_backend,
)
from prism_infer.engine.llm_engine import (
    LLMEngine,
    select_distributed_init_method,
)
from prism_infer.engine.request import MonotonicRequestIdAllocator, RequestState
from prism_infer.engine.sequence import Sequence
from prism_infer.models.model_registry import validate_model_architecture
from prism_infer.models.qwen3_vl_architecture import Qwen3VLArchitecture
from prism_infer.sampling_params import SamplingParams
from prism_infer.utils.context import Context, get_context, reset_context, use_context


def _spawn_config_probe(config: Config, sender) -> None:
    sender.send(
        (
            config.execution_backend,
            config.eos,
            config.num_kvcache_blocks,
            config.num_cpu_blocks,
            config.kvcache_block_size,
        )
    )
    sender.close()


def _patch_hf_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        max_position_embeddings=2048,
        text_config=SimpleNamespace(max_position_embeddings=2048),
    )
    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda _path: hf_config,
    )
    return hf_config


def _qwen3_vl_config() -> SimpleNamespace:
    text = SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=36,
        head_dim=128,
        vocab_size=151936,
        rope_theta=5_000_000,
        rms_norm_eps=1.0e-6,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        },
    )
    vision = SimpleNamespace(
        hidden_act="gelu_pytorch_tanh",
        hidden_size=1152,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=16,
        num_heads=16,
        intermediate_size=4304,
        depth=27,
        out_hidden_size=4096,
        num_position_embeddings=2304,
        spatial_merge_size=2,
        deepstack_visual_indexes=[8, 16, 24],
    )
    return SimpleNamespace(
        model_type="qwen3_vl",
        text_config=text,
        vision_config=vision,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        tie_word_embeddings=False,
    )


def test_unknown_flat_option_fails_before_model_or_gpu_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects: list[str] = []

    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda _path: effects.append("model_config"),
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.torch.cuda.device_count",
        lambda: effects.append("cuda") or 0,
    )

    with pytest.raises(TypeError, match="unknown Prism config option.*max_num_seq"):
        LLMEngine(str(tmp_path), max_num_seq=4)

    assert effects == []


def test_runtime_capabilities_fail_before_tokenizer_or_gpu_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects: list[str] = []
    config = SimpleNamespace(
        execution_backend="cuda_graph",
        compression_mode="off",
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.Config",
        lambda *args, **kwargs: config,
    )

    def reject_capabilities(**kwargs):
        effects.append("capabilities")
        raise RuntimeError("missing optimized kernel")

    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.validate_runtime_capabilities",
        reject_capabilities,
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: effects.append("tokenizer"),
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.torch.cuda.device_count",
        lambda: effects.append("cuda") or 1,
    )

    with pytest.raises(RuntimeError, match="missing optimized kernel"):
        LLMEngine("unused")

    assert effects == ["capabilities"]


def test_model_registry_rejects_unknown_family_before_tokenizer_or_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects: list[str] = []
    config = SimpleNamespace(
        execution_backend="eager",
        compression_mode="off",
        hf_config=SimpleNamespace(model_type="unsupported_vl"),
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.Config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.validate_runtime_capabilities",
        lambda **kwargs: effects.append("capabilities"),
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: effects.append("tokenizer"),
    )
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.torch.cuda.device_count",
        lambda: effects.append("cuda") or 1,
    )

    with pytest.raises(ValueError, match="unsupported model_type"):
        LLMEngine("unused")

    assert effects == ["capabilities"]


def test_qwen3_vl_architecture_contract_fails_closed() -> None:
    config = _qwen3_vl_config()
    architecture = Qwen3VLArchitecture.from_config(config)
    assert validate_model_architecture(config).value == "qwen3_vl"
    assert architecture.text.num_heads == 32
    assert architecture.vision.deepstack_visual_indexes == (8, 16, 24)

    config.vision_config.num_position_embeddings = 2305
    with pytest.raises(ValueError, match="perfect square"):
        Qwen3VLArchitecture.from_config(config)
    config.vision_config.num_position_embeddings = 2304

    config.text_config.attention_bias = True
    with pytest.raises(ValueError, match="attention_bias"):
        Qwen3VLArchitecture.from_config(config)
    config.text_config.attention_bias = False

    config.vision_config.out_hidden_size = 2048
    with pytest.raises(ValueError, match="vision output size"):
        Qwen3VLArchitecture.from_config(config)


def test_distributed_rendezvous_uses_a_dynamic_local_port() -> None:
    first = select_distributed_init_method()
    second = select_distributed_init_method()

    assert first.startswith("tcp://127.0.0.1:")
    assert second.startswith("tcp://127.0.0.1:")
    assert int(first.rsplit(":", 1)[1]) > 0
    assert int(second.rsplit(":", 1)[1]) > 0


def test_nested_config_and_flat_compatibility_adapter_are_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hf_config(monkeypatch)
    nested = PrismConfig(
        model=ModelConfig(
            model=str(tmp_path),
            max_model_len=1024,
            tensor_parallel_size=1,
            tensor_parallel_timeout_seconds=37,
            logits_precision="fp32",
            mlp_projection_mode="legacy",
        ),
        multimodal=MultimodalConfig(
            image_max_pixels=602112,
            video_max_pixels=802816,
            max_vision_patches_per_batch=7000,
            vision_encoder_microbatch_patches=3500,
            vision_attention_backend="flash_attn",
        ),
        cache=CacheConfig(
            gpu_memory_utilization=0.75,
            page_size=32,
            num_gpu_blocks=24,
            cpu_kv_cache_ratio=0.25,
            enable_prefix_caching=False,
        ),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_seqs=8,
            enable_chunked_prefill=False,
            max_chunk_size=256,
            max_queue_size=16,
            max_consecutive_prefill_batches=2,
        ),
        execution=ExecutionConfig(backend=ExecutionBackendName.EAGER),
    )
    nested_runtime = Config(nested)
    flat_runtime = Config(
        str(tmp_path),
        max_model_len=1024,
        tensor_parallel_size=1,
        tensor_parallel_timeout_seconds=37,
        logits_precision="fp32",
        mlp_projection_mode="legacy",
        image_max_pixels=602112,
        video_max_pixels=802816,
        max_vision_patches_per_batch=7000,
        vision_encoder_microbatch_patches=3500,
        vision_attention_backend="flash_attn",
        gpu_memory_utilization=0.75,
        kvcache_block_size=32,
        num_kvcache_blocks=24,
        cpu_kv_cache_ratio=0.25,
        enable_prefix_caching=False,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_chunked_prefill=False,
        max_chunk_size=256,
        max_queue_size=16,
        max_consecutive_prefill_batches=2,
        enforce_eager=True,
    )

    assert flat_runtime.prism_config == nested_runtime.prism_config
    assert flat_runtime.max_model_len == nested_runtime.max_model_len == 1024
    assert flat_runtime.tensor_parallel_timeout_seconds == 37
    assert flat_runtime.image_max_pixels == 602112
    assert flat_runtime.video_max_pixels == 802816
    assert flat_runtime.max_vision_patches_per_batch == 7000
    assert flat_runtime.vision_encoder_microbatch_patches == 3500
    assert flat_runtime.vision_attention_backend.value == "flash_attn"
    with pytest.raises(TypeError, match="cannot be combined with flat options"):
        Config(nested, max_num_seqs=4)


def test_vision_attention_backend_rejects_implicit_auto_policy() -> None:
    assert MultimodalConfig().vision_attention_backend.value == "sdpa"
    with pytest.raises(ValueError, match="vision attention backend"):
        MultimodalConfig(vision_attention_backend="auto")


def test_runtime_config_is_frozen_replaced_and_pickle_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hf_config(monkeypatch)
    initial = Config(
        str(tmp_path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        enforce_eager=True,
    )

    with pytest.raises(FrozenInstanceError):
        initial.eos = 42
    with pytest.raises(FrozenInstanceError):
        initial.scheduler_config.max_num_seqs = 1

    with_eos = initial.with_eos(42)
    resolved = with_eos.with_cache_capacity(
        num_kvcache_blocks=96,
        num_cpu_blocks=48,
    )

    assert initial.eos == -1
    assert initial.num_kvcache_blocks == -1
    assert with_eos is not initial
    assert with_eos.eos == 42
    assert with_eos.num_kvcache_blocks == -1
    assert resolved is not with_eos
    assert resolved.eos == 42
    assert resolved.num_kvcache_blocks == 96
    assert resolved.num_cpu_blocks == 48

    restored = pickle.loads(pickle.dumps(resolved))
    assert restored == resolved
    assert restored.prism_config == resolved.prism_config

    receiver, sender = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(
        target=_spawn_config_probe,
        args=(resolved, sender),
    )
    process.start()
    sender.close()
    assert receiver.recv() == ("eager", 42, 96, 48, 256)
    receiver.close()
    process.join(timeout=10)
    assert process.exitcode == 0


def test_compile_graph_requires_stateless_region_before_hf_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loads: list[str] = []
    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda path: model_loads.append(path),
    )

    with pytest.raises(ValueError, match="compile_graph.*requires.*stateless"):
        Config(str(tmp_path), execution_backend="compile_graph")

    assert model_loads == []


@pytest.mark.parametrize(
    "invalid_options",
    (
        {"enforce_eager": "false"},
        {"enable_chunked_prefill": 1},
        {"enable_prefix_caching": "yes"},
        {"decode_compile_force_same_precision": 0},
    ),
)
def test_typed_config_rejects_implicit_boolean_coercion_before_hf_loading(
    invalid_options: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loads: list[str] = []
    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda path: model_loads.append(path),
    )

    with pytest.raises(TypeError, match="must be a boolean"):
        Config(str(tmp_path), **invalid_options)

    assert model_loads == []


def test_cuda_graph_batch_limit_fails_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hf_config(monkeypatch)

    with pytest.raises(ValueError, match="cuda_graph.*max_num_seqs <= 512"):
        Config(str(tmp_path), max_num_seqs=513)

    eager = Config(str(tmp_path), max_num_seqs=513, enforce_eager=True)
    assert eager.max_num_seqs == 513


def test_all_production_sequence_construction_has_explicit_page_contract() -> None:
    with pytest.raises(TypeError, match="block_size"):
        Sequence([1])  # type: ignore[call-arg]

    package_root = Path(__file__).parents[1] / "prism_infer"
    missing: list[str] = []
    constructors = {
        "Sequence",
        "from_image_inputs",
        "from_single_image_inputs",
        "from_video_inputs",
    }
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name not in constructors:
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            if not {"block_size", "request_id"}.issubset(keyword_names):
                relative = source_path.relative_to(package_root.parent)
                missing.append(f"{relative}:{node.lineno}")

    assert missing == []
    assert not hasattr(Sequence, "set_block_size")
    assert not hasattr(Sequence, "block_size")


def test_engine_allocator_and_pickle_preserve_request_identity() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    engine.request_id_allocator = MonotonicRequestIdAllocator(next_request_id=700)
    engine.config = SimpleNamespace(kvcache_block_size=16)
    submitted: list[Sequence] = []

    class SchedulerStub:
        def add(self, seq: Sequence, *, raise_on_reject: bool):
            submitted.append(seq)
            return SimpleNamespace(accepted=True)

    engine.scheduler = SchedulerStub()
    first_id = engine.add_request([1, 2], SamplingParams(max_tokens=1))
    second_id = engine.add_request([3, 4], SamplingParams(max_tokens=1))

    assert (first_id, second_id) == (700, 701)
    first = submitted[0]
    first.transition_to(RequestState.PREFILLING, reason="test")
    first.transition_to(RequestState.DECODING, reason="test")
    first.append_token(9)
    restored = pickle.loads(pickle.dumps(first))

    assert restored.seq_id == first.seq_id == 700
    assert restored.status is RequestState.DECODING
    assert restored.block_size == first.block_size == 16
    with pytest.raises(TypeError, match="legacy Sequence payloads"):
        Sequence.__new__(Sequence).__setstate__((700, 16, [1, 2]))

    serialized = first.__getstate__()
    invalid_id = dict(serialized, seq_id="700")
    with pytest.raises(ValueError, match="serialized seq_id"):
        Sequence.__new__(Sequence).__setstate__(invalid_id)
    invalid_page = dict(serialized, block_size=True)
    with pytest.raises(ValueError, match="serialized block_size"):
        Sequence.__new__(Sequence).__setstate__(invalid_page)
    invalid_state = dict(serialized, request_state="DECODING")
    with pytest.raises(TypeError, match="state must be RequestState"):
        Sequence.__new__(Sequence).__setstate__(invalid_state)


def test_device_batch_is_frozen_sequence_free_and_phase_checked() -> None:
    inputs = DeviceModelInputs(
        input_ids=torch.tensor([1, 2]),
        position_ids=torch.tensor([0, 1]),
    )
    batch = DeviceBatch(
        phase=BatchPhase.PREFILL,
        sequence_ids=(10,),
        scheduled_token_counts=(2,),
        model_inputs=inputs,
        attention_context=Context(is_prefill=True),
        temperatures=torch.tensor([0.0]),
        execution_bucket=1,
    )

    with pytest.raises(FrozenInstanceError):
        batch.phase = BatchPhase.DECODE
    assert not any(isinstance(getattr(batch, field.name), Sequence) for field in fields(batch))
    assert "sequences" not in {field.name for field in fields(batch)}

    with pytest.raises(ValueError, match="phase/context mismatch"):
        DeviceBatch(
            phase=BatchPhase.PREFILL,
            sequence_ids=(10,),
            scheduled_token_counts=(2,),
            model_inputs=inputs,
            attention_context=Context(is_prefill=False),
            temperatures=torch.tensor([0.0]),
            execution_bucket=1,
        )

    with pytest.raises(TypeError, match="sequence_ids must be an immutable tuple"):
        DeviceBatch(
            phase=BatchPhase.PREFILL,
            sequence_ids=[10],  # type: ignore[arg-type]
            scheduled_token_counts=(2,),
            model_inputs=inputs,
            attention_context=Context(is_prefill=True),
            temperatures=torch.tensor([0.0]),
            execution_bucket=1,
        )


def test_model_executor_dispatches_only_the_run_plan_contract() -> None:
    seq = Sequence([1], block_size=16, request_id=5)
    plan = BatchPlan(
        phase=BatchPhase.PREFILL,
        sequences=(seq,),
        scheduled_token_counts=(1,),
    )

    class RunPlanOnlyRunner:
        kv_cache_dtype = torch.bfloat16

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def call(self, method_name: str, *args: object) -> ExecutionResult:
            self.calls.append((method_name, args))
            if method_name != "run_plan":
                raise AssertionError(f"unexpected runner method: {method_name}")
            return ExecutionResult(token_ids=(8,))

    runner = RunPlanOnlyRunner()
    executor = ModelExecutor(
        SimpleNamespace(compression_mode="off"),
        runner,
        SimpleNamespace(),
    )

    assert executor.execute(plan).token_ids == (8,)
    assert runner.calls == [("run_plan", (plan,))]


def test_executor_rejects_runner_result_with_wrong_batch_cardinality() -> None:
    seq = Sequence([1], block_size=16, request_id=5)
    plan = BatchPlan(
        phase=BatchPhase.PREFILL,
        sequences=(seq,),
        scheduled_token_counts=(1,),
    )

    runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        call=lambda _method, *_args: ExecutionResult(token_ids=(8, 9)),
    )
    executor = ModelExecutor(
        SimpleNamespace(compression_mode="off"),
        runner,
        SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="planned batch size.*2 != 1"):
        executor.execute(plan)


def test_execution_backend_resets_installed_context_on_failure() -> None:
    runner = SimpleNamespace(
        config=SimpleNamespace(execution_backend="eager"),
        rank=0,
    )

    def fail_forward(_inputs, *, is_prefill):
        raise RuntimeError("synthetic backend failure")

    runner.run_model_eager = fail_forward
    backend = EagerExecutionBackend(runner)
    batch = DeviceBatch(
        phase=BatchPhase.PREFILL,
        sequence_ids=(10,),
        scheduled_token_counts=(1,),
        model_inputs=DeviceModelInputs(
            input_ids=torch.tensor([1]),
            position_ids=torch.tensor([0]),
        ),
        attention_context=Context(
            is_prefill=True,
            slot_mapping=torch.tensor([7], dtype=torch.int32),
        ),
        temperatures=torch.tensor([0.0]),
        execution_bucket=1,
    )

    reset_context()
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        backend.execute(batch)
    assert get_context().slot_mapping is None


def test_execution_backend_factory_is_typed_and_breaks_runner_ownership() -> None:
    eager_runner = SimpleNamespace(
        config=SimpleNamespace(execution_backend="eager"),
    )
    graph_runner = SimpleNamespace(
        config=SimpleNamespace(execution_backend="cuda_graph"),
    )

    eager = create_execution_backend(eager_runner)
    graph = create_execution_backend(graph_runner)

    assert isinstance(eager, EagerExecutionBackend)
    assert isinstance(graph, CudaGraphExecutionBackend)
    eager.release()
    with pytest.raises(RuntimeError, match="released"):
        _ = eager.runner


def test_cuda_graph_backend_forbids_runtime_eager_fallback() -> None:
    calls: list[str] = []
    runner = SimpleNamespace(
        config=SimpleNamespace(execution_backend="cuda_graph"),
        graph_bs=[1],
        run_model_eager=lambda *args, **kwargs: calls.append("eager"),
        run_model_cudagraph=lambda *args, **kwargs: calls.append("graph"),
    )
    backend = CudaGraphExecutionBackend(runner)
    batch = DeviceBatch(
        phase=BatchPhase.DECODE,
        sequence_ids=(10,),
        scheduled_token_counts=(1,),
        model_inputs=DeviceModelInputs(
            input_ids=torch.tensor([1]),
            position_ids=torch.tensor([0]),
        ),
        attention_context=Context(
            is_prefill=False,
            compression_metadata=SimpleNamespace(mode="visual_prune"),
        ),
        temperatures=torch.tensor([0.0]),
        execution_bucket=1,
    )

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        backend.forward_logits(batch)
    assert calls == []


def test_execution_context_restores_nested_scopes() -> None:
    reset_context()
    outer_mapping = torch.tensor([11], dtype=torch.int32)
    inner_mapping = torch.tensor([22], dtype=torch.int32)

    with use_context(Context(is_prefill=True, slot_mapping=outer_mapping)):
        assert get_context().slot_mapping is outer_mapping
        with use_context(Context(is_prefill=False, slot_mapping=inner_mapping)):
            assert get_context().slot_mapping is inner_mapping
        assert get_context().slot_mapping is outer_mapping

    assert get_context().slot_mapping is None
