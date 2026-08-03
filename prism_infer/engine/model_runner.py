# ═══════════════════════════════════════════════════════════════
# model_runner.py —— 模型执行器 (engine 层最核心的文件)
#
# 职责: 把 scheduler 的调度结果 (一批序列) 转换成 GPU tensor,
#       喂给模型前向推理, 采样出下一个 token
#
# 关键概念:
#   - Prefill 准备: 拼接多条序列的 token, 计算位置和 slot_mapping
#   - Decode 准备: 每条序列只取最后一个 token
#   - KV Cache 分配: 一次性预分配 GPU 显存, 分配给各 attention 层
#   - CUDA Graph: 预录制 Decode 的 GPU 操作, 消除 CPU launch 开销
#   - Tensor Parallel: 多 GPU 通过变长 IPC 控制通道同步调用
#
# C++ 类比: 整个文件 ≈ inference engine 的 execute() 函数
# ═══════════════════════════════════════════════════════════════

import traceback
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection
from time import perf_counter

import torch
import torch.distributed as dist  # 分布式通信 (NCCL)

from prism_infer.analysis.schema_constants import (
    DECODE_COMPILE_KV_BOUNDARY,
    DECODE_COMPILE_STATELESS_SUBGRAPH,
    DECODE_COMPILE_SUBGRAPH,
)
from prism_infer.config import (
    MAX_CUDA_GRAPH_BATCH_SIZE,
    Config,
)
from prism_infer.engine.compression import (
    COMPRESSION_OFF,
    CompressionMetadata,
    build_compression_metadata,
    build_visual_pruning_config,
    compression_mode_supports_cuda_graph,
    compression_mode_uses_fp8_payload,
    compression_mode_uses_token_head_scales,
    compression_supports_cuda_graph,
)
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    DeviceBatch,
    DeviceModelInputs,
    ExecutionResult,
    PrefillSlice,
    PreparedModelInputs,
    SingleGreedyDecodeState,
)
from prism_infer.engine.execution_backend import create_execution_backend
from prism_infer.engine.input_preparation import ModelInputPreparer
from prism_infer.engine.kv_layout import KVCompactionPlan
from prism_infer.engine.kv_quantization import (
    KV_COMPONENT_COUNT,
    KV_SCALE_DTYPE,
    kv_block_storage_bytes,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.tp_control import (
    TPCommand,
    TPControlPlane,
    TPMethod,
    TPResponse,
)
from prism_infer.engine.visual_pruning import (
    finalize_attention_pruning_decisions,
)
from prism_infer.observability import (
    is_trace_enabled,
    profile_region,
    register_model_config,
)
from prism_infer.ops.kv_compaction import compact_kv_slots

try:
    from prism_infer.models.qwen3 import Qwen3ForCausalLM  # Qwen3 纯文本模型 (legacy)
except ImportError:
    Qwen3ForCausalLM = None  # VL 项目中纯文本模型可能不存在, 用 VL 版替代
from prism_infer.layers.sampler import (  # 采样器 (温度采样/贪婪)
    SAMPLING_NUMERICAL_EPSILON,
    Sampler,
)
from prism_infer.models.model_registry import ModelFamily, resolve_model_family
from prism_infer.models.qwen3_vl import (
    Qwen3VLForCausalLM,
    Qwen3VLTextForwardState,
    compile_decode_fp8_lm_head,
    compile_decode_o_proj,
)
from prism_infer.models.qwen3_vl_architecture import MROPE_AXIS_COUNT
from prism_infer.utils.context import (
    Context,
    get_context,
    install_context,
    reset_context,
    use_context,
)
from prism_infer.utils.loader import load_model  # 权重加载
from prism_infer.vision.vision_encoder import VisionEncoderForwardState

CUDA_GRAPH_EXACT_BATCH_LIMIT = 8
CUDA_GRAPH_BATCH_BUCKET_STRIDE = 16
MROPE_DECODE_POSITION_RANK = 2
VISUAL_EMBEDDING_CACHE_MAX_BYTES = 256 * 1024 * 1024


@dataclass(slots=True)
class _CooperativeVisionPayload:
    """One image or video payload split only at attention-safe boundaries."""

    token_id: int
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    microbatches: tuple[
        tuple[int, int, tuple[tuple[int, int, int], ...]],
        ...,
    ]
    next_microbatch: int = 0
    active_state: VisionEncoderForwardState | None = None
    main_parts: list[torch.Tensor] | None = None
    deepstack_parts: list[list[torch.Tensor]] | None = None


@dataclass(slots=True)
class _PendingPrefillExecution:
    """One prefill paused at an exact vision or language block boundary."""

    plan: BatchPlan
    device_batch: DeviceBatch
    model_state: Qwen3VLTextForwardState | None
    vision_payloads: list[_CooperativeVisionPayload]
    next_vision_payload: int = 0
    sampled_tokens: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _VisualEmbeddingCacheEntry:
    """Exact Vision Encoder outputs retained for one content fingerprint."""

    visual_embeds: torch.Tensor
    deepstack_visual_embeds: tuple[torch.Tensor, ...]
    storage_bytes: int


def _resolve_model_dtype(hf_config) -> torch.dtype:
    """从 HF config 或 text_config 中解析模型权重 dtype。"""

    text_config = getattr(hf_config, "text_config", None)
    dtype = (
        getattr(hf_config, "torch_dtype", None)
        or getattr(hf_config, "dtype", None)
        or getattr(text_config, "torch_dtype", None)
        or getattr(text_config, "dtype", None)
        or torch.bfloat16
    )
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
    return dtype


def _text_hf_config(hf_config):
    """Qwen3-VL 的 LLM 配置在 text_config；纯文本模型直接用顶层 config。"""

    return getattr(hf_config, "text_config", hf_config)


@contextmanager
def _model_initialization_defaults(dtype: torch.dtype) -> Iterator[None]:
    """Scope global torch defaults used while constructing model parameters."""

    default_dtype = torch.get_default_dtype()
    default_device = torch.get_default_device()
    try:
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")
        yield
    finally:
        try:
            torch.set_default_device(default_device)
        finally:
            torch.set_default_dtype(default_dtype)


class ModelRunner:
    # ─────────────────────────────────────────────────────────
    # __init__: 初始化模型、KV Cache、CUDA Graph
    # ─────────────────────────────────────────────────────────
    def __init__(
        self,
        config: Config,
        rank: int,
        control_channel: Connection | list[Connection],
        distributed_init_method: str,
    ) -> None:
        self.config = config
        hf_config = config.hf_config  # HuggingFace 模型配置 (层数/head数等)
        self.block_size = config.kvcache_block_size  # KV Cache 块大小 (如 16)
        self.enforce_eager = config.enforce_eager  # True = 禁用 CUDA Graph, 每次都 eager 执行
        self.world_size = config.tensor_parallel_size  # 几张 GPU (Tensor Parallel 并行度)
        self.rank = rank  # 当前 GPU 编号 (0, 1, 2, ...)
        self.control_channel = control_channel  # rank0 为发送端列表, worker 为接收端
        self.tp_control = TPControlPlane(
            rank=rank,
            world_size=self.world_size,
            channel=control_channel,
            timeout_seconds=config.tensor_parallel_timeout_seconds,
        )
        self.input_preparer = ModelInputPreparer(config, block_size=self.block_size)
        self.model_dtype = _resolve_model_dtype(hf_config)
        self.kv_cache_dtype = self._resolve_kv_cache_dtype(config)
        self.uses_token_head_scales = compression_mode_uses_token_head_scales(
            config.compression_mode
        )
        self.kv_scale_cache: torch.Tensor | None = None
        self.cpu_kv_scale_cache: torch.Tensor | None = None
        self.cudagraph_capture_ms = 0.0
        self.decode_compile_first_call_ms = 0.0
        self.decode_fp8_lm_head_quantization_ms = 0.0
        self.decode_compile_first_call_pending = False
        self.last_cudagraph_actual_batch_size: int | None = None
        self.last_cudagraph_replay_batch_size: int | None = None
        self._visual_embedding_cache: OrderedDict[
            str,
            _VisualEmbeddingCacheEntry,
        ] = OrderedDict()
        self._visual_embedding_cache_resident_bytes = 0
        self._visual_embedding_cache_hits = 0
        self._visual_embedding_cache_misses = 0
        self._visual_embedding_cache_evictions = 0
        self._visual_embedding_cache_oversize_skips = 0
        register_model_config(config)
        self._initialize_distributed(distributed_init_method)
        self._initialize_model_runtime(hf_config)
        self._synchronize_tensor_parallel_startup()

    def _initialize_distributed(self, distributed_init_method: str) -> None:
        """Bind the rank-local GPU and initialize the process group."""

        if not isinstance(distributed_init_method, str):
            raise TypeError("distributed_init_method must be a string")
        torch.cuda.set_device(self.rank)
        distributed_backend = "nccl" if self.world_size > 1 else "gloo"
        init_options = {
            "backend": distributed_backend,
            "init_method": distributed_init_method,
            "world_size": self.world_size,
            "rank": self.rank,
            "timeout": timedelta(seconds=self.tp_control.timeout()),
        }
        if distributed_backend == "nccl":
            init_options["device_id"] = torch.device("cuda", self.rank)
        dist.init_process_group(**init_options)

    def _create_model(self, hf_config) -> None:
        """Instantiate exactly one validated model family on the active GPU."""

        model_family = resolve_model_family(hf_config)
        if model_family is ModelFamily.QWEN3_VL:
            self.model = Qwen3VLForCausalLM(
                hf_config,
                mlp_projection_mode=self.config.mlp_projection_mode,
                vision_encoder_microbatch_patches=(self.config.vision_encoder_microbatch_patches),
                vision_attention_backend=self.config.vision_attention_backend,
                enable_vision_tensor_cudagraph=(self.config.enable_vision_tensor_cudagraph),
            )
            self.model.logits_precision = self.config.logits_precision
            for layer in self.model.model.language_model.layers:
                layer.self_attn.fused_qk_rmsnorm_enabled = getattr(
                    self.config, "enable_fused_qk_rmsnorm", False
                )
                layer.self_attn.fused_qk_mrope_enabled = getattr(
                    self.config,
                    "enable_fused_qk_mrope",
                    False,
                )
                layer.self_attn.packed_kv_projection_enabled = getattr(
                    self.config,
                    "enable_packed_kv_projection",
                    False,
                )
            self.model.model.language_model.fused_add_rmsnorm_enabled = getattr(
                self.config,
                "enable_fused_add_rmsnorm",
                False,
            )
            self.is_vl_model = True
            return
        if Qwen3ForCausalLM is None:
            raise RuntimeError("Qwen3 text model support is unavailable in this build")
        if self.config.logits_precision != "model":
            raise ValueError(
                "logits_precision='fp32' historical reproduction is "
                "currently supported only for Qwen3-VL"
            )
        self.model = Qwen3ForCausalLM(hf_config)
        self.is_vl_model = False

    def _initialize_model_runtime(self, hf_config) -> None:
        """Load weights, warm kernels, allocate KV storage, and select a backend."""

        try:
            with _model_initialization_defaults(self.model_dtype):
                self._create_model(hf_config)
                load_model(self.model, self.config.model)
                self.sampler = Sampler()
                self.execution_backend = create_execution_backend(self)
                self.execution_backend.warmup()
                self.allocate_kv_cache()
                self._configure_decode_compile()
                if not self.enforce_eager:
                    self.execution_backend.capture()
        except BaseException:
            self._release_execution_backend()
            reset_context()
            raise

    def _release_execution_backend(self) -> None:
        backend = getattr(self, "execution_backend", None)
        if backend is None:
            return
        backend.release()
        del self.execution_backend

    def _synchronize_tensor_parallel_startup(self) -> None:
        if self.world_size <= 1:
            return
        self._tp_barrier()
        if self.rank > 0:
            self.loop()

    # ─────────────────────────────────────────────────────────
    # exit: 清理资源
    # ─────────────────────────────────────────────────────────
    def _tp_barrier(self) -> None:
        """Synchronize NCCL ranks on each process's explicitly bound GPU."""

        dist.barrier(device_ids=[self.rank])

    @property
    def control_timeout_seconds(self) -> float:
        """Compatibility view of the control-plane timeout."""

        return self.tp_control.timeout_seconds

    @control_timeout_seconds.setter
    def control_timeout_seconds(self, value: float) -> None:
        self.tp_control.timeout_seconds = value

    def exit(self) -> None:
        # The control pipe remains open until workers acknowledge this method.
        if self.world_size > 1:
            self._tp_barrier()
        # All model, cache, and CUDA Graph tensors must remain alive until
        # their last queued kernel has completed.
        torch.cuda.synchronize()
        visual_embedding_cache = getattr(self, "_visual_embedding_cache", None)
        if visual_embedding_cache is not None:
            visual_embedding_cache.clear()
        self._visual_embedding_cache_resident_bytes = 0
        self._release_execution_backend()
        reset_context()
        dist.destroy_process_group()

    def _control_timeout(self) -> float:
        """Compatibility delegate for older tests and integrations."""

        return self.tp_control.timeout()

    def _rank0_control_channels(self) -> list[Connection]:
        """Compatibility delegate for older tests and integrations."""

        return self.tp_control.rank0_channels()

    def _next_control_command(
        self,
        method_name: str,
        args: tuple[object, ...],
    ) -> TPCommand:
        """Compatibility delegate for older tests and integrations."""

        return self.tp_control.next_command(method_name, args)

    @staticmethod
    def _deserialize_control_message(data: bytes) -> tuple[str, list[object]]:
        """Compatibility adapter around the typed command decoder."""

        return TPControlPlane.deserialize_message(data)

    def read_control_command(self) -> TPCommand:
        """Read and validate one typed command on a worker rank."""

        return self.tp_control.read_command()

    def read_control_message(self) -> tuple[str, list[object]]:
        """Compatibility view of :meth:`read_control_command`."""

        command = self.read_control_command()
        return command.method.value, list(command.args)

    def _broadcast_control_command(
        self,
        method_name: str,
        args: tuple[object, ...],
    ) -> tuple[TPCommand, int]:
        """Compatibility delegate for older tests and integrations."""

        return self.tp_control.broadcast(method_name, args)

    def write_control_message(self, method_name: str, *args: object) -> int:
        """Broadcast one typed command and return its serialized byte size."""

        _, payload_bytes = self._broadcast_control_command(
            method_name,
            tuple(args),
        )
        return payload_bytes

    def _send_control_response(self, response: TPResponse) -> None:
        """Compatibility delegate for older tests and integrations."""

        self.tp_control.send_response(response)

    def _await_worker_responses(self, command: TPCommand) -> None:
        """Compatibility delegate for older tests and integrations."""

        self.tp_control.await_responses(command)

    def _invoke_local(self, method_name: str, args: tuple[object, ...]) -> object:
        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"unknown ModelRunner method: {method_name!r}")
        return method(*args)

    def loop(self) -> None:
        """Execute typed commands and report completion or failure to rank 0."""

        try:
            while True:
                command = self.tp_control.read_command()
                try:
                    self._invoke_local(command.method.value, command.args)
                except BaseException as exc:
                    response = TPResponse.error(
                        command,
                        worker_rank=self.rank,
                        error=exc,
                        traceback_text=traceback.format_exc(),
                    )
                    self.tp_control.send_response(response)
                    raise
                self.tp_control.send_response(TPResponse.ok(command, worker_rank=self.rank))
                if command.method is TPMethod.EXIT:
                    break
        finally:
            self.tp_control.close_worker_channel()

    def call(self, method_name: str, *args: object) -> object:
        """Invoke a runner method and keep every TP rank in lockstep."""

        # Resolve locally before broadcasting so an invalid method cannot leave
        # workers waiting in a command that rank 0 never executes.
        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"unknown ModelRunner method: {method_name!r}")
        if self.world_size <= 1 or self.rank != 0:
            return method(*args)

        command, _ = self.tp_control.broadcast(
            method_name,
            tuple(args),
        )
        local_result: object = None
        local_error: BaseException | None = None
        try:
            local_result = method(*args)
        except BaseException as exc:
            local_error = exc
        try:
            self.tp_control.await_responses(command)
        except BaseException as control_error:
            if local_error is not None:
                raise control_error from local_error
            raise
        if local_error is not None:
            raise local_error
        return local_result

    # ═══════════════════════════════════════════════════════════
    # 初始化阶段: warmup + allocate_kv_cache
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_kv_cache_dtype(config: Config) -> torch.dtype:
        """根据 compression mode 选择物理 KV cache dtype。"""

        if compression_mode_uses_fp8_payload(config.compression_mode):
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError(
                    f"compression_mode={config.compression_mode!r} requires torch.float8_e4m3fn"
                )
            return torch.float8_e4m3fn
        return _resolve_model_dtype(config.hf_config)

    def _forward_model(self, inputs: DeviceModelInputs):
        """Call either the new Qwen3-VL interface or the legacy text model."""
        if self.is_vl_model:
            return self.model(
                input_ids=inputs.input_ids,
                position_ids=inputs.position_ids,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                pixel_values_videos=inputs.pixel_values_videos,
                video_grid_thw=inputs.video_grid_thw,
                visual_embeds=inputs.visual_embeds,
                deepstack_visual_embeds=(inputs.deepstack_visual_embeds),
            )
        return self.model(inputs.input_ids, inputs.position_ids)

    @staticmethod
    def _visual_cache_device_tensor(value: torch.Tensor) -> torch.Tensor:
        if value.is_cuda:
            return value
        return value.pin_memory().cuda(non_blocking=True)

    @staticmethod
    def _visual_cache_storage_bytes(
        visual_embeds: torch.Tensor,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
    ) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (visual_embeds, *deepstack_visual_embeds)
        )

    @staticmethod
    def _attach_visual_embedding_cache_entry(
        seq: Sequence,
        entry: _VisualEmbeddingCacheEntry,
    ) -> None:
        seq.precomputed_visual_embeds = entry.visual_embeds
        seq.precomputed_deepstack_visual_embeds = entry.deepstack_visual_embeds

    @torch.inference_mode()
    def hydrate_visual_embedding_cache(self, seq: Sequence) -> None:
        """Attach or compute exact visual outputs for one admitted request."""

        if not self.config.enable_visual_embedding_cache:
            return
        if self.world_size != 1 or not self.is_vl_model:
            raise RuntimeError("visual embedding cache currently requires Qwen3-VL with TP1")
        key = seq.visual_embedding_cache_key
        if key is None:
            return

        cached = self._visual_embedding_cache.get(key)
        if cached is not None:
            self._visual_embedding_cache.move_to_end(key)
            self._visual_embedding_cache_hits += 1
            self._attach_visual_embedding_cache_entry(seq, cached)
            return

        payloads = [
            (seq.pixel_values, seq.image_grid_thw),
            (seq.pixel_values_videos, seq.video_grid_thw),
        ]
        present = [(payload, grid) for payload, grid in payloads if payload is not None]
        if len(present) != 1:
            raise RuntimeError(
                "visual embedding cache requires exactly one visual modality "
                f"per request, got {len(present)} for seq={seq.seq_id}"
            )
        payload, grid = present[0]
        if grid is None:
            raise RuntimeError(
                f"visual cache payload is missing grid metadata for seq={seq.seq_id}"
            )
        payload = self._visual_cache_device_tensor(payload)
        grid = self._visual_cache_device_tensor(grid)
        self._visual_embedding_cache_misses += 1
        with profile_region(
            "model.vision.embedding_cache_miss",
            metadata={
                "patch_tokens": int(payload.shape[0]),
                "seq_id": seq.seq_id,
            },
        ):
            visual_embeds, deepstack = self.model.model._encode_visual_payload(payload, grid)
        visual_embeds = visual_embeds.detach()
        deepstack_visual_embeds = tuple(value.detach() for value in deepstack)
        expected_rows = seq.image_token_count + seq.video_token_count
        if (
            int(visual_embeds.shape[0]) != expected_rows
            or not deepstack_visual_embeds
            or any(value.shape != visual_embeds.shape for value in deepstack_visual_embeds)
        ):
            raise RuntimeError(
                "Vision Encoder cache output does not match visual placeholders: "
                f"seq={seq.seq_id} rows={visual_embeds.shape[0]} "
                f"expected={expected_rows}"
            )
        storage_bytes = self._visual_cache_storage_bytes(
            visual_embeds,
            deepstack_visual_embeds,
        )
        entry = _VisualEmbeddingCacheEntry(
            visual_embeds=visual_embeds,
            deepstack_visual_embeds=deepstack_visual_embeds,
            storage_bytes=storage_bytes,
        )
        self._attach_visual_embedding_cache_entry(seq, entry)
        if storage_bytes > VISUAL_EMBEDDING_CACHE_MAX_BYTES:
            self._visual_embedding_cache_oversize_skips += 1
            return
        while (
            self._visual_embedding_cache
            and self._visual_embedding_cache_resident_bytes + storage_bytes
            > VISUAL_EMBEDDING_CACHE_MAX_BYTES
        ):
            _, evicted = self._visual_embedding_cache.popitem(last=False)
            self._visual_embedding_cache_resident_bytes -= evicted.storage_bytes
            self._visual_embedding_cache_evictions += 1
        self._visual_embedding_cache[key] = entry
        self._visual_embedding_cache_resident_bytes += storage_bytes

    def visual_embedding_cache_metadata(self) -> dict[str, object]:
        """Return bounded cache state and measured-run counters."""

        return {
            "enabled": self.config.enable_visual_embedding_cache,
            "identity": "sha256_model_processor_media_layout_v1",
            "scope": "vision_encoder_main_and_deepstack_outputs_only",
            "max_bytes": VISUAL_EMBEDDING_CACHE_MAX_BYTES,
            "resident_bytes": self._visual_embedding_cache_resident_bytes,
            "entries": len(self._visual_embedding_cache),
            "hits": self._visual_embedding_cache_hits,
            "misses": self._visual_embedding_cache_misses,
            "evictions": self._visual_embedding_cache_evictions,
            "oversize_skips": (self._visual_embedding_cache_oversize_skips),
        }

    def reset_visual_embedding_cache_metrics(self) -> None:
        """Reset counters while retaining warm exact encoder outputs."""

        self._visual_embedding_cache_hits = 0
        self._visual_embedding_cache_misses = 0
        self._visual_embedding_cache_evictions = 0
        self._visual_embedding_cache_oversize_skips = 0

    def _configure_decode_compile(self) -> None:
        """Enable exactly one explicitly configured decode compile region."""

        region = self.config.decode_compile_region
        if region == "none":
            return
        if not self.is_vl_model:
            raise RuntimeError("decode compile currently supports only Qwen3-VL")
        attention_layers = [layer.self_attn for layer in self.model.model.language_model.layers]
        if not attention_layers:
            raise RuntimeError("decode compile found no decoder layers")
        if region == "attention":
            for attention in attention_layers:
                attention.enable_decode_compile(
                    mode=self.config.decode_compile_mode,
                    emulate_precision_casts=(self.config.decode_compile_emulate_precision_casts),
                    force_same_precision=(self.config.decode_compile_force_same_precision),
                )
            self.decode_compile_first_call_pending = True
            return
        if region != "stateless":
            raise RuntimeError(f"unsupported decode compile region: {region!r}")
        compiled_o_proj = compile_decode_o_proj(
            mode=self.config.decode_compile_mode,
            emulate_precision_casts=self.config.decode_compile_emulate_precision_casts,
            force_same_precision=self.config.decode_compile_force_same_precision,
        )
        for attention in attention_layers:
            attention.enable_decode_o_proj_compile(compiled_o_proj)
        torch.cuda.synchronize()
        quantize_start = perf_counter()
        fp8_lm_head_inputs = self.model.prepare_decode_fp8_greedy_lm_head()
        torch.cuda.synchronize()
        self.decode_fp8_lm_head_quantization_ms = (perf_counter() - quantize_start) * 1000.0
        compiled_fp8_lm_head = compile_decode_fp8_lm_head(
            mode=self.config.decode_compile_mode,
            emulate_precision_casts=self.config.decode_compile_emulate_precision_casts,
            force_same_precision=self.config.decode_compile_force_same_precision,
        )
        self.model.enable_decode_fp8_greedy_lm_head_compile(compiled_fp8_lm_head)
        probe = torch.zeros(
            1,
            attention_layers[0].o_proj.in_features,
            device=attention_layers[0].o_proj.weight.device,
            dtype=attention_layers[0].o_proj.weight.dtype,
        )
        torch.cuda.synchronize()
        compile_start = perf_counter()
        compiled_outputs = [
            compiled_o_proj(probe, attention_layers[0].o_proj.weight),
            compiled_fp8_lm_head(probe, *fp8_lm_head_inputs),
        ]
        torch.cuda.synchronize()
        self.decode_compile_first_call_ms = (perf_counter() - compile_start) * 1000.0
        del compiled_outputs, probe
        torch.cuda.empty_cache()

    def compile_metadata(self) -> dict[str, object]:
        """返回 decode ``torch.compile`` 的可审计配置和 cold first call。"""

        enabled = self.config.decode_compile_region != "none"
        region = self.config.decode_compile_region
        subgraph = {
            "attention": DECODE_COMPILE_SUBGRAPH,
            "stateless": DECODE_COMPILE_STATELESS_SUBGRAPH,
        }.get(region, "none")
        return {
            "enabled": enabled,
            "region": (f"decode_{region}" if enabled else "none"),
            "subgraph": subgraph,
            "kv_cache_boundary": DECODE_COMPILE_KV_BOUNDARY if enabled else "none",
            "backend": "inductor" if enabled else "none",
            "mode": self.config.decode_compile_mode if enabled else "none",
            "emulate_precision_casts": (
                self.config.decode_compile_emulate_precision_casts if enabled else False
            ),
            "force_same_precision": (
                self.config.decode_compile_force_same_precision if enabled else False
            ),
            "first_call_ms": self.decode_compile_first_call_ms if enabled else 0.0,
            "fp8_lm_head_quantization_ms": (
                getattr(self, "decode_fp8_lm_head_quantization_ms", 0.0)
                if region == "stateless"
                else 0.0
            ),
        }

    def vision_tensor_cudagraph_metadata(self) -> dict[str, object]:
        """Return fixed-shape Vision tensor-region Graph captures."""

        if not self.is_vl_model:
            return {
                "enabled": False,
                "capture_count": 0,
                "captures": [],
            }
        return self.model.model.visual.tensor_cudagraph_metadata()

    def warmup_model(self):
        """热身: 用最大尺寸输入跑一次模型, 触发 CUDA kernel 编译/分配"""
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        # max_num_batched_tokens: 一批最多处理多少 token
        # max_model_len: 单条序列最大长度
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 计算最大并发序列数 (受 token 数和序列数上限约束)
        seqs = [
            Sequence(
                [0] * max_model_len,
                block_size=self.block_size,
                request_id=synthetic_request_id,
            )
            for synthetic_request_id in range(num_seqs)
        ]
        # 造假序列: 每条都是 max_model_len 个 0
        self.run(seqs, True)  # 跑一次 Prefill (前向传播)
        # 目的: 让 PyTorch/CUDA 编译 kernel, 知道峰值内存
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 GPU 剩余显存, 计算能分配多少 KV Cache block, 一次性分配"""
        text_config = _text_hf_config(self.config.hf_config)
        num_kv_heads, head_dim = self._kv_cache_geometry(text_config)
        storage_bytes = kv_block_storage_bytes(
            num_layers=text_config.num_hidden_layers,
            page_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            payload_dtype=self.kv_cache_dtype,
            token_head_scales=self.uses_token_head_scales,
        )
        self.kv_payload_block_bytes = storage_bytes.payload
        self.kv_scale_block_bytes = storage_bytes.scales
        self.kv_block_bytes = storage_bytes.total
        self.num_kvcache_blocks = self._select_num_kv_blocks(storage_bytes.total)
        self._allocate_gpu_kv_storage(text_config, num_kv_heads, head_dim)
        self._bind_attention_kv_views(text_config.num_hidden_layers)
        self._allocate_cpu_kv_storage(text_config, num_kv_heads, head_dim)

    def _kv_cache_geometry(self, text_config) -> tuple[int, int]:
        total_kv_heads = text_config.num_key_value_heads
        if total_kv_heads % self.world_size != 0:
            raise ValueError(
                "num_key_value_heads must be divisible by tensor parallel size: "
                f"heads={total_kv_heads}, tp_size={self.world_size}"
            )
        num_kv_heads = total_kv_heads // self.world_size
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            if text_config.hidden_size % text_config.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            head_dim = text_config.hidden_size // text_config.num_attention_heads
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive integer, got {head_dim!r}")
        return num_kv_heads, head_dim

    def _select_num_kv_blocks(self, block_bytes: int) -> int:
        free, total = torch.cuda.mem_get_info()
        memory_stats = torch.cuda.memory_stats()
        used = total - free
        peak = memory_stats["allocated_bytes.all.peak"]
        current = memory_stats["allocated_bytes.all.current"]
        available_bytes = int(total * self.config.gpu_memory_utilization - used - peak + current)
        max_blocks = available_bytes // block_bytes
        requested_blocks = self.config.num_kvcache_blocks
        if requested_blocks > 0 and requested_blocks > max_blocks:
            raise RuntimeError(
                "requested num_kvcache_blocks exceeds available memory: "
                f"requested={requested_blocks}, max={max_blocks}, "
                f"kv_cache_dtype={self.kv_cache_dtype}, "
                f"payload_block_bytes={self.kv_payload_block_bytes}, "
                f"scale_block_bytes={self.kv_scale_block_bytes}"
            )
        selected_blocks = requested_blocks if requested_blocks > 0 else max_blocks
        if selected_blocks <= 0:
            raise RuntimeError("no KV cache blocks fit in the configured GPU memory budget")
        return selected_blocks

    def _allocate_gpu_kv_storage(self, text_config, num_kv_heads: int, head_dim: int) -> None:
        self.kv_cache = torch.empty(
            KV_COMPONENT_COUNT,
            text_config.num_hidden_layers,
            self.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=self.kv_cache_dtype,
        )
        if self.uses_token_head_scales:
            self.kv_scale_cache = torch.empty(
                KV_COMPONENT_COUNT,
                text_config.num_hidden_layers,
                self.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                dtype=KV_SCALE_DTYPE,
            )
        else:
            self.kv_scale_cache = None

    def _bind_attention_kv_views(self, expected_layers: int) -> None:
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        if len(attention_modules) != expected_layers:
            raise RuntimeError(
                f"KV cache layer count mismatch: assigned={len(attention_modules)}, "
                f"expected={expected_layers}"
            )
        for layer_id, module in enumerate(attention_modules):
            module.layer_idx = layer_id
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            module.k_scale_cache = (
                None if self.kv_scale_cache is None else self.kv_scale_cache[0, layer_id]
            )
            module.v_scale_cache = (
                None if self.kv_scale_cache is None else self.kv_scale_cache[1, layer_id]
            )

    def _allocate_cpu_kv_storage(self, text_config, num_kv_heads: int, head_dim: int) -> None:
        self.num_cpu_blocks = int(self.num_kvcache_blocks * self.config.cpu_kv_cache_ratio)
        if self.num_cpu_blocks > 0:
            self.cpu_kv_cache = torch.empty(
                KV_COMPONENT_COUNT,
                text_config.num_hidden_layers,
                self.num_cpu_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=self.kv_cache_dtype,
                device="cpu",
                pin_memory=True,  # pinned memory: 加速 GPU↔CPU 传输
            )
            if self.uses_token_head_scales:
                self.cpu_kv_scale_cache = torch.empty(
                    KV_COMPONENT_COUNT,
                    text_config.num_hidden_layers,
                    self.num_cpu_blocks,
                    self.block_size,
                    num_kv_heads,
                    dtype=KV_SCALE_DTYPE,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                self.cpu_kv_scale_cache = None
        else:
            self.cpu_kv_cache = None
            self.cpu_kv_scale_cache = None

    # ═══════════════════════════════════════════════════════════
    # 准备阶段: 把 Sequence 列表转成 GPU tensor
    # ═══════════════════════════════════════════════════════════

    # ── copy_kv_blocks: CoW (Copy-on-Write) GPU 端 KV 数据复制 ──
    # 当多个序列共享同一个 KV Cache block 时,
    # 写入前需要先复制一份, 避免污染其他序列的数据
    #
    # C++ 类比: memcpy(new_page, old_page, PAGE_SIZE)
    # CUDA 类比: cudaMemcpyDeviceToDevice
    def _bound_gpu_scale_cache(self) -> torch.Tensor | None:
        """Return the optional scale cache after checking runner ownership."""

        scale_cache = getattr(self, "kv_scale_cache", None)
        expected = getattr(
            self,
            "uses_token_head_scales",
            scale_cache is not None,
        )
        if expected != (scale_cache is not None):
            raise RuntimeError(
                "runner scaled-KV configuration/cache ownership mismatch: "
                f"expected={expected}, cache_bound={scale_cache is not None}"
            )
        return scale_cache

    def copy_kv_blocks(self, cow_pairs: list[tuple[int, int]]):
        """Copy complete source KV blocks into private destination blocks."""

        scale_cache = self._bound_gpu_scale_cache()
        for src_block_id, dst_block_id in cow_pairs:
            self.kv_cache[:, :, dst_block_id].copy_(self.kv_cache[:, :, src_block_id])
            if scale_cache is not None:
                scale_cache[:, :, dst_block_id].copy_(scale_cache[:, :, src_block_id])

    def copy_kv_block_prefixes(
        self,
        copies: list[tuple[int, int, int]],
    ) -> None:
        """Copy only valid rows from shared compact-prefix tail blocks."""

        scale_cache = self._bound_gpu_scale_cache()
        block_size = self.kv_cache.shape[3]
        for src_block_id, dst_block_id, num_rows in copies:
            if num_rows <= 0 or num_rows > block_size:
                raise ValueError(
                    f"KV block-prefix rows must be in [1, {block_size}], got {num_rows}"
                )
            self.kv_cache[:, :, dst_block_id, :num_rows].copy_(
                self.kv_cache[:, :, src_block_id, :num_rows]
            )
            if scale_cache is not None:
                scale_cache[:, :, dst_block_id, :num_rows].copy_(
                    scale_cache[:, :, src_block_id, :num_rows]
                )

    def compact_kv_cache(self, plans: list[KVCompactionPlan]) -> None:
        """按已验证 plan 把 retained prompt KV 移到页表前部。

        kv_cache: [2, layers, blocks, block_size, kv_heads, head_dim]
        每条 plan先 gather到临时 tensor，再写 destination slots，避免重叠 copy
        覆盖尚未读取的 source。Block/free-list 提交由主进程 BlockManager 完成。
        """

        if not plans:
            return
        scale_cache = self._bound_gpu_scale_cache()
        seen_sequence_ids: set[int] = set()
        flat_cache = self.kv_cache.reshape(
            self.kv_cache.shape[0],
            self.kv_cache.shape[1],
            -1,
            self.kv_cache.shape[-2],
            self.kv_cache.shape[-1],
        )
        flat_scale_cache = (
            None
            if scale_cache is None
            else scale_cache.reshape(
                scale_cache.shape[0],
                scale_cache.shape[1],
                -1,
                scale_cache.shape[-1],
            )
        )
        # Validate the complete plan set before mutating either payload or
        # scales.  A late duplicate/dtype error must not leave a half-committed
        # physical layout.
        for plan in plans:
            plan.validate(block_size=self.block_size)
            if plan.kv_dtype != str(self.kv_cache.dtype):
                raise RuntimeError(
                    "compaction plan/cache dtype mismatch: "
                    f"plan={plan.kv_dtype}, cache={self.kv_cache.dtype}"
                )
            if plan.seq_id in seen_sequence_ids:
                raise RuntimeError(f"duplicate compaction plan for seq_id={plan.seq_id}")
            seen_sequence_ids.add(plan.seq_id)
        for plan in plans:
            source_slots = torch.tensor(
                plan.source_slots,
                dtype=torch.long,
                device=self.kv_cache.device,
            )
            destination_slots = torch.tensor(
                plan.destination_slots,
                dtype=torch.long,
                device=self.kv_cache.device,
            )
            compact_kv_slots(flat_cache, source_slots, destination_slots)
            if flat_scale_cache is not None:
                compact_kv_slots(
                    flat_scale_cache,
                    source_slots,
                    destination_slots,
                )
        if self.world_size > 1:
            self._tp_barrier()

    # ── swap_blocks: GPU ↔ CPU KV Cache 数据搬运 ──
    # C++ 类比: cudaMemcpyAsync(dst, src, size, direction, stream)
    #   swap_out: DeviceToHost (GPU→CPU)
    #   swap_in:  HostToDevice (CPU→GPU)
    def swap_blocks(self, swap_map: list[tuple[int, int]], direction: str):
        """搬运 KV Cache blocks
        direction='out': GPU→CPU (swap_map = [(gpu_id, cpu_id), ...])
        direction='in':  CPU→GPU (swap_map = [(cpu_id, gpu_id), ...])
        """
        if not swap_map:
            return
        if direction not in ("out", "in"):
            raise ValueError(f"swap direction must be 'out' or 'in', got {direction!r}")
        if self.cpu_kv_cache is None:
            raise RuntimeError("KV swap requested without an allocated CPU payload cache")
        scale_cache = self._bound_gpu_scale_cache()
        cpu_scale_cache = getattr(self, "cpu_kv_scale_cache", None)
        if (scale_cache is None) != (cpu_scale_cache is None):
            raise RuntimeError("GPU/CPU scale caches must either both exist or both be absent")
        for src_id, dst_id in swap_map:
            if direction == "out":
                # GPU → CPU (异步, non_blocking=True)
                self.cpu_kv_cache[:, :, dst_id].copy_(
                    self.kv_cache[:, :, src_id], non_blocking=True
                )
                if scale_cache is not None:
                    cpu_scale_cache[:, :, dst_id].copy_(
                        scale_cache[:, :, src_id], non_blocking=True
                    )
            else:
                # CPU → GPU (异步)
                self.kv_cache[:, :, dst_id].copy_(
                    self.cpu_kv_cache[:, :, src_id], non_blocking=True
                )
                if scale_cache is not None:
                    scale_cache[:, :, dst_id].copy_(
                        cpu_scale_cache[:, :, src_id], non_blocking=True
                    )
        # 确保搬运完成后再继续 (类似 cudaStreamSynchronize)
        torch.cuda.synchronize()

    def _get_input_preparer(self) -> ModelInputPreparer:
        """Return the composed preparer, including lightweight test runners."""

        preparer = getattr(self, "input_preparer", None)
        if (
            preparer is None
            or preparer.config is not self.config
            or preparer.block_size != self.block_size
        ):
            preparer = ModelInputPreparer(self.config, block_size=self.block_size)
            self.input_preparer = preparer
        return preparer

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Compatibility adapter for the dedicated input-preparation boundary."""

        return self._get_input_preparer().prepare_block_tables(seqs)

    def _prepare_prefill_batch(
        self,
        seqs: list[Sequence],
        *,
        prefill_slices: tuple[PrefillSlice, ...] | None = None,
    ) -> PreparedModelInputs:
        """Compatibility delegate for the dedicated input-preparation boundary."""

        return self._get_input_preparer().prepare_prefill(
            seqs,
            prefill_slices=prefill_slices,
        )

    def prepare_prefill(
        self,
        seqs: list[Sequence],
        *,
        prefill_slices: tuple[PrefillSlice, ...] | None = None,
    ) -> DeviceModelInputs:
        """Compatibility adapter; production execution consumes the pair directly."""

        prepared = self._prepare_prefill_batch(
            seqs,
            prefill_slices=prefill_slices,
        )
        install_context(prepared.attention_context)
        return prepared.model_inputs

    def _prepare_decode_batch(self, seqs: list[Sequence]) -> PreparedModelInputs:
        """Compatibility delegate for the dedicated input-preparation boundary."""

        return self._get_input_preparer().prepare_decode(seqs)

    def prepare_decode(self, seqs: list[Sequence]) -> DeviceModelInputs:
        """Compatibility adapter; production execution consumes the pair directly."""

        prepared = self._prepare_decode_batch(seqs)
        install_context(prepared.attention_context)
        return prepared.model_inputs

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数: 收集每条序列的温度"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )
        return temperatures

    @staticmethod
    def resolve_sampling_mode(seqs: list[Sequence]) -> str:
        """Resolve the batch sampling branch from host-owned request state."""

        greedy = [seq.temperature <= SAMPLING_NUMERICAL_EPSILON for seq in seqs]
        if all(greedy):
            return "greedy"
        if not any(greedy):
            return "random"
        return "mixed"

    @staticmethod
    def _as_mrope_decode_positions(position_ids: torch.Tensor) -> torch.Tensor:
        """把 decode position ids 规范为 `[3, batch]`。

        text-only decode 原本是一维 `[batch]`；CUDA Graph replay 统一使用
        `[3, max_bs]` 占位，三轴同值与文本 M-RoPE 语义等价。
        """

        if position_ids.ndim == 1:
            return position_ids.view(1, -1).expand(MROPE_AXIS_COUNT, -1)
        if (
            position_ids.ndim == MROPE_DECODE_POSITION_RANK
            and position_ids.shape[0] == MROPE_AXIS_COUNT
        ):
            return position_ids
        raise ValueError(
            f"decode position_ids must be [batch] or [3, batch], got {list(position_ids.shape)}"
        )

    @staticmethod
    def _cudagraph_batch_sizes(max_bs: int) -> list[int]:
        """生成 CUDA Graph decode batch 档位，确保覆盖 `max_bs`。

        小 batch 对 BF16 GEMM shape 很敏感，因此 1 到 8 每个 shape 都精确
        录制；更大的 batch 使用 16 的步长控制启动 capture 成本。当
        `max_num_seqs` 是 9、17 等非标准档位时，额外录制最后的 `max_bs`，
        避免 replay 查找失败。
        """

        if max_bs < 1:
            raise ValueError(f"max_bs must be >= 1, got {max_bs}")
        graph_bs = list(
            range(
                1,
                min(max_bs, CUDA_GRAPH_EXACT_BATCH_LIMIT) + 1,
            )
        )
        graph_bs += list(
            range(
                CUDA_GRAPH_BATCH_BUCKET_STRIDE,
                max_bs + 1,
                CUDA_GRAPH_BATCH_BUCKET_STRIDE,
            )
        )
        if not graph_bs or graph_bs[-1] != max_bs:
            graph_bs.append(max_bs)
        return sorted(set(graph_bs))

    def cudagraph_metadata(self, requested_batch_size: int) -> dict[str, object]:
        """返回可审计的 CUDA Graph capture/replay 配置。"""

        if requested_batch_size < 1:
            raise ValueError(f"requested_batch_size must be >= 1, got {requested_batch_size}")
        if self.enforce_eager:
            return {
                "enabled": False,
                "capture_scope": "none",
                "capture_ms": 0.0,
                "batch_sizes": [],
                "requested_batch_size": requested_batch_size,
                "selected_batch_size": requested_batch_size,
                "batch_padding": 0,
            }
        selected_batch_size = next(
            batch_size for batch_size in self.graph_bs if batch_size >= requested_batch_size
        )
        return {
            "enabled": True,
            "capture_scope": "decode_model_forward_logits_greedy",
            "capture_ms": self.cudagraph_capture_ms,
            "batch_sizes": list(self.graph_bs),
            "requested_batch_size": requested_batch_size,
            "selected_batch_size": selected_batch_size,
            "batch_padding": selected_batch_size - requested_batch_size,
        }

    # ═══════════════════════════════════════════════════════════
    # 执行阶段: run_model + run (对外入口)
    # ═══════════════════════════════════════════════════════════

    @torch.inference_mode()
    def run_model_eager(
        self,
        model_inputs: DeviceModelInputs,
        *,
        is_prefill: bool,
    ):
        """Execute the eager/compiled-attention tensor path."""

        with profile_region(
            "runner.model.forward",
            metadata={
                "backend": (
                    "torch_compile_attention"
                    if (not is_prefill and self.config.decode_compile_region == "attention")
                    else "eager"
                )
            },
        ):
            if not is_prefill and self.decode_compile_first_call_pending:
                torch.cuda.synchronize()
                compile_start = perf_counter()
                hidden_states = self._forward_model(model_inputs)
                torch.cuda.synchronize()
                self.decode_compile_first_call_ms = (perf_counter() - compile_start) * 1000.0
                self.decode_compile_first_call_pending = False
            else:
                hidden_states = self._forward_model(model_inputs)
        with profile_region("runner.model.compute_logits"):
            return self.model.compute_logits(hidden_states)

    @torch.inference_mode()
    def run_model_cudagraph(
        self,
        model_inputs: DeviceModelInputs,
        *,
        return_greedy_tokens: bool = False,
    ):
        """Replay the captured decode graph; no eager fallback is allowed."""

        input_ids = model_inputs.input_ids
        context = get_context()
        required_tensors = {
            "slot_mapping": context.slot_mapping,
            "context_lens": context.context_lens,
            "decode_max_context_len": context.decode_max_context_len,
            "block_tables": context.block_tables,
        }
        missing = [name for name, value in required_tensors.items() if value is None]
        if missing:
            raise RuntimeError(
                f"CUDA Graph decode is missing context tensors: {', '.join(missing)}"
            )
        batch_size = input_ids.size(0)
        try:
            captured_batch_size = next(size for size in self.graph_bs if size >= batch_size)
        except StopIteration as exc:
            raise RuntimeError(
                f"no CUDA Graph bucket covers decode batch size {batch_size}"
            ) from exc
        graph = (
            self.greedy_graphs[captured_batch_size]
            if return_greedy_tokens
            else self.graphs[captured_batch_size]
        )
        self.last_cudagraph_actual_batch_size = batch_size
        self.last_cudagraph_replay_batch_size = captured_batch_size
        graph_vars = self.graph_vars[captured_batch_size]
        block_table_width = context.block_tables.size(1)
        if block_table_width > graph_vars["block_tables"].size(1):
            raise RuntimeError(
                "decode block table exceeds captured CUDA Graph capacity: "
                f"actual={block_table_width} "
                f"captured={graph_vars['block_tables'].size(1)}"
            )

        with profile_region("runner.cudagraph.copy_inputs"):
            packed_model_inputs = model_inputs.packed_decode_inputs
            packed_decode_metadata = context.packed_decode_metadata
            use_packed_staging = (
                batch_size == captured_batch_size
                and packed_model_inputs is not None
                and packed_decode_metadata is not None
                and packed_model_inputs.numel() == graph_vars["packed_model_inputs"].numel()
                and packed_decode_metadata.numel() <= graph_vars["packed_decode_metadata"].numel()
                and (batch_size == 1 or block_table_width == graph_vars["block_tables"].size(1))
            )
            if use_packed_staging:
                if packed_model_inputs is not graph_vars["host_packed_model_inputs"]:
                    graph_vars["host_packed_model_inputs"].copy_(packed_model_inputs)
                if packed_decode_metadata is not graph_vars["host_packed_decode_metadata"]:
                    graph_vars["host_packed_decode_metadata"][
                        : packed_decode_metadata.numel()
                    ].copy_(packed_decode_metadata)
            else:
                graph_vars["host_input_ids"][:batch_size] = input_ids
                graph_vars["host_positions"][:, :batch_size] = self._as_mrope_decode_positions(
                    model_inputs.position_ids
                )
                graph_vars["host_slot_mapping"].fill_(-1)
                graph_vars["host_slot_mapping"][:batch_size] = context.slot_mapping
                graph_vars["host_context_lens"].zero_()
                graph_vars["host_context_lens"][:batch_size] = context.context_lens
                graph_vars["host_decode_max_context_len"].copy_(context.decode_max_context_len)
                graph_vars["host_block_tables"].fill_(-1)
                graph_vars["host_block_tables"][
                    :batch_size,
                    :block_table_width,
                ] = context.block_tables

        with profile_region(
            "runner.cudagraph.replay",
            metadata={
                "actual_batch_size": batch_size,
                "captured_batch_size": captured_batch_size,
            },
        ):
            graph.replay()
        if return_greedy_tokens:
            return self.graph_greedy_tokens[captured_batch_size][:batch_size]
        captured_logits = self.graph_logits[captured_batch_size]
        if captured_logits is None:
            if self.rank == 0:
                raise RuntimeError("rank 0 CUDA Graph replay did not produce logits")
            return None
        return captured_logits[:batch_size]

    @torch.inference_mode()
    def run_model(self, model_inputs: DeviceModelInputs, is_prefill: bool):
        """Compatibility adapter for direct legacy runner callers."""

        compression_requires_eager = not compression_supports_cuda_graph(
            get_context().compression_metadata
        )
        if is_prefill or self.enforce_eager or compression_requires_eager:
            return self.run_model_eager(
                model_inputs,
                is_prefill=is_prefill,
            )
        return self.run_model_cudagraph(model_inputs)

    def run_plan(self, plan: BatchPlan) -> ExecutionResult:
        """Execute one immutable host plan through a tensor-only DeviceBatch."""

        self._validate_execution_plan(plan)
        return self._run_plan(plan)

    def _run_plan(self, plan: BatchPlan) -> ExecutionResult:
        """Execute a validated plan on the caller-selected CUDA stream."""

        backend = self._get_execution_backend()
        device_batch = backend.prepare(plan)
        execution = backend.execute(device_batch)
        token_ids = list(execution.token_ids)
        if plan.is_prefill:
            self._commit_prefill_execution(plan, device_batch, token_ids)
        return ExecutionResult(token_ids=tuple(token_ids))

    def _build_cooperative_vision_payloads(
        self,
        inputs: DeviceModelInputs,
    ) -> list[_CooperativeVisionPayload]:
        """Materialize attention-safe vision microbatch plans without encoding."""

        visual_model = self.model.model
        patch_limit = visual_model.vision_encoder_microbatch_patches
        payloads = []
        for token_id, pixel_values, grid_thw in (
            (
                visual_model.image_token_id,
                inputs.pixel_values,
                inputs.image_grid_thw,
            ),
            (
                visual_model.video_token_id,
                inputs.pixel_values_videos,
                inputs.video_grid_thw,
            ),
        ):
            if pixel_values is None:
                if grid_thw is not None:
                    raise RuntimeError("visual grid exists without a payload")
                continue
            if grid_thw is None:
                raise RuntimeError("visual payload exists without a grid")
            microbatches = visual_model._plan_visual_microbatches(
                grid_thw,
                patch_limit=patch_limit,
            )
            expected_patches = microbatches[-1][1] if microbatches else 0
            if expected_patches != int(pixel_values.shape[0]):
                raise ValueError(
                    "visual grid/payload patch count mismatch: "
                    f"grid={expected_patches} "
                    f"payload={int(pixel_values.shape[0])}"
                )
            payloads.append(
                _CooperativeVisionPayload(
                    token_id=token_id,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    microbatches=microbatches,
                    main_parts=[],
                )
            )
        return payloads

    def _begin_pending_language_forward(
        self,
        pending: _PendingPrefillExecution,
    ) -> None:
        """Install completed visual outputs and create the language state."""

        inputs = pending.device_batch.model_inputs
        visual_embeds = inputs.visual_embeds
        deepstack_visual_embeds = inputs.deepstack_visual_embeds
        if pending.vision_payloads:
            if visual_embeds is not None:
                raise RuntimeError("cooperative prefill cannot mix raw and cached vision inputs")
            modality_outputs = []
            for payload in pending.vision_payloads:
                if (
                    payload.next_microbatch != len(payload.microbatches)
                    or not payload.main_parts
                    or payload.deepstack_parts is None
                ):
                    raise RuntimeError("language forward started before vision completed")
                modality_outputs.append(
                    (
                        payload.token_id,
                        torch.cat(payload.main_parts, dim=0),
                        tuple(torch.cat(parts, dim=0) for parts in payload.deepstack_parts),
                    )
                )

            input_ids = inputs.input_ids
            visual_token_mask = torch.zeros_like(
                input_ids,
                dtype=torch.bool,
            )
            for token_id, _, _ in modality_outputs:
                visual_token_mask |= input_ids == token_id
            visual_token_ids = input_ids[visual_token_mask]
            hidden_size = int(modality_outputs[0][1].shape[-1])
            visual_embeds = modality_outputs[0][1].new_empty(
                (visual_token_ids.numel(), hidden_size)
            )
            deepstack_depth = len(modality_outputs[0][2])
            deepstack_visual_embeds = tuple(
                visual_embeds.new_empty(visual_embeds.shape) for _ in range(deepstack_depth)
            )
            for token_id, main, deepstack in modality_outputs:
                if len(deepstack) != deepstack_depth:
                    raise RuntimeError("vision modalities returned inconsistent DeepStack")
                positions = torch.nonzero(
                    visual_token_ids == token_id,
                    as_tuple=False,
                ).flatten()
                if positions.numel() != main.shape[0]:
                    raise ValueError(
                        "visual output/token count mismatch: "
                        f"token_id={token_id} output={main.shape[0]} "
                        f"positions={positions.numel()}"
                    )
                visual_embeds[positions] = main
                for target, value in zip(
                    deepstack_visual_embeds,
                    deepstack,
                    strict=True,
                ):
                    target[positions] = value

        with use_context(pending.device_batch.attention_context):
            (
                inputs_embeds,
                visual_pos_masks,
                resolved_deepstack,
            ) = self.model.model.prepare_language_inputs(
                input_ids=inputs.input_ids,
                visual_embeds=visual_embeds,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            pending.model_state = self.model.model.language_model.begin_cooperative_forward(
                inputs_embeds=inputs_embeds,
                position_ids=inputs.position_ids,
                attention_mask=None,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=resolved_deepstack,
            )

    def _advance_pending_vision(
        self,
        pending: _PendingPrefillExecution,
        *,
        max_blocks: int,
    ) -> bool:
        """Execute one vision block quantum across the active microbatch."""

        if pending.next_vision_payload >= len(pending.vision_payloads):
            return True
        payload = pending.vision_payloads[pending.next_vision_payload]
        if payload.active_state is None:
            if payload.next_microbatch >= len(payload.microbatches):
                raise RuntimeError("completed vision payload remained active")
            start, end, rows = payload.microbatches[payload.next_microbatch]
            chunk_grid = torch.tensor(
                rows,
                dtype=payload.grid_thw.dtype,
                device=payload.grid_thw.device,
            )
            with profile_region(
                "runner.prefill.vision_begin",
                metadata={
                    "microbatch": payload.next_microbatch,
                    "patches": end - start,
                },
            ):
                payload.active_state = self.model.model.visual.begin_cooperative_forward(
                    payload.pixel_values[start:end],
                    chunk_grid,
                )

        state = payload.active_state
        with profile_region(
            "runner.prefill.vision_block_quantum",
            metadata={
                "blocks": max_blocks,
                "microbatch": payload.next_microbatch,
                "next_block": state.next_block,
                "patches": int(state.hidden_states.shape[0]),
            },
        ):
            complete = self.model.model.visual.advance_cooperative_forward(
                state,
                max_blocks=max_blocks,
            )
        if not complete:
            return False

        main, deepstack = self.model.model.visual.finish_cooperative_forward(state)
        payload.main_parts.append(main)
        if payload.deepstack_parts is None:
            payload.deepstack_parts = [[] for _ in deepstack]
        if len(deepstack) != len(payload.deepstack_parts):
            raise RuntimeError("vision microbatches returned inconsistent DeepStack")
        for parts, value in zip(
            payload.deepstack_parts,
            deepstack,
            strict=True,
        ):
            parts.append(value)
        payload.active_state = None
        payload.next_microbatch += 1
        if payload.next_microbatch == len(payload.microbatches):
            pending.next_vision_payload += 1
        return pending.next_vision_payload == len(pending.vision_payloads)

    @torch.inference_mode()
    def begin_prefill_plan(self, plan: BatchPlan) -> _PendingPrefillExecution:
        """Prepare a prefill paused before its first vision or language block."""

        if not self.config.enable_cooperative_prefill:
            raise RuntimeError("cooperative prefill is not enabled")
        if self.world_size != 1:
            raise RuntimeError("cooperative prefill currently supports TP1 only")
        if not self.is_vl_model:
            raise RuntimeError("cooperative prefill currently requires Qwen3-VL")
        if plan.phase is not BatchPhase.PREFILL:
            raise ValueError("begin_prefill_plan requires a prefill plan")
        self._validate_execution_plan(plan)
        backend = self._get_execution_backend()
        device_batch = backend.prepare(plan)
        inputs = device_batch.model_inputs
        vision_payloads = self._build_cooperative_vision_payloads(inputs)
        pending = _PendingPrefillExecution(
            plan=plan,
            device_batch=device_batch,
            model_state=None,
            vision_payloads=vision_payloads,
        )
        if not vision_payloads:
            self._begin_pending_language_forward(pending)
        return pending

    @torch.inference_mode()
    def advance_prefill_plan(
        self,
        pending: _PendingPrefillExecution,
        max_layers: int,
        *,
        max_vision_blocks: int | None = None,
    ) -> bool:
        """Execute one bounded vision-block or language-layer quantum."""

        if not isinstance(pending, _PendingPrefillExecution):
            raise TypeError("pending prefill handle is invalid")
        if pending.sampled_tokens is not None:
            return True
        if pending.model_state is None:
            if max_vision_blocks is None:
                max_vision_blocks = self.config.cooperative_prefill_vision_block_quantum
            vision_complete = self._advance_pending_vision(
                pending,
                max_blocks=max_vision_blocks,
            )
            if not vision_complete:
                return False
            with profile_region("runner.prefill.language_begin"):
                self._begin_pending_language_forward(pending)
            return False
        language_model = self.model.model.language_model
        with use_context(pending.device_batch.attention_context):
            with profile_region(
                "runner.prefill.language_layer_quantum",
                metadata={
                    "layers": max_layers,
                    "next_layer": pending.model_state.next_layer,
                    "tokens": int(pending.model_state.hidden_states.shape[0]),
                },
            ):
                complete = language_model.advance_cooperative_forward(
                    pending.model_state,
                    max_layers=max_layers,
                )
            if complete:
                hidden_states = language_model.finish_cooperative_forward(
                    pending.model_state,
                )
                logits = self.model.compute_logits(hidden_states)
                pending.sampled_tokens = self.sampler(
                    logits,
                    pending.device_batch.temperatures,
                    sampling_mode=pending.device_batch.sampling_mode,
                )
        if not complete:
            return False
        return True

    def finish_prefill_plan(
        self,
        pending: _PendingPrefillExecution,
    ) -> ExecutionResult:
        """Finish all remaining layers, sample and commit the prefill."""

        if not isinstance(pending, _PendingPrefillExecution):
            raise TypeError("pending prefill handle is invalid")
        max_quantum = max(
            len(self.model.model.visual.blocks),
            len(self.model.model.language_model.layers),
        )
        while pending.sampled_tokens is None:
            self.advance_prefill_plan(
                pending,
                max_quantum,
                max_vision_blocks=max_quantum,
            )
        token_ids = list(pending.sampled_tokens.tolist())
        self._commit_prefill_execution(
            pending.plan,
            pending.device_batch,
            token_ids,
        )
        return ExecutionResult(token_ids=tuple(token_ids))

    def prepare_single_greedy_decode_cudagraph(
        self,
        plan: BatchPlan,
    ) -> DeviceBatch | None:
        """Fill persistent Graph host buffers for the latency-critical B1 path.

        General decode preparation intentionally materializes immutable staging
        tensors.  For CUDA-Graph-safe KV state and greedy batch-one replay, each
        TP rank already owns equivalent stable pinned Graph buffers and
        ``sampled_tokens.tolist()`` makes each step sequential.  Reusing those
        buffers avoids two small tensor allocations, a second host copy, and
        repeated Context/DeviceBatch setup.
        """

        if (
            plan.phase is not BatchPhase.DECODE
            or plan.batch_size != 1
            or not compression_mode_supports_cuda_graph(self.config.compression_mode)
            or getattr(self.config, "enable_visual_pruning_shadow", False)
            or is_trace_enabled()
            or 1 not in getattr(self, "graph_vars", {})
        ):
            return None
        seq = plan.sequences[0]
        if seq.temperature > SAMPLING_NUMERICAL_EPSILON:
            return None
        if not seq.block_table or seq.physical_last_block_num_tokens <= 0:
            raise RuntimeError(f"batch-one Graph decode has no writable KV slot: {seq.seq_id}")

        logical_position = len(seq) - 1
        if seq.rope_delta is not None:
            logical_position += int(seq.rope_delta.item())
        physical_context_len = seq.physical_kv_len
        state = SingleGreedyDecodeState(
            sequence_id=seq.seq_id,
            last_token=seq.last_token,
            logical_position=logical_position,
            kv_slot=(
                seq.block_table[-1] * self.block_size + seq.physical_last_block_num_tokens - 1
            ),
            physical_context_len=physical_context_len,
            logical_context_len=len(seq),
            block_table=tuple(seq.block_table),
        )
        cache = getattr(self, "_single_greedy_decode_batch_cache", None)
        if cache is None or cache[0] != seq.seq_id:
            compression_metadata = build_compression_metadata(
                self.config,
                [seq],
                is_prefill=False,
            )
        else:
            compression_metadata = None
        return self._prepare_single_greedy_decode_cudagraph_state(
            state,
            compression_metadata=compression_metadata,
        )

    def single_greedy_decode_state(
        self,
        plan: BatchPlan,
    ) -> SingleGreedyDecodeState | None:
        """Build the minimal compression-off payload needed by TP workers."""

        if (
            plan.phase is not BatchPhase.DECODE
            or plan.batch_size != 1
            or self.config.compression_mode != COMPRESSION_OFF
            or getattr(self.config, "enable_visual_pruning_shadow", False)
            or is_trace_enabled()
            or 1 not in getattr(self, "graph_vars", {})
        ):
            return None
        seq = plan.sequences[0]
        if seq.temperature > SAMPLING_NUMERICAL_EPSILON:
            return None
        if not seq.block_table or seq.physical_last_block_num_tokens <= 0:
            raise RuntimeError(f"batch-one Graph decode has no writable KV slot: {seq.seq_id}")
        logical_position = len(seq) - 1
        if seq.rope_delta is not None:
            logical_position += int(seq.rope_delta.item())
        return SingleGreedyDecodeState(
            sequence_id=seq.seq_id,
            last_token=seq.last_token,
            logical_position=logical_position,
            kv_slot=(
                seq.block_table[-1] * self.block_size + seq.physical_last_block_num_tokens - 1
            ),
            physical_context_len=seq.physical_kv_len,
            logical_context_len=len(seq),
            block_table=tuple(seq.block_table),
        )

    def _prepare_single_greedy_decode_cudagraph_state(
        self,
        state: SingleGreedyDecodeState,
        *,
        compression_metadata: CompressionMetadata | None,
    ) -> DeviceBatch:
        """Fill persistent Graph host buffers from one compact decode state."""

        graph_vars = self.graph_vars[1]
        host_model_values = graph_vars["host_packed_model_inputs_numpy"]
        host_metadata_values = graph_vars["host_packed_decode_metadata_numpy"]
        host_model_values[0] = state.last_token
        host_model_values[1 : 1 + MROPE_AXIS_COUNT] = state.logical_position
        host_metadata_values[0] = state.kv_slot
        host_metadata_values[1] = state.physical_context_len
        host_metadata_values[2] = state.logical_context_len
        host_metadata_values[3] = state.physical_context_len
        cache = getattr(self, "_single_greedy_decode_batch_cache", None)
        if cache is None or cache[0] != state.sequence_id:
            host_metadata_values[4:] = -1
            host_packed_model_inputs = graph_vars["host_packed_model_inputs"]
            host_packed_decode_metadata = graph_vars["host_packed_decode_metadata"]
            scale_cache = self._bound_gpu_scale_cache()
            context = Context(
                is_prefill=False,
                slot_mapping=graph_vars["host_slot_mapping"],
                context_lens=graph_vars["host_context_lens"],
                logical_context_lens=host_packed_decode_metadata[2:3],
                block_tables=graph_vars["host_block_tables"],
                decode_max_context_len=graph_vars["host_decode_max_context_len"],
                packed_decode_metadata=host_packed_decode_metadata,
                paged_decode_block_n=self.config.paged_decode_block_n,
                compression_metadata=compression_metadata,
            )
            device_batch = DeviceBatch(
                phase=BatchPhase.DECODE,
                sequence_ids=(state.sequence_id,),
                scheduled_token_counts=(1,),
                model_inputs=DeviceModelInputs(
                    input_ids=graph_vars["host_input_ids"],
                    position_ids=graph_vars["host_positions"],
                    packed_decode_inputs=host_packed_model_inputs,
                ),
                attention_context=context,
                temperatures=None,
                execution_bucket=1,
                sampling_mode="greedy",
                kv_scale_views=(() if scale_cache is None else (scale_cache[0], scale_cache[1])),
            )
            cache = (state.sequence_id, device_batch)
            self._single_greedy_decode_batch_cache = cache
        host_metadata_values[4 : 4 + len(state.block_table)] = state.block_table
        return cache[1]

    def execute_single_greedy_decode_cudagraph(
        self,
        plan: BatchPlan,
    ) -> ExecutionResult | None:
        """Execute the persistent batch-one Graph path without generic wrappers."""

        device_batch = self.prepare_single_greedy_decode_cudagraph(plan)
        if device_batch is None:
            return None
        with use_context(device_batch.attention_context):
            with profile_region("runner.run_model"):
                sampled_tokens = self.run_model_cudagraph(
                    device_batch.model_inputs,
                    return_greedy_tokens=True,
                )
            with profile_region("runner.sampler"):
                token_ids = tuple(sampled_tokens.tolist())
        return ExecutionResult(token_ids=token_ids)

    def execute_single_greedy_decode_cudagraph_state(
        self,
        state: SingleGreedyDecodeState,
    ) -> ExecutionResult:
        """Execute the compact TP2 batch-one Graph control payload."""

        if self.config.compression_mode != COMPRESSION_OFF:
            raise RuntimeError("compact TP2 Graph decode requires compression_mode='off'")
        device_batch = self._prepare_single_greedy_decode_cudagraph_state(
            state,
            compression_metadata=None,
        )
        with use_context(device_batch.attention_context):
            with profile_region("runner.run_model"):
                sampled_tokens = self.run_model_cudagraph(
                    device_batch.model_inputs,
                    return_greedy_tokens=True,
                )
            with profile_region("runner.sampler"):
                token_ids = tuple(sampled_tokens.tolist())
        return ExecutionResult(token_ids=token_ids)

    def _validate_execution_plan(self, plan: BatchPlan) -> None:
        if not isinstance(plan, BatchPlan):
            raise TypeError(f"run_plan requires BatchPlan, got {type(plan).__name__}")
        if not plan.is_prefill or not self.config.enable_chunked_prefill:
            return
        if any(not seq.block_table for seq in plan.sequences):
            return
        max_chunk = self.config.max_chunk_size
        for seq, prefill_slice in zip(plan.sequences, plan.prefill_slices, strict=False):
            if prefill_slice.num_tokens > max_chunk:
                raise ValueError(
                    "invalid scheduled prefill chunk: "
                    f"seq={seq.seq_id} chunk={prefill_slice.num_tokens} "
                    f"max_chunk={max_chunk}"
                )

    def _get_execution_backend(self):
        backend = getattr(self, "execution_backend", None)
        if backend is None:
            backend = create_execution_backend(self)
            self.execution_backend = backend
        return backend

    def _commit_prefill_execution(
        self,
        plan: BatchPlan,
        device_batch: DeviceBatch,
        token_ids: list[int | None],
    ) -> None:
        seqs = list(plan.sequences)
        scorer = device_batch.attention_context.visual_pruning_scorer
        if scorer is not None:
            with profile_region("runner.visual_prune.finalize_attention_scores"):
                finalize_attention_pruning_decisions(
                    seqs,
                    build_visual_pruning_config(self.config),
                    scorer,
                )
        for index, (seq, prefill_slice) in enumerate(zip(seqs, plan.prefill_slices, strict=False)):
            if (
                seq.kv_layout is not None
                and seq.kv_layout.logical_context_len < seq.num_prompt_tokens
            ):
                seq.kv_layout.append_prefill_tokens(prefill_slice.num_tokens)
            seq.num_computed_tokens = prefill_slice.token_end
            if not seq.is_prefill_finished:
                token_ids[index] = None

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        scheduled_token_counts: list[int] | None = None,
    ) -> list[int | None]:
        """One-cycle compatibility adapter for direct legacy runner calls."""

        if scheduled_token_counts is None:
            is_warmup = any(not seq.block_table for seq in seqs)
            chunked_active = is_prefill and self.config.enable_chunked_prefill and not is_warmup
            scheduled_token_counts = [
                (
                    max(
                        1,
                        min(
                            seq.num_prompt_tokens
                            - max(
                                seq.num_computed_tokens,
                                seq.num_cached_tokens,
                            ),
                            self.config.max_chunk_size,
                        )
                        if chunked_active
                        else seq.num_prompt_tokens
                        - max(
                            seq.num_computed_tokens,
                            seq.num_cached_tokens,
                        ),
                    )
                    if is_prefill
                    else 1
                )
                for seq in seqs
            ]
        plan = BatchPlan(
            phase=BatchPhase.PREFILL if is_prefill else BatchPhase.DECODE,
            sequences=tuple(seqs),
            scheduled_token_counts=tuple(scheduled_token_counts),
            policy_name="legacy_runner_adapter",
        )
        return list(self.run_plan(plan).token_ids)

    # ═══════════════════════════════════════════════════════════
    # CUDA Graph: 预录制 Decode 操作, 消除 CPU 开销
    # ═══════════════════════════════════════════════════════════

    @torch.inference_mode()
    def capture_cudagraph(self):
        """录制不同 batch size 的 CUDA Graph"""
        torch.cuda.synchronize()
        capture_start = perf_counter()
        config = self.config
        hf_config = config.hf_config
        text_config = _text_hf_config(hf_config)
        max_bs = min(
            self.config.max_num_seqs,
            MAX_CUDA_GRAPH_BATCH_SIZE,
        )
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 最多需要多少个 block (向上取整)

        # ── 创建"占位"tensor (graph 录制时绑定这些 tensor 的地址) ──
        outputs = torch.zeros(max_bs, text_config.hidden_size)

        # ── 要录制的 batch size 列表 ──
        self.graph_bs = self._cudagraph_batch_sizes(max_bs)
        # [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, ...]
        # 不是每个 bs 都录, 只录这些 "档位"
        # 大 batch 不在列表里时, 向上找最近的 (如 bs=9 → 用 bs=16 的 graph)

        self.graphs = {}
        self.greedy_graphs = {}
        self.graph_vars = {}
        self.graph_logits = {}
        self.graph_greedy_tokens = {}
        self.graph_pool = None  # 共享 GPU 内存池

        for bs in reversed(self.graph_bs):  # 从大到小录
            greedy_graph = torch.cuda.CUDAGraph()

            packed_model_inputs = torch.zeros(
                bs * (1 + MROPE_AXIS_COUNT),
                dtype=torch.int64,
            )
            input_ids = packed_model_inputs[:bs]
            positions = packed_model_inputs[bs:].view(MROPE_AXIS_COUNT, bs)
            metadata_prefix_size = 3 * bs + 1
            packed_decode_metadata = torch.full(
                (metadata_prefix_size + bs * max_num_blocks,),
                -1,
                dtype=torch.int32,
            )
            host_packed_model_inputs = torch.zeros(
                packed_model_inputs.numel(),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            host_packed_decode_metadata = torch.full(
                (packed_decode_metadata.numel(),),
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            host_packed_model_inputs_numpy = host_packed_model_inputs.numpy()
            host_packed_decode_metadata_numpy = host_packed_decode_metadata.numpy()
            slot_mapping = packed_decode_metadata[:bs]
            context_lens = packed_decode_metadata[bs : 2 * bs]
            decode_max_context_len = packed_decode_metadata[3 * bs]
            block_tables = packed_decode_metadata[metadata_prefix_size:].view(
                bs,
                max_num_blocks,
            )
            host_input_ids = host_packed_model_inputs[:bs]
            host_positions = host_packed_model_inputs[bs:].view(MROPE_AXIS_COUNT, bs)
            host_slot_mapping = host_packed_decode_metadata[:bs]
            host_context_lens = host_packed_decode_metadata[bs : 2 * bs]
            host_decode_max_context_len = host_packed_decode_metadata[3 * bs]
            host_block_tables = host_packed_decode_metadata[metadata_prefix_size:].view(
                bs, max_num_blocks
            )

            # warmup: 先跑一次, 让 CUDA 编译 kernel
            slot_mapping.copy_(torch.arange(bs, dtype=torch.int32, device=slot_mapping.device))
            context_lens.fill_(1)
            decode_max_context_len.fill_(1)
            block_tables.zero_()
            host_slot_mapping.copy_(torch.arange(bs, dtype=torch.int32, device="cpu"))
            host_context_lens.fill_(1)
            host_decode_max_context_len.fill_(1)
            host_block_tables.zero_()
            capture_context = Context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                decode_max_context_len=decode_max_context_len,
                block_tables=block_tables,
                paged_decode_block_n=config.paged_decode_block_n,
            )
            compute_greedy_tokens = getattr(
                self.model,
                "compute_greedy_tokens",
                None,
            )
            with use_context(capture_context):
                packed_model_inputs.copy_(host_packed_model_inputs, non_blocking=True)
                packed_decode_metadata.copy_(
                    host_packed_decode_metadata,
                    non_blocking=True,
                )
                capture_inputs = DeviceModelInputs(input_ids[:bs], positions[:, :bs])
                if self.decode_compile_first_call_pending:
                    torch.cuda.synchronize()
                    compile_start = perf_counter()
                    outputs[:bs] = self._forward_model(capture_inputs)
                    torch.cuda.synchronize()
                    self.decode_compile_first_call_ms = (perf_counter() - compile_start) * 1000.0
                    self.decode_compile_first_call_pending = False
                else:
                    outputs[:bs] = self._forward_model(capture_inputs)
                if compute_greedy_tokens is None:
                    self.model.compute_logits(outputs[:bs]).argmax(dim=-1)
                else:
                    compute_greedy_tokens(outputs[:bs])
                with torch.cuda.graph(greedy_graph, self.graph_pool):
                    packed_model_inputs.copy_(
                        host_packed_model_inputs,
                        non_blocking=True,
                    )
                    packed_decode_metadata.copy_(
                        host_packed_decode_metadata,
                        non_blocking=True,
                    )
                    outputs[:bs] = self._forward_model(
                        DeviceModelInputs(input_ids[:bs], positions[:, :bs])
                    )
                    if compute_greedy_tokens is None:
                        captured_greedy_tokens = self.model.compute_logits(outputs[:bs]).argmax(
                            dim=-1
                        )
                    else:
                        captured_greedy_tokens = compute_greedy_tokens(outputs[:bs])
            # torch.cuda.graph(graph, pool):
            #   pool: 共享内存池, 不同 bs 的 graph 共享 GPU workspace
            #   with 块内的所有 GPU 操作被录制到 graph 里
            #   不会真正执行, 只是记录 "要执行哪些 kernel"

            if self.graph_pool is None:
                self.graph_pool = greedy_graph.pool()  # 第一个 graph 创建池

            logits_graph = torch.cuda.CUDAGraph()
            with use_context(capture_context):
                packed_model_inputs.copy_(host_packed_model_inputs, non_blocking=True)
                packed_decode_metadata.copy_(
                    host_packed_decode_metadata,
                    non_blocking=True,
                )
                outputs[:bs] = self._forward_model(
                    DeviceModelInputs(input_ids[:bs], positions[:, :bs])
                )
                self.model.compute_logits(outputs[:bs])
                with torch.cuda.graph(logits_graph, self.graph_pool):
                    packed_model_inputs.copy_(
                        host_packed_model_inputs,
                        non_blocking=True,
                    )
                    packed_decode_metadata.copy_(
                        host_packed_decode_metadata,
                        non_blocking=True,
                    )
                    outputs[:bs] = self._forward_model(
                        DeviceModelInputs(input_ids[:bs], positions[:, :bs])
                    )
                    captured_logits = self.model.compute_logits(outputs[:bs])

            self.graphs[bs] = logits_graph  # 存起来, 按 bs 索引
            self.greedy_graphs[bs] = greedy_graph
            self.graph_vars[bs] = dict(
                packed_model_inputs=packed_model_inputs,
                packed_decode_metadata=packed_decode_metadata,
                host_packed_model_inputs=host_packed_model_inputs,
                host_packed_decode_metadata=host_packed_decode_metadata,
                host_packed_model_inputs_numpy=host_packed_model_inputs_numpy,
                host_packed_decode_metadata_numpy=host_packed_decode_metadata_numpy,
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                decode_max_context_len=decode_max_context_len,
                block_tables=block_tables,
                host_input_ids=host_input_ids,
                host_positions=host_positions,
                host_slot_mapping=host_slot_mapping,
                host_context_lens=host_context_lens,
                host_decode_max_context_len=host_decode_max_context_len,
                host_block_tables=host_block_tables,
                outputs=outputs,
            )
            self.graph_logits[bs] = captured_logits
            self.graph_greedy_tokens[bs] = captured_greedy_tokens
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        self.cudagraph_capture_ms = (perf_counter() - capture_start) * 1000.0
        # 回放时: 把实际数据拷到这些 tensor 里 → replay → 读 outputs
        # 这些 tensor 的 GPU 地址在整个生命周期内不变 (CUDA Graph 的要求)
