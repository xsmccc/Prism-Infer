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

import pickle                                      # 序列化, 用于多 GPU 间传递数据
from multiprocessing.connection import Connection
from time import perf_counter

import torch
import torch.distributed as dist                   # 分布式通信 (NCCL)

from prism_infer.analysis.performance_profile import profile_region
from prism_infer.config import (
    Config,
    ExecutionBackendName,
    MAX_CUDA_GRAPH_BATCH_SIZE,
)
from prism_infer.engine.compression import (
    build_compression_metadata,
    compression_supports_cuda_graph,
    compression_mode_uses_fp8_payload,
    compression_mode_uses_token_head_scales,
    build_visual_pruning_config,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    DeviceBatch,
    DeviceModelInputs,
    ExecutionResult,
)
from prism_infer.engine.kv_layout import KVCompactionPlan
from prism_infer.engine.kv_quantization import (
    KV_COMPONENT_COUNT,
    KV_SCALE_DTYPE,
    kv_block_storage_bytes,
)
from prism_infer.engine.visual_pruning import (
    build_retained_slot_mapping,
    build_runtime_visual_token_scorer,
    finalize_attention_pruning_decisions,
)
from prism_infer.ops.kv_compaction import compact_kv_slots
try:
    from prism_infer.models.qwen3 import Qwen3ForCausalLM     # Qwen3 纯文本模型 (legacy)
except ImportError:
    Qwen3ForCausalLM = None  # VL 项目中纯文本模型可能不存在, 用 VL 版替代
from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM
from prism_infer.layers.sampler import Sampler              # 采样器 (温度采样/贪婪)
from prism_infer.analysis.kv_trace import build_trace_metadata, register_model_config
from prism_infer.utils.context import (  # 全局上下文
    get_context,
    install_context,
    reset_context,
    set_context,
)
from prism_infer.utils.loader import load_model             # 权重加载


CUDA_GRAPH_SMALL_BATCH_BUCKETS = (1, 2, 4, 8)
CUDA_GRAPH_BATCH_BUCKET_STRIDE = 16


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


class ModelRunnerExecutionBackend:
    """Prepare tensor-only batches and execute the startup-selected backend."""

    def __init__(self, runner: "ModelRunner") -> None:
        self.runner = runner
        self.name = ExecutionBackendName(
            runner.config.execution_backend
        ).value
        self._released = False

    def prepare(self, plan: BatchPlan) -> DeviceBatch:
        if self._released:
            raise RuntimeError("execution backend was released")
        seqs = list(plan.sequences)
        try:
            with profile_region(
                "runner.prepare_inputs",
                metadata={"phase": plan.phase.value},
            ):
                model_inputs = (
                    self.runner.prepare_prefill(seqs)
                    if plan.is_prefill
                    else self.runner.prepare_decode(seqs)
                )
            if not isinstance(model_inputs, DeviceModelInputs):
                model_inputs = DeviceModelInputs(
                    input_ids=model_inputs.input_ids,
                    position_ids=model_inputs.position_ids,
                    pixel_values=getattr(model_inputs, "pixel_values", None),
                    image_grid_thw=getattr(model_inputs, "image_grid_thw", None),
                    pixel_values_videos=getattr(
                        model_inputs,
                        "pixel_values_videos",
                        None,
                    ),
                    video_grid_thw=getattr(model_inputs, "video_grid_thw", None),
                )
            with profile_region("runner.prepare_sample_inputs"):
                temperatures = (
                    self.runner.prepare_sample(seqs)
                    if self.runner.rank == 0
                    else None
                )
            attention_context = get_context()
            execution_bucket = plan.batch_size
            if (
                plan.phase is BatchPhase.DECODE
                and self.name == ExecutionBackendName.CUDA_GRAPH.value
                and hasattr(self.runner, "graph_bs")
            ):
                execution_bucket = next(
                    size
                    for size in self.runner.graph_bs
                    if size >= plan.batch_size
                )
            kv_scale_cache = getattr(self.runner, "kv_scale_cache", None)
            return DeviceBatch(
                phase=plan.phase,
                sequence_ids=plan.sequence_ids,
                scheduled_token_counts=plan.scheduled_token_counts,
                model_inputs=model_inputs,
                attention_context=attention_context,
                temperatures=temperatures,
                execution_bucket=execution_bucket,
                kv_scale_views=(
                    ()
                    if kv_scale_cache is None
                    else (
                        kv_scale_cache[0],
                        kv_scale_cache[1],
                    )
                ),
            )
        finally:
            # The only state crossing prepare -> execute is the immutable
            # DeviceBatch.  Attention state is re-installed explicitly below.
            reset_context()

    def warmup(self, bucket: int | None = None) -> None:
        if self._released:
            raise RuntimeError("execution backend was released")
        if bucket is not None and bucket <= 0:
            raise ValueError(f"warmup bucket must be positive, got {bucket}")
        self.runner.warmup_model()

    def capture(self, bucket: int | None = None) -> None:
        if self._released:
            raise RuntimeError("execution backend was released")
        if self.name != ExecutionBackendName.CUDA_GRAPH.value:
            raise RuntimeError(
                f"execution backend {self.name!r} does not support capture"
            )
        if bucket is not None and bucket <= 0:
            raise ValueError(f"capture bucket must be positive, got {bucket}")
        self.runner.capture_cudagraph()
        if bucket is not None and bucket not in self.runner.graph_bs:
            raise ValueError(
                f"capture bucket {bucket} is not in {self.runner.graph_bs}"
            )

    def execute(self, device_batch: DeviceBatch) -> ExecutionResult:
        if self._released:
            raise RuntimeError("execution backend was released")
        if not isinstance(device_batch, DeviceBatch):
            raise TypeError(
                "execute requires DeviceBatch, "
                f"got {type(device_batch).__name__}"
            )
        install_context(device_batch.attention_context)
        try:
            with profile_region("runner.run_model"):
                logits = self.runner.run_model(
                    device_batch.model_inputs,
                    device_batch.phase is BatchPhase.PREFILL,
                )
            if self.runner.rank == 0:
                if device_batch.temperatures is None:
                    raise RuntimeError(
                        "rank 0 DeviceBatch requires temperatures"
                    )
                with profile_region("runner.sampler"):
                    token_ids = tuple(
                        self.runner.sampler(
                            logits,
                            device_batch.temperatures,
                        ).tolist()
                    )
            else:
                token_ids = tuple(None for _ in device_batch.sequence_ids)
            return ExecutionResult(token_ids=token_ids)
        finally:
            reset_context()

    def release(self) -> None:
        if self._released:
            return
        for name in ("graphs", "graph_pool", "graph_vars"):
            if hasattr(self.runner, name):
                delattr(self.runner, name)
        self._released = True


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
        hf_config = config.hf_config                # HuggingFace 模型配置 (层数/head数等)
        self.block_size = config.kvcache_block_size  # KV Cache 块大小 (如 16)
        self.enforce_eager = config.enforce_eager    # True = 禁用 CUDA Graph, 每次都 eager 执行
        self.world_size = config.tensor_parallel_size  # 几张 GPU (Tensor Parallel 并行度)
        self.rank = rank                             # 当前 GPU 编号 (0, 1, 2, ...)
        self.control_channel = control_channel       # rank0 为发送端列表, worker 为接收端
        self.model_dtype = _resolve_model_dtype(hf_config)
        self.kv_cache_dtype = self._resolve_kv_cache_dtype(config)
        self.uses_token_head_scales = compression_mode_uses_token_head_scales(
            config.compression_mode
        )
        self.kv_scale_cache: torch.Tensor | None = None
        self.cpu_kv_scale_cache: torch.Tensor | None = None
        self.cudagraph_capture_ms = 0.0
        self.decode_compile_first_call_ms = 0.0
        self.decode_compile_first_call_pending = False
        self.last_cudagraph_actual_batch_size: int | None = None
        self.last_cudagraph_replay_batch_size: int | None = None
        register_model_config(config)

        # ── 初始化分布式通信 ──
        if not isinstance(distributed_init_method, str):
            raise TypeError("distributed_init_method must be a string")
        torch.cuda.set_device(rank)                  # 绑定当前进程到第 rank 号 GPU
        distributed_backend = "nccl" if self.world_size > 1 else "gloo"
        init_options = {
            "backend": distributed_backend,
            "init_method": distributed_init_method,
            "world_size": self.world_size,
            "rank": rank,
        }
        if distributed_backend == "nccl":
            init_options["device_id"] = torch.device("cuda", rank)
        dist.init_process_group(**init_options)
        # TP ranks 使用 NCCL；TP1 用 Gloo 提供 rank/world contract，避免空 collective
        # 仍提前占用 CUDA communicator memory。
        # rendezvous 地址由 engine 每次启动动态选择，避免固定端口冲突。
        # C++ 类比: MPI_Init() + MPI_Comm_rank()

        # ── 创建模型、KV Cache 和 CUDA Graph ──
        default_dtype = torch.get_default_dtype()     # 保存调用方默认类型/设备
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(self.model_dtype)   # 设为模型精度 (如 bfloat16)
            torch.set_default_device("cuda")          # 后续 torch.empty() 等默认在 GPU 上
            if self._is_vl_config(hf_config) or Qwen3ForCausalLM is None:
                self.model = Qwen3VLForCausalLM(
                    hf_config,
                    mlp_projection_mode=config.mlp_projection_mode,
                )  # Qwen3-VL 模型结构
                self.model.logits_precision = config.logits_precision
                self.is_vl_model = True
            else:
                if config.logits_precision != "model":
                    raise ValueError(
                        "logits_precision='fp32' historical reproduction is "
                        "currently supported only for Qwen3-VL"
                    )
                self.model = Qwen3ForCausalLM(hf_config)  # legacy Qwen3 纯文本结构
                self.is_vl_model = False
            load_model(self.model, config.model)       # 从文件加载权重到模型
            self.sampler = Sampler()                   # 创建采样器
            self.execution_backend = ModelRunnerExecutionBackend(self)

            self.execution_backend.warmup()            # 热身: 跑一次前向, 触发 CUDA kernel 编译
            self.allocate_kv_cache()                   # 根据剩余显存计算能分配多少 block, 分配 KV Cache
            self._configure_decode_compile()
            if not self.enforce_eager:
                self.execution_backend.capture()       # 录制 CUDA Graph (Decode 加速)
        except BaseException:
            # Break the runner <-> backend ownership cycle even when model
            # loading, warmup, KV allocation, or Graph capture fails.
            backend = getattr(self, "execution_backend", None)
            if backend is not None:
                backend.release()
                del self.execution_backend
            reset_context()
            raise
        finally:
            try:
                torch.set_default_device(default_device)
            finally:
                torch.set_default_dtype(default_dtype)

        # ── 多 GPU 同步 (Tensor Parallel) ──
        if self.world_size > 1:
            if rank == 0:
                if not isinstance(self.control_channel, list):
                    raise TypeError("rank 0 requires a list of TP control senders")
                if len(self.control_channel) != self.world_size - 1:
                    raise ValueError(
                        "rank 0 TP control sender count must equal world_size - 1: "
                        f"senders={len(self.control_channel)}, world_size={self.world_size}"
                    )
                self._tp_barrier()                      # 等所有 worker 完成模型初始化
            else:
                if not isinstance(self.control_channel, Connection):
                    raise TypeError("TP worker requires one control receiver")
                self._tp_barrier()
                self.loop()                             # 从进程进入无限循环, 等待主进程指令
                # 注意: 只有 rank>0 会进入 loop(), rank=0 继续执行正常流程

    # ─────────────────────────────────────────────────────────
    # exit: 清理资源
    # ─────────────────────────────────────────────────────────
    def _tp_barrier(self) -> None:
        """Synchronize NCCL ranks on each process's explicitly bound GPU."""

        dist.barrier(device_ids=[self.rank])

    def exit(self) -> None:
        if self.world_size > 1:
            channels = (
                self.control_channel
                if isinstance(self.control_channel, list)
                else [self.control_channel]
            )
            for channel in channels:
                channel.close()
            self._tp_barrier()                          # 等所有进程都关了
        backend = getattr(self, "execution_backend", None)
        if backend is not None:
            backend.release()
            # Backend owns a runner reference.  Removing the reverse edge is
            # required for deterministic in-process model/KV reclamation.
            del self.execution_backend
        reset_context()
        torch.cuda.synchronize()                        # 等所有 GPU 操作完成
        dist.destroy_process_group()                    # 销毁分布式通信组
        # C++ 类比: MPI_Finalize()

    # ─────────────────────────────────────────────────────────
    # 多 GPU 通信: loop / read_control_message / write_control_message / call
    #
    # 工作原理:
    #   rank 0 调用 call("run", seqs, True)
    #   → call 把方法名+参数 pickle 后通过每个 worker 的单向 Pipe 发送
    #   → rank>0 的 loop 阻塞读取一条完整的变长消息
    #   → rank>0 调用 self.run(seqs, True) → 同步执行相同操作
    # ─────────────────────────────────────────────────────────

    def loop(self) -> None:
        """从进程的无限循环: 等待主进程发指令, 执行相同的方法"""
        while True:
            method_name, args = self.read_control_message()
            self.call(method_name, *args)               # 执行对应方法
            if method_name == "exit":
                break                                    # 收到 exit 指令则退出循环

    @staticmethod
    def _deserialize_control_message(data: bytes) -> tuple[str, list[object]]:
        """反序列化并校验一条 TP 控制消息。"""

        try:
            message = pickle.loads(data)
        except (pickle.PickleError, EOFError, AttributeError, ImportError, IndexError) as exc:
            raise RuntimeError("failed to deserialize TP control message") from exc
        if not isinstance(message, list) or not message:
            raise ValueError("TP control message must be a non-empty list")
        method_name, *args = message
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("TP control method name must be a non-empty string")
        return method_name, args

    def read_control_message(self) -> tuple[str, list[object]]:
        """worker 阻塞读取一条有边界的变长控制消息。"""

        assert self.world_size > 1 and self.rank > 0
        if not isinstance(self.control_channel, Connection):
            raise TypeError("TP worker control receiver is not a Connection")
        try:
            data = self.control_channel.recv_bytes()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"TP worker rank {self.rank} lost its control channel"
            ) from exc
        return self._deserialize_control_message(data)

    def write_control_message(self, method_name: str, *args: object) -> int:
        """rank 0 向全部 worker 广播一条变长控制消息。"""

        assert self.world_size > 1 and self.rank == 0
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("TP control method name must be a non-empty string")
        if not isinstance(self.control_channel, list):
            raise TypeError("rank 0 TP control senders must be a list")
        try:
            data = pickle.dumps(
                [method_name, *args],
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (pickle.PickleError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"failed to serialize TP control message for method={method_name!r}"
            ) from exc
        for worker_rank, channel in enumerate(self.control_channel, start=1):
            try:
                channel.send_bytes(data)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(
                    "failed to send TP control message: "
                    f"worker_rank={worker_rank}, method={method_name!r}, "
                    f"payload_bytes={len(data)}"
                ) from exc
        return len(data)

    def call(self, method_name: str, *args: object) -> object:
        """调用自身的方法, 同时通知从进程也调用相同方法"""
        if self.world_size > 1 and self.rank == 0:
            self.write_control_message(method_name, *args)
        method = getattr(self, method_name, None)       # 反射: 通过字符串找到方法
        if method is None or not callable(method):
            raise AttributeError(f"unknown ModelRunner method: {method_name!r}")
        # getattr(self, "run") 等价于 self.run
        # C++ 类比: 类似 std::unordered_map<string, function_ptr>
        return method(*args)                            # 调用方法

    # ═══════════════════════════════════════════════════════════
    # 初始化阶段: warmup + allocate_kv_cache
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _is_vl_config(hf_config) -> bool:
        model_type = getattr(hf_config, "model_type", "")
        return (
            "vl" in str(model_type).lower()
            or hasattr(hf_config, "vision_config")
            or hasattr(hf_config, "image_token_id")
        )

    @staticmethod
    def _resolve_kv_cache_dtype(config: Config) -> torch.dtype:
        """根据 compression mode 选择物理 KV cache dtype。"""

        if compression_mode_uses_fp8_payload(config.compression_mode):
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError(
                    f"compression_mode={config.compression_mode!r} requires "
                    "torch.float8_e4m3fn"
                )
            return torch.float8_e4m3fn
        return _resolve_model_dtype(config.hf_config)

    def _forward_model(self, inputs: DeviceModelInputs):
        """Call either the new Qwen3-VL interface or the legacy text model."""
        if self.is_vl_model:
            return self.model(input_ids=inputs.input_ids,
                              position_ids=inputs.position_ids,
                              pixel_values=inputs.pixel_values,
                              image_grid_thw=inputs.image_grid_thw,
                              pixel_values_videos=inputs.pixel_values_videos,
                              video_grid_thw=inputs.video_grid_thw)
        return self.model(inputs.input_ids, inputs.position_ids)

    def _configure_decode_compile(self) -> None:
        """按显式配置启用 attention-only decode compile preflight。"""

        region = self.config.decode_compile_region
        if region == "none":
            return
        if region != "attention" or not self.is_vl_model:
            raise RuntimeError(
                "decode compile currently supports only Qwen3-VL attention"
            )
        attention_layers = [
            layer.self_attn
            for layer in self.model.model.language_model.layers
        ]
        if not attention_layers:
            raise RuntimeError("decode attention compile found no decoder layers")
        for attention in attention_layers:
            attention.enable_decode_compile(
                mode=self.config.decode_compile_mode,
                emulate_precision_casts=(
                    self.config.decode_compile_emulate_precision_casts
                ),
                force_same_precision=(
                    self.config.decode_compile_force_same_precision
                ),
            )
        self.decode_compile_first_call_pending = True

    def compile_metadata(self) -> dict[str, object]:
        """返回 decode ``torch.compile`` 的可审计配置和 cold first call。"""

        enabled = self.config.decode_compile_region != "none"
        return {
            "enabled": enabled,
            "region": (
                "decode_attention" if enabled else "none"
            ),
            "backend": "inductor" if enabled else "none",
            "mode": self.config.decode_compile_mode if enabled else "none",
            "emulate_precision_casts": (
                self.config.decode_compile_emulate_precision_casts
                if enabled else False
            ),
            "force_same_precision": (
                self.config.decode_compile_force_same_precision
                if enabled else False
            ),
            "first_call_ms": self.decode_compile_first_call_ms if enabled else 0.0,
        }

    def warmup_model(self):
        """热身: 用最大尺寸输入跑一次模型, 触发 CUDA kernel 编译/分配"""
        torch.cuda.empty_cache()                        # 清空 GPU 缓存
        torch.cuda.reset_peak_memory_stats()            # 重置峰值内存统计
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
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
        self.run(seqs, True)                            # 跑一次 Prefill (前向传播)
        # 目的: 让 PyTorch/CUDA 编译 kernel, 知道峰值内存
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 GPU 剩余显存, 计算能分配多少 KV Cache block, 一次性分配"""
        config = self.config
        hf_config = config.hf_config
        text_config = _text_hf_config(hf_config)
        free, total = torch.cuda.mem_get_info()         # GPU 显存: 空闲 / 总量
        used = total - free                             # 已使用
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # PyTorch 峰值占用
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # PyTorch 当前占用

        # ── 计算每个 block 的字节数 ──
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        # GQA/MQA: KV head 数可能比 Q head 少, 再除以 TP 并行度
        head_dim = getattr(text_config, "head_dim",
                           text_config.hidden_size // text_config.num_attention_heads)
        # 每个 head 的维度, 如 128

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
        block_bytes = self.kv_block_bytes
        # C++ 类比: sizeof(Block_KV) = 2 * layers * block_size * heads * dim * sizeof(half)

        # ── 计算能分配多少个 block ──
        max_num_kvcache_blocks = int(
            total * config.gpu_memory_utilization        # GPU 总量 × 利用率 (如 0.9)
            - used                                       # 减去已用
            - peak + current                             # 减去峰值瞬时占用
        ) // block_bytes
        requested_blocks = config.num_kvcache_blocks
        if requested_blocks > 0:
            if requested_blocks > max_num_kvcache_blocks:
                raise RuntimeError(
                    "requested num_kvcache_blocks exceeds available memory: "
                    f"requested={requested_blocks}, max={max_num_kvcache_blocks}, "
                    f"kv_cache_dtype={self.kv_cache_dtype}, "
                    f"payload_block_bytes={self.kv_payload_block_bytes}, "
                    f"scale_block_bytes={self.kv_scale_block_bytes}"
                )
            num_kvcache_blocks = requested_blocks
        else:
            num_kvcache_blocks = max_num_kvcache_blocks
        # 这就是 "剩余空间能放多少个 block"
        if num_kvcache_blocks <= 0:
            raise RuntimeError(
                "no KV cache blocks fit in the configured GPU memory budget"
            )
        self.num_kvcache_blocks = num_kvcache_blocks

        # ── 一次性分配整个 KV Cache 张量 ──
        self.kv_cache = torch.empty(
            KV_COMPONENT_COUNT,                          # 0=K_cache, 1=V_cache
            text_config.num_hidden_layers,               # 每层一份
            self.num_kvcache_blocks,                     # block 个数
            self.block_size,                             # 每 block token 数
            num_kv_heads,                                # KV head 数
            head_dim,                                    # head 维度
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
        # shape 示例: [2, 28, 500, 16, 8, 128]
        # 2 = K/V, 28层, 500个block, 每block 16 token, 8个KV head, 128维
        # 整个 KV Cache 在一个连续张量里! block_id 就是第 2 维的索引

        # ── 把 KV Cache 分配给每一层的 Attention ──
        layer_id = 0
        for module in self.model.modules():              # 遍历模型所有子模块
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 找到 Attention 层 (有 k_cache 和 v_cache 属性的模块)
                module.layer_idx = layer_id
                module.k_cache = self.kv_cache[0, layer_id]  # shape: [num_blocks, block_size, heads, dim]
                module.v_cache = self.kv_cache[1, layer_id]
                module.k_scale_cache = (
                    None
                    if self.kv_scale_cache is None
                    else self.kv_scale_cache[0, layer_id]
                )
                module.v_scale_cache = (
                    None
                    if self.kv_scale_cache is None
                    else self.kv_scale_cache[1, layer_id]
                )
                layer_id += 1
        assert layer_id == text_config.num_hidden_layers, (
            f"KV cache layer count mismatch: assigned={layer_id}, "
            f"expected={text_config.num_hidden_layers}"
        )
        # 这样每一层的 Attention 直接引用这个大张量的一个切片

        # ── 分配 CPU 端 KV Cache (Swap 用, pinned memory 加速 GPU↔CPU 传输) ──
        # C++ 类比: cudaMallocHost (pinned memory, page-locked)
        # 比普通 CPU 内存传输快 2-3 倍, 因为避免了额外的内存拷贝
        self.num_cpu_blocks = int(
            self.num_kvcache_blocks * config.cpu_kv_cache_ratio
        )
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
                pin_memory=True  # pinned memory: 加速 GPU↔CPU 传输
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
        """复制 KV Cache blocks: 把 src block 的数据复制到 dst block"""
        scale_cache = self._bound_gpu_scale_cache()
        for src_block_id, dst_block_id in cow_pairs:
            # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            # 复制所有层的 K 和 V
            self.kv_cache[:, :, dst_block_id].copy_(self.kv_cache[:, :, src_block_id])
            if scale_cache is not None:
                scale_cache[:, :, dst_block_id].copy_(
                    scale_cache[:, :, src_block_id]
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
                raise RuntimeError(
                    f"duplicate compaction plan for seq_id={plan.seq_id}"
                )
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
            raise RuntimeError(
                "GPU/CPU scale caches must either both exist or both be absent"
            )
        for src_id, dst_id in swap_map:
            if direction == "out":
                # GPU → CPU (异步, non_blocking=True)
                self.cpu_kv_cache[:, :, dst_id].copy_(
                    self.kv_cache[:, :, src_id], non_blocking=True)
                if scale_cache is not None:
                    cpu_scale_cache[:, :, dst_id].copy_(
                        scale_cache[:, :, src_id], non_blocking=True
                    )
            else:
                # CPU → GPU (异步)
                self.kv_cache[:, :, dst_id].copy_(
                    self.cpu_kv_cache[:, :, src_id], non_blocking=True)
                if scale_cache is not None:
                    scale_cache[:, :, dst_id].copy_(
                        cpu_scale_cache[:, :, src_id], non_blocking=True
                    )
        # 确保搬运完成后再继续 (类似 cudaStreamSynchronize)
        torch.cuda.synchronize()

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把每条序列的 block_table 对齐并转成 GPU 张量"""
        swapped = [seq.seq_id for seq in seqs if getattr(seq, "cpu_block_table", [])]
        if swapped:
            raise RuntimeError(
                "prepare_block_tables requires GPU block_table; "
                f"swapped sequences still on CPU: {swapped}"
            )
        max_len = max(len(seq.block_table) for seq in seqs)  # 找最长的 block_table
        # 不同序列的 block 数可能不同, 需要 padding 到相同长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 短的用 -1 填充 (padding)
        # 例: [[0,1,2], [0,1]] → [[0,1,2], [0,1,-1]]
        block_tables = torch.tensor(block_tables, dtype=torch.int32,
                                     pin_memory=True).cuda(non_blocking=True)
        # pin_memory=True: 分配在锁页内存 (CPU), 加速 CPU→GPU 传输
        # .cuda(non_blocking=True): 异步拷贝到 GPU, 不等拷贝完成
        # C++ 类比: cudaMemcpyAsync(gpu_ptr, cpu_pinned_ptr, size, cudaMemcpyHostToDevice)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prefill 准备: 拼接多条序列的 token, 计算位置和 slot_mapping"""
        register_model_config(self.config)
        swapped = [seq.seq_id for seq in seqs if getattr(seq, "cpu_block_table", [])]
        if swapped:
            raise RuntimeError(f"cannot prepare prefill for swapped sequences: {swapped}")
        input_ids = []          # 所有序列的 token 拼成一个一维列表
        positions = []          # 每个 token 的位置编号 (RoPE 用)
        vl_position_chunks = []
        cu_seqlens_q = [0]      # cumulative sequence lengths for Q (前缀和)
        cu_seqlens_k = [0]      # cumulative sequence lengths for K
        max_seqlen_q = 0        # 最长的 Q 序列长度 (Flash Attention 需要)
        max_seqlen_k = 0        # 最长的 K 序列长度
        slot_mapping = []       # 每个 token 在 KV Cache 中的全局槽位编号
        block_tables = None     # Prefix Cache 命中时才需要
        context_lens = None     # paged chunk/prefix prefill 的完整 K 长度
        pixel_value_chunks = []
        image_grid_chunks = []
        video_value_chunks = []
        video_grid_chunks = []
        has_vl_positions = any(seq.position_ids is not None for seq in seqs)

        for seq in seqs:
            seqlen = len(seq)                           # 序列总长度

            input_ids.extend(seq[seq.num_cached_tokens:])
            # seq[start:] = 从 num_cached_tokens 位置开始的 token
            # Prefix Cache 命中的 token 不用喂给模型 → 跳过前 num_cached_tokens 个
            # 如果没有 cache hit, num_cached_tokens=0 → 喂全部 token

            if has_vl_positions:
                if seq.position_ids is not None:
                    # seq.position_ids: [3, 1, seqlen] -> current chunk [3, seqlen_q]
                    vl_position_chunks.append(seq.position_ids[:, 0, seq.num_cached_tokens:seqlen])
                else:
                    text_pos = torch.arange(seq.num_cached_tokens, seqlen, dtype=torch.long)
                    vl_position_chunks.append(text_pos.view(1, -1).expand(3, -1))
                current_tokens = seq[seq.num_cached_tokens:seqlen]
                if seq.pixel_values is not None:
                    image_tokens = current_tokens.count(seq.image_token_id)
                    if image_tokens not in (0, seq.image_token_count):
                        raise ValueError(
                            "chunk boundary splits image token payload: "
                            f"seq={seq.seq_id} chunk_tokens={image_tokens} "
                            f"expected={seq.image_token_count}"
                        )
                    if image_tokens:
                        pixel_value_chunks.append(seq.pixel_values)
                        image_grid_chunks.append(seq.image_grid_thw)
                if seq.pixel_values_videos is not None:
                    video_tokens = current_tokens.count(seq.video_token_id)
                    if video_tokens not in (0, seq.video_token_count):
                        raise ValueError(
                            "chunk boundary splits video token payload: "
                            f"seq={seq.seq_id} chunk_tokens={video_tokens} "
                            f"expected={seq.video_token_count}"
                        )
                    if video_tokens:
                        video_value_chunks.append(seq.pixel_values_videos)
                        video_grid_chunks.append(seq.video_grid_thw)
            else:
                positions.extend(list(range(seq.num_cached_tokens, seqlen)))
                # 位置编号: [num_cached_tokens, num_cached_tokens+1, ..., seqlen-1]
                # RoPE 需要每个 token 的绝对位置

            seqlen_q = seqlen - seq.num_cached_tokens   # Q 的实际长度 (去掉 cached 部分)
            seqlen_k = seqlen                           # K 的长度 = 整个序列 (包括 cached)
            # Q ≠ K 的情况: Prefix Cache 命中时, Q 只算不在 cache 里的, K 包含全部

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # Q 的累积长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # K 的累积长度
            # Flash Attention 的 varlen API 需要: 告诉它每条序列在哪里开始/结束

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup 时没有 block_table
                continue

            # ── slot_mapping: 计算每个(非 cached) token 存到 KV Cache 的哪个位置 ──
            for token_index in range(seq.num_cached_tokens, seqlen):
                block_index = token_index // self.block_size
                block_offset = token_index % self.block_size
                slot_mapping.append(
                    seq.block_table[block_index] * self.block_size
                    + block_offset
                )

        paged_prefill = cu_seqlens_k[-1] > cu_seqlens_q[-1]

        # ── 全部转成 GPU tensor ──
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if has_vl_positions:
            positions = torch.cat(vl_position_chunks, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if paged_prefill:
            block_tables = self.prepare_block_tables(seqs)
            context_lens = torch.tensor(
                [len(seq) for seq in seqs],
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)

        trace_metadata = build_trace_metadata(
            seqs,
            is_prefill=True,
            input_ids=input_ids,
            position_ids=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=self.block_size,
        )
        compression_metadata = build_compression_metadata(
            self.config,
            seqs,
            is_prefill=True,
        )
        visual_pruning_scorer = None
        pruning_config = compression_metadata.visual_pruning_config
        if (
            pruning_config is not None
            and pruning_config.get("strategy") == "attention"
            and compression_metadata.enabled
            and compression_metadata.total_visual_tokens > 0
            and (pixel_value_chunks or video_value_chunks)
        ):
            text_config = _text_hf_config(self.config.hf_config)
            visual_pruning_scorer = build_runtime_visual_token_scorer(
                seqs,
                num_hidden_layers=int(text_config.num_hidden_layers),
                attention_last_n_layers=int(
                    pruning_config["attention_last_n_layers"]
                ),
            )

        # ── 设置全局上下文 (attention 层会读取这些信息) ──
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, context_lens, block_tables, trace_metadata=trace_metadata,
                    compression_metadata=compression_metadata,
                    visual_pruning_scorer=visual_pruning_scorer)
        # True = is_prefill
        # attention 层通过 get_context() 获取这些信息来决定怎么计算

        pixel_values = torch.cat(pixel_value_chunks, dim=0) if pixel_value_chunks else None
        image_grid_thw = torch.cat(image_grid_chunks, dim=0) if image_grid_chunks else None
        pixel_values_videos = torch.cat(video_value_chunks, dim=0) if video_value_chunks else None
        video_grid_thw = torch.cat(video_grid_chunks, dim=0) if video_grid_chunks else None
        if pixel_values is not None:
            pixel_values = pixel_values.pin_memory().cuda(non_blocking=True)
            image_grid_thw = image_grid_thw.pin_memory().cuda(non_blocking=True)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.pin_memory().cuda(non_blocking=True)
            video_grid_thw = video_grid_thw.pin_memory().cuda(non_blocking=True)

        return DeviceModelInputs(input_ids=input_ids,
                           position_ids=positions,
                           pixel_values=pixel_values,
                           image_grid_thw=image_grid_thw,
                           pixel_values_videos=pixel_values_videos,
                           video_grid_thw=video_grid_thw)

    def prepare_decode(self, seqs: list[Sequence]):
        """Decode 准备: 每条序列只取最后一个 token"""
        register_model_config(self.config)
        swapped = [seq.seq_id for seq in seqs if getattr(seq, "cpu_block_table", [])]
        if swapped:
            raise RuntimeError(f"cannot prepare decode for swapped sequences: {swapped}")
        input_ids = []
        positions = []
        vl_position_chunks = []
        slot_mapping = []
        context_lens = []           # 每条序列的上下文长度 (= 总 token 数)
        logical_context_lens = []   # 未压缩逻辑长度，M-RoPE position仍按它生成
        has_vl_positions = any(seq.rope_delta is not None for seq in seqs)

        for seq in seqs:
            input_ids.append(seq.last_token)            # 只取最后一个 token
            if has_vl_positions:
                delta = int(seq.rope_delta.item()) if seq.rope_delta is not None else 0
                pos = torch.arange(1, dtype=torch.long) + (len(seq) - 1 + delta)
                vl_position_chunks.append(pos.expand(3, -1))
            else:
                positions.append(len(seq) - 1)          # 位置 = 序列长度 - 1
            context_lens.append(seq.physical_kv_len)    # attention读取实际物理 KV 长度
            logical_context_lens.append(len(seq))       # M-RoPE/trace 保留逻辑长度

            slot_mapping.append(
                seq.block_table[-1] * self.block_size
                + seq.physical_last_block_num_tokens
                - 1
            )
            # 新 token 的 KV 存到: 最后一个 block 的最后一个已占用位置
            # 例: block_table[-1]=5, block_size=16, last_block_num_tokens=3
            # → slot = 5*16 + 3 - 1 = 82 (0-indexed)
            # 注意: last_block_num_tokens 已经包含了刚 append 的新 token

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if has_vl_positions:
            positions = torch.cat(vl_position_chunks, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logical_context_lens = torch.tensor(
            logical_context_lens,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        trace_metadata = build_trace_metadata(
            seqs,
            is_prefill=False,
            input_ids=input_ids,
            position_ids=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=self.block_size,
        )
        compression_metadata = build_compression_metadata(
            self.config,
            seqs,
            is_prefill=False,
        )
        visual_pruning_slot_mappings: tuple[torch.Tensor, ...] = ()
        if compression_metadata.visual_pruning_effective:
            records = compression_metadata.visual_pruning_records_by_batch
            if len(records) != len(seqs):
                raise RuntimeError(
                    "visual pruning records must align with decode batch: "
                    f"records={len(records)}, sequences={len(seqs)}"
                )
            with profile_region(
                "runner.visual_prune.build_slot_mappings",
                metadata={"batch_size": len(seqs)},
            ):
                visual_pruning_slot_mappings = tuple(
                    build_retained_slot_mapping(
                        record,
                        len(seq),
                        seq.block_table,
                        self.block_size,
                        device=block_tables.device,
                    )
                    for seq, record in zip(seqs, records)
                )
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    logical_context_lens=logical_context_lens,
                    block_tables=block_tables, trace_metadata=trace_metadata,
                    compression_metadata=compression_metadata,
                    visual_pruning_slot_mappings=visual_pruning_slot_mappings)
        # False = is_decode
        return DeviceModelInputs(input_ids=input_ids, position_ids=positions)

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数: 收集每条序列的温度"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32,
                                     pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @staticmethod
    def _as_mrope_decode_positions(position_ids: torch.Tensor) -> torch.Tensor:
        """把 decode position ids 规范为 `[3, batch]`。

        text-only decode 原本是一维 `[batch]`；CUDA Graph replay 统一使用
        `[3, max_bs]` 占位，三轴同值与文本 M-RoPE 语义等价。
        """

        if position_ids.ndim == 1:
            return position_ids.view(1, -1).expand(3, -1)
        if position_ids.ndim == 2 and position_ids.shape[0] == 3:
            return position_ids
        raise ValueError(
            "decode position_ids must be [batch] or [3, batch], "
            f"got {list(position_ids.shape)}"
        )

    @staticmethod
    def _cudagraph_batch_sizes(max_bs: int) -> list[int]:
        """生成 CUDA Graph decode batch 档位，确保覆盖 `max_bs`。

        常用档位保持 1/2/4/8/16...；当 `max_num_seqs` 是 3、5、17 等
        非标准档位时，额外录制最后的 `max_bs`，避免 replay 查找失败。
        """

        if max_bs < 1:
            raise ValueError(f"max_bs must be >= 1, got {max_bs}")
        graph_bs = [
            batch_size
            for batch_size in CUDA_GRAPH_SMALL_BATCH_BUCKETS
            if batch_size <= max_bs
        ]
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
            raise ValueError(
                f"requested_batch_size must be >= 1, got {requested_batch_size}"
            )
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
            batch_size
            for batch_size in self.graph_bs
            if batch_size >= requested_batch_size
        )
        return {
            "enabled": True,
            "capture_scope": "decode_model_forward",
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
    # @torch.inference_mode(): 禁用梯度计算 + 自动求导, 推理更快更省内存
    # 比 torch.no_grad() 更彻底 (连 autograd 元数据都不创建)
    def run_model(self, model_inputs: DeviceModelInputs, is_prefill: bool):
        """执行模型前向推理, 返回 logits"""
        input_ids = model_inputs.input_ids
        context = get_context()
        compression_metadata = context.compression_metadata
        compression_requires_eager = not compression_supports_cuda_graph(
            compression_metadata
        )
        if (
            is_prefill
            or self.enforce_eager
            or compression_requires_eager
        ):
            # ---- Prefill / Eager / batch 太大 → 直接跑模型 ----
            # logical visual_prune 必须走 eager；其 retained-slot gather 依赖
            # 动态 Python metadata。physical compact/FP8 由静态 tensor 表达。
            with profile_region(
                "runner.model.forward",
                metadata={
                    "backend": (
                        "torch_compile_attention"
                        if (
                            not is_prefill
                            and self.config.decode_compile_region == "attention"
                        )
                        else "eager"
                    )
                },
            ):
                if (
                    not is_prefill
                    and self.decode_compile_first_call_pending
                ):
                    torch.cuda.synchronize()
                    compile_start = perf_counter()
                    hidden_states = self._forward_model(model_inputs)
                    torch.cuda.synchronize()
                    self.decode_compile_first_call_ms = (
                        perf_counter() - compile_start
                    ) * 1000.0
                    self.decode_compile_first_call_pending = False
                else:
                    hidden_states = self._forward_model(model_inputs)
            with profile_region("runner.model.compute_logits"):
                return self.model.compute_logits(hidden_states)
            # self.model(input_ids, positions): 前向传播, 返回 hidden_states
            # self.model.compute_logits(...): 乘以 lm_head 权重 → [batch, vocab_size]
        else:
            # ---- Decode + CUDA Graph → 回放预录制的 Graph ----
            bs = input_ids.size(0)                      # batch size (序列数)

            # 找到 >= bs 的最小预录制 batch size
            captured_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[captured_bs]
            self.last_cudagraph_actual_batch_size = bs
            self.last_cudagraph_replay_batch_size = captured_bs
            # self.graph_bs = [1, 2, 4, 8, 16, 32, ...]
            # 例: bs=3 → 找到 4 → 用为 batch=4 录制的 graph
            # next + generator: 找第一个满足条件的

            graph_vars = self.graph_vars
            # graph_vars 是录制时的 tensor → 回放前把实际数据拷进去

            with profile_region("runner.cudagraph.copy_inputs"):
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:, :bs] = self._as_mrope_decode_positions(
                    model_inputs.position_ids
                )
                graph_vars["slot_mapping"].fill_(-1)        # 先全填 -1 (padding)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()           # 先全填 0
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"].fill_(-1)
                graph_vars["block_tables"][
                    :bs,
                    :context.block_tables.size(1),
                ] = context.block_tables

            with profile_region(
                "runner.cudagraph.replay",
                metadata={
                    "actual_batch_size": bs,
                    "captured_batch_size": captured_bs,
                },
            ):
                graph.replay()                           # 回放! GPU 执行预录制的所有 kernel
            # 回放不经过 Python → 无 CPU launch overhead → 极快

            with profile_region("runner.model.compute_logits"):
                return self.model.compute_logits(graph_vars["outputs"][:bs])
            # graph 输出写到 graph_vars["outputs"], 取前 bs 个

    def run_plan(self, plan: BatchPlan) -> ExecutionResult:
        """Execute one immutable host plan through a tensor-only DeviceBatch."""

        if not isinstance(plan, BatchPlan):
            raise TypeError(f"run_plan requires BatchPlan, got {type(plan).__name__}")
        seqs = list(plan.sequences)
        enable_chunked = self.config.enable_chunked_prefill
        max_chunk = self.config.max_chunk_size
        is_warmup = any(not seq.block_table for seq in seqs)
        chunked_active = plan.is_prefill and enable_chunked and not is_warmup
        if chunked_active:
            for index, seq in enumerate(seqs):
                remaining = seq.num_prompt_tokens - seq.num_computed_tokens
                chunk = plan.scheduled_token_counts[index]
                if chunk > remaining or chunk > max_chunk:
                    raise ValueError(
                        "invalid scheduled prefill chunk: "
                        f"seq={seq.seq_id} chunk={chunk} "
                        f"remaining={remaining} max_chunk={max_chunk}"
                    )
                seq._orig_num_cached_tokens = seq.num_cached_tokens
                seq.num_cached_tokens = seq.num_computed_tokens
                seq._orig_num_tokens = seq.num_tokens
                seq._orig_token_ids = seq.token_ids
                seq.num_tokens = seq.num_computed_tokens + chunk
                seq.token_ids = seq.token_ids[:seq.num_tokens]

        try:
            backend = getattr(self, "execution_backend", None)
            if backend is None:
                backend = ModelRunnerExecutionBackend(self)
                self.execution_backend = backend
            device_batch = backend.prepare(plan)
            execution = backend.execute(device_batch)
            token_ids = list(execution.token_ids)

            if plan.is_prefill:
                scorer = device_batch.attention_context.visual_pruning_scorer
                if scorer is not None:
                    with profile_region(
                        "runner.visual_prune.finalize_attention_scores"
                    ):
                        finalize_attention_pruning_decisions(
                            seqs,
                            build_visual_pruning_config(self.config),
                            scorer,
                        )

            if chunked_active:
                for index, seq in enumerate(seqs):
                    chunk = seq.num_tokens - seq.num_computed_tokens
                    seq.num_computed_tokens += chunk
                    seq.num_cached_tokens = seq._orig_num_cached_tokens
                    seq.num_tokens = seq._orig_num_tokens
                    seq.token_ids = seq._orig_token_ids
                    del (
                        seq._orig_num_tokens,
                        seq._orig_token_ids,
                        seq._orig_num_cached_tokens,
                    )
                    if not seq.is_prefill_finished:
                        token_ids[index] = None
            return ExecutionResult(token_ids=tuple(token_ids))
        finally:
            if chunked_active:
                for seq in seqs:
                    if hasattr(seq, "_orig_num_tokens"):
                        seq.num_tokens = seq._orig_num_tokens
                        seq.token_ids = seq._orig_token_ids
                        seq.num_cached_tokens = seq._orig_num_cached_tokens
                        del (
                            seq._orig_num_tokens,
                            seq._orig_token_ids,
                            seq._orig_num_cached_tokens,
                        )
            reset_context()

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        scheduled_token_counts: list[int] | None = None,
    ) -> list[int | None]:
        """One-cycle compatibility adapter for direct P1-P7 runner calls."""

        if scheduled_token_counts is None:
            is_warmup = any(not seq.block_table for seq in seqs)
            chunked_active = (
                is_prefill
                and self.config.enable_chunked_prefill
                and not is_warmup
            )
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
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(3, max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.ones(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, text_config.hidden_size)

        # ── 要录制的 batch size 列表 ──
        self.graph_bs = self._cudagraph_batch_sizes(max_bs)
        # [1, 2, 4, 8, 16, 32, 48, 64, ...]
        # 不是每个 bs 都录, 只录这些 "档位"
        # 实际 bs 不在列表里时, 向上找最近的 (如 bs=3 → 用 bs=4 的 graph)

        self.graphs = {}
        self.graph_pool = None                           # 共享 GPU 内存池

        for bs in reversed(self.graph_bs):               # 从大到小录
            graph = torch.cuda.CUDAGraph()

            # warmup: 先跑一次, 让 CUDA 编译 kernel
            slot_mapping[:bs] = torch.arange(bs, dtype=torch.int32, device=slot_mapping.device)
            context_lens[:bs] = 1
            block_tables[:bs].zero_()
            set_context(False, slot_mapping=slot_mapping[:bs],
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self._forward_model(
                DeviceModelInputs(input_ids[:bs], positions[:, :bs])
            )

            # capture: 录制!
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self._forward_model(
                    DeviceModelInputs(input_ids[:bs], positions[:, :bs])
                )
            # torch.cuda.graph(graph, pool):
            #   pool: 共享内存池, 不同 bs 的 graph 共享 GPU workspace
            #   with 块内的所有 GPU 操作被录制到 graph 里
            #   不会真正执行, 只是记录 "要执行哪些 kernel"

            if self.graph_pool is None:
                self.graph_pool = graph.pool()           # 第一个 graph 创建池

            self.graphs[bs] = graph                      # 存起来, 按 bs 索引
            torch.cuda.synchronize()
            reset_context()

        # ── 保存占位 tensor 的引用 ──
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        torch.cuda.synchronize()
        self.cudagraph_capture_ms = (perf_counter() - capture_start) * 1000.0
        # 回放时: 把实际数据拷到这些 tensor 里 → replay → 读 outputs
        # 这些 tensor 的 GPU 地址在整个生命周期内不变 (CUDA Graph 的要求)
