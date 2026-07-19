import atexit  # atexit.register: 注册进程退出时的清理函数(类似C++ std::atexit)
import gc
import socket
from dataclasses import dataclass
from time import perf_counter, perf_counter_ns
from typing import Any, cast
from tqdm.auto import tqdm  # 进度条库, auto版本自动适配终端/Jupyter
from transformers import AutoTokenizer  # HuggingFace分词器: 文本↔token_ids
import torch
import torch.multiprocessing as mp  # PyTorch多进程模块(比标准multiprocessing多CUDA tensor共享支持)

from prism_infer.observability import (
    get_performance_profile_session,
    profile_region,
)
from prism_infer.config import Config, PrismConfig
from prism_infer.runtime_capabilities import validate_runtime_capabilities
from prism_infer.sampling_params import SamplingParams
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.vl_inputs import (
    ImageInputs,
    VideoInputs,
    load_vl_processor,
    prepare_image_inputs,
    prepare_interleaved_image_inputs,
    prepare_video_inputs,
)
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.contracts import MetricsSink, StepResult
from prism_infer.engine.executor import ModelExecutor
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.engine.tp_control import DEFAULT_TP_CONTROL_TIMEOUT_SECONDS
from prism_infer.engine.request import (
    MonotonicRequestIdAllocator,
    RequestIdAllocator,
    validate_request_id,
)
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from prism_infer.models.model_registry import validate_model_architecture


@dataclass(slots=True)
class _GenerationProgress:
    """Own optional progress reporting for every synchronous generation API."""

    progress_bar: Any | None
    prefill_throughput: float = 0.0
    decode_throughput: float = 0.0

    def observe_step(self, *, signed_tokens: int, elapsed_seconds: float) -> None:
        if self.progress_bar is None or elapsed_seconds <= 0:
            return
        if signed_tokens > 0:
            self.prefill_throughput = signed_tokens / elapsed_seconds
        else:
            self.decode_throughput = -signed_tokens / elapsed_seconds
        self.progress_bar.set_postfix(
            {
                "Prefill": f"{int(self.prefill_throughput)}tok/s",
                "Decode": f"{int(self.decode_throughput)}tok/s",
            }
        )

    def complete_request(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


@dataclass(frozen=True, slots=True)
class _MixedRequestSpec:
    """Validated host-side description of one heterogeneous request."""

    request_type: str
    prompt: str | list[int]
    media: Any | None = None


_MIXED_MEDIA_FIELD_BY_TYPE = {
    "image": "image",
    "images": "images",
    "video": "video",
}


def select_distributed_init_method() -> str:
    """Select an engine-local TCP rendezvous instead of a fixed magic port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def validate_tensor_parallel_environment(config: Config, device_count: int) -> None:
    """在 spawn/NCCL 前校验 TP 硬件和模型分片约束。"""

    tp_size = config.tensor_parallel_size
    if tp_size > device_count:
        raise RuntimeError(
            f"tensor_parallel_size={tp_size} requires at least {tp_size} visible "
            f"CUDA devices, but only {device_count} are available"
        )
    hf_config = config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    dimensions = {
        "num_attention_heads": getattr(text_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "intermediate_size": getattr(text_config, "intermediate_size", None),
        "vocab_size": getattr(text_config, "vocab_size", None),
    }
    invalid = {
        name: value
        for name, value in dimensions.items()
        if not isinstance(value, int) or value <= 0 or value % tp_size != 0
    }
    if invalid:
        raise ValueError(
            f"tensor_parallel_size={tp_size} cannot evenly shard model dimensions: {invalid}"
        )


class LLMEngine:
    """推理引擎主控: 管理多进程、调度、主循环
    用户通过LLM(继承自LLMEngine)使用, 所有核心逻辑都在这里
    """

    def __init__(
        self,
        model: str | PrismConfig,
        *,
        metrics_sink: MetricsSink | None = None,
        clock_ns=perf_counter_ns,
        request_id_allocator: RequestIdAllocator | None = None,
        **config_options: object,
    ):
        config = Config(model, **config_options)
        validate_runtime_capabilities(
            execution_backend=config.execution_backend,
            compression_mode=config.compression_mode,
        )
        validate_model_architecture(config.hf_config)
        validate_tensor_parallel_environment(config, torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config = config.with_eos(self.tokenizer.eos_token_id)
        if request_id_allocator is None:
            request_id_allocator = MonotonicRequestIdAllocator()
        if not isinstance(request_id_allocator, RequestIdAllocator):
            raise TypeError("request_id_allocator must implement allocate() -> int")
        self.request_id_allocator = request_id_allocator

        # --- 多进程TP初始化 ---
        self.ps = []  # 子进程列表
        self.control_senders = []  # rank 0 <-> worker 的 typed duplex 控制通道
        distributed_init_method = select_distributed_init_method()
        ctx = mp.get_context("spawn")  # spawn模式: 新建Python解释器(fork对CUDA不安全)
        try:
            # 创建子进程: rank=1,2,...,N-1 分别在GPU1,2,...,N-1上运行
            for i in range(1, config.tensor_parallel_size):
                rank0_channel, worker_channel = ctx.Pipe(duplex=True)
                process = ctx.Process(
                    target=ModelRunner,
                    args=(
                        config,
                        i,
                        worker_channel,
                        distributed_init_method,
                    ),
                )
                process.start()
                worker_channel.close()
                self.ps.append(process)
                self.control_senders.append(rank0_channel)
            # 主进程自己也创建ModelRunner: rank=0, 在GPU0上运行
            self.model_runner = ModelRunner(
                config,
                0,
                self.control_senders,
                distributed_init_method,
            )
        except BaseException:
            for channel in self.control_senders:
                channel.close()
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=config.tensor_parallel_timeout_seconds)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=config.tensor_parallel_timeout_seconds)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
            raise
        config = config.with_cache_capacity(
            num_kvcache_blocks=self.model_runner.num_kvcache_blocks,
            num_cpu_blocks=self.model_runner.num_cpu_blocks,
        )
        self.model_runner.config = config

        # --- Tokenizer + Scheduler ---
        self.vl_processor = (
            load_vl_processor(
                config.model,
                image_max_pixels=config.image_max_pixels,
                video_max_pixels=config.video_max_pixels,
            )
            if self.model_runner.is_vl_model
            else None
        )
        self.config = config
        self.clock_ns = clock_ns
        self.scheduler = Scheduler(config, clock_ns=clock_ns)  # 调度器
        self.executor = ModelExecutor(
            config,
            self.model_runner,
            self.scheduler.block_manager,
        )
        self.metrics: MetricsSink = metrics_sink if metrics_sink is not None else EngineMetrics()
        atexit.register(self.exit)  # 注册退出清理函数(类似RAII析构+atexit)

    def _allocate_request_id(self) -> int:
        """Allocate request identity from the engine-owned source."""

        allocator = getattr(self, "request_id_allocator", None)
        if allocator is None:
            # Lightweight ``__new__`` construction used by focused tests still
            # gets engine-owned deterministic identity rather than Sequence
            # class state.
            allocator = MonotonicRequestIdAllocator()
            self.request_id_allocator = allocator
        request_id = allocator.allocate()
        try:
            return validate_request_id(
                request_id,
                name="allocated request id",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"request id allocator returned an invalid value: {request_id!r}"
            ) from exc

    @staticmethod
    def _first_cleanup_failure(
        current: BaseException | None,
        candidate: BaseException | None,
    ) -> BaseException | None:
        return current if current is not None else candidate

    def _release_model_runner(self) -> BaseException | None:
        failure: BaseException | None = None
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            try:
                model_runner.call("exit")  # 通过IPC通知所有子进程退出无限循环
            except BaseException as exc:
                failure = exc
            finally:
                # A partial runner exit must not preserve the backend -> runner
                # ownership edge or the model/KV tensors behind it.
                backend = getattr(model_runner, "execution_backend", None)
                if backend is not None:
                    try:
                        backend.release()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                    if hasattr(model_runner, "execution_backend"):
                        del model_runner.execution_backend
                # ModelExecutor intentionally owns a runner reference.  Drop
                # both public references before releasing the CUDA cache.
                if hasattr(self, "model_runner"):
                    del self.model_runner
        if hasattr(self, "executor"):
            del self.executor
        return failure

    def _close_control_channels(self) -> BaseException | None:
        failure: BaseException | None = None
        channels = tuple(getattr(self, "control_senders", ()))
        self.control_senders = []
        for sender in channels:
            try:
                sender.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        return failure

    def _worker_shutdown_timeout(self) -> float:
        return float(
            getattr(
                getattr(self, "config", None),
                "tensor_parallel_timeout_seconds",
                DEFAULT_TP_CONTROL_TIMEOUT_SECONDS,
            )
        )

    @staticmethod
    def _shutdown_worker(
        process: Any,
        *,
        timeout: float,
        terminate_first: bool,
    ) -> BaseException | None:
        try:
            if terminate_first and process.is_alive():
                process.terminate()
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                process.join(timeout=timeout)
            if process.is_alive():
                return RuntimeError(f"TP worker process {process.pid} did not exit within timeout")
        except BaseException as exc:
            return exc
        return None

    def _shutdown_workers(self, *, terminate_first: bool) -> BaseException | None:
        failure: BaseException | None = None
        processes = tuple(getattr(self, "ps", ()))
        self.ps = []
        timeout = self._worker_shutdown_timeout()
        for process in processes:
            worker_failure = self._shutdown_worker(
                process,
                timeout=timeout,
                terminate_first=terminate_first,
            )
            failure = self._first_cleanup_failure(failure, worker_failure)
        return failure

    def exit(self) -> None:
        """Stop workers and release all runner-owned CPU/GPU resources once."""

        if getattr(self, "_exited", False):
            return
        self._exited = True
        # ``atexit`` keeps a strong reference to bound methods. Explicit
        # shutdown must unregister it so completed engines do not accumulate
        # in a long-lived process.
        atexit.unregister(self.exit)
        failure = self._release_model_runner()
        failure = self._first_cleanup_failure(
            failure,
            self._close_control_channels(),
        )
        failure = self._first_cleanup_failure(
            failure,
            self._shutdown_workers(terminate_first=failure is not None),
        )
        # Model modules and hooks may contain cycles outside the explicit
        # runner/backend edge.  Collection is exit-only, never on the hot path.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if failure is not None:
            raise failure

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """添加一条推理请求: 文本→tokenize→创建Sequence→加入调度队列"""

        seq = self._prepare_text_sequence(prompt, sampling_params)
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def _prepare_text_sequence(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Convert one text request into scheduler-owned host state."""

        if isinstance(prompt, str):
            with profile_region("preprocess.tokenizer", cuda=False):
                prompt = self.tokenizer.encode(prompt)  # 文本→token_ids
        elif not isinstance(prompt, list):
            raise TypeError("text prompt must be a string or list of token ids")
        return Sequence(
            prompt,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
        )

    def _submit_sequence(
        self,
        seq: Sequence,
        *,
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """Submit one prepared request through admission and metrics contracts."""

        if not hasattr(self, "metrics"):
            # Lightweight construction tests and embedders may instantiate an
            # engine shell via ``__new__`` and attach scheduler/processor only.
            self.metrics = EngineMetrics()
        clock_ns = getattr(self, "clock_ns", perf_counter_ns)
        arrival_ns = clock_ns() if submitted_ns is None else submitted_ns
        self.metrics.on_request_submitted(seq, timestamp_ns=arrival_ns)
        try:
            decision = self.scheduler.add(
                seq,
                raise_on_reject=raise_on_reject,
            )
        except Exception:
            marker = getattr(self.metrics, "mark_terminal", None)
            if marker is not None:
                marker(
                    seq.seq_id,
                    reason="rejected",
                    timestamp_ns=clock_ns(),
                )
            raise
        if not decision.accepted:
            marker = getattr(self.metrics, "mark_terminal", None)
            if marker is not None:
                marker(
                    seq.seq_id,
                    reason="rejected",
                    timestamp_ns=clock_ns(),
                )
        return seq.seq_id

    def _submit_image_inputs(
        self,
        inputs: ImageInputs,
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None,
        raise_on_reject: bool,
    ) -> int:
        """将已通过 processor 校验的单图/多图输入提交给统一 runtime。"""

        seq = self._prepare_image_sequence(inputs, sampling_params)
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def _prepare_image_sequence(
        self,
        inputs: ImageInputs,
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Build one image sequence without mutating scheduler queues."""

        with profile_region("preprocess.mrope_positions", cuda=False):
            position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
                inputs.input_ids,
                config=self.config.hf_config,
                image_grid_thw=inputs.image_grid_thw,
                attention_mask=inputs.attention_mask,
            )
        return Sequence.from_image_inputs(
            inputs,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
            position_ids=position_ids,
            rope_delta=rope_delta,
        )

    def _prepare_image_request(
        self,
        prompt: str,
        image: Any,
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Run image preprocessing without publishing a request."""

        if self.vl_processor is None:
            raise ValueError("image generation requires a Qwen3-VL model config")
        with profile_region("preprocess.image_processor", cuda=False):
            inputs = prepare_image_inputs(self.vl_processor, prompt, image)
        return self._prepare_image_sequence(inputs, sampling_params)

    def add_vl_request(
        self,
        prompt: str,
        image,
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """添加一条图像 VL 请求。

        image 可以是一张图片，也可以是多张图片的 list/tuple。processor 作为
        非核心预处理工具使用；position ids 和 engine 推理由 Prism-Infer 自实现。
        """

        seq = self._prepare_image_request(prompt, image, sampling_params)
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def add_images_request(
        self,
        prompt: str,
        images,
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """添加一条多图 VL 请求，语义化别名。"""

        return self.add_vl_request(
            prompt,
            images,
            sampling_params,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def add_interleaved_images_request(
        self,
        prompt: str,
        images,
        sampling_params: SamplingParams,
        *,
        image_marker: str = "<image>",
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """提交图片按 marker 穿插在文本中的多图请求。"""

        if self.vl_processor is None:
            raise ValueError("add_interleaved_images_request requires a Qwen3-VL model config")
        with profile_region("preprocess.image_processor", cuda=False):
            inputs = prepare_interleaved_image_inputs(
                self.vl_processor,
                prompt,
                images,
                image_marker=image_marker,
            )
        return self._submit_image_inputs(
            inputs,
            sampling_params,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def add_video_request(
        self,
        prompt: str,
        video,
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None = None,
        raise_on_reject: bool = True,
    ) -> int:
        """添加一条视频 VL 请求。"""

        seq = self._prepare_video_request(prompt, video, sampling_params)
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def _submit_video_inputs(
        self,
        inputs: VideoInputs,
        sampling_params: SamplingParams,
        *,
        submitted_ns: int | None,
        raise_on_reject: bool,
    ) -> int:
        """将已通过 processor 校验的视频输入提交给统一 runtime。"""

        seq = self._prepare_video_sequence(inputs, sampling_params)
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

    def _prepare_video_sequence(
        self,
        inputs: VideoInputs,
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Build one video sequence without mutating scheduler queues."""

        with profile_region("preprocess.mrope_positions", cuda=False):
            position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
                inputs.input_ids,
                config=self.config.hf_config,
                video_grid_thw=inputs.video_grid_thw,
                attention_mask=inputs.attention_mask,
            )
        return Sequence.from_video_inputs(
            inputs,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
            position_ids=position_ids,
            rope_delta=rope_delta,
        )

    def _prepare_video_request(
        self,
        prompt: str,
        video: Any,
        sampling_params: SamplingParams,
    ) -> Sequence:
        """Run video preprocessing without publishing a request."""

        if self.vl_processor is None:
            raise ValueError("video generation requires a Qwen3-VL model config")
        with profile_region("preprocess.video_processor", cuda=False):
            inputs = prepare_video_inputs(self.vl_processor, prompt, video)
        return self._prepare_video_sequence(inputs, sampling_params)

    def step_result(self) -> StepResult:
        """Execute one strongly typed schedule → execute → commit cycle."""

        profile_session = get_performance_profile_session()
        if profile_session is not None:
            profile_session.begin_step()
        step_status = "error"
        try:
            with profile_region("engine.scheduler.schedule", cuda=False):
                plan = self.scheduler.schedule()
            self.metrics.on_batch_planned(plan)
            if profile_session is not None:
                profile_session.annotate_step(
                    phase=plan.phase.value,
                    batch_size=plan.batch_size,
                    sequence_ids=list(plan.sequence_ids),
                    sequence_lengths=[len(seq) for seq in plan.sequences],
                    prompt_tokens=sum(seq.num_prompt_tokens for seq in plan.sequences),
                    image_tokens=sum(seq.image_token_count for seq in plan.sequences),
                    video_tokens=sum(seq.video_token_count for seq in plan.sequences),
                    vision_patches=plan.num_scheduled_vision_patches,
                    scheduled_tokens=plan.num_scheduled_tokens,
                    scheduler_policy=plan.policy_name,
                )
            clock_ns = getattr(self, "clock_ns", perf_counter_ns)
            started_ns = clock_ns()
            execution = self.executor.execute(plan)
            finished_ns = clock_ns()
            self.metrics.on_batch_completed(
                plan,
                execution,
                started_ns=started_ns,
                finished_ns=finished_ns,
            )
            with profile_region("engine.scheduler.postprocess", cuda=False):
                outputs = self.scheduler.postprocess(plan, execution.token_ids)
            completed_ns = clock_ns()
            self.metrics.on_requests_finished(
                outputs,
                timestamp_ns=completed_ns,
            )
            result = StepResult(
                plan=plan,
                outputs=outputs,
                execution=execution,
                elapsed_ns=completed_ns - plan.created_ns,
            )
            step_status = "ok"
            return result
        finally:
            if profile_session is not None:
                profile_session.end_step(status=step_status)

    def step(self):
        """Compatibility API returning ``(finished, signed_token_count)``."""

        return self.step_result().as_legacy_tuple()

    def is_finished(self):
        """所有请求是否处理完毕(waiting和running都为空)"""
        return self.scheduler.is_finished()

    def cancel_request(self, request_id: int) -> bool:
        """Cancel an admitted request and release owned KV blocks."""

        cancelled = self.scheduler.cancel(request_id)
        if cancelled:
            marker = getattr(self.metrics, "mark_terminal", None)
            if marker is not None:
                marker(
                    request_id,
                    reason="cancelled",
                    timestamp_ns=getattr(self, "clock_ns", perf_counter_ns)(),
                )
        return cancelled

    def metrics_snapshot(self) -> dict[str, object]:
        """Return an immutable-by-convention copy of engine metrics records."""

        snapshot = getattr(self.metrics, "snapshot", None)
        if snapshot is None:
            raise RuntimeError("configured metrics sink does not expose snapshots")
        return snapshot()

    def reset_metrics(self) -> None:
        """Reset request/batch/scheduler ledgers between idle benchmark runs."""

        reset = getattr(self.metrics, "reset", None)
        if reset is None:
            raise RuntimeError("configured metrics sink cannot be reset")
        self.scheduler.reset_metrics()
        reset()

    def request_state(self, request_id: int):
        """Return the authoritative FSM state for a submitted request."""

        seq = self.scheduler.get_request(request_id)
        return None if seq is None else seq.status

    @staticmethod
    def _normalize_sampling_params(
        sampling_params: SamplingParams | list[SamplingParams],
        *,
        request_count: int,
        request_label: str,
    ) -> list[SamplingParams]:
        normalized = (
            list(sampling_params)
            if isinstance(sampling_params, list)
            else [sampling_params] * request_count
        )
        if len(normalized) != request_count:
            raise ValueError(
                f"sampling_params must contain one entry per {request_label}: "
                f"{len(normalized)} != {request_count}"
            )
        invalid = [
            type(params).__name__ for params in normalized if not isinstance(params, SamplingParams)
        ]
        if invalid:
            raise TypeError(f"sampling_params entries must be SamplingParams, got {invalid}")
        return normalized

    def _run_generation(
        self,
        sequence_ids: list[int],
        *,
        progress_bar: Any | None,
    ) -> list[dict[str, Any]]:
        """Drive one synchronous generation set through the shared engine loop."""

        requested_ids = frozenset(sequence_ids)
        completed: dict[int, list[int]] = {}
        progress = _GenerationProgress(progress_bar)
        try:
            while not self.is_finished():
                started = perf_counter()
                step_outputs, signed_tokens = self.step()
                progress.observe_step(
                    signed_tokens=signed_tokens,
                    elapsed_seconds=perf_counter() - started,
                )
                for request_id, token_ids in step_outputs:
                    if request_id not in requested_ids:
                        continue
                    if request_id not in completed:
                        progress.complete_request()
                    completed[request_id] = token_ids
            missing = [request_id for request_id in sequence_ids if request_id not in completed]
            if missing:
                raise RuntimeError(
                    f"generation finished without terminal outputs for request ids: {missing}"
                )
            return [
                self._format_generation_output(completed[request_id]) for request_id in sequence_ids
            ]
        finally:
            progress.close()

    def generate(
        self,
        prompts: list[str] | list[list[int]],  # 输入: 字符串列表或token_id列表的列表
        sampling_params: SamplingParams
        | list[SamplingParams],  # 采样参数: 单个(所有请求共用)或列表
        use_tqdm: bool = True,  # 是否显示进度条
    ) -> list[dict[str, Any]]:
        """对外公开接口: 批量输入prompt, 返回生成结果"""
        normalized_params = self._normalize_sampling_params(
            sampling_params,
            request_count=len(prompts),
            request_label="prompt",
        )
        if not prompts:
            return []
        sequences = [
            self._prepare_text_sequence(prompt, params)
            for prompt, params in zip(prompts, normalized_params, strict=True)
        ]
        # Request identifiers are allocator-defined and need not be monotonic.
        # Preserve submission order explicitly instead of sorting identifiers.
        submitted_ids = [self._submit_sequence(sequence) for sequence in sequences]
        progress_bar = (
            tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        )
        return self._run_generation(
            submitted_ids,
            progress_bar=progress_bar,
        )

    def _format_generation_output(self, token_ids: list[int]) -> dict[str, Any]:
        """Expose user text and a lossless special-token decode alongside IDs."""

        decode_options = {"clean_up_tokenization_spaces": False}
        return {
            "text": self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                **decode_options,
            ),
            "raw_text": self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                **decode_options,
            ),
            "token_ids": token_ids,
        }

    def _finish_single_generation(
        self,
        seq_id: int,
        progress_bar: Any | None,
    ) -> dict[str, Any]:
        """统一收集 image/video 单请求，避免公开 VL API 漂移。"""

        return self._run_generation(
            [seq_id],
            progress_bar=progress_bar,
        )[0]

    def generate_vl(
        self,
        prompt: str,
        image,
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> dict:
        """Qwen3-VL 图像生成入口。

        image 可以是一张图片，也可以是多张图片的 list/tuple。当前仍是
        单请求 eager correctness 路径；视频和 batch VL 会在 P3 后续阶段扩展。
        """

        seq_id = self.add_vl_request(prompt, image, sampling_params)
        pbar = tqdm(total=1, desc="Generating VL", dynamic_ncols=True) if use_tqdm else None
        return self._finish_single_generation(seq_id, pbar)

    def generate_images(
        self,
        prompt: str,
        images,
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> dict:
        """多图生成入口，语义化别名。"""

        return self.generate_vl(prompt, images, sampling_params, use_tqdm=use_tqdm)

    def generate_interleaved_images(
        self,
        prompt: str,
        images,
        sampling_params: SamplingParams,
        *,
        image_marker: str = "<image>",
        use_tqdm: bool = True,
    ) -> dict[str, Any]:
        """生成 marker-interleaved 多图请求。"""

        seq_id = self.add_interleaved_images_request(
            prompt,
            images,
            sampling_params,
            image_marker=image_marker,
        )
        pbar = (
            tqdm(total=1, desc="Generating Interleaved Images", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        return self._finish_single_generation(seq_id, pbar)

    def generate_prepared_image_inputs(
        self,
        inputs: ImageInputs,
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = True,
    ) -> dict[str, Any]:
        """生成一次已预处理 image 输入，供需要审计 prompt IDs 的工具使用。"""

        seq_id = self._submit_image_inputs(
            inputs,
            sampling_params,
            submitted_ns=None,
            raise_on_reject=True,
        )
        pbar = (
            tqdm(total=1, desc="Generating Prepared Images", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        return self._finish_single_generation(seq_id, pbar)

    def generate_video(
        self,
        prompt: str,
        video,
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> dict:
        """Qwen3-VL 视频生成入口。"""

        seq_id = self.add_video_request(prompt, video, sampling_params)
        pbar = tqdm(total=1, desc="Generating Video", dynamic_ncols=True) if use_tqdm else None
        return self._finish_single_generation(seq_id, pbar)

    def generate_prepared_video_inputs(
        self,
        inputs: VideoInputs,
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = True,
    ) -> dict[str, Any]:
        """生成一次已预处理 video 输入，供质量工具复核 prompt IDs。"""

        seq_id = self._submit_video_inputs(
            inputs,
            sampling_params,
            submitted_ns=None,
            raise_on_reject=True,
        )
        pbar = (
            tqdm(total=1, desc="Generating Prepared Video", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        return self._finish_single_generation(seq_id, pbar)

    @staticmethod
    def _parse_mixed_request(
        request: object,
        *,
        index: int,
    ) -> _MixedRequestSpec:
        if not isinstance(request, dict):
            raise TypeError(f"mixed request {index} must be a dict")
        request_type = request.get("type", "text")
        if not isinstance(request_type, str):
            raise TypeError(f"mixed request {index} type must be a string")
        if "prompt" not in request or request["prompt"] is None:
            raise ValueError(f"mixed request {index} is missing 'prompt'")
        prompt = request["prompt"]
        if request_type == "text":
            if not isinstance(prompt, (str, list)):
                raise TypeError(
                    f"mixed text request {index} prompt must be a string or token-id list"
                )
            return _MixedRequestSpec(request_type=request_type, prompt=prompt)

        media_field = _MIXED_MEDIA_FIELD_BY_TYPE.get(request_type)
        if media_field is None:
            raise ValueError(f"unsupported mixed request type: {request_type}")
        if not isinstance(prompt, str):
            raise TypeError(f"mixed {request_type} request {index} prompt must be a string")
        if media_field not in request:
            raise ValueError(f"{request_type} request must provide {media_field!r}")
        return _MixedRequestSpec(
            request_type=request_type,
            prompt=prompt,
            media=request[media_field],
        )

    def _prepare_mixed_sequence(
        self,
        request: _MixedRequestSpec,
        sampling_params: SamplingParams,
    ) -> Sequence:
        if request.request_type == "text":
            return self._prepare_text_sequence(request.prompt, sampling_params)
        prompt = cast(str, request.prompt)
        if request.request_type in {"image", "images"}:
            return self._prepare_image_request(prompt, request.media, sampling_params)
        if request.request_type == "video":
            return self._prepare_video_request(prompt, request.media, sampling_params)
        raise RuntimeError(f"unvalidated mixed request type: {request.request_type}")

    def generate_mixed(
        self,
        requests: list[dict[str, Any]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, Any]]:
        """混合 text/image/images/video 请求批量生成。

        request 格式:
          - {"type": "text", "prompt": "..."}
          - {"type": "image", "prompt": "...", "image": image}
          - {"type": "images", "prompt": "...", "images": [image0, image1]}
          - {"type": "video", "prompt": "...", "video": frames}

        P3.3 当前覆盖 non-prefix mixed batch eager correctness。
        """

        if not requests:
            return []
        normalized_params = self._normalize_sampling_params(
            sampling_params,
            request_count=len(requests),
            request_label="mixed request",
        )
        specs = [
            self._parse_mixed_request(request, index=index)
            for index, request in enumerate(requests)
        ]
        # Complete host preprocessing before publishing any request to the
        # scheduler. A malformed later media item cannot strand earlier work.
        sequences = [
            self._prepare_mixed_sequence(request, params)
            for request, params in zip(specs, normalized_params, strict=True)
        ]
        sequence_ids = [self._submit_sequence(sequence) for sequence in sequences]
        progress_bar = (
            tqdm(total=len(requests), desc="Generating Mixed", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        return self._run_generation(
            sequence_ids,
            progress_bar=progress_bar,
        )
