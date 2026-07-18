import atexit                           # atexit.register: 注册进程退出时的清理函数(类似C++ std::atexit)
import gc
import socket
from time import perf_counter, perf_counter_ns
from typing import Any
from tqdm.auto import tqdm              # 进度条库, auto版本自动适配终端/Jupyter
from transformers import AutoTokenizer  # HuggingFace分词器: 文本↔token_ids
import torch
import torch.multiprocessing as mp      # PyTorch多进程模块(比标准multiprocessing多CUDA tensor共享支持)

from prism_infer.analysis.performance_profile import (
    get_performance_profile_session,
    profile_region,
)
from prism_infer.config import Config, PrismConfig
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
from prism_infer.engine.request import (
    MonotonicRequestIdAllocator,
    RequestIdAllocator,
    validate_request_id,
)
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config


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
            f"tensor_parallel_size={tp_size} cannot evenly shard model dimensions: "
            f"{invalid}"
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
        validate_tensor_parallel_environment(config, torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config = config.with_eos(self.tokenizer.eos_token_id)
        if request_id_allocator is None:
            request_id_allocator = MonotonicRequestIdAllocator()
        if not isinstance(request_id_allocator, RequestIdAllocator):
            raise TypeError(
                "request_id_allocator must implement allocate() -> int"
            )
        self.request_id_allocator = request_id_allocator

        # --- 多进程TP初始化 ---
        self.ps = []                 # 子进程列表
        self.control_senders = []    # rank 0 -> worker 的变长控制消息通道
        distributed_init_method = select_distributed_init_method()
        ctx = mp.get_context("spawn")  # spawn模式: 新建Python解释器(fork对CUDA不安全)
        try:
            # 创建子进程: rank=1,2,...,N-1 分别在GPU1,2,...,N-1上运行
            for i in range(1, config.tensor_parallel_size):
                receiver, sender = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=ModelRunner,
                    args=(
                        config,
                        i,
                        receiver,
                        distributed_init_method,
                    ),
                )
                process.start()
                receiver.close()        # rank 0 不持有 worker 的接收端
                self.ps.append(process)
                self.control_senders.append(sender)
            # 主进程自己也创建ModelRunner: rank=0, 在GPU0上运行
            self.model_runner = ModelRunner(
                config,
                0,
                self.control_senders,
                distributed_init_method,
            )
        except BaseException:
            for sender in self.control_senders:
                sender.close()
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join()
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
        self.metrics: MetricsSink = (
            metrics_sink if metrics_sink is not None else EngineMetrics()
        )
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
                "request id allocator returned an invalid value: "
                f"{request_id!r}"
            ) from exc

    def exit(self):
        """清理: 通知子进程退出, 释放GPU资源, 等待子进程结束"""
        if getattr(self, "_exited", False):
            return
        self._exited = True
        # ``atexit`` keeps a strong reference to bound methods.  Explicit
        # shutdown must unregister it so completed engines do not accumulate
        # in a long-lived process.
        atexit.unregister(self.exit)
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
                if hasattr(self, "executor"):
                    del self.executor
                del self.model_runner
                backend = None
                model_runner = None
        for sender in getattr(self, "control_senders", []):
            try:
                sender.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        for p in getattr(self, "ps", []):
            if failure is not None and p.is_alive():
                p.terminate()
            p.join()                     # 等待所有子进程退出(类似C++ thread.join)
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
        if isinstance(prompt, str):
            with profile_region("preprocess.tokenizer", cuda=False):
                prompt = self.tokenizer.encode(prompt)  # 文本→token_ids
        seq = Sequence(
            prompt,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
        )
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
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

        with profile_region("preprocess.mrope_positions", cuda=False):
            position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
                inputs.input_ids,
                config=self.config.hf_config,
                image_grid_thw=inputs.image_grid_thw,
                attention_mask=inputs.attention_mask,
            )
        seq = Sequence.from_image_inputs(
            inputs,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

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

        if self.vl_processor is None:
            raise ValueError("generate_vl requires a Qwen3-VL model config")
        with profile_region("preprocess.image_processor", cuda=False):
            inputs = prepare_image_inputs(self.vl_processor, prompt, image)
        return self._submit_image_inputs(
            inputs,
            sampling_params,
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
            raise ValueError(
                "add_interleaved_images_request requires a Qwen3-VL model config"
            )
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

        if self.vl_processor is None:
            raise ValueError("generate_video requires a Qwen3-VL model config")
        with profile_region("preprocess.video_processor", cuda=False):
            inputs = prepare_video_inputs(self.vl_processor, prompt, video)
        return self._submit_video_inputs(
            inputs,
            sampling_params,
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

        with profile_region("preprocess.mrope_positions", cuda=False):
            position_ids, rope_delta = get_qwen3_vl_rope_index_from_config(
                inputs.input_ids,
                config=self.config.hf_config,
                video_grid_thw=inputs.video_grid_thw,
                attention_mask=inputs.attention_mask,
            )
        seq = Sequence.from_video_inputs(
            inputs,
            sampling_params,
            block_size=self.config.kvcache_block_size,
            request_id=self._allocate_request_id(),
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
        return self._submit_sequence(
            seq,
            submitted_ns=submitted_ns,
            raise_on_reject=raise_on_reject,
        )

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
                    prompt_tokens=sum(
                        seq.num_prompt_tokens for seq in plan.sequences
                    ),
                    image_tokens=sum(
                        seq.image_token_count for seq in plan.sequences
                    ),
                    video_tokens=sum(
                        seq.video_token_count for seq in plan.sequences
                    ),
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
                outputs = self.scheduler.postprocess(
                    plan, execution.token_ids
                )
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

    def generate(
        self,
        prompts: list[str] | list[list[int]],          # 输入: 字符串列表或token_id列表的列表
        sampling_params: SamplingParams | list[SamplingParams],  # 采样参数: 单个(所有请求共用)或列表
        use_tqdm: bool = True,                          # 是否显示进度条
    ) -> list[dict[str, Any]]:
        """对外公开接口: 批量输入prompt, 返回生成结果"""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 如果sampling_params不是列表, 复制N份(所有请求共用同一参数)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 逐个添加请求到调度队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # --- 主推理循环 ---
        outputs = {}                                   # {seq_id: completion_token_ids}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()                         # 计时开始
            output, num_tokens = self.step()            # 执行一步推理
            # 更新吞吐量统计(进度条显示用)
            if use_tqdm:
                if num_tokens > 0:  # prefill
                    prefill_throughput = num_tokens / (perf_counter() - t)  # tokens/sec
                else:               # decode
                    decode_throughput = -num_tokens / (perf_counter() - t)  # tokens/sec
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 收集本步完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 进度条+1(每完成一条序列, 不是每步+1)

        # --- 结果整理 ---
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]  # 按seq_id排序, 保证与输入顺序一致
        outputs = [self._format_generation_output(token_ids) for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs  # 返回: [{"text": "...", "token_ids": [...]}, ...]

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

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            started = perf_counter()
            output, num_tokens = self.step()
            if progress_bar is not None:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - started)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - started)
                progress_bar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for done_seq_id, token_ids in output:
                outputs[done_seq_id] = token_ids
                if progress_bar is not None:
                    progress_bar.update(1)
        if progress_bar is not None:
            progress_bar.close()
        token_ids = outputs[seq_id]
        return self._format_generation_output(token_ids)

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

        if use_tqdm:
            pbar = tqdm(total=1, desc="Generating VL", dynamic_ncols=True)
        else:
            pbar = None

        seq_id = self.add_vl_request(prompt, image, sampling_params)
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

        pbar = (
            tqdm(total=1, desc="Generating Interleaved Images", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        seq_id = self.add_interleaved_images_request(
            prompt,
            images,
            sampling_params,
            image_marker=image_marker,
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

        pbar = (
            tqdm(total=1, desc="Generating Prepared Images", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        seq_id = self._submit_image_inputs(
            inputs,
            sampling_params,
            submitted_ns=None,
            raise_on_reject=True,
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

        if use_tqdm:
            pbar = tqdm(total=1, desc="Generating Video", dynamic_ncols=True)
        else:
            pbar = None

        seq_id = self.add_video_request(prompt, video, sampling_params)
        return self._finish_single_generation(seq_id, pbar)

    def generate_prepared_video_inputs(
        self,
        inputs: VideoInputs,
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = True,
    ) -> dict[str, Any]:
        """生成一次已预处理 video 输入，供质量工具复核 prompt IDs。"""

        pbar = (
            tqdm(total=1, desc="Generating Prepared Video", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        seq_id = self._submit_video_inputs(
            inputs,
            sampling_params,
            submitted_ns=None,
            raise_on_reject=True,
        )
        return self._finish_single_generation(seq_id, pbar)

    def generate_mixed(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
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
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(requests)
        if len(sampling_params) != len(requests):
            raise ValueError(
                "sampling_params length must match requests length, "
                f"got {len(sampling_params)} vs {len(requests)}"
            )

        if use_tqdm:
            pbar = tqdm(total=len(requests), desc="Generating Mixed", dynamic_ncols=True)
        else:
            pbar = None

        seq_ids = []
        for request, sp in zip(requests, sampling_params):
            request_type = request.get("type", "text")
            prompt = request.get("prompt")
            if prompt is None:
                raise ValueError(f"mixed request missing prompt: {request}")
            if request_type == "text":
                seq_ids.append(self.add_request(prompt, sp))
            elif request_type == "image":
                if "image" not in request:
                    raise ValueError("image request must provide 'image'")
                seq_ids.append(self.add_vl_request(prompt, request["image"], sp))
            elif request_type == "images":
                if "images" not in request:
                    raise ValueError("images request must provide 'images'")
                seq_ids.append(self.add_images_request(prompt, request["images"], sp))
            elif request_type == "video":
                if "video" not in request:
                    raise ValueError("video request must provide 'video'")
                seq_ids.append(self.add_video_request(prompt, request["video"], sp))
            else:
                raise ValueError(f"unsupported mixed request type: {request_type}")

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if pbar is not None:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for done_seq_id, token_ids in output:
                outputs[done_seq_id] = token_ids
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        ordered = [outputs[seq_id] for seq_id in seq_ids]
        return [self._format_generation_output(token_ids) for token_ids in ordered]
