import atexit                           # atexit.register: 注册进程退出时的清理函数(类似C++ std::atexit)
from dataclasses import fields          # fields(Config): 获取dataclass的所有字段信息
from time import perf_counter, perf_counter_ns
from tqdm.auto import tqdm              # 进度条库, auto版本自动适配终端/Jupyter
from transformers import AutoTokenizer  # HuggingFace分词器: 文本↔token_ids
import torch
import torch.multiprocessing as mp      # PyTorch多进程模块(比标准multiprocessing多CUDA tensor共享支持)

from prism_infer.analysis.performance_profile import (
    get_performance_profile_session,
    profile_region,
)
from prism_infer.config import Config
from prism_infer.sampling_params import SamplingParams
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.vl_inputs import (
    load_vl_processor,
    prepare_image_inputs,
    prepare_video_inputs,
)
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.model_runner import ModelRunner
from prism_infer.engine.contracts import MetricsSink, StepResult
from prism_infer.engine.executor import ModelExecutor
from prism_infer.engine.metrics import EngineMetrics
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config


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

    def __init__(self, model, **kwargs):
        metrics_sink = kwargs.pop("metrics_sink", None)
        clock_ns = kwargs.pop("clock_ns", perf_counter_ns)
        # --- 参数过滤: 只保留Config认识的参数, 忽略其他 ---
        config_fields = {field.name for field in fields(Config)}  # Config的所有字段名集合
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}  # 过滤
        config = Config(model, **config_kwargs)  # 创建配置(触发__post_init__校验)
        validate_tensor_parallel_environment(config, torch.cuda.device_count())

        # --- 多进程TP初始化 ---
        self.ps = []                 # 子进程列表
        self.control_senders = []    # rank 0 -> worker 的变长控制消息通道
        ctx = mp.get_context("spawn")  # spawn模式: 新建Python解释器(fork对CUDA不安全)
        # 创建子进程: rank=1,2,...,N-1 分别在GPU1,2,...,N-1上运行
        for i in range(1, config.tensor_parallel_size):
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, receiver),
            )
            process.start()
            receiver.close()            # rank 0 不持有 worker 的接收端
            self.ps.append(process)
            self.control_senders.append(sender)
        # 主进程自己也创建ModelRunner: rank=0, 在GPU0上运行
        self.model_runner = ModelRunner(config, 0, self.control_senders)

        # --- Tokenizer + Scheduler ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)  # 分词器(只在主进程)
        self.vl_processor = load_vl_processor(config.model) if self.model_runner.is_vl_model else None
        self.config = config
        config.eos = self.tokenizer.eos_token_id  # 从tokenizer获取EOS token id(Config创建时还不知道)
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

    def exit(self):
        """清理: 通知子进程退出, 释放GPU资源, 等待子进程结束"""
        if getattr(self, "_exited", False):
            return
        self._exited = True
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")  # 通过IPC通知所有子进程退出无限循环
            # ModelExecutor intentionally owns a backend reference.  Drop that
            # boundary before deleting the public compatibility attribute so
            # model weights/KV cache are actually released between engines.
            if hasattr(self, "executor"):
                del self.executor
            del self.model_runner      # 释放主进程的ModelRunner(含GPU资源)
        for p in getattr(self, "ps", []):
            p.join()                     # 等待所有子进程退出(类似C++ thread.join)

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
        seq = Sequence(prompt, sampling_params)      # 创建序列对象
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
            position_ids=position_ids,
            rope_delta=rope_delta,
        )
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
    ) -> list[str]:
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
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]  # token_ids→文本
        if use_tqdm:
            pbar.close()
        return outputs  # 返回: [{"text": "...", "token_ids": [...]}, ...]

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
        token_ids = outputs[seq_id]
        return {
            "text": self.tokenizer.decode(token_ids),
            "token_ids": token_ids,
        }

    def generate_images(
        self,
        prompt: str,
        images,
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> dict:
        """多图生成入口，语义化别名。"""

        return self.generate_vl(prompt, images, sampling_params, use_tqdm=use_tqdm)

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
        token_ids = outputs[seq_id]
        return {
            "text": self.tokenizer.decode(token_ids),
            "token_ids": token_ids,
        }

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
        return [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in ordered
        ]
