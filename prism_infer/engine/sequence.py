from copy import copy              # copy模块: 浅拷贝(复制列表本身, 不复制元素)
from enum import Enum, auto        # Enum: 枚举类(类似C++ enum class); auto: 自动分配值
from itertools import count        # count(): 无限自增迭代器(0,1,2,3,...)

import torch

from prism_infer.engine.vl_inputs import ImageInputs, SingleImageInputs, VideoInputs
from prism_infer.sampling_params import SamplingParams


# 序列状态枚举: WAITING→RUNNING→FINISHED 三阶段生命周期
class SequenceStatus(Enum):
    WAITING = auto()    # 等待prefill(在scheduler.waiting队列中)
    RUNNING = auto()    # 正在decode(在scheduler.running队列中)
    FINISHED = auto()   # 生成结束(从队列移除)
    SWAPPED = auto()    # KV Cache已换出到CPU内存(等待换回)


class Sequence:
    block_size = 256    # 类变量: KV Cache块大小(所有实例共享)
    counter = count()   # 类变量: 全局自增ID计数器(类似C++ static atomic<int>)

    @classmethod
    def set_block_size(cls, block_size: int) -> None:
        """同步全局 Sequence block size。

        当前 Sequence 仍用类变量保存 block size；engine 初始化时必须把
        Config.kvcache_block_size 写入这里，避免 Sequence.num_blocks 与
        BlockManager/ModelRunner 的物理 KV block size 不一致。
        """

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        cls.block_size = int(block_size)

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
        image_token_id: int | None = None,
        image_token_count: int = 0,
        video_token_id: int | None = None,
        video_token_count: int = 0,
    ):
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if sampling_params is None:
            sampling_params = SamplingParams()
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be provided together")

        self.seq_id = next(Sequence.counter)          # 全局唯一ID: 0, 1, 2, ...
        self.block_size = int(type(self).block_size)  # 实例级快照，避免多 Config 运行时互相污染
        self.status = SequenceStatus.WAITING           # 初始状态=等待prefill
        self.token_ids = copy(token_ids)               # 浅拷贝prompt的token列表(值语义, 类似C++ vector拷贝)
        self.last_token = token_ids[-1]                # 最后一个token(序列化优化用)
        self.num_tokens = len(self.token_ids)           # 当前总token数(prompt+生成)
        self.num_prompt_tokens = len(token_ids)         # prompt token数(固定不变)
        self.num_cached_tokens = 0                      # 已在KV Cache中缓存的token数 (Prefix Cache)
        self.num_computed_tokens = 0                      # 已Prefill计算的token数 (Chunked Prefill)
        self.block_table = []                           # GPU 物理块映射表: [gpu_block_id_0, gpu_block_id_1, ...]
        self.cpu_block_table = []                       # Swap 后的 CPU 物理块映射表，不能污染 GPU block_table
        self.cpu_block_hashes = []                      # Swap 后每个 CPU block 对应的 prefix hash
        self.cpu_block_token_ids = []                   # Swap 后满块 token 副本，用于恢复 prefix-cache 索引
        # 从SamplingParams展开存储(避免序列化时携带整个SamplingParams对象)
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens    # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos    # 是否忽略EOS
        # VL 请求元数据。Prefill 需要 visual payload/position_ids；
        # Decode 只需要 rope_delta 延续 3D position_ids。
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.pixel_values_videos = pixel_values_videos
        self.video_grid_thw = video_grid_thw
        self.position_ids = position_ids
        self.rope_delta = rope_delta
        self.image_token_id = image_token_id
        self.image_token_count = image_token_count
        self.video_token_id = video_token_id
        self.video_token_count = video_token_count

    @classmethod
    def from_image_inputs(
        cls,
        inputs: ImageInputs,
        sampling_params: SamplingParams | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从图像预处理结果构造多模态 Sequence。"""

        return cls(
            inputs.token_ids,
            sampling_params,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            position_ids=position_ids,
            rope_delta=rope_delta,
            image_token_id=inputs.image_token_id,
            image_token_count=inputs.image_token_count,
        )

    @classmethod
    def from_single_image_inputs(
        cls,
        inputs: SingleImageInputs,
        sampling_params: SamplingParams | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从 P2.1 单图预处理结果构造多模态 Sequence。"""

        return cls.from_image_inputs(
            inputs,
            sampling_params,
            position_ids=position_ids,
            rope_delta=rope_delta,
        )

    @classmethod
    def from_video_inputs(
        cls,
        inputs: VideoInputs,
        sampling_params: SamplingParams | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从视频预处理结果构造多模态 Sequence。"""

        return cls(
            inputs.token_ids,
            sampling_params,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            position_ids=position_ids,
            rope_delta=rope_delta,
            video_token_id=inputs.video_token_id,
            video_token_count=inputs.video_token_count,
        )

    def __len__(self):           # len(seq) → 总token数
        return self.num_tokens

    def __getitem__(self, key):  # seq[0], seq[5:10] → 访问token_ids
        return self.token_ids[key]

    @property
    def is_finished(self):       # seq.is_finished → bool
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):  # 已生成的token数 = 总数 - prompt数
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):      # 切片取prompt部分: token_ids[:num_prompt]
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):  # 切片取生成部分: token_ids[num_prompt:]
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_multimodal(self) -> bool:
        """该序列是否携带或曾携带 VL 元数据。"""
        return (
            self.pixel_values is not None
            or self.image_grid_thw is not None
            or self.pixel_values_videos is not None
            or self.video_grid_thw is not None
            or self.position_ids is not None
            or self.rope_delta is not None
        )

    @property
    def num_cached_blocks(self):     # 已缓存的完整块数(整除)
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):            # 总共需要的块数(向上取整: (n+BS-1)//BS)
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):  # 最后一个块中的token数(可能不满)
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):              # 取第i个块对应的token子列表(用于hash匹配KV复用)
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    @property
    def is_prefill_finished(self) -> bool:
        """是否已完成所有 Prefill (Chunked Prefill 用)"""
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        """还需要 Prefill 多少 token"""
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    def append_token(self, token_id: int):  # 追加新生成的token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # === 跨进程序列化优化 ===
    # Python pickle序列化对象时自动调用这两个方法
    # 目的: 减少主进程→子进程的数据传输量

    def __getstate__(self):
        """序列化: 决定发送什么数据给子进程
        - Prefill(未生成): 发完整token_ids列表(子进程需要所有prompt做计算)
        - Decode(已生成): 只发last_token(1个int, 子进程已有KV Cache)
        - VL Decode: 不再发送 visual payload, 只保留 rope_delta
        """
        is_prefill_payload = self.num_completion_tokens == 0
        return {
            "block_size": self.block_size,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_computed_tokens": self.num_computed_tokens,
            "block_table": self.block_table,
            "cpu_block_table": self.cpu_block_table,
            "cpu_block_hashes": self.cpu_block_hashes,
            "cpu_block_token_ids": self.cpu_block_token_ids,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "payload": self.token_ids if is_prefill_payload else self.last_token,
            "is_prefill_payload": is_prefill_payload,
            "pixel_values": self.pixel_values if is_prefill_payload else None,
            "image_grid_thw": self.image_grid_thw if is_prefill_payload else None,
            "pixel_values_videos": self.pixel_values_videos if is_prefill_payload else None,
            "video_grid_thw": self.video_grid_thw if is_prefill_payload else None,
            "position_ids": self.position_ids if is_prefill_payload else None,
            "rope_delta": self.rope_delta,
            "image_token_id": self.image_token_id,
            "image_token_count": self.image_token_count,
            "video_token_id": self.video_token_id,
            "video_token_count": self.video_token_count,
        }

    def __setstate__(self, state):
        """反序列化: 子进程收到数据后恢复对象
        state就是__getstate__返回的那个元组
        state[:-1] = 前4个值, state[-1] = token_ids(list)或last_token(int)
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        if isinstance(state, dict):
            self.block_size = int(state.get("block_size", Sequence.block_size))
            self.temperature = state.get("temperature", 1.0)
            self.max_tokens = state.get("max_tokens", 0)
            self.ignore_eos = state.get("ignore_eos", False)
            self.num_tokens = state["num_tokens"]
            self.num_prompt_tokens = state["num_prompt_tokens"]
            self.num_cached_tokens = state["num_cached_tokens"]
            self.num_computed_tokens = state.get("num_computed_tokens", self.num_cached_tokens)
            self.block_table = state["block_table"]
            self.cpu_block_table = state.get("cpu_block_table", [])
            self.cpu_block_hashes = state.get("cpu_block_hashes", [])
            self.cpu_block_token_ids = [
                list(token_ids) for token_ids in state.get("cpu_block_token_ids", [])
            ]
            if state["is_prefill_payload"]:
                self.token_ids = state["payload"]
                self.last_token = self.token_ids[-1]
            else:
                self.last_token = state["payload"]
            self.pixel_values = state.get("pixel_values")
            self.image_grid_thw = state.get("image_grid_thw")
            self.pixel_values_videos = state.get("pixel_values_videos")
            self.video_grid_thw = state.get("video_grid_thw")
            self.position_ids = state.get("position_ids")
            self.rope_delta = state.get("rope_delta")
            self.image_token_id = state.get("image_token_id")
            self.image_token_count = state.get("image_token_count", 0)
            self.video_token_id = state.get("video_token_id")
            self.video_token_count = state.get("video_token_count", 0)
        else:
            self.block_size = int(Sequence.block_size)
            self.temperature = 1.0
            self.max_tokens = 0
            self.ignore_eos = False
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
            self.num_computed_tokens = self.num_cached_tokens
            self.cpu_block_table = []
            self.cpu_block_hashes = []
            self.cpu_block_token_ids = []
            if self.num_completion_tokens == 0:
                self.token_ids = state[-1]   # Prefill: state[-1]是完整列表
                self.last_token = self.token_ids[-1]
            else:
                self.last_token = state[-1]  # Decode: state[-1]是一个int
            self.pixel_values = None
            self.image_grid_thw = None
            self.pixel_values_videos = None
            self.video_grid_thw = None
            self.position_ids = None
            self.rope_delta = None
            self.image_token_id = None
            self.image_token_count = 0
            self.video_token_id = None
            self.video_token_count = 0
