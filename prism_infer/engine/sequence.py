from copy import copy  # copy模块: 浅拷贝(复制列表本身, 不复制元素)

import torch

from prism_infer.engine.vl_inputs import ImageInputs, SingleImageInputs, VideoInputs
from prism_infer.engine.kv_layout import KVCacheLayoutDescriptor
from prism_infer.engine.request import (
    RequestLifecycle,
    RequestState,
    SequenceStatus,
    validate_request_id,
)
from prism_infer.sampling_params import SamplingParams


class Sequence:
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        block_size: int,
        request_id: int,
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
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
        validate_request_id(request_id)
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be provided together")

        self.seq_id = request_id
        self.block_size = block_size
        self.lifecycle = RequestLifecycle(self.seq_id)
        self.token_ids = copy(token_ids)  # 浅拷贝prompt的token列表(值语义, 类似C++ vector拷贝)
        self.last_token = token_ids[-1]  # 最后一个token(序列化优化用)
        self.num_tokens = len(self.token_ids)  # 当前总token数(prompt+生成)
        self.num_prompt_tokens = len(token_ids)  # prompt token数(固定不变)
        self.num_cached_tokens = 0  # 已在KV Cache中缓存的token数 (Prefix Cache)
        self.num_computed_tokens = 0  # 已Prefill计算的token数 (Chunked Prefill)
        self.block_table = []  # GPU 物理块映射表: [gpu_block_id_0, gpu_block_id_1, ...]
        self.cpu_block_table = []  # Swap 后的 CPU 物理块映射表，不能污染 GPU block_table
        self.cpu_block_hashes = []  # Swap 后每个 CPU block 对应的 prefix hash
        self.cpu_block_token_ids = []  # Swap 后满块 token 副本，用于恢复 prefix-cache 索引
        # 从SamplingParams展开存储(避免序列化时携带整个SamplingParams对象)
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略EOS
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
        self.visual_pruning_decision_record: dict[str, object] | None = None
        self.kv_layout: KVCacheLayoutDescriptor | None = None

    @classmethod
    def from_image_inputs(
        cls,
        inputs: ImageInputs,
        sampling_params: SamplingParams | None = None,
        *,
        block_size: int,
        request_id: int,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从图像预处理结果构造多模态 Sequence。"""

        return cls(
            inputs.token_ids,
            sampling_params,
            block_size=block_size,
            request_id=request_id,
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
        block_size: int,
        request_id: int,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从 P2.1 单图预处理结果构造多模态 Sequence。"""

        return cls.from_image_inputs(
            inputs,
            sampling_params,
            block_size=block_size,
            request_id=request_id,
            position_ids=position_ids,
            rope_delta=rope_delta,
        )

    @classmethod
    def from_video_inputs(
        cls,
        inputs: VideoInputs,
        sampling_params: SamplingParams | None = None,
        *,
        block_size: int,
        request_id: int,
        position_ids: torch.Tensor | None = None,
        rope_delta: torch.Tensor | None = None,
    ) -> "Sequence":
        """从视频预处理结果构造多模态 Sequence。"""

        return cls(
            inputs.token_ids,
            sampling_params,
            block_size=block_size,
            request_id=request_id,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            position_ids=position_ids,
            rope_delta=rope_delta,
            video_token_id=inputs.video_token_id,
            video_token_count=inputs.video_token_count,
        )

    def __len__(self):  # len(seq) → 总token数
        return self.num_tokens

    def __getitem__(self, key):  # seq[0], seq[5:10] → 访问token_ids
        return self.token_ids[key]

    @property
    def is_finished(self):  # seq.is_finished → bool
        return self.status == SequenceStatus.FINISHED

    @property
    def status(self) -> RequestState:
        """Backwards-compatible lifecycle state view."""

        return self.lifecycle.state

    @status.setter
    def status(self, state: RequestState) -> None:
        # Older tests/integrations assign status directly.  Scheduler code uses
        # transition_to(), which validates the finite-state machine.
        if not hasattr(self, "lifecycle"):
            self.lifecycle = RequestLifecycle(self.seq_id, state=state)
        else:
            self.lifecycle.restore(state)

    def transition_to(self, state: RequestState, *, reason: str) -> None:
        """Apply one validated main-process request lifecycle transition."""

        self.lifecycle.transition(state, reason=reason)

    @property
    def num_completion_tokens(self):  # 已生成的token数 = 总数 - prompt数
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):  # 切片取prompt部分: token_ids[:num_prompt]
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):  # 切片取生成部分: token_ids[num_prompt:]
        return self.token_ids[self.num_prompt_tokens :]

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
    def vision_patch_count(self) -> int:
        """Raw vision-encoder patch rows owned by this request."""

        return sum(
            int(payload.shape[0])
            for payload in (self.pixel_values, self.pixel_values_videos)
            if payload is not None
        )

    def vision_patch_count_for_prefill_range(self, start: int, end: int) -> int:
        """Return payload patches materialized by one atomic prefill range."""

        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= self.num_prompt_tokens
        ):
            raise ValueError(
                "invalid prefill range for vision patch accounting: "
                f"[{start}, {end}) prompt_tokens={self.num_prompt_tokens}"
            )
        current_tokens = self.token_ids[start:end]
        patches = 0
        for modality, payload, token_id, expected_tokens in (
            (
                "image",
                self.pixel_values,
                self.image_token_id,
                self.image_token_count,
            ),
            (
                "video",
                self.pixel_values_videos,
                self.video_token_id,
                self.video_token_count,
            ),
        ):
            if payload is None:
                continue
            if token_id is None or expected_tokens <= 0:
                raise RuntimeError(f"{modality} payload is missing token identity metadata")
            observed_tokens = current_tokens.count(token_id)
            if observed_tokens not in (0, expected_tokens):
                raise ValueError(
                    f"prefill range splits {modality} token payload: "
                    f"seq={self.seq_id} range=[{start}, {end}) "
                    f"tokens={observed_tokens} expected={expected_tokens}"
                )
            if observed_tokens:
                patches += int(payload.shape[0])
        return patches

    @property
    def num_cached_blocks(self):  # 已缓存的完整块数(整除)
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):  # 总共需要的块数(向上取整: (n+BS-1)//BS)
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):  # 最后一个块中的token数(可能不满)
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def physical_kv_len(self) -> int:
        """当前 attention 实际可见的 KV token 数。"""

        return self.num_tokens if self.kv_layout is None else self.kv_layout.physical_kv_len

    @property
    def physical_num_blocks(self) -> int:
        """当前 physical KV tail 所需 page 数。"""

        return (self.physical_kv_len + self.block_size - 1) // self.block_size

    @property
    def physical_last_block_num_tokens(self) -> int:
        """当前 physical KV 最后一页的有效 token 数。"""

        return self.physical_kv_len - (self.physical_num_blocks - 1) * self.block_size

    @property
    def has_compact_kv_layout(self) -> bool:
        """是否已提交 visual KV physical compaction。"""

        return self.kv_layout is not None

    def install_kv_layout(self, layout: KVCacheLayoutDescriptor) -> None:
        """在 GPU copy 和 block 回收后提交 physical layout。"""

        if self.kv_layout is not None:
            raise RuntimeError("sequence KV layout is already compacted")
        expected_blocks = (layout.physical_kv_len + self.block_size - 1) // self.block_size
        if len(self.block_table) != expected_blocks:
            raise ValueError("layout install requires the final compact block table")
        layout.validate(block_size=self.block_size, block_table=self.block_table)
        if layout.logical_context_len != self.num_tokens:
            raise ValueError("layout logical length must match sequence length at install")
        self.kv_layout = layout

    def block(self, i):  # 取第i个块对应的token子列表(用于hash匹配KV复用)
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError(f"block index must be an integer, got {i!r}")
        if not 0 <= i < self.num_blocks:
            raise IndexError(f"block index {i} outside [0, {self.num_blocks})")
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

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
        if self.kv_layout is not None:
            self.kv_layout.append_generated_token()

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
            "seq_id": self.seq_id,
            "request_state": self.status,
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
            "visual_pruning_decision_record": self.visual_pruning_decision_record,
            "kv_layout": (
                None
                if self.kv_layout is None
                else self.kv_layout.to_record(block_table=self.block_table or self.cpu_block_table)
            ),
        }

    def __setstate__(self, state):
        """Restore an explicit request/page contract in a TP worker."""

        if not isinstance(state, dict):
            raise TypeError(
                "legacy Sequence payloads without explicit request/page state are not supported"
            )
        missing = {"seq_id", "block_size", "num_tokens", "payload"} - set(state)
        if missing:
            raise ValueError(
                "serialized Sequence is missing explicit fields: " + ", ".join(sorted(missing))
            )
        self.seq_id = validate_request_id(
            state["seq_id"],
            name="serialized seq_id",
        )
        request_state = state.get("request_state", RequestState.WAITING)
        self.lifecycle = RequestLifecycle(self.seq_id, state=request_state)
        serialized_block_size = state["block_size"]
        if (
            isinstance(serialized_block_size, bool)
            or not isinstance(serialized_block_size, int)
            or serialized_block_size <= 0
        ):
            raise ValueError(
                f"serialized block_size must be a positive integer, got {serialized_block_size!r}"
            )
        self.block_size = serialized_block_size
        self.temperature = state.get("temperature", 1.0)
        self.max_tokens = state.get("max_tokens", 0)
        self.ignore_eos = state.get("ignore_eos", False)
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.num_computed_tokens = state.get(
            "num_computed_tokens",
            self.num_cached_tokens,
        )
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
        self.visual_pruning_decision_record = state.get("visual_pruning_decision_record")
        layout_record = state.get("kv_layout")
        self.kv_layout = (
            None if layout_record is None else KVCacheLayoutDescriptor.from_record(layout_record)
        )
        if self.kv_layout is not None:
            self.kv_layout.validate(
                block_size=self.block_size,
                block_table=self.block_table or self.cpu_block_table,
                allow_pending_append=True,
            )
