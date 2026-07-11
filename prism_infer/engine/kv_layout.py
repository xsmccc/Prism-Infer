"""P6.4 visual KV physical layout contract。

该模块定义逻辑 token 位置与物理 KV 排布之间的稳定边界。KV 数据搬移、block
分配和 attention kernel 仍由各自模块实现；descriptor/plan 只携带经过验证的
长度、位置和页表变更，不依赖第三方压缩实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


KV_LAYOUT_SCHEMA_VERSION = 1
KV_LAYOUT_DENSE = "dense"
KV_LAYOUT_VISUAL_COMPACT = "visual_compact"


@dataclass
class KVCacheLayoutDescriptor:
    """一条 active sequence 的逻辑/物理 KV 状态。"""

    mode: str
    logical_context_len: int
    physical_kv_len: int
    prompt_logical_len: int
    compressed_prompt_kv_len: int
    retained_original_positions: tuple[int, ...]
    kv_dtype: str
    compression_record: dict[str, object]
    schema_version: int = KV_LAYOUT_SCHEMA_VERSION

    def validate(
        self,
        *,
        block_size: int,
        block_table: list[int],
        allow_pending_append: bool = False,
    ) -> None:
        """校验 descriptor 与当前物理页表的一致性。"""

        if self.schema_version != KV_LAYOUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported KV layout schema_version: {self.schema_version}"
            )
        if self.mode != KV_LAYOUT_VISUAL_COMPACT:
            raise ValueError(f"unsupported active KV layout mode: {self.mode!r}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if self.prompt_logical_len < 1:
            raise ValueError("prompt_logical_len must be positive")
        if self.logical_context_len < self.prompt_logical_len:
            raise ValueError("logical context cannot be shorter than prompt")
        if not 1 <= self.compressed_prompt_kv_len <= self.prompt_logical_len:
            raise ValueError("compressed prompt KV length is outside prompt bounds")
        generated_tokens = self.logical_context_len - self.prompt_logical_len
        expected_physical_len = self.compressed_prompt_kv_len + generated_tokens
        if self.physical_kv_len != expected_physical_len:
            raise ValueError(
                "physical KV length must equal compact prompt + generated tokens: "
                f"physical={self.physical_kv_len}, expected={expected_physical_len}"
            )
        retained = self.retained_original_positions
        if len(retained) != self.compressed_prompt_kv_len:
            raise ValueError(
                "retained position count must equal compressed prompt KV length"
            )
        if tuple(sorted(set(retained))) != retained:
            raise ValueError("retained original positions must be sorted and unique")
        if retained and (retained[0] < 0 or retained[-1] >= self.prompt_logical_len):
            raise ValueError("retained original positions are outside prompt")
        required_blocks = (self.physical_kv_len + block_size - 1) // block_size
        pending_append = (
            allow_pending_append
            and self.physical_kv_len % block_size == 1
            and len(block_table) == required_blocks - 1
        )
        if len(block_table) != required_blocks and not pending_append:
            raise ValueError(
                "compact block table length mismatch: "
                f"required={required_blocks}, actual={len(block_table)}"
            )
        if any(block_id < 0 for block_id in block_table):
            raise ValueError("compact block table contains a negative block id")
        if not self.kv_dtype:
            raise ValueError("compact KV dtype must be recorded")
        if not bool(self.compression_record.get("physical_compaction", False)):
            raise ValueError("compact compression record must mark physical_compaction")

    def append_generated_token(self) -> None:
        """为下一次 decode KV 写入同时推进逻辑与物理长度。"""

        self.logical_context_len += 1
        self.physical_kv_len += 1

    def to_record(self, *, block_table: list[int]) -> dict[str, object]:
        """生成可跨进程序列化和 benchmark 记录的 layout 数据。"""

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "logical_context_len": self.logical_context_len,
            "physical_kv_len": self.physical_kv_len,
            "prompt_logical_len": self.prompt_logical_len,
            "compressed_prompt_kv_len": self.compressed_prompt_kv_len,
            "retained_original_positions": list(self.retained_original_positions),
            "block_table": list(block_table),
            "kv_dtype": self.kv_dtype,
            "compression_record": dict(self.compression_record),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "KVCacheLayoutDescriptor":
        """从 Sequence pickle record 恢复 descriptor。"""

        return cls(
            schema_version=int(record["schema_version"]),
            mode=str(record["mode"]),
            logical_context_len=int(record["logical_context_len"]),
            physical_kv_len=int(record["physical_kv_len"]),
            prompt_logical_len=int(record["prompt_logical_len"]),
            compressed_prompt_kv_len=int(record["compressed_prompt_kv_len"]),
            retained_original_positions=tuple(
                int(position)
                for position in record["retained_original_positions"]
            ),
            kv_dtype=str(record["kv_dtype"]),
            compression_record=dict(record["compression_record"]),
        )


@dataclass(frozen=True)
class KVCompactionPlan:
    """GPU copy 完成前不可提交的 per-sequence compaction plan。"""

    seq_id: int
    logical_prompt_len: int
    physical_prompt_len: int
    old_block_table: tuple[int, ...]
    new_block_table: tuple[int, ...]
    released_block_ids: tuple[int, ...]
    retained_original_positions: tuple[int, ...]
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]
    kv_dtype: str
    compression_record: dict[str, object]

    def validate(self, *, block_size: int) -> None:
        """校验 copy mapping、页表缩减和 decision record。"""

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if self.logical_prompt_len < 1:
            raise ValueError("compaction logical prompt length must be positive")
        if self.physical_prompt_len != len(self.retained_original_positions):
            raise ValueError("physical prompt length must equal retained positions")
        if self.physical_prompt_len != len(self.source_slots):
            raise ValueError("physical prompt length must equal source slots")
        if self.physical_prompt_len != len(self.destination_slots):
            raise ValueError("physical prompt length must equal destination slots")
        if tuple(sorted(set(self.retained_original_positions))) != (
            self.retained_original_positions
        ):
            raise ValueError("plan retained positions must be sorted and unique")
        expected_blocks = (self.physical_prompt_len + block_size - 1) // block_size
        if len(self.new_block_table) != expected_blocks:
            raise ValueError("plan compact block count is inconsistent")
        if self.old_block_table[:expected_blocks] != self.new_block_table:
            raise ValueError("plan must compact into the existing page-table prefix")
        if self.old_block_table[expected_blocks:] != self.released_block_ids:
            raise ValueError("plan released blocks must be the old page-table suffix")
        expected_destinations = tuple(
            self.new_block_table[index // block_size] * block_size
            + index % block_size
            for index in range(self.physical_prompt_len)
        )
        if self.destination_slots != expected_destinations:
            raise ValueError("plan destination slots are not a dense physical tail")
        if not self.kv_dtype:
            raise ValueError("plan KV dtype must be recorded")
        if bool(self.compression_record.get("physical_compaction", False)):
            raise ValueError("uncommitted plan record must not claim compaction complete")
