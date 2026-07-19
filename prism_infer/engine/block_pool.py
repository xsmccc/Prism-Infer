"""Physical KV block-pool primitives.

The scheduler-facing block manager owns sequence page-table orchestration. This
module owns allocator invariants: free/used membership, reference counts,
prefix-hash indexing, and CPU swap-page ownership.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


NO_BLOCK_HASH = -1


@dataclass(slots=True)
class Block:
    """Metadata for one physical GPU KV-cache block."""

    block_id: int
    ref_count: int = 0
    hash: int = NO_BLOCK_HASH
    token_ids: list[int] = field(default_factory=list)

    def update(self, block_hash: int, token_ids: list[int]) -> None:
        self.hash = block_hash
        self.token_ids = list(token_ids)

    def activate(self) -> None:
        if self.ref_count != 0:
            raise RuntimeError(
                f"cannot activate referenced block {self.block_id}: ref_count={self.ref_count}"
            )
        self.ref_count = 1
        self.hash = NO_BLOCK_HASH
        self.token_ids.clear()

    def mark_free(self) -> None:
        if self.ref_count != 0:
            raise RuntimeError(
                f"cannot free referenced block {self.block_id}: ref_count={self.ref_count}"
            )
        self.hash = NO_BLOCK_HASH
        self.token_ids.clear()

    # Historical compatibility for internal callers.
    reset = activate


class GpuBlockPool:
    """Reference-counted GPU block allocator with a prefix-hash index."""

    def __init__(self, num_blocks: int):
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int):
            raise TypeError("num_blocks must be an integer")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.blocks = [Block(block_id) for block_id in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.free_block_id_set = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def capacity(self) -> int:
        return len(self.blocks)

    @property
    def free_count(self) -> int:
        return len(self.free_block_id_set)

    def _block(self, block_id: int) -> Block:
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise TypeError(f"GPU block id must be an integer, got {block_id!r}")
        if block_id < 0 or block_id >= self.capacity:
            raise IndexError(f"GPU block id {block_id} outside [0, {self.capacity})")
        return self.blocks[block_id]

    def remove_hash_index(self, block: Block) -> None:
        if block.hash == NO_BLOCK_HASH:
            return
        if self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    def allocate(self, block_id: int) -> Block:
        block = self._block(block_id)
        if block_id not in self.free_block_id_set:
            raise RuntimeError(f"GPU block {block_id} is not free")
        if block.ref_count != 0:
            raise RuntimeError(f"free GPU block {block_id} has ref_count={block.ref_count}")
        self.remove_hash_index(block)
        block.activate()
        self.free_block_id_set.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def allocate_free(self) -> Block:
        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_id_set:
                return self.allocate(block_id)
        raise RuntimeError("no free GPU KV-cache block available")

    def allocate_many(self, count: int) -> list[Block]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"block allocation count must be non-negative: {count!r}")
        if count > self.free_count:
            raise RuntimeError(
                "insufficient free GPU KV-cache blocks: "
                f"requested={count}, available={self.free_count}"
            )
        allocated: list[Block] = []
        try:
            for _ in range(count):
                allocated.append(self.allocate_free())
        except Exception:
            for block in reversed(allocated):
                self.release_reference(block.block_id)
            raise
        return allocated

    def retain(self, block_id: int) -> Block:
        block = self._block(block_id)
        if block_id not in self.used_block_ids or block.ref_count <= 0:
            raise RuntimeError(f"cannot retain unowned GPU block {block_id}")
        block.ref_count += 1
        return block

    def validate_owned(self, block_ids: Iterable[int]) -> tuple[int, ...]:
        owned = tuple(block_ids)
        blocks = tuple(self._block(block_id) for block_id in owned)
        if len(set(owned)) != len(owned):
            raise RuntimeError("GPU block table contains duplicate block ids")
        invalid = []
        for block_id, block in zip(owned, blocks):
            if block_id not in self.used_block_ids or block.ref_count <= 0:
                invalid.append(block_id)
        if invalid:
            raise RuntimeError(f"GPU block table contains unowned ids: {invalid}")
        return owned

    def release_reference(self, block_id: int) -> bool:
        """Release one owner and return whether the block became free."""

        block = self._block(block_id)
        if block_id not in self.used_block_ids or block.ref_count <= 0:
            raise RuntimeError(f"cannot release unowned GPU block {block_id}")
        block.ref_count -= 1
        if block.ref_count:
            return False
        self.deallocate(block_id)
        return True

    def deallocate(self, block_id: int) -> None:
        block = self._block(block_id)
        if block.ref_count != 0:
            raise RuntimeError(f"GPU block {block_id} still has ref_count={block.ref_count}")
        if block_id not in self.used_block_ids:
            raise RuntimeError(f"GPU block {block_id} is not allocated")
        self.remove_hash_index(block)
        block.mark_free()
        self.used_block_ids.remove(block_id)
        self.free_block_id_set.add(block_id)
        self.free_block_ids.append(block_id)

    def register_hash(
        self,
        block_id: int,
        block_hash: int,
        token_ids: list[int],
    ) -> None:
        block = self._block(block_id)
        if block_id not in self.used_block_ids or block.ref_count <= 0:
            raise RuntimeError(f"cannot hash unowned GPU block {block_id}")
        self.remove_hash_index(block)
        block.update(block_hash, token_ids)
        if block_hash != NO_BLOCK_HASH:
            self.hash_to_block_id[block_hash] = block_id

    def clear_hash(self, block_id: int) -> None:
        block = self._block(block_id)
        self.remove_hash_index(block)
        block.hash = NO_BLOCK_HASH
        block.token_ids.clear()

    def lookup(self, block_hash: int, token_ids: list[int]) -> Block | None:
        if block_hash == NO_BLOCK_HASH:
            return None
        block_id = self.hash_to_block_id.get(block_hash)
        if block_id is None:
            return None
        block = self._block(block_id)
        if block_id not in self.used_block_ids or block.ref_count <= 0:
            raise RuntimeError(f"prefix hash points to unowned GPU block {block_id}")
        return block if block.token_ids == token_ids else None


class CpuBlockPool:
    """Ownership-tracked fixed-capacity CPU swap-page allocator."""

    def __init__(self, num_blocks: int):
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int):
            raise TypeError("num_cpu_blocks must be an integer")
        if num_blocks < 0:
            raise ValueError(f"num_cpu_blocks must be non-negative, got {num_blocks}")
        self.capacity = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self._free_block_id_set = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def free_count(self) -> int:
        return len(self._free_block_id_set)

    def can_allocate(self, count: int) -> bool:
        return 0 <= count <= self.free_count

    def _validate_block_id(self, block_id: int) -> int:
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise TypeError(f"CPU block id must be an integer, got {block_id!r}")
        if not 0 <= block_id < self.capacity:
            raise IndexError(f"CPU block id {block_id} outside [0, {self.capacity})")
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"CPU block count must be non-negative: {count!r}")
        if not self.can_allocate(count):
            raise RuntimeError(
                "insufficient free CPU KV-cache blocks: "
                f"requested={count}, available={self.free_count}"
            )
        allocated: list[int] = []
        try:
            for _ in range(count):
                block_id = self._validate_block_id(self.free_block_ids.popleft())
                if block_id not in self._free_block_id_set:
                    raise RuntimeError(f"CPU free-list corruption at block {block_id}")
                self._free_block_id_set.remove(block_id)
                self.used_block_ids.add(block_id)
                allocated.append(block_id)
        except Exception:
            for block_id in reversed(allocated):
                self.used_block_ids.remove(block_id)
                self._free_block_id_set.add(block_id)
                self.free_block_ids.appendleft(block_id)
            raise
        return allocated

    def validate_owned(self, block_ids: Iterable[int]) -> tuple[int, ...]:
        owned = tuple(block_ids)
        for block_id in owned:
            self._validate_block_id(block_id)
        if len(set(owned)) != len(owned):
            raise RuntimeError("CPU block table contains duplicate block ids")
        invalid = [block_id for block_id in owned if block_id not in self.used_block_ids]
        if invalid:
            raise RuntimeError(f"CPU block table contains unowned ids: {invalid}")
        return owned

    def release_many(self, block_ids: Iterable[int]) -> None:
        owned = self.validate_owned(block_ids)
        for block_id in owned:
            self.used_block_ids.remove(block_id)
            self._free_block_id_set.add(block_id)
            self.free_block_ids.append(block_id)


__all__ = ["Block", "CpuBlockPool", "GpuBlockPool", "NO_BLOCK_HASH"]
