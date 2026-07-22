"""Host-side request state to tensor-only model input preparation.

This module is the boundary between mutable ``Sequence`` objects and the
immutable tensors consumed by eager, compile, and CUDA Graph backends.  It
owns no model or KV-cache storage; its only persistent inputs are the validated
runtime configuration and the KV page size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from prism_infer.config import Config
from prism_infer.engine.compression import build_compression_metadata
from prism_infer.engine.contracts import DeviceModelInputs, PrefillSlice, PreparedModelInputs
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.visual_pruning import (
    build_retained_slot_mapping,
    build_runtime_visual_token_scorer,
)
from prism_infer.models.qwen3_vl_architecture import (
    MROPE_AXIS_COUNT,
    MROPE_POSITION_TENSOR_RANK,
)
from prism_infer.observability import build_trace_metadata, profile_region, register_model_config
from prism_infer.utils.context import Context


@dataclass(slots=True)
class _PrefillHostBatch:
    """CPU-side accumulators for one flattened variable-length prefill."""

    input_ids: list[int] = field(default_factory=list)
    text_positions: list[int] = field(default_factory=list)
    mrope_position_chunks: list[torch.Tensor] = field(default_factory=list)
    cu_seqlens_q: list[int] = field(default_factory=lambda: [0])
    cu_seqlens_k: list[int] = field(default_factory=lambda: [0])
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: list[int] = field(default_factory=list)
    pixel_value_chunks: list[torch.Tensor] = field(default_factory=list)
    image_grid_chunks: list[torch.Tensor] = field(default_factory=list)
    video_value_chunks: list[torch.Tensor] = field(default_factory=list)
    video_grid_chunks: list[torch.Tensor] = field(default_factory=list)


class ModelInputPreparer:
    """Prepare prefill/decode tensors without owning model execution state."""

    def __init__(self, config: Config, *, block_size: int) -> None:
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
        self.config = config
        self.block_size = block_size

    @staticmethod
    def _validate_batch(seqs: list[Sequence], *, phase: str) -> None:
        if not isinstance(seqs, list):
            raise TypeError(f"{phase} sequences must be a list")
        if not seqs:
            raise ValueError(f"cannot prepare an empty {phase} batch")
        if any(not isinstance(seq, Sequence) for seq in seqs):
            raise TypeError(f"{phase} batch must contain Sequence objects")
        swapped = [seq.seq_id for seq in seqs if seq.cpu_block_table]
        if swapped:
            raise RuntimeError(f"cannot prepare {phase} for swapped sequences: {swapped}")

    @staticmethod
    def _to_cuda_tensor(values: object, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

    @staticmethod
    def _position_tensor(
        text_positions: list[int],
        mrope_chunks: list[torch.Tensor],
        *,
        uses_mrope: bool,
    ) -> torch.Tensor:
        if uses_mrope:
            if not mrope_chunks:
                raise RuntimeError("M-RoPE batch produced no position chunks")
            return (
                torch.cat(mrope_chunks, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
            )
        return ModelInputPreparer._to_cuda_tensor(text_positions, dtype=torch.int64)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Pad GPU-resident page tables and transfer them to the active GPU."""

        self._validate_batch(seqs, phase="block-table")
        missing = [seq.seq_id for seq in seqs if not seq.block_table]
        if missing:
            raise RuntimeError(f"GPU block_table is missing for sequences: {missing}")
        max_blocks = max(len(seq.block_table) for seq in seqs)
        padded = [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs]
        return self._to_cuda_tensor(padded, dtype=torch.int32)

    @staticmethod
    def _normalize_prefill_slices(
        seqs: list[Sequence],
        prefill_slices: tuple[PrefillSlice, ...] | None,
    ) -> tuple[PrefillSlice, ...]:
        if prefill_slices is None:
            prefill_slices = tuple(
                PrefillSlice(
                    sequence_id=seq.seq_id,
                    token_start=max(seq.num_cached_tokens, seq.num_computed_tokens),
                    token_end=seq.num_prompt_tokens,
                )
                for seq in seqs
            )
        elif not isinstance(prefill_slices, tuple):
            raise TypeError("prefill_slices must be an immutable tuple")
        if len(prefill_slices) != len(seqs):
            raise ValueError("prefill_slices must match prefill sequences")
        for seq, prefill_slice in zip(seqs, prefill_slices):
            if not isinstance(prefill_slice, PrefillSlice):
                raise TypeError("prefill_slices must contain PrefillSlice values")
            if prefill_slice.sequence_id != seq.seq_id:
                raise ValueError(
                    "prefill slice sequence id mismatch: "
                    f"{prefill_slice.sequence_id} != {seq.seq_id}"
                )
            if prefill_slice.token_end > seq.num_prompt_tokens:
                raise ValueError(
                    "prefill slice exceeds the prompt: "
                    f"seq={seq.seq_id} end={prefill_slice.token_end} "
                    f"prompt_tokens={seq.num_prompt_tokens}"
                )
        return prefill_slices

    @staticmethod
    def _validate_position_ids(seq: Sequence, *, token_end: int) -> None:
        position_ids = seq.position_ids
        if position_ids is None:
            return
        expected_prefix = (MROPE_AXIS_COUNT, 1)
        if (
            position_ids.ndim != MROPE_POSITION_TENSOR_RANK
            or tuple(position_ids.shape[:2]) != expected_prefix
        ):
            raise ValueError(
                "multimodal position_ids must have shape "
                f"[{MROPE_AXIS_COUNT}, 1, sequence_length], "
                f"got {tuple(position_ids.shape)} for seq={seq.seq_id}"
            )
        if position_ids.shape[2] < token_end:
            raise ValueError(
                "multimodal position_ids are shorter than the prefill slice: "
                f"seq={seq.seq_id} positions={position_ids.shape[2]} end={token_end}"
            )

    @staticmethod
    def _append_sequence_media(
        host: _PrefillHostBatch,
        seq: Sequence,
        current_tokens: list[int],
    ) -> None:
        media = (
            (
                "image",
                seq.pixel_values,
                seq.image_grid_thw,
                seq.image_token_id,
                seq.image_token_count,
                host.pixel_value_chunks,
                host.image_grid_chunks,
            ),
            (
                "video",
                seq.pixel_values_videos,
                seq.video_grid_thw,
                seq.video_token_id,
                seq.video_token_count,
                host.video_value_chunks,
                host.video_grid_chunks,
            ),
        )
        for (
            modality,
            payload,
            grid,
            token_id,
            expected_tokens,
            payload_chunks,
            grid_chunks,
        ) in media:
            if payload is None:
                continue
            if grid is None:
                raise RuntimeError(
                    f"{modality} payload is missing grid metadata for seq={seq.seq_id}"
                )
            if token_id is None or expected_tokens <= 0:
                raise RuntimeError(
                    f"{modality} payload is missing token identity metadata for seq={seq.seq_id}"
                )
            observed_tokens = current_tokens.count(token_id)
            if observed_tokens not in (0, expected_tokens):
                raise ValueError(
                    f"chunk boundary splits {modality} token payload: "
                    f"seq={seq.seq_id} chunk_tokens={observed_tokens} "
                    f"expected={expected_tokens}"
                )
            if observed_tokens:
                payload_chunks.append(payload)
                grid_chunks.append(grid)

    def _append_prefill_sequence(
        self,
        host: _PrefillHostBatch,
        seq: Sequence,
        prefill_slice: PrefillSlice,
        *,
        uses_mrope: bool,
    ) -> None:
        query_start = prefill_slice.token_start
        token_end = prefill_slice.token_end
        current_tokens = seq[query_start:token_end]
        host.input_ids.extend(current_tokens)

        if uses_mrope:
            self._validate_position_ids(seq, token_end=token_end)
            if seq.position_ids is None:
                text_positions = torch.arange(query_start, token_end, dtype=torch.long)
                host.mrope_position_chunks.append(
                    text_positions.view(1, -1).expand(MROPE_AXIS_COUNT, -1)
                )
            else:
                host.mrope_position_chunks.append(seq.position_ids[:, 0, query_start:token_end])
            self._append_sequence_media(host, seq, current_tokens)
        else:
            host.text_positions.extend(range(query_start, token_end))

        query_length = token_end - query_start
        key_length = token_end
        host.cu_seqlens_q.append(host.cu_seqlens_q[-1] + query_length)
        host.cu_seqlens_k.append(host.cu_seqlens_k[-1] + key_length)
        host.max_seqlen_q = max(query_length, host.max_seqlen_q)
        host.max_seqlen_k = max(key_length, host.max_seqlen_k)

        if not seq.block_table:  # warmup does not own KV pages
            return
        required_blocks = (token_end + self.block_size - 1) // self.block_size
        if len(seq.block_table) < required_blocks:
            raise RuntimeError(
                "prefill block_table does not cover scheduled tokens: "
                f"seq={seq.seq_id} blocks={len(seq.block_table)} required={required_blocks}"
            )
        for token_index in range(query_start, token_end):
            block_index, block_offset = divmod(token_index, self.block_size)
            host.slot_mapping.append(seq.block_table[block_index] * self.block_size + block_offset)

    @staticmethod
    def _multimodal_inputs(host: _PrefillHostBatch) -> dict[str, torch.Tensor | None]:
        def concatenate(chunks: list[torch.Tensor]) -> torch.Tensor | None:
            if not chunks:
                return None
            return torch.cat(chunks, dim=0).pin_memory().cuda(non_blocking=True)

        return {
            "pixel_values": concatenate(host.pixel_value_chunks),
            "image_grid_thw": concatenate(host.image_grid_chunks),
            "pixel_values_videos": concatenate(host.video_value_chunks),
            "video_grid_thw": concatenate(host.video_grid_chunks),
        }

    def _visual_pruning_scorer(
        self,
        seqs: list[Sequence],
        prefill_slices: tuple[PrefillSlice, ...],
        *,
        has_visual_payload: bool,
        compression_metadata,
    ):
        pruning_config = compression_metadata.visual_pruning_config
        enabled = (
            pruning_config is not None
            and pruning_config.get("strategy") == "attention"
            and compression_metadata.enabled
            and compression_metadata.total_visual_tokens > 0
            and has_visual_payload
        )
        if not enabled:
            return None
        incomplete = [
            prefill_slice.sequence_id
            for seq, prefill_slice in zip(seqs, prefill_slices)
            if prefill_slice.token_start != 0 or prefill_slice.token_end != seq.num_prompt_tokens
        ]
        if incomplete:
            raise RuntimeError(
                "attention-based visual pruning requires one complete "
                f"prefill slice; chunked requests={incomplete}"
            )
        hf_config = self.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        return build_runtime_visual_token_scorer(
            seqs,
            num_hidden_layers=int(text_config.num_hidden_layers),
            attention_last_n_layers=int(pruning_config["attention_last_n_layers"]),
        )

    def prepare_prefill(
        self,
        seqs: list[Sequence],
        *,
        prefill_slices: tuple[PrefillSlice, ...] | None = None,
    ) -> PreparedModelInputs:
        """Flatten a variable-length text/image/video prefill batch."""

        register_model_config(self.config)
        self._validate_batch(seqs, phase="prefill")
        slices = self._normalize_prefill_slices(seqs, prefill_slices)
        uses_mrope = any(seq.position_ids is not None for seq in seqs)
        missing_positions = [
            seq.seq_id
            for seq in seqs
            if (seq.pixel_values is not None or seq.pixel_values_videos is not None)
            and seq.position_ids is None
        ]
        if missing_positions:
            raise RuntimeError(
                "multimodal prefill requires model-specific position_ids: "
                f"sequences={missing_positions}"
            )

        host = _PrefillHostBatch()
        for seq, prefill_slice in zip(seqs, slices):
            self._append_prefill_sequence(
                host,
                seq,
                prefill_slice,
                uses_mrope=uses_mrope,
            )

        input_ids = self._to_cuda_tensor(host.input_ids, dtype=torch.int64)
        positions = self._position_tensor(
            host.text_positions,
            host.mrope_position_chunks,
            uses_mrope=uses_mrope,
        )
        cu_seqlens_q = self._to_cuda_tensor(host.cu_seqlens_q, dtype=torch.int32)
        cu_seqlens_k = self._to_cuda_tensor(host.cu_seqlens_k, dtype=torch.int32)
        slot_mapping = self._to_cuda_tensor(host.slot_mapping, dtype=torch.int32)

        block_tables = None
        context_lens = None
        paged_prefill = host.cu_seqlens_k[-1] > host.cu_seqlens_q[-1]
        if paged_prefill:
            block_tables = self.prepare_block_tables(seqs)
            context_lens = self._to_cuda_tensor(
                [prefill_slice.token_end for prefill_slice in slices],
                dtype=torch.int32,
            )

        trace_metadata = build_trace_metadata(
            seqs,
            is_prefill=True,
            input_ids=input_ids,
            position_ids=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=self.block_size,
            query_ranges=[
                (prefill_slice.token_start, prefill_slice.token_end) for prefill_slice in slices
            ],
        )
        compression_metadata = build_compression_metadata(
            self.config,
            seqs,
            is_prefill=True,
        )
        visual_pruning_scorer = self._visual_pruning_scorer(
            seqs,
            slices,
            has_visual_payload=bool(host.pixel_value_chunks or host.video_value_chunks),
            compression_metadata=compression_metadata,
        )
        multimodal_inputs = self._multimodal_inputs(host)
        return PreparedModelInputs(
            model_inputs=DeviceModelInputs(
                input_ids=input_ids,
                position_ids=positions,
                **multimodal_inputs,
            ),
            attention_context=Context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=host.max_seqlen_q,
                max_seqlen_k=host.max_seqlen_k,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                trace_metadata=trace_metadata,
                compression_metadata=compression_metadata,
                visual_pruning_scorer=visual_pruning_scorer,
            ),
        )

    def prepare_decode(self, seqs: list[Sequence]) -> PreparedModelInputs:
        """Prepare one token per request using physical KV and logical M-RoPE state."""

        register_model_config(self.config)
        self._validate_batch(seqs, phase="decode")
        missing_tables = [seq.seq_id for seq in seqs if not seq.block_table]
        if missing_tables:
            raise RuntimeError(f"decode requires GPU block_table: sequences={missing_tables}")

        input_ids: list[int] = []
        text_positions: list[int] = []
        mrope_positions: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        logical_context_lens: list[int] = []
        uses_mrope = any(seq.rope_delta is not None for seq in seqs)

        for seq in seqs:
            input_ids.append(seq.last_token)
            logical_position = len(seq) - 1
            if uses_mrope:
                delta = int(seq.rope_delta.item()) if seq.rope_delta is not None else 0
                mrope_positions.append(logical_position + delta)
            else:
                text_positions.append(logical_position)
            context_lens.append(seq.physical_kv_len)
            logical_context_lens.append(len(seq))
            if seq.physical_last_block_num_tokens <= 0:
                raise RuntimeError(f"decode has no writable KV slot for seq={seq.seq_id}")
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.physical_last_block_num_tokens - 1
            )

        batch_size = len(seqs)
        position_values = (
            mrope_positions * MROPE_AXIS_COUNT if uses_mrope else text_positions
        )
        packed_model_inputs = self._to_cuda_tensor(
            [*input_ids, *position_values],
            dtype=torch.int64,
        )
        input_ids_tensor = packed_model_inputs[:batch_size]
        positions_flat = packed_model_inputs[batch_size:]
        positions_tensor = (
            positions_flat.view(MROPE_AXIS_COUNT, batch_size)
            if uses_mrope
            else positions_flat
        )

        max_blocks = max(len(seq.block_table) for seq in seqs)
        padded_block_tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in seqs
        ]
        packed_attention_metadata = self._to_cuda_tensor(
            [
                *slot_mapping,
                *context_lens,
                *logical_context_lens,
                max(context_lens),
                *(block_id for row in padded_block_tables for block_id in row),
            ],
            dtype=torch.int32,
        )
        metadata_offset = 0
        slot_mapping_tensor = packed_attention_metadata[
            metadata_offset : metadata_offset + batch_size
        ]
        metadata_offset += batch_size
        context_lens_tensor = packed_attention_metadata[
            metadata_offset : metadata_offset + batch_size
        ]
        metadata_offset += batch_size
        logical_context_lens_tensor = packed_attention_metadata[
            metadata_offset : metadata_offset + batch_size
        ]
        metadata_offset += batch_size
        decode_max_context_len = packed_attention_metadata[metadata_offset]
        metadata_offset += 1
        block_tables = packed_attention_metadata[metadata_offset:].view(
            batch_size,
            max_blocks,
        )

        trace_metadata = build_trace_metadata(
            seqs,
            is_prefill=False,
            input_ids=input_ids_tensor,
            position_ids=positions_tensor,
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables,
            context_lens=context_lens_tensor,
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

        return PreparedModelInputs(
            model_inputs=DeviceModelInputs(
                input_ids=input_ids_tensor,
                position_ids=positions_tensor,
            ),
            attention_context=Context(
                is_prefill=False,
                slot_mapping=slot_mapping_tensor,
                context_lens=context_lens_tensor,
                logical_context_lens=logical_context_lens_tensor,
                block_tables=block_tables,
                decode_max_context_len=decode_max_context_len,
                paged_decode_block_n=self.config.paged_decode_block_n,
                trace_metadata=trace_metadata,
                compression_metadata=compression_metadata,
                visual_pruning_slot_mappings=visual_pruning_slot_mappings,
            ),
        )


__all__ = ["ModelInputPreparer"]
