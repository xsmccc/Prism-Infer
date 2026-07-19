"""KV Cache trace 与视觉 token 行为分析工具。

本模块只在显式开启 trace 时生效，普通推理路径通过
`is_trace_enabled()` 的快速判断直接跳过。trace 记录采用 JSONL:

- 第一行是 `trace_header`，包含 schema、模型配置和用户 metadata。
- 后续每行是 `attention_layer`，记录某一步某一层的 Q/K/V、KV span、
  last-query/current-query attention mass 和 token importance。

核心模型、attention 和 KV cache 仍由 Prism-Infer 自实现；本模块只负责
观测与离线统计，不替代推理路径。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Iterator
import json
import math

import torch

from prism_infer.observability.kv_trace import install_kv_trace_provider


SCHEMA_VERSION = 1
_ACTIVE_SESSION: TraceSession | None = None


def _jsonable(value: Any) -> Any:
    """把 tensor/config 标量转换为 JSON 可序列化对象。"""

    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().tolist()
    elif isinstance(value, torch.dtype):
        result = str(value).replace("torch.", "")
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result = value
    elif isinstance(value, dict):
        result = {str(k): _jsonable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_jsonable(v) for v in value]
    else:
        result = str(value)
    return result


def _shape(tensor: torch.Tensor | None) -> list[int] | None:
    return None if tensor is None else list(tensor.shape)


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """返回 detached tensor 的轻量统计，不保留 GPU tensor 引用。"""

    if tensor is None:
        return None
    data = tensor.detach().float()
    if data.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "numel": 0,
            "mean": None,
            "std": None,
            "abs_max": None,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "abs_max": float(data.abs().max().item()),
    }


def _tensor_meta(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """只记录 tensor 元信息，避免扫描整块预分配 KV cache。"""

    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


@dataclass(frozen=True)
class TraceConfig:
    """KV trace 运行配置。

    output_path: JSONL 输出路径；为空时只在内存中保留 records。
    include_attention: 是否计算 last/current query 的精确 attention mass。
    top_k_tokens: 每条序列记录 top-k token importance。
    include_warmup: 是否记录 ModelRunner 初始化 warmup 的假输入。
    """

    output_path: str | None = None
    include_attention: bool = True
    top_k_tokens: int = 8
    include_warmup: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "output_path": self.output_path,
                "include_attention": self.include_attention,
                "top_k_tokens": self.top_k_tokens,
                "include_warmup": self.include_warmup,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class TokenSpan:
    """一段连续同模态 token。

    start/end 是序列内 local token 下标，flat_start/flat_end 是本次
    prefill/decode 输入中的 flatten 下标；decode 中只有当前 query token
    可能有 flat 坐标。
    """

    modality: str
    start: int
    end: int
    index: int
    flat_start: int | None = None
    flat_end: int | None = None

    @property
    def token_count(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "flat_start": self.flat_start,
            "flat_end": self.flat_end,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class SequenceTraceInfo:
    """一次 batch 中单条 sequence 的 trace 元数据。"""

    seq_id: int
    prompt_len: int
    total_len: int
    query_start: int
    query_end: int
    flat_start: int | None
    flat_end: int | None
    block_table: list[int]
    image_token_id: int | None
    image_token_count: int
    image_grid_thw: list[list[int]] | None
    video_token_id: int | None
    video_token_count: int
    video_grid_thw: list[list[int]] | None
    spans: list[TokenSpan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "prompt_len": self.prompt_len,
            "total_len": self.total_len,
            "query_start": self.query_start,
            "query_end": self.query_end,
            "flat_start": self.flat_start,
            "flat_end": self.flat_end,
            "block_table": self.block_table,
            "image_token_id": self.image_token_id,
            "image_token_count": self.image_token_count,
            "image_grid_thw": self.image_grid_thw,
            "video_token_id": self.video_token_id,
            "video_token_count": self.video_token_count,
            "video_grid_thw": self.video_grid_thw,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class TraceMetadata:
    """ModelRunner 写入 Context、Attention 读取的 batch 级元数据。"""

    step_id: int
    phase: str
    batch_size: int
    input_ids_shape: list[int]
    position_ids_shape: list[int]
    slot_mapping_shape: list[int] | None
    block_tables_shape: list[int] | None
    context_lens: list[int] | None
    sequences: list[SequenceTraceInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "phase": self.phase,
            "batch_size": self.batch_size,
            "input_ids_shape": self.input_ids_shape,
            "position_ids_shape": self.position_ids_shape,
            "slot_mapping_shape": self.slot_mapping_shape,
            "block_tables_shape": self.block_tables_shape,
            "context_lens": self.context_lens,
            "sequences": [seq.to_dict() for seq in self.sequences],
        }


class TraceSession:
    """一次 KV trace 会话，负责收集 records 并写出 JSONL。"""

    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self.records: list[dict[str, Any]] = []
        self.created_at = time()
        self._step_id = 0
        self.model_config: dict[str, Any] = {}

    def next_step_id(self) -> int:
        step_id = self._step_id
        self._step_id += 1
        return step_id

    def set_model_config(self, config: Any) -> None:
        """从 engine Config 中提取可复现的模型与 KV cache 配置。"""

        hf_config = getattr(config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        vision_config = getattr(hf_config, "vision_config", None)
        self.model_config = _jsonable(
            {
                "model_path": getattr(config, "model", None),
                "model_type": getattr(hf_config, "model_type", None),
                "text": {
                    "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
                    "hidden_size": getattr(text_config, "hidden_size", None),
                    "num_attention_heads": getattr(text_config, "num_attention_heads", None),
                    "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
                    "head_dim": getattr(text_config, "head_dim", None),
                    "vocab_size": getattr(text_config, "vocab_size", None),
                },
                "vision": {
                    "hidden_size": getattr(vision_config, "hidden_size", None),
                    "patch_size": getattr(vision_config, "patch_size", None),
                    "spatial_merge_size": getattr(vision_config, "spatial_merge_size", None),
                },
                "engine": {
                    "kvcache_block_size": getattr(config, "kvcache_block_size", None),
                    "max_model_len": getattr(config, "max_model_len", None),
                    "max_num_batched_tokens": getattr(config, "max_num_batched_tokens", None),
                    "max_num_seqs": getattr(config, "max_num_seqs", None),
                    "tensor_parallel_size": getattr(config, "tensor_parallel_size", None),
                    "enforce_eager": getattr(config, "enforce_eager", None),
                },
            }
        )

    def add_record(self, record: dict[str, Any]) -> None:
        self.records.append(_jsonable(record))

    def header(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "trace_header",
            "created_at": self.created_at,
            "trace_config": self.config.to_dict(),
            "model_config": self.model_config,
        }

    def flush(self, output_path: str | None = None) -> None:
        path_value = output_path or self.config.output_path
        if path_value is None:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self.header(), ensure_ascii=False, sort_keys=True) + "\n")
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def kv_trace(
    output_path: str | Path | None = None,
    *,
    include_attention: bool = True,
    top_k_tokens: int = 8,
    include_warmup: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Iterator[TraceSession]:
    """开启一次 KV trace 会话。

    用法:

    ```python
    with kv_trace("trace.jsonl", metadata={"case": "single_image"}):
        llm.generate_vl(...)
    ```
    """

    global _ACTIVE_SESSION
    previous = _ACTIVE_SESSION
    session = TraceSession(
        TraceConfig(
            output_path=str(output_path) if output_path is not None else None,
            include_attention=include_attention,
            top_k_tokens=top_k_tokens,
            include_warmup=include_warmup,
            metadata=metadata or {},
        )
    )
    _ACTIVE_SESSION = session
    try:
        yield session
    finally:
        session.flush()
        _ACTIVE_SESSION = previous


def get_trace_session() -> TraceSession | None:
    return _ACTIVE_SESSION


def is_trace_enabled() -> bool:
    return _ACTIVE_SESSION is not None


def register_model_config(config: Any) -> None:
    session = get_trace_session()
    if session is not None:
        session.set_model_config(config)


def locate_token_spans(
    token_ids: list[int],
    *,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> list[TokenSpan]:
    """扫描 token ids，返回 text/image/video 连续 span。"""

    if not token_ids:
        return []

    counters = {"text": 0, "image": 0, "video": 0}

    def modality(token_id: int) -> str:
        if image_token_id is not None and token_id == image_token_id:
            return "image"
        if video_token_id is not None and token_id == video_token_id:
            return "video"
        return "text"

    spans: list[TokenSpan] = []
    start = 0
    current = modality(token_ids[0])
    for idx, token_id in enumerate(token_ids[1:], start=1):
        next_modality = modality(token_id)
        if next_modality == current:
            continue
        span_index = counters[current]
        counters[current] += 1
        spans.append(TokenSpan(current, start, idx, span_index))
        start = idx
        current = next_modality

    span_index = counters[current]
    spans.append(TokenSpan(current, start, len(token_ids), span_index))
    return spans


def _grid_to_list(grid: torch.Tensor | None) -> list[list[int]] | None:
    if grid is None:
        return None
    values = grid.detach().cpu().to(torch.int64).tolist()
    return [[int(x) for x in row] for row in values]


def _with_flat_coordinates(
    spans: list[TokenSpan],
    *,
    query_start: int,
    query_end: int,
    flat_start: int | None,
) -> list[TokenSpan]:
    result = []
    for span in spans:
        overlap_start = max(span.start, query_start)
        overlap_end = min(span.end, query_end)
        if flat_start is not None and overlap_start < overlap_end:
            span_flat_start = flat_start + overlap_start - query_start
            span_flat_end = flat_start + overlap_end - query_start
        else:
            span_flat_start = None
            span_flat_end = None
        result.append(
            TokenSpan(
                span.modality,
                span.start,
                span.end,
                span.index,
                span_flat_start,
                span_flat_end,
            )
        )
    return result


def build_trace_metadata(
    seqs: list[Any],
    *,
    is_prefill: bool,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    block_tables: torch.Tensor | None,
    context_lens: torch.Tensor | None,
    block_size: int,
    query_ranges: list[tuple[int, int]] | None = None,
) -> TraceMetadata | None:
    """从 ModelRunner 当前 batch 构造 trace metadata。

    warmup 阶段没有真实 block_table，默认不记录，避免污染分析报告。
    """

    session = get_trace_session()
    if session is None:
        return None
    is_warmup = any(not getattr(seq, "block_table", []) for seq in seqs)
    if is_warmup and not session.config.include_warmup:
        return None

    phase = "prefill" if is_prefill else "decode"
    if query_ranges is not None and len(query_ranges) != len(seqs):
        raise ValueError("query_ranges must match traced sequences")
    flat_cursor = 0
    sequence_infos: list[SequenceTraceInfo] = []
    for seq_idx, seq in enumerate(seqs):
        token_ids = list(getattr(seq, "token_ids", []))
        if not token_ids and hasattr(seq, "last_token"):
            token_ids = [int(seq.last_token)]
        total_len = int(getattr(seq, "num_tokens", len(token_ids)))
        prompt_len = int(getattr(seq, "num_prompt_tokens", total_len))
        if is_prefill:
            if query_ranges is None:
                query_start = int(getattr(seq, "num_cached_tokens", 0))
                query_end = total_len
            else:
                query_start, query_end = query_ranges[seq_idx]
                if not 0 <= query_start < query_end <= prompt_len:
                    raise ValueError(
                        "invalid trace prefill query range: "
                        f"seq={getattr(seq, 'seq_id', seq_idx)} "
                        f"range=[{query_start}, {query_end}) "
                        f"prompt_len={prompt_len}"
                    )
                total_len = query_end
            seq_flat_start = flat_cursor
            seq_flat_end = flat_cursor + max(0, query_end - query_start)
            flat_cursor = seq_flat_end
        else:
            query_start = max(0, total_len - 1)
            query_end = total_len
            seq_flat_start = seq_idx
            seq_flat_end = seq_idx + 1

        spans = locate_token_spans(
            token_ids,
            image_token_id=getattr(seq, "image_token_id", None),
            video_token_id=getattr(seq, "video_token_id", None),
        )
        spans = _with_flat_coordinates(
            spans,
            query_start=query_start,
            query_end=query_end,
            flat_start=seq_flat_start,
        )
        sequence_infos.append(
            SequenceTraceInfo(
                seq_id=int(getattr(seq, "seq_id", seq_idx)),
                prompt_len=prompt_len,
                total_len=total_len,
                query_start=query_start,
                query_end=query_end,
                flat_start=seq_flat_start,
                flat_end=seq_flat_end,
                block_table=[int(x) for x in getattr(seq, "block_table", [])],
                image_token_id=getattr(seq, "image_token_id", None),
                image_token_count=int(getattr(seq, "image_token_count", 0)),
                image_grid_thw=_grid_to_list(getattr(seq, "image_grid_thw", None)),
                video_token_id=getattr(seq, "video_token_id", None),
                video_token_count=int(getattr(seq, "video_token_count", 0)),
                video_grid_thw=_grid_to_list(getattr(seq, "video_grid_thw", None)),
                spans=spans,
            )
        )

    context_lens_list = None
    if context_lens is not None:
        context_lens_list = [int(x) for x in context_lens.detach().cpu().tolist()]
    return TraceMetadata(
        step_id=session.next_step_id(),
        phase=phase,
        batch_size=len(seqs),
        input_ids_shape=list(input_ids.shape),
        position_ids_shape=list(position_ids.shape),
        slot_mapping_shape=_shape(slot_mapping),
        block_tables_shape=_shape(block_tables),
        context_lens=context_lens_list,
        sequences=sequence_infos,
    )


def _expand_kv_heads(
    values: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return values
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads, got {num_heads}/{num_kv_heads}"
        )
    return values.repeat_interleave(num_heads // num_kv_heads, dim=1)


def _modality_indices(
    seq: SequenceTraceInfo,
    *,
    modality: str,
    length: int,
) -> list[int]:
    indices: list[int] = []
    for span in seq.spans:
        if span.modality != modality:
            continue
        start = max(0, span.start)
        end = min(length, span.end)
        if start < end:
            indices.extend(range(start, end))
    return indices


def _top_tokens(
    probs: torch.Tensor,
    *,
    token_offset: int,
    candidate_indices: list[int] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or probs.numel() == 0:
        return []
    if candidate_indices is None:
        local_indices = torch.arange(probs.numel(), device=probs.device)
        values = probs
    elif candidate_indices:
        local_indices = torch.tensor(candidate_indices, device=probs.device, dtype=torch.long)
        values = probs.index_select(0, local_indices)
    else:
        return []
    k = min(top_k, int(values.numel()))
    scores, order = torch.topk(values, k=k)
    selected = local_indices.index_select(0, order)
    return [
        {
            "token_index": int(token_offset + selected[i].item()),
            "score": float(scores[i].item()),
        }
        for i in range(k)
    ]


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """按最后一维计算 Shannon entropy，输入需为概率分布。"""

    safe_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
    return -(probs * safe_probs.log()).sum(dim=-1)


def _normalized_entropy(entropy: torch.Tensor, token_count: int) -> torch.Tensor:
    """把 entropy 归一化到约 [0, 1]；token_count<=1 时定义为 0。"""

    if token_count <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(token_count)


def _conditional_entropy_for_indices(
    probs: torch.Tensor,
    indices: list[int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """计算候选 token 内的条件 entropy。

    返回每个 head 的 `(entropy, normalized_entropy)`；没有候选 token 时返回
    `(None, None)`。
    """

    if not indices:
        return None, None
    index = torch.tensor(indices, device=probs.device, dtype=torch.long)
    selected = probs.index_select(-1, index)
    mass = selected.sum(dim=-1, keepdim=True)
    normalized = selected / mass.clamp_min(torch.finfo(selected.dtype).tiny)
    entropy = _entropy_from_probs(normalized)
    return entropy, _normalized_entropy(entropy, len(indices))


def _span_kv_stats(
    metadata: TraceMetadata,
    k: torch.Tensor,
    v: torch.Tensor,
) -> list[dict[str, Any]]:
    k_norm = k.detach().float().norm(dim=-1)
    v_norm = v.detach().float().norm(dim=-1)
    stats: list[dict[str, Any]] = []
    for seq in metadata.sequences:
        for span in seq.spans:
            if span.flat_start is None or span.flat_end is None:
                continue
            if span.flat_end <= span.flat_start:
                continue
            k_slice = k_norm[span.flat_start : span.flat_end]
            v_slice = v_norm[span.flat_start : span.flat_end]
            if k_slice.numel() == 0 or v_slice.numel() == 0:
                continue
            stats.append(
                {
                    "seq_id": seq.seq_id,
                    "modality": span.modality,
                    "span_index": span.index,
                    "start": span.start,
                    "end": span.end,
                    "token_count": span.flat_end - span.flat_start,
                    "k_norm_mean": float(k_slice.mean().item()),
                    "k_norm_std": float(k_slice.std(unbiased=False).item()),
                    "v_norm_mean": float(v_slice.mean().item()),
                    "v_norm_std": float(v_slice.std(unbiased=False).item()),
                    "k_norm_by_head": _to_float_list(k_slice.mean(dim=0)),
                    "v_norm_by_head": _to_float_list(v_slice.mean(dim=0)),
                }
            )
    return stats


def _head_norm_stats(
    metadata: TraceMetadata,
    k: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, Any]:
    k_norm = k.detach().float().norm(dim=-1)
    v_norm = v.detach().float().norm(dim=-1)
    result: dict[str, Any] = {
        "current_tokens": {
            "k_norm_mean_by_head": _to_float_list(k_norm.mean(dim=0)),
            "v_norm_mean_by_head": _to_float_list(v_norm.mean(dim=0)),
        },
        "by_modality": {},
    }
    for modality in ("text", "image", "video"):
        ranges = []
        for seq in metadata.sequences:
            for span in seq.spans:
                if span.modality != modality:
                    continue
                if span.flat_start is not None and span.flat_end is not None:
                    ranges.extend(range(span.flat_start, span.flat_end))
        if not ranges:
            continue
        index = torch.tensor(ranges, device=k_norm.device, dtype=torch.long)
        k_mod = k_norm.index_select(0, index)
        v_mod = v_norm.index_select(0, index)
        result["by_modality"][modality] = {
            "token_count": int(index.numel()),
            "k_norm_mean": float(k_mod.mean().item()),
            "v_norm_mean": float(v_mod.mean().item()),
            "k_norm_mean_by_head": _to_float_list(k_mod.mean(dim=0)),
            "v_norm_mean_by_head": _to_float_list(v_mod.mean(dim=0)),
        }
    return result


def _attention_for_prefill(
    metadata: TraceMetadata,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    top_k_tokens: int,
) -> dict[str, Any]:
    q_data = q.detach().float()
    k_data = k.detach().float()
    sequence_stats = []
    for seq in metadata.sequences:
        if seq.flat_start is None or seq.flat_end is None or seq.flat_end <= seq.flat_start:
            continue
        q_last = q_data[seq.flat_end - 1]
        keys = k_data[seq.flat_start : seq.flat_end]
        keys = _expand_kv_heads(keys, num_heads=num_heads, num_kv_heads=num_kv_heads)
        # q_last: [heads, dim], keys: [tokens, heads, dim] -> scores: [heads, tokens]
        scores = torch.einsum("hd,thd->ht", q_last, keys) * scale
        probs = torch.softmax(scores, dim=-1)
        mean_probs = probs.mean(dim=0)
        span_masses = []
        visual_head_mass = torch.zeros(num_heads, device=q.device, dtype=torch.float32)
        text_head_mass = torch.zeros_like(visual_head_mass)
        for span in seq.spans:
            start = max(span.start, seq.query_start) - seq.query_start
            end = min(span.end, seq.query_end) - seq.query_start
            if end <= start:
                continue
            mass = probs[:, start:end].sum(dim=-1)
            if span.modality in {"image", "video"}:
                visual_head_mass += mass
            elif span.modality == "text":
                text_head_mass += mass
            span_masses.append(
                {
                    "modality": span.modality,
                    "span_index": span.index,
                    "start": span.start,
                    "end": span.end,
                    "mass_mean": float(mass.mean().item()),
                    "mass_std": float(mass.std(unbiased=False).item()),
                    "mass_by_head": _to_float_list(mass),
                }
            )
        visual_indices = []
        for modality in ("image", "video"):
            visual_indices.extend(
                idx - seq.query_start
                for idx in _modality_indices(seq, modality=modality, length=seq.query_end)
                if seq.query_start <= idx < seq.query_end
            )
        entropy = _entropy_from_probs(probs)
        entropy_norm = _normalized_entropy(entropy, probs.shape[-1])
        visual_entropy, visual_entropy_norm = _conditional_entropy_for_indices(
            probs,
            visual_indices,
        )
        sequence_stats.append(
            {
                "seq_id": seq.seq_id,
                "query_token_index": seq.query_end - 1,
                "key_token_count": seq.query_end - seq.query_start,
                "attention_entropy_mean": float(entropy.mean().item()),
                "attention_entropy_std": float(entropy.std(unbiased=False).item()),
                "attention_entropy_by_head": _to_float_list(entropy),
                "attention_entropy_normalized_mean": float(entropy_norm.mean().item()),
                "attention_entropy_normalized_by_head": _to_float_list(entropy_norm),
                "visual_attention_entropy_mean": (
                    None if visual_entropy is None else float(visual_entropy.mean().item())
                ),
                "visual_attention_entropy_normalized_mean": (
                    None
                    if visual_entropy_norm is None
                    else float(visual_entropy_norm.mean().item())
                ),
                "visual_attention_entropy_by_head": (
                    [] if visual_entropy is None else _to_float_list(visual_entropy)
                ),
                "visual_mass_mean": float(visual_head_mass.mean().item()),
                "visual_mass_std": float(visual_head_mass.std(unbiased=False).item()),
                "text_mass_mean": float(text_head_mass.mean().item()),
                "text_mass_std": float(text_head_mass.std(unbiased=False).item()),
                "head_visual_mass": _to_float_list(visual_head_mass),
                "head_text_mass": _to_float_list(text_head_mass),
                "span_masses": span_masses,
                "top_tokens": _top_tokens(
                    mean_probs,
                    token_offset=seq.query_start,
                    candidate_indices=None,
                    top_k=top_k_tokens,
                ),
                "top_visual_tokens": _top_tokens(
                    mean_probs,
                    token_offset=seq.query_start,
                    candidate_indices=visual_indices,
                    top_k=top_k_tokens,
                ),
            }
        )
    return {
        "available": True,
        "kind": "prefill_last_query",
        "sequence_stats": sequence_stats,
    }


def _gather_paged_cache_tokens(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    context_len: int,
) -> torch.Tensor:
    block_size = cache.shape[1]
    pieces = []
    remaining = context_len
    for block_id in block_ids.detach().cpu().tolist():
        if remaining <= 0:
            break
        if block_id < 0:
            break
        take = min(block_size, remaining)
        pieces.append(cache[int(block_id), :take])
        remaining -= take
    if remaining != 0 or not pieces:
        raise RuntimeError(
            "invalid block table while tracing decode attention: "
            f"context_len={context_len}, remaining={remaining}"
        )
    return torch.cat(pieces, dim=0)


def _attention_for_decode(
    metadata: TraceMetadata,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    context: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    top_k_tokens: int,
) -> dict[str, Any]:
    if context.block_tables is None or context.context_lens is None or k_cache.numel() == 0:
        return {
            "available": False,
            "kind": "decode_current_query",
            "reason": "decode cache metadata is unavailable",
        }
    q_data = q.detach().float()
    sequence_stats = []
    for seq_order, seq in enumerate(metadata.sequences):
        context_len = int(context.context_lens[seq_order].item())
        keys = (
            _gather_paged_cache_tokens(
                k_cache,
                context.block_tables[seq_order],
                context_len,
            )
            .detach()
            .float()
        )
        keys = _expand_kv_heads(keys, num_heads=num_heads, num_kv_heads=num_kv_heads)
        scores = torch.einsum("hd,thd->ht", q_data[seq_order], keys) * scale
        probs = torch.softmax(scores, dim=-1)
        mean_probs = probs.mean(dim=0)
        span_masses = []
        visual_head_mass = torch.zeros(num_heads, device=q.device, dtype=torch.float32)
        text_head_mass = torch.zeros_like(visual_head_mass)
        for span in seq.spans:
            start = max(0, span.start)
            end = min(context_len, span.end)
            if end <= start:
                continue
            mass = probs[:, start:end].sum(dim=-1)
            if span.modality in {"image", "video"}:
                visual_head_mass += mass
            elif span.modality == "text":
                text_head_mass += mass
            span_masses.append(
                {
                    "modality": span.modality,
                    "span_index": span.index,
                    "start": span.start,
                    "end": span.end,
                    "mass_mean": float(mass.mean().item()),
                    "mass_std": float(mass.std(unbiased=False).item()),
                    "mass_by_head": _to_float_list(mass),
                }
            )
        visual_indices = []
        for modality in ("image", "video"):
            visual_indices.extend(
                idx for idx in _modality_indices(seq, modality=modality, length=context_len)
            )
        entropy = _entropy_from_probs(probs)
        entropy_norm = _normalized_entropy(entropy, probs.shape[-1])
        visual_entropy, visual_entropy_norm = _conditional_entropy_for_indices(
            probs,
            visual_indices,
        )
        sequence_stats.append(
            {
                "seq_id": seq.seq_id,
                "query_token_index": context_len - 1,
                "key_token_count": context_len,
                "attention_entropy_mean": float(entropy.mean().item()),
                "attention_entropy_std": float(entropy.std(unbiased=False).item()),
                "attention_entropy_by_head": _to_float_list(entropy),
                "attention_entropy_normalized_mean": float(entropy_norm.mean().item()),
                "attention_entropy_normalized_by_head": _to_float_list(entropy_norm),
                "visual_attention_entropy_mean": (
                    None if visual_entropy is None else float(visual_entropy.mean().item())
                ),
                "visual_attention_entropy_normalized_mean": (
                    None
                    if visual_entropy_norm is None
                    else float(visual_entropy_norm.mean().item())
                ),
                "visual_attention_entropy_by_head": (
                    [] if visual_entropy is None else _to_float_list(visual_entropy)
                ),
                "visual_mass_mean": float(visual_head_mass.mean().item()),
                "visual_mass_std": float(visual_head_mass.std(unbiased=False).item()),
                "text_mass_mean": float(text_head_mass.mean().item()),
                "text_mass_std": float(text_head_mass.std(unbiased=False).item()),
                "head_visual_mass": _to_float_list(visual_head_mass),
                "head_text_mass": _to_float_list(text_head_mass),
                "span_masses": span_masses,
                "top_tokens": _top_tokens(
                    mean_probs,
                    token_offset=0,
                    candidate_indices=None,
                    top_k=top_k_tokens,
                ),
                "top_visual_tokens": _top_tokens(
                    mean_probs,
                    token_offset=0,
                    candidate_indices=visual_indices,
                    top_k=top_k_tokens,
                ),
            }
        )
    return {
        "available": True,
        "kind": "decode_current_query",
        "sequence_stats": sequence_stats,
    }


def record_attention_layer(
    *,
    layer_id: int | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context: Any,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> None:
    """记录单层 attention/KV 行为。"""

    session = get_trace_session()
    metadata = getattr(context, "trace_metadata", None)
    if session is None or metadata is None:
        return
    if layer_id is None:
        return

    attention: dict[str, Any]
    if not session.config.include_attention:
        attention = {"available": False, "reason": "include_attention is false"}
    elif metadata.phase == "prefill":
        attention = _attention_for_prefill(
            metadata,
            q,
            k,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=scale,
            top_k_tokens=session.config.top_k_tokens,
        )
    else:
        attention = _attention_for_decode(
            metadata,
            q,
            k_cache,
            context,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=scale,
            top_k_tokens=session.config.top_k_tokens,
        )

    session.add_record(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "attention_layer",
            "step_id": metadata.step_id,
            "phase": metadata.phase,
            "layer_id": layer_id,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "scale": scale,
            "batch": metadata.to_dict(),
            "tensor_stats": {
                "q": _tensor_stats(q),
                "k": _tensor_stats(k),
                "v": _tensor_stats(v),
                "output": _tensor_stats(output),
                "k_cache": _tensor_meta(k_cache) if k_cache.numel() else None,
                "v_cache": _tensor_meta(v_cache) if v_cache.numel() else None,
            },
            "head_stats": _head_norm_stats(metadata, k, v),
            "span_stats": _span_kv_stats(metadata, k, v),
            "attention": attention,
        }
    )


def read_trace_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL trace 文件。"""

    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(dot / (norm_a * norm_b))


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 trace records，输出 P4 报告需要的核心统计。"""

    headers = [r for r in records if r.get("record_type") == "trace_header"]
    layer_records = [r for r in records if r.get("record_type") == "attention_layer"]
    phases = sorted({str(r.get("phase")) for r in layer_records})
    layers = sorted({int(r["layer_id"]) for r in layer_records if r.get("layer_id") is not None})
    per_layer = {
        layer: _summarize_trace_layer(
            [record for record in layer_records if record.get("layer_id") == layer]
        )
        for layer in layers
    }
    adjacent_redundancy = _adjacent_layer_redundancy(layers, per_layer)

    return {
        "schema_version": SCHEMA_VERSION,
        "num_headers": len(headers),
        "num_layer_records": len(layer_records),
        "num_steps": len({r.get("step_id") for r in layer_records}),
        "phases": phases,
        "layers": layers,
        "model_config": headers[-1].get("model_config") if headers else {},
        "trace_config": headers[-1].get("trace_config") if headers else {},
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "adjacent_layer_redundancy": adjacent_redundancy,
    }


def _new_layer_value_lists() -> dict[str, list[Any]]:
    return {
        "visual_mass": [],
        "text_mass": [],
        "entropy": [],
        "entropy_norm": [],
        "visual_entropy_norm": [],
        "visual_head_std": [],
        "visual_k_norm": [],
        "text_k_norm": [],
        "visual_k_by_head": [],
    }


def _collect_trace_layer_values(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    values = _new_layer_value_lists()
    for record in records:
        attention = record.get("attention", {})
        for sequence_stats in attention.get("sequence_stats", []):
            _collect_attention_summary_values(values, sequence_stats)
        _collect_norm_summary_values(values, record)
    return values


def _collect_attention_summary_values(
    values: dict[str, list[Any]],
    sequence_stats: dict[str, Any],
) -> None:
    field_targets = {
        "visual_mass_mean": "visual_mass",
        "text_mass_mean": "text_mass",
        "attention_entropy_mean": "entropy",
        "attention_entropy_normalized_mean": "entropy_norm",
    }
    for field_name, target in field_targets.items():
        if field_name in sequence_stats:
            values[target].append(float(sequence_stats[field_name]))
    visual_entropy = sequence_stats.get("visual_attention_entropy_normalized_mean")
    if visual_entropy is not None:
        values["visual_entropy_norm"].append(float(visual_entropy))
    head_visual_mass = sequence_stats.get("head_visual_mass")
    if head_visual_mass:
        mean_value = sum(head_visual_mass) / len(head_visual_mass)
        variance = sum((item - mean_value) ** 2 for item in head_visual_mass) / len(
            head_visual_mass
        )
        values["visual_head_std"].append(math.sqrt(variance))


def _collect_norm_summary_values(
    values: dict[str, list[Any]],
    record: dict[str, Any],
) -> None:
    by_modality = record.get("head_stats", {}).get("by_modality", {})
    for modality in ("image", "video"):
        if modality in by_modality:
            values["visual_k_norm"].append(float(by_modality[modality]["k_norm_mean"]))
            values["visual_k_by_head"].append(by_modality[modality]["k_norm_mean_by_head"])
    if "text" in by_modality:
        values["text_k_norm"].append(float(by_modality["text"]["k_norm_mean"]))


def _mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    return [
        float(sum(vector[index] for vector in vectors) / len(vectors))
        for index in range(len(vectors[0]))
    ]


def _summarize_trace_layer(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = _collect_trace_layer_values(records)
    visual_norm_mean = _mean(values["visual_k_norm"])
    text_norm_mean = _mean(values["text_k_norm"])
    ratio = (
        None
        if visual_norm_mean is None or text_norm_mean in (None, 0.0)
        else float(visual_norm_mean / text_norm_mean)
    )
    return {
        "record_count": len(records),
        "visual_attention_mass_mean": _mean(values["visual_mass"]),
        "text_attention_mass_mean": _mean(values["text_mass"]),
        "attention_entropy_mean": _mean(values["entropy"]),
        "attention_entropy_normalized_mean": _mean(values["entropy_norm"]),
        "visual_attention_entropy_normalized_mean": _mean(values["visual_entropy_norm"]),
        "visual_head_mass_std_mean": _mean(values["visual_head_std"]),
        "visual_k_norm_mean": visual_norm_mean,
        "text_k_norm_mean": text_norm_mean,
        "visual_text_k_norm_ratio": ratio,
        "visual_k_norm_by_head_mean": _mean_vector(values["visual_k_by_head"]),
    }


def _adjacent_layer_redundancy(
    layers: list[int],
    per_layer: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    adjacent = []
    for previous, following in zip(layers, layers[1:]):
        previous_vector = per_layer[previous].get("visual_k_norm_by_head_mean")
        following_vector = per_layer[following].get("visual_k_norm_by_head_mean")
        similarity = (
            _cosine(previous_vector, following_vector)
            if previous_vector and following_vector
            else None
        )
        adjacent.append(
            {
                "prev_layer": previous,
                "next_layer": following,
                "visual_k_head_cosine": similarity,
            }
        )
    return adjacent


def format_summary_markdown(summary: dict[str, Any]) -> str:
    """把 trace summary 渲染为中文 Markdown。"""

    lines = [
        "# KV Trace Summary",
        "",
        f"- layer records: `{summary['num_layer_records']}`",
        f"- steps: `{summary['num_steps']}`",
        f"- phases: `{', '.join(summary['phases'])}`",
        f"- layers: `{len(summary['layers'])}`",
        "",
        "## Per-layer Visual KV / Attention",
        "",
        "| layer | visual attn mass | text attn mass | entropy | norm entropy | visual norm entropy | visual K norm | text K norm | K ratio | head mass std |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in summary["layers"]:
        stats = summary["per_layer"][str(layer)]

        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.6f}"

        lines.append(
            f"| {layer} | {fmt(stats['visual_attention_mass_mean'])} | "
            f"{fmt(stats['text_attention_mass_mean'])} | "
            f"{fmt(stats['attention_entropy_mean'])} | "
            f"{fmt(stats['attention_entropy_normalized_mean'])} | "
            f"{fmt(stats['visual_attention_entropy_normalized_mean'])} | "
            f"{fmt(stats['visual_k_norm_mean'])} | "
            f"{fmt(stats['text_k_norm_mean'])} | "
            f"{fmt(stats['visual_text_k_norm_ratio'])} | "
            f"{fmt(stats['visual_head_mass_std_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Adjacent-layer Redundancy",
            "",
            "| prev | next | visual K head cosine |",
            "|---:|---:|---:|",
        ]
    )
    for item in summary["adjacent_layer_redundancy"]:
        value = item["visual_k_head_cosine"]
        lines.append(
            f"| {item['prev_layer']} | {item['next_layer']} | "
            f"{'NA' if value is None else f'{float(value):.6f}'} |"
        )
    return "\n".join(lines) + "\n"


def render_summary_svg(summary: dict[str, Any], *, width: int = 960, height: int = 420) -> str:
    """Render a dependency-free SVG chart for visual mass and K norm ratio."""

    layers = summary.get("layers", [])
    if not layers:
        raise ValueError("summary does not contain layers")
    margin_left = 54
    margin_right = 28
    margin_top = 32
    margin_bottom = 54
    chart_width = max(1, width - margin_left - margin_right)
    chart_height = max(1, height - margin_top - margin_bottom)
    values = []
    ratios = []
    for layer in layers:
        stats = summary["per_layer"][str(layer)]
        values.append(float(stats["visual_attention_mass_mean"] or 0.0))
        ratios.append(float(stats["visual_text_k_norm_ratio"] or 0.0))

    max_mass = max(max(values), 1e-6)
    max_ratio = max(max(ratios), 1.0)
    bar_width = chart_width / len(layers)

    def x_pos(idx: int) -> float:
        return margin_left + idx * bar_width

    def y_mass(value: float) -> float:
        return margin_top + chart_height * (1.0 - value / max_mass)

    def y_ratio(value: float) -> float:
        return margin_top + chart_height * (1.0 - value / max_ratio)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="22" font-family="monospace" font-size="16" fill="#111827">KV Trace Layer Summary</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#374151" stroke-width="1"/>',
        f'<text x="{margin_left}" y="{height - 16}" font-family="monospace" font-size="12" fill="#374151">layer</text>',
        f'<text x="{width - 250}" y="22" font-family="monospace" font-size="12" fill="#2563eb">bar: visual attention mass</text>',
        f'<text x="{width - 250}" y="40" font-family="monospace" font-size="12" fill="#dc2626">line: visual/text K norm ratio</text>',
    ]
    for idx, (layer, value) in enumerate(zip(layers, values)):
        x = x_pos(idx)
        y = y_mass(value)
        bar_h = margin_top + chart_height - y
        parts.append(
            f'<rect x="{x + bar_width * 0.15:.2f}" y="{y:.2f}" '
            f'width="{max(1.0, bar_width * 0.7):.2f}" height="{bar_h:.2f}" '
            'fill="#93c5fd" stroke="#2563eb" stroke-width="0.5"/>'
        )
        if idx % max(1, len(layers) // 12) == 0 or idx == len(layers) - 1:
            parts.append(
                f'<text x="{x + bar_width * 0.1:.2f}" y="{height - 34}" '
                'font-family="monospace" font-size="10" fill="#374151">'
                f"{layer}</text>"
            )
    points = " ".join(
        f"{x_pos(idx) + bar_width * 0.5:.2f},{y_ratio(value):.2f}"
        for idx, value in enumerate(ratios)
    )
    parts.append(f'<polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="2"/>')
    for tick, label in [(0.0, "0"), (max_mass, f"{max_mass:.3f}")]:
        y = y_mass(tick)
        parts.append(
            f'<text x="8" y="{y + 4:.2f}" font-family="monospace" font-size="10" fill="#2563eb">{label}</text>'
        )
        parts.append(
            f'<line x1="{margin_left - 4}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#2563eb" stroke-width="1"/>'
        )
    for tick, label in [(0.0, "0"), (max_ratio, f"{max_ratio:.3f}")]:
        y = y_ratio(tick)
        parts.append(
            f'<text x="{width - margin_right + 4}" y="{y + 4:.2f}" font-family="monospace" font-size="10" fill="#dc2626">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


install_kv_trace_provider(
    is_enabled_provider=is_trace_enabled,
    register_model_provider=register_model_config,
    build_metadata_provider=build_trace_metadata,
    record_layer_provider=record_attention_layer,
)
