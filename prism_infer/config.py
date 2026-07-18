"""Typed, validated configuration domains for Prism-Infer.

The public ``LLM(model, **flat_options)`` form remains available for one
compatibility cycle, but it is now a fail-closed adapter into immutable domain
objects.  Runtime-derived values such as effective model length and physical
KV capacity live in :class:`Config`, rather than mutating user input objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from transformers import AutoConfig

from prism_infer.engine.compression import (
    COMPRESSION_FP8_KV,
    COMPRESSION_OFF,
    COMPRESSION_SCALED_FP8_KV,
    COMPRESSION_VISUAL_COMPACT_FP8,
    COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
    compression_mode_supports_cuda_graph,
    normalize_compression_mode,
)
from prism_infer.engine.visual_pruning import VisualPruningConfig


AUTO_KV_CACHE_BLOCKS = -1
UNSET_EOS_TOKEN_ID = -1
DEFAULT_KV_CACHE_PAGE_SIZE = 256
DEFAULT_CPU_KV_CACHE_RATIO = 0.5
SUPPORTED_KV_CACHE_PAGE_SIZES = frozenset({16, 32, 64, 128, 256})
MAX_CUDA_GRAPH_BATCH_SIZE = 512


class ExecutionBackendName(str, Enum):
    """Startup-selected execution backend; runtime fallback is forbidden."""

    EAGER = "eager"
    COMPILE = "compile"
    CUDA_GRAPH = "cuda_graph"
    COMPILE_GRAPH = "compile_graph"


class KVCacheFormat(str, Enum):
    """Physical KV payload format represented by the current runtime."""

    AUTO = "auto"
    MODEL = "model"
    FP8_E4M3FN = "fp8_e4m3fn"


class KVScaleMode(str, Enum):
    """Scale lifecycle for the KV format.

    ``UNIT`` identifies the rejected P5 direct-cast baseline.  P9-C will add
    the separately quality-gated scaled format rather than relabeling it.
    """

    AUTO = "auto"
    NONE = "none"
    UNIT = "unit"
    PER_TOKEN_HEAD = "per_token_head"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    logits_precision: str = "model"
    mlp_projection_mode: str = "packed"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty local path")
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be an existing local directory: {self.model!r}")
        _positive_int(self.max_model_len, name="max_model_len")
        _positive_int(self.tensor_parallel_size, name="tensor_parallel_size")
        if self.logits_precision not in ("fp32", "model"):
            raise ValueError(
                "logits_precision must be 'fp32' or 'model', "
                f"got {self.logits_precision!r}"
            )
        if self.mlp_projection_mode not in ("legacy", "packed"):
            raise ValueError(
                "mlp_projection_mode must be 'legacy' or 'packed', "
                f"got {self.mlp_projection_mode!r}"
            )


@dataclass(frozen=True, slots=True)
class CacheConfig:
    gpu_memory_utilization: float = 0.9
    page_size: int = DEFAULT_KV_CACHE_PAGE_SIZE
    num_gpu_blocks: int = AUTO_KV_CACHE_BLOCKS
    cpu_kv_cache_ratio: float = DEFAULT_CPU_KV_CACHE_RATIO
    enable_prefix_caching: bool = True
    compression_mode: str = COMPRESSION_OFF
    enable_visual_pruning_shadow: bool = False
    visual_pruning_keep_ratio: float = 0.6
    visual_pruning_min_keep_tokens: int = 32
    visual_pruning_strategy: str = "uniform"
    visual_pruning_attention_last_n_layers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.gpu_memory_utilization, bool) or not isinstance(
            self.gpu_memory_utilization, (int, float)
        ):
            raise ValueError(
                "gpu_memory_utilization must be a number in (0, 1], "
                f"got {self.gpu_memory_utilization!r}"
            )
        if not 0.0 < float(self.gpu_memory_utilization) <= 1.0:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization!r}"
            )
        object.__setattr__(
            self,
            "gpu_memory_utilization",
            float(self.gpu_memory_utilization),
        )
        if self.page_size not in SUPPORTED_KV_CACHE_PAGE_SIZES:
            supported = ", ".join(
                str(value) for value in sorted(SUPPORTED_KV_CACHE_PAGE_SIZES)
            )
            raise ValueError(
                f"kvcache_block_size must be one of {{{supported}}}, "
                f"got {self.page_size!r}"
            )
        if self.num_gpu_blocks != AUTO_KV_CACHE_BLOCKS:
            _positive_int(self.num_gpu_blocks, name="num_kvcache_blocks")
        if isinstance(self.cpu_kv_cache_ratio, bool) or not isinstance(
            self.cpu_kv_cache_ratio,
            (int, float),
        ):
            raise TypeError(
                "cpu_kv_cache_ratio must be a non-negative finite number, "
                f"got {self.cpu_kv_cache_ratio!r}"
            )
        cpu_ratio = float(self.cpu_kv_cache_ratio)
        if cpu_ratio < 0.0 or not isfinite(cpu_ratio):
            raise ValueError(
                "cpu_kv_cache_ratio must be a non-negative finite number, "
                f"got {self.cpu_kv_cache_ratio!r}"
            )
        object.__setattr__(self, "cpu_kv_cache_ratio", cpu_ratio)
        _boolean(
            self.enable_prefix_caching,
            name="enable_prefix_caching",
        )
        _boolean(
            self.enable_visual_pruning_shadow,
            name="enable_visual_pruning_shadow",
        )
        if isinstance(self.visual_pruning_keep_ratio, bool) or not isinstance(
            self.visual_pruning_keep_ratio,
            (int, float),
        ):
            raise TypeError(
                "visual_pruning_keep_ratio must be a number, "
                f"got {self.visual_pruning_keep_ratio!r}"
            )
        keep_ratio = float(self.visual_pruning_keep_ratio)
        object.__setattr__(self, "visual_pruning_keep_ratio", keep_ratio)
        _positive_int(
            self.visual_pruning_min_keep_tokens,
            name="visual_pruning_min_keep_tokens",
        )
        _positive_int(
            self.visual_pruning_attention_last_n_layers,
            name="visual_pruning_attention_last_n_layers",
        )
        if not isinstance(self.compression_mode, str):
            raise TypeError(
                "compression_mode must be a string, "
                f"got {self.compression_mode!r}"
            )
        normalized_mode = normalize_compression_mode(self.compression_mode)
        object.__setattr__(self, "compression_mode", normalized_mode)
        VisualPruningConfig(
            keep_ratio=keep_ratio,
            min_keep_tokens=self.visual_pruning_min_keep_tokens,
            strategy=self.visual_pruning_strategy,
            attention_last_n_layers=self.visual_pruning_attention_last_n_layers,
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    enable_chunked_prefill: bool = True
    max_chunk_size: int = 512
    scheduler_policy: str = "fcfs"
    max_queue_size: int | None = None
    max_consecutive_prefill_batches: int = 1

    def __post_init__(self) -> None:
        _positive_int(
            self.max_num_batched_tokens,
            name="max_num_batched_tokens",
        )
        _positive_int(self.max_num_seqs, name="max_num_seqs")
        _positive_int(self.max_chunk_size, name="max_chunk_size")
        _positive_int(
            self.max_consecutive_prefill_batches,
            name="max_consecutive_prefill_batches",
        )
        _boolean(
            self.enable_chunked_prefill,
            name="enable_chunked_prefill",
        )
        if self.scheduler_policy != "fcfs":
            raise ValueError(
                "scheduler_policy currently supports only 'fcfs', "
                f"got {self.scheduler_policy!r}"
            )
        if self.max_queue_size is not None:
            _positive_int(self.max_queue_size, name="max_queue_size")


@dataclass(frozen=True, slots=True)
class MultimodalConfig:
    """HF processor 的显式视觉像素预算；``None`` 保留模型默认值。"""

    image_max_pixels: int | None = None
    video_max_pixels: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("image_max_pixels", self.image_max_pixels),
            ("video_max_pixels", self.video_max_pixels),
        ):
            if value is not None:
                _positive_int(value, name=name)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    backend: ExecutionBackendName | str = ExecutionBackendName.CUDA_GRAPH
    compile_region: str = "none"
    compile_mode: str = "default"
    compile_emulate_precision_casts: bool = True
    compile_force_same_precision: bool = True
    allow_unsafe_compile: bool = False

    def __post_init__(self) -> None:
        try:
            backend = ExecutionBackendName(self.backend)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(item.value for item in ExecutionBackendName)
            raise ValueError(
                f"execution backend must be one of {supported}; got {self.backend!r}"
            ) from exc
        object.__setattr__(self, "backend", backend)
        _boolean(
            self.compile_emulate_precision_casts,
            name="decode_compile_emulate_precision_casts",
        )
        _boolean(
            self.compile_force_same_precision,
            name="decode_compile_force_same_precision",
        )
        _boolean(
            self.allow_unsafe_compile,
            name="allow_unsafe_decode_compile",
        )
        if self.compile_region not in ("none", "attention"):
            raise ValueError(
                "decode_compile_region must be 'none' or 'attention', "
                f"got {self.compile_region!r}"
            )
        if self.compile_mode not in ("default", "reduce-overhead"):
            raise ValueError(
                "decode_compile_mode must be 'default' or 'reduce-overhead', "
                f"got {self.compile_mode!r}"
            )
        if backend is ExecutionBackendName.COMPILE_GRAPH:
            raise ValueError(
                "execution backend 'compile_graph' is not implemented; "
                "startup fallback is forbidden"
            )
        if backend is ExecutionBackendName.COMPILE:
            if self.compile_region != "attention":
                raise ValueError(
                    "execution backend 'compile' currently requires "
                    "decode_compile_region='attention'"
                )
            if not self.allow_unsafe_compile:
                raise ValueError(
                    "decode torch.compile is a rejected P6.3 preflight candidate; "
                    "set allow_unsafe_decode_compile=True only to reproduce "
                    "benchmark evidence"
                )
        elif self.compile_region != "none":
            raise ValueError(
                f"execution backend {backend.value!r} cannot use "
                f"decode_compile_region={self.compile_region!r}"
            )


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    kv_cache_format: KVCacheFormat | str = KVCacheFormat.AUTO
    kv_scale_mode: KVScaleMode | str = KVScaleMode.AUTO
    weight_format: str = "model"
    activation_format: str = "model"

    def resolve(self, *, compression_mode: str) -> "QuantizationConfig":
        try:
            requested_format = KVCacheFormat(self.kv_cache_format)
            requested_scale = KVScaleMode(self.kv_scale_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported KV quantization format or scale mode") from exc
        if self.weight_format != "model" or self.activation_format != "model":
            raise ValueError(
                "weight/activation quantization has no connected backend in P9-B"
            )
        unit_scale_fp8_active = compression_mode in (
            COMPRESSION_FP8_KV,
            COMPRESSION_VISUAL_COMPACT_FP8,
        )
        token_head_scale_active = compression_mode in (
            COMPRESSION_SCALED_FP8_KV,
            COMPRESSION_VISUAL_COMPACT_SCALED_FP8,
        )
        fp8_active = unit_scale_fp8_active or token_head_scale_active
        expected_format = (
            KVCacheFormat.FP8_E4M3FN if fp8_active else KVCacheFormat.MODEL
        )
        expected_scale = (
            KVScaleMode.PER_TOKEN_HEAD
            if token_head_scale_active
            else (KVScaleMode.UNIT if unit_scale_fp8_active else KVScaleMode.NONE)
        )
        if requested_format not in (KVCacheFormat.AUTO, expected_format):
            raise ValueError(
                "kv_cache_format conflicts with compression_mode: "
                f"format={requested_format.value!r}, mode={compression_mode!r}"
            )
        if requested_scale not in (KVScaleMode.AUTO, expected_scale):
            raise ValueError(
                "kv_scale_mode conflicts with compression_mode: "
                f"scale={requested_scale.value!r}, mode={compression_mode!r}"
            )
        return QuantizationConfig(
            kv_cache_format=expected_format,
            kv_scale_mode=expected_scale,
            weight_format=self.weight_format,
            activation_format=self.activation_format,
        )


@dataclass(frozen=True, slots=True)
class ServingConfig:
    enabled: bool = False

    def __post_init__(self) -> None:
        _boolean(self.enabled, name="serving.enabled")
        if self.enabled:
            raise ValueError(
                "network serving is not implemented in P9-B; use the engine API"
            )


@dataclass(frozen=True, slots=True)
class PrismConfig:
    """Validated user configuration before HF-derived resolution."""

    model: ModelConfig
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def __post_init__(self) -> None:
        domains = {
            "model": (self.model, ModelConfig),
            "multimodal": (self.multimodal, MultimodalConfig),
            "cache": (self.cache, CacheConfig),
            "scheduler": (self.scheduler, SchedulerConfig),
            "execution": (self.execution, ExecutionConfig),
            "quantization": (self.quantization, QuantizationConfig),
            "serving": (self.serving, ServingConfig),
        }
        for name, (value, expected_type) in domains.items():
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"PrismConfig.{name} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        resolved_quantization = self.quantization.resolve(
            compression_mode=self.cache.compression_mode
        )
        object.__setattr__(self, "quantization", resolved_quantization)
        if (
            self.execution.backend is ExecutionBackendName.CUDA_GRAPH
            and not compression_mode_supports_cuda_graph(
                self.cache.compression_mode
            )
        ):
            raise ValueError(
                f"compression_mode={self.cache.compression_mode!r} requires "
                "execution backend 'eager' because its dynamic decode "
                "metadata is not CUDA Graph safe"
            )
        if (
            self.execution.backend is ExecutionBackendName.CUDA_GRAPH
            and self.scheduler.max_num_seqs > MAX_CUDA_GRAPH_BATCH_SIZE
        ):
            raise ValueError(
                "execution backend 'cuda_graph' supports max_num_seqs <= "
                f"{MAX_CUDA_GRAPH_BATCH_SIZE}; got "
                f"{self.scheduler.max_num_seqs}"
            )
        if (
            self.execution.backend is ExecutionBackendName.COMPILE
            and self.cache.compression_mode != COMPRESSION_OFF
        ):
            raise ValueError(
                "P6.3 decode compile preflight requires compression_mode='off'"
            )

    @classmethod
    def from_flat_options(
        cls,
        model: str,
        options: Mapping[str, object],
    ) -> "PrismConfig":
        """Strict one-cycle adapter for the historical flat public API."""

        model_fields = {
            "max_model_len": "max_model_len",
            "tensor_parallel_size": "tensor_parallel_size",
            "logits_precision": "logits_precision",
            "mlp_projection_mode": "mlp_projection_mode",
        }
        multimodal_fields = {
            "image_max_pixels": "image_max_pixels",
            "video_max_pixels": "video_max_pixels",
        }
        cache_fields = {
            "gpu_memory_utilization": "gpu_memory_utilization",
            "kvcache_block_size": "page_size",
            "num_kvcache_blocks": "num_gpu_blocks",
            "cpu_kv_cache_ratio": "cpu_kv_cache_ratio",
            "enable_prefix_caching": "enable_prefix_caching",
            "compression_mode": "compression_mode",
            "enable_visual_pruning_shadow": "enable_visual_pruning_shadow",
            "visual_pruning_keep_ratio": "visual_pruning_keep_ratio",
            "visual_pruning_min_keep_tokens": "visual_pruning_min_keep_tokens",
            "visual_pruning_strategy": "visual_pruning_strategy",
            "visual_pruning_attention_last_n_layers": (
                "visual_pruning_attention_last_n_layers"
            ),
        }
        scheduler_fields = {
            "max_num_batched_tokens": "max_num_batched_tokens",
            "max_num_seqs": "max_num_seqs",
            "enable_chunked_prefill": "enable_chunked_prefill",
            "max_chunk_size": "max_chunk_size",
            "scheduler_policy": "scheduler_policy",
            "max_queue_size": "max_queue_size",
            "max_consecutive_prefill_batches": (
                "max_consecutive_prefill_batches"
            ),
        }
        execution_fields = {
            "decode_compile_region": "compile_region",
            "decode_compile_mode": "compile_mode",
            "decode_compile_emulate_precision_casts": (
                "compile_emulate_precision_casts"
            ),
            "decode_compile_force_same_precision": (
                "compile_force_same_precision"
            ),
            "allow_unsafe_decode_compile": "allow_unsafe_compile",
        }
        control_fields = {"enforce_eager", "execution_backend"}
        allowed = (
            set(model_fields)
            | set(multimodal_fields)
            | set(cache_fields)
            | set(scheduler_fields)
            | set(execution_fields)
            | control_fields
        )
        unknown = sorted(set(options) - allowed)
        if unknown:
            joined = ", ".join(repr(name) for name in unknown)
            raise TypeError(f"unknown Prism config option(s): {joined}")

        def select(mapping: Mapping[str, str]) -> dict[str, object]:
            return {
                target: options[source]
                for source, target in mapping.items()
                if source in options
            }

        compile_region = options.get("decode_compile_region", "none")
        if not isinstance(compile_region, str):
            raise TypeError(
                "decode_compile_region must be a string, "
                f"got {compile_region!r}"
            )
        enforce_eager = _boolean(
            options.get("enforce_eager", False),
            name="enforce_eager",
        )
        explicit_backend = options.get("execution_backend")
        if explicit_backend is None:
            if compile_region != "none" and not enforce_eager:
                raise ValueError(
                    "decode torch.compile region and CUDA Graph are mutually exclusive"
                )
            backend = (
                ExecutionBackendName.COMPILE
                if compile_region != "none"
                else (
                    ExecutionBackendName.EAGER
                    if enforce_eager
                    else ExecutionBackendName.CUDA_GRAPH
                )
            )
        else:
            try:
                backend = ExecutionBackendName(explicit_backend)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"unsupported execution_backend: {explicit_backend!r}"
                ) from exc
            if "enforce_eager" in options:
                expected_eager = backend is not ExecutionBackendName.CUDA_GRAPH
                if enforce_eager != expected_eager:
                    raise ValueError(
                        "enforce_eager conflicts with execution_backend: "
                        f"enforce_eager={enforce_eager}, backend={backend.value!r}"
                    )

        execution_options = select(execution_fields)
        execution_options["backend"] = backend
        return cls(
            model=ModelConfig(model=model, **select(model_fields)),
            multimodal=MultimodalConfig(**select(multimodal_fields)),
            cache=CacheConfig(**select(cache_fields)),
            scheduler=SchedulerConfig(**select(scheduler_fields)),
            execution=ExecutionConfig(**execution_options),
        )


@dataclass(frozen=True, slots=True, init=False)
class Config:
    """Immutable HF-resolved runtime configuration.

    Flat properties intentionally preserve existing internal call sites while
    the domain objects remain the single source of truth.
    """

    prism_config: PrismConfig
    hf_config: Any
    effective_max_model_len: int
    eos: int
    resolved_num_kvcache_blocks: int
    num_cpu_blocks: int

    def __init__(self, model: str | PrismConfig, **flat_options: object) -> None:
        if isinstance(model, PrismConfig):
            if flat_options:
                unknown = ", ".join(repr(name) for name in sorted(flat_options))
                raise TypeError(
                    "nested PrismConfig cannot be combined with flat options: "
                    f"{unknown}"
                )
            prism_config = model
        elif isinstance(model, str):
            prism_config = PrismConfig.from_flat_options(model, flat_options)
        else:
            raise TypeError(
                "model must be a local path or PrismConfig, "
                f"got {type(model).__name__}"
            )

        hf_config = AutoConfig.from_pretrained(prism_config.model.model)
        text_config = getattr(hf_config, "text_config", None)
        model_limit = getattr(hf_config, "max_position_embeddings", None)
        if model_limit is None and text_config is not None:
            model_limit = getattr(text_config, "max_position_embeddings", None)
        if not isinstance(model_limit, int) or model_limit <= 0:
            model_limit = prism_config.model.max_model_len
        effective_max_model_len = min(
            prism_config.model.max_model_len,
            model_limit,
        )
        if (
            prism_config.scheduler.max_num_batched_tokens
            < effective_max_model_len
        ):
            raise ValueError(
                "max_num_batched_tokens must be >= effective max_model_len: "
                f"{prism_config.scheduler.max_num_batched_tokens} < "
                f"{effective_max_model_len}"
            )
        self._install(
            prism_config=prism_config,
            hf_config=hf_config,
            effective_max_model_len=effective_max_model_len,
            eos=UNSET_EOS_TOKEN_ID,
            resolved_num_kvcache_blocks=(
                prism_config.cache.num_gpu_blocks
            ),
            num_cpu_blocks=0,
        )

    def _install(
        self,
        *,
        prism_config: PrismConfig,
        hf_config: Any,
        effective_max_model_len: int,
        eos: int,
        resolved_num_kvcache_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        object.__setattr__(self, "prism_config", prism_config)
        object.__setattr__(self, "hf_config", hf_config)
        object.__setattr__(
            self,
            "effective_max_model_len",
            effective_max_model_len,
        )
        object.__setattr__(self, "eos", eos)
        object.__setattr__(
            self,
            "resolved_num_kvcache_blocks",
            resolved_num_kvcache_blocks,
        )
        object.__setattr__(self, "num_cpu_blocks", num_cpu_blocks)

    def _with_runtime_values(
        self,
        *,
        eos: int,
        num_kvcache_blocks: int,
        num_cpu_blocks: int,
    ) -> "Config":
        resolved = object.__new__(Config)
        resolved._install(
            prism_config=self.prism_config,
            hf_config=self.hf_config,
            effective_max_model_len=self.effective_max_model_len,
            eos=eos,
            resolved_num_kvcache_blocks=num_kvcache_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )
        return resolved

    def with_eos(self, eos: int | None) -> "Config":
        if eos is None:
            eos = UNSET_EOS_TOKEN_ID
        if isinstance(eos, bool) or not isinstance(eos, int):
            raise ValueError(f"eos token id must be an integer or None, got {eos!r}")
        if eos < 0 and eos != UNSET_EOS_TOKEN_ID:
            raise ValueError(
                "eos token id must be non-negative, None, or the unset "
                f"sentinel {UNSET_EOS_TOKEN_ID}; got {eos}"
            )
        return self._with_runtime_values(
            eos=eos,
            num_kvcache_blocks=self.num_kvcache_blocks,
            num_cpu_blocks=self.num_cpu_blocks,
        )

    def with_cache_capacity(
        self,
        *,
        num_kvcache_blocks: int,
        num_cpu_blocks: int,
    ) -> "Config":
        _positive_int(num_kvcache_blocks, name="num_kvcache_blocks")
        if (
            isinstance(num_cpu_blocks, bool)
            or not isinstance(num_cpu_blocks, int)
            or num_cpu_blocks < 0
        ):
            raise ValueError(
                f"num_cpu_blocks must be a non-negative integer, got {num_cpu_blocks!r}"
            )
        return self._with_runtime_values(
            eos=self.eos,
            num_kvcache_blocks=num_kvcache_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )

    @property
    def model_config(self) -> ModelConfig:
        return self.prism_config.model

    @property
    def multimodal_config(self) -> MultimodalConfig:
        return self.prism_config.multimodal

    @property
    def cache_config(self) -> CacheConfig:
        return self.prism_config.cache

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self.prism_config.scheduler

    @property
    def execution_config(self) -> ExecutionConfig:
        return self.prism_config.execution

    @property
    def quantization_config(self) -> QuantizationConfig:
        return self.prism_config.quantization

    @property
    def serving_config(self) -> ServingConfig:
        return self.prism_config.serving

    @property
    def model(self) -> str:
        return self.model_config.model

    @property
    def max_model_len(self) -> int:
        return self.effective_max_model_len

    @property
    def tensor_parallel_size(self) -> int:
        return self.model_config.tensor_parallel_size

    @property
    def logits_precision(self) -> str:
        return self.model_config.logits_precision

    @property
    def mlp_projection_mode(self) -> str:
        return self.model_config.mlp_projection_mode

    @property
    def image_max_pixels(self) -> int | None:
        return self.multimodal_config.image_max_pixels

    @property
    def video_max_pixels(self) -> int | None:
        return self.multimodal_config.video_max_pixels

    @property
    def max_num_batched_tokens(self) -> int:
        return self.scheduler_config.max_num_batched_tokens

    @property
    def max_num_seqs(self) -> int:
        return self.scheduler_config.max_num_seqs

    @property
    def enable_chunked_prefill(self) -> bool:
        return self.scheduler_config.enable_chunked_prefill

    @property
    def max_chunk_size(self) -> int:
        return self.scheduler_config.max_chunk_size

    @property
    def scheduler_policy(self) -> str:
        return self.scheduler_config.scheduler_policy

    @property
    def max_queue_size(self) -> int | None:
        return self.scheduler_config.max_queue_size

    @property
    def max_consecutive_prefill_batches(self) -> int:
        return self.scheduler_config.max_consecutive_prefill_batches

    @property
    def gpu_memory_utilization(self) -> float:
        return float(self.cache_config.gpu_memory_utilization)

    @property
    def kvcache_block_size(self) -> int:
        return self.cache_config.page_size

    @property
    def num_kvcache_blocks(self) -> int:
        return self.resolved_num_kvcache_blocks

    @property
    def enable_prefix_caching(self) -> bool:
        return self.cache_config.enable_prefix_caching

    @property
    def cpu_kv_cache_ratio(self) -> float:
        return self.cache_config.cpu_kv_cache_ratio

    @property
    def compression_mode(self) -> str:
        return self.cache_config.compression_mode

    @property
    def enable_visual_pruning_shadow(self) -> bool:
        return self.cache_config.enable_visual_pruning_shadow

    @property
    def visual_pruning_keep_ratio(self) -> float:
        return self.cache_config.visual_pruning_keep_ratio

    @property
    def visual_pruning_min_keep_tokens(self) -> int:
        return self.cache_config.visual_pruning_min_keep_tokens

    @property
    def visual_pruning_strategy(self) -> str:
        return self.cache_config.visual_pruning_strategy

    @property
    def visual_pruning_attention_last_n_layers(self) -> int:
        return self.cache_config.visual_pruning_attention_last_n_layers

    @property
    def execution_backend(self) -> str:
        return self.execution_config.backend.value

    @property
    def enforce_eager(self) -> bool:
        return self.execution_config.backend is not ExecutionBackendName.CUDA_GRAPH

    @property
    def decode_compile_region(self) -> str:
        return self.execution_config.compile_region

    @property
    def decode_compile_mode(self) -> str:
        return self.execution_config.compile_mode

    @property
    def decode_compile_emulate_precision_casts(self) -> bool:
        return self.execution_config.compile_emulate_precision_casts

    @property
    def decode_compile_force_same_precision(self) -> bool:
        return self.execution_config.compile_force_same_precision

    @property
    def allow_unsafe_decode_compile(self) -> bool:
        return self.execution_config.allow_unsafe_compile
