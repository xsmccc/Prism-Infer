"""Validated Qwen3-VL architecture contract and canonical component defaults."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isqrt
from typing import Any


SUPPORTED_QWEN3_VL_MODEL_TYPES = frozenset({"qwen3_vl"})
MROPE_AXIS_COUNT = 3
MROPE_POSITION_TENSOR_RANK = 3
ROTARY_PAIR_SIZE = 2
VISION_GRID_DIMENSIONS = 3
VISION_GRID_TEMPORAL_AXIS = 0
VISION_GRID_HEIGHT_AXIS = 1
VISION_GRID_WIDTH_AXIS = 2

CANONICAL_TEXT_VOCAB_SIZE = 151_936
CANONICAL_TEXT_HIDDEN_SIZE = 4_096
CANONICAL_TEXT_NUM_HEADS = 32
CANONICAL_TEXT_NUM_KV_HEADS = 8
CANONICAL_TEXT_NUM_LAYERS = 36
CANONICAL_TEXT_INTERMEDIATE_SIZE = 12_288
CANONICAL_TEXT_HEAD_DIM = 128
CANONICAL_TEXT_ROPE_THETA = 5_000_000.0
CANONICAL_TEXT_RMS_NORM_EPS = 1.0e-6
CANONICAL_MROPE_SECTION = (24, 20, 20)

CANONICAL_IMAGE_TOKEN_ID = 151_655
CANONICAL_VIDEO_TOKEN_ID = 151_656
CANONICAL_VISION_START_TOKEN_ID = 151_652

CANONICAL_VISION_HIDDEN_SIZE = 1_152
CANONICAL_VISION_IN_CHANNELS = 3
CANONICAL_VISION_TEMPORAL_PATCH_SIZE = 2
CANONICAL_VISION_PATCH_SIZE = 16
CANONICAL_VISION_NUM_HEADS = 16
CANONICAL_VISION_INTERMEDIATE_SIZE = 4_304
CANONICAL_VISION_DEPTH = 27
CANONICAL_VISION_OUTPUT_SIZE = 4_096
CANONICAL_VISION_POSITION_EMBEDDINGS = 2_304
CANONICAL_VISION_SPATIAL_MERGE_SIZE = 2
CANONICAL_VISION_ROPE_THETA = 10_000.0
CANONICAL_DEEPSTACK_VISUAL_INDEXES = (8, 16, 24)


_MISSING = object()


def _read(config: object, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not _MISSING:
        return default
    joined = " or ".join(repr(name) for name in names)
    raise ValueError(f"Qwen3-VL config is missing required field {joined}")


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    normalized = float(value)
    if normalized <= 0.0 or not isfinite(normalized):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return normalized


def _text_config(config: object) -> object:
    return _read(config, "text_config", default=config)


def _vision_config(config: object) -> object:
    return _read(config, "vision_config", "vision_config_dict")


def _rope_config(text_config: object) -> object:
    return _read(text_config, "rope_parameters", "rope_scaling")


def _mrope_section(text_config: object) -> tuple[int, ...]:
    rope_config = _read(
        text_config,
        "rope_parameters",
        "rope_scaling",
        default=None,
    )
    section = (
        _read(rope_config, "mrope_section")
        if rope_config is not None
        else _read(text_config, "mrope_section")
    )
    if not isinstance(section, (list, tuple)):
        raise TypeError("mrope_section must be a list or tuple")
    normalized = tuple(
        _positive_int(value, name=f"mrope_section[{index}]") for index, value in enumerate(section)
    )
    if len(normalized) != MROPE_AXIS_COUNT:
        raise ValueError(f"mrope_section must contain {MROPE_AXIS_COUNT} axes, got {normalized}")
    return normalized


@dataclass(frozen=True, slots=True)
class Qwen3VLTextArchitecture:
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    intermediate_size: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    mrope_section: tuple[int, ...]

    @classmethod
    def canonical(cls) -> "Qwen3VLTextArchitecture":
        return cls(
            vocab_size=CANONICAL_TEXT_VOCAB_SIZE,
            hidden_size=CANONICAL_TEXT_HIDDEN_SIZE,
            num_heads=CANONICAL_TEXT_NUM_HEADS,
            num_kv_heads=CANONICAL_TEXT_NUM_KV_HEADS,
            num_layers=CANONICAL_TEXT_NUM_LAYERS,
            intermediate_size=CANONICAL_TEXT_INTERMEDIATE_SIZE,
            head_dim=CANONICAL_TEXT_HEAD_DIM,
            rope_theta=CANONICAL_TEXT_ROPE_THETA,
            rms_norm_eps=CANONICAL_TEXT_RMS_NORM_EPS,
            mrope_section=CANONICAL_MROPE_SECTION,
        )

    @classmethod
    def from_config(cls, config: object) -> "Qwen3VLTextArchitecture":
        text = _text_config(config)
        hidden_size = _positive_int(
            _read(text, "hidden_size"),
            name="text_config.hidden_size",
        )
        num_heads = _positive_int(
            _read(text, "num_attention_heads"),
            name="text_config.num_attention_heads",
        )
        num_kv_heads = _positive_int(
            _read(text, "num_key_value_heads"),
            name="text_config.num_key_value_heads",
        )
        head_dim = _positive_int(
            _read(
                text,
                "head_dim",
                default=(hidden_size // num_heads if hidden_size % num_heads == 0 else _MISSING),
            ),
            name="text_config.head_dim",
        )
        section = _mrope_section(text)
        rope_theta = _read(text, "rope_theta", default=None)
        if rope_theta is None:
            rope_theta = _read(_rope_config(text), "rope_theta")
        architecture = cls(
            vocab_size=_positive_int(
                _read(text, "vocab_size"),
                name="text_config.vocab_size",
            ),
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=_positive_int(
                _read(text, "num_hidden_layers", "num_layers"),
                name="text_config.num_hidden_layers",
            ),
            intermediate_size=_positive_int(
                _read(text, "intermediate_size"),
                name="text_config.intermediate_size",
            ),
            head_dim=head_dim,
            rope_theta=_positive_float(
                rope_theta,
                name="text_config.rope_theta",
            ),
            rms_norm_eps=_positive_float(
                _read(text, "rms_norm_eps"),
                name="text_config.rms_norm_eps",
            ),
            mrope_section=section,
        )
        architecture.validate()
        return architecture

    def validate(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                "Qwen3-VL query heads must be divisible by KV heads: "
                f"{self.num_heads} % {self.num_kv_heads} != 0"
            )
        if self.head_dim % ROTARY_PAIR_SIZE:
            raise ValueError(f"Qwen3-VL head_dim must be even, got {self.head_dim}")
        if sum(self.mrope_section) * ROTARY_PAIR_SIZE != self.head_dim:
            raise ValueError(
                "mrope_section/head_dim mismatch: "
                f"sum={sum(self.mrope_section)}, head_dim={self.head_dim}"
            )


@dataclass(frozen=True, slots=True)
class Qwen3VLVisionArchitecture:
    hidden_size: int
    in_channels: int
    temporal_patch_size: int
    patch_size: int
    num_heads: int
    intermediate_size: int
    depth: int
    output_size: int
    num_position_embeddings: int
    spatial_merge_size: int
    deepstack_visual_indexes: tuple[int, ...]

    @classmethod
    def canonical(cls) -> "Qwen3VLVisionArchitecture":
        return cls(
            hidden_size=CANONICAL_VISION_HIDDEN_SIZE,
            in_channels=CANONICAL_VISION_IN_CHANNELS,
            temporal_patch_size=CANONICAL_VISION_TEMPORAL_PATCH_SIZE,
            patch_size=CANONICAL_VISION_PATCH_SIZE,
            num_heads=CANONICAL_VISION_NUM_HEADS,
            intermediate_size=CANONICAL_VISION_INTERMEDIATE_SIZE,
            depth=CANONICAL_VISION_DEPTH,
            output_size=CANONICAL_VISION_OUTPUT_SIZE,
            num_position_embeddings=CANONICAL_VISION_POSITION_EMBEDDINGS,
            spatial_merge_size=CANONICAL_VISION_SPATIAL_MERGE_SIZE,
            deepstack_visual_indexes=CANONICAL_DEEPSTACK_VISUAL_INDEXES,
        )

    @classmethod
    def from_config(cls, config: object) -> "Qwen3VLVisionArchitecture":
        vision = (
            _vision_config(config)
            if _read(config, "vision_config", "vision_config_dict", default=None) is not None
            else config
        )
        indexes = _read(
            vision,
            "deepstack_visual_indexes",
            "deepstack_visual_indices",
        )
        if not isinstance(indexes, (list, tuple)):
            raise TypeError("deepstack_visual_indexes must be a list or tuple")
        architecture = cls(
            hidden_size=_positive_int(
                _read(vision, "hidden_size", "embed_dim"),
                name="vision_config.hidden_size",
            ),
            in_channels=_positive_int(
                _read(vision, "in_channels"),
                name="vision_config.in_channels",
            ),
            temporal_patch_size=_positive_int(
                _read(vision, "temporal_patch_size"),
                name="vision_config.temporal_patch_size",
            ),
            patch_size=_positive_int(
                _read(vision, "patch_size"),
                name="vision_config.patch_size",
            ),
            num_heads=_positive_int(
                _read(vision, "num_heads", "num_attention_heads"),
                name="vision_config.num_heads",
            ),
            intermediate_size=_positive_int(
                _read(vision, "intermediate_size", "mlp_hidden_size"),
                name="vision_config.intermediate_size",
            ),
            depth=_positive_int(
                _read(vision, "depth", "num_hidden_layers", "num_layers"),
                name="vision_config.depth",
            ),
            output_size=_positive_int(
                _read(vision, "out_hidden_size", "output_hidden_size"),
                name="vision_config.out_hidden_size",
            ),
            num_position_embeddings=_positive_int(
                _read(
                    vision,
                    "num_position_embeddings",
                    "max_position_embeddings",
                ),
                name="vision_config.num_position_embeddings",
            ),
            spatial_merge_size=_positive_int(
                _read(vision, "spatial_merge_size"),
                name="vision_config.spatial_merge_size",
            ),
            deepstack_visual_indexes=tuple(
                _non_negative_int(
                    value,
                    name=f"deepstack_visual_indexes[{index}]",
                )
                for index, value in enumerate(indexes)
            ),
        )
        architecture.validate()
        return architecture

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError(
                "vision hidden_size must be divisible by num_heads: "
                f"{self.hidden_size} % {self.num_heads} != 0"
            )
        head_dim = self.hidden_size // self.num_heads
        if head_dim % (ROTARY_PAIR_SIZE * ROTARY_PAIR_SIZE):
            raise ValueError(f"vision head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
        grid_side = isqrt(self.num_position_embeddings)
        if grid_side * grid_side != self.num_position_embeddings:
            raise ValueError(
                "vision num_position_embeddings must be a perfect square, "
                f"got {self.num_position_embeddings}"
            )
        indexes = self.deepstack_visual_indexes
        if tuple(sorted(set(indexes))) != indexes:
            raise ValueError(f"deepstack_visual_indexes must be unique and sorted, got {indexes}")
        if any(index >= self.depth for index in indexes):
            raise ValueError(
                "deepstack visual index exceeds vision depth: "
                f"indexes={indexes}, depth={self.depth}"
            )


@dataclass(frozen=True, slots=True)
class Qwen3VLArchitecture:
    text: Qwen3VLTextArchitecture
    vision: Qwen3VLVisionArchitecture
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    tie_word_embeddings: bool

    @classmethod
    def canonical(cls) -> "Qwen3VLArchitecture":
        return cls(
            text=Qwen3VLTextArchitecture.canonical(),
            vision=Qwen3VLVisionArchitecture.canonical(),
            image_token_id=CANONICAL_IMAGE_TOKEN_ID,
            video_token_id=CANONICAL_VIDEO_TOKEN_ID,
            vision_start_token_id=CANONICAL_VISION_START_TOKEN_ID,
            tie_word_embeddings=False,
        )

    @classmethod
    def from_config(cls, config: object) -> "Qwen3VLArchitecture":
        model_type = _read(config, "model_type")
        if model_type not in SUPPORTED_QWEN3_VL_MODEL_TYPES:
            raise ValueError(
                f"unsupported multimodal model_type {model_type!r}; "
                f"supported={sorted(SUPPORTED_QWEN3_VL_MODEL_TYPES)}"
            )
        text_config = _text_config(config)
        vision_config = _vision_config(config)
        if _read(text_config, "attention_bias") is not False:
            raise ValueError("Qwen3-VL attention_bias=True is not supported")
        if float(_read(text_config, "attention_dropout")) != 0.0:
            raise ValueError("Qwen3-VL attention_dropout must be zero")
        if _read(text_config, "hidden_act") != "silu":
            raise ValueError("Qwen3-VL text hidden_act must be 'silu'")
        if _read(vision_config, "hidden_act") != "gelu_pytorch_tanh":
            raise ValueError("Qwen3-VL vision hidden_act must be 'gelu_pytorch_tanh'")
        rope_config = _rope_config(text_config)
        if _read(rope_config, "mrope_interleaved") is not True:
            raise ValueError("only interleaved Qwen3-VL M-RoPE is supported")
        tie_word_embeddings = _read(
            config,
            "tie_word_embeddings",
            default=_read(text_config, "tie_word_embeddings", default=False),
        )
        if not isinstance(tie_word_embeddings, bool):
            raise TypeError("tie_word_embeddings must be bool")
        architecture = cls(
            text=Qwen3VLTextArchitecture.from_config(config),
            vision=Qwen3VLVisionArchitecture.from_config(vision_config),
            image_token_id=_positive_int(
                _read(config, "image_token_id"),
                name="image_token_id",
            ),
            video_token_id=_positive_int(
                _read(config, "video_token_id"),
                name="video_token_id",
            ),
            vision_start_token_id=_positive_int(
                _read(config, "vision_start_token_id"),
                name="vision_start_token_id",
            ),
            tie_word_embeddings=tie_word_embeddings,
        )
        architecture.validate()
        return architecture

    def validate(self) -> None:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError(
                "vision output size must equal text hidden size: "
                f"vision={self.vision.output_size}, text={self.text.hidden_size}"
            )
        if len(self.vision.deepstack_visual_indexes) > self.text.num_layers:
            raise ValueError(
                "DeepStack outputs exceed text decoder depth: "
                f"outputs={len(self.vision.deepstack_visual_indexes)}, "
                f"layers={self.text.num_layers}"
            )
        token_ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
        )
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(f"Qwen3-VL special token ids must be distinct: {token_ids}")


__all__ = [
    "CANONICAL_DEEPSTACK_VISUAL_INDEXES",
    "CANONICAL_IMAGE_TOKEN_ID",
    "CANONICAL_MROPE_SECTION",
    "CANONICAL_TEXT_HEAD_DIM",
    "CANONICAL_TEXT_HIDDEN_SIZE",
    "CANONICAL_TEXT_INTERMEDIATE_SIZE",
    "CANONICAL_TEXT_NUM_HEADS",
    "CANONICAL_TEXT_NUM_KV_HEADS",
    "CANONICAL_TEXT_NUM_LAYERS",
    "CANONICAL_TEXT_RMS_NORM_EPS",
    "CANONICAL_TEXT_ROPE_THETA",
    "CANONICAL_TEXT_VOCAB_SIZE",
    "CANONICAL_VIDEO_TOKEN_ID",
    "CANONICAL_VISION_DEPTH",
    "CANONICAL_VISION_HIDDEN_SIZE",
    "CANONICAL_VISION_IN_CHANNELS",
    "CANONICAL_VISION_INTERMEDIATE_SIZE",
    "CANONICAL_VISION_NUM_HEADS",
    "CANONICAL_VISION_OUTPUT_SIZE",
    "CANONICAL_VISION_PATCH_SIZE",
    "CANONICAL_VISION_POSITION_EMBEDDINGS",
    "CANONICAL_VISION_ROPE_THETA",
    "CANONICAL_VISION_SPATIAL_MERGE_SIZE",
    "CANONICAL_VISION_START_TOKEN_ID",
    "CANONICAL_VISION_TEMPORAL_PATCH_SIZE",
    "MROPE_AXIS_COUNT",
    "MROPE_POSITION_TENSOR_RANK",
    "Qwen3VLArchitecture",
    "Qwen3VLTextArchitecture",
    "Qwen3VLVisionArchitecture",
    "ROTARY_PAIR_SIZE",
    "VISION_GRID_DIMENSIONS",
    "VISION_GRID_HEIGHT_AXIS",
    "VISION_GRID_TEMPORAL_AXIS",
    "VISION_GRID_WIDTH_AXIS",
]
