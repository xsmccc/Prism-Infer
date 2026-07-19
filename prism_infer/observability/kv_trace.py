"""No-op-by-default KV trace boundary used by the inference runtime."""

from __future__ import annotations

from typing import Any, Callable


_is_enabled_provider: Callable[[], bool] | None = None
_register_model_provider: Callable[[Any], None] | None = None
_build_metadata_provider: Callable[..., Any | None] | None = None
_record_layer_provider: Callable[..., None] | None = None


def install_kv_trace_provider(
    *,
    is_enabled_provider: Callable[[], bool],
    register_model_provider: Callable[[Any], None],
    build_metadata_provider: Callable[..., Any | None],
    record_layer_provider: Callable[..., None],
) -> None:
    providers = (
        is_enabled_provider,
        register_model_provider,
        build_metadata_provider,
        record_layer_provider,
    )
    if any(not callable(provider) for provider in providers):
        raise TypeError("KV trace observability providers must be callable")
    global _is_enabled_provider
    global _register_model_provider
    global _build_metadata_provider
    global _record_layer_provider
    _is_enabled_provider = is_enabled_provider
    _register_model_provider = register_model_provider
    _build_metadata_provider = build_metadata_provider
    _record_layer_provider = record_layer_provider


def is_trace_enabled() -> bool:
    provider = _is_enabled_provider
    return False if provider is None else bool(provider())


def register_model_config(config: Any) -> None:
    provider = _register_model_provider
    if provider is not None:
        provider(config)


def build_trace_metadata(*args: Any, **kwargs: Any) -> Any | None:
    provider = _build_metadata_provider
    return None if provider is None else provider(*args, **kwargs)


def record_attention_layer(*args: Any, **kwargs: Any) -> None:
    provider = _record_layer_provider
    if provider is not None:
        provider(*args, **kwargs)
