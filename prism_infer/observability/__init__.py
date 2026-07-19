"""Dependency-inverted runtime observability hooks.

The inference runtime depends only on these no-op-by-default facades.  Optional
analysis collectors register concrete providers when their public modules are
imported, keeping experiment/reporting code out of the production dependency
direction.
"""

from prism_infer.observability.kv_trace import (
    build_trace_metadata,
    is_trace_enabled,
    record_attention_layer,
    register_model_config,
)
from prism_infer.observability.performance import (
    get_performance_profile_session,
    profile_region,
)

__all__ = [
    "build_trace_metadata",
    "get_performance_profile_session",
    "is_trace_enabled",
    "profile_region",
    "record_attention_layer",
    "register_model_config",
]
