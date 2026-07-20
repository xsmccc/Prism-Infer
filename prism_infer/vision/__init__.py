from prism_infer.vision.backends import VisionAttentionBackendName

__all__ = ["VisionAttentionBackendName", "VisionEncoder"]


def __getattr__(name: str):
    """Keep backend policy imports independent from the torch vision implementation."""

    if name == "VisionEncoder":
        from prism_infer.vision.vision_encoder import VisionEncoder

        return VisionEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
