"""Public Prism-Infer API with lazy loading for heavyweight engine dependencies."""

from prism_infer.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from prism_infer.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
