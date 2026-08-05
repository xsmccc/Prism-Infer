"""Request-level generation parameters."""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Generation settings attached to one request."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
