"""Request lifecycle contracts used by the scheduler and online engine.

The original engine encoded lifecycle changes as unconstrained assignments to
``Sequence.status``.  Keeping the state machine in a dependency-light module
makes those transitions explicit without coupling request admission to CUDA or
model execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import perf_counter_ns


class RequestState(Enum):
    """Authoritative request states in the main engine process."""

    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    # Backwards-compatible name used by pre-P7 tests and integrations.
    RUNNING = DECODING
    SWAPPED = auto()
    FINISHED = auto()
    CANCELLED = auto()
    REJECTED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.REJECTED,
        }


# Preserve the public name imported throughout the P1-P6 code and tests.
SequenceStatus = RequestState


_ALLOWED_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.WAITING: frozenset(
        {
            RequestState.PREFILLING,
            RequestState.CANCELLED,
            RequestState.REJECTED,
        }
    ),
    RequestState.PREFILLING: frozenset(
        {
            RequestState.DECODING,
            RequestState.WAITING,
            RequestState.CANCELLED,
            RequestState.FINISHED,
        }
    ),
    RequestState.DECODING: frozenset(
        {
            RequestState.SWAPPED,
            RequestState.WAITING,
            RequestState.FINISHED,
            RequestState.CANCELLED,
        }
    ),
    RequestState.SWAPPED: frozenset(
        {
            RequestState.DECODING,
            RequestState.WAITING,
            RequestState.CANCELLED,
        }
    ),
    RequestState.FINISHED: frozenset(),
    RequestState.CANCELLED: frozenset(),
    RequestState.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class RequestTransition:
    """One auditable lifecycle transition."""

    source: RequestState
    target: RequestState
    reason: str
    timestamp_ns: int


@dataclass(slots=True)
class RequestLifecycle:
    """Finite-state machine owned by one request/sequence."""

    request_id: int
    state: RequestState = RequestState.WAITING
    transitions: list[RequestTransition] = field(default_factory=list)

    def transition(
        self,
        target: RequestState,
        *,
        reason: str,
        timestamp_ns: int | None = None,
    ) -> None:
        if not isinstance(target, RequestState):
            raise TypeError(f"target must be RequestState, got {type(target)!r}")
        source = self.state
        if target == source:
            return
        if target not in _ALLOWED_TRANSITIONS[source]:
            raise RuntimeError(
                "invalid request state transition: "
                f"request={self.request_id} {source.name}->{target.name} "
                f"reason={reason!r}"
            )
        self.state = target
        self.transitions.append(
            RequestTransition(
                source=source,
                target=target,
                reason=reason,
                timestamp_ns=(
                    perf_counter_ns() if timestamp_ns is None else timestamp_ns
                ),
            )
        )

    def restore(self, target: RequestState) -> None:
        """Restore externally serialized/test state without inventing history.

        Main-engine code must use :meth:`transition`.  ``restore`` exists for
        worker deserialization and backwards-compatible direct ``status``
        assignment in older integrations.
        """

        if not isinstance(target, RequestState):
            raise TypeError(f"target must be RequestState, got {type(target)!r}")
        self.state = target
