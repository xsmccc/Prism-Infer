"""No-op-by-default performance instrumentation boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable


ProfileRegionProvider = Callable[..., AbstractContextManager[None]]
ProfileSessionProvider = Callable[[], Any | None]


_profile_region_provider: ProfileRegionProvider | None = None
_profile_session_provider: ProfileSessionProvider | None = None


def install_performance_provider(
    *,
    profile_region_provider: ProfileRegionProvider,
    profile_session_provider: ProfileSessionProvider,
) -> None:
    """Install the optional analysis collector behind the runtime facade."""

    if not callable(profile_region_provider) or not callable(profile_session_provider):
        raise TypeError("performance observability providers must be callable")
    global _profile_region_provider, _profile_session_provider
    _profile_region_provider = profile_region_provider
    _profile_session_provider = profile_session_provider


def profile_region(
    name: str,
    *,
    cuda: bool = True,
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    """Return an installed semantic region or a zero-cost no-op context."""

    provider = _profile_region_provider
    if provider is None:
        return nullcontext()
    return provider(name, cuda=cuda, metadata=metadata)


def get_performance_profile_session() -> Any | None:
    provider = _profile_session_provider
    return None if provider is None else provider()
