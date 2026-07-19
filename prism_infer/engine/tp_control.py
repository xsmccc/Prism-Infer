"""Typed request/ack protocol for tensor-parallel worker control."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection, wait
from time import perf_counter


DEFAULT_TP_CONTROL_TIMEOUT_SECONDS = 120


class TPMethod(str, Enum):
    """ModelRunner methods that are safe to dispatch to every TP rank."""

    COPY_KV_BLOCKS = "copy_kv_blocks"
    SWAP_BLOCKS = "swap_blocks"
    RUN_PLAN = "run_plan"
    COMPACT_KV_CACHE = "compact_kv_cache"
    RUN = "run"  # one-cycle compatibility adapter
    EXIT = "exit"


class TPResponseStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class TPControlPlane:
    """Own TP command transport, request ordering, ACKs, and timeouts.

    Model execution is deliberately outside this class. Workers decode one
    command, invoke their local runner, and return a typed response; rank 0
    coordinates transport without embedding pipe state in model execution.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        channel: Connection | list[Connection],
        timeout_seconds: float = DEFAULT_TP_CONTROL_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"TP world_size must be a positive integer, got {world_size!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
        self.rank = rank
        self.world_size = world_size
        self.channel = channel
        self.timeout_seconds = timeout_seconds
        self._next_request_id = 0
        self._validate_topology()

    def _validate_topology(self) -> None:
        if self.world_size <= 1:
            return
        if self.rank == 0:
            if not isinstance(self.channel, list):
                raise TypeError("rank 0 requires a list of TP control channels")
            expected = self.world_size - 1
            if len(self.channel) != expected:
                raise ValueError(
                    "rank 0 TP control channel count must equal world_size - 1: "
                    f"channels={len(self.channel)}, world_size={self.world_size}"
                )
            if any(not isinstance(item, Connection) for item in self.channel):
                raise TypeError("rank 0 TP control channels must be Connections")
            return
        if not isinstance(self.channel, Connection):
            raise TypeError("TP worker requires one duplex control Connection")

    def timeout(self) -> float:
        value = self.timeout_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"TP control timeout must be positive, got {value!r}")
        return float(value)

    def rank0_channels(self) -> list[Connection]:
        if self.rank != 0 or self.world_size <= 1:
            raise RuntimeError("only TP rank 0 owns worker control channels")
        self._validate_topology()
        if not isinstance(self.channel, list):  # narrowed by _validate_topology
            raise AssertionError("rank 0 TP channel validation did not narrow to a list")
        return self.channel

    def next_command(
        self,
        method_name: str,
        args: tuple[object, ...],
    ) -> "TPCommand":
        try:
            method = TPMethod(method_name)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(item.value for item in TPMethod)
            raise ValueError(
                f"unsupported TP control method {method_name!r}; supported methods: {supported}"
            ) from exc
        command = TPCommand(
            request_id=self._next_request_id,
            method=method,
            args=args,
        )
        self._next_request_id += 1
        return command

    @staticmethod
    def deserialize_message(data: bytes) -> tuple[str, list[object]]:
        """Return the legacy method/args view of one typed command."""

        command = deserialize_command(data)
        return command.method.value, list(command.args)

    def read_command(self) -> "TPCommand":
        if self.world_size <= 1 or self.rank <= 0:
            raise RuntimeError("only TP worker ranks can read control commands")
        if not isinstance(self.channel, Connection):
            raise TypeError("TP worker control channel is not a Connection")
        try:
            data = self.channel.recv_bytes()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"TP worker rank {self.rank} lost its control channel") from exc
        return deserialize_command(data)

    def broadcast(
        self,
        method_name: str,
        args: tuple[object, ...],
    ) -> tuple["TPCommand", int]:
        if self.world_size <= 1 or self.rank != 0:
            raise RuntimeError("only TP rank 0 can broadcast control commands")
        command = self.next_command(method_name, args)
        try:
            data = serialize_command(command)
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to serialize TP control message for method={method_name!r}"
            ) from exc
        for worker_rank, channel in enumerate(self.rank0_channels(), start=1):
            try:
                channel.send_bytes(data)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(
                    "failed to send TP control message: "
                    f"worker_rank={worker_rank}, method={method_name!r}, "
                    f"request_id={command.request_id}, payload_bytes={len(data)}"
                ) from exc
        return command, len(data)

    def send_response(self, response: "TPResponse") -> None:
        if not isinstance(self.channel, Connection):
            raise TypeError("TP worker control channel is not a Connection")
        data = serialize_response(response)
        try:
            self.channel.send_bytes(data)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"TP worker rank {self.rank} failed to send command response"
            ) from exc

    def await_responses(self, command: "TPCommand") -> None:
        """Wait for one matching ACK/error from every worker with a deadline."""

        channels = self.rank0_channels()
        pending = {worker_rank: channel for worker_rank, channel in enumerate(channels, start=1)}
        deadline = perf_counter() + self.timeout()
        failures: list[TPResponse] = []
        while pending:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise self._timeout_error(command, pending)
            ready_channels = wait(list(pending.values()), timeout=remaining)
            if not ready_channels:
                raise self._timeout_error(command, pending)
            for channel in ready_channels:
                worker_rank = next(
                    rank for rank, candidate in pending.items() if candidate is channel
                )
                try:
                    response = deserialize_response(channel.recv_bytes())
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"TP worker rank {worker_rank} closed before ACK") from exc
                if response.worker_rank != worker_rank:
                    raise RuntimeError(
                        "TP response rank mismatch: "
                        f"channel_rank={worker_rank}, response_rank={response.worker_rank}"
                    )
                if response.request_id != command.request_id:
                    raise RuntimeError(
                        "stale or out-of-order TP response: "
                        f"worker_rank={worker_rank}, "
                        f"expected_request_id={command.request_id}, "
                        f"actual_request_id={response.request_id}"
                    )
                del pending[worker_rank]
                if response.status is TPResponseStatus.ERROR:
                    failures.append(response)
        if failures:
            details = "\n".join(
                f"rank {response.worker_rank}: {response.error_type}: "
                f"{response.error_message}\n{response.traceback_text}"
                for response in failures
            )
            raise RuntimeError(
                "TP worker command failed: "
                f"method={command.method.value!r}, "
                f"request_id={command.request_id}\n{details}"
            )

    @staticmethod
    def _timeout_error(
        command: "TPCommand",
        pending: dict[int, Connection],
    ) -> TimeoutError:
        return TimeoutError(
            "timed out waiting for TP worker responses: "
            f"method={command.method.value!r}, request_id={command.request_id}, "
            f"pending_ranks={sorted(pending)}"
        )

    def close_worker_channel(self) -> None:
        if isinstance(self.channel, Connection):
            self.channel.close()


def _request_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"TP control request_id must be a non-negative integer, got {value!r}")
    return value


def _worker_rank(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"TP worker rank must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class TPCommand:
    request_id: int
    method: TPMethod
    args: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _request_id(self.request_id)
        if not isinstance(self.method, TPMethod):
            raise TypeError("TP command method must be TPMethod")
        if not isinstance(self.args, tuple):
            raise TypeError("TP command args must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class TPResponse:
    request_id: int
    worker_rank: int
    status: TPResponseStatus
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _request_id(self.request_id)
        _worker_rank(self.worker_rank)
        if not isinstance(self.status, TPResponseStatus):
            raise TypeError("TP response status must be TPResponseStatus")
        error_fields = (
            self.error_type,
            self.error_message,
            self.traceback_text,
        )
        if self.status is TPResponseStatus.OK:
            if any(field is not None for field in error_fields):
                raise ValueError("successful TP response must not contain error fields")
            return
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("failed TP response requires error_type")
        if not isinstance(self.error_message, str):
            raise ValueError("failed TP response requires error_message")
        if not isinstance(self.traceback_text, str):
            raise ValueError("failed TP response requires traceback_text")

    @classmethod
    def ok(cls, command: TPCommand, *, worker_rank: int) -> "TPResponse":
        return cls(
            request_id=command.request_id,
            worker_rank=worker_rank,
            status=TPResponseStatus.OK,
        )

    @classmethod
    def error(
        cls,
        command: TPCommand,
        *,
        worker_rank: int,
        error: BaseException,
        traceback_text: str,
    ) -> "TPResponse":
        return cls(
            request_id=command.request_id,
            worker_rank=worker_rank,
            status=TPResponseStatus.ERROR,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_text=traceback_text,
        )


def _serialize(message: TPCommand | TPResponse, *, kind: str) -> bytes:
    try:
        return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, AttributeError, TypeError) as exc:
        raise RuntimeError(f"failed to serialize TP control {kind}") from exc


def serialize_command(command: TPCommand) -> bytes:
    if not isinstance(command, TPCommand):
        raise TypeError("command must be TPCommand")
    command.validate()
    return _serialize(command, kind="command")


def serialize_response(response: TPResponse) -> bytes:
    if not isinstance(response, TPResponse):
        raise TypeError("response must be TPResponse")
    response.validate()
    return _serialize(response, kind="response")


def _deserialize(data: bytes, *, expected_type: type, kind: str):
    if not isinstance(data, bytes):
        raise TypeError(f"serialized TP {kind} must be bytes")
    try:
        message = pickle.loads(data)
    except (
        pickle.PickleError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
    ) as exc:
        raise RuntimeError(f"failed to deserialize TP control {kind}") from exc
    if not isinstance(message, expected_type):
        raise ValueError(
            f"TP control {kind} must decode to {expected_type.__name__}, "
            f"got {type(message).__name__}"
        )
    message.validate()
    return message


def deserialize_command(data: bytes) -> TPCommand:
    return _deserialize(data, expected_type=TPCommand, kind="command")


def deserialize_response(data: bytes) -> TPResponse:
    return _deserialize(data, expected_type=TPResponse, kind="response")


__all__ = [
    "DEFAULT_TP_CONTROL_TIMEOUT_SECONDS",
    "TPCommand",
    "TPControlPlane",
    "TPMethod",
    "TPResponse",
    "TPResponseStatus",
    "deserialize_command",
    "deserialize_response",
    "serialize_command",
    "serialize_response",
]
