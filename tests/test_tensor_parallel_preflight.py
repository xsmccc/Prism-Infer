"""P6.8 Tensor Parallel 启动与变长控制通道门禁测试。"""

import pickle
import threading
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from types import SimpleNamespace

import pytest
import torch

from prism_infer.engine.llm_engine import validate_tensor_parallel_environment
from prism_infer.engine.model_runner import ModelRunner


def _config(tp_size: int = 2) -> SimpleNamespace:
    text_config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        intermediate_size=12288,
        vocab_size=151936,
    )
    return SimpleNamespace(
        tensor_parallel_size=tp_size,
        hf_config=SimpleNamespace(text_config=text_config),
    )


def test_tp_preflight_rejects_missing_visible_gpu() -> None:
    with pytest.raises(RuntimeError, match="only 1 are available"):
        validate_tensor_parallel_environment(_config(tp_size=2), device_count=1)
    print("P6.8 TP visible-device guard: PASS")


def test_tp_preflight_checks_all_sharded_dimensions() -> None:
    config = _config(tp_size=2)
    validate_tensor_parallel_environment(config, device_count=2)

    config.hf_config.text_config.num_key_value_heads = 7
    with pytest.raises(ValueError, match="num_key_value_heads"):
        validate_tensor_parallel_environment(config, device_count=2)
    print("P6.8 TP model-dimension guard: PASS")


def _runner(
    rank: int,
    channel: Connection | list[Connection],
) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size = 3
    runner.rank = rank
    runner.control_channel = channel
    return runner


def test_tp_variable_control_channel_broadcasts_large_vl_payload() -> None:
    receivers = []
    senders = []
    for _ in range(2):
        receiver, sender = Pipe(duplex=False)
        receivers.append(receiver)
        senders.append(sender)

    rank0 = _runner(rank=0, channel=senders)
    workers = [
        _runner(rank=index + 1, channel=receiver)
        for index, receiver in enumerate(receivers)
    ]
    sequence_like = SimpleNamespace(
        # [visual_tokens, vision_hidden], 超过旧协议的 1 MiB 上限。
        pixel_values=torch.arange(784 * 1536, dtype=torch.float32).view(784, 1536),
        prompt_token_ids=[1, 2, 3],
    )
    result: dict[str, int] = {}

    small_payload_bytes = rank0.write_control_message("copy_kv_blocks", [(1, 2)])
    small_messages = [worker.read_control_message() for worker in workers]
    assert small_payload_bytes < 2**20
    assert small_messages == [
        ("copy_kv_blocks", [[(1, 2)]]),
        ("copy_kv_blocks", [[(1, 2)]]),
    ]

    def _send() -> None:
        result["payload_bytes"] = rank0.write_control_message(
            "run",
            [sequence_like],
            True,
        )

    sender_thread = threading.Thread(target=_send)
    sender_thread.start()
    messages = [worker.read_control_message() for worker in workers]
    sender_thread.join(timeout=10)

    assert not sender_thread.is_alive()
    assert result["payload_bytes"] > 2**20
    for method_name, args in messages:
        received = args[0][0]
        assert method_name == "run"
        assert args[1] is True
        assert received.prompt_token_ids == sequence_like.prompt_token_ids
        assert torch.equal(received.pixel_values, sequence_like.pixel_values)
    for channel in [*receivers, *senders]:
        channel.close()
    print(
        "P6.8 TP variable control payload: "
        f"bytes={result['payload_bytes']}, workers={len(workers)}, PASS"
    )


def test_tp_control_channel_rejects_malformed_message() -> None:
    with pytest.raises(RuntimeError, match="deserialize"):
        ModelRunner._deserialize_control_message(b"not-a-pickle")
    with pytest.raises(ValueError, match="non-empty list"):
        ModelRunner._deserialize_control_message(pickle.dumps({"method": "run"}))
    with pytest.raises(ValueError, match="method name"):
        ModelRunner._deserialize_control_message(pickle.dumps([None]))

    print("P6.8 TP malformed control message guards: PASS")


def test_tp_control_channel_reports_serialize_and_send_failures() -> None:
    receiver, sender = Pipe(duplex=False)
    rank0 = _runner(rank=0, channel=[sender])
    rank0.world_size = 2

    with pytest.raises(RuntimeError, match="failed to serialize"):
        rank0.write_control_message("run", lambda: None)
    receiver.close()
    with pytest.raises(RuntimeError, match="worker_rank=1"):
        rank0.write_control_message("run", b"payload")
    sender.close()
    print("P6.8 TP serialize/send failure reporting: PASS")
