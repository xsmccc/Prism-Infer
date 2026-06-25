"""P2.6 greedy sampler 验证。"""

import torch

from prism_infer.layers.sampler import Sampler
from prism_infer.sampling_params import SamplingParams


def test_sampling_params_allows_temperature_zero():
    """temperature=0 是 P2 greedy 对齐门禁需要的合法配置。"""

    params = SamplingParams(temperature=0.0, max_tokens=2)
    assert params.temperature == 0.0
    assert params.max_tokens == 2
    print("sampling params greedy temperature: PASS")


def test_sampler_temperature_zero_uses_argmax():
    """temperature=0 必须 deterministic argmax，不走随机采样。"""

    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 3.0, 2.0],
            [4.0, 1.0, 4.5],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.0], dtype=torch.float32)

    token_ids = sampler(logits, temperatures)
    print(f"greedy logits shape: {list(logits.shape)}")
    print(f"greedy token_ids: {token_ids.tolist()}")

    assert token_ids.tolist() == [1, 2]
    print("sampler greedy argmax: PASS")


def test_sampler_mixed_greedy_and_random_shapes():
    """同一 batch 允许部分请求 greedy、部分请求随机采样。"""

    torch.manual_seed(20260624)
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 3.0, 2.0],
            [4.0, 1.0, 4.5],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.7], dtype=torch.float32)

    token_ids = sampler(logits, temperatures)
    print(f"mixed logits shape: {list(logits.shape)}")
    print(f"mixed temperatures: {temperatures.tolist()}")
    print(f"mixed token_ids shape: {list(token_ids.shape)}")

    assert list(token_ids.shape) == [2]
    assert token_ids[0].item() == 1
    print("sampler mixed greedy/random: PASS")
