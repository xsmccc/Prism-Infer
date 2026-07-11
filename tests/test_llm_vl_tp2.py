"""P6.8 Qwen3-VL TP1/TP2 真实模型 greedy smoke test。"""

import gc
import os

import pytest
import torch
from PIL import Image

from conftest import get_model_path
from prism_infer import LLM
from prism_infer.sampling_params import SamplingParams


def _require_tp2_environment() -> None:
    """只在显式请求且至少两张 GPU 可见时运行重型 TP2 测试。"""

    if os.environ.get("PRISM_RUN_TP2") != "1":
        pytest.skip("set PRISM_RUN_TP2=1 to run the TP2 integration test")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP2 integration test requires at least two visible CUDA devices")


def _generate_tokens(model_path: str, tp_size: int) -> list[int]:
    """用固定单图请求生成 greedy token，并完整释放当前 TP engine。"""

    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
    )
    try:
        output = llm.generate_vl(
            "Describe this image.",
            image,
            SamplingParams(temperature=0.0, max_tokens=8),
            use_tqdm=False,
        )
        return output["token_ids"]
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def test_qwen3_vl_tp2_greedy_matches_tp1() -> None:
    """同一 VL prompt 的 TP1/TP2 greedy token ids 必须完全一致。"""

    _require_tp2_environment()
    model_path = get_model_path()

    tp1_token_ids = _generate_tokens(model_path, tp_size=1)
    tp2_token_ids = _generate_tokens(model_path, tp_size=2)

    print(f"TP1 token ids: {tp1_token_ids}")
    print(f"TP2 token ids: {tp2_token_ids}")
    assert tp2_token_ids == tp1_token_ids
    print("P6.8 Qwen3-VL TP1/TP2 greedy token exact: PASS")
