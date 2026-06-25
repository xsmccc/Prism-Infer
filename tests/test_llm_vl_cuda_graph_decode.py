"""P3.5 VL CUDA Graph decode correctness 验证。"""

import gc

import torch
from PIL import Image

from conftest import get_model_path
from prism_infer import LLM
from prism_infer.sampling_params import SamplingParams
from test_processor_pipeline_video import demo_video_frames


MAX_TOKENS = 2


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    pytest = __import__("pytest")
    pytest.skip("VL CUDA Graph decode test requires CUDA")


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _graph_kwargs(max_num_seqs: int = 1) -> dict:
    return {
        "tensor_parallel_size": 1,
        "max_model_len": 1280,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": 0.9,
        "enable_chunked_prefill": False,
    }


def _new_llm(model_path: str, *, enforce_eager: bool, max_num_seqs: int = 1) -> LLM:
    return LLM(
        model_path,
        enforce_eager=enforce_eager,
        **_graph_kwargs(max_num_seqs=max_num_seqs),
    )


def _run_single_case(model_path: str, case: dict, *, enforce_eager: bool) -> list[int]:
    llm = _new_llm(model_path, enforce_eager=enforce_eager, max_num_seqs=1)
    try:
        sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        if case["kind"] == "image":
            output = llm.generate_vl(case["prompt"], case["payload"], sampling, use_tqdm=False)
        elif case["kind"] == "images":
            output = llm.generate_images(case["prompt"], case["payload"], sampling, use_tqdm=False)
        elif case["kind"] == "video":
            output = llm.generate_video(case["prompt"], case["payload"], sampling, use_tqdm=False)
        else:
            raise ValueError(case["kind"])
        return output["token_ids"]
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()


def _cases() -> list[dict]:
    return [
        {
            "name": "single-image",
            "kind": "image",
            "prompt": "Describe this image.",
            "payload": _image((100, 150, 200)),
        },
        {
            "name": "multi-image",
            "kind": "images",
            "prompt": "Compare these images.",
            "payload": [_image((100, 150, 200)), _image((200, 120, 80))],
        },
        {
            "name": "video",
            "kind": "video",
            "prompt": "Describe this video.",
            "payload": demo_video_frames(),
        },
    ]


def _mixed_requests() -> list[dict]:
    return [
        {"type": "text", "prompt": "Hello"},
        {
            "type": "images",
            "prompt": "Compare these images.",
            "images": [_image((100, 150, 200)), _image((200, 120, 80))],
        },
        {"type": "video", "prompt": "Describe this video.", "video": demo_video_frames()},
    ]


def test_vl_cuda_graph_single_multi_video_match_eager() -> None:
    """单图/多图/视频 graph decode token ids 必须与 eager 完全一致。"""

    _require_cuda()
    model_path = get_model_path()
    for case in _cases():
        eager_tokens = _run_single_case(model_path, case, enforce_eager=True)
        graph_tokens = _run_single_case(model_path, case, enforce_eager=False)
        print(f"{case['name']} eager token_ids: {eager_tokens}")
        print(f"{case['name']} graph token_ids: {graph_tokens}")
        print(f"{case['name']} graph input coverage: max_tokens={MAX_TOKENS}, batch=1")
        assert graph_tokens == eager_tokens
        print(f"{case['name']} VL CUDA Graph decode token equivalence: PASS")


def test_vl_cuda_graph_mixed_batch_matches_eager() -> None:
    """text/single-image/multi-image/video mixed batch graph decode 应对齐 eager。"""

    _require_cuda()
    model_path = get_model_path()
    requests = _mixed_requests()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    eager_llm = _new_llm(model_path, enforce_eager=True, max_num_seqs=4)
    try:
        eager_outputs = eager_llm.generate_mixed(requests, sampling, use_tqdm=False)
    finally:
        eager_llm.exit()
        gc.collect()
        torch.cuda.empty_cache()

    graph_llm = _new_llm(model_path, enforce_eager=False, max_num_seqs=4)
    try:
        graph_outputs = graph_llm.generate_mixed(requests, sampling, use_tqdm=False)
    finally:
        graph_llm.exit()
        gc.collect()
        torch.cuda.empty_cache()

    eager_ids = [output["token_ids"] for output in eager_outputs]
    graph_ids = [output["token_ids"] for output in graph_outputs]
    print(f"mixed eager token_ids: {eager_ids}")
    print(f"mixed graph token_ids: {graph_ids}")
    print(f"mixed graph batch size: {len(requests)}")
    print("mixed graph replay rounding: requested batch=3, replay graph batch=4")

    assert graph_ids == eager_ids
    print("LLM.generate_mixed VL CUDA Graph decode equivalence: PASS")
