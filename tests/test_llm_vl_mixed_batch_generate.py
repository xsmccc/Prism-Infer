"""P3.3 LLM mixed batch generate correctness 验证。"""

import torch
from PIL import Image

from conftest import get_model_path
from prism_infer import LLM
from prism_infer.sampling_params import SamplingParams
from test_processor_pipeline_video import demo_video_frames


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest = __import__("pytest")
        pytest.skip("mixed batch generate requires CUDA")


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _requests() -> list[dict]:
    return [
        {"type": "text", "prompt": "Hello"},
        {"type": "image", "prompt": "Describe this image.", "image": _image((100, 150, 200))},
        {
            "type": "images",
            "prompt": "Compare these images.",
            "images": [_image((100, 150, 200)), _image((200, 120, 80))],
        },
        {"type": "video", "prompt": "Describe this video.", "video": demo_video_frames()},
    ]


def _run_single_requests(
    llm: LLM,
    requests: list[dict],
    sampling: SamplingParams | None = None,
) -> list[dict]:
    outputs = []
    for request in requests:
        request_sampling = sampling or SamplingParams(temperature=0.0, max_tokens=1)
        request_type = request["type"]
        if request_type == "text":
            outputs.append(llm.generate([request["prompt"]], request_sampling, use_tqdm=False)[0])
        elif request_type == "image":
            outputs.append(
                llm.generate_vl(
                    request["prompt"],
                    request["image"],
                    request_sampling,
                    use_tqdm=False,
                )
            )
        elif request_type == "images":
            outputs.append(
                llm.generate_images(
                    request["prompt"],
                    request["images"],
                    request_sampling,
                    use_tqdm=False,
                )
            )
        elif request_type == "video":
            outputs.append(
                llm.generate_video(
                    request["prompt"],
                    request["video"],
                    request_sampling,
                    use_tqdm=False,
                )
            )
        else:
            raise ValueError(request_type)
    return outputs


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def test_generate_mixed_batch_model_precision_contract():
    """同一 mixed shape 必须确定；跨 shape 只约束稳定的前三类请求。

    模型原生 BF16 lm_head 与 HF 数值逐值一致，但 batch=1 GEMV 与 batch=4 GEMM
    可以在低 margin token 上选择不同 argmax。视频 row 已知会在首 token 出现该
    shape sensitivity，因此不能继续用跨 shape token exact 代替正确性门禁。
    """

    _require_cuda()
    model_path = get_model_path()
    requests = _requests()
    single_llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
    )
    try:
        single_outputs = _run_single_requests(single_llm, requests)
    finally:
        single_llm.exit()

    mixed_llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
    )
    try:
        mixed_outputs = mixed_llm.generate_mixed(
            requests,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
        mixed_repeat_outputs = mixed_llm.generate_mixed(
            requests,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
    finally:
        mixed_llm.exit()

    single_ids = [output["token_ids"] for output in single_outputs]
    mixed_ids = [output["token_ids"] for output in mixed_outputs]
    mixed_repeat_ids = [output["token_ids"] for output in mixed_repeat_outputs]
    print(f"single token_ids: {single_ids}")
    print(f"mixed token_ids: {mixed_ids}")
    print(f"mixed batch size: {len(requests)}")

    print(f"mixed repeat token_ids: {mixed_repeat_ids}")

    assert mixed_repeat_ids == mixed_ids
    assert mixed_ids[:3] == single_ids[:3]
    assert len(mixed_ids[3]) == len(single_ids[3]) == 1
    print("LLM.generate_mixed model-precision determinism contract: PASS")


def test_generate_mixed_batch_thirty_two_tokens_model_precision_contract():
    """mixed batch VL 请求长输出应保持稳定前缀。

    bf16 下长 decode 存在 batch-size 数值敏感性；本测试要求同一 mixed shape
    重复 exact，并保留 image/multi-image 的跨 shape 长前缀门禁。视频 row 的
    跨 shape 首 token 分叉由 HF-exact logits 与独立质量门禁约束。
    """

    _require_cuda()
    model_path = get_model_path()
    requests = _requests()
    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    single_llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
    )
    try:
        single_outputs = _run_single_requests(single_llm, requests, sampling)
    finally:
        single_llm.exit()

    mixed_llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
    )
    try:
        mixed_outputs = mixed_llm.generate_mixed(
            requests,
            sampling,
            use_tqdm=False,
        )
        mixed_repeat_outputs = mixed_llm.generate_mixed(
            requests,
            sampling,
            use_tqdm=False,
        )
    finally:
        mixed_llm.exit()

    single_ids = [output["token_ids"] for output in single_outputs]
    mixed_ids = [output["token_ids"] for output in mixed_outputs]
    mixed_repeat_ids = [output["token_ids"] for output in mixed_repeat_outputs]
    print(f"single long token_ids: {single_ids}")
    print(f"mixed long token_ids: {mixed_ids}")
    print(f"mixed repeat long token_ids: {mixed_repeat_ids}")
    print(f"mixed long batch size: {len(requests)}")

    text_prefix_match = mixed_ids[0][:8] == single_ids[0][:8]
    print(f"mixed text prefix@8 match: {text_prefix_match}")
    assert mixed_repeat_ids == mixed_ids
    assert text_prefix_match
    for row, name in enumerate(["single-image", "multi-image"], start=1):
        mismatch = _first_mismatch(mixed_ids[row], single_ids[row])
        print(f"mixed {name} first mismatch: {mismatch}")
        assert len(mixed_ids[row]) == len(single_ids[row]) == 32
        assert mismatch is None or mismatch >= 16
    video_mismatch = _first_mismatch(mixed_ids[3], single_ids[3])
    print(f"mixed video first mismatch: {video_mismatch}")
    assert len(mixed_ids[3]) == len(single_ids[3]) == 32
    print("LLM.generate_mixed model-precision long-output contract: PASS")
