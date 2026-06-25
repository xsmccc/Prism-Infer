"""P3.4 长输出多 token greedy 质量验证。"""

import gc

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration

from conftest import get_model_path, require_transformers
from prism_infer import LLM
from prism_infer.sampling_params import SamplingParams
from test_processor_pipeline_video import demo_video_frames


MAX_TOKENS = 32
PREFIX_CHECKPOINTS = (8, 16, 32)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest = __import__("pytest")
        pytest.skip("long VL generate alignment requires CUDA")


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _cases() -> list[dict]:
    return [
        {
            "name": "single-image",
            "prompt": "Describe this image.",
            "messages_content": [
                {"type": "image", "image": _image((100, 150, 200))},
                {"type": "text", "text": "Describe this image."},
            ],
            "processor_kwargs": {
                "images": [_image((100, 150, 200))],
            },
            "prism_call": "generate_vl",
            "payload": _image((100, 150, 200)),
        },
        {
            "name": "multi-image",
            "prompt": "Compare these images.",
            "messages_content": [
                {"type": "image", "image": _image((100, 150, 200))},
                {"type": "image", "image": _image((200, 120, 80))},
                {"type": "text", "text": "Compare these images."},
            ],
            "processor_kwargs": {
                "images": [_image((100, 150, 200)), _image((200, 120, 80))],
            },
            "prism_call": "generate_images",
            "payload": [_image((100, 150, 200)), _image((200, 120, 80))],
        },
        {
            "name": "video",
            "prompt": "Describe this video.",
            "messages_content": [
                {"type": "video", "video": demo_video_frames()},
                {"type": "text", "text": "Describe this video."},
            ],
            "processor_kwargs": {
                "videos": [demo_video_frames()],
            },
            "prism_call": "generate_video",
            "payload": demo_video_frames(),
        },
    ]


def _hf_generate_tokens(model, processor, case: dict) -> tuple[list[int], int]:
    messages = [{"role": "user", "content": case["messages_content"]}]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_inputs = processor(
        text=prompt_text,
        return_tensors="pt",
        **case["processor_kwargs"],
    ).to("cuda")
    with torch.inference_mode():
        hf_output = model.generate(
            **hf_inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
        )
    token_ids = hf_output[0, hf_inputs["input_ids"].shape[1]:].tolist()
    prompt_len = int(hf_inputs["input_ids"].shape[1])
    del hf_output, hf_inputs
    torch.cuda.empty_cache()
    return token_ids, prompt_len


def _prism_generate_tokens(llm: LLM, case: dict) -> list[int]:
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    method = getattr(llm, case["prism_call"])
    output = method(case["prompt"], case["payload"], sampling, use_tqdm=False)
    return output["token_ids"]


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def test_vl_long_generate_eight_tokens_match_hf_greedy():
    """单图/多图/视频 max_tokens=8/16/32 greedy token ids 必须与 HF 完全一致。"""

    _require_cuda()
    transformers = require_transformers()
    model_path = get_model_path()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    hf_tokens_by_case = {}
    prompt_lens = {}
    try:
        for case in _cases():
            token_ids, prompt_len = _hf_generate_tokens(hf_model, processor, case)
            hf_tokens_by_case[case["name"]] = token_ids
            prompt_lens[case["name"]] = prompt_len
    finally:
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1280,
        max_num_batched_tokens=1280,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        for case in _cases():
            prism_tokens = _prism_generate_tokens(llm, case)
            hf_tokens = hf_tokens_by_case[case["name"]]
            mismatch = _first_mismatch(hf_tokens, prism_tokens)
            print(f"{case['name']} prompt tokens: {prompt_lens[case['name']]}")
            print(f"{case['name']} HF token_ids: {hf_tokens}")
            print(f"{case['name']} Prism token_ids: {prism_tokens}")
            print(f"{case['name']} first mismatch: {mismatch}")
            for checkpoint in PREFIX_CHECKPOINTS:
                prefix_match = prism_tokens[:checkpoint] == hf_tokens[:checkpoint]
                print(f"{case['name']} prefix@{checkpoint} match: {prefix_match}")
                assert prefix_match
            assert prism_tokens == hf_tokens
            print(f"{case['name']} long greedy {MAX_TOKENS} tokens: PASS")
    finally:
        llm.exit()
