"""P2.6 LLM.generate_vl 单图入口验证。"""

from types import SimpleNamespace

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration

from conftest import get_model_path, require_transformers
from prism_infer import LLM
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.scheduler import Scheduler
from prism_infer.sampling_params import SamplingParams


def _make_minimal_engine() -> LLMEngine:
    transformers = require_transformers()
    model_path = get_model_path()
    config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        model=model_path,
        hf_config=config,
        enforce_eager=True,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        num_cpu_blocks=0,
        enable_chunked_prefill=True,
        max_chunk_size=512,
        max_queue_size=None,
        max_consecutive_prefill_batches=1,
        enable_prefix_caching=True,
        eos=-1,
    )
    engine.vl_processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    engine.model_runner = SimpleNamespace(is_vl_model=True)
    engine.scheduler = Scheduler(engine.config)
    return engine


def test_add_vl_request_builds_single_image_sequence():
    """add_vl_request 应构造带 VL payload 和 rope_delta 的 Sequence。"""

    engine = _make_minimal_engine()
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    seq_id = engine.add_vl_request(
        "Describe this image.",
        image,
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    seq = engine.scheduler.waiting[0]

    print(f"vl seq_id: {seq_id}")
    print(f"vl token count: {len(seq)}")
    print(f"vl pixel_values shape: {list(seq.pixel_values.shape)}")
    print(f"vl image_grid_thw shape: {list(seq.image_grid_thw.shape)}")
    print(f"vl position_ids shape: {list(seq.position_ids.shape)}")
    print(f"vl rope_delta shape: {list(seq.rope_delta.shape)}")

    assert seq.seq_id == seq_id
    assert seq.is_multimodal
    assert seq.temperature == 0.0
    assert list(seq.position_ids.shape) == [3, 1, len(seq)]
    assert list(seq.rope_delta.shape) == [1, 1]
    assert list(seq.image_grid_thw.shape) == [1, 3]
    print("LLMEngine add_vl_request: PASS")


def test_add_vl_request_builds_multi_image_sequence():
    """add_vl_request 应支持单请求多图 payload 和 rope_delta。"""

    engine = _make_minimal_engine()
    images = [
        Image.new("RGB", (448, 448), color=(100, 150, 200)),
        Image.new("RGB", (448, 448), color=(200, 120, 80)),
    ]
    seq_id = engine.add_vl_request(
        "Compare these images.",
        images,
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    seq = engine.scheduler.waiting[0]

    print(f"multi vl seq_id: {seq_id}")
    print(f"multi vl token count: {len(seq)}")
    print(f"multi vl pixel_values shape: {list(seq.pixel_values.shape)}")
    print(f"multi vl image_grid_thw shape: {list(seq.image_grid_thw.shape)}")
    print(f"multi vl position_ids shape: {list(seq.position_ids.shape)}")
    print(f"multi vl rope_delta shape: {list(seq.rope_delta.shape)}")

    assert seq.seq_id == seq_id
    assert seq.is_multimodal
    assert seq.temperature == 0.0
    assert list(seq.image_grid_thw.shape) == [2, 3]
    assert list(seq.position_ids.shape) == [3, 1, len(seq)]
    assert list(seq.rope_delta.shape) == [1, 1]
    assert seq.image_token_count == 392
    print("LLMEngine add_vl_request multi image: PASS")


def _demo_video_frames() -> list[Image.Image]:
    return [
        Image.new("RGB", (448, 448), color=(80 + i * 30, 120, 180))
        for i in range(4)
    ]


def test_add_video_request_builds_video_sequence():
    """add_video_request 应支持单请求视频 payload 和 rope_delta。"""

    engine = _make_minimal_engine()
    frames = _demo_video_frames()
    seq_id = engine.add_video_request(
        "Describe this video.",
        frames,
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    seq = engine.scheduler.waiting[0]

    print(f"video vl seq_id: {seq_id}")
    print(f"video vl token count: {len(seq)}")
    print(f"video vl pixel_values_videos shape: {list(seq.pixel_values_videos.shape)}")
    print(f"video vl video_grid_thw shape: {list(seq.video_grid_thw.shape)}")
    print(f"video vl position_ids shape: {list(seq.position_ids.shape)}")
    print(f"video vl rope_delta shape: {list(seq.rope_delta.shape)}")

    assert seq.seq_id == seq_id
    assert seq.is_multimodal
    assert seq.temperature == 0.0
    assert list(seq.video_grid_thw.shape) == [1, 3]
    assert list(seq.position_ids.shape) == [3, 1, len(seq)]
    assert list(seq.rope_delta.shape) == [1, 1]
    assert seq.video_token_count == 392
    print("LLMEngine add_video_request: PASS")


def test_add_vl_request_allows_graph_mode_sequence_building():
    """P3.5 后 VL 请求在 graph mode 下也应能构造 Sequence。"""

    engine = _make_minimal_engine()
    engine.config.enforce_eager = False
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))

    seq_id = engine.add_vl_request(
        "Describe this image.",
        image,
        SamplingParams(temperature=0.0),
    )
    seq = engine.scheduler.waiting[0]

    print(f"graph mode vl seq_id: {seq_id}")
    print(f"graph mode vl position_ids shape: {list(seq.position_ids.shape)}")
    print(f"graph mode vl rope_delta shape: {list(seq.rope_delta.shape)}")

    assert seq.seq_id == seq_id
    assert list(seq.position_ids.shape) == [3, 1, len(seq)]
    assert list(seq.rope_delta.shape) == [1, 1]
    print("LLMEngine generate_vl graph mode sequence building: PASS")


def test_generate_vl_one_token_matches_hf_greedy():
    """单图 generate_vl 第一个 greedy token 必须与 HF 完全一致。"""

    if not torch.cuda.is_available():
        pytest = __import__("pytest")
        pytest.skip("generate_vl token alignment requires CUDA")

    transformers = require_transformers()
    model_path = get_model_path()
    image = Image.new("RGB", (448, 448), color=(100, 150, 200))
    prompt = "Describe this image."

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to("cuda")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=1,
            do_sample=False,
        )
    hf_token_ids = hf_output[0, hf_inputs["input_ids"].shape[1]:].tolist()
    del hf_model, hf_output, hf_inputs
    torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        our_output = llm.generate_vl(
            prompt,
            image,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
    finally:
        llm.exit()

    print(f"HF token_ids: {hf_token_ids}")
    print(f"Prism token_ids: {our_output['token_ids']}")

    assert our_output["token_ids"] == hf_token_ids
    print("LLM.generate_vl one-token greedy HF alignment: PASS")



def test_generate_vl_multi_image_one_token_matches_hf_greedy():
    """多图 generate_vl 第一个 greedy token 必须与 HF 完全一致。"""

    if not torch.cuda.is_available():
        pytest = __import__("pytest")
        pytest.skip("multi-image generate_vl token alignment requires CUDA")

    transformers = require_transformers()
    model_path = get_model_path()
    images = [
        Image.new("RGB", (448, 448), color=(100, 150, 200)),
        Image.new("RGB", (448, 448), color=(200, 120, 80)),
    ]
    prompt = "Compare these images."

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_inputs = processor(text=prompt_text, images=images, return_tensors="pt").to("cuda")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=1,
            do_sample=False,
        )
    hf_token_ids = hf_output[0, hf_inputs["input_ids"].shape[1]:].tolist()
    del hf_model, hf_output, hf_inputs
    torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=768,
        max_num_batched_tokens=768,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        our_output = llm.generate_vl(
            prompt,
            images,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
    finally:
        llm.exit()

    print(f"HF multi-image token_ids: {hf_token_ids}")
    print(f"Prism multi-image token_ids: {our_output['token_ids']}")

    assert our_output["token_ids"] == hf_token_ids
    print("LLM.generate_vl multi-image one-token greedy HF alignment: PASS")


def test_generate_video_one_token_matches_hf_greedy():
    """视频 generate_video 第一个 greedy token 必须与 HF 完全一致。"""

    if not torch.cuda.is_available():
        pytest = __import__("pytest")
        pytest.skip("video generate token alignment requires CUDA")

    transformers = require_transformers()
    model_path = get_model_path()
    frames = _demo_video_frames()
    prompt = "Describe this video."

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_inputs = processor(text=prompt_text, videos=[frames], return_tensors="pt").to("cuda")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=1,
            do_sample=False,
        )
    hf_token_ids = hf_output[0, hf_inputs["input_ids"].shape[1]:].tolist()
    del hf_model, hf_output, hf_inputs
    torch.cuda.empty_cache()

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=768,
        max_num_batched_tokens=768,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        our_output = llm.generate_video(
            prompt,
            frames,
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
    finally:
        llm.exit()

    print(f"HF video token_ids: {hf_token_ids}")
    print(f"Prism video token_ids: {our_output['token_ids']}")

    assert our_output["token_ids"] == hf_token_ids
    print("LLM.generate_video one-token greedy HF alignment: PASS")
