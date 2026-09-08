"""Shared CPU preprocessing checks without model weights or GPU execution."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import torch
from PIL import Image

from prism_infer.engine import media_preprocessing
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.media_preprocessing import MediaPreprocessingCache
from prism_infer.engine.online import OnlineRequest, OnlineServingSession
from prism_infer.engine.request import MonotonicRequestIdAllocator
from prism_infer.engine.vl_inputs import ImageInputs, VideoInputs
from prism_infer.sampling_params import SamplingParams

IMAGE_TOKEN = 151655
PLACEHOLDER = "<|image_pad|>"


class _Tokenizer:
    def __call__(self, text, **_options):
        chunks = text.split(PLACEHOLDER)
        ids = []
        for index, chunk in enumerate(chunks):
            if index:
                ids.append(IMAGE_TOKEN)
            ids.extend(ord(character) for character in chunk)
        return {"input_ids": ids}


def _image_inputs(prompt, media, *, image_marker=None):
    images = list(media) if isinstance(media, list | tuple) else [media]
    if image_marker is None:
        prompt_text = PLACEHOLDER * len(images) + prompt + "<assistant>"
    else:
        prompt_text = prompt.replace(image_marker, PLACEHOLDER) + "<assistant>"
    template_ids = _Tokenizer()(prompt_text)["input_ids"]
    ids = [
        value
        for token in template_ids
        for value in ([token] * 4 if token == IMAGE_TOKEN else [token])
    ]
    input_ids = torch.tensor([ids], dtype=torch.long)
    pixel_parts = []
    for image in images:
        values = list(image.convert("RGB").tobytes())
        repeated = (values * ((48 + len(values) - 1) // len(values)))[:48]
        pixel_parts.append(torch.tensor(repeated, dtype=torch.float32).reshape(16, 3))
    return ImageInputs(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.cat(pixel_parts),
        image_grid_thw=torch.tensor([[1, 4, 4]] * len(images)),
        image_token_id=IMAGE_TOKEN,
        image_token_count=4 * len(images),
        expected_image_tokens=4 * len(images),
        prompt_text=prompt_text,
    )


def _engine(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(kvcache_block_size=4, hf_config=SimpleNamespace())
    engine.vl_processor = SimpleNamespace(tokenizer=_Tokenizer())
    engine._media_preprocess_cache = MediaPreprocessingCache("test-shared")
    engine.request_id_allocator = MonotonicRequestIdAllocator()
    calls = []
    submitted = []

    def process_image(prompt, media):
        calls.append(("images", prompt))
        return _image_inputs(prompt, media)

    def process_interleaved(prompt, media, *, image_marker):
        calls.append(("interleaved_images", prompt, image_marker))
        return _image_inputs(prompt, media, image_marker=image_marker)

    def submit(sequence, **_options):
        submitted.append(sequence)
        return sequence.seq_id

    def positions(input_ids, **_options):
        length = input_ids.shape[-1]
        return torch.arange(length).repeat(3, 1), torch.tensor([length])

    engine._process_image_inputs = process_image
    engine._process_interleaved_image_inputs = process_interleaved
    engine._submit_sequence = submit
    monkeypatch.setattr(
        "prism_infer.engine.llm_engine.get_qwen3_vl_rope_index_from_config", positions
    )
    return engine, calls, submitted


def test_public_online_and_cpu_interface_share_results_and_identity(monkeypatch):
    engine, calls, submitted = _engine(monkeypatch)
    picture = Image.new("RGB", (2, 2), color=(30, 40, 50))
    sampling = SamplingParams(max_tokens=2)
    identity_calls = []
    original = media_preprocessing._visual_embedding_fingerprint
    original_per_image = media_preprocessing._per_image_media_hashes

    def fingerprint(*args, **kwargs):
        identity_calls.append(("group", args[1]))
        return original(*args, **kwargs)

    def per_image(*args, **kwargs):
        identity_calls.append(("per_image", args[1]))
        return original_per_image(*args, **kwargs)

    def unexpected_identity_rescan(_inputs):
        raise AssertionError("prepared cached media was hashed again during Sequence construction")

    monkeypatch.setattr(media_preprocessing, "_visual_embedding_fingerprint", fingerprint)
    monkeypatch.setattr(media_preprocessing, "_per_image_media_hashes", per_image)
    engine._image_media_identity = unexpected_identity_rescan
    engine.add_images_request("Describe.", [picture], sampling)
    first = submitted[0]
    second = OnlineServingSession(engine)._prepare_media_sequence(
        OnlineRequest(
            request_key="online",
            arrival_offset_s=0,
            payload={"type": "image", "prompt": "Describe.", "image": picture.copy()},
            sampling_params=sampling,
        ),
        engine._allocate_request_id(),
    )
    question = "What is the main color?"
    third = engine._prepare_media_request(
        "images",
        question,
        [picture.copy()],
        sampling,
        request_id=engine._allocate_request_id(),
    )

    assert len(calls) == 1
    assert identity_calls == [("group", "images"), ("per_image", "images")]
    assert len({first.seq_id, second.seq_id, third.seq_id}) == 3
    assert first.pixel_values is second.pixel_values is third.pixel_values
    assert first.image_grid_thw is third.image_grid_thw
    assert first.multimodal_prefix_cache_key == third.multimodal_prefix_cache_key
    assert first.multimodal_media_token_hashes is not None
    assert first.multimodal_media_token_hashes == third.multimodal_media_token_hashes
    assert third.prompt_token_ids == _image_inputs(question, [picture]).token_ids
    assert third.position_ids.shape[-1] == len(third.prompt_token_ids)
    assert third.position_ids is not first.position_ids
    assert third.rope_delta.item() == len(third.prompt_token_ids)
    assert engine.media_preprocess_cache_metadata()["hits"] == 2
    assert engine.media_preprocess_cache_metadata()["prompt_rebind_hits"] == 1


def test_same_pil_object_changed_pixels_do_not_hit_cache(monkeypatch):
    engine, calls, _submitted = _engine(monkeypatch)
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    sampling = SamplingParams(max_tokens=1)
    before = engine._prepare_media_request("images", "Describe.", image, sampling, request_id=0)
    image.putpixel((0, 0), (200, 201, 202))
    after = engine._prepare_media_request("images", "Describe.", image, sampling, request_id=1)

    assert len(calls) == 2
    assert not torch.equal(before.pixel_values, after.pixel_values)
    assert before.multimodal_prefix_cache_key != after.multimodal_prefix_cache_key
    assert engine.media_preprocess_cache_metadata()["hits"] == 0


def test_same_palette_image_with_changed_colors_does_not_hit_cache(monkeypatch):
    engine, calls, _submitted = _engine(monkeypatch)
    image = Image.new("P", (2, 2), color=0)
    palette = [10, 20, 30] + [0] * 765
    image.putpalette(palette)
    original_indices = image.tobytes()
    sampling = SamplingParams(max_tokens=1)
    before = engine._prepare_media_request("images", "Describe.", image, sampling, request_id=0)
    palette[:3] = [70, 80, 90]
    image.putpalette(palette)
    assert image.tobytes() == original_indices

    after = engine._prepare_media_request("images", "Describe.", image, sampling, request_id=1)

    assert len(calls) == 2
    assert not torch.equal(before.pixel_values, after.pixel_values)
    assert before.multimodal_prefix_cache_key != after.multimodal_prefix_cache_key
    assert engine.media_preprocess_cache_metadata()["hits"] == 0


def test_interleaved_marker_and_changed_layout_keep_full_processor_fallback(monkeypatch):
    engine, calls, _submitted = _engine(monkeypatch)
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    sampling = SamplingParams(max_tokens=1)
    prompt = "A: <image> B: [img] Describe."
    first = engine._prepare_media_request(
        "interleaved_images",
        prompt,
        [image],
        sampling,
        request_id=0,
        image_marker="<image>",
    )
    second = engine._prepare_media_request(
        "interleaved_images",
        prompt,
        [image],
        sampling,
        request_id=1,
        image_marker="[img]",
    )
    changed = "A different introduction: <image> Describe."
    third = engine._prepare_media_request(
        "interleaved_images",
        changed,
        [image],
        sampling,
        request_id=2,
    )

    assert len(calls) == 3
    assert first.prompt_token_ids != second.prompt_token_ids
    expected = _image_inputs(changed, [image], image_marker="<image>")
    assert third.prompt_token_ids == expected.token_ids
    assert first.multimodal_media_token_hashes == second.multimodal_media_token_hashes


def test_concurrent_cold_processing_runs_outside_cache_lock(monkeypatch):
    engine, _calls, _submitted = _engine(monkeypatch)
    rendezvous = Barrier(2, timeout=5)
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    sampling = SamplingParams(max_tokens=1)

    def process_image(prompt, media):
        rendezvous.wait()
        return _image_inputs(prompt, media)

    engine._process_image_inputs = process_image
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                engine._prepare_media_request,
                "images",
                "Describe.",
                image,
                sampling,
                request_id=request_id,
            )
            for request_id in range(2)
        ]
        sequences = [future.result() for future in futures]

    assert sequences[0].prompt_token_ids == sequences[1].prompt_token_ids
    assert torch.equal(sequences[0].pixel_values, sequences[1].pixel_values)
    assert engine.media_preprocess_cache_metadata()["entries"] == 1
    assert engine.media_preprocess_cache_metadata()["misses"] == 2


def test_public_video_and_cpu_preparation_use_the_same_cache(monkeypatch):
    engine, _calls, submitted = _engine(monkeypatch)
    calls = []
    frame = Image.new("RGB", (2, 2), color=(10, 20, 30))
    sampling = SamplingParams(max_tokens=1)

    def process_video(prompt, media):
        calls.append((prompt, media))
        ids = torch.tensor([[151656, 7]])
        return VideoInputs(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values_videos=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            video_grid_thw=torch.tensor([[1, 2, 2]]),
            video_token_id=151656,
            video_token_count=1,
            expected_video_tokens=1,
            prompt_text=prompt,
        )

    engine._process_video_inputs = process_video
    engine.add_video_request("Describe.", [frame], sampling)
    prepared = engine._prepare_media_request(
        "video",
        "Describe.",
        [frame.copy()],
        sampling,
        request_id=engine._allocate_request_id(),
    )
    assert len(calls) == 1
    assert prepared.pixel_values_videos is submitted[0].pixel_values_videos
    assert prepared.multimodal_prefix_cache_key == submitted[0].multimodal_prefix_cache_key
    assert prepared.multimodal_media_token_hashes is None
