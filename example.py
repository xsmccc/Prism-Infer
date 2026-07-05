"""Minimal Qwen3-VL example for Prism-Infer.

Set PRISM_MODEL_PATH to a local Qwen3-VL-8B-Instruct snapshot containing
config.json, tokenizer/processor files, and model weights.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

from prism_infer import LLM, SamplingParams


def make_demo_image() -> Image.Image:
    image = Image.new("RGB", (448, 448), color=(70, 120, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 408, 408], outline=(255, 255, 255), width=8)
    draw.text((64, 64), "BLUE", fill=(255, 255, 255))
    return image


def main() -> None:
    model_path = os.environ.get("PRISM_MODEL_PATH")
    if not model_path:
        raise SystemExit("Set PRISM_MODEL_PATH to a local Qwen3-VL model snapshot")

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=False,
    )
    try:
        output = llm.generate_vl(
            "Describe this image in one short sentence.",
            make_demo_image(),
            SamplingParams(temperature=0.0, max_tokens=8),
            use_tqdm=False,
        )
    finally:
        llm.exit()

    print(f"Token IDs: {output['token_ids']}")
    print(f"Text: {output['text']!r}")


if __name__ == "__main__":
    main()
