"""Check the repaired request path on the existing Qwen3-VL-8B snapshot."""

import json
import os

from PIL import Image

from prism_infer import LLM, SamplingParams


def text_tokens(rows):
    return [row["token_ids"] for row in rows]


def main():
    llm = LLM(
        os.environ["PRISM_MODEL_PATH"],
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        max_chunk_size=256,
        num_kvcache_blocks=32,
        compression_mode="scaled_fp8_kv",
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=4)
    try:
        manager = llm.scheduler.block_manager
        text = "This is a reading exercise. " * 80 + "The capital of France is"
        manager.enable_prefix_caching = False
        text_reference = llm.generate([text, text], sampling, use_tqdm=False)
        manager.enable_prefix_caching = True
        text_cold = llm.generate([text, text], sampling, use_tqdm=False)
        text_warm = llm.generate([text, text], sampling, use_tqdm=False)
        assert text_tokens(text_reference) == text_tokens(text_cold) == text_tokens(text_warm)

        picture = Image.new("RGB", (448, 448), color=(240, 20, 20))
        question = "What is the main color? Answer with one color word."
        manager.enable_prefix_caching = False
        image_reference = llm.generate_vl(question, picture, sampling, use_tqdm=False)
        manager.enable_prefix_caching = True
        long_question = "Ignore irrelevant details in the request. " * 60 + question
        llm.generate_vl(long_question, picture, sampling, use_tqdm=False)
        before = manager.multimodal_prefix_cache_metadata()["hits"]
        image_warm = [
            llm.generate_vl(question, picture, sampling, use_tqdm=False) for _ in range(2)
        ]
        assert all(row["token_ids"] == image_reference["token_ids"] for row in image_warm)
        after = manager.multimodal_prefix_cache_metadata()["hits"]
        assert after - before == 2
        print(
            "MODEL_RESULT "
            + json.dumps(
                {
                    "model": "Qwen3-VL-8B-Instruct",
                    "compression_mode": "scaled_fp8_kv",
                    "execution": "eager, TP1, chunk_size=256, 32 KV pages",
                    "duplicate_text_reference": text_tokens(text_reference),
                    "duplicate_text_cold": text_tokens(text_cold),
                    "duplicate_text_warm": text_tokens(text_warm),
                    "image_reference": image_reference["token_ids"],
                    "short_question_after_long_question": [row["token_ids"] for row in image_warm],
                    "image_prefix_hits": after - before,
                }
            )
        )
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
