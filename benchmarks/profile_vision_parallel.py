"""Capture one cold request and one full visual-prefix hit after model warmup."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

from benchmarks.bench_vision_parallel import DEFAULT_PROMPT, FIXTURE_COLORS, _run_requests
from prism_infer import LLM, SamplingParams
from prism_infer.observability.performance import install_performance_provider


@contextmanager
def _nvtx_region(name, *, cuda=True, metadata=None):
    # This module also runs in spawned TP workers, so both ranks carry labels.
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


install_performance_provider(
    profile_region_provider=_nvtx_region,
    profile_session_provider=lambda: None,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = [Image.new("RGB", (448, 448), color) for color in FIXTURE_COLORS]
    llm = LLM(
        args.model,
        tensor_parallel_size=2,
        vision_encoder_parallel_mode="data",
        execution_backend="cuda_graph",
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        num_kvcache_blocks=64,
        compression_mode="scaled_fp8_kv",
        enable_prefix_caching=True,
        enable_visual_embedding_cache=False,
        enable_chunked_prefill=False,
        paged_decode_block_n=256,
        enable_fused_qk_rmsnorm=True,
        enable_fused_qk_mrope=True,
        enable_fused_add_rmsnorm=True,
        enable_packed_kv_projection=True,
    )
    rows = []
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

    def run(phase, prefix):
        with _nvtx_region(phase):
            _run_requests(
                llm,
                images,
                [DEFAULT_PROMPT],
                sampling,
                phase=phase,
                prefix_enabled=prefix,
                output_rows=rows,
                replica_id="0",
            )

    try:
        run("warmup", False)
        run("prime", True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        run("trace_cold", False)
        run("trace_full_prefix_hit", True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    finally:
        llm.exit()
        for image in images:
            image.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"requests": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
