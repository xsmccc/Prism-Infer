"""P7.3 real-engine online text/VL integration gates."""

import torch

try:
    import pytest
except ImportError:
    pytest = None

from benchmarks.harness import materialize_requests
from conftest import get_model_path
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.engine.online import OnlineRequest, OnlineServingSession


pytestmark = (
    []
    if pytest is None
    else [
        pytest.mark.model,
        pytest.mark.gpu,
        pytest.mark.integration,
        pytest.mark.slow,
    ]
)

MANIFEST = "benchmarks/workloads/p7_online_smoke.json"


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    if pytest is not None:
        pytest.skip("online engine integration requires CUDA")
    raise SystemExit("SKIP: online engine integration requires CUDA")


def _case(case_id: str) -> dict:
    manifest = load_workload_manifest(MANIFEST)
    return next(case for case in manifest["cases"] if case["id"] == case_id)


def _llm(*, max_model_len: int, max_chunk_size: int, max_num_seqs: int, blocks: int):
    return LLM(
        get_model_path(),
        enforce_eager=True,
        compression_mode="off",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=blocks,
        kvcache_block_size=256,
        enable_chunked_prefill=True,
        max_chunk_size=max_chunk_size,
        enable_prefix_caching=False,
        logits_precision="model",
    )


def test_online_long_text_chunked_matches_single_prefill() -> None:
    _require_cuda()
    prompt = _case("text_long_chunked")["requests"][0]["prompt"]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
    )

    chunked = _llm(
        max_model_len=512,
        max_chunk_size=128,
        max_num_seqs=1,
        blocks=4,
    )
    try:
        chunked_run = OnlineServingSession(chunked).run(
            (
                OnlineRequest(
                    request_key="chunked",
                    arrival_offset_s=0.0,
                    payload={"type": "text", "prompt": prompt},
                    sampling_params=sampling,
                ),
            )
        )
    finally:
        chunked.exit()

    full = _llm(
        max_model_len=512,
        max_chunk_size=512,
        max_num_seqs=1,
        blocks=4,
    )
    try:
        full_run = OnlineServingSession(full).run(
            (
                OnlineRequest(
                    request_key="full",
                    arrival_offset_s=0.0,
                    payload={"type": "text", "prompt": prompt},
                    sampling_params=sampling,
                ),
            )
        )
    finally:
        full.exit()

    chunked_tokens = chunked_run.requests[0].token_ids
    full_tokens = full_run.requests[0].token_ids
    prefill_counts = [
        batch["scheduled_tokens"]
        for batch in chunked_run.engine_metrics["batches"]
        if batch["phase"] == "prefill"
    ]
    assert prefill_counts == [128, 128, 45]
    assert chunked_tokens == full_tokens
    print(f"online long-text chunk sizes: {prefill_counts}")
    print(f"online chunked/full tokens: {chunked_tokens}/{full_tokens} PASS")


def test_online_mixed_vl_matches_same_shape_offline_batch() -> None:
    _require_cuda()
    requests = materialize_requests(_case("mixed_text_image_video_online"))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
    )
    llm = _llm(
        max_model_len=1024,
        max_chunk_size=512,
        max_num_seqs=3,
        blocks=16,
    )
    try:
        online = OnlineServingSession(llm).run(
            tuple(
                OnlineRequest(
                    request_key=f"mixed-{index}",
                    arrival_offset_s=0.0,
                    payload=request,
                    sampling_params=sampling,
                )
                for index, request in enumerate(requests)
            )
        )
        llm.reset_metrics()
        offline = llm.generate_mixed(
            requests,
            sampling,
            use_tqdm=False,
        )
    finally:
        llm.exit()

    online_tokens = [list(request.token_ids) for request in online.requests]
    offline_tokens = [output["token_ids"] for output in offline]
    assert online_tokens == offline_tokens
    assert online.scheduler_metrics["peak_active"] == 3
    print(f"online mixed VL tokens: {online_tokens}")
    print("online/offline same-shape mixed VL: PASS")
