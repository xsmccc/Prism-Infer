# P14 Loaded Multimodal Decode Closure

## Result

P14 closes Prism-Infer's loaded decode target on Qwen3-VL-8B-Instruct and
the frozen 60-request `conditional_video` trace at 4 requests/s. The accepted
configuration combines:

- asynchronous media preprocessing that preserves the original arrival time;
- resumable vision and language prefill at exact Transformer block boundaries;
- exact fused add/RMSNorm and Q/K RMSNorm + M-RoPE through batch 8;
- row-wise FP8 LM-head candidate generation followed by original-weight FP32
  reranking;
- exact-batch CUDA Graph replay for decode batch sizes 1--8;
- scaled-FP8 KV storage and physical visual-token compaction.

All four accepted runs beat the declared SGLang TPOT reference. The four-run
TPOT median is 13.929 ms, 3.94% below SGLang's 14.500 ms. The final run from
the cleaned source is 13.776 ms, 4.99% below SGLang.

This is a bounded loaded-TPOT result, not a claim of universal engine
superiority. Prism does not beat vLLM TPOT, and it does not beat either
external runtime on TTFT, raw throughput, or class-SLO Goodput.

## Protocol and References

- GPU class: NVIDIA GeForce RTX 5090
- model: Qwen3-VL-8B-Instruct, snapshot
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- workload: frozen `conditional_video` class and Poisson-arrival trace
- measured requests: 60; warmup requests: 10
- request rate: 4/s; output length: 64
- maximum active decode batch: 8
- Prism source base: `ea330bbf35904b086795fa4fb2ffcb8edc8c8d46`
- correctness mode: greedy decoding with selective-FP32 logits

The external records use the same model and class schedule, but their
network-serving harness is not byte-for-byte identical to the current Prism
in-process harness. They are therefore declared cross-runtime references.
GPU UUID is retained for auditability, while a new baseline is not required
when continuing on the same GPU model and software environment.

| Runtime | Raw output tok/s | TPOT p50 (ms) | TTFT p50 (ms) | Class-SLO Goodput (tok/s) | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prism pre-P14 baseline | 214.623 | 27.590 | 397.173 | 21.462 | 24,018 |
| vLLM external reference | 222.462 | 13.659 | 145.463 | 211.339 | 23,826 |
| SGLang external reference | 221.181 | 14.500 | 161.562 | 191.690 | 23,666 |

## Accepted Repeated Runs

| Run | TPOT p50 (ms) | TPOT p90 (ms) | TTFT p50 (ms) | Raw tok/s | Goodput tok/s | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 14.145 | 17.515 | 2,404.738 | 184.531 | 9.227 | 23,990 |
| r2 | 13.741 | 17.465 | 2,507.674 | 185.076 | 9.254 | 23,990 |
| r3 | 14.081 | 16.845 | 2,501.216 | 184.455 | 9.223 | 23,990 |
| r4, cleaned source | 13.776 | 17.253 | 2,483.435 | 186.885 | 9.344 | 23,990 |

The pre-clean three-run median is 14.081 ms, 2.89% below SGLang. Including
the behavior-neutral source cleanup, the four-run median is 13.929 ms,
3.94% below SGLang. Relative to the Prism baseline, median TPOT falls by
49.5%. Raw throughput falls by about 13.9%, which is the principal tradeoff
of aggressive decode protection.

Artifacts:

- `data/p14_loaded_decode/quarter_latch_fp8_b8_fusions_async_q1_formal_n60_dirty_r1.json`
- `data/p14_loaded_decode/quarter_latch_fp8_b8_fusions_async_q1_formal_n60_dirty_r2.json`
- `data/p14_loaded_decode/quarter_latch_fp8_b8_fusions_async_q1_formal_n60_dirty_r3.json`
- `data/p14_loaded_decode/quarter_latch_fp8_b8_fusions_async_q1_formal_n60_clean_r4.json`

## Why the Accepted Design Works

### Cooperative multimodal prefill

Profiling showed that decode CUDA Graphs were fast in isolation, but long
vision and language prefills created 190--247 ms intervals in which decode
could not run. Splitting work at arbitrary tensor ranges changed numerical
behavior, so Prism instead pauses only at semantic block boundaries:

1. execute a bounded number of ViT blocks;
2. replay one runnable decode CUDA Graph;
3. resume the next ViT block quantum;
4. after vision completes, do the same at language-layer boundaries.

The original vision microbatch partition, all 27 ViT blocks, DeepStack
mergers after blocks 8/16/24, and the final main merger retain their original
operator order. No duplicate model weights or second long-lived CUDA stream
are required.

Quantum 1 is latched after decode reaches one quarter of the configured
capacity, batch 2 for `max_num_seqs=8`. Before that point, a quantum floor of
4 avoids over-fragmenting startup. The latch remains active until the engine
becomes idle. This one-quarter policy removed the run-to-run instability of
the earlier one-third threshold.

### Async media preprocessing

Image/video CPU preprocessing previously blocked request admission. A single
worker now preprocesses the next media request while the main thread advances
GPU serving. The request ID and original arrival timestamp are allocated
before submission, so TTFT continues to include preprocessing and the trace
semantics are not weakened.

### Exact B5--B8 fused decode path

Profiler data exposed a batch-5/6 latency cliff: exact add/RMSNorm,
Q/K RMSNorm + M-RoPE, and selective top-k fast paths stopped at batch 4.
Their supported range was expanded to batch 8 and checked elementwise against
the original operations. This removed the avoidable fallback cost without
changing residuals, normalized Q/K values, or selected candidates.

The accepted run observed CUDA Graph replay at batches 1--6; the batch-8
support remains valid for higher-concurrency traces without changing the
current result.

### FP8 candidates with FP32 exact reranking

The LM head is bandwidth-heavy. Prism uses row-wise activation scales to
generate 64 likely token IDs with FP8 matrix multiplication, then gathers
the original BF16 weights and recomputes only those logits in FP32. FP8 never
decides the returned token; it only narrows the candidate set.

The compiled function is dynamic across batches 1--8. A component probe
recovered every exact winner:

| Batch | Exact winners present in FP8 top-64 | FP32 rerank winner |
| ---: | ---: | ---: |
| 2 | 64/64 | exact |
| 4 | 128/128 | exact |
| 8 | 256/256 | exact |

Artifact:
`data/p14_loaded_decode/batched_fp8_lm_head_recall_b2_b4_b8_dirty.log`.

## Correctness and KV Evidence

The final operator set preserves the established isolated H1 and H2 greedy
trajectories for 64 output tokens:

| Case | Reference | Token hash |
| --- | --- | --- |
| H1, eight images | `data/p13_phase/tuning/h1_off_chunk2048_dirty.json` | `14950e722866d99edc833472e1e7c34ae0ea6e9302b5f3d2bea894ccaa75cb0d` |
| H2, 16-frame video | `data/p13_phase/tuning/h2_off_chunk2048_dirty.json` | `51404847d2bad1b4c7f24a9c1db4ba0337dc3d37d9f376aa1cd6c5f83bea7727` |

Current artifacts:

- `data/p14_loaded_decode/isolated_h1_quarter_fp8_b8_candidate_o64_dirty.json`
- `data/p14_loaded_decode/isolated_h2_quarter_fp8_b8_candidate_o64_dirty.json`

Loaded schedules can form different BF16 GEMM batch shapes, so low-margin
greedy trajectories are not required to match a differently batched baseline
request by request. The correctness contract is same-shape determinism,
isolated H1/H2 equality, exact fused-operator outputs, and exact final
reranking of the FP8 candidate set.

KV evidence from the final run:

- scaled-FP8 payload: 4,152,360,960 bytes;
- scale metadata: 129,761,280 bytes;
- total KV pool: 4,282,122,240 bytes;
- equivalent BF16 element storage: 8,304,721,920 bytes;
- storage reduction versus BF16: 48.44%;
- physical visual compaction: 11,286 of 33,252 logical prompt tokens removed,
  or 33.94%;
- physical blocks reclaimed: 48 of 144, or 33.33%;
- peak serving memory: 23,990 MiB.

## Rejected Candidates and What They Taught Us

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Phase-decomposed/mixed prefill | changed H1/H2 trajectories; q512 TPOT 25.624 ms | removed from source |
| Separate cooperative CUDA stream | only 0.7% TPOT gain and +180 MiB | rejected; no duplicate stream state |
| Packed QKV B2--B8 | saved 0.3--0.9 ms/step but BF16 K/V differed by up to 0.03125 | rejected; exact K/V recompute erased gain |
| KV pre-touch | TPOT 16.36 vs 16.44 ms and stalls remained | rejected |
| SwiGLU block-64 retune | B5 24.49 vs 24.00 us; B8 24.20 vs 17.18 us | retained existing block-1024 path |
| One-third cooperative latch | runs at 14.102, 15.733, and 14.796 ms | replaced by stable one-quarter latch |
| FP8 scalar activation scale | invalid `_scaled_mm` scale shape at B8 | changed to per-row scales |
| Static compiled FP8 batches | Dynamo recompile-limit failure across B1--B8 | changed to one dynamic compiled function |
| Generic GEMV replacement | remaining projections are memory-bound; no evidence of a better kernel | not pursued without shape-specific proof |

Rejected runtime code was removed rather than left behind as dormant
configuration. Logs and result artifacts remain on the server data disk.

## Interview Narrative

The concise story is:

1. **Problem:** isolated decode was already competitive, but multimodal
   prefills monopolized the GPU and caused loaded TPOT stalls.
2. **Evidence:** semantic profiling separated ViT blocks, language layers,
   CUDA Graph replay, and B1--B8 kernel shapes; it exposed atomic vision
   intervals and a B5/6 exact-fusion fallback cliff.
3. **Design:** make prefill resumable only at numerically safe Transformer
   boundaries, overlap CPU media preprocessing, extend exact kernels through
   batch 8, and reduce LM-head bandwidth with guarded FP8 candidates plus
   FP32 reranking.
4. **Tradeoff:** decode receives stronger service priority, so TPOT improves
   sharply while TTFT, raw throughput, and SLO Goodput regress.
5. **Result:** four-run median loaded TPOT is 13.929 ms, 49.5% below the Prism
   baseline and 3.94% below the declared SGLang reference, while preserving
   isolated multimodal token hashes and the 48.44% KV storage reduction.

The defensible project claim is therefore: Prism-Infer closes and slightly
surpasses SGLang on this frozen loaded multimodal TPOT trace through
block-boundary scheduling, CUDA Graph/`torch.compile` shape closure, exact
kernel fusion, and guarded mixed precision. It is not yet an across-the-board
vLLM/SGLang serving win.
