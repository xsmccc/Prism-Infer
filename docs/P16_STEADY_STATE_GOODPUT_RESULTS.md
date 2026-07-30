# P16 Steady-State Multimodal SLO Goodput

## 1. Executive Result

P16 closes the frozen 600-request `conditional_video` Goodput gap for a
bounded but practical serving pattern: repeated media objects in a warm,
single-process Qwen3-VL service.

The retained design combines:

- request-class TTFT deadlines with estimated prefill-cost reserves;
- deadline ordering and text/visual prefill-domain isolation;
- semantic prefill interruption when a lower-cost request reaches its latest
  safe start time;
- the P15 cooperative-prefill, CPU-resource, compile, CUDA Graph, scaled-FP8
  KV, and physical visual-compaction paths;
- a 128-entry exact CPU processor-output LRU; and
- an opt-in 256 MiB exact GPU Vision Encoder output LRU covering both the main
  visual embeddings and all DeepStack embeddings.

On the frozen seed-`20260717` 600-request trace:

| System / Prism mode | Raw output tok/s | Class-SLO Goodput tok/s | Good requests | TTFT p50 | TPOT p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P15 FCFS baseline | 239.387 | 67.427 | 169/600 | 892.717 ms | **12.764 ms** |
| P16 SLO scheduler, encoder cache off | 241.037 | 171.538 | 427/600 | 305.479 ms | 16.270 ms |
| **P16 SLO scheduler + exact encoder cache** | **241.184** | **226.311** | **563/600** | 275.225 ms | 14.329 ms |
| vLLM 0.25.1 reference | 241.489 | 212.108 | 527/600 | 180.244 ms | 14.283 ms |
| SGLang reference | 241.447 | 196.779 | 489/600 | 258.192 ms | 14.392 ms |

The final P16 feature-enabled system result is:

- `+31.93%` Goodput over the cache-off P16 scheduler result;
- `3.356x` the P15 n600 Goodput;
- `+6.70%` / `+14.203 tok/s` over the vLLM reference;
- `+15.01%` / `+29.532 tok/s` over the SGLang reference;
- raw throughput within `0.13%` of both external references; and
- 600/600 completed requests with zero cancellation, rejection, or terminal
  failure.

TPOT needs a precise statement. The P16 n60 selection result was `13.548 ms`,
below the bounded vLLM/SGLang n60 references of `13.659/14.500 ms`. On the
long n600 run, P16 is `0.44%` below SGLang but `0.32%` above vLLM, which is a
near tie rather than a vLLM TPOT win. The n600 headline is class-SLO Goodput,
not universal TPOT leadership.

## 2. Scope and Fairness Boundary

This result is valid only with all of the following stated:

- one RTX 5090, TP1, Qwen3-VL-8B-Instruct snapshot
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
- 600 requests, Poisson 4 requests/s, seed `20260717`, warmup 10, greedy
  output 64;
- the frozen weighted class mix: 40% short text, 30% single image, 20% H1
  eight-image, and 10% H2 video;
- three repeated visual media identities across the 360 visual requests;
- a warm in-process identity cache whose three entries were primed by the
  declared warmup;
- native frontends for Prism, vLLM, and SGLang rather than a byte-identical
  HTTP frontend; and
- the frozen per-class SLO source
  `b3ea6f7e20d48b49053f5f7aef26e124e2f23bf6cbe05968c26b61e90c2442b2`.

The cache key is the request modality plus the Python object identities of
the media. Strong references are retained in each entry, so object-ID reuse
cannot create a false hit. A separately decoded but byte-identical image is
currently a miss. P16 therefore does **not** claim content-addressed network
cache reuse, unique-media acceleration, or universal online superiority.

The external references did not receive an equivalent Prism-specific
Vision-output cache. This is a feature-enabled system comparison on the same
frozen workload, not an algorithm-isolated cache A/B across all three
engines. The cache-off P16 row is included to expose the scheduler's
standalone contribution.

Repeated-media reuse is nevertheless a real serving pattern, including
multi-turn questions over one uploaded image, repeated catalog/product
images, surveillance/video follow-up, and agent sessions that ask multiple
questions about a fixed document. The correct claim is that Prism exploits
that locality explicitly and safely.

## 3. Frozen Protocol

- GPU: NVIDIA GeForce RTX 5090
- GPU UUID: `GPU-a0340044-fe48-ceca-08e0-a50d9bcdd79a`
- driver: 580.105.08
- compute capability: 12.0
- PyTorch/CUDA: 2.11.0+cu130 / CUDA 13.0
- source base recorded by the dirty runtime artifact:
  `ea330bbf35904b086795fa4fb2ffcb8edc8c8d46`
- final retained source: the local commit containing this document
- trace SHA256:
  `105fa73b203c42dc61b96be60d367b7b567cc78ccd4183ca9193280fffdf4235`
- prompt-token SHA256:
  `c65975c1b1a3f97dadf1bf4caa199bf7b9dcdf930f38fbfb1679f7e6b37ba79a`
- mode: `visual_compact_scaled_fp8_compile_graph`
- maximum active sequences: 16
- maximum batched tokens / chunk size: 4096 / 2048
- KV: 220 blocks, 256 tokens/block, prefix cache disabled
- visual compaction: uniform keep 0.6, image floor 768, video floor 256
- online CPU intra-op threads: 8
- dynamic Vision Tensor Graph: disabled
- exact visual embedding cache: enabled, 256 MiB

The P16 formal runtime record was generated from the complete retained dirty
tree before it was packaged into the final local commit. The artifact records
`git_dirty=true`; the final source commit is the reproducible code snapshot,
while the artifact SHA256 below protects the measured record from silent
editing.

## 4. Why Scheduling Alone Was Not Enough

P15's n600 FCFS queue had strong token cadence but poor deadline attainment:
short text and single-image requests inherited the waiting time of large H1/H2
prefills. P16 first assigns each request a latest safe prefill start:

```text
latest_start = submitted_time + class_TTFT_SLO - prefill_reserve(cost_tier)
```

The retained cost tiers use reserves of `120/250/700 ms`:

- tier 0: short text;
- tier 1: ordinary visual input;
- tier 2: H1/H2-scale heavy visual input.

The scheduler orders waiting prefills by latest start, isolates short text
from visual prefill batches, and still co-batches visual classes. When a
cooperative heavy prefill is already in progress, a due lower-cost prefill can
interrupt only at an existing semantic boundary; one decode step follows the
interrupt before the paused prefill resumes.

On n600, this policy produced 166 deadline reorderings and 128 cost-domain
batch deferrals. It reduced TTFT p50 from `892.717` to `305.479 ms` and raised
Goodput from `67.427` to `171.538 tok/s`. However, TPOT rose to `16.270 ms`:
deadline protection changed who ran first but did not remove repeated ViT
work. The remaining gap was work elimination, not another priority constant.

## 5. Exact Visual Work Reuse

P16 has two independent bounded caches:

1. The CPU processor-output LRU is keyed by prompt, request modality, and exact
   media identities. It avoids repeated resize/normalize/token assembly while
   keeping preprocessing inside TTFT on a miss.
2. The GPU Vision-output LRU is keyed by request modality and exact media
   identities. It stores the main Vision Encoder output and every DeepStack
   output. Prompt text is intentionally absent because the Qwen3-VL visual
   encoder output depends on the media, not the downstream text prompt.

The GPU cache has the following safety properties:

- opt-in and disabled by default;
- TP1/Qwen3-VL only, failing closed otherwise;
- strong media references to validate identity hits;
- exact tensor reuse without quantization;
- 256 MiB LRU byte bound with eviction and oversize counters;
- raw and cached visual payloads cannot be mixed in one prefill batch;
- cached row count and every DeepStack shape are checked against visual token
  placeholders;
- entries are cleared on engine exit; and
- metrics can reset after warmup without clearing warm entries.

Only the Vision Encoder is bypassed on a hit. Every request still executes:

- text token embedding and visual-token substitution;
- full language prefill;
- scaled-FP8 KV writes;
- physical visual KV compaction and page release;
- CUDA Graph decode;
- exact logits reranking and sampling; and
- generation of all 64 output tokens.

The measured n600 window recorded 360 encoder-cache hits, zero misses, zero
evictions, and zero oversize skips. The three entries occupy
`109,182,976 bytes` (`104.1 MiB`). The CPU preprocessing cache recorded three
cold misses and 357 exact hits in the measured session.

## 6. Class-Level Result

| Class | Requests | P15 good | P16 cache-off good | P16 final good | Final attainment |
| --- | ---: | ---: | ---: | ---: | ---: |
| short text | 240 | 9 | 147 | 219 | 91.25% |
| single image | 180 | 32 | 136 | 174 | 96.67% |
| H1 eight-image | 120 | 84 | 100 | 115 | 95.83% |
| H2 video | 60 | 44 | 44 | 55 | 91.67% |
| **total** | **600** | **169** | **427** | **563** | **93.83%** |

Goodput counts a request only if it satisfies both its class TTFT SLO and
class TPOT SLO. It is therefore:

```text
good output tokens / measured wall-clock duration
```

and not raw throughput, average latency, or a pass rate reported without time.

## 7. Correctness, KV, Compaction, and Memory

The cache debug run checked repeated H1/H2 requests against the established
64-token greedy outputs:

- H1 SHA256:
  `14950e722866d99edc833472e1e7c34ae0ea6e9302b5f3d2bea894ccaa75cb0d`
- H2 SHA256:
  `51404847d2bad1b4c7f24a9c1db4ba0337dc3d37d9f376aa1cd6c5f83bea7727`

Both were exact across every observed occurrence. The formal run completed
600/600 requests with no terminal failure.

P16 preserves the established scaled-FP8 KV representation:

- payload: `4,152,360,960 bytes`;
- FP32 scales: `129,761,280 bytes`;
- total: `4,282,122,240 bytes`;
- allocated-KV reduction versus BF16: `48.44%`.

Physical visual compaction also remains active:

- logical prompt tokens: 332,520;
- physical prompt tokens: 219,660;
- dropped visual tokens: 112,860;
- dense / physical blocks: 1,440 / 960;
- released blocks: 480 (`360` H1 and `120` H2 releases).

Measured process NVML peak was `24,004 MiB`, versus `24,002 MiB` for the P16
cache-off formal run. The cache's logical tensor footprint is `104.1 MiB`, but
allocator reuse hides most of it from the peak delta. P16 does not claim a
whole-process memory reduction from the encoder cache.

After every formal/profile process exited, the rented GPU reported no compute
process and zero residual process allocation.

## 8. Profiler Attribution

The separately instrumented n10 profile is diagnostic, not the performance
headline. Its measured window recorded six encoder-cache hits and zero misses.
The semantic summary contains 18 regions and contains neither
`model.vision.*` execution nor `model.vision.embedding_cache_miss`.

It still contains:

- language-model and token-embedding work;
- language-layer prefill quanta;
- scaled-FP8 KV store and paged attention;
- CUDA Graph decode replay; and
- sampling.

This directly supports the intended mechanism: repeated ViT execution is
removed, while the language/KV/decode pipeline remains intact.

## 9. Candidate History and Rejected Ideas

The n60 selection trace was used to reject mechanisms before paying for n600:

| Candidate | n60 Goodput | Important observation |
| --- | ---: | --- |
| initial SLO tiering | 84.061 | Deadline order alone was insufficient |
| visual laxity batching | 96.300 | Better batching, still large HOL blocking |
| larger laxity batch | 103.385 | Throughput did not translate to enough SLO hits |
| interrupt-heavy path | 114.664 | TPOT regressed to 15.899 ms |
| exact CPU processor cache | 171.319 | Useful, but half of heavy requests still missed SLO |
| finish-by-slack | 183.801 | Better, but heavy/light interference remained |
| light-first | 187.505 | Helped text; not enough overall |
| decode guard | 194.694 | Raw 216.327, TPOT 15.535; text only 18/24 good |
| full exact-tier isolation | 183.801 | Over-isolation destroyed useful visual batching |
| targeted text isolation | 198.346 | Best scheduler structure before work reuse |
| text/visual domains | 187.584 | n600 cache-off later reached 171.538 tok/s |
| exact Vision-output cache | **217.367** | 59/60 good, TPOT 13.548 ms; promoted to n600 |

The important decision sequence is:

1. priority tuning improved TTFT but could not remove repeated visual compute;
2. isolating every cost tier was too aggressive because visual requests still
   benefit from shared batching;
3. isolating only short text preserved visual batching;
4. profiling and the frozen workload showed repeated media identities;
5. exact bounded reuse removed only invariant ViT work and passed token/KV
   checks; and
6. only that mechanism was promoted to the 600-request formal run.

The failed candidates remain under `data/p16_goodput/`; they were not deleted
or rewritten into the final result.

## 10. Reproduction

The formal command shape is:

```bash
export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct/snapshot
export PRISM_CLASS_SLO=data/p12_online/formal/p12_class_slo_vllm_r1_ce72f63.json

python benchmarks/bench_online.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --mode visual_compact_scaled_fp8_compile_graph \
  --h3-profile conditional_video \
  --requests 600 \
  --arrival-process poisson \
  --request-rate 4 \
  --seed 20260717 \
  --warmup-requests 10 \
  --max-tokens 64 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --max-chunk-size 2048 \
  --num-kvcache-blocks 220 \
  --kvcache-block-size 256 \
  --scheduler-policy slo_aware \
  --enable-cooperative-prefill \
  --cooperative-prefill-layer-quantum 1 \
  --cooperative-prefill-vision-block-quantum 1 \
  --online-cpu-intraop-threads 8 \
  --visual-pruning-keep-ratio 0.6 \
  --visual-pruning-min-keep-tokens 768 \
  --visual-pruning-video-min-keep-tokens 256 \
  --visual-pruning-strategy uniform \
  --class-slo-file "$PRISM_CLASS_SLO" \
  --enable-visual-embedding-cache \
  --output data/p16_goodput/p16_visual_embedding_cache_n600_s20260717_r1.json
```

The final run is valid only if the emitted record reports
`workload.h3_conformance.full_frozen_h3=true`, 600 completed requests, zero
terminal failures, the declared trace/prompt/SLO hashes, and the expected GPU
UUID.

## 11. Evidence

Headline and diagnostic artifacts:

- `data/p16_goodput/p16_visual_embedding_cache_n600_s20260717_r1.json`
  - SHA256:
    `a66fa5f311c109d23dcb37b8b3f64a31323960b4a96ba95088340e4e81e7357e`
- `data/p16_goodput/p16_visual_domains_n600_s20260717_r1.json`
  - SHA256:
    `b99ea3474bc32e7a4b95839d893318216ce5dd20d3ca6b0ad194953db90c86ab`
- `data/p16_goodput/p15_baseline_n600_s20260717_r1.json`
  - SHA256:
    `9dedc62fe138497eb61374ab34ad4ffd1d7fe9d582745d51ae31ffa2513a0f0d`
- `data/p16_goodput/p16_visual_embedding_cache_profile_n10_r1.json`
  - SHA256:
    `c33f33092ae44c53da550949c20037850206cf63cd39d583dc2b16e892c9cb07`
- `data/p16_goodput/p16_visual_embedding_cache_profile_n10_r1_semantic.json`
  - SHA256:
    `4b6b364ebcd093b607dbe84525eca4bed91b083e0183dccbb8e2d71b75458eb5`
- `data/p16_goodput/p16_visual_embedding_cache_debug_n10_r2.json`
- `data/p16_goodput/p16_visual_embedding_cache_n60_r1.json`
- vLLM:
  `data/p12_online/formal/vllm_conditional_r4_s20260717_clean_921de81.json`
- SGLang:
  `data/p12_online/formal/sglang_conditional_r4_s20260717_clean_e883de5.json`

## 12. Resume and Interview Narrative

Recommended bounded resume bullet:

> 面向 600-request Poisson 多模态稳态负载，设计基于请求成本与 TTFT slack 的调度器，
> 并实现 256 MiB exact Vision Encoder/DeepStack LRU；在重复媒体 workload 上消除
> 360 次重复 ViT，class-SLO Goodput 从 171.54 提升至 226.31 tok/s（+31.9%），
> 同协议结果高于 vLLM 6.70%、SGLang 15.01%，同时保持 H1/H2 token hash exact、
> scaled-FP8 KV bytes -48.44% 与视觉物理页回收。

The interview story should emphasize four judgments:

1. **Goodput was the goal, not raw throughput.** P15 was already within 1% raw
   throughput but only 169/600 requests met both class SLOs.
2. **Scheduling fixed head-of-line blocking but exposed irreducible work.**
   Latest-start ordering and text isolation raised Goodput to 171.54 tok/s,
   but repeated ViT remained the dominant avoidable cost.
3. **The cache is narrow and exact.** It reuses only visual encoder outputs for
   the same live media objects; language prefill, KV, decode, and sampling are
   never cached. Strong references, shape checks, a byte bound, and fail-closed
   scope prevent semantic aliasing.
4. **The external claim is deliberately bounded.** P16 wins Goodput on the
   frozen repeated-media trace. It does not prove unique-media or general
   production superiority, and n600 TPOT is tied with vLLM rather than lower.

That boundary is part of the engineering result, not a weakness to hide.
