# Results

## 1. Measurement environment

The headline results use the following frozen environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GB |
| Driver | 580.105.08 |
| CUDA | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.14.1 |
| Model | Qwen3-VL-8B-Instruct |
| Model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Model dtype / primary external comparison | BF16 / TP1 |
| External engines | vLLM 0.25.1, SGLang 0.5.15.post1 |

The TP2 extension uses two RTX 5090 GPUs in one host with the same software
stack. The GPUs have no direct CUDA P2P/NVLink path; NCCL traverses the host
`NODE` path. Results are grouped by complete protocol rather than by physical
GPU identity. Section 6 includes a separate TP2 Prism/vLLM/SGLang cell; it must
not be mixed with the TP1 H1/H2 and serving protocols above it.

## 2. Offline compiler/Graph latency

Protocol:

- batch 1, greedy decode, output 128, ignore EOS;
- warmup 2 and measured repeat 5;
- profiler disabled during timing;
- approximately 4 GiB KV budget; and
- exact prompt-token SHA256 across Prism, vLLM, and SGLang.

| Case | Prompt | Prompt tokens | Prompt SHA256 |
|---|---|---:|---|
| H1 | 8 synthetic 448×448 images | 1,618 | `04205e4593a1c294efa78f78462246266c6469d59decbe161973aeba757786b9` |
| H2 | 16 synthetic 448×448 video frames | 1,667 | `a3241f512bbb1a3e825585d118dc00383119a33302901642137cfd95c16bc5b2` |

### BF16 latency profile

| Case | System | TPOT median | TTFT median | E2E median |
|---|---|---:|---:|---:|
| H1 | **Prism** | **9.8821 ms** | **245.349 ms** | **1,598.843 ms** |
| H1 | SGLang | 10.3520 ms | 284.844 ms | 1,600.005 ms |
| H1 | vLLM | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| H2 | **Prism** | **9.8680 ms** | **240.175 ms** | **1,601.801 ms** |
| H2 | SGLang | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| H2 | vLLM | 10.5278 ms | 323.819 ms | 1,673.800 ms |

Prism TPOT is 4.54%–4.83% lower than SGLang and 6.13%–6.27% lower than
vLLM in these two cells. H1 E2E versus SGLang differs by only 0.07% and should
be treated as a tie.

### Scaled-FP8 capacity profile

| Case | Prism scaled-FP8 TPOT | SGLang BF16 | vLLM BF16 |
|---|---:|---:|---:|
| H1 | **10.2363 ms** | 10.3520 ms | 10.5276 ms |
| H2 | **10.2588 ms** | 10.3689 ms | 10.5278 ms |

The capacity profile remains 1.06%–1.12% lower in TPOT than SGLang and
2.55%–2.77% lower than vLLM. It is slower than Prism's own BF16 latency
profile, so the result supports capacity with bounded latency, not “FP8 makes
decode faster.”

## 3. KV storage, process memory, and quality

Memory sampling was separated from latency timing.

| Profile | Pages / capacity | KV bytes | NVML process peak | Torch peak allocated |
|---|---:|---:|---:|---:|
| BF16 | 113 / 28,928 tokens | 4,068.000 MiB | 23,938 MiB | 21,637.368 MiB |
| Scaled-FP8, same capacity | 113 / 28,928 tokens | 2,097.562 MiB | 21,966 MiB | 19,667.298 MiB |
| Scaled-FP8, ~4 GiB budget | 220 / 56,320 tokens | 4,083.750 MiB | 23,952 MiB | 21,653.298 MiB |

Derived results:

- allocated KV storage: **-48.4375%** at the same capacity;
- process NVML peak: **-1,972 MiB / -8.24%** at the same capacity; and
- KV token capacity: **28,928 → 56,320 / +94.69%** at a similar budget.

Scaled-FP8 passed six frozen non-inferiority cells across DocVQA, MuirBench,
and MVBench development/final splits. Unit-scale FP8 did not pass and is not
used by the final profile.

The combined modality-aware compaction policy also passed its frozen
DocVQA/MuirBench/MVBench development cells. In a capacity-constrained H1
batch-2 experiment, prompt pages fell from 7 to 4 per request; 378 of 384
decode steps could run as batch 2, increasing requests/s by 58.83%. This is an
isolated page-pressure result, not general online throughput.

## 4. Loaded multimodal serving

The H3 workload contains text, single-image, multi-image, and conditional-video
requests with Poisson arrivals at 4 requests/s. It uses greedy output 64 and
class-conditioned TTFT/TPOT SLOs.

### Fresh-object repeat matrix

Every request decodes a new media object. Repeated cells are byte-identical;
unique cells include a deterministic content marker while preserving shapes.

| Repeat | Prism raw | Prism Goodput | Good | vLLM raw | vLLM Goodput | Good |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 216.188 | 133.316 | 37/60 | **223.079** | **215.643** | 58/60 |
| 25% | 216.441 | 187.582 | 52/60 | **223.474** | **216.025** | 58/60 |
| 50% | 217.083 | 206.228 | 57/60 | **223.521** | **219.796** | 59/60 |
| 75% | 224.279 | 224.279 | 60/60 | **225.112** | **225.112** | 60/60 |
| 100% | 224.369 | 224.369 | 60/60 | **225.004** | **225.004** | 60/60 |
| 100%, different question | 224.301 | 224.301 | 60/60 | **225.004** | **225.004** | 60/60 |

At 75%–100% repeat, Prism and vLLM differ by 0.28%–0.37% raw throughput
and both satisfy every SLO. Prism's 100% result is 0.82% higher in raw
throughput and 4.30% higher in Goodput than the available SGLang cache-on
reference of 222.538/215.120 tok/s.

At 0%–50% repeat, vLLM remains ahead. Profiling attributes the gap primarily
to cold multimodal prefill disturbing decode cadence, rather than the network
front end.

### 600-request closure

The 100%-repeat long run records:

| Metric | Result |
|---|---:|
| Raw throughput | 241.428 tok/s |
| Class-SLO Goodput | 241.428 tok/s |
| SLO attainment | 600/600 |
| TTFT p50 | 146.418 ms |
| TPOT p50 | 13.041 ms |
| Processor / Vision / DeepStack / prefix hits | 360 / 360 / 360 / 360 |
| Allocated scaled-FP8 KV | 4,282,122,240 bytes |
| Released physical pages | 480 |
| Process peak | 24,006 MiB |
| Terminal failures | 0 |

Compared with the prior object-identity encoder cache on the same repeated
workload, raw throughput changes by only +0.10%, while Goodput rises from
226.311 to 241.428 tok/s (+6.68%). This comparison measures persistent
compacted-prefix reuse and tail-page pooling, not unique-media performance.

## 5. Prefix ownership evidence

In the n60 100%-repeat run:

- prefix hits/misses: 36/0;
- first tail copies / later tail leases: 2/34;
- resident compacted-prefix bytes: 233,570,304;
- copied bytes: 33,454,080;
- avoided repeated copy bytes: 529,486,848; and
- evictions, rejections, and terminal failures: zero.

An isolated H1 cold/copy/reuse sequence produced the same 64-token SHA256
`3b81c4a3e5ec1c9b9d1a67d06a6ad56ffae3320ccdbb89e0dbfc25ad14082b0d`
for all three requests.

## 6. Dual-GPU TP2

The TP2 path shards the Qwen3-VL language model rather than merely launching
two replicas:

- attention Q/K/V heads, MLP gate/up, vocabulary embedding, LM head, and
  per-rank KV heads are column sharded;
- attention output and MLP down projections are row sharded and reduced with
  NCCL;
- fixed-bucket CUDA Graph replay includes the distributed collectives; and
- exact greedy selection computes local BF16 logits/top-1 and all-gathers two
  FP32 scalars per row: the winning value and global token ID.

The frozen external cell uses one 210-token image prompt, batch 1, greedy
output 32, warmup 1 / repeat 3, BF16 weights, max model length 512, and TP2 in
all three engines.

| Engine | TPOT median | TTFT median | E2E median | Output status |
|---|---:|---:|---:|---|
| **Prism-Infer** | **5.970135 ms** | 86.057 ms | 281.098 ms | exact to vLLM |
| vLLM 0.25.1 | 6.161151 ms | 61.251 ms | 252.744 ms | reference |
| SGLang 0.5.15.post1 | 5.970127 ms | 45.452 ms | 230.601 ms | diverges at token 21 |

Prism TPOT is 3.10% lower than vLLM. Prism and SGLang are numerically tied to
the shown precision, but SGLang is a performance reference only: it shares the
prompt-token hash, then selects a different output token at position 21. A
separate SGLang TP1 run also diverged, so this is not evidence of a TP2-only
correctness failure. Prism and vLLM produced the same 32 token IDs and output
SHA256.

### TP2 optimization attribution

The original correct distributed Graph was profiled before changing the
algorithm. Reusing existing fused QK-Norm/M-RoPE/residual kernels, packed QKV,
and the tuned Paged Attention tile reduced TPOT from 8.4720 to 6.5958 ms. A
rank-local compile region then fused only QKV projection, QK-Norm, and M-RoPE;
paired 217-step A/B reduced 6.6079 to 6.5709 ms (-0.56%). The outer Graph still
owns KV state and NCCL.

Two host-control changes closed the remaining gap: every TP rank reuses its
persistent batch-one Graph buffers, and rank 0 broadcasts only token/position/
KV-slot/block-table state instead of the complete mutable sequence plan. The
paired seven-repeat control A/B reduced TPOT from 6.3212 to 5.9928 ms (-5.19%).

Nsight Systems captured 248 Graph replays. Median projected Graph GPU time was
5.9689 ms, with 5.69/5.76 ms of active kernels per replay on rank 0/1. Kernel
time was approximately 84% cuBLAS BF16 GEMV, 8% NCCL, 2.6% Paged Attention,
and under 2% fused residual normalization. The ranks were balanced; NCCL and
Graph launch were not the primary decode bottlenecks.

The mixed text+image+video batch-3 cell remained exact while TPOT fell from
11.0999 to 8.3603 ms (-24.68%) and decode throughput rose from 268.998 to
357.712 tok/s (+32.98%). This path uses the ordinary Graph bucket rather than
the batch-one compact control specialization.

The earlier sharding memory cell remains valid: single-image Torch peak
allocation was 17,082.5 MiB for TP1 and 9,126.5 MiB per TP2 rank (-46.57% per
rank); mixed batch 3 was 17,429.6 versus 9,318.1 MiB per rank (-46.54%).

The per-rank memory reduction must not be presented as a reduction in total
GPU memory: the single-image TP2 sum is 6.85% above TP1. TP2 improves the
decode-heavy region and removes the single-card weight/KV ceiling, but TTFT is
worse because the vision path is still replicated and this host lacks a
direct peer link.

Correctness and execution evidence:

- TP1 and TP2 produced the same tokens in both cells and across repeats;
- single-image output SHA256:
  `7ba5f8b0fafb6ee12454eed1de92a884c8854c3b2aa499396f7554719c9b9a8f`;
- mixed-batch output SHA256:
  `43a12cc7a7d21009b3c23cf14c1a179b586fd1c02849c8cbe4b9aafa7323b3d8`;
- compile boundary: `qkv_projection_qk_norm_mrope`, outside the validated KV
  boundary `validated_runtime_store_and_paged_decode`;
- Graph capture scope: `decode_model_forward_logits_greedy`; the final
  single-image record observed 93 batch-1 replays; and
- replacing a full-vocabulary gather with distributed local top-1 reduced the
  TP2 batch-1 decode step from 8.9240 to 8.5249 ms in the attribution cell
  (-4.47%).

The formal online TP2 run admitted and completed 6/6 burst requests covering
text, image, and video with zero rejection, cancellation, or terminal failure.
Continuous batching reached four active sequences; CUDA Graph recorded seven
batch-2 and seven batch-4 replays. Native HTTP serving separately returned
successful non-streaming text/image responses and a complete SSE token stream.
The final `configs/tp2_graph.json` compile/Graph profile also started on both
GPUs, returned HTTP 200 with eight deterministic text tokens, and shut down
cleanly. These checks establish an end-to-end TP2 serving path; the small
online runs are not used as throughput headlines.

Evidence artifact SHA256:

| Artifact | SHA256 |
|---|---|
| TP1 single image | `dd3f125869b49b99e00257f59eeda77ced7158436af1c6140bf4c8ce048aca81` |
| Final Prism TP2 single image | `f0da699a87092996f5d33e573e42bbda8e4f297711fdca0d9965bfd6bee77b58` |
| vLLM TP2 single image | `8143fa42222206657f905b593e082a1077892a49d84789020b19bce162134582` |
| SGLang TP2 single image | `2bf613039b7ca80613f5b71a1776940d0f5f78c31f94bfdcafc10ac9b19bdc25` |
| TP1 mixed batch 3 | `68814a0ced2d792e50059a29290f57a3e9a6f9abf5f5a5875f341af7d15bc7fc` |
| Final Prism TP2 mixed batch 3 | `1980261f3890f3438985b54f81988d939f7bdcfbedac6e96a5df74d86897c065` |
| Compact-control seven-repeat cell | `19fd6a625dfa4c7eedb374aa74d2fa461d981dc271a6eaf08dd1487e948066fa` |
| Rank-level Nsight report | `62ca42b91e059e044f75c658d33662142028e43e292ef91fd4d8046d03cc9e81` |
| TP2 online mixed n6 | `ea923c736d104c3f2efcc27fc144bb203525bec39eeb3ed5d8352655c403cc63` |

## 7. Interpretation

The defensible project result is:

1. Prism-Infer demonstrates a compiler/Graph decode path that wins the frozen
   H1/H2 TPOT cells.
2. It nearly doubles KV capacity with a quality-gated scaled-FP8 lifecycle.
3. Its distinctive feature is content-addressed reuse of physically compacted
   multimodal prefix pages with correct M-RoPE and page ownership semantics.
4. This design reaches parity with vLLM and exceeds the available SGLang
   reference only under high media locality.
5. vLLM remains stronger on unique and low-reuse media; universal superiority
   is not supported.
6. TP2 demonstrates real Qwen3-VL language/KV sharding, rank-local compile,
   Graph-captured NCCL, and a scoped TPOT win over vLLM on two RTX 5090 GPUs.
   It does not establish lower TTFT/E2E, lower aggregate memory, arbitrary-
   batch superiority, or multi-node scalability.
