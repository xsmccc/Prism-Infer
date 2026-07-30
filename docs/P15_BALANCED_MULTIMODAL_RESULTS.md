# P15 Balanced Multimodal Serving Closure

## 1. Result

P15 removes P14's loaded-serving tradeoff on the frozen 60-request
`conditional_video` trace. The retained path combines:

- deadline-aware prefill coalescing while decode work is active;
- semantic ViT-block and language-layer cooperative prefill for underfilled
  or deadline-expired batches;
- atomic prefill for batches that have accumulated at least three requests;
- an explicit eight-thread CPU intra-op budget for online media preprocessing;
- the existing exact-batch B1--B8 CUDA Graph decode path, guarded FP8
  LM-head candidates with FP32 reranking, scaled-FP8 KV, and physical visual
  compaction.

Four independent loaded runs all satisfy the P15 targets. Their medians are:

- raw throughput: **215.628 output tok/s**;
- TTFT p50: **776.863 ms**;
- TPOT p50: **12.490 ms**;
- class-SLO Goodput: **75.566 output tok/s**.

The worst of the four runs still reaches 214.041 tok/s, 784.716 ms TTFT,
12.737 ms TPOT, and 75.391 tok/s Goodput. P15 therefore does not depend on
one favorable sample.

Against the declared cross-runtime references, the P15 median TPOT is 8.56%
below vLLM's 13.659 ms and 13.86% below SGLang's 14.500 ms. Raw throughput
remains 3.07% below vLLM and 2.51% below SGLang; TTFT and class-SLO Goodput
also remain behind both external systems. This is a bounded loaded-TPOT win,
not a universal serving-ranking claim.

## 2. Frozen Protocol

- GPU: NVIDIA GeForce RTX 5090
- GPU UUID: `GPU-a0340044-fe48-ceca-08e0-a50d9bcdd79a`
- driver: 580.105.08
- PyTorch/CUDA: 2.11.0+cu130 / CUDA 13.0
- model: Qwen3-VL-8B-Instruct snapshot
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- source base: `ea330bbf35904b086795fa4fb2ffcb8edc8c8d46`
- P14 retained commit: `7ea7f804a7d20548c615442171980b94b2fbabb8`
- workload: 60 requests, Poisson 4 requests/s, seed `20260717`
- trace SHA256:
  `b7948e4deb75e174ca76b2fc3ca1cae4aeb8a4676e163f4cd0f8165a5f0e954b`
- warmup: 10 requests
- output: 64 greedy tokens per request
- maximum active sequences: 8
- KV: 220 blocks, 256 tokens per block, prefix caching disabled
- visual policy: keep 0.6, image floor 768, video floor 256
- CPU intra-op budget: 8 threads

The external vLLM/SGLang records use the same model and class schedule, but
their serving frontends are not byte-for-byte identical to Prism's in-process
harness. They remain bounded references rather than a universal ranking.

## 3. Four-Run Evidence

| Run | Raw tok/s | TTFT p50 (ms) | TPOT p50 (ms) | Goodput tok/s | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1 | 215.951 | 774.878 | 12.453 | 75.583 | 23,990 |
| r2 | 214.041 | 767.172 | 12.737 | 85.616 | 23,992 |
| r3 | 215.854 | 778.847 | 12.412 | 75.549 | 23,990 |
| r4 | 215.402 | 784.716 | 12.527 | 75.391 | 23,990 |
| **median** | **215.628** | **776.863** | **12.490** | **75.566** | **23,990** |

Artifacts:

- `data/p15_balanced/final_cpu8_q1_formal_n60_dirty_r1.json`
- `data/p15_balanced/final_cpu8_q1_formal_n60_dirty_r2.json`
- `data/p15_balanced/final_cpu8_q1_formal_n60_dirty_r3.json`
- `data/p15_balanced/final_cpu8_q1_formal_n60_dirty_r4.json`

All four runs completed 60/60 requests with zero terminal failures. Each run
observed nine atomic coalesced prefills and sixteen cooperative underfilled
prefills; the number of decode steps used to accumulate or protect the
backlog ranged from 162 to 172.

## 4. Improvement Over P14

P14 deliberately protected every loaded decode cohort with one-layer/one-block
prefill quanta. It achieved a four-run TPOT median of 13.929 ms, but a pending
one- or two-request prefill could occupy the only cooperative handle for about
one second. Later arrivals could not join it, so queueing and total work grew.

Using four-run medians:

| Metric | P14 | P15 | Change |
| --- | ---: | ---: | ---: |
| Raw output tok/s | 184.804 | 215.628 | **+16.68%** |
| TTFT p50 | 2,492.326 ms | 776.863 ms | **-68.83%** |
| TPOT p50 | 13.929 ms | 12.490 ms | **-10.33%** |
| Class-SLO Goodput | 9.241 tok/s | 75.566 tok/s | **8.18x** |

P15 keeps P14's semantic preemption boundaries but no longer fragments every
prefill. When decode is active:

1. one or two newly waiting requests may wait up to 250 ms while decode
   continues;
2. if at least three requests are available, the resulting prefill plan runs
   atomically to recover batching efficiency;
3. if the oldest request reaches the 250 ms deadline while the batch remains
   underfilled, it enters the existing block/layer cooperative path;
4. the policy returns to ordinary scheduling when no decode work exists.

This is a two-sided policy: the wait deadline protects TTFT, while semantic
cooperative execution protects decode when a sufficiently efficient atomic
batch cannot be formed.

## 5. CPU Submission Root Cause

The first n60 coalescing run recovered throughput to 203.180 tok/s, but still
reported 1,364.878 ms TTFT, 15.756 ms TPOT, and only 13.545 tok/s Goodput.
Batch records showed a stable 11--14 ms CUDA Graph path interrupted by
50--155 ms wall-time spikes during media arrivals.

The server exposed 208 logical CPUs, while PyTorch defaulted to 104 intra-op
threads and 104 inter-op threads. The single asynchronous media-preprocessing
worker could therefore fan one resize/tensor operation across a large native
thread pool and delay the Python/CUDA launch thread. Adding a second
preprocessing worker made this contention worse and had already been rejected.

A single-variable diagnostic bounded only CPU intra-op parallelism to eight:

| Metric | 104-thread default | 8-thread diagnostic |
| --- | ---: | ---: |
| Raw output tok/s | 203.180 | 208.825 |
| TTFT p50 | 1,364.878 ms | 763.265 ms |
| TPOT p50 | 15.756 ms | 12.347 ms |
| Class-SLO Goodput | 13.545 tok/s | 83.530 tok/s |

The final implementation exposes
`--online-cpu-intraop-threads` (default 8), applies it before model and media
processing, validates it, and records the effective value in both benchmark
and profiler artifacts. This is host-resource isolation, not a GPU-kernel
speedup or a hidden exclusion of preprocessing: media preprocessing remains
inside TTFT.

Diagnostic artifacts:

- `data/p15_balanced/deadline_coalesced_q1_formal_n60_dirty_r1.json`
- `data/p15_balanced/cpu_threads8_deadline_coalesced_q1_n60_dirty_r1.json`

## 6. Final Semantic Profile

The final code was profiled separately from the uninstrumented headline runs.
Profiling overhead is therefore not mixed into the performance table.

| Region | Calls | CPU median | CUDA median | CUDA p90 |
| --- | ---: | ---: | ---: | ---: |
| `runner.cudagraph.replay` | 233 | 0.072 ms | 10.527 ms | 11.058 ms |
| `runner.cudagraph.copy_inputs` | 233 | 0.084 ms | 0.106 ms | 0.254 ms |
| `runner.prefill.vision_block_quantum` | 95 | 1.742 ms | 1.933 ms | 6.658 ms |
| `runner.prefill.language_layer_quantum` | 41 | 3.790 ms | 4.157 ms | 16.929 ms |
| atomic `runner.model.forward` | 2 | 167.250 ms | 180.439 ms | 235.755 ms |
| `engine.scheduler.postprocess` | 238 | 0.055 ms | n/a | n/a |

The Graph CPU submission window is only 0.072 ms while GPU replay occupies
10.527 ms. The remaining decode cost is therefore inside the captured GPU
work, not Python launch overhead. The accepted P15 gain comes from preventing
CPU preprocessing from starving launches and from choosing when to pay an
atomic prefill, rather than pretending that two non-preemptible GPU kernels
can overlap.

Artifacts:

- `data/p15_balanced/final_cpu8_q1_semantic_profile_n10_dirty_r1.json`
- `data/p15_balanced/final_cpu8_q1_profiled_n10_dirty_r1.json`

## 7. Correctness, KV, and Visual Compaction

The final isolated H1/H2 runs preserve the established 64-token greedy
trajectories:

| Case | Expected token hash | Final token hash | Result |
| --- | --- | --- | --- |
| H1, eight images | `14950e722866d99edc833472e1e7c34ae0ea6e9302b5f3d2bea894ccaa75cb0d` | same | exact |
| H2, 16-frame video | `51404847d2bad1b4c7f24a9c1db4ba0337dc3d37d9f376aa1cd6c5f83bea7727` | same | exact |

Artifacts:

- `data/p15_balanced/final_isolated_h1_cpu8_q1_o64_dirty_r2.json`
- `data/p15_balanced/final_isolated_h2_cpu8_q1_o64_dirty_r1.json`

KV storage is unchanged:

- scaled-FP8 payload: 4,152,360,960 bytes;
- FP32 scale metadata: 129,761,280 bytes;
- total: 4,282,122,240 bytes;
- equivalent BF16 element storage: 8,304,721,920 bytes;
- storage reduction: **48.44%**.

Every final n60 run also preserves the loaded visual-compaction result:

- 11,286 of 33,252 logical prompt tokens removed: **33.94%**;
- 48 of 144 physical prompt blocks reclaimed: **33.33%**;
- peak serving memory: 23,990--23,992 MiB.

H1 reclaims three of seven prompt pages and H2 reclaims two of seven pages in
the isolated correctness runs.

## 8. Rejected Attempts

| Candidate | Evidence | Decision and lesson |
| --- | --- | --- |
| Low-priority prefill stream + high-priority decode stream | valid n10: raw 138.414 vs 139.377 tok/s, TPOT 12.876 vs 12.186 ms, TTFT 1,418.859 vs 1,373.246 ms | Rejected. CUDA stream priority cannot preempt a running long GEMM; an early version also exposed shared Graph workspace before final logits were separated. |
| Two CPU preprocessing workers | raw 139.100 tok/s, TPOT 12.980 ms, TTFT 1,388.882 ms | Rejected. More workers increased host contention instead of hiding it. |
| Fixed cooperative quantum 2 | n10 looked promising, but n60 TPOT rose to 14.945 ms | Rejected by the full trace. Short traces did not expose steady-state contention. |
| Full compiled prefill MLP | slower than the exact BF16 operator path on measured shapes | Rejected. Compile coverage is not automatically a speedup. |
| Packed prefill QKV | BF16 K/V were not exact; exact recomputation removed the gain | Rejected by numerical contract. |
| Generic GEMV replacement | profiler did not identify a shape with defensible end-to-end gain over cuBLAS | Not implemented without shape-specific evidence. |
| Unbounded 104-thread CPU preprocessing | decode wall-time spikes and n60 TPOT 15.756 ms | Replaced by explicit eight-thread resource budgeting. |

Raw stream-overlap evidence is retained under
`data/p15_balanced/stream_overlap_q1_n10_dirty_r*.json`. The accidentally
concurrent OOM sample and the isolated H1 command that omitted the final mode
are retained as invalid operational evidence, not counted as model failures.

## 9. Interview Narrative

The concise technical story is:

1. **P14 solved only TPOT.** One-layer/one-block cooperative prefill protected
   decode but serialized underfilled multimodal prefills, cutting throughput
   and making TTFT/Goodput unacceptable.
2. **The first structural fix was batching, not another quantum sweep.**
   I used original request timestamps to give underfilled prefills a 250 ms
   accumulation window, ran batches of at least three atomically, and retained
   semantic cooperative fallback when the deadline expired.
3. **The first n60 run exposed a different bottleneck.** GPU replay was stable,
   but decode wall time had 50--155 ms spikes correlated with asynchronous
   media preprocessing.
4. **The root cause was host oversubscription.** A single worker could use
   PyTorch's 104-thread intra-op pool and starve CUDA submission. Bounding it
   to eight preserved preprocessing inside TTFT while restoring stable Graph
   cadence.
5. **I rejected fake overlap.** Low-priority streams did not preempt long
   GEMMs, so they added contention and correctness risk. The retained design
   schedules at semantic boundaries and budgets host resources explicitly.
6. **Result.** Four-run medians are 215.628 tok/s, 776.863 ms TTFT,
   12.490 ms TPOT, and 75.566 tok/s Goodput, while keeping exact H1/H2 hashes,
   48.44% KV storage reduction, and physical visual-page reclamation.

The important limitation to state voluntarily is that Prism wins the declared
loaded TPOT comparison, but vLLM/SGLang still lead raw throughput, TTFT, and
Goodput on their bounded reference records.

## 10. Resume-Ready Bullets

- 基于 semantic profiler 将多模态 loaded stall 拆解为 ViT block、language
  layer、CUDA Graph replay 与 CPU preprocessing，定位 104 路 CPU intra-op
  过度并行导致的 CUDA launch starvation；显式限制为 8 线程后，冻结 60-request
  trace 的 TTFT p50 从 1.365 s 降至 0.763 s、TPOT 从 15.756 ms 降至
  12.347 ms。
- 设计 deadline-aware prefill coalescing：小批量在 250 ms 内积累，达到 3 请求后
  原子执行，超时则在 ViT block/语言层边界协作执行；相对 P14 四次中位数，吞吐
  `+16.68%`、TTFT `-68.83%`、TPOT `-10.33%`、class-SLO Goodput 提升
  `8.18x`。
- 在 RTX 5090/Qwen3-VL-8B 冻结 loaded trace 上完成四次复测，TPOT 中位数
  `12.490 ms`，较 vLLM/SGLang 参考低 `8.56%/13.86%`；同时保持 H1/H2
  64-token hash exact、scaled-FP8 KV bytes `-48.44%` 与视觉物理页回收。
