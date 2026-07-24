# P12 Mixed-Multimodal Online Goodput

## Scope

P12 tests whether Prism-Infer's visual-token KV compaction remains useful in a
real arrival-driven mixed workload. The target is not an isolated-kernel win:
the result must include queueing, TTFT, TPOT, output-token goodput, peak memory,
correctness, and evidence that the compiler and CUDA Graph paths actually ran.

The frozen conditional-video H3 mix is:

- 40% short text;
- 30% one 448x448 image;
- 20% eight 448x448 images (H1);
- 10% sixteen 448x448 video frames (H2).

Requests use smooth weighted round robin within a period of ten and Poisson
arrival offsets. Formal runs use 600 completed requests, 64 output tokens,
rates 1/2/4 requests per second, and seeds 20260717/20260718/20260719.
Per-class SLOs are frozen from the best stable vLLM low-load run:
`TTFT <= 5 * p50` and `TPOT <= 2 * p50`. Headline goodput is the number of
output tokens per second from requests that satisfy both limits.

## Environment boundary

Results in this document are scoped to the current AutoDL RTX 5090:

- GPU UUID: `GPU-1bf42358-f8ac-c597-c1c9-30289ce22ba7`;
- driver: 580.105.08;
- model snapshot:
  `Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
- Prism branch: `codex/torch28-p9d`.

Numbers from an earlier GPU UUID are retained as history but are not pooled
with this environment.

## Problems, attempts, and resolutions

| Problem or attempt | Evidence | Resolution / lesson |
| --- | --- | --- |
| The old online harness did not expose final P10 compile/Graph modes and used only a six-request mix. | It could not prove compiler/Graph activation or represent a stable H3 distribution. | Added final P10 modes, the frozen smooth-WRR H3 schedule, prompt/trace hashes, per-class summaries, compaction accounting, and execution evidence. |
| The requested compile region was named `decode_stateless`, but the implementation contract uses `stateless`. | The first integration path rejected the configuration before measurement. | Bound the harness to the implementation's canonical region and record that effective region in every artifact. |
| KV storage accounting was treated like a record object. | `kv_cache_storage_bytes` returns a `NamedTuple`, so `.to_record()` failed. | Serialize explicit payload, scale, and total byte fields. |
| Warmup compaction records polluted measured page-reclaim totals. | Formal summaries initially included scheduler history from warmup. | Reset engine metrics after warmup and summarize only measured request IDs. |
| A 512-token prefill chunk rejected H1/H2. | H1 has about 1,582 visual placeholders and H2 about 1,633; the visual encoder payload is atomic at this boundary. | Use a 2,048-token chunk. This is a multimodal scheduling constraint, not a KV-capacity failure. |
| `max_num_seqs` limited only each batch, not resident running requests. | With a 24-page pressure pool, 12 requests became resident, causing page rotation and inter-token gaps near 1.3 seconds. | Enforce the limit on resident running plus swapped-in requests. A cap sweep showed 8 is the useful operating point; 4 over-queues. |
| A resident cap of 4 looked attractive for decode but damaged TTFT. | Versus the old scheduler, TPOT p50 improved 55.46%, but TTFT p50 regressed 238%. | Reject cap 4 as over-conservative; retain the failed candidate as evidence for the latency tradeoff. |
| Reducing the resident cap from 8 to 6 lowered TPOT but queued short text requests. | On the same 20-request rate-4 trace, TPOT p50 improved from 24.74 to 22.60 ms, but throughput fell from 233.54 to 226.76 tokens/s, TTFT p50 rose from 69.81 to 247.22 ms, and goodput fell from 58.39 to 56.69 tokens/s. | Reject cap 6. A single global concurrency cap cannot resolve class-dependent prefill/decode interference. |
| Parent-process Torch memory was zero for vLLM V1. | EngineCore owns the CUDA allocations in a child process. | Sample total NVML compute-process memory on the dedicated GPU and record all observed PIDs. |
| Prism reported Torch allocator memory while external adapters reported NVML process memory. | The values have different scopes, so a cross-framework peak-VRAM comparison would be invalid even though both are measured in MiB. | Add the same 10 ms dedicated-GPU NVML compute-process sampler to Prism; keep allocator numbers as diagnostics only. |
| Previous external numbers were offline closed-loop results. | They did not include Poisson arrivals or controller queueing and therefore could not support an H3 goodput claim. | Added in-process online adapters with identical class/arrival schedules and audited prompt-token hashes. |
| The first SGLang online smoke produced a different global prompt hash. | Image, H1, and H2 hashes were already exact, but the text prompt had 21 tokens instead of 13 because the adapter applied a chat template that vLLM does not apply to text-only H3. | Pass the frozen text prompt directly while retaining chat-template handling for multimodal inputs, then rerun the identity check. |
| The first visual-guard sweep exited before touching the GPU. | The command omitted `benchmarks/workloads/p9_headline.json`, so the default manifest did not contain the frozen H3 contract. | Keep the fail-closed error as evidence, add the explicit manifest, and rerun without treating the failed launch as a measurement. |
| A size-tiered guard initially starved H1/H2. | Single-image and heavy-visual prefills shared one decode-progress counter; every single-image prefill reset the heavy request's progress, pushing H1 TTFT to 2.1--2.6 seconds. | Prototype independent progress counters, verify that starvation disappears, then reject the tiered policy anyway because frequent single-image prefills reduce goodput. |
| Deadline promotion reduced single-image queue time but moved failures to TPOT. | With guard 32, 200--350 ms single-image deadlines caused extra visual interruptions; TPOT returned to 20.61--23.32 ms and goodput fell to 65.86--98.89 tokens/s. | Remove deadline/tier complexity from production code. Keep only the uniform guard whose end-to-end result is positive. |

## Scheduler pressure result

These diagnostic runs use the same 20-request conditional-video trace,
rate 4, seed 20260717, 64 output tokens, and a deliberately small 24-page FP8
pool. They are mechanism evidence, not the fixed-memory external headline.

| Scheduler | Peak resident | Peak pages | Duration (s) | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous batch-only limit | 12 | 24 | 7.1922 | 177.97 | 293.67 | 42.53 |
| Resident cap 4 (rejected) | 4 | 15 | 6.8128 | 187.88 | 993.59 | 18.94 |
| Resident cap 8 | 8 | 23 | 5.7652 | 222.02 | 268.59 | 29.15 |

Cap 8 improves output throughput by 24.75%, duration by 19.84%, TTFT p50 by
8.54%, and TPOT p50 by 31.45% versus the previous scheduler on the identical
trace. It also preserves enough concurrency to avoid the cap-4 TTFT collapse.

The same run recorded 16 effective page releases: H1 released three pages per
request across four requests, and H2 released two pages per request across two
requests. Compaction decisions reduced 48 dense pages to 32 physical pages for
the affected requests. Text and single-image no-op decisions did not claim
releases.

After adding common NVML sampling, a clean same-commit A/B isolated compaction
from scaled-FP8 KV quantization. Both runs used cap 8, the same 20-request
trace, and the same 24-page pool:

| Mode | Peak GPU pages | Swap/preemptions | Output tok/s | SLO goodput tok/s | TTFT p50 (ms) | H1 TTFT p50 (ms) | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scaled FP8 only | 24 | 1 | 231.720 | 46.344 | 72.089 | 323.046 | 20,818 |
| Scaled FP8 + visual compaction | 22 | 0 | 233.539 | 58.385 | 69.808 | 254.892 | 20,732 |

Compaction eliminated the swap, improved H1 TTFT by 21.1%, improved overall
TTFT by 3.16%, and increased throughput by 0.78%. The 25.98% goodput increase
is caused by one additional request meeting both SLOs in a 20-request sample,
so it is explicitly a pressure-mechanism result rather than a formal goodput
claim.

## Visual-prefill guard tuning

The loaded trace showed that a global resident cap cannot control prefill
interference. The retained candidate instead requires a configurable number
of decode batches between visual prefill batches while allowing text prefill
to bypass deferred visual work. The setting is disabled by default and emits
deferral/bypass counters.

All rows below use the identical 20-request rate-4 trace, 220 scaled-FP8 pages,
cap 8, the frozen class SLO file, and dirty tuning builds. They select a
candidate; they are not formal claims.

| Candidate | Output tok/s | SLO goodput tok/s | Good requests | TTFT p50 (ms) | TPOT p50 (ms) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No visual guard | 233.54 | 58.39 | 5/20 | 69.81 | 24.74 | Baseline |
| Uniform guard 8 | 232.99 | 58.25 | 5/20 | 76.26 | 24.73 | Reject: no effect |
| Uniform guard 16 | 232.52 | 69.76 | 6/20 | 80.65 | 23.54 | Reject: TPOT still over SLO |
| Uniform guard 32 | 224.62 | 123.54 | 11/20 | 336.00 | 19.65 | Retain for formal validation |
| Guard 32, cap 12 | 222.69 | 100.21 | 9/20 | 407.50 | 19.59 | Reject: more concurrency does not help |
| Guard 32, cap 16 | 222.54 | 100.14 | 9/20 | 411.50 | 19.53 | Reject: same batches, lower goodput |
| Tiered single/heavy guard | 176.83--230.35 | 56.77--131.89 | 5--14/20 | 83.80--257.50 | 17.46--23.34 | Reject: either heavy TTFT starvation or frequent interruption |
| Guard 32 plus deadline promotion | 219.54--221.05 | 65.86--98.89 | 6--9/20 | 267.70--448.40 | 20.61--23.32 | Reject: shifts TTFT failures to TPOT |

Uniform guard 32 reduces the number of visual prefill batches from ten to
seven by coalescing queued visual work. Its small-sample goodput is 111.6%
higher than the unguarded run, but raw throughput is 3.8% lower and
single-image TTFT moves toward its SLO. It therefore needs the full clean
600-request run before any headline claim.

## External protocol audit

The clean 20-request vLLM adapter validation has:

- trace SHA256:
  `be4e1f58736cc2ff915c1c1031045deab625dba053c0f39b4fa4a4c082d011d7`;
- prompt-token SHA256:
  `88302a418a7c80c45b362e8f9be9cb143bc1e40a683e377b44ad3172ae64f452`,
  exactly equal to Prism for all four request classes;
- effective vLLM backend: inductor compile, full-and-piecewise CUDA Graph,
  FlashAttention, asynchronous scheduling, and chunked prefill;
- fixed 4 GiB KV allocation;
- NVML process memory: 22,680 MiB after initialization and 23,670 MiB peak
  while serving.

The fixed-pool comparison separates storage capacity from total process
memory:

| System | KV format | Accounted KV bytes | Physical KV-token capacity |
| --- | --- | ---: | ---: |
| vLLM | BF16 | 4,294,967,296 | about 29,127 |
| SGLang | BF16 | 4,265,607,168 theoretical | 28,928 |
| Prism | scaled FP8 | 4,282,122,240 including scales | 56,320 |

Prism therefore has about 1.94x the physical KV-token capacity at approximately
the same pool size. This is a capacity result, not by itself a total-process
memory result; the latter uses the common NVML sampler.

The 20-request run is an adapter/protocol validation only. It is not a formal
claim because H3 requires 600 completed requests and repeated seeds.

The formal low-load vLLM run completed 600/600 requests on the clean
`ce72f63` harness. Its duration was 633.773 seconds, peak in-flight requests
were five, output throughput was 60.590 tokens/s, and peak process memory was
23,816 MiB. The frozen SLO record SHA256 is
`b3ea6f7e20d48b49053f5f7aef26e124e2f23bf6cbe05968c26b61e90c2442b2`.

| Class | Requests | vLLM TTFT p50 (ms) | TTFT SLO (ms) | vLLM TPOT p50 (ms) | TPOT SLO (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 240 | 33.991 | 169.953 | 10.521 | 21.043 |
| Single image | 180 | 98.079 | 490.394 | 10.632 | 21.265 |
| H1 eight images | 120 | 270.549 | 1,352.745 | 10.548 | 21.096 |
| H2 video | 60 | 292.300 | 1,461.498 | 10.544 | 21.087 |

After fixing the text-only prompt path, the SGLang 10-request smoke matched
both the vLLM trace hash and the full prompt-token hash
`690f5745f57405c060c2cef4ec8833ce9a31172be84273f6efc6489541640cbb`.
It used decode CUDA Graph batches 1/2/4/8; SGLang explicitly disabled
multimodal prefill Graph. The smoke reached 219.621 output tokens/s and
197.659 SLO-goodput tokens/s with 9/10 requests meeting both limits. These
numbers validate the adapter only and are not a formal SGLang result.

## Formal results

The first clean Prism rate-1 run completed 600/600 requests with no rejection.
Its trace hash, prompt-token hash, and SLO-file hash match vLLM. It reached
60.588 output tokens/s and 59.478 SLO-goodput tokens/s (589/600 requests).
Applying the same frozen SLOs to the vLLM source run gives 59.681 goodput
tokens/s (591/600), so Prism is 0.34% lower at low load.

| Class | Prism TTFT p50 (ms) | vLLM TTFT p50 (ms) | Prism TPOT p50 (ms) | vLLM TPOT p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Text | 43.575 | 33.991 | 11.069 | 10.521 |
| Single image | 67.170 | 98.079 | 11.425 | 10.632 |
| H1 eight images | 233.663 | 270.549 | 11.376 | 10.548 |
| H2 video | 274.286 | 292.300 | 11.598 | 10.544 |

Prism therefore improves single-image, H1, and H2 TTFT by 31.5%, 13.6%, and
6.2%, respectively, while text TTFT and TPOT remain behind. This is a useful
mechanism result, not yet the loaded-rate headline.

The rate-1 run also recorded 180 effective reclaim events and 480 released
pages: 360 from 120 H1 requests and 120 from 60 H2 requests. Dense visual
prompts required 1,440 pages; compaction reduced them to 960 physical pages
and dropped 112,860 visual KV tokens under the previously validated quality
policy. Decode CUDA Graph replayed 25,043 times and Vision Tensor Graph
captured three shapes with no capacity fallback.

Loaded-rate Prism and external rows remain pending completion of the clean
fixed-memory matrix. The rate-1 Prism artifact predates the common NVML
sampler, so it is not used for a cross-framework peak-VRAM headline.

The first clean fixed-memory rate-4 comparison identified the remaining
bottleneck:

| System | Output tok/s | SLO-good requests | SLO goodput tok/s | TTFT p50 (ms) | TPOT p50 (ms) | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM | 241.489 | 527/600 | 212.108 | 81.890 | 14.350 | 23,874 |
| Prism | 239.607 | 163/600 | 65.093 | 249.920 | 23.840 | 24,456 |

Prism is only 0.78% behind in raw output throughput and retains the 1.94x KV
capacity advantage, but is 69.31% behind in SLO goodput. Batch traces show
7,123 decode batches at a 14.13 ms median plus 520 prefill batches: text and
single-image prefills take about 41 and 56 ms, while H1/H2 atomic visual
prefills take roughly 180--210 ms. The FCFS policy permits a prefill after one
decode batch, so loaded visual arrivals repeatedly interrupt token delivery.
This is a scheduler-interference problem, not evidence that the decode kernels
or KV compression lost raw throughput.

## Claim boundary

- Do not compare the 24-page Prism pressure diagnostic with vLLM's 4 GiB run
  as a fixed-memory headline.
- Do not call offline closed-loop numbers online goodput.
- Do not pool results from different GPU UUIDs.
- A throughput win does not count as an H3 win unless per-class TTFT and TPOT
  use the frozen vLLM SLO file and correctness/token hashes pass.
- Dynamic page releases are a mechanism claim. A total-memory or goodput claim
  requires the corresponding measured artifact.

## Interview narrative

The concise story is: profiling showed that multimodal KV compression was not
enough by itself. Under a small page pool the scheduler admitted too many
resident requests, so reclaimed pages were immediately rotated among twelve
sequences and decode developed long gaps. I fixed resident admission and then
found a second bottleneck at fixed memory: 180--210 ms atomic visual prefills
repeatedly interrupted 14 ms decode batches. A modality-aware guard coalesces
visual work while text prefills bypass it. I rejected global cap reduction,
larger caps, size tiers, and deadline promotion using same-trace data rather
than retaining knobs that only move failures between TTFT and TPOT. The
comparison uses output-token goodput under frozen class-specific SLOs, exact
prompt/arrival hashes, and common NVML process memory. This connects page-level
compaction to arrival-driven scheduling without hiding queueing or changing
the workload.
