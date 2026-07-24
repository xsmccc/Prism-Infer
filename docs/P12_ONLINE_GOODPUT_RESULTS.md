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
| Deadline promotion reduced single-image queue time but moved failures to TPOT. | With guard 32, 200--350 ms single-image deadlines caused extra visual interruptions; TPOT returned to 20.61--23.32 ms and goodput fell to 65.86--98.89 tokens/s. | Remove deadline/tier complexity. It moves failures between TTFT and TPOT instead of eliminating atomic prefill stalls. |
| Uniform guard 32 looked strong on 20 requests but failed at the frozen 600-request horizon. | The short run reached 123.54 goodput tokens/s, but the clean full run reached only 61.84, 5.0% below the unguarded Prism result, as queued visual work accumulated. | Reject the fixed cadence. Candidate selection and the formal horizon must remain separate gates. |
| Backlog-adaptive cadence looked strong on the bursty 100-request prefix but collapsed in steady state. | Base 64 moved goodput from 15.62 to 94.48 tokens/s on 100 requests, then reached only 6.96 on 600; base 32 reached 4.99 as peak pending visual work grew to 23. | Reject cadence control entirely. A scalar interval cannot balance an atomic 180--210 ms prefill against approximately 14 ms decode; the execution boundary must change. |
| The generic summary uses the CLI's default 500/50 ms SLO, not the frozen per-class SLOs. | The guard-32 artifact reports 313 generic-good requests, while the four class-aware counts sum to 157 and produce the actual 61.84 headline goodput. | Compute every headline from `class_aware_summary`; use `summary` only for latency and raw-throughput fields. |
| `terminal_failures` is a structured object, so its mapping length is not a failure count. | The object has three keys (`count`, `by_reason`, and `requests`) even when `count` is zero. | Read the explicit `count` field. Treat schema-aware parsing as part of the benchmark protocol. |

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

## Visual-prefill cadence experiments

The loaded trace showed that a global resident cap cannot control prefill
interference. The first candidate requires a configurable number
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
| Uniform guard 32 | 224.62 | 123.54 | 11/20 | 336.00 | 19.65 | Select, then reject in formal validation |
| Guard 32, cap 12 | 222.69 | 100.21 | 9/20 | 407.50 | 19.59 | Reject: more concurrency does not help |
| Guard 32, cap 16 | 222.54 | 100.14 | 9/20 | 411.50 | 19.53 | Reject: same batches, lower goodput |
| Tiered single/heavy guard | 176.83--230.35 | 56.77--131.89 | 5--14/20 | 83.80--257.50 | 17.46--23.34 | Reject: either heavy TTFT starvation or frequent interruption |
| Guard 32 plus deadline promotion | 219.54--221.05 | 65.86--98.89 | 6--9/20 | 267.70--448.40 | 20.61--23.32 | Reject: shifts TTFT failures to TPOT |

Uniform guard 32 reduces the number of visual prefill batches from ten to
seven by coalescing queued visual work. Its small-sample goodput is 111.6%
higher than the unguarded run, but raw throughput is 3.8% lower and
single-image TTFT moves toward its SLO. The clean 600-request validation
rejected it:

| Prism policy | Output tok/s | Class-aware good requests | SLO goodput tok/s | TTFT p50 (ms) | TPOT p50 (ms) | Queue p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unguarded | 239.607 | 163/600 | 65.093 | 249.920 | 23.840 | not recorded |
| Fixed guard 32 | 236.337 | 157/600 | 61.842 | 456.550 | 21.405 | 288.960 |

The fixed guard lowers raw throughput by 1.36% and class-aware goodput by
5.0%. It improves TPOT but allows visual requests to accumulate, so it is not
retained.

A second candidate makes cadence backlog-aware:
`effective_interval = ceil(base_interval / pending_visual_prefills)`. Text
prefills may still bypass deferred visual requests. This keeps long decode
runs when visual pressure is low but releases visual batches faster when the
queue grows. The 100-request selection runs use the same rate-4 seed and
settings; they are dirty candidate-selection evidence, not formal claims:

| Base interval | Output tok/s | Class-aware good requests | SLO goodput tok/s | TTFT p50 (ms) | TPOT p50 (ms) | Queue p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 223.142 | 7/100 | 15.620 | 571.768 | 28.808 | 408.344 |
| 32 | 238.114 | 37/100 | 88.102 | 318.537 | 21.644 | 154.511 |
| 48 | 236.519 | 34/100 | 80.416 | 381.376 | 21.766 | 208.679 |
| 64 | 236.189 | 40/100 | 94.476 | 404.065 | 20.752 | 239.710 |

Base 64 is selected because it has the best class-aware goodput and is the
only candidate whose overall TPOT p50 is below the approximately 21 ms frozen
class limits. Relative to interval 0 on this bursty prefix, it improves raw
throughput by 5.85% and goodput by 6.05x.

The full horizon rejects the adaptive policy:

| Base interval | Output tok/s | Class-aware good requests | SLO goodput tok/s | TTFT p50 (ms) | TPOT p50 (ms) | Queue p50 (ms) | Peak pending visual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 232.116 | 18/600 | 6.963 | 1,349.332 | 29.182 | 1,155.589 | 18 |
| 32 | 230.264 | 13/600 | 4.989 | 2,147.446 | 31.612 | 2,038.355 | 23 |

At base 64, backlog lowers the effective interval to four decode batches, but
visual work still accumulates. Base 32 lowers it to two and releases visual
prefills more often; this both grows the queue further and fragments decode,
so TTFT and TPOT fail together. Both runs complete 600/600 requests with zero
terminal failures, exact formal trace/prompt/SLO hashes, and the same 24,456
MiB NVML peak. They are dirty diagnostic runs because a clean rerun is
unnecessary after a decisive rejection. All cadence code and its unit test are
removed from the retained implementation.

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

The clean SGLang rate-4 run then completed the full 600-request H3 schedule
with the exact formal trace, prompt, and SLO hashes. It used SGLang
0.5.15.post1, Torch 2.11 with CUDA 13.0, BF16 KV, decode CUDA Graph sizes
1/2/4/8, and no multimodal prefill Graph. It reached 241.447 output tokens/s,
196.779 class-aware goodput tokens/s (489/600), and a 23,560 MiB NVML peak.
Its 4,265,607,168-byte theoretical KV pool holds 28,928 BF16 tokens.

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
low-load mechanism result, separate from the loaded-rate headline.

The rate-1 run also recorded 180 effective reclaim events and 480 released
pages: 360 from 120 H1 requests and 120 from 60 H2 requests. Dense visual
prompts required 1,440 pages; compaction reduced them to 960 physical pages
and dropped 112,860 visual KV tokens under the previously validated quality
policy. Decode CUDA Graph replayed 25,043 times and Vision Tensor Graph
captured three shapes with no capacity fallback.

The rate-1 Prism artifact predates the common NVML sampler, so it is not used
for a cross-framework peak-VRAM headline.

The clean fixed-memory rate-4 matrix identifies the remaining bottleneck:

| System | Output tok/s | Class-aware good requests | SLO goodput tok/s | NVML peak (MiB) | Physical KV tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | 241.489 | 527/600 | 212.108 | 23,874 | about 29,127 |
| SGLang | 241.447 | 489/600 | 196.779 | 23,560 | 28,928 |
| Prism | 239.607 | 163/600 | 65.093 | 24,456 | 56,320 |

Prism is only 0.78% behind vLLM and 0.76% behind SGLang in raw output
throughput and retains 1.93x/1.95x their physical KV-token capacity. It is,
however, 69.31%/66.92% behind in SLO goodput and uses 582/896 MiB more total
process memory. This is not a loaded-rate win.

There is one bounded multimodal latency characteristic worth retaining:
Prism's H1 and H2 TTFT p50 are 288.260 and 289.730 ms versus SGLang's 311.659
and 591.749 ms, improvements of 7.5% and 51.0%. Prism is slower on text and
single-image classes, so these numbers cannot be generalized to overall
goodput.

Batch traces show 7,123 decode batches at a 14.13 ms median plus 520 prefill
batches: text and single-image prefills take about 41 and 56 ms, while H1/H2
atomic visual prefills take roughly 180--210 ms. The FCFS policy permits a
prefill after one decode batch, so loaded visual arrivals repeatedly interrupt
token delivery. This is a scheduler-interference problem, not evidence that
the decode kernels or KV compression lost raw throughput.

## Evidence ledger

All rate-4 formal artifacts use arrival trace
`105fa73b203c42dc61b96be60d367b7b567cc78ccd4183ca9193280fffdf4235`,
prompt-token hash
`c65975c1b1a3f97dadf1bf4caa199bf7b9dcdf930f38fbfb1679f7e6b37ba79a`,
and frozen SLO hash
`b3ea6f7e20d48b49053f5f7aef26e124e2f23bf6cbe05968c26b61e90c2442b2`.

| Artifact | SHA256 | Status |
| --- | --- | --- |
| `data/p12_online/formal/p12_class_slo_vllm_r1_ce72f63.json` | `b3ea6f7e20d48b49053f5f7aef26e124e2f23bf6cbe05968c26b61e90c2442b2` | Frozen SLO source |
| `data/p12_online/formal/vllm_conditional_r4_s20260717_clean_921de81.json` | `d7548224e0a2a7659bed68c92a76525cad6dceb702dec5f80ab48abd9e4ec308` | Clean external baseline |
| `data/p12_online/formal/sglang_conditional_r4_s20260717_clean_e883de5.json` | `85f368b27b75377ed5d0d2edb236c69c68eab1497e23ecb208b32b24ecf80bd3` | Clean external baseline |
| `data/p12_online/formal/prism_visual_compact_scaled_fp8_r4_s20260717_clean_921de81.json` | `b4f9707889a14decbaa1ca4bd51c984f8e115dab8a62405d11963f8c3a742024` | Retained Prism baseline |
| `data/p12_online/formal/prism_visual_guard32_scaled_fp8_r4_s20260717_clean_e883de5.json` | `e6026aa4dcbb3482f0e8ab674fa8bb739de3d7a294e5bb63ac654c70ebe7fd35` | Clean rejected fixed cadence |
| `data/p12_online/tuning/prism_adaptive64_fp8_r4_s20260717_dirty.json` | `92f1e812d41e230930069c48f555f22efdb0e00f18ca0659b9778c038019dfc1` | Dirty rejected adaptive cadence |
| `data/p12_online/tuning/prism_adaptive32_fp8_r4_s20260717_dirty.json` | `bdbf198404de84679e8034bd3fac215b665384feb5f0ff9dd66ded3b89a8f57c` | Dirty rejected recovery experiment |

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
sequences and decode developed long gaps. I fixed resident admission and
verified that compaction eliminated a swap and improved H1 TTFT by 21.1% under
pressure.

At fixed memory, the bottleneck changed: 180--210 ms atomic visual prefills
repeatedly interrupted roughly 14 ms decode batches. I tested global
concurrency caps, fixed visual cadence, size tiers, deadlines, and a
backlog-adaptive cadence. Short traces made several policies look attractive,
but the 600-request horizon exposed queue accumulation or TPOT regressions. I
removed policies that merely moved failures between TTFT and TPOT rather than
keeping extra knobs.

The external conclusion is deliberately bounded. Prism matches raw throughput
within 0.8% while providing about 1.94x KV-token capacity and faster H1/H2 TTFT
than SGLang, but it does not beat vLLM or SGLang loaded goodput. The next
high-value architecture step is to split or overlap multimodal prefill so that
the scheduler can preempt at a finer boundary; another FCFS cadence constant
cannot remove a 180--210 ms atomic region.

Useful interview follow-ups:

- **Why did KV quantization not improve TPOT?** It increases resident-token
  capacity and avoids page pressure, but decode still reads model weights and
  is interrupted by atomic prefills. Capacity and token cadence are different
  bottlenecks.
- **Why not claim lower total VRAM?** The fixed KV pool holds about 1.94x more
  tokens, but Prism's measured process peak is higher because weights, compiler
  artifacts, and Graph pools are also resident. The honest claim is KV
  capacity, not total-process memory.
- **Why did the 20-request guard result fail?** It did not reach steady queue
  pressure. At 600 requests, deferred visual work accumulated and later
  releases damaged both TTFT and TPOT. This is why candidate selection and
  formal validation are separate.
- **What evidence proves compiler/Graph paths ran?** The artifacts record the
  Inductor region and mode, decode Graph capture sizes and replay counts, and
  three Vision Tensor Graph shapes with no capacity fallback.
- **What would be implemented next?** Chunked or phase-decomposed vision
  prefill with explicit decode-priority scheduling, followed by a stream/event
  overlap experiment only if profiling proves kernels can overlap without
  memory-bandwidth contention.
