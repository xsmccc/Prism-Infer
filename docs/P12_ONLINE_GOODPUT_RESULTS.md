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
| Parent-process Torch memory was zero for vLLM V1. | EngineCore owns the CUDA allocations in a child process. | Sample total NVML compute-process memory on the dedicated GPU and record all observed PIDs. |
| Previous external numbers were offline closed-loop results. | They did not include Poisson arrivals or controller queueing and therefore could not support an H3 goodput claim. | Added in-process online adapters with identical class/arrival schedules and audited prompt-token hashes. |
| The first SGLang online smoke produced a different global prompt hash. | Image, H1, and H2 hashes were already exact, but the text prompt had 21 tokens instead of 13 because the adapter applied a chat template that vLLM does not apply to text-only H3. | Pass the frozen text prompt directly while retaining chat-template handling for multimodal inputs, then rerun the identity check. |

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

The low-load vLLM SLO source is complete. Prism and loaded-rate external rows
remain pending completion of the clean fixed-memory matrix.

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
sequences and decode developed long gaps. I fixed the resident-admission
semantics, swept the cap, rejected an over-conservative cap of four, and kept
eight because it improved both TTFT and TPOT on the identical trace. I then
rebuilt the comparison around output-token goodput under class-specific SLOs,
audited prompt-token identity across frameworks, and measured child-process
GPU memory correctly. This connects a page-level optimization to an
arrival-driven system result without hiding queueing or changing the workload.
