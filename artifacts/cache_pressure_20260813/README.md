# Cache-pressure comparison after the KV compaction fix

This directory contains historical request-level evidence from 2026-08-13, based on `c4c8550`
with the int64 KV-copy fix and then-uncommitted benchmark changes. It is not a benchmark of the
2026-09-07 APC repairs. The earlier compact-prefix performance and compressed-quality artifacts under
`artifacts/working_set` predate the FP8 KV compaction address fix and must not be used for current
latency or quality claims.

The experiment uses one RTX 5090, Qwen3-VL-8B-Instruct revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, page size 256, and the same 600-request MuirBench
`pressure` stream for Prism-Infer, vLLM 0.25.1, and SGLang 0.5.15.post1. Each engine is run with
220 pages (4,282,122,240 bytes) and 352 pages (6,851,395,584 bytes). All engines finish model
initialization, graph capture, and the 42-request population phase before measurement; process
startup is not part of any latency below.

The primary timing starts when the controller takes a due request, immediately before that
framework's prompt/media preparation. The main cross-engine result uses three independent paired
runs: two in 352-to-220 order and one in reverse order. Each number below is the median of the
three run-level changes; the range keeps the observed run variation visible:

| Engine | start-to-first p50 median | three-run range | start-to-first mean median | three-run range |
|---|---:|---:|---:|---:|
| Prism-Infer | +1.63% | +0.78% to +1.67% | -0.02% | -1.04% to +0.50% |
| vLLM 0.25.1 | +14.30% | +6.37% to +14.93% | +35.34% | +31.96% to +39.11% |
| SGLang 0.5.15.post1 | +36.98% | +23.97% to +41.96% | +37.68% | +28.21% to +38.32% |

The uncached logical prompt-token proxy increases by 9.79% for Prism, 299.67% for vLLM, and
347.76% for SGLang when moving from 352 to 220 pages. This is consistent with reduced cache
coverage under pressure, but it is derived from each engine's own `cached_tokens` signal and is
not a cross-engine measure of physical Prefill or Vision work.

The planned-arrival fields are retained as a secondary open-loop view, but they additionally
include controller lateness. The controller-start comparison removes that source of distortion
and remains a framework-native serving comparison, not a claim that internal API and preprocessing
implementations are identical.

As a direction check, the three-run planned-arrival p50 change medians are -1.77% for Prism,
+16.77% for vLLM, and +26.90% for SGLang. Planned arrival can include harness/controller delay,
while controller start can exclude wait before request-specific processing begins; the two scopes
answer different questions, and both retain the same capacity-sensitivity conclusion here.

At 220 pages, the Prism Dense/Compact comparison uses the same runtime, request stream, and GPU.
Compact Prefix increases resident media groups from 27 to 40 and reduces Prefix evictions from 96
to 15. Under the same controller-start origin used above, TTFT p50/p99 falls from
101.356/721.983 ms to 90.264/523.621 ms (-10.94%/-27.47%); E2E p50/p99 falls from
379.096/1,062.074 ms to 324.390/845.583 ms (-14.43%/-20.38%).

The fixed MuirBench quality rerun contains 85 media-first questions. Dense answers 46/85 and
Uniform Compact answers 47/85; on the 49 samples that actually remove visual tokens, the results
are 27/49 and 28/49. Uniform and Dense differ on only 3/85 predictions. This supports “no observed
accuracy loss on this sample set,” not an accuracy-improvement or equivalence claim. The old
Attention Top-k and MVBench compact results also use the faulty copy kernel and have not been rerun.

The defect was an int32 overflow in Triton pointer offsets after flattening K/V and 36 layers into
72 cache rows. With the old kernel, changing only the pool size from 220 to 352 pages changed the
full output of 24/42 cold population requests and the first token of 367/600 measured requests.
Promoting the Triton row/token indices to int64 makes all 42 cold trajectories and all 600 measured
first tokens agree across the two pool sizes.

`summary.json` is the compact machine-readable view. The `.json.gz` files retain all request-level
records.

File scopes:

- `*_controller_start.json.gz`: first six-cell 220/352 comparison and both timing origins;
- `*_controller_start_repeat.json.gz`: second six-cell run in 352-to-220 order;
- `*_controller_start_reverse.json.gz`: third six-cell run in 220-to-352 order;
- `prism_{dense,compact}_prefix_pressure_220_controller_start.json.gz`: controller-start Prism
  internal Dense/Compact pair;
- `prism_*_int64fix*.json.gz`: first fixed-kernel Prism pair and the Dense 220-page internal cell;
- `muir_{dense_media_first,uniform_reuse}.json.gz`: fixed-kernel MuirBench quality rerun;
- `prism_compact_pressure_{220,352}.json.gz`: pre-fix diagnostic pair retained only to show the
  address-overflow failure; it must not be used for latency or quality claims;
- `vllm_pressure_{220,352}.json.gz` and `sglang_pressure_{220,352}.json.gz`: earlier
  planned-arrival-only external cells, superseded for the primary comparison by the
  controller-start files.
