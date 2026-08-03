# Repeated visual-context artifacts

This directory contains portable evidence for the fixed-budget, repeated-visual-context study.
Dataset media and model weights are not included.

## Experiment identity

- Model: Qwen3-VL-8B-Instruct, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Device: RTX 5090, TP1
- Performance measurement commit: `c389c3445390490293b204f65bb46020d34d8b4d`
- Quality and trace measurement commit: `0673658f625e59023e40759c994f4269d6c7387c`
- KV budget: 220 Scaled-FP8 pages, 256 tokens/page, 4,282,122,240 bytes
- Working-set plan SHA256:
  `74102cb70c6a62cdb62c1c6ed72c92a878d3a11eda4c43da101c2305fabb2739`
- Dense-page pre-run SHA256:
  `028f471748d7431f63b7cfbb859fc73680aa7a8b97b4c9ff9c4d969eff7ebbef`
- Quality selection JSON SHA256 recorded by the request-level quality runs:
  `c511ab44dca420a5b4ef65ae378104ce046f21f81703203a26a73620f9a9651e`

The performance plan contains only media groups with at least two different questions. Its
`fit`/`knee`/`pressure` streams cover 42/56/85 questions; every measured request switches questions
within its media group.

## Files

### Overview

- `highlights.json`: compact, screenshot-friendly summary of the main performance result, quality
  trade-off, decode result, KV capacity and limitations.

### Protocol

- `protocol/muirbench_working_set_plan.json`: the shared plan consumed by Prism-Infer, vLLM and
  SGLang. It records ordered media SHA256 values, media-first prompts, question IDs, arrival times,
  generation settings and the three working sets.
- `protocol/muirbench_dense_prefix_pages_8192.json`: the resumable cold pre-run that measured each
  media group's Dense Scaled-FP8 Prefix pages.
- `protocol/mvbench_repeated_selection.json`: the exact 123-video, 252-question MVBench subset.
- `protocol/mvbench_repeated_materialization.json`: archive revision, HTTP Range, CRC and media
  SHA256 records for the MVBench files.
- `protocol/quality_raw/*.json.gz`: exact measurement-time quality protocol, evaluator and
  selection JSON. These preserve the source identities stored in the request-level quality
  records; the files under `benchmarks/workloads/` are the cleaned current entry points.

### Performance

- `performance/working_set_summary.png`: resident-media and TTFT overview.
- `performance/working_set_prism_ablation.*`: Vision-only, Dense Prefix and Compact Prefix results.
- `performance/working_set_engine_comparison.*`: Prism-Infer, vLLM 0.25.1 and
  SGLang 0.5.15.post1 under the same KV-byte budget.
- `performance/working_set_summary.json`: machine-readable summary with plan/request/prompt
  identities, request latency, cache counters and process peak memory.
- `performance/performance_raw/*.json.gz`: request-level records for every engine, variant and
  working set.

### Quality

- `quality/working_set_quality.*`: compact table used by the project documentation.
- `quality/quality_summary.json`: paired Prompt-layout and actual-token-deletion comparisons.
- `quality/quality_raw/*.json.gz`: request-level quality records for all evaluated configurations.

### Trace

- `trace/prefix_hit.nsys-rep`: representative cold-request and Prefix-hit Nsight Systems capture.
- `trace/trace_evidence.json`: runtime counters and cached-token evidence.
- `trace/trace_audit.json`: machine-readable cold/hit execution-path observations.
- `trace/nsys_summary.json`: exported range and kernel summary.

Use `gzip -dk <file>.json.gz` to inspect a compressed raw record. The raw records are authoritative;
Markdown tables, CSV files, `highlights.json` and the PNG are derived views. `SHA256SUMS` covers every
portable artifact in this directory except itself.
