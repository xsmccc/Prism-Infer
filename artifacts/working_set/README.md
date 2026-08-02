# Repeated visual-context study artifacts

This directory contains the portable evidence for the fixed-budget repeated-visual-context study.
Dataset media and model weights are not included.

## Experiment identity

- Model: Qwen3-VL-8B-Instruct, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Device: RTX 5090, TP1
- Measurement commit: `0673658f625e59023e40759c994f4269d6c7387c`
- KV budget: 220 Scaled-FP8 pages, 256 tokens/page, 4,282,122,240 bytes
- Working-set plan SHA256:
  `3af9e6e29656c9fe93d08c760e64a71ee4c6deb67782cbe228cc27cd56ff984a`
- Dense-page pre-run SHA256:
  `028f471748d7431f63b7cfbb859fc73680aa7a8b97b4c9ff9c4d969eff7ebbef`

## Files

### Protocol

- `protocol/muirbench_working_set_plan.json`: the shared plan consumed by Prism-Infer, vLLM and
  SGLang. It records ordered media SHA256 values, labeled media-first prompts, arrival times,
  generation settings and the `fit`/`knee`/`pressure` streams.
- `protocol/muirbench_dense_prefix_pages_8192.json`: the resumable cold pre-run that measured each
  media group's dense Scaled-FP8 Prefix pages.
- `protocol/p9_mvbench_repeated_selection.json`: the exact 123-video, 252-question MVBench subset.
- `protocol/p9_mvbench_repeated_materialization.json`: archive revision, HTTP Range, CRC and media
  SHA256 records for the MVBench files.

### Performance

- `performance/working_set_summary.png`: resident-media and TTFT overview.
- `performance/working_set_prism_ablation.*`: Vision-only, Dense Prefix and Compact Prefix results.
- `performance/working_set_engine_comparison.*`: Prism-Infer, vLLM 0.25.1 and
  SGLang 0.5.15.post1 under the same KV-byte budget.
- `performance/working_set_summary.json`: schema-v2 machine-readable summary, including process
  peak memory.
- `performance/matrix_progress.json`: identities and SHA256 values for all 15 completed cells.
- `performance/performance_raw/*.json.gz`: the 15 request-level benchmark records.

### Quality

- `quality/working_set_quality.*`: compact table used by the project documentation.
- `quality/quality_summary.json`: paired prompt-layout and actual-token-deletion comparisons.
- `quality/quality_raw/*.json.gz`: all nine completed request-level quality records.

### Trace

- `trace/prefix_hit.nsys-rep`: representative cold-request and Prefix-hit Nsight Systems capture.
- `trace/trace_evidence.json`: runtime counters and cached-token evidence.
- `trace/trace_audit.json`: six explicit cold/hit checks; `passed` is `true`.
- `trace/nsys_summary.json`: exported range and kernel summary.

Use `gzip -dk <file>.json.gz` to inspect a compressed raw record. The raw records are authoritative;
Markdown tables and the PNG are derived views. `SHA256SUMS` covers every portable artifact in this
directory except itself.
