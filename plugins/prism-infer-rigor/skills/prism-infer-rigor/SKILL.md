---
name: prism-infer-rigor
description: "Mandatory Prism-Infer quality workflow for Codex. Use for every task in /data/Prism-Infer involving code, docs, tests, architecture, verification, benchmarks, Qwen3-VL, vision, M-RoPE, engine, KV cache analysis, or compression research. Enforces CLAUDE.md flow: recover context, plan, implement, verify, teach, document, and block unsupported claims."
---

# Prism-Infer Rigor

This skill is mandatory for Prism-Infer work. Follow it before code edits, during implementation, and before the final answer.

Write user-facing project explanations in Chinese unless the user explicitly asks otherwise.

## Priority

When rules conflict, use this priority order:

```text
Evidence and honesty > Verification > Implementation quality > Teaching and context > Language polish
```

User instructions override workflow preferences, but never override honesty: do not fabricate evidence, tests, benchmark numbers, or external implementation details.

## Session Recovery

At the start of a fresh Prism-Infer task, before editing files:

1. Read `CLAUDE.md`.
2. Read `docs/ROADMAP.md`.
3. Read `docs/VERIFICATION.md`.
4. Run `git status --short`.
5. Read task-relevant source files and tests.
6. State the current phase, touched area, and verification target in a short update.

If a handoff summary exists in the conversation, use it only as a pointer. Verify project state from files before making code claims.

If this recovery was skipped and later discovered, say so explicitly, complete the recovery, and do not claim full workflow compliance for work done before recovery.

## Required Workflow

Every non-trivial task follows:

```text
Plan -> Implement -> Verify -> Teach -> Document/KB -> Next Task
```

- **Plan**: give 3-5 concrete steps, including files or modules to inspect/edit and the verification command. Include rationale, alternatives, and reference source for design decisions.
- **Implement**: keep edits scoped. Use project patterns. Do not refactor unrelated code.
- **Verify**: run the narrowest meaningful checks first, then broader regression when risk warrants it. Report commands actually run and key output.
- **Teach**: explain module position in the architecture, key choices, limitations, and how to reason about the code.
- **Document/KB**: update project docs when milestone state, verification commands, or public behavior changes. If a configured knowledge-base path is available, write the module note there; if the Claude-era `E:\知识库\03-项目实践\` path is unavailable in Linux, state that KB write was skipped because the path is not mounted.
- **Next Task**: do not start the next project phase until the user confirms or explicitly says to continue.

For tiny read-only questions, answer directly after recovery if no code or milestone claim is involved.

## Evidence Rules

Ground every project claim in one of:

- local file path and line number,
- command output from this session,
- cited prior output clearly marked as prior evidence,
- external URL or source file line when making external implementation claims.

Do not claim "vLLM does X", "SGLang uses Y", "industry standard", or "similar to X" without inspected evidence. If evidence is missing but reasoning is still useful, write `[UNCERTAIN]` and explain the uncertainty.

Every non-trivial design decision must include:

- chosen approach and rationale,
- rejected alternatives and why,
- reference source or "no prior reference found, reasoning from first principles",
- risks and limits.

Use this hard block when an evidence rule is violated:

```text
[RIGOR BLOCK] Rule A: evidence required
Violation: <unsupported claim>
Action required: inspect source, cite evidence, or mark the claim [UNCERTAIN].
```

## Implementation Rules

Core Prism-Infer modules must be self-implemented:

- model and weight loading behavior,
- vision encoder and M-RoPE,
- attention and sampler behavior,
- engine, scheduler, sequence, block manager, context,
- KV cache layout, tracing, compression, pruning, quantization, and kernels.

Third-party code may be used only for non-core utilities, mature infrastructure with no project benefit to rewrite, or ground-truth verification. If used in a module, document the exception and source in the module docstring.

Implementation requirements:

- No silent fallback from failed accuracy or compression paths. Raise explicit errors for unsupported states.
- Preserve the FP baseline for all compression work.
- Keep hyperparameters, modes, thresholds, and model dimensions config-driven or derived from input/config.
- Public functions and methods need type hints.
- Tensor operations need shape comments when shapes are not obvious.
- No `pass`, `...`, placeholder logic, or bare `TODO`. A TODO must be dated as `TODO(YYYY-MM-DD): reason`.
- No simplified implementation pretending to be equivalent to the target behavior.
- No imports of nonexistent required modules that break package import.
- Work with dirty user changes; never revert unrelated changes without explicit permission.

Use this hard block for core implementation shortcuts:

```text
[RIGOR BLOCK] Rule B: core implementation shortcut
Violation: <shortcut, wrapper, silent fallback, or placeholder>
Action required: implement the local behavior or document a valid exception.
```

## Verification Rules

A PASS claim must be tied to a command actually run in this session or explicitly labeled as prior output.

For tensor/model correctness checks, verification output must include:

- input shape,
- output shape,
- max absolute difference versus independent reference,
- mean/std comparison when applicable,
- explicit PASS/FAIL conclusion.

Accuracy thresholds:

```text
Same precision:        max diff < 1e-5
Cross precision:       max diff < 1e-2
End-to-end greedy:     output token ids identical
Sampling/distribution: perplexity diff < 0.1 or documented logits-distribution gate
Compression on:        compression ratio, quality degradation, memory, and latency/throughput data
```

Compression-specific rule:

- `compression_mode="off"` may be verified as exact no-op baseline.
- Any active compression/pruning/quantization mode must provide measured compression ratio, quality degradation, memory effect, and performance data before it is called complete.
- If active compression is not implemented, it must fail loudly. Do not silently run uncompressed and call it compressed.

Benchmark reports must include:

- warmup count,
- measured repeat count,
- `torch.cuda.synchronize()` timing boundary,
- input shapes and key config,
- GPU memory allocated/reserved/peak,
- median, p90, min, and max.

If GPU, model files, or environment prerequisites are unavailable, state the unverified risk instead of reporting PASS.

Use this hard block for unsupported completion claims:

```text
[RIGOR BLOCK] Rule C: verification required
Violation: <completion or correctness claim without required verification>
Action required: run verification or restate as unverified risk.
```

## Prism-Infer Recovery Facts

Do not rely on these as permanent truth; confirm from docs and tests at task start.

- Repo root: `/data/Prism-Infer`.
- Model path used by current verification docs: `/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Set `HF_HUB_OFFLINE=1` for local model verification.
- Current roadmap source of truth: `docs/ROADMAP.md`.
- Current verification source of truth: `docs/VERIFICATION.md`.
- P5 compression work must preserve the P1-P4 FP and trace baselines.

Common verification commands, subject to the current `docs/VERIFICATION.md`:

```bash
.venv-local/bin/python -m compileall -q prism_infer tests
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
.venv-local/bin/python -m pytest -q tests -s
```

Use narrower focused pytest commands for local changes before the full suite.

## Final Response Checklist

Before final answer:

1. Did you read required project context for this session?
2. Did you cite local files or command output for concrete claims?
3. Did you run verification, or clearly state what was not run?
4. Did you avoid unverified benchmark/performance claims?
5. Did you explain architecture position, design choices, and residual risk for new modules?
6. Did you update docs/KB when milestone status or verification changed?
7. Did you avoid starting the next phase without user confirmation?

If any required item failed, lead with the risk or use a `[RIGOR BLOCK]` and wait for user acknowledgement when the block mechanism applies.

## Context Handoff

If a handoff or compaction boundary is needed, write:

```text
=== SESSION HANDOFF ===
阶段: <current phase>
已完成: <key outputs>
进行中: <unfinished task>
关键决策: <decisions and rationale>
下一步: <first task next session>
模型/硬件: <model path, GPU, key env>
知识库路径: <configured path or unavailable>
=== END HANDOFF ===
```
