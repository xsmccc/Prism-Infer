---
name: prism-infer-rigor
description: Enforce Prism-Infer development rigor for Codex. Use when working in /data/Prism-Infer on code changes, architecture decisions, verification, benchmarks, external implementation comparisons, Qwen3-VL model accuracy fixes, vision encoder work, M-RoPE, engine integration, KV cache analysis, or compression research.
---

# Prism-Infer Rigor

Apply these rules before and after every Prism-Infer change. This project is Chinese-first: write project explanations, docstrings, validation summaries, and handoff notes in Chinese unless the user asks otherwise.

## Evidence

- Ground claims in files, tests, command output, or cited references. Do not claim an external project does something unless you inspected it and can cite a concrete file path plus line number, or a URL for documentation.
- If citing vLLM, SGLang, Hugging Face, or another external implementation, include the exact source location used for the comparison. If no citation is available, mark the claim with `[UNCERTAIN]` and state the reasoning.
- Do not say "industry standard", "common practice", or "similar to X" unless you provide evidence. If the claim matters to an implementation decision and evidence is missing, stop with:

```text
[RIGOR BLOCK] Rule A.1: external implementation claim needs evidence
Violation: <what was claimed without evidence>
Action required: inspect the source or mark the claim [UNCERTAIN].
```

- Every non-trivial design decision must state rationale, rejected alternatives, reference source or first-principles basis, and risk.

## Implementation

- Preserve self-implementation for core modules. Do not replace Prism-Infer core model, vision, attention, M-RoPE, KV cache, compression, scheduler, or engine behavior with a Hugging Face, vLLM, SGLang, or other third-party wrapper.
- Third-party code is allowed only for non-core utilities, mature infrastructure with no project benefit to rewrite, or ground-truth verification. Document the reason in the module docstring when this exception is used.
- Keep changes config-driven. Do not introduce magic constants for model dimensions, patch counts, layer indexes, thresholds, or shape assumptions when they can be read from config or input tensors.
- Add type hints to public functions and methods. Add shape comments near tensor transformations when the shape is not obvious.
- Do not leave `pass`, `...`, placeholder logic, or bare `TODO`. A TODO must be dated as `TODO(YYYY-MM-DD): reason`.
- Do not silently fallback from a failed compression or accuracy path to an uncompressed or simplified path. Raise an explicit error or report the unsupported state.

Use this hard block for core implementation shortcuts:

```text
[RIGOR BLOCK] Rule B.1: core module must be self-implemented
Violation: <shortcut or wrapper used>
Action required: implement the Prism-Infer module or document a valid exception.
```

## Verification

- Run the narrowest meaningful verification for the touched code. For tensor/model modules, validation must include input shape, output shape, max absolute difference versus an independent reference, mean/std comparison, and explicit PASS/FAIL.
- Do not claim a module or milestone is complete if validation was not run. Say what was not run and why.
- Accuracy targets:
  - Same precision max diff `< 1e-5`.
  - Cross precision max diff `< 1e-2`.
  - End-to-end greedy tokens must match exactly.
  - Sampling mode must compare logits distribution or perplexity, with perplexity diff `< 0.1`.
- If GPU or model files are unavailable, downgrade only by explicitly stating the unverified risk. Do not convert missing verification into a PASS.
- Benchmark reports must include warmup count, measured iterations, `torch.cuda.synchronize()` timing boundary, GPU memory stats, input shape, key parameters, median, p90, min, and max. Do not estimate benchmark numbers.

Use this hard block when a completion claim lacks verification:

```text
[RIGOR BLOCK] Rule C.1: completion claim needs verification
Violation: <claim made without required check>
Action required: run verification or restate the result as unverified risk.
```

## Prism-Infer Status

Treat these as current project facts until rerun:

- Model path: `/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Main module/vision/LLM alignment suite has passed as `20 passed`.
- `tests/test_full_model.py` full logits check is currently `MARGINAL`, with max diff about `3.125e-01` and mean diff about `2.480617e-02`.
- Do not claim strict end-to-end Qwen3-VL full-model alignment until the full logits issue is fixed or a rerun shows strict PASS.

Preferred verification commands:

```bash
.venv-local/bin/python -m compileall prism_infer tests
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_full_model_structure.py \
  tests/test_patch_embed.py \
  tests/test_vit_mlp.py \
  tests/test_mrope.py \
  tests/test_vit_attention.py \
  tests/test_vit_attention_rope.py \
  tests/test_vision_encoder.py \
  tests/test_qwen3_vl.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model.py
```

## Delivery

- After module-level work, explain in Chinese: module position in the architecture, key implementation decisions, verification results, and remaining risk.
- If a knowledge base path is available, write the module note there. The Claude-era default `E:\知识库\03-项目实践\` is not available in this Linux environment unless the user maps or overrides it; if unavailable, state that the knowledge-base write was skipped.
- Before final response, check that all PASS claims correspond to commands actually run in this session or clearly cited prior output.

## Context

- At the start of a fresh Prism-Infer session, read `CLAUDE.md`, `docs/ROADMAP.md`, relevant docs, and current `git status` before editing code.
- If context is being compacted or handed off, include: current phase, decisions made, files changed, validation status, model path, GPU availability, and next tasks.
