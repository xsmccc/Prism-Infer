# CLAUDE.md — Prism-Infer Project Conventions

## Project Overview

Prism-Infer is a multi-modal inference engine built from nano-vllm.  
Target model: Qwen3-VL-8B-Instruct. Focus: vision token KV Cache analysis & compression.  
Hardware: 本地 4070 Laptop 7GB (debug), 4090 24GB (main), 5090 32GB (long-seq), multi-card rentable.

## Directory Structure

```
nano-vllm/
├── prism_infer/           # Python package (main code)
│   ├── engine/            # llm_engine, model_runner, scheduler, block_manager, sequence
│   ├── layers/            # attention, linear, layernorm, rotary_embedding, sampler, activation
│   ├── models/            # qwen3.py (existing), qwen3_vl.py (new)
│   ├── vision/            # VisionEncoder, Projector, M-RoPE (all new)
│   ├── ops/               # Custom Triton/CUDA kernels (new)
│   ├── analysis/          # KV Cache analysis tools (new)
│   └── utils/             # context, loader
├── scripts/               # Exploration & trace scripts
├── docs/                  # ROADMAP.md, DAY_XX.md task plans
├── tests/                 # Unit & integration tests
├── data/                  # Experiment outputs (gitignored)
├── learnning/             # (removed — now in E:\知识库)
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Code Conventions

### Python Style
- Type hints on all function signatures: `def forward(x: torch.Tensor) -> torch.Tensor:`
- Docstrings in Chinese (this project is Chinese-first)
- `# ---- section divider ----` for logical sections within files
- Imports sorted: stdlib → third-party → prism_infer internal
- No `import *` ever

### Naming
- Classes: PascalCase (`VisionEncoder`, `BlockManager`)
- Functions/methods: snake_case (`prepare_prefill`, `allocate_kv_cache`)
- Files: snake_case (`qwen3_vl.py`, `model_runner.py`)
- Constants: UPPER_SNAKE_CASE

### Tensor Conventions
- Always document shape in comments: `# [batch, seqlen, hidden]`
- Use `torch.bfloat16` for model weights, `torch.float32` for computation when needed
- `pin_memory=True` for CPU tensors being transferred to GPU
- Explicit `.cuda(non_blocking=True)` with `torch.cuda.synchronize()` where needed

## Architecture Constraints

- **No silent fallback**: if a compression strategy fails, raise explicit error. Never silently fallback to uncompressed.
- **Preserve FP baseline**: all compression paths must have FP reference to compare against.
- **One module, one responsibility**: don't put vision logic in engine/, don't put scheduling logic in layers/.
- **Config-driven**: all hyperparameters go through `config.py` or a dedicated dataclass. No magic numbers.
- **Keep nano-vllm's design**: multi-process TP via shared memory, set_context/get_context for attention metadata, @torch.inference_mode() for forward passes.

---

## AI Collaboration Rules

### Must Do
1. **Read before write**: inspect relevant existing code before proposing any implementation.
2. **State assumptions explicitly**: "I assume X because the HF model does Y".
3. **Prefer small changes**: one logical change per commit, one module per task.
4. **Add tests for behavior changes**: if a function's output changes, add a test.
5. **Report what was run and what was NOT run**: be explicit about untested paths.
6. **Distinguish measured from expected**: "Measured: ppl 8.2" vs "Expected: throughput should improve".
7. **Keep code paths explicit**: no `try: compressed; except: fallback_to_uncompressed` without logging.

### Must NOT Do
1. **Silently skip or fallback**: all failures must be visible (assert, raise, or log.WARNING).
2. **Claim speedup without benchmark**: must include warmup, repeat count, `torch.cuda.synchronize()`, memory stats, median/p90.
3. **Claim correctness without validation**: must show comparison against reference output.
4. **Edit unrelated modules**: don't touch scheduler when working on vision encoder.
5. **Hide TODOs**: every TODO must have a date and owner. `# TODO(2024-06-14): implement M-RoPE for visual tokens`
6. **Invent benchmark data**: don't say "throughput improved 40%" without running the actual benchmark.
7. **Implement demo-only code**: it must work end-to-end, not just in a notebook.
8. **Remove or overwrite user's changes**: ask before editing files outside the current task scope.

### Before Each Implementation
- List all files that will be touched.
- Identify which existing behavior will change.
- State the verification criteria (exact error tolerance, expected output, etc.).

### After Each Implementation
- Summarize what was changed and why.
- State what was tested and the result.
- Document what was NOT tested and the risk.
- Suggest the next step.

### Verification Language
Use precise words:
- "Tested with input shape [1, 3, 448, 448], output matches HF within 1e-5" — ✅
- "Expected to work for batch_size > 1 but not tested" — ✅
- "should be fine" — ❌
- "probably works" — ❌
- "it's faster" without numbers — ❌

---

## Task Workflow

Every task follows this cycle:
```
Plan → Implement → Verify → Knowledge Base → Next Task
```

1. **Plan**: 3-5 bullet points specifying exactly what will be done.
2. **Implement**: code changes scoped to the task.
3. **Verify**: run the verification script, compare against expected output.
4. **Knowledge Base**: write one 500-word note answering "how would I explain this in an interview?"
5. **Next Task**: only proceed after verification passes.

## Benchmark Rules

Every benchmark must print:
- Warmup iterations and repeat count
- `torch.cuda.synchronize()` before timing
- GPU memory stats (allocated, reserved, peak)
- Input shapes and config parameters
- Median, p90, min, max latencies (not just mean)
