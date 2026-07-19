# Test tiers

Prism-Infer keeps durable automated contracts under `tests/`. Test count is
not treated as a cleanup target: a test is removed only when its behavior is
duplicated or its contract is obsolete.

The registered pytest markers define resource boundaries:

- `unit`: deterministic CPU-capable logic;
- `integration`: multiple Prism-Infer components;
- `model`: a local Qwen3-VL checkpoint, with downloads forbidden;
- `gpu`: CUDA execution;
- `slow`: outside the default presubmit latency budget;
- `distributed`: multiple processes or GPUs.

Run the CPU presubmit with:

```bash
python -m pytest -q -m "not model and not gpu and not slow and not distributed"
```

Run the full local suite, allowing resource-gated tests to skip themselves:

```bash
python -m pytest -q
```

Exploratory scripts that print diagnostics without a stable assertion contract
belong in `tools/debug/`, not here. Executable performance experiments belong
in `benchmarks/`; tests for their schemas and pure helpers remain here.
