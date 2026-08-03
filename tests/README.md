# Tests

The test directory covers Prism-Infer's inference path:

- Qwen3-VL model structure, multimodal positions, and processor integration;
- paged KV storage, scaled-FP8 quantization, and physical token compaction;
- `torch.compile` and CUDA Graph execution boundaries;
- scheduling, online serving, and request lifecycle behavior.

Performance scripts and result derivation live under `benchmarks/`; tests here
focus on runtime behavior and numerical correctness.

Run the focused logic tests with:

```bash
python -m pytest -q -m "not model and not gpu and not slow and not distributed"
```

Run all tests whose local model and GPU requirements are available with:

```bash
python -m pytest -q
```

Resource-dependent tests declare `model`, `gpu`, `slow`, or `distributed`
markers and skip when their prerequisites are absent.
