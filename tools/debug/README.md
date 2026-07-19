# Manual GPU diagnostics

These scripts preserve one-off layerwise investigations that are useful when
full-model parity regresses. They are deliberately outside `tests/`: each one
loads a large local checkpoint, prints exploratory diagnostics, and is not a
stable automated pass/fail contract.

Set `PRISM_MODEL_PATH` to a local Qwen3-VL checkpoint, then run a script from
the repository root, for example:

```bash
PRISM_MODEL_PATH=/models/Qwen3-VL-8B-Instruct \
  python tools/debug/full_model_layerwise.py
```

Automated regressions distilled from these investigations remain under
`tests/`. A diagnostic should move back into that directory only after it has
deterministic inputs, explicit assertions, bounded resources, and a declared
test tier.
