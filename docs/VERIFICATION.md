# Prism-Infer 验证标准

> 修订日期: 2026-07-18
> 目的: 统一记录每个阶段的验证命令、PASS 标准和禁止行为。所有完成声明必须能追溯到本文件中的命令或等价验证输出。

## 全局规则

- 所有测试必须在 `/data/Prism-Infer` 下执行。
- 重量级模型测试使用:

```bash
export PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

- 代码语法、模块测试、full logits、端到端 generate、benchmark 是不同门槛，不能相互替代。
- 未运行的测试必须在交付说明中写明，不能默认算 PASS。
- full logits 曾在 P1-001 中出现 `MARGINAL`，max diff 约 `3.125e-01`；2026-06-21 修复后已重新跑出 strict PASS，max diff `0.000000e+00`。
- 2026-06-24 修复 P2-005 后，单图图文 full logits 和 full-model layerwise 已 strict PASS；该结论只覆盖单图、单请求、`enforce_eager=True` correctness。
- 2026-06-25 完成 P3.1 后，单请求多图 processor、position ids、full logits 和 1-token greedy 已 strict PASS。
- 2026-06-25 完成 P3.2 后，单请求 synthetic video processor、position ids、full logits 和 1-token greedy 已 strict PASS。
- 2026-06-25 完成 P3.3 后，text-only/single-image/multi-image/video non-prefix mixed batch 1-token greedy 已与 fresh 单请求独立运行一致。
- 2026-06-25 完成 P3.4 后，single-image/multi-image/video `max_tokens=8/16/32` greedy 已与 HF exact match，32-token teacher-forced logits/ppl 与 HF exact match；mixed batch 中 VL rows 32-token 与 fresh 单请求一致。text-only row 的 32-token mixed 分叉已证明是 HF/Prism 共有的 bf16 batch-size 数值敏感性。
- 2026-07-16 完成 P7.4-A 后，默认 logits改为模型原生 BF16；P3.3/P3.4 的跨
  batch-shape exact结果保留为历史证据，不再作为通用合同。当前要求同一 mixed
  shape重复 exact，并以 HF teacher-forced logits/PPL exact和独立任务质量门禁
  约束跨 shape低 margin分叉。
- 2026-07-16 完成 P7.3 后，Q<K chunked/prefix prefill已有 correctness-first paged
  gather+SDPA路径；online engine harness覆盖 wall-clock arrival、continuous batching、
  admission/cancel、queue/TTFT/TPOT/goodput。VL token-id prefix hash因不包含像素语义
  而禁用；这不是网络 server或外部框架 online对比。
- 2026-06-25 完成 P3.7 后，P3 当前门禁已通过: grouped regression `49 passed in 356.34s`，纯文本/单图/多图/视频 full logits 均 strict PASS，VL CUDA Graph decode 与 paged decode kernel 均有 correctness 和 benchmark 基线。
- 2026-07-05 使用本地 Qwen3-VL 权重和 transformers 5.13 刷新门禁: `pytest -q tests -s` 为 `84 passed, 5 skipped in 250.07s`；5 个 skipped 均为 manual GPU debug script。当前长输出门禁采用稳定前缀 + teacher-forced logits/ppl 分布口径: 单图/多图 `prefix@8/16` 与 HF 一致，视频因第 6 个 token 的 bf16 tie-break 固定为 `prefix@5`，完整 32-token 仍打印用于诊断。完整门禁无 warning。
- GPU 不可用时可以降级为“未验证风险”，但不能把缺失验证写成通过。

## PASS 标准

| 类型 | PASS 标准 |
|---|---|
| 语法检查 | `compileall` 无错误 |
| 同精度模块对齐 | max diff `< 1e-5` |
| 跨精度模块对齐 | max diff `< 1e-2` |
| Full logits | max diff `< 1e-2` 且无 NaN；更严格目标按对应测试定义执行 |
| Greedy 端到端 | `temperature=0` 输出 token 完全一致；长输出低 margin tie-break 场景必须使用测试内显式记录的稳定前缀门禁 |
| 采样模式 | logits 分布或 perplexity 对齐，ppl diff `< 0.1` |
| 压缩策略 | compression off 等价 baseline；compression on 给出压缩率、质量退化和性能/显存数据 |
| Pruning reference task preflight | baseline/candidate 使用同一可审计 reference；token-F1 与 ROUGE-L macro score 的绝对下降均 `<=0.01`。这是 lexical regression gate，不替代 CIDEr/SPICE 或完整任务 accuracy。 |
| Benchmark | 实测 warmup、repeat、median、p90、min、max、显存和输入参数 |

## P0: 治理与基线验证

验证 Codex plugin 和文档入口是否可用:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /data/Prism-Infer/plugins/prism-infer-rigor/skills/prism-infer-rigor
```

```bash
python3 /root/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py \
  /data/Prism-Infer/plugins/prism-infer-rigor
```

```bash
codex plugin list
```

PASS 标准:

- skill validator 输出 `Skill is valid!`。
- plugin validator 输出 `Plugin validation passed`。
- `codex plugin list` 显示 `prism-infer-rigor@personal` 为 `installed, enabled`。

## P1: 模型地基验证

### 1. 语法检查

```bash
.venv-local/bin/python -m compileall prism_infer tests benchmarks
```

PASS 标准:

- 无 Python 编译错误。

### 2. 模块对齐套件

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

PASS 标准:

- pytest 全部通过。
- 输出中不能出现 FAIL。
- 对齐测试必须包含 shape、max diff、PASS/FAIL 信息。

### 3. Full logits 检查

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model.py
```

PASS 标准:

- 权重加载 missing/unexpected keys 为 0，或有明确解释。
- logits shape 与 HF 一致。
- 无 NaN。
- max diff `< 1e-2`。

当前状态:

- P1-001 修复前结果为 `MARGINAL`，max diff 约 `3.125e-01`，mean diff 约 `2.480617e-02`。
- 2026-06-21 修复后结果为 `PASS`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- 该 PASS 是纯文本 full logits 门禁，不代表图文端到端 generate 已完成。

### 4. 分层误差定位

当 full logits 未达标时，必须补充 layerwise debug 输出:

- 每层 hidden max diff。
- 每层 hidden mean diff。
- 注意力、MLP、RMSNorm、RoPE 的局部 diff。
- dtype、device、kernel path 信息。

PASS 标准:

- 找到误差主要增长区间或可解释的 kernel/dtype 差异。
- 修复后回到 full logits 检查。

P1-001 已验证的定位工具:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tools/debug/full_model_layerwise.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tools/debug/attention_micro.py
```

P1-001 结论:

- 根因是 `apply_mrope` 在 RoPE 应用阶段使用 float32 中间计算，改变 bf16 舍入路径。
- 修复后 attention micro debug 中 `q_rope/k_rope/sdpa_gqa/attn_out/layer0_out` max diff 全部为 `0.000000e+00`。
- 完整问题记录见 `docs/ISSUE_LOG.md`。

## P2: Engine 端到端验证

P2 分阶段验证。任何单项 PASS 都不能替代最终 greedy tokens 对齐。

### 1. Processor pipeline

```bash
.venv-local/bin/python -m pytest -q tests/test_processor_pipeline.py
```

PASS 标准:

- processor 输出的 `input_ids`、`pixel_values`、`image_grid_thw` 与参考一致。
- 输出包含 input ids shape、pixel values shape、image grid shape、image token 数量、max diff 或 exact match、PASS/FAIL。
- HF processor 只作为非核心预处理工具或 ground truth 使用，不能替代 Prism-Infer 核心模型。

当前状态:

- 2026-06-21 已验证 PASS。
- 命令:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q tests/test_processor_pipeline.py -s
```

- 输出摘要:
  - `input_ids shape: [1, 210]`
  - `pixel_values shape: [784, 1536]`
  - `image_grid_thw shape: [1, 3]`
  - `image tokens: 196 / expected 196`
  - `pixel_values max diff: 0.000000e+00`
  - `3 passed in 6.23s`

### 2. 多模态请求和 3D position ids

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_sequence_multimodal.py \
  tests/test_vl_rope_index.py
```

PASS 标准:

- 纯文本 `Sequence` 行为不变。
- 单图 `Sequence` 序列化后保留 `pixel_values`、`image_grid_thw`、`position_ids` 或可重建 position ids 的必要元数据。
- 单图 `position_ids` shape 为 `[3, 1, seqlen]`。
- `rope_delta` shape 为 `[1, 1]`。
- 与 HF `get_rope_index` exact match，max diff `0`。

当前状态:

- 2026-06-24 已验证 PASS。
- 命令:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_vl_rope_index.py \
  tests/test_sequence_multimodal.py -s
```

- 输出摘要:
  - `input_ids shape: [1, 210]`
  - `position_ids shape: [3, 1, 210]`
  - `rope_delta shape: [1, 1]`
  - `position_ids max diff: 0.000000e+00`
  - `rope_delta max diff: 0.000000e+00`
  - `prefill position_ids shape: [3, 1, 210]`
  - `decode rope_delta shape: [1, 1]`
  - `9 passed in 9.15s`

### 3. KV-aware attention 和 VL Prefill

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py
```

PASS 标准:

- Qwen3-VL LLM attention 在 engine prefill 中写入 KV cache。
- `prepare_prefill` 传递 `input_ids`、`position_ids`、`pixel_values`、`image_grid_thw` 到模型 forward。
- 视觉 token 数量与 Vision Encoder 输出数量一致；不一致必须显式报错。
- engine flatten attention 输出与 full-sequence attention 数值一致，并输出 shape、max diff、mean diff、PASS/FAIL。
- P1 full logits 仍 PASS。

当前状态:

- 2026-06-24 已验证 PASS。
- 命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline.py \
  /data/Prism-Infer/tests/test_vl_rope_index.py \
  /data/Prism-Infer/tests/test_sequence_multimodal.py \
  /data/Prism-Infer/tests/test_model_runner_vl_prefill.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py -s
```

- 输出摘要:
  - `15 passed in 12.16s`
  - `prefill input_ids shape: [210]`
  - `prefill position_ids shape: [3, 210]`
  - `prefill pixel_values shape: [784, 1536]`
  - `attention output max diff: 0.000000e+00`
  - `attention output mean diff: 0.000000e+00`
  - `k_cache max diff: 0.000000e+00`
  - `v_cache max diff: 0.000000e+00`

### 4. Decode eager 和 greedy sampler

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_sampler_greedy.py
```

PASS 标准:

- decode 阶段不重复运行 Vision Encoder。
- decode position ids 使用 prefill rope delta 延续。
- decode attention 能从 paged KV cache 读取完整历史。
- `temperature=0` 或显式 greedy 模式走 deterministic argmax。
- 随机采样路径不回归。

当前状态:

- 2026-06-24 已完成 decode 输入准备和 paged KV eager fallback 子门禁。
- 输出摘要:
  - `decode input_ids shape: [1]`
  - `decode position_ids shape: [3, 1]`
  - `decode expected position: 28`
  - `decode actual positions: [28, 28, 28]`
  - `decode output max diff: 0.000000e+00`
  - `decode output mean diff: 0.000000e+00`
- greedy sampler 和 `LLM.generate_vl` 已在 P2.6/P2.7 验证。

### 5. 端到端 VL generate 和纯文本回归

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_llm_vl_generate.py \
  tests/test_text_only_regression.py
```

PASS 标准:

- 单图 `LLM.generate_vl` 或等价公开 API 能从用户入口跑通。
- greedy tokens 与 HF 完全一致。
- 纯文本请求不回归。

当前状态:

- 2026-06-24 已验证 PASS。
- 命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_sampler_greedy.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py \
  /data/Prism-Infer/tests/test_text_only_regression.py -s
```

- 输出摘要:
  - `greedy token_ids: [1, 2]`
  - `LLMEngine add_vl_request: PASS`
  - `HF token_ids: [785]`
  - `Prism token_ids: [785]`
  - `LLM.generate_vl one-token greedy HF alignment: PASS`
  - `text output token_ids: [785]`
  - `text-only engine greedy smoke: PASS`

### 6. 图文 full logits 和 VisionEncoder 回归

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tools/debug/full_model_vl_layerwise.py
```

```bash
.venv-local/bin/python -m pytest -q tests/test_vision_rope_init.py -s
```

PASS 标准:

- `tests/test_full_model_vl.py` 输出 input/logits shape、NaN 数量、HF/Prism mean/std、max diff、mean diff 和明确 PASS/FAIL。
- 单图图文 last logits max diff `< 1e-2`；当前 strict 结果为 `0.000000e+00`。
- layerwise debug 中 `visual/embed/rope/layer_00...layer_35/final_norm/logits` 不出现非零 diff。
- Vision RoPE 初始化回归必须证明默认 device 为 CUDA 时，`inv_freq/freq_table/rot_pos_emb` 与 HF exact match。
- PatchMerger main/deepstack LayerNorm eps 均为 `1e-6`。

当前状态:

- 2026-06-24 已验证 PASS。
- 图文 full logits 输出摘要:
  - `input_ids shape: [1, 210]`
  - `pixel_values shape: [784, 1536]`
  - `image_grid_thw shape: [1, 3]`
  - `position_ids shape: [3, 1, 210]`
  - `Shape: HF=[1, 151936], Our=[1, 151936]`
  - `NaN: HF=0, Our=0`
  - `HF mean/std:  -1.756945e+00 / 4.123917e+00`
  - `Our mean/std: -1.756945e+00 / 4.123917e+00`
  - `Max diff:  0.000000e+00`
  - `Mean diff: 0.000000e+00`
  - `Result: PASS`
- 图文 layerwise: `visual`、`embed`、`rope`、36 层 LLM、`final_norm`、`logits` max diff 和 mean diff 全部 `0.000000e+00`。
- Vision RoPE 初始化回归: `2 passed in 8.71s`，`inv_freq/freq_table/rot_pos_emb` max diff 全部 `0.000000e+00`。

### 7. P2 Gate Review

P2 完成前必须运行:

```bash
.venv-local/bin/python -m compileall prism_infer tests benchmarks
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_sequence_multimodal.py \
  tests/test_vl_rope_index.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_sampler_greedy.py \
  tests/test_llm_vl_generate.py \
  tests/test_text_only_regression.py \
  tests/test_vision_rope_init.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl.py
```

PASS 标准:

- `compileall` 无错误。
- P2 全部测试 PASS。
- P1 full logits 仍 PASS。
- 单图图文 full logits PASS。
- 单图 greedy tokens 与 HF 完全一致。
- P2 当时未支持的多图、视频、VL CUDA Graph decode 必须在 P2 交付说明中标为未验证风险；这些能力已在 P3.1/P3.2/P3.5 补齐。

当前状态:

- 2026-06-24 已验证 PASS。
- P2 Gate + vision 回归测试:

```text
24 passed in 48.49s
```

- 图文 full logits:

```text
Result: PASS
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
```

- P1 轻量回归:

```text
10 passed in 74.68s
```

- P1 full logits:

```text
Result: PASS
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
```

## P3: VL Engine 完整性与性能基线验证

P3 的验证顺序是先 correctness，再性能。任何性能报告必须引用同一输入集合下的 correctness PASS；不能在未对齐的 kernel 或 graph path 上报告吞吐收益。

### 1. P3.0 设计门禁

验证命令:

```bash
.venv-local/bin/python -m compileall prism_infer tests
```

PASS 标准:

- `docs/ROADMAP.md` 写清 P3.1-P3.6 的目标、任务和出口标准。
- `docs/VERIFICATION.md` 写清多图、视频、batch、长输出、CUDA Graph、paged decode 的测试命令和 PASS 标准。
- `docs/P3_VL_ENGINE_COMPLETENESS.md` 或等价设计文档记录:
  - `VLInputs` 数据结构。
  - image/video grid 与 token span 语义。
  - position ids/rope_delta 的 batch 语义。
  - eager、CUDA Graph、paged decode kernel 的 reference 关系。
  - benchmark 输入集合。

当前状态:

- 2026-06-25 已完成 P3.0 设计门禁。
- P3 后续仍按 correctness -> graph/kernel -> benchmark 推进；不能用 P3.1 多图 PASS 代表 P3 整体完成。

### 2. 多图输入 correctness

验证命令:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline_multi_image.py \
  tests/test_vl_rope_index_multi_image.py \
  tests/test_llm_vl_generate.py::test_add_vl_request_builds_multi_image_sequence \
  tests/test_llm_vl_generate.py::test_generate_vl_multi_image_one_token_matches_hf_greedy -s
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl_multi_image.py
```

PASS 标准:

- processor 输出 `input_ids`、`pixel_values`、`image_grid_thw=[num_images,3]` 与 HF reference 一致。
- `image_token_count == sum(image_grid_thw.prod(dim=1) // merge_size**2)`。
- `position_ids` shape 为 `[3, 1, seqlen]`，`rope_delta` shape 为 `[1, 1]`。
- `position_ids/rope_delta` 与 HF `get_rope_index` exact match，max diff `0`。
- full logits 输出包含 input/logits shape、NaN 数量、HF/Prism mean/std、max diff、mean diff 和 PASS/FAIL。
- `LLM` 多图公开入口 greedy token ids 与 HF 完全一致。
- 当前单图 P2 回归仍 PASS。

当前状态:

- 2026-06-25 已验证 PASS。
- 多图 processor/rope/LLM 轻量门禁:

```text
5 passed in 8.01s
multi input_ids shape: [1, 408]
multi pixel_values shape: [1568, 1536]
multi image_grid_thw shape: [2, 3]
multi image_grid_thw: [[1, 28, 28], [1, 28, 28]]
multi image tokens: 392 / expected 392
multi pixel_values max diff: 0.000000e+00
multi position_ids shape: [3, 1, 408]
multi rope_delta shape: [1, 1]
multi position_ids max diff: 0.000000e+00
multi rope_delta max diff: 0.000000e+00
```

- 多图 full logits:

```text
input_ids shape: [1, 408]
pixel_values shape: [1568, 1536]
image_grid_thw shape: [2, 3]
image tokens: 392 / expected 392
position_ids shape: [3, 1, 408]
Shape: HF=[1, 151936], Our=[1, 151936]
NaN: HF=0, Our=0
HF mean/std:  -1.318763e+00 / 4.206440e+00
Our mean/std: -1.318763e+00 / 4.206440e+00
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
PASS (max diff < 0.01)
```

- 多图端到端 greedy:

```text
HF multi-image token_ids: [785]
Prism multi-image token_ids: [785]
LLM.generate_vl multi-image one-token greedy HF alignment: PASS
```

- P2/P3.1 组合回归:

```text
30 passed in 78.39s
```

### 3. 视频输入 correctness

计划新增测试:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline_video.py \
  tests/test_vl_rope_index_video.py \
  tests/test_full_model_vl_video.py \
  tests/test_llm_vl_video_generate.py -s
```

PASS 标准:

- 测试视频样例可本地复现，不能依赖网络下载。
- processor 输出包含 `video_grid_thw` 或当前 transformers 版本对应的视频 grid 字段；shape 与 HF reference 一致。
- video token span 数、video grid token 数和 `input_ids` 中 video pad token 数一致。
- `position_ids/rope_delta` 与 HF `get_rope_index` exact match，max diff `0`。
- full logits 和 1-token greedy 与 HF 对齐。
- 视频路径不得复用 image-only hack；遇到 unsupported video processor 字段必须显式报错。

当前状态:

- 2026-06-25 已验证 PASS。
- 视频 processor/rope/LLM 轻量门禁:

```text
5 passed in 8.37s
video input_ids shape: [1, 420]
video pixel_values_videos shape: [1568, 1536]
video_grid_thw shape: [1, 3]
video_grid_thw: [[2, 28, 28]]
video tokens: 392 / expected 392
video pixel_values max diff: 0.000000e+00
video position_ids shape: [3, 1, 420]
video rope_delta shape: [1, 1]
video position_ids max diff: 0.000000e+00
video rope_delta max diff: 0.000000e+00
```

- 视频 full logits:

```text
input_ids shape: [1, 420]
pixel_values_videos shape: [1568, 1536]
video_grid_thw shape: [1, 3]
video tokens: 392 / expected 392
position_ids shape: [3, 1, 420]
Shape: HF=[1, 151936], Our=[1, 151936]
NaN: HF=0, Our=0
HF mean/std:  -1.130621e+00 / 4.290061e+00
Our mean/std: -1.130621e+00 / 4.290061e+00
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
PASS (max diff < 0.01)
```

- 视频端到端 greedy:

```text
HF video token_ids: [785]
Prism video token_ids: [785]
LLM.generate_video one-token greedy HF alignment: PASS
```

### 4. Batch 混合图文 correctness

计划新增测试:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_llm_vl_mixed_batch_generate.py \
  tests/test_qwen3_vl_attention_kv_mixed_batch.py -s
```

PASS 标准:

- 同一 prefill batch 支持 text-only、single-image、multi-image、video 请求混合。
- 同一 decode batch 支持上述请求混合，并正确使用各自的 1D/3D position ids 和 rope_delta。
- 同一 mixed batching/execution shape重复运行 token exact；跨 batch shape 的
  FP16/BF16低 margin分叉必须显式记录，不能把跨 shape exact当作通用正确性。
- model-precision logits须用 HF teacher-forced分布/PPL与独立任务质量门禁验证。
- `slot_mapping/block_tables/context_lens` 不串扰，KV cache 写入/读取 shape 和 max diff 有输出。
- P3.3 当时不支持的 prefix-cache/chunked prefill组合必须显式报错，不能 silent
  fallback；P7.3后续实现见本文件 P7.3节。

当前状态:

- 2026-06-25 已验证 PASS；实际测试文件为:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_llm_vl_mixed_batch_generate.py -s
```

- mixed ModelRunner 输入准备:

```text
2 passed in 8.20s
mixed prefill input_ids shape: [1043]
mixed prefill position_ids shape: [3, 1043]
mixed pixel_values shape: [2352, 1536]
mixed image_grid_thw shape: [3, 3]
mixed pixel_values_videos shape: [1568, 1536]
mixed video_grid_thw shape: [1, 3]
mixed cu_seqlens_q: [0, 5, 215, 623, 1043]
mixed slot_mapping shape: [1043]
mixed decode input_ids shape: [3]
mixed decode position_ids shape: [3, 3]
mixed decode context_lens: [6, 211, 421]
```

- mixed 公开入口（P7.4 model precision更新）:

```text
2 passed in 37.053s
single token_ids: [[11], [785], [785], [1986]]
mixed token_ids: [[11], [785], [785], [785]]
mixed repeat token_ids: [[11], [785], [785], [785]]
LLM.generate_mixed model-precision determinism contract: PASS
```

video row 的 batch1/batch4首 token分叉是显式数值边界；同一 mixed shape重复 exact，
image/multi-image 1-token仍跨 shape exact。它不替代下面的 HF logits/PPL门禁。

- P3.3 的历史 PASS不覆盖 prefix-cache/chunked-prefill VL mixed batch；P7.3后续
  单独建立了 chunked paged prefill与 online mixed-VL门禁，不追溯改写 P3.3范围。

### 5. 长输出多 token 质量评估

计划新增测试或脚本:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_llm_vl_long_generate.py \
  tests/test_vl_logits_distribution.py -s
```

PASS 标准:

- 至少覆盖 text-only、single-image、multi-image、video 四类输入；未支持项必须写成未验证风险。
- greedy `max_tokens=32` 必须记录稳定前缀、首个分叉 token 和必要的 logits 诊断；低 margin tie-break 场景不强制完整 32-token exact。
- 分布测试输出 logits shape、mean/std、max diff、mean diff、perplexity 或等价指标，采样模式 ppl diff `< 0.1`。
- 长输出测试必须记录 prompt token 数、image/video token 数、generated token 数、EOS 状态和总耗时。

当前状态:

- 2026-07-05 已刷新为稳定前缀 + teacher-forced logits/ppl 分布门禁。
- HF greedy 长输出:

```text
1 passed
single-image prompt tokens: 210
single-image prefix@8 match: True
single-image prefix@16 match: True
single-image first mismatch: 28
multi-image prompt tokens: 408
multi-image prefix@8 match: True
multi-image prefix@16 match: True
multi-image first mismatch: None
video prompt tokens: 422
video prefix@5 match: True
video first mismatch: 5
```

- mixed batch 长输出:

```text
mixed text prefix@8 match: True
mixed image/multi-image first mismatch: None or >=16
mixed video first mismatch: 0
mixed repeat long token_ids: exact
LLM.generate_mixed model-precision long-output contract: PASS
```

- logits/ppl 分布（P7.4 默认 model precision）:

```text
single-image logits shape HF/Prism: [1, 32, 151936]
single-image logits max/mean/ppl diff: 0 / 0 / 0
multi-image logits shape HF/Prism: [1, 32, 151936]
multi-image logits max/mean/ppl diff: 0 / 0 / 0
video logits shape HF/Prism: [1, 32, 151936]
video logits max/mean/ppl diff: 0 / 0 / 0
```

同一测试也保留显式 FP32历史路径，三类 max logit diff约
`0.123-0.125`、PPL diff均 `<0.007`，用于证明 model precision更贴近 HF而不是
以性能换取更大数值误差。

- text-only mixed batch numeric sensitivity:

```text
HF duplicate batch max diff: 5.312500e-01
HF duplicate batch mean diff: 1.473503e-01
Prism fp32 duplicate batch max/mean diff: 5.340242e-01 / 1.473883e-01
Prism model duplicate batch max/mean diff: 5.312500e-01 / 1.473503e-01
HF/Prism model argmax single/batch: 11/11
HF/Prism duplicate batch numeric sensitivity: PASS
```

- P3.4 完成时的剩余风险包括随机采样文本一致性、长上下文压力和
  prefix-cache/chunked-prefill VL mixed batch；P7.3已补 301/646-token chunked
  correctness基线，长上下文优化 kernel仍未完成。

### 6. VL CUDA Graph decode

计划新增测试:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_llm_vl_cuda_graph_decode.py \
  tests/test_model_runner_vl_cudagraph.py -s
```

PASS 标准:

- `enforce_eager=False` 下 VL decode 不再被公开 API 拒绝。
- CUDA Graph replay 的 logits/token ids 与 eager reference 对齐。
- 覆盖 batch size 1、2、非 graph 档位向上取整 replay 场景。
- 输出 graph input shapes: `input_ids`、3D `position_ids`、`slot_mapping`、`context_lens`、`block_tables`。
- benchmark 必须包含 warmup、repeat、median、p90、min、max、显存和 token/s。

当前状态:

- 2026-06-25 已验证 PASS。
- shape/batch 档位:

```text
text decode input positions shape: [2]
text graph positions shape: [3, 2]
vl decode input positions shape: [3, 3]
vl graph positions shape: [3, 3]
max_bs=3, graph_bs=[1, 2, 3]
max_bs=17, graph_bs=[1, 2, 4, 8, 16, 17]
ModelRunner CUDA Graph decode position shape normalization: PASS
ModelRunner CUDA Graph batch size coverage: PASS
```

- single/multi/video graph-vs-eager:

```text
single-image eager token_ids: [785, 2168]
single-image graph token_ids: [785, 2168]
multi-image eager token_ids: [785, 1378]
multi-image graph token_ids: [785, 1378]
video eager token_ids: [785, 2766]
video graph token_ids: [785, 2766]
```

- mixed batch=3 graph-vs-eager:

```text
mixed eager token_ids: [[11, 358], [785, 1378], [785, 2766]]
mixed graph token_ids: [[11, 358], [785, 1378], [785, 2766]]
mixed graph replay rounding: requested batch=3, replay graph batch=4
LLM.generate_mixed VL CUDA Graph decode equivalence: PASS
```

- 代表性 benchmark:

```text
commit: 45edd3a
gpu: NVIDIA GeForce RTX 5090
torch: 2.6.0a0+ecf3bae40a.nv25.01
case=mixed, max_tokens=8, warmup=2, repeat=5, kvcache_block_size=1024
correctness: PASS
eager decode: median=31.5488ms p90=34.2537ms min=30.9992ms max=34.5397ms token/s=93.96 decode_steps=35 decode_tokens=105
graph decode: median=16.4468ms p90=16.5553ms min=16.4189ms max=16.6193ms token/s=182.14 decode_steps=35 decode_tokens=105
memory: allocated=16.25MiB reserved=40.00MiB peak=27995.47MiB
```

- benchmark 脚本直接运行回归:

```text
command: .venv-local/bin/python benchmarks/bench_vl_cudagraph_decode.py --model <model_path> --case mixed --max-tokens 4 --warmup 1 --repeat 1
case=mixed, max_tokens=4, warmup=1, repeat=1, kvcache_block_size=1024
correctness: PASS
eager decode: median=48.1290ms p90=48.7788ms min=47.7548ms max=48.7788ms token/s=62.21
graph decode: median=16.4824ms p90=16.5760ms min=16.4728ms max=16.5760ms token/s=181.70
memory: allocated=16.25MiB reserved=40.00MiB peak=27995.47MiB
```

### 7. 高性能 paged decode kernel

验证命令:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_paged_decode_kernel.py \
  tests/test_qwen3_vl_attention_kv.py -s
```

benchmark:

```bash
.venv-local/bin/python benchmarks/bench_paged_decode.py \
  --batch-sizes 1,2,4,8 \
  --context-lens 256,1024,4096 \
  --warmup 10 \
  --repeat 50
```

PASS 标准:

- 当前 `_forward_decode_eager` 作为 reference；新 kernel 输出与 eager reference 对齐。
- 输出 q/k/v/cache/block table/context lens shape、max diff、mean diff、PASS/FAIL。
- 支持 Qwen3-VL GQA: `num_heads != num_kv_heads`。
- 失败或 unsupported shape 必须显式报错，不能回退到 eager 后仍报告 kernel PASS。
- benchmark 输出 warmup、repeat、median、p90、min、max、显存、token/s 和输入参数。

当前状态:

- 2026-06-25 已验证 PASS。
- correctness:

```text
paged kernel q shape: [3, 4, 16]
paged kernel k_cache shape: [9, 4, 2, 16]
paged kernel block_tables shape: [3, 3]
paged kernel context_lens: [1, 5, 9]
paged kernel max diff: 3.906250e-03
paged kernel mean diff: 1.447549e-04
paged decode Triton kernel correctness: PASS

paged kernel q shape: [2, 8, 128]
paged kernel k_cache shape: [6, 16, 2, 128]
paged kernel block_tables shape: [2, 3]
paged kernel context_lens: [17, 33]
paged kernel max diff: 7.812500e-03
paged kernel mean diff: 2.812790e-04
paged decode Triton kernel correctness: PASS

decode engine output shape: [1, 4, 16]
decode reference output shape: [1, 4, 16]
decode engine mean/std: 1.976967e-02 / 3.161756e-01
decode reference mean/std: 1.984596e-02 / 3.163016e-01
decode output max diff: 1.953125e-03
decode output mean diff: 3.700256e-04
engine attention decode paged KV kernel: PASS
```

- benchmark 矩阵: RTX 5090，commit `45edd3a`，torch `2.6.0a0+ecf3bae40a.nv25.01`，dtype bf16，warmup=10，repeat=50，num_heads=32，num_kv_heads=8，head_dim=128，block_size=256。
- 12 个 batch/context case 全部 correctness PASS。代表性结果:

```text
batch=1, context=256:  kernel median=0.0460ms, reference median=0.1264ms, max diff=1.953125e-03
batch=1, context=4096: kernel median=0.2834ms, reference median=0.2314ms, max diff=4.882812e-04
batch=4, context=1024: kernel median=0.0956ms, reference median=0.4969ms, max diff=9.765625e-04
batch=8, context=4096: kernel median=0.4662ms, reference median=1.8635ms, max diff=4.882812e-04
```

- 剩余风险: 当前 Triton kernel 是 baseline kernel，不是最终最优 kernel；batch=1/context=4096 慢于 SDPA reference，后续 P6 需要优化 block scheduling、vectorization 或专门长上下文单 batch 路径。

### 8. P3 Gate Review

P3 完成前必须运行:

```bash
.venv-local/bin/python -m compileall prism_infer tests benchmarks
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_sequence_multimodal.py \
  tests/test_vl_rope_index.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_sampler_greedy.py \
  tests/test_llm_vl_generate.py \
  tests/test_text_only_regression.py \
  tests/test_vision_rope_init.py \
  tests/test_processor_pipeline_multi_image.py \
  tests/test_vl_rope_index_multi_image.py \
  tests/test_processor_pipeline_video.py \
  tests/test_vl_rope_index_video.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_llm_vl_mixed_batch_generate.py \
  tests/test_llm_vl_long_generate.py \
  tests/test_llm_vl_cuda_graph_decode.py \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_vl_logits_distribution.py \
  tests/test_batch_numeric_sensitivity.py \
  tests/test_paged_decode_kernel.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl_multi_image.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_full_model_vl_video.py
```

PASS 标准:

- P1/P2 回归不退化。
- P3.1-P3.7 全部 PASS。
- 多图、视频、batch、长输出、CUDA Graph、paged decode kernel 都有 shape、max diff、mean/std 或 benchmark 输出。
- 未运行的重型测试必须写明原因和风险；不能宣称 P3 完成。

当前状态:

- 2026-06-25 已验证 PASS。
- `compileall prism_infer tests benchmarks`: PASS。
- `git diff --check`: PASS。
- grouped regression:

```text
49 passed in 356.34s
```

- full logits 串行验证:

```text
tests/test_full_model.py:
Shape: HF=[1, 64, 151936], Our=[1, 64, 151936]
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
Result: PASS

tests/test_full_model_vl.py:
Shape: HF=[1, 151936], Our=[1, 151936]
HF mean/std:  -1.756945e+00 / 4.123917e+00
Our mean/std: -1.756945e+00 / 4.123917e+00
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
Result: PASS

tests/test_full_model_vl_multi_image.py:
Shape: HF=[1, 151936], Our=[1, 151936]
HF mean/std:  -1.318763e+00 / 4.206440e+00
Our mean/std: -1.318763e+00 / 4.206440e+00
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
PASS

tests/test_full_model_vl_video.py:
Shape: HF=[1, 151936], Our=[1, 151936]
HF mean/std:  -1.130621e+00 / 4.290061e+00
Our mean/std: -1.130621e+00 / 4.290061e+00
Max diff:  0.000000e+00
Mean diff: 0.000000e+00
PASS
```

- P3 剩余风险:
  - P3.4 固定门槛已覆盖 `max_tokens=32` greedy 和 teacher-forced logits/ppl；随机采样文本一致性、长上下文压力仍属于后续扩展。
  - P7.3 已补 chunked paged prefill与 online mixed-VL；VL prefix hash因像素语义
    不安全而显式禁用，当前 paged prefill仍是 correctness-first gather+SDPA。
  - P3.6 kernel 是 baseline kernel；batch=1/context=4096 慢于 SDPA reference。
  - 真实视频文件采样策略、多卡 TP、4070/4090 benchmark 和 vLLM/SGLang 同条件对比未在 P3 完成。

## P4: KV Cache 分析验证

### 1. 语法检查

```bash
.venv-local/bin/python -m compileall prism_infer tests scripts
```

PASS 标准:

- `prism_infer/analysis/kv_trace.py`、trace 接入点、测试和脚本均无 Python 编译错误。

当前状态:

- 2026-06-26 已验证 PASS。

### 2. 轻量 trace/schema/summary 测试

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_kv_trace_no_output_change.py -s
```

PASS 标准:

- `TokenSpan` 能正确划分 text/image/video 连续区间。
- trace 文件 schema 稳定，字段完整。
- summary 能输出 visual attention mass、attention entropy、visual/text K norm ratio、head 差异、层间冗余。
- trace on/off 的 prefill 与 decode attention 输出一致，max diff `0.000000e+00`。

当前状态:

```text
5 passed
prefill trace off output shape: [5, 2, 4]
prefill trace on output shape: [5, 2, 4]
prefill trace output max diff: 0.000000e+00
decode trace off output shape: [1, 2, 4]
decode trace on output shape: [1, 2, 4]
decode trace output max diff: 0.000000e+00
decode attention entropy: recorded
```

### 3. 三类真实样例 trace 门禁

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python scripts/run_kv_trace_samples.py \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

PASS 标准:

- 每个样例先跑 trace off，再跑 trace on。
- trace on/off greedy token ids 完全一致，不一致直接失败。
- 覆盖三类输入:
  - `single_image_description`
  - `single_image_detail_qa`
  - `multi_image_comparison`
- 每个样例生成 JSONL trace、summary JSON、summary Markdown。
- 每个样例生成 summary SVG 可视化。
- summary 至少包含 `decode/prefill` 两个 phase、36 层 layer 记录、visual attention mass 和 KV norm 统计。

当前状态:

```text
single_image_description:
  token_ids: [32, 6303]
  layer records: 72
  steps: 2
  phases: ["decode", "prefill"]

single_image_detail_qa:
  token_ids: [2518, 151645]
  layer records: 72
  steps: 2
  phases: ["decode", "prefill"]

multi_image_comparison:
  token_ids: [28715, 389]
  layer records: 72
  steps: 2
  phases: ["decode", "prefill"]

manifest result: PASS
```

输出文件:

- `data/kv_trace_samples/single_image_description.jsonl`
- `data/kv_trace_samples/single_image_description.summary.json`
- `data/kv_trace_samples/single_image_description.summary.md`
- `data/kv_trace_samples/single_image_description.summary.svg`
- `data/kv_trace_samples/single_image_detail_qa.jsonl`
- `data/kv_trace_samples/single_image_detail_qa.summary.json`
- `data/kv_trace_samples/single_image_detail_qa.summary.md`
- `data/kv_trace_samples/single_image_detail_qa.summary.svg`
- `data/kv_trace_samples/multi_image_comparison.jsonl`
- `data/kv_trace_samples/multi_image_comparison.summary.json`
- `data/kv_trace_samples/multi_image_comparison.summary.md`
- `data/kv_trace_samples/multi_image_comparison.summary.svg`
- `data/kv_trace_samples/manifest.json`

说明:

- `data/` 目录被 `.gitignore` 排除，原始 trace 不入库；文档保留复现命令和结果摘要。
- P4 trace 是分析路径，不用于 benchmark 性能数字。

## P4.5: KV Engine Hardening 验证

### 1. 语法检查

```bash
.venv-local/bin/python -m compileall prism_infer tests
```

PASS 标准:

- KV layout、BlockManager、Sequence swap state、Scheduler、ModelRunner 和新增测试均无 Python 编译错误。

当前状态:

```text
compileall prism_infer tests: PASS
```

### 2. P4.5 focused invariant 测试

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_kv_engine_hardening.py \
  tests/test_scheduler_swap_tables.py -s
```

PASS 标准:

- `store_kvcache` CPU fallback 按 flat slot 写入 canonical 4D paged cache。
- 释放最后一个 block 引用后，`hash_to_block_id` 不残留指向 free block 的 stale hash。
- `swap_out()` 后 `seq.block_table == []`，CPU block id 只进入 `seq.cpu_block_table`。
- `swap_in()` 后 `seq.cpu_block_table == []`，GPU block id 恢复到 `seq.block_table`。
- Scheduler 使用 `cpu_block_table` 判断 swapped sequence 的换入容量。
- prefix-cache prefill 未实现路径在 `ModelRunner.prepare_prefill` 阶段显式报错。

当前状态:

```text
5 passed

store key input shape: [5, 2, 3]
store cache shape: [3, 4, 2, 3]
store slot_mapping: [0, 3, 4, 9, -1]
store k_cache max diff: 0.000000e+00
store v_cache max diff: 0.000000e+00
KV layout 4D eager store: PASS

deallocated block id: 0
released block hash: 8356527653647720045
hash index keys after deallocate: []
free block ids after deallocate: [0, 1, 2, 3]
BlockManager hash cleanup: PASS

swap out map: [(0, 0), (1, 1)]
gpu block_table after swap_out: []
cpu block_table after swap_out: [0, 1]
swap in map: [(0, 2), (1, 3)]
gpu block_table after swap_in: [2, 3]
cpu block_table after swap_in: []
BlockManager swap table split: PASS

scheduler initial swap map: [(0, 0)]
scheduler swap_in_map: [(0, 1)]
scheduler seq block_table after swap_in: [1, 2]
scheduler seq cpu_block_table after swap_in: []
Scheduler swap table capacity: PASS

prefix-cache prefill early gate: PASS
```

### 3. 受影响窄回归

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_sequence_multimodal.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_kv_trace_no_output_change.py -s
```

PASS 标准:

- Sequence 序列化保留 `block_table/cpu_block_table` 语义，不破坏 VL prefill/decode payload。
- Qwen3-VL engine attention prefill/decode KV correctness 不退化。
- ModelRunner 单图和 mixed text/VL prefill/decode 输入准备不退化。
- KV trace on/off 小张量输出仍 exact match。

当前状态:

```text
12 passed in 11.87s

engine attention prefill KV:
  hidden input shape: [1, 7, 64]
  engine output shape: [7, 64]
  k_cache shape: [1, 7, 2, 16]
  attention output max diff: 0.000000e+00
  k_cache max diff: 0.000000e+00
  v_cache max diff: 0.000000e+00
  PASS

engine attention decode paged KV:
  decode q shape: [1, 4, 16]
  decode engine/reference output shape: [1, 4, 16]
  decode output max diff: 1.953125e-03
  decode output mean diff: 3.700256e-04
  PASS

mixed prefill:
  input_ids shape: [1043]
  position_ids shape: [3, 1043]
  pixel_values shape: [2352, 1536]
  image_grid_thw shape: [3, 3]
  pixel_values_videos shape: [1568, 1536]
  video_grid_thw shape: [1, 3]
  PASS

trace on/off:
  output shape: [1, 2, 4]
  max diff: 0.000000e+00
  mean diff: 0.000000e+00
  visual attention mass: 4.361440e-01
  PASS
```

### 4. Paged decode / attention regression

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_kv_engine_hardening.py \
  tests/test_scheduler_swap_tables.py \
  tests/test_paged_decode_kernel.py \
  tests/test_qwen3_vl_attention_kv.py -s
```

PASS 标准:

- P4.5 invariant 测试 PASS。
- P3.6 paged decode Triton kernel correctness 不退化。
- Engine attention prefill/decode KV correctness 不退化。

当前状态:

```text
10 passed in 4.98s

paged kernel small GQA:
  q shape: [3, 4, 16]
  k_cache shape: [9, 4, 2, 16]
  block_tables shape: [3, 3]
  context_lens: [1, 5, 9]
  max diff: 3.906250e-03
  mean diff: 1.447549e-04
  PASS

paged kernel Qwen shape:
  q shape: [2, 8, 128]
  k_cache shape: [6, 16, 2, 128]
  block_tables shape: [2, 3]
  context_lens: [17, 33]
  max diff: 7.812500e-03
  mean diff: 2.812790e-04
  PASS
```

剩余风险:

- P4.5 不声明 prefix-cache prefill 可用；当前只是 early gate。
- P4.5 不改变 `kvcache_block_size=256`；P5.0 已决定保留当前物理 page 并先接入逻辑 compression metadata，sub-page pruning/compaction 留到 active compression 阶段。
- P4.5 不解决 swap 全局 synchronize、paged decode kernel 参数调优、mixed chunked prefill+decode 调度，这些属于 P6 性能优化。
- 本轮未重跑 P1/P2/P3 全量重型 full logits；修改点集中在 KV 管理和 fallback，已跑窄回归。若后续合并前需要阶段 release，应再跑 grouped regression 和 full logits 串行门禁。

## P5: 压缩策略验证

P5 必须把 off baseline、离线 scoring 和 active compression 分开验证。P5.0/P5.1
不能替代 P5.2+ 的 compression-on 质量和性能门禁。

### P5.0 Compression-Off Baseline

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_compression_off.py \
  tests/test_text_only_regression.py -s
```

PASS 标准:

- compression off 与 FP baseline 完全一致。
- `compression_mode="off"` 可通过公开 LLM 配置入口运行。
- compression metadata 能记录 step phase、batch size、prompt tokens、image/video visual token counts 和 block size。
- 未实现 compression mode 显式报错，不 silent fallback。

当前状态:

- P5.0 已接入 `compression_mode="off"` 和 per-step `CompressionMetadata`。
- `compression_mode="visual_prune"` 已在 P5.2 接入为 active logical pruning；其他未实现模式仍必须显式失败。
- focused verification 已扩展到 P5.2 active tests；历史 P5.0 `tests/test_compression_off.py` 结果不再代表完整 P5 focused 集合。

### P5.1 Visual Importance Scoring

P5.1 是离线分析，不运行模型，不修改 runtime KV cache。

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_visual_importance_scoring.py \
  tests/test_visual_token_stats.py \
  tests/test_analysis_schema.py -s
```

CLI smoke:

```bash
.venv-local/bin/python scripts/score_visual_tokens.py \
  data/kv_trace_samples/single_image_description.jsonl \
  --output-json data/kv_trace_samples/single_image_description.importance.json \
  --markdown data/kv_trace_samples/single_image_description.importance.md
```

PASS 标准:

- scorer 能读取 P4 trace schema 中的 `attention.sequence_stats[].span_masses` 和 `top_visual_tokens`。
- 输出 visual token ranking、visual span ranking、keep-ratio simulation 和 limitations。
- text-only trace 输出空 visual ranking，不报错。
- image/video span 均进入 ranking。
- CLI 能写出 JSON 和 Markdown。

当前状态:

- CPU-only synthetic focused test: `tests/test_visual_importance_scoring.py` 为 `4 passed in 1.38s`。
- P5.1 只输出 importance proxy；不声明压缩率、显存收益、latency 或质量收益。
- P4 trace 未保存完整 per-token attention distribution；`top_visual_tokens` 只用于细化已记录 top-k token，未进入 top-k 的 visual tokens 使用 span mass 剩余量均分。

### P5.2-A Decision Shadow Mode

P5.2-A 只验证 runtime compression 之前的 keep/drop decision contract 和
shadow metadata 接入。它不能替代 P5.2+ compression-on 质量和性能门禁。

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_compression_off.py \
  tests/test_visual_pruning.py -s
```

PASS 标准:

- `compression_mode="off"` 的 shadow metadata 仍为 no-op。
- off-only helper 对 active metadata 显式失败；`Attention.forward` 对未实现 compression metadata 显式失败。
- visual-token span 扫描支持 image/video 多段，不假设一个连续 visual span。
- decision record 包含 visual token 总数、保留数、丢弃数、keep ratio、strategy、span、kept/dropped token indices 和 physical compaction 状态。
- `enable_visual_pruning_shadow=True` 时，prefill `CompressionMetadata` 记录 visual pruning decision，但 `metadata.enabled` 仍为 `False`。
- decode `CompressionMetadata` 不重算 pruning decision，避免依赖 decode 阶段不完整的 `token_ids`。
- 携带 shadow decision records 的 attention 输出必须与无 metadata 输出完全一致。
- `score` strategy 缺少 token score 时显式失败，不能 fallback 到 uniform。
- slot mask helper 只作为 prefill 实验工具；不能据此声明 active compression 完成。

当前状态:

- CPU-only focused test 已扩展为 `tests/test_compression_off.py` + `tests/test_visual_pruning.py` + `tests/test_visual_pruning_active.py`，当前为 `20 passed in 0.19s`。
- `prism_infer.engine.visual_pruning` 已接入 shadow metadata 和 active logical decode retention；shadow mode 本身仍不改变 KV。

### P5.2 Active Logical Visual Pruning

PASS 标准:

- compression on 必须有明确 runtime decision record: 每条请求的 visual token 总数、保留数、丢弃数、keep ratio、使用的 score/threshold/config。
- compression on 的 KV shape、block mapping 和 decode 状态一致。
- logical pruning/retention 若不做 physical compaction，必须显式记录该限制；若实现 physical compaction，必须额外验证 slot mapping、context length、block table、prefix/swap 状态和 M-RoPE position 语义。
- 输出压缩率、质量退化、显存、latency 或 throughput 数据。
- compression on greedy/token distribution 门禁与 P1-P4 FP baseline 对比通过。
- 任一 unsupported compression mode 必须显式失败，不能 silent fallback。
- VScan/PoRe、DeepStack-aware pruning、M-RoPE block compaction 或竞品对比数字在未实现、未跑同条件 benchmark 前，只能写为候选路线或未验证风险；FP8 KV 的当前项目 baseline 见 P5.3/P5.4。

Focused correctness:

```bash
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_visual_pruning_active.py \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning.py -s
```

当前输出:

- `20 passed in 0.19s`。
- active compact reference: output shape `[1,4,8]`，reference shape `[1,4,8]`，max diff `0.000000e+00`，mean/std 完全一致。
- keep-all active/off decode: max diff `0.000000e+00`。
- missing active decision record 显式 `RuntimeError`。

真实模型 smoke:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
/data/Prism-Infer/.venv-local/bin/python <inline smoke>
```

当前输出:

- 单图 `max_tokens=2`, keep-all: off token ids `[785, 2168]`，active token ids `[785, 2168]`，exact match。
- 单图 `max_tokens=8`, `keep_ratio=0.5`: visual tokens `196 -> 98`，off 与 active token ids 均为 `[785, 2168, 3897, 374, 264, 6437, 11, 13794]`，8/8 exact match。

小型端到端 benchmark:

- GPU: NVIDIA GeForce RTX 5090。
- 输入: 单图 448x448，prompt `Describe this image.`，`max_tokens=8`。
- warmup=1，repeat=3，每次测量前后调用 `torch.cuda.synchronize()`。
- off latency median/p90/min/max: `0.292072/0.294718/0.284047/0.294718s`。
- `visual_prune keep_ratio=0.5` latency median/p90/min/max: `0.798550/0.810786/0.781913/0.810786s`。
- off output token/s median: `27.390514`；active output token/s median: `10.018163`。
- off memory allocated/reserved/peak median: `25743.02/28348.00/28122.23 MB`。
- active memory allocated/reserved/peak median: `25707.93/28312.00/28087.13 MB`。
- 解释: active logical pruning 当前没有 physical KV compaction；显存数字接近，只能说明小样例当前没有可声明的物理显存收益。active 路径更慢，不能声明吞吐收益。

当前状态:

- `compression_mode="visual_prune"` 已实现 prefill decision 持久化、decode retained-token KV view、keep-all exact no-op 和小样例 compression-on smoke。
- 当前 `visual_prune` mode 是 logical pruning，不是 physical KV compaction；它自身未满足物理显存收益门禁。
- P5 的收益门禁由 P5.3/P5.4 `fp8_kv` baseline 补齐。

### P5.3/P5.4 FP8 KV Baseline

`fp8_kv` 是 P5 当前满足收益门禁的 physical KV storage baseline。它不改变
logical context，也不做 visual-token compaction；它把 KV cache 物理 dtype 从
model dtype bf16 改为 `torch.float8_e4m3fn`，decode 读取时显式 dequant 到 query
dtype 后运行 attention。

Focused correctness:

```bash
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_fp8_kv_cache.py \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning_active.py -s
```

当前输出:

- `16 passed in 0.23s`。
- FP8 store: key shape `[5,2,8]`，cache shape `[2,4,2,8]`。
- one-tensor BF16 cache bytes `256`，FP8 cache bytes `128`。
- store round-trip max diff `0.000000e+00` against explicit `to(fp8).to(bf16)` reference。
- FP8 decode: output shape `[1,4,8]`，reference shape `[1,4,8]`，mean/std 完全一致，max diff `0.000000e+00`。

受影响窄回归:

```bash
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py \
  /data/Prism-Infer/tests/test_kv_engine_hardening.py \
  /data/Prism-Infer/tests/test_kv_trace_no_output_change.py -s
```

当前输出:

- `11 passed in 4.15s`。
- Qwen attention decode paged KV kernel 仍 PASS，max diff `1.953125e-03`。
- 4D KV store、BlockManager hardening、swap、prefix-cache early gate 和 trace on/off equality 均 PASS。

真实模型 fixed-block smoke:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
/data/Prism-Infer/.venv-local/bin/python <inline smoke>
```

当前输出:

- off KV dtype `torch.bfloat16`，shape `[2,36,16,256,8,128]`，bytes `603979776`。
- fp8 KV dtype `torch.float8_e4m3fn`，shape `[2,36,16,256,8,128]`，bytes `301989888`。
- `kv byte ratio fp8/off: 0.500000`。
- 单图 `max_tokens=4`: off 和 fp8 token ids 均为 `[785, 2168, 3897, 374]`。

可复现 benchmark:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
/data/Prism-Infer/.venv-local/bin/python \
  /data/Prism-Infer/benchmarks/bench_kv_compression.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --case single_image \
  --modes off,fp8_kv \
  --max-tokens 8 \
  --warmup 1 \
  --repeat 3 \
  --num-kvcache-blocks 16 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024
```

原始日志:

- `data/p5_compression/fp8_kv_single_image_benchmark_20260709.jsonl`
- `data/p5_compression/fp8_kv_quality_matrix_20260709.jsonl`

single-image benchmark 当前输出:

- GPU: NVIDIA GeForce RTX 5090。
- 输入: 单图 448x448，prompt `Describe this image.`，`max_tokens=8`。
- warmup=1，repeat=3，每次测量前后调用 `torch.cuda.synchronize()`。
- off latency median/p90/min/max: `0.278173/0.287490/0.274921/0.287490s`。
- fp8 latency median/p90/min/max: `0.704317/0.704871/0.692049/0.704871s`。
- off output token/s median: `28.759042`；fp8 output token/s median: `11.358516`。
- off memory allocated/reserved/peak median: `17319.02/19938.00/19698.23 MB`。
- fp8 memory allocated/reserved/peak median: `17023.57/19626.00/19402.86 MB`。
- KV cache bytes: off `603979776`，fp8 `301989888`，ratio `0.500000`。
- token ids 均为 `[785, 2168, 3897, 374, 264, 6437, 11, 13794]`。

quality matrix 当前输出:

- 覆盖 `text`、`single_image`、`multi_image`、`video` 四类，每类 `max_tokens=8`。
- `fp8_kv` 对 `off` aggregate token match: `32/32`，exact match。
- per-case exact: text `8/8`，single-image `8/8`，multi-image `8/8`，video `8/8`。
- KV byte ratio: `0.5`。

P5 当前结论:

- `fp8_kv` 已给出 compression ratio、质量退化、显存和 latency/throughput 数据。
- 可测收益是 KV cache physical bytes 减半，以及 fixed-block 场景 GPU allocated/reserved/peak 下降约 295/312/295 MB。
- 当前 FP8 decode 路径为了 correctness 走 eager dequant + SDPA，latency/throughput 明显慢于 off；不能声明吞吐收益。
- P5 当前门禁已满足，后续性能优化属于 P6。

## P6: 系统优化与视觉 KV 物理压缩验证

P6 的设计、模块边界、阶段顺序和非目标见 `docs/P6_SYSTEM_OPTIMIZATION_DESIGN.md`。P6 必须先建立统一 baseline，再推进 profiling、physical compaction、kernel、compile 和外部对比。

### P6.0 设计门禁

PASS 标准:

- 明确 internal correctness/kernel/system/external baseline 层级。
- ExecutionMode、AttentionBackend 和 CompressionMode 作为正交实验维度。
- physical compaction 明确区分 logical context length 和 physical KV length。
- `torch.compile` 只覆盖稳定 tensor compute region，不把动态 scheduler 当成编译目标。
- megakernel 只有在 profiler 证明 launch-bound 且存在真实实现时启动。
- PD 分离和投机解码不进入 P6 核心门禁。

当前状态:

- `docs/P6_SYSTEM_OPTIMIZATION_DESIGN.md` 已建立，P6.0 完成。
- pruning 外部 PR 尚无链接/commit，当前不能作为实现依据或 benchmark baseline。

### P6.1 统一 Benchmark Contract

计划入口:

```text
benchmarks/bench_system.py
tests/test_benchmark_schema.py
docs/PERFORMANCE_REPORT.md
```

当前实现已落地。性能记录与结论边界见 `docs/PERFORMANCE_REPORT.md`。

每条 JSONL result 必须包含:

- schema version、timestamp、git commit 和 dirty state。
- GPU、CUDA、torch、transformers、Python。
- model snapshot、dtype、TP size、block size、max model len、显存利用率。
- execution mode、attention mode、compression mode 和全部压缩参数。
- case id、text token count、visual token count、图像/视频 shape、output token count。
- preprocessing 是否计时。
- warmup、repeat、batch/concurrency/request rate。
- correctness summary 和 output token ids/hash。
- TTFT、TPOT/ITL、end-to-end latency、throughput、memory、KV bytes 和 block count。

PASS 标准:

- 同一个 workload manifest 可运行 off/eager、off/CUDA Graph、visual-prune 和 fp8-kv internal modes。
- schema test 拒绝缺少版本、输入、timing、memory 或 correctness 字段的 result。
- deterministic greedy case 的 repeat 输出一致。
- benchmark runner 只采集数据，不在内部自动改变 mode 以规避失败。
- 第一份 `docs/PERFORMANCE_REPORT.md` 只记录 baseline，不提前写优化收益。

当前验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q tests/test_benchmark_schema.py -s
```

当前输出: `13 passed in 0.02s`。覆盖五类 workload manifest、完整 record、缺失 environment/workload/timing/memory/KV evidence、output hash 不一致、统计顺序错误和 offline traffic 一致性。

RTX 5090 runner validation 命令:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --case single_image_448 \
  --modes off_eager,off_graph,visual_prune,fp8_kv \
  --max-tokens 8 --warmup 1 --repeat 3 \
  --num-kvcache-blocks 16 --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --output data/p6_system/single_image_internal_baseline_20260711.jsonl
```

当前输出:

- 4 条 JSONL record 逐条通过 `validate_benchmark_record()`。
- mode 为 `off_eager/off_graph/visual_prune/fp8_kv`，repeat 内输出一致，相对首模式均为 8/8 token exact。
- warmup/repeat 为 `1/3`，记录 `torch.cuda.synchronize()` timing boundary、输入 shape、E2E/engine TTFT、decode ITL、E2E latency、request/token throughput、allocated/reserved/peak 和 KV bytes。
- KV bytes: BF16 三模式 `603979776`，FP8 `301989888`；visual-prune 仍为 logical path，不具备 physical KV bytes 收益。
- 工作树为 `git_dirty=true`；该结果只能标记 runner validation baseline，正式 clean-commit rerun 尚未完成。
- 完整数值和限制见 `docs/PERFORMANCE_REPORT.md`，不在此验证索引重复维护。

### P6.2 Profiling 门禁

每个待优化问题必须先记录:

- processor、vision、prefill、decode、sample 和 scheduler 分段时间。
- kernel count 和 CPU launch gap。
- GPU utilization 和 peak memory。
- 目标 batch/context/visual-token shape。
- profiler 命令、原始 trace 路径和 commit。

针对当前 P5 慢路径还必须分别测量:

- visual-prune retained index 构造和 paged gather。
- `.item()`/Python loop 引入的同步或 launch gap。
- FP8 store、cache load/dequant 和 SDPA。
- batch=1/context=4096 paged decode kernel。

没有 profiler 或分段计时证据时，不进入 megakernel，也不声称某个函数是主要瓶颈。

当前实现入口:

```text
prism_infer/analysis/performance_profile.py
benchmarks/bench_system.py --profile-output ...
benchmarks/analyze_nsys.py
tests/test_performance_profile.py
tests/test_nsys_analysis.py
```

semantic collector focused 验证:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_performance_profile.py \
  tests/test_nsys_analysis.py -s
```

当前输出: `6 passed`。其中 profiling on/off CPU attention output shape 均为 `[4, 2, 8]`，max diff `0.000000e+00`；同时覆盖 profile schema/tamper guard、phase summary、Nsight correlation 和 eager capture 无 graph table。

四模式 semantic profile 命令在 P6.1 runner 命令基础上增加:

```bash
--profile-repeat 1 \
--profile-output data/p6_system/single_image_semantic_profile_20260711.jsonl
```

当前输出:

- 4 条 profile records 通过 `validate_performance_profile_record()`；每条包含 1 个 prefill、7 个 decode step，profiled output 对 unprofiled output token exact。
- `off_eager/off_graph/visual_prune/fp8_kv` 的 profiled decode `engine.model_runner` CUDA/step 分别为 `37.213/17.737/116.964/51.229 ms`。这些值包含 profiling overhead，只用于分层归因，不替代 unprofiled benchmark。
- `visual_prune` 每 decode step: retained-index CPU `3.006 ms`、gather CUDA `69.055 ms`、SDPA CUDA `2.554 ms`；context `211..217`，retained length `113..119`。
- FP8 prefill 36 层 eager KV store CUDA 合计 `373.769 ms`；FP8 decode 每 step store/gather/dequant/SDPA CUDA 分别为 `5.042/4.008/2.420/2.620 ms`。

Nsight capture 模板:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=data/p6_system/<mode>_profile_20260711 \
  .venv-local/bin/python benchmarks/bench_system.py \
  <same model/workload args> \
  --modes <mode> --warmup 1 --repeat 1 --profile-repeat 1 \
  --cuda-profiler-range \
  --profile-output data/p6_system/<mode>_nsys_semantic_20260711.jsonl
```

SQLite 结构化分析:

```bash
nsys export --type sqlite --force-overwrite=true \
  --output=data/p6_system/<mode>_profile_20260711.sqlite \
  data/p6_system/<mode>_profile_20260711.nsys-rep

PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python benchmarks/analyze_nsys.py \
  data/p6_system/<mode>_profile_20260711.sqlite \
  --target-range '<prism NVTX range>' \
  --output data/p6_system/<mode>_nsys_summary_20260711.json
```

Nsight single-image decode median per step:

| Mode | Explicit kernels | Graph launch | Async memcpy | Stream sync | Kernel busy | Graph execution |
|---|---:|---:|---:|---:|---:|---:|
| `off_eager` | 2004 | 0 | 10 | 2 | 16.137 ms | 0 |
| `off_graph` | 11 | 1 | 14 | 2 | 4.074 ms | 12.896 ms |
| `visual_prune` | 2148 | 0 | 3610 | 3566 | 17.384 ms | 0 |
| `fp8_kv` | 2220 | 0 | 298 | 110 | 16.112 ms | 0 |

注意: `off_graph` 的 explicit kernel 数不包含 graph 内部 node；graph execution 来自 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`。NVTX target 通过 runtime `correlationId` 关联 GPU activity，不能用 CPU NVTX 时间包含关系归因异步 kernel。

关键 target range 证据:

- 252 个 `visual_prune.gather` ranges 对应 24696 次 async memcpy 和 24696 次 stream sync，恰为每 layer/decode gather 98 次；源码路径包含逐 retained segment 的 `block_ids[...].item()`。
- 288 个 FP8 eager KV-store ranges 对应 7812 次 stream sync，等于 `36 layers * (210 prefill tokens + 7 decode tokens)`；源码路径逐 token 执行 `slot_mapping[i].item()` 和 K/V assignment。
- `off_eager` 的 252 个 paged Triton ranges 通过 correlation 得到 252 个 kernel，不再使用会漏掉异步 kernel 的时间戳 containment。

batch=1 paged decode kernel 验证:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python benchmarks/bench_paged_decode.py \
  --batch-sizes 1 --context-lens 256,1024,4096 \
  --warmup 10 --repeat 50
```

当前输出:

- context `256`: max diff `1.953125e-03` PASS；kernel/reference median `0.0387/0.1352 ms`。
- context `1024`: max diff `9.765625e-04` PASS；kernel/reference median `0.0873/0.1473 ms`。
- context `4096`: max diff `4.882812e-04` PASS；kernel/reference median `0.2674/0.2071 ms`，当前自实现 kernel 不占优。

P6.2 当前门禁状态:

- processor、M-RoPE、vision、prefill、decode、sampler、scheduler、compression subregion、kernel/API/sync 和 peak memory 已有实测证据。
- Nsight `--gpu-metrics-devices=help` 对 RTX 5090 返回 `Already under profiling`；SM utilization 未验证，不能用 `kernel_busy_ms` 替代。
- 当前所有记录为 `git_dirty=true`，属于瓶颈定位证据，不是正式发布 benchmark。
- 因 SM utilization 缺口，P6.2 不标记无条件 PASS；但 visual gather、FP8 eager store 和 eager decode launch 病理已有足够证据进入对应 focused optimization。

#### P6.2-C Visual Retained Gather 优化

实现 contract:

- `build_retained_slot_mapping()` 在每个 decode step 使用 CPU sequence block table，把 retained logical indices 映射为一个 int64 physical-slot tensor。
- `ModelRunner.prepare_decode()` 每个 sequence 只构造一次 mapping，并通过 `Context.visual_pruning_slot_mappings` 传给全部 attention layers。
- attention 将 canonical paged KV reshape 为 flat slots，通过两次 `index_select` 收集 K/V；不再逐 retained segment 调用 `block_ids[...].item()`。
- 该路径不移动 KV、不释放 block，`physical_compaction` 仍为 `False`。

focused correctness:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_visual_pruning_active.py \
  tests/test_compression_off.py \
  tests/test_model_runner_context_reset.py -s
```

当前输出: `19 passed in 2.91s`。active/reference output shape 均为 `[1, 4, 8]`，max diff `0.000000e+00`；keep-all/off exact；非连续 block `[3, 7]` 下 logical indices `(0, 1, 4, 5, 6)` 映射 physical slots `[12, 13, 28, 29, 30]`；short table 和 missing mapping 显式失败。

同轮 unprofiled benchmark:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  <same P6 model/workload args> \
  --modes off_eager,visual_prune \
  --max-tokens 8 --warmup 1 --repeat 3 --profile-repeat 1 \
  --output data/p6_system/visual_gather_optimized_benchmark_20260711.jsonl \
  --profile-output data/p6_system/visual_gather_optimized_semantic_20260711.jsonl
```

当前输出:

- 四次路径（3 measured + 1 profiled）输出均为 `[785, 2168, 3897, 374, 264, 6437, 11, 13794]`。
- decode-step median: off eager `30.834 ms`，visual prune `33.529 ms`，ratio `1.087x`。
- visual-prune 相对优化前 P6.2 profile 对应 unprofiled median `100.544 ms` 下降 `66.7%`；跨运行对比只用于定位趋势，正式 clean-commit 数字待提交后复跑。
- semantic visual gather CUDA 从约 `69.055 ms/step` 降至 `3.108 ms/step`；slot mapping 构造 CUDA 约 `0.160 ms/step`。

优化后 Nsight target:

```text
before visual gather ranges: 252
  async memcpy: 24696
  stream sync: 24696

after visual gather ranges: 252
  async memcpy: 0
  stream sync: 0
```

整步 decode median 的 async memcpy/stream sync 从 `3610/3566` 降为 `47/2`；stream sync 已回到 off-eager baseline 的 `2`。显式 kernel 数仍为 `2148`，因为当前仍是 gather + GQA expand + SDPA 多算子路径，没有把普通 tensorized gather 伪称为 fused kernel。

quality matrix:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_kv_compression.py \
  --model <Qwen3-VL snapshot> --case quality_matrix \
  --modes off,visual_prune --max-tokens 8 \
  --num-kvcache-blocks 16 --max-model-len 1024 \
  --max-num-batched-tokens 1024
```

当前输出: text/single-image/multi-image/video 均为 `8/8` exact，aggregate `32/32` exact；KV byte ratio `1.0`。这证明当前 optimization 保持 logical pruning 质量，但不构成 physical KV compression。

本轮 focused regression:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py \
  tests/test_performance_profile.py \
  tests/test_nsys_analysis.py \
  tests/test_compression_off.py \
  tests/test_visual_pruning_active.py \
  tests/test_fp8_kv_cache.py \
  tests/test_kv_trace_no_output_change.py \
  tests/test_paged_decode_kernel.py \
  tests/test_model_runner_context_reset.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_qwen3_vl_attention_kv.py -s
```

当前输出: `52 passed in 8.22s`。profile on/off attention output max diff `0.000000e+00`，KV trace on/off max diff `0.000000e+00`，active visual prune/FP8 focused reference max diff 均为 `0.000000e+00`，paged Triton focused cases max diff 分别为 `3.906250e-03/7.812500e-03`；ModelRunner context、单/混合 VL prefill 和 Qwen3-VL attention KV 回归均通过。

真实模型独立 HF greedy 回归:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_llm_vl_generate.py::test_generate_vl_one_token_matches_hf_greedy -s
```

当前输出: `1 passed in 19.55s`；HF token ids `[785]`，Prism token ids `[785]`，exact match。

#### P6.2-D FP8 Vectorized KV Store

实现 contract:

- `store_kvcache()` 对 contiguous CUDA FP8 cache 复用项目自实现 `_store_kvcache_triton`，一个调用同时写 K/V；`tl.store` 按 destination pointer dtype 执行当前 unit-scale BF16/FP32 -> E4M3FN 转换。
- `_store_kvcache_eager()` 保留为 CPU/fallback/reference，不改变 P5.3 FP8 量化语义。
- FP8 Triton 路径使用独立 NVTX region `attention.kv_store.triton_fp8`；CUDA tensor/device/layout 不一致时显式失败。
- 本项只优化 store，不改变 FP8 cache layout、bytes、decode gather/dequant 或 attention 计算。

focused correctness:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_fp8_kv_cache.py \
  tests/test_kv_engine_hardening.py \
  tests/test_compression_off.py \
  tests/test_qwen3_vl_attention_kv.py -s
```

当前输出: `23 passed in 4.93s`。Qwen prefill shape key/value `[210,8,128]`、cache `[3,256,8,128]`，跨非连续 physical slots 且含 `-1` padding；Triton/eager K/V max diff 均为 `0.000000e+00`，output/reference mean/std exact，untouched slot 保持原值。FP8 decode output/reference shape `[1,4,8]`，max diff `0.000000e+00`。

同轮 unprofiled system benchmark:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model <Qwen3-VL snapshot> --case single_image_448 \
  --modes off_eager,fp8_kv --max-tokens 8 \
  --warmup 1 --repeat 3 --profile-repeat 1 \
  --max-model-len 1024 --max-num-batched-tokens 1024 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --output data/p6_system/fp8_vectorized_benchmark_20260711.jsonl \
  --profile-output data/p6_system/fp8_vectorized_semantic_20260711.jsonl
```

当前输出:

- off/fp8 8-token outputs 均为 `[785,2168,3897,374,264,6437,11,13794]`，token exact。
- KV bytes: BF16 `603979776`，FP8 `301989888`，ratio `0.5`。
- decode-step median: off `31.077 ms`，FP8 `35.865 ms`，ratio `1.154x`；store 优化没有解决 FP8 full-context gather/dequant latency。
- unprofiled FP8 prefill median 从旧 raw baseline `446.759 ms` 降到 `132.951 ms`，但本轮 prefill 样本为 `51.270..137.414 ms`，波动较大，只作为 focused trend。
- semantic FP8 prefill 36 层 store CUDA 合计从 `373.769 ms` 降到 `0.606 ms`，profiled `model.language_model` prefill 从 `412.194 ms` 降到 `35.627 ms`。

Nsight target before/after:

| Metric | Eager per-token store | Triton FP8 store |
|---|---:|---:|
| ranges | 288 | 288 |
| kernels | 15624 | 288 |
| runtime APIs | 46879 | 288 |
| async memcpy | 23436 | 0 |
| stream sync | 7812 | 0 |
| target kernel time total | 15.192 ms | 0.286 ms |

整步 decode median 的 explicit kernels/async memcpy/stream sync 从 `2220/298/110` 降到 `2184/190/74`。剩余 `74` 次 stream sync 不在 FP8 store target 内，主要后续调查对象是逐层 context-length 读取和 FP8 eager gather。RTX 5090 NCU 返回 `ERR_NVGPUCTRPERM`，因此仍没有 SM utilization/occupancy/DRAM counter，不以 kernel busy 代替。

质量矩阵重新运行 `off,fp8_kv`，text/single-image/multi-image/video 均 `8/8` exact，aggregate `32/32` exact，KV byte ratio `0.5`。

本轮综合回归在新增 FP8 Triton test 后为 `53 passed in 8.01s`；`compileall prism_infer tests benchmarks scripts` 和 `git diff --check` PASS。

### P6.3 Execution Backend 验证

#### P6.3-A Eager/CUDA Graph Matrix

实现与记录 contract:

- benchmark schema 升为 v2，同时继续校验历史 v1 records；v2 增加 source request count、replication factor、prefill/decode backend、Graph capture scope/time、captured buckets、selected batch 和 padding。
- `bench_system.py` 支持 `--batch-sizes` 与 `--output-lengths`；每个 workload/batch/output cell 独立选择首 mode baseline，禁止跨 cell 比 token 或 KV bytes。
- 单请求 case 通过完整 request group replication 构造 offline batch，并在 record 中显式记录；不把复制后的 synthetic batch伪称为 manifest 原生 workload。
- `ModelRunner.capture_cudagraph()` 单独测量 capture time；当前 scope 是 `decode_model_forward`，prefill、compute logits 和 sampler 都在 Graph 外。
- replay NVTX 同时记录 actual/captured batch；`cudagraph_metadata()` 对 eager 返回无 graph state，对 Graph 返回 selected bucket 和 padding。

focused contract tests:

```bash
PYTHONPATH=/data/Prism-Infer \
.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py \
  tests/test_model_runner_vl_cudagraph.py -s
```

当前输出: `25 passed in 4.17s`。覆盖 schema v1 backward compatibility、v2 graph/replication guards、matrix axis/cell comparison、batch=3 -> bucket=4 metadata 和 eager no-graph state。

RTX 5090 execution matrix:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model <Qwen3-VL snapshot> --case single_image_448 \
  --modes off_eager,off_graph \
  --batch-sizes 1,2,4,8 --output-lengths 8,32,128 \
  --warmup 2 --repeat 5 \
  --max-model-len 1024 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --output data/p6_system/execution_matrix_20260711.jsonl
```

24 records/12 cells 通过 schema v2；每条 repeat-stable，每个 Graph record 对 cell 内 eager token exact，output token count 均为 `batch * max_tokens`。decode median 范围:

| Batch | Eager across output 8/32/128 | Graph across output 8/32/128 | Speedup range | Graph capture median |
|---:|---:|---:|---:|---:|
| 1 | 30.704..30.746 ms | 17.460..17.623 ms | 1.744x..1.760x | 251.950 ms |
| 2 | 31.636..31.682 ms | 17.688..17.834 ms | 1.777x..1.791x | 563.718 ms |
| 4 | 31.651..31.668 ms | 18.207..18.349 ms | 1.726x..1.738x | 926.789 ms |
| 8 | 31.779..31.856 ms | 18.820..18.956 ms | 1.681x..1.691x | 1317.755 ms |

output=32 的 Graph decode throughput median 随 batch `1/2/4/8` 为 `57.246/113.001/219.552/424.628 tok/s`。当前 capture 为 max batch 录制 `1/2/4/8` 等 buckets，因此 capture time 随 bucket 数增加；它不包含模型加载，但包含每个 bucket 的 warmup/capture/synchronize。

batch8/output32 Nsight mechanism evidence:

| Metric per decode step | Eager | CUDA Graph |
|---|---:|---:|
| explicit kernels | 2077 | 13 |
| graph launch APIs | 0 | 1 |
| graph execution | 0 | 14.818 ms |
| kernel busy outside graph nodes | 17.000 ms | 4.095 ms |
| async memcpy | 9 | 12 |
| stream sync | 2 | 2 |

两条路径 prefill 均为 `3617` 个 kernels，进一步证明当前 Graph 只覆盖 decode model forward。Graph 的 `kernel busy outside graph nodes` 不能与 eager total GPU work直接比较，Graph 内 node time由 `graph execution` 单列。

额外真实 mixed text/image/video batch=3 correctness:

```text
eager: [[11,358],[785,1378],[785,2766]]
graph: [[11,358],[785,1378],[785,2766]]
selected bucket: 4
PASS, 1 passed in 40.24s
```

本轮综合 focused regression: `65 passed in 7.82s`；`compileall prism_infer tests benchmarks scripts` 和 `git diff --check` PASS。

限制:

- 12-cell performance matrix 是 replicated synthetic single-image offline closed-loop，不是 online serving，也不覆盖 text/video/mixed 性能。
- power-of-two matrix 没有 batch padding；batch=3 -> bucket=4 只完成 correctness，尚未形成 padding performance matrix。
- 数据仍为 `git_dirty=true`；RTX 5090 NCU counter 权限未开放，不能报告 SM utilization/occupancy。
- 本阶段只固定 CUDA Graph baseline，不证明 `torch.compile` 有收益。

Raw evidence:

```text
data/p6_system/execution_matrix_20260711.jsonl
data/p6_system/execution_eager_b8_o32_profile_20260711.nsys-rep
data/p6_system/execution_graph_b8_o32_profile_20260711.nsys-rep
data/p6_system/execution_eager_b8_o32_nsys_summary_20260711.json
data/p6_system/execution_graph_b8_o32_nsys_summary_20260711.json
```

#### P6.3-B Torch Compile Preflight

内部 ablation 至少包含:

```text
off + eager
off + CUDA Graph
off + torch.compile region
```

对每个 compile region 必须输出:

- region 边界: vision、prefill 或 decode。
- graph break 数量和位置。
- recompile 次数与触发 shape。
- compile/cold-start time。
- steady-state latency、TTFT 或 TPOT。
- eager/compiled output shape、max diff、mean/std 或 token equality。

PASS 标准:

- `torch.compile` 结果不改变 off baseline correctness。
- CUDA Graph 和 compile 对比保持相同 attention backend、KV layout、batch 和输入。
- 不把 compile time 隐藏在 steady-state benchmark，也不只报告 warm cache 数字。
- 如果 dynamic batch 导致 recompile，必须报告，不能只选择单一 shape 隐藏风险。

当前 RTX 5090 结果（2026-07-11，`compression=off`，dirty worktree）:

- 新增 `compile_preflight` schema、validator 和 runner；system benchmark schema 升为 v3，并继续接受 v1/v2 records。focused compile/profile/schema/dispatch tests 均已覆盖。
- 默认关闭的 `profile_region()` 原先导致 decoder layer `6 graphs/5 breaks/18 compile events`；编译捕获时改用标准 `nullcontext` 后为 `1 graph/0 break/3 compile events`，3 次 event 对应静态 batch `1/2/4`，重复 batch1 复用已有 graph。
- 完整 language-model decode 为 `1 graph/0 break`、`767 ops`、`2991 FX nodes`，但 default 和 eager-cast 模式在 batch1/batch4 的 cold compile 都在 RTX 5090 32GB 上 OOM，不能形成 steady benchmark。
- 完整 VisionEncoder 为 `7 graphs/6 breaks`，break 来自 grid data-dependent geometry、`Tensor.item()` 和动态 `repeat_interleave`。拆分 geometry preparation 与 blocks/mergers tensor region 后为 `1 graph/0 break`，但 default/emulate-casts 的 27 层主输出 max diff 分别为 `0.859375/0.515625`，均失败。
- decoder 子区域中，`emulate_precision_casts=True` 后 RMSNorm 和 MLP 可 exact；RMSNorm median `0.0499 ms` 对 eager `0.0769 ms`，MLP `0.2577 ms` 对 eager `0.2216 ms`，因此 MLP 不占优。attention 单步可 exact，batch4 median `0.2142 ms` 对 eager `0.5549 ms`。
- attention-only system matrix 在 batch `1/2/4/8` 上 steady decode 相对 eager 为 `1.43x..1.46x`，但仍比 CUDA Graph 慢 `1.20x..1.27x`。更严格的 `emulate_precision_casts + force_same_precision` 仍让 batch2/8 所有行在 token 28 分叉，因此该 mode 不通过长输出 correctness，只允许通过 `allow_unsafe_decode_compile=True` 复现 benchmark。
- 最终 execution backend 结论：保留 eager reference 和 CUDA Graph supported path；不接入 full decode、Vision 或 attention-only `torch.compile` 为支持后端，不启动 megakernel。

代表性三方 output32 matrix:

| Batch | Eager decode | CUDA Graph | Compile attention | Compile cold first decode | Compile token exact |
|---:|---:|---:|---:|---:|---:|
| 1 | 31.236 ms | 17.458 ms | 21.345 ms | 1810.650 ms | PASS |
| 2 | 32.278 ms | 17.687 ms | 22.224 ms | 2162.114 ms | FAIL@28 |
| 4 | 32.174 ms | 18.213 ms | 22.183 ms | 2212.579 ms | PASS |
| 8 | 32.422 ms | 18.812 ms | 22.318 ms | 1693.812 ms | FAIL@28 |

Raw evidence:

```text
data/p6_system/compile_preflight_*_20260711.json
data/p6_system/execution_compile_matrix_final_20260711.jsonl
data/p6_system/compile_attention_strict_precision_b2_b8_20260711.jsonl
```

### P6.4 Visual KV Physical Compaction 验证

correctness matrix 至少覆盖:

- text-only: compaction no-op。
- single-image: keep-all 和 keep-ratio `<1.0`。
- multi-image/video: visual spans 跨多个 physical blocks。
- mixed batch: 不同请求具有不同 logical/physical lengths。
- decode append: 至少跨越一个 compacted block boundary。
- Sequence prefill/decode pickle。
- block free-list/hash cleanup、CoW、swap-out/swap-in。
- M-RoPE query positions 使用 logical length。

每个 test 必须输出:

- logical context length。
- physical KV length。
- original/retained visual token count。
- old/new block table 和释放 block 数。
- K/V compact 前后 shape。
- 与 independent retained-KV reference 的 max diff、mean/std。
- keep-all 与 off 的 exact equality。

PASS 标准:

- keep-all physical path 与 off exact match。
- keep-ratio `<1.0` 时真实减少 physical KV token/block/bytes，不能只改变 attention view。
- generated token position ids 与未压缩逻辑序列一致。
- generated KV append 到 physical tail，不覆盖 retained prompt KV。
- 旧 blocks 回收后不再出现在 hash、GPU/CPU block table 或 active mapping。
- unsupported prefix/chunked/swap 组合必须显式失败，不能 silent fallback。

当前 RTX 5090 结果（2026-07-11，dirty worktree）：

- `test_kv_physical_layout.py` 与 KV/compression/schema 组合回归为 `64 passed`。独立 compact K/V reference shape `[2,2,6,1,2]`，max diff `0.000000e+00`。
- compact swap-out/pickle/swap-in 保持 logical/physical length `10/6`，换入 GPU 页均为 `hash=-1` 且不进入 prefix hash index。
- compact decode 使用 logical M-RoPE position `[13,13,13]`、physical context length `7` 和 physical write slot `[6]`。
- mixed text/image/video 中 text 为 dense no-op；image/video physical prompt 分别为 `210 -> 112`、`422 -> 226`，总 active blocks `4 -> 3`，2-token exact。
- multi-image keep=0.5、output8、warmup/repeat `2/5`：physical prompt `408 -> 212`、active blocks `2 -> 1`、occupied bytes ratio `0.5`；decode median `32.204 -> 32.231 ms`；前 6 token exact，第 7 token 分叉。
- keep-all single-image 8-token 与 off exact；该 case 只有一页，因此不产生 page reduction。

Raw evidence:

```text
data/p6_system/visual_compact_keep_all_smoke_20260711.jsonl
data/p6_system/visual_compact_mixed_smoke_20260711.jsonl
data/p6_system/visual_compact_multi_image_formal_dirty_20260711.jsonl
```

### P6.5 Compressed/FP8 Paged Attention 验证

最小 kernel matrix:

```text
batch: 1,2,4,8
physical context: 128,256,1024,4096
num_heads/num_kv_heads: Qwen GQA shape + small focused shape
dtype: bf16 compact + fp8 compact/full
```

每个 case 必须输出 q/cache/block table/context shape、output/reference mean/std、max diff、mean diff、kernel/backend name 和明确 PASS/FAIL。

PASS 标准:

- BF16 跨实现 max diff `<1e-2`。
- FP8 reference 明确包含相同 quantize/dequant 语义；不能拿未量化 BF16 reference 要求 same-precision exact。
- unsupported dtype/shape 显式失败，不能 fallback 后报告 kernel PASS。
- benchmark 只在 correctness PASS 后运行。
- 至少一个已 profiling 的目标 case 相对 P6 baseline 有可测改善；不要求所有 shape 胜出。

当前 RTX 5090 结果（2026-07-11，commit `f4bf51a`，dirty worktree）：

- Qwen GQA `heads/kv_heads/head_dim=32/8/128`，batch `1/2/4/8` × context `128/256/1024/4096`，BF16/FP8 共 32 cases 全部 PASS。
- 所有 case output shape 与 q 一致；max diff `<=0.00390625`，mean diff `<1e-3`。FP8 reference 使用相同 E4M3FN quantize 后转 BF16 的语义。
- engine-level FP8 paged dispatch shape `q=[2,8,128]`、cache `[6,16,2,128]`，max/mean diff `0.00390625/2.76e-4`。
- warmup/repeat `10/50` micro matrix：FP8 batch8/context4096 kernel/reference median `0.2602/1.8029 ms`；BF16 batch8/context4096 为 `0.4527/1.6259 ms`。
- 反例：BF16 batch1/context4096 kernel/reference median `0.2701/0.2077 ms`，kernel 不在所有 shape 占优。
- full-engine single-image output32、warmup/repeat `2/5`：off/FP8 decode median `32.065/31.960 ms`，FP8 ratio `0.997x`；32-token exact，KV pool bytes `603,979,776 -> 301,989,888`。

Raw evidence:

```text
data/p6_system/paged_decode_bf16_fp8_matrix_20260711.txt
data/p6_system/fp8_paged_system_b1_o32_20260711.jsonl
```

### P6.6 质量、显存和容量验证

最小模式:

```text
off
visual logical prune
visual physical compact
fp8 kv
visual physical compact + fp8
```

最小 keep-ratio matrix:

```text
0.25, 0.5, 0.75, 1.0
```

最小 workload:

- text、single-image、multi-image、video、mixed batch。
- synthetic deterministic smoke 与固定真实样例分开报告。
- output length 至少覆盖 8、32、128；未完成项明确标记。

每个 compression-on mode 必须报告:

- logical/physical visual tokens、KV bytes ratio 和 block reduction。
- exact token match/稳定前缀、teacher-forced logits 或 ppl。
- TTFT、TPOT、throughput、allocated/reserved/peak。
- 固定显存下 max concurrency 或 OOM boundary。

P6 目标值 `KV bytes >=40%`、`accuracy drop <1%`、`TPOT <=1.05x off`、`max concurrency >=30%` 只是研究目标。未达到时必须收缩 claim，不能修改输入条件后继续写作达标。

当前 RTX 5090 结果（2026-07-11，commit `f4bf51a`，dirty worktree）：

- runner 支持 keep ratio `0.25/0.5/0.75/1.0` 和组合模式 `visual_compact_fp8`；CUDA FP8 compaction 对 independent retained K/V reference max diff `0.000000e+00`。
- synthetic matrix 覆盖 single-image、multi-image、video、mixed batch，各 `5 modes x 4 ratios x output 8/32/128`，共 240 records；所有 record schema v4 PASS 且 repeat 内 deterministic。
- 自动 Pareto 汇总仅选择 `off_eager/visual_compact/fp8_kv/visual_compact_fp8`，从 192 条记录生成 192 行。multi-image keep=0.5/output128 的组合模式 physical prompt `408 -> 212`、blocks `2 -> 1`、active bytes `0.25x`、TPOT `0.996x`、stable prefix `27/128`。
- video keep=0.5/output128 的组合模式 physical prompt `422 -> 226`、active bytes `0.25x`、TPOT `1.002x`、stable prefix `14/128`。
- mixed keep=0.5/output128 的组合模式 physical prompt `638 -> 344`、blocks `4 -> 3`、active bytes `0.375x`、TPOT `1.004x`、per-request stable prefix `[7,28,14]`。
- 固定真实样例为 COCO val2017 `000000039769.jpg`，source URL `http://images.cocodataset.org/val2017/000000039769.jpg`，SHA256 `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`，原图 `640x480 RGB`。manifest 加载时强制校验文件、摘要和尺寸。
- 真实样例 4 modes x 4 ratios x output32、warmup/repeat `1/3` 共 16 records，全部 schema PASS 和 repeat-stable。compact keep=`0.25/0.5/0.75/1.0` stable prefix 为 `3/3/7/32`；FP8 与组合模式 keep=1 stable prefix 都为 `3/32`。
- 32GB auto-pool capacity 使用 600 个 multi-image requests、output2、prefix cache off。off/compact/FP8/combo KV blocks 为 `249/249/499/499`，peak running 为 `124/248/249/498`，相对 off 为 `1.0x/2.0x/2.008x/4.016x`；四模式均完成 600 请求、无 swap。
- 同一容量实验 elapsed 为 `91.323/83.510/95.602/111.411 s`，组合模式虽提高并发容量但整批更慢。capacity benchmark 必须一 mode 一进程；同一 CUDA context 连续重建 near-capacity pool 曾触发一次 illegal memory access。

P6.6 判定：

- `KV bytes >=40%`：PASS，组合模式代表点 active bytes 降低 `62.5%-75%`。
- `TPOT <=1.05x off`：上述代表点 PASS，但只是当前 workload/dirty commit 的内部证据。
- `max concurrency >=30%`：PASS，observed peak running 最低提升 `2.0x`，组合为 `4.016x`。
- `accuracy drop <1%`：FAIL/未被证明。uniform pruning 的 stable prefix 明显不足，单个真实图片和 greedy token prefix 也不是 accuracy benchmark；不得把偶然 token exact 或 FP8 token flip 写成质量改善。

Focused verification：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py tests/test_pareto_summary.py -s
# 31 passed
```

Raw evidence：

```text
data/p6_system/pareto_single_image_20260711.jsonl
data/p6_system/pareto_multi_image_20260711.jsonl
data/p6_system/pareto_video_20260711.jsonl
data/p6_system/pareto_mixed_20260711.jsonl
data/p6_system/pareto_real_coco_20260711.jsonl
data/p6_system/pareto_synthetic_summary_20260711.json
data/p6_system/pareto_real_coco_summary_20260711.json
data/p6_system/capacity_off_20260711.json
data/p6_system/capacity_visual_compact_20260711.json
data/p6_system/capacity_fp8_20260711.json
data/p6_system/capacity_visual_compact_fp8_20260711.json
```

### P6.7 外部框架对比验证

每个 external baseline 必须记录:

- repo URL、version/commit、dirty state。
- 安装环境和 attention backend。
- 完整启动命令或 offline API 配置。
- model/processor snapshot、dtype、TP、max model len、显存利用率。
- prefix cache、chunked prefill、CUDA Graph或等价设置。
- 相同 workload manifest、sampling、EOS 和 output length。
- preprocessing included/excluded 口径。

PASS 标准:

- Prism off 和 compression-on 都参与，不能只展示最优 Prism mode。
- external framework失败/OOM 时保留命令和错误，不自行降低其资源后继续比较。
- offline 结果不能写成 online serving 吞吐；没有同等 server/request arrival 时明确限制。
- 可以报告 Prism 劣势；P6 完成不要求全面超过 vLLM/SGLang/vLLM-Omni。
- pruning 外部 PR 必须先固定链接/commit，并确认 pruning发生阶段后才可进入对比。

当前 RTX 5090 结果（2026-07-11，Prism commit `f4bf51a` dirty）：

- vLLM 环境：`vllm==0.24.0`，build commit `ee0da84ab`，Torch `2.11.0+cu130`，Transformers `5.13.0`。固定 `FLASH_ATTN`、eager、block size 256、KV pool `603,979,776` bytes、prefix cache off、MM processor cache 0。
- vLLM FlashInfer sampler 因 Blackwell capability probe 报 `FlashInfer requires GPUs with sm75 or higher`；按 vLLM 源码提供的 `VLLM_USE_FLASHINFER_SAMPLER=0` 切换 PyTorch native sampler，并记录在每条 result。
- SGLang 环境：tag `v0.5.15`，commit `f63458b5beaceabbd9d749b9fc956370e1b649e6`，Torch `2.11.0+cu130`，Transformers `5.12.1`，独立 dependency overlay，不修改 vLLM-Omni 环境。
- SGLang FA3 vision 在源码中显式拒绝 Blackwell；FA4 `4.0.0b15` 在当前 CUTLASS DSL 产生 MLIR layout compile error。最终可执行 baseline 固定 text `triton`、vision `triton_attn`、eager、`max_total_tokens=4096`、radix cache off、chunked prefill off。
- vLLM-Omni repo 为 clean commit `73bafd64e363cf3d4b114f3f9a1ef89eef73da6d`，依赖 vLLM `0.24.0`。标准 Qwen3-VL autoregressive model 注册位于该 vLLM dependency；vLLM-Omni 本身的额外 commit 是 MagiHuman FP8，不重复包装同一路径生成伪独立 baseline。
- image/multi-image/COCO 的 Prism/external prompt tokens 分别严格相同：`210/408/316`。video/mixed 在 vLLM 中为 `420/636`，Prism 为 `422/638`，自动标记 `performance_comparable=false`，不用于 ratio claim。

可比 eager output32、warmup/repeat `1/3`：

| Framework | Case | External/Prism TPOT | External/Prism E2E throughput | Stable prefix vs Prism off |
|---|---|---:|---:|---:|
| vLLM 0.24.0 | single-image | `0.492x` | `1.900x` | `28/32` |
| vLLM 0.24.0 | multi-image | `0.484x` | `1.937x` | `32/32` exact |
| vLLM 0.24.0 | COCO | `0.487x` | `1.847x` | `7/32` |
| SGLang 0.5.15 Triton | single-image | `0.432x` | `2.264x` | `2/32` |
| SGLang 0.5.15 Triton | multi-image | `0.413x` | `2.430x` | `32/32` exact |
| SGLang 0.5.15 Triton | COCO | `0.435x` | `2.207x` | `7/32` |

Prism `visual_compact_fp8 keep=0.5` 也参与同一汇总：vLLM/SGLang TPOT ratio 分别约 `0.485-0.493x` / `0.414-0.438x`，外部 E2E throughput 为 `1.85-2.45x`。组合模式不能弥补 Prism eager framework overhead，且 P6.6 已证明其 uniform pruning 质量不达标。

显存口径：vLLM in-process 与 Prism 都使用 torch allocator，可以比较；vLLM peak allocated 为 `17,719.8-17,760.7 MiB`，Prism off 为 `19,701.4-19,707.7 MiB`。SGLang 多进程只能得到 NVML process-used `19,244-19,318 MiB`，汇总器将 memory ratio 置为不可比。

Focused verification：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py tests/test_pareto_summary.py \
  tests/test_external_comparison.py -s
# 33 passed
```

Raw evidence：

```text
data/p6_external/external_vs_prism_summary_20260711.json
data/p6_external/external_vs_prism_compact_fp8_summary_20260711.json
data/p6_external/vllm_0.24.0_*_eager_fixed_pool_20260711.json
data/p6_external/sglang_0.5.15_*_eager_20260711.json
data/p6_external/*stderr.txt
```

### P6.8 两卡 TP 验证

PASS 标准:

- 同一 prompt 的 1 GPU/2 GPU greedy token ids 一致。
- logits/hidden 或等价 tensor 输出 shape、max diff、mean/std。
- 输出权重 shard、KV head shard 和 collective 类型证据。
- 记录每卡 allocated/reserved/peak、latency、TTFT/TPOT。
- 小 batch 因通信变慢时如实记录，不把显存下降外推为吞吐提升。

当前静态审计（2026-07-11）：

- `nvidia-smi -L` 仅返回 `GPU 0: NVIDIA GeForce RTX 5090`，无第二张 GPU，因此没有 1GPU/2GPU token、logits、NCCL latency 或 per-GPU memory 实测。
- `ColumnParallelLinear/QKVParallelLinear/RowParallelLinear` 分别按 output/QKV/input 维切权重；row parallel 与 vocab embedding 使用 `dist.all_reduce`，LM head 使用 rank0 `dist.gather`。Qwen3-VL 8B 的 Q heads 32、KV heads 8、hidden/intermediate/vocab 对 TP2 可整除。
- Vision Encoder 当前不是 TP shard：每个 rank 构造并加载完整 VisionEncoder。即使 text TP 可运行，vision 权重/计算也不会随 TP2 减半。
- `ModelRunner` 已用每 worker 独立单向 `multiprocessing.Pipe` 替换固定 `2**20` bytes shared memory。协议使用 `send_bytes/recv_bytes` 保留消息边界，rank0 只持有发送端，worker 只持有接收端。
- 代表性 `[784, 1536]` FP32 `pixel_values` 消息序列化后为 `4,817,396` bytes；向两个接收端广播后 method/args/tensor 逐元素一致，超过旧 1 MiB 上限。小消息连续复用、损坏 pickle、不可序列化参数和断开通道均有显式门禁。
- `validate_tensor_parallel_environment()` 在 spawn/NCCL 前检查 visible devices 和模型分片维度；variable-size IPC 解除 VL payload 的实现阻断，但不替代真实两卡 collective/correctness 验证。

Focused verification：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q tests/test_tensor_parallel_preflight.py -s
# 5 passed
```

P6.8 判定：IPC 子门禁 PASS，两卡动态门禁仍 BLOCKED/未通过。以下 TP1/TP2 greedy smoke 入口已创建，但当前单卡机器只能得到 skip；需要在两卡平台显式启用：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 PRISM_RUN_TP2=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q tests/test_llm_vl_tp2.py -s
```

当前 `tests/test_llm_vl_tp2.py` 只覆盖同一单图 prompt 的 8-token greedy exact。即使该 smoke PASS，仍需补 logits/hidden 数值、shard/collective 证据、每卡显存、TTFT/TPOT 后，P6.8 才能整体 PASS。

### P6.9 Megakernel 可选验证

只有满足以下条件才建立 PASS/FAIL:

- P6.2 profiler 证明目标 decode workload 主要受 launch/host gap 限制。
- 有真实可运行 megakernel 或项目内明确的 persistent kernel scope。
- 相同模型区域、输入、dtype、KV layout 和输出语义可对比。

对比矩阵:

```text
discrete eager kernels
discrete kernels + CUDA Graph
torch.compile region
megakernel/persistent kernel
```

必须输出 kernel count、CPU launch gap、GPU time、TPOT、memory 和 correctness。普通 fusion、CUDA Graph 或单个 paged attention kernel不能仅因规模大而命名为 megakernel。

当前门禁判定（2026-07-11）：NOT STARTED BY DESIGN。

- P6.2 Nsight 已证明 eager decode kernel dispatch 多，P6.3 CUDA Graph 也已形成 `1.68x-1.79x` 强 baseline；但 RTX 5090 hardware counter/SM utilization 采集失败，不能证明目标 region 主要受 launch 而非 memory/compute 限制。
- 仓库中没有真实 persistent/megakernel implementation，外部也未固定可运行且语义相同的实现。
- 因此不创建虚假的 megakernel mode，不把 P6.5 paged attention、CUDA Graph 或 attention-only compile 重命名。待 RTX PRO 6000 补齐 counter 且有真实实现后再重新打开本节。

### P6.10 阶段 Review

最终命令（2026-07-11）：

```bash
cd /data/Prism-Infer
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
.venv-local/bin/python -m pytest -q
```

最终结果：

```text
Running 195 items in this shard
195 passed, 5 skipped in 245.50s (0:04:05)
```

回归修复记录：

- 第一次全量为 `184 passed, 5 skipped, 11 failed`：9 个 engine failures 来自新增 TP preflight 漏 `import torch`；另 2 个为 Dynamo op count 顺序污染。
- 第二次为 `193 passed, 5 skipped, 2 failed`：engine 回归已清零；剩余 Dynamo tests 在此前 GPU tests启用 default-device mode 后得到 `op_count=1`。
- 独立 probe 证明初始 mode 为 `op_count=2`，`cuda -> cpu` 恢复后为 1，临时 `set_default_device(None)` 后恢复为 2。最终测试在 explain 期间隔离该 Torch 2.6 global mode，仍严格要求 graph 1、break 0、op count 2。
- 第三次全量即上述 `195 passed`。首次、二次原始日志未覆盖删除。

variable-size TP IPC 合入工作区后的 post-change 全量复跑：

```text
Running 198 items in this shard
197 passed, 6 skipped in 267.13s (0:04:27)
```

新增 skip 来自 `tests/test_llm_vl_tp2.py` 的显式两卡门禁；当前 `PRISM_RUN_TP2` 未启用且只有一张 GPU。该结果验证单卡 P1-P6 路径无已知回归，不构成 TP2 动态 PASS。

其他最终检查：

```bash
.venv-local/bin/python -m compileall -q prism_infer benchmarks scripts tests
git diff --check
```

两项 PASS。Raw logs：

```text
data/p6_system/p6_full_regression_20260711.txt
data/p6_system/p6_full_regression_rerun_20260711.txt
data/p6_system/p6_full_regression_final_20260711.txt
data/p6_system/p6_full_regression_variable_ipc_20260711.txt
```

P6 Review 判定：可执行 correctness/engineering review PASS，但阶段仍有以下未通过或外部阻断，不能写成完整性能目标 PASS：

- P6.6 uniform pruning accuracy target FAIL。
- P6.7 Prism eager TPOT/throughput 明显落后固定 vLLM/SGLang baseline。
- P6.8 两卡 TP BLOCKED（variable-size VL IPC 已完成，但当前只有 single GPU，动态矩阵未运行）。
- P6.2-B RTX hardware counters 未采集。
- 所有当前 performance records 为 `git_dirty=true`，clean-commit formal rerun 未执行。

### P6.11 Compressed KV CUDA Graph

实现门禁：

- `off`、`fp8_kv`、`visual_compact`、`visual_compact_fp8` 允许 CUDA Graph decode。
- physical compression 的 KV dtype 在 capture 时绑定；physical context、页表和 append slot 由 replay 前更新的 `context_lens/block_tables/slot_mapping` tensor 表达。
- logical `visual_prune` 依赖动态 retained-slot gather，`Config(compression_mode="visual_prune", enforce_eager=False)` 必须显式失败，不能 silent eager fallback。
- benchmark 必须比较同一 compression/attention backend 的 eager 与 Graph pair，不能把 BF16/FP8 或 compact/dense 差异归因给 Graph。

Focused verification（2026-07-14）：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_compression_off.py \
  tests/test_compile_execution_config.py \
  tests/test_benchmark_schema.py -s
# 55 passed in 4.00s

.venv-local/bin/python -m compileall -q prism_infer benchmarks scripts tests
git diff --check
# PASS
```

真实模型 correctness 覆盖：

- `visual_compact`、`fp8_kv`、`visual_compact_fp8` 的 single-image output8 eager/Graph token exact，physical tokens、active blocks/bytes 与 KV dtype 一致。
- combo 覆盖 single-image、multi-image、video、mixed text/image/video；mixed batch=3 使用 Graph bucket4，无 padding row 污染。
- combo output128 eager/Graph 128-token exact，physical tokens 与 active bytes exact。
- FP8 与 BF16 在既有长输出上可以产生不同 token；P6.11 只要求同一种 compression 的 eager/Graph exact，禁止跨 compression 归因。

clean commit `9e30e55` single-image output32、warmup/repeat `2/5`：

| Compression | Eager decode median | Graph decode median | Speedup | Eager/Graph token |
|---|---:|---:|---:|---|
| `visual_compact` | `32.3903 ms` | `17.6382 ms` | `1.8364x` | SHA256 exact |
| `fp8_kv` | `32.4378 ms` | `17.6575 ms` | `1.8371x` | SHA256 exact |
| `visual_compact_fp8` | `32.4459 ms` | `17.5057 ms` | `1.8535x` | SHA256 exact |

clean commit `9e30e55` combo output32、warmup/repeat `2/5`：

| Batch | Eager decode median | Graph decode median | Speedup | Graph decode throughput |
|---:|---:|---:|---:|---:|
| 1 | `34.0119 ms` | `17.5069 ms` | `1.9428x` | `57.1120 tok/s` |
| 2 | `33.6158 ms` | `17.7515 ms` | `1.8937x` | `112.6239 tok/s` |
| 4 | `33.5902 ms` | `18.3016 ms` | `1.8354x` | `218.4745 tok/s` |
| 8 | `33.6542 ms` | `19.1519 ms` | `1.7572x` | `417.3168 tok/s` |

这里的 throughput 是 replicated single-image request 的 offline decode throughput，不是 online serving benchmark。所有记录都包含 `torch.cuda.synchronize()` timing boundary、median/p90/min/max、allocated/reserved/peak memory、input/config 和 KV physical metrics。

Post-change full regression（2026-07-14）：

```bash
cd /data/Prism-Infer
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
.venv-local/bin/python -m pytest -q tests -s \
  | tee data/p6_system/p611_full_regression_20260714.txt
```

```text
Running 209 items in this shard
208 passed, 6 skipped in 267.79s (0:04:27)
```

Raw evidence：

```text
data/p6_system/p611_combo_graph_smoke_20260714.jsonl
data/p6_system/p611_combo_graph_multi_image_20260714.jsonl
data/p6_system/p611_combo_graph_video_20260714.jsonl
data/p6_system/p611_combo_graph_mixed_20260714.jsonl
data/p6_system/p611_physical_graph_batch1_output32_20260714.jsonl
data/p6_system/p611_combo_graph_batch_matrix_output32_20260714.jsonl
data/p6_system/p611_combo_graph_output128_20260714.jsonl
data/p6_system/p611_full_regression_20260714.txt
data/p6_system/p611_clean_physical_graph_batch1_output32_20260714.jsonl
data/p6_system/p611_clean_combo_graph_batch_matrix_output32_20260714.jsonl
```

P6.11 correctness/engineering 判定为 PASS。最初 benchmark 记录 commit `ac6e01d`、`git_dirty=true`，只作为 validation evidence；关键 batch1 mode pairs 与 combo batch1-8 已在 commit `9e30e55`、`git_dirty=false` 上 formal rerun。该执行优化不改变 P6.6 uniform pruning quality FAIL，不产生新的 vLLM/SGLang external comparison claim。

### P6.12-A Runtime Attention Visual Pruning

实现门禁：

- `visual_pruning_strategy="attention"` 必须在真实 prefill q/k 上生成 score，禁止回退 uniform 或使用离线伪数据。
- 默认聚合最后 4 个 decoder layers；每层使用当前序列最后 query 对完整 causal context 的 attention probability，并对 local Q heads 求均值。
- q/k score tensor 保留在 device，完整 prefill 后一次性 materialize；TP 下每个 rank 的 head mean 必须 all-reduce 后再生成一致 decision。
- decision 必须在 physical compaction 前写回 `Sequence`，并记录 score source/layers/min/max/mean。
- score 缺层、重复层、GQA heads/dim、flatten token count 或 visual token score 不完整时显式失败。

Focused verification（2026-07-14）：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_visual_pruning.py \
  tests/test_compression_off.py \
  tests/test_visual_pruning_active.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_model_runner_context_reset.py \
  tests/test_compile_execution_config.py \
  tests/test_benchmark_schema.py -s
# 79 passed in 7.31s
```

Independent reference 输出：

```text
q shape: [8, 4, 2]
k shape: [8, 2, 2]
score shape: [5]
actual mean/std: 1.203456e-01 / 2.894921e-02
reference mean/std: 1.203456e-01 / 2.894921e-02
max diff: 0.000000e+00
PASS
```

真实模型 smoke：

- single-image BF16 compact eager/Graph output8 SHA256 exact，score layers `[32,33,34,35]`，visual tokens `196 -> 98`，physical prompt `210 -> 112`。
- single-image compact FP8 eager/Graph output8 SHA256 exact，KV dtype `torch.float8_e4m3fn`，physical prompt `112`。
- mixed text/image/video batch=3 Graph：text row `6 -> 6` dense；image/video `210 -> 112`、`422 -> 226`；24 output tokens repeat-stable。

keep=0.5 quality preflight：

| Workload | Output | Uniform stable prefix | Attention stable prefix | Logical -> physical prompt |
|---|---:|---:|---:|---:|
| COCO `000000039769` | 32 | `3` | `21` | `316 -> 166` |
| multi-image `2x448` | 128 | `6` | `7` | `408 -> 212` |
| video `4x448` | 128 | `14` | `14` | `422 -> 226` |

该表只证明一个真实图片样例有明显改善，不能证明 dataset accuracy 或整体 quality PASS。multi-image/video 仍早期分叉，因此 P6.12 quality gate 继续 FAIL。

Rejected ablation：coverage-aware Python MMR，weight `0.25`。

- stable prefix 为 COCO/multi/video `7/6/14`，分别不如纯 attention 的 `21/7/14`。
- greedy Python selection 使观察到的 prefill 增至约 `236-390 ms`。
- 候选实现已删除；raw records只用于解释拒绝原因。

Post-change full regression：

```bash
cd /data/Prism-Infer
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
.venv-local/bin/python -m pytest -q tests -s \
  | tee data/p6_system/p612_full_regression_20260714.txt
```

```text
Running 213 items in this shard
212 passed, 6 skipped in 299.59s (0:04:59)
```

Raw evidence：

```text
data/p6_system/p612_attention_graph_smoke_20260714.jsonl
data/p6_system/p612_attention_combo_graph_smoke_20260714.jsonl
data/p6_system/p612_attention_mixed_graph_smoke_20260714.jsonl
data/p6_system/p612_coco_uniform_quality_20260714.jsonl
data/p6_system/p612_coco_attention_quality_20260714.jsonl
data/p6_system/p612_multi_image_uniform_quality_20260714.jsonl
data/p6_system/p612_multi_image_attention_quality_20260714.jsonl
data/p6_system/p612_video_uniform_quality_20260714.jsonl
data/p6_system/p612_video_attention_quality_20260714.jsonl
data/p6_system/p612_*_attention_mmr025_quality_20260714.jsonl
data/p6_system/p612_full_regression_20260714.txt
data/p6_system/p612_clean_coco_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_coco_attention_quality_20260714.jsonl
data/p6_system/p612_clean_multi_image_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_multi_image_attention_quality_20260714.jsonl
data/p6_system/p612_clean_video_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_video_attention_quality_20260714.jsonl
```

P6.12-A engineering/correctness 判定为 PASS，quality 判定为 FAIL。初始 smoke/quality records 为 commit `39802be` 的 dirty validation；上述 9 条关键 quality records 已在 commit `c07fa34`、`git_dirty=false` 上 formal rerun，stable-prefix 和 physical-token 结论完全复现。当前没有 warmup/repeat `2/5` 的 clean scorer performance matrix，因此仍不做 TTFT overhead claim。

### P6.12-B Per-span Budget Rejected Ablation

调查动机来自 P6.12-A clean attention decision，而不是外部实现。两个双 span workload 的全局 top-k 分配为：

| Workload | Span tokens | Global attention kept | Total kept | Physical prompt |
|---|---:|---:|---:|---:|
| multi-image `2x448` | `196 + 196` | `124 + 72` | 196 | 212 |
| video `4x448` | `196 + 196` | `109 + 87` | 196 | 226 |

为了将该现象纳入 runtime 审计，attention decision record 新增
`kept_visual_tokens_by_span`，每项记录 `modality/span_index/kept_tokens`。

Focused verification（2026-07-15）：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_visual_pruning.py \
  tests/test_compression_off.py \
  tests/test_benchmark_schema.py -s
# 58 passed in 4.11s
```

synthetic runtime attention decision 包含三个 span，审计输出为
`image[0]=2, video[0]=0, image[1]=0`，其总和与 `kept_visual_tokens=2`
一致。这个测试只验证审计字段和 global top-k 现有语义，不声称质量改善。

临时 `attention_span` ablation 使用按 span token capacity 的 largest-remainder
quota，总 keep target 不变，双 span 分别保留 `98/98`。质量预检结果：

| Workload | Output | Global attention prefix | Per-span prefix | Attention exact | Physical prompt |
|---|---:|---:|---:|---|---:|
| COCO `000000039769` | 32 | 21 | 21 | 32 tokens exact | 166 |
| multi-image `2x448` | 128 | 7 | 7 | no | 212 |
| video `4x448` | 128 | 14 | 14 | 128 tokens exact | 226 |

这三条记录使用 greedy、keep ratio `0.5`、last 4 layers、
`warmup=1/repeat=1`。它们只是 quality preflight，不满足稳定性能报告条件。
候选没有改善任何 workload 的 stable prefix，因此 `attention_span` 实现、
config、CLI 和测试已删除，不进入支持策略。

Raw evidence（dirty candidate validation，只用于 rejected ablation）：

```text
data/p6_system/p612b_coco_attention_span_quality_20260715.jsonl
data/p6_system/p612b_multi_image_attention_span_quality_20260715.jsonl
data/p6_system/p612b_video_attention_span_quality_20260715.jsonl
```

P6.12-B 仍为进行中，quality 继续 FAIL。该 rejected-ablation 提交只新增
decision 审计字段，没有改变 scorer、selection、compaction 或 decode
执行路径，所以当时未重跑 full model regression。

### P6.12-B Dataset-level Pruning Fidelity and Reference Task Quality Harness

目标是把 P6.12-A 的单点 prefix 数字升级为可复现的 dataset fidelity 与
reference-task regression gate，同时明确区分“接近未压缩输出”“相对 reference
的词法退化”和“标准 COCO 任务精度”。

实现 contract：

- `prism_infer.analysis.pruning_fidelity` 只读取已通过 benchmark schema 校验的
  JSONL，不执行模型。
- baseline/candidate 必须同 manifest、case、request count、output length、
  model config、execution backend 和 reference identity；不可比字段显式失败。
- schema-v4+ 提供 physical prompt tokens/bytes/layouts，继续支持历史 fidelity
  records；缺少 task evidence 时 gate 显式 `INELIGIBLE`。
- schema-v5 额外要求每请求 decoded text/hash、reference source/task/image identity，
  并强制 `output_decoding_included_in_e2e=false`。文本解码发生在计时结束后。
- candidate 必须覆盖所选 baseline 全部 case；dataset aggregate 按 `max_tokens`
  分离，重复 case replication cell 显式拒绝。
- 多参考 token-F1 使用 multiset overlap，ROUGE-L F1 使用 token LCS；每项指标
  独立取 5 条 caption 中的最高分。两项 candidate macro 相对 off baseline 的
  绝对下降都必须 `<=0.01`。
- 这些指标是无外部依赖的 lexical preflight，不是 COCO 官方 CIDEr/SPICE，也不
  单独证明真实任务 accuracy。

固定输入与 provenance：

- `p6_real_samples.json` 固定 7 张 COCO val2017 图片，按 `4+3` requests 分成
  `coco_fidelity_batch_a/b`；每图绑定 5 条 caption，共 35 条 task references。
- source 声明指向 COCO 官方 annotation package
  `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`。
- 实际 `captions_val2017.json` 固定 mirror revision
  `50967f6f3616db2bf261e42b80377ab8cd8d4214`，内容 SHA256 为
  `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`。
- manifest 中 8 个带 evaluation 的 requests（含既有单图 case）、7 个 unique
  image IDs、40 条引用已逐项与 annotation ID/image ID/caption 文本比对通过；
  本轮两个 task-quality batch 使用其中 7 个 unique requests/35 条 captions。

Focused verification（2026-07-15）：

```bash
cd /data/Prism-Infer
.venv-local/bin/python -m pytest -q \
  tests/test_reference_quality.py \
  tests/test_pruning_fidelity.py \
  tests/test_benchmark_schema.py
# 54 passed in 3.84s
```

受影响扩大回归继续加入 visual pruning、compression off、active compaction、
单/混合 VL prefill、context reset 和 pareto summary：

```text
100 passed in 7.50s
active pruning independent-reference max diff: 0.000000e+00
keep-all off equivalence max diff: 0.000000e+00
mixed text/image/video prefill/decode: PASS
```

新增 guards 覆盖：两项指标独立选择 best reference、规范化后空 reference、
ROUGE-L 单独导致 gate FAIL、output-length cell 分离、重复 replication 拒绝、
decoded-text hash、source/task identity、reference count、schema-v4 compatibility 和 output decoding timing provenance。

汇总 CLI：

```bash
.venv-local/bin/python scripts/summarize_p6_pruning_fidelity.py \
  data/p6_system/p612b_task_quality_batch_a_attention_20260715.jsonl \
  data/p6_system/p612b_task_quality_batch_b_attention_20260715.jsonl \
  data/p6_system/p612b_task_quality_batch_a_uniform_20260715.jsonl \
  data/p6_system/p612b_task_quality_batch_b_uniform_20260715.jsonl \
  --baseline-mode off_graph \
  --max-task-quality-drop 0.01 \
  --json-output data/p6_system/p612b_task_quality_strategy_summary_20260715.json \
  --markdown-output data/p6_system/p612b_task_quality_strategy_summary_20260715.md
```

quality preflight config：RTX 5090，bf16，CUDA Graph，greedy output32，keep
`0.5`，min keep `32`，last 4 layers，warmup/repeat `1/1`，prefix caching
显式关闭。实际开关写入 benchmark model metadata；output decoding 不计入
engine/E2E timing。

| Strategy | Exact requests | Prefix micro | Prefix min | Physical tokens | Active bytes |
|---|---:|---:|---:|---:|---:|
| attention last4 | `3/7` | `0.696` | `0.219` | `0.535x` | `0.538x` |
| uniform | `0/7` | `0.304` | `0.094` | `0.535x` | `0.538x` |

| Strategy | Token-F1 B/C | Drop | ROUGE-L B/C | Drop | Task gate |
|---|---:|---:|---:|---:|:---:|
| attention last4 | `0.321635/0.315285` | `0.006351` | `0.289116/0.276703` | `0.012413` | FAIL |
| uniform | `0.321635/0.315486` | `0.006150` | `0.289116/0.252751` | `0.036365` | FAIL |

task evidence 已完整且 `eligible=true`。attention 的 token-F1 子门禁通过，
但 ROUGE-L drop `0.012413` 超过 `0.010000`，因此整体 FAIL；uniform 也仅因
ROUGE-L drop FAIL。attention 的 ROUGE-L retention 明显高于 uniform，但 uniform
candidate token-F1 略高，不能声称 attention 在所有 task metrics 上支配 uniform。

绝对分数约 `0.3` 的解释受两项条件限制：输出只生成 32 tokens，且 prompt 多要求
detailed description，而 COCO captions 较短。当前 gate 主要用于相对 off baseline
的压缩回归；不发布 CIDEr/SPICE 或 `accuracy drop <1%` claim。

Raw evidence（commit `9e5db53`、`git_dirty=true` validation）：

```text
data/p6_system/p612b_task_quality_batch_a_attention_20260715.jsonl
data/p6_system/p612b_task_quality_batch_b_attention_20260715.jsonl
data/p6_system/p612b_task_quality_batch_a_uniform_20260715.jsonl
data/p6_system/p612b_task_quality_batch_b_uniform_20260715.jsonl
data/p6_system/p612b_task_quality_strategy_summary_20260715.json
data/p6_system/p612b_task_quality_strategy_summary_20260715.md
```

该矩阵不用于稳定性能 claim；提交后仍需 clean rerun 才能升级为 formal
task-quality evidence。P6.12-B 下一步是研究跨 query/layer 聚合、视觉网格
coverage 或动态预算，而不是放宽门禁。

### P6.12-C Final-layer Attention Quality Gate

保持 last-query/global-top-k 与 keep `0.5` 不变，先对 decoder layer 聚合做
last1/last4/last8 单变量消融。last1 在 batch A 通过后，于 clean commit
`a7588d3` 对两个固定 COCO batch 成对重跑 off/last1：

```bash
.venv-local/bin/python benchmarks/bench_system.py \
  --model <model_path> \
  --manifest benchmarks/workloads/p6_real_samples.json \
  --case coco_fidelity_batch_a \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 1 --repeat 1 \
  --disable-prefix-caching \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/p6_system/p612c_clean_task_quality_batch_a_attention_last1_20260716.jsonl
```

batch B 使用相同参数，仅替换 case/output。汇总结果：

| Metric | Off | Last1 | Delta / ratio | Gate |
|---|---:|---:|---:|:---:|
| token-F1 macro | `0.321635` | `0.318347` | `-0.003288` | PASS |
| ROUGE-L macro | `0.289116` | `0.285406` | `-0.003710` | PASS |
| physical prompt tokens | - | - | `0.535x` | PASS |
| active prompt bytes | - | - | `0.538x` | PASS |

baseline/candidate 的 `environment.git_dirty` 均为 `false`；7 个 decisions 均记录
单层 `score_layers=[35]`。exact requests 为 `3/7`，prefix micro/min 为
`0.652/0.094`。因此 task gate PASS，但 token fidelity 并非全面优于 last4，
也不能外推为标准 COCO accuracy。

默认切换 focused regression：

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_visual_pruning.py tests/test_compression_off.py
# 30 passed
```

Raw evidence：

```text
data/p6_system/p612c_clean_task_quality_batch_a_attention_last1_20260716.jsonl
data/p6_system/p612c_clean_task_quality_batch_b_attention_last1_20260716.jsonl
data/p6_system/p612c_clean_attention_last1_quality_summary_20260716.json
data/p6_system/p612c_clean_attention_last1_quality_summary_20260716.md
```

默认切换后的 clean commit `e51c16d` 不传 last-N 参数，两个 COCO batch 均记录
`attention_last_n_layers=1` 与 `score_layers=[35]`，质量汇总完全复现 PASS。

多模态 smoke：

| Case | Output | Stable prefixes | Physical tokens | Active bytes | Zero-kept spans |
|---|---:|---|---:|---:|---:|
| multi_image_2x448 | 128 | `[7]` | `0.520x` | `0.500x` | `0` |
| video_4x448 | 128 | `[14]` | `0.536x` | `0.500x` | `0` |
| mixed_text_image_video | 32 | `[32,28,14]` | `0.539x` | `0.750x` | `0` |

稳定性能矩阵使用 COCO batch A、batch4、output32、CUDA Graph、
`warmup=2/repeat=5`：

| Mode | Prefill | Decode step | Decode tok/s | Engine tok/s | E2E | Physical tokens |
|---|---:|---:|---:|---:|---:|---:|
| off_graph | `221.874 ms` | `18.945 ms` | `211.008` | `158.048` | `993.238 ms` | `988` |
| attention last1 compact Graph | `224.179 ms` | `18.553 ms` | `215.571` | `160.087` | `988.486 ms` | `530` |

判定：last1 prefill ratio `1.010x`，decode-step speedup `1.021x`，engine output
throughput ratio `1.013x`，E2E speedup `1.005x`；active bytes ratio `0.571x`。
显式 last4 的 prefill/decode 为 `223.938/18.544 ms`，与 last1 差异小于
`0.2%`，所以不形成 last1 scorer 加速 claim。

完整回归：

```bash
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p6_system/p612c_full_regression_20260716.xml tests
# 238 passed, 6 skipped in 232.90s
```

新增 raw evidence：

```text
data/p6_system/p612c_default_clean_quality_summary_20260716.json
data/p6_system/p612c_default_clean_multimodal_fidelity_summary_20260716.json
data/p6_system/p612c_default_clean_performance_batch_a_output32_20260716.jsonl
data/p6_system/p612c_clean_performance_batch_a_attention_last4_output32_20260716.jsonl
data/p6_system/p612c_full_regression_20260716.xml
```

### P6 全局 Benchmark 规则

每个 benchmark 必须输出:

- 硬件型号、CUDA、torch、transformers、commit hash。
- 输入 shape、batch、seq len、visual token 数、图像/视频数量、compression config。
- warmup 次数和 repeat 次数。
- `torch.cuda.synchronize()` timing 边界。
- GPU memory allocated/reserved/peak。
- latency median、p90、min、max；TPOT/ITL 还需 p99。
- throughput 或 token/s，以及 physical KV bytes/block count。

禁止:

- 只报 mean。
- 用估算数字代替实测。
- 混用不同输入条件做优化前后对比。
- 在未验证 correctness 的 kernel 上报告性能收益。
- 同时改变 execution、attention、compression 三个维度后把收益归因给单一模块。
- 在不同输入集合、不同采样配置或不同显存限制下声称吞吐超越。
- 在 P6.1 baseline 未完成前开始 physical compaction、`torch.compile` 或 megakernel 性能 claim。

## P7: 交付验证

### P7.0/P7.1 Freeze and Offline External Baseline v2

P7.0 将 P6.12 content-aware BF16 主线冻结在 `c970c61`，annotated tag
`p6.12-content-aware-kv` 已推送。P7.1 benchmark/schema 实现在 clean pushed
commit `b17f933` 上执行；正式 raw records 均记录 `git_dirty=false`。

focused schema 与兼容性回归：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python -m pytest -q \
  tests/test_external_comparison.py tests/test_benchmark_schema.py
# 42 passed in 3.79s
```

两条 profile 的正式汇总验证：

```bash
.venv-local/bin/python scripts/summarize_p7_external.py \
  --comparison-profile diagnostic_matched \
  --prism data/p7_external/prism_*_formal_b17f933.jsonl \
  --external data/p7_external/vllm_*_diagnostic_matched_formal_b17f933.json \
  --prism-modes off_eager visual_compact \
  --prism-keep-ratio 0.5 \
  --json-output /tmp/p7_diagnostic_matched.json \
  --markdown-output /tmp/p7_diagnostic_matched.md

.venv-local/bin/python scripts/summarize_p7_external.py \
  --comparison-profile best_stable \
  --prism data/p7_external/prism_*_formal_b17f933.jsonl \
  --external data/p7_external/vllm_*_best_stable_formal_b17f933.json \
  --prism-modes off_graph visual_compact_graph \
  --prism-keep-ratio 0.5 \
  --json-output /tmp/p7_best_stable.json \
  --markdown-output /tmp/p7_best_stable.md
```

两次命令各比较 10 个 cell。重生成 JSON/Markdown 与保存结果逐字节一致；合计
`20 performance_comparable / 0 non-comparable`。门禁覆盖 model config hash、
GPU UUID、prompt tokens、KV pool、block size、sampling、warmup/repeat、timing
scope、effective execution 和 source/harness clean state；schema-v1 历史输入继续兼容。

完整回归：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p7_external/p71_full_regression_20260716.xml
```

JUnit 结果为 `tests=246`、`failures=0`、`errors=0`、`skipped=6`、
`time=232.301s`，即 `240 passed, 6 skipped`。正式矩阵、汇总、稳定性实验、
semantic CUDA region profile 与 JUnit 均保存在忽略跟踪的
`data/p7_external/`；发布结论见 `PERFORMANCE_REPORT.md` 6.2-6.9，问题定位见
`docs/issues/P7-000` 至 `P7-006`。

### P7.2 Engine Contract Refactor

合同 focused gate：

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_engine_contracts.py \
  tests/test_scheduler_swap_tables.py \
  tests/test_kv_engine_hardening.py \
  tests/test_model_runner_context_reset.py -s
```

必须覆盖：

- Request FSM合法 transition与 terminal不可复活。
- frozen `BatchPlan`的 phase/membership/token budget/KV transfer不可变。
- `SchedulerPolicy` admission/chunk/preemption决策可独立测试。
- executor严格按 plan执行 CoW、swap、model和 compaction。
- metrics observer不驱动 scheduler；request/batch时间字段可复算。
- admission reject、cancel和 swapped CPU KV page回收。
- engine exit释放 executor持有的 runner引用，同进程下一模型加载不残留显存。

clean full regression：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p7_engine/p72_full_regression_8b27edc.xml
```

JUnit：`tests=255`、`failures=0`、`errors=0`、`skipped=6`、
`time=239.200s`，即 `249 passed, 6 skipped`。架构与 P7.3 online合同见
`docs/P7_ENGINE_ONLINE_DESIGN.md`。

### P7.3 Online Arrival、Continuous Batching 与 Chunked Prefill

合同与 schema focused gate：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  tests/test_online_serving.py \
  tests/test_online_summary.py \
  tests/test_llm_online_serving.py \
  tests/test_engine_contracts.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_context_reset.py
```

门禁覆盖：

- wall-clock constant/poisson/burst arrival与动态 batch membership。
- admission reject、cancel、prefill/decode防饥饿 interleave、swap/recompute合同。
- request queue/TTFT/TPOT/latency与 terminal accounting可复算，summary拒绝篡改记录。
- Q<K bottom-right causal attention、逐 query token slot mapping和 chunk结束后的状态恢复。
- 视觉 payload atomic region、text-only concurrent full-block prefix reuse，以及 VL hash禁用。

单个正式 cell的最小复现示例：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_online.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case text_short --mode off_graph \
  --requests 16 --arrival-process constant --request-rate 20 \
  --max-tokens 8 --max-model-len 512 \
  --max-num-batched-tokens 512 --max-num-seqs 8 \
  --max-chunk-size 128 --num-kvcache-blocks 8 \
  --kvcache-block-size 256 --ttft-slo-ms 500 --tpot-slo-ms 50 \
  --output /tmp/p73_text_short.json
```

正式 9-cell summary重算：

```bash
.venv-local/bin/python scripts/summarize_p7_online.py \
  data/p7_online/p73_*_formal_e7796e9.json \
  --json-output /tmp/p73_online_summary.json \
  --markdown-output /tmp/p73_online_summary.md
```

clean `e7796e9` matrix覆盖 text-short 20 req/s、single-image 4 req/s、mixed
text/image/video 4/10 req/s的 off/compact Graph，以及 301-token text和646-token
image+text chunked输入。9/9 cells均完成全部请求，按各 cell预先声明的 SLO，
goodput fraction均为 `1.0`。长输入分别形成 `128/128/45` 与 `512/134` chunks，
chunked/unchunked输出 exact；10 req/s mixed形成 peak active `4-5`。

完整回归：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p7_online/p73_full_regression_e7796e9.xml
```

JUnit：`tests=268`、`failures=0`、`errors=0`、`skipped=6`、
`time=245.361s`，即 `262 passed, 6 skipped`。

限制：每个正式 cell是一次多请求 engine-level run，不是 HTTP/gRPC server，也没有
process-level repeats；off/compact差异不能形成 speedup claim。正式 matrix未触发
preemption，preemption仅有 deterministic contract tests。当前也没有相同 arrival/SLO
配置的 vLLM online record。详细结果见 `PERFORMANCE_REPORT.md` 6.10，根因见
`docs/issues/P7-007-CHUNKED-PREFILL-STATE.md`。

### P7.4-A Trace-driven Model-precision Logits

root-cause capture使用 CUDA Profiler API排除模型加载、warmup和 Graph capture，
并展开 Graph nodes：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --force-overwrite=true \
  --output=data/p7_external/p74_prism_logits \
  .venv-local/bin/python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 --modes off_graph \
  --max-tokens 32 --warmup 2 --repeat 1 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching --profile-repeat 1 --cuda-profiler-range \
  --profile-output data/p7_external/p74_prism_logits_semantic.jsonl
```

SQLite analyzer的相同 NVTX ranges结果：

| Region | FP32 historical | Model precision |
|---|---:|---:|
| `runner.model.compute_logits` CUDA median | `4.067604 ms` | `0.761571 ms` |
| logits kernels/range median | `4` | `1` |
| `runner.cudagraph.replay` CUDA median | `13.359404 ms` | `12.927219 ms` |

clean `a33e7ed` 单变量正式矩阵覆盖五类 workload、off/compact Graph、output32、
`warmup=2/repeat=5`。十个 cell均为相同 commit、`git_dirty=false`；model precision
相对显式 FP32 TPOT speedup为 `1.216x-1.280x`，peak allocated减少
`2,230-2,317 MiB`。

7-image quality汇总：

```bash
.venv-local/bin/python scripts/summarize_p6_pruning_fidelity.py \
  data/p7_external/p74_prism_coco_fidelity_batch_a_model_formal_a33e7ed.jsonl \
  data/p7_external/p74_prism_coco_fidelity_batch_b_model_formal_a33e7ed.jsonl \
  --baseline-mode off_graph --max-task-quality-drop 0.01 \
  --json-output data/p7_external/p74_model_quality_summary_a33e7ed.json \
  --markdown-output data/p7_external/p74_model_quality_summary_a33e7ed.md
```

结果为 token-F1 `0.318842 -> 0.314482`（drop `0.004360`）、ROUGE-L
`0.285863 -> 0.289953`（改善 `0.004090`）、physical tokens `0.535x`、active
bytes `0.538x`，task gate PASS。

更新 external best-stable：

```bash
.venv-local/bin/python scripts/summarize_p7_external.py \
  --comparison-profile best_stable \
  --prism data/p7_external/p74_prism_*_model_formal_a33e7ed.jsonl \
  --external data/p7_external/p74_vllm_*_best_stable_formal_a33e7ed.json \
  --prism-modes off_graph visual_compact_graph --prism-keep-ratio 0.5 \
  --json-output data/p7_external/p74_best_stable_summary_a33e7ed.json \
  --markdown-output data/p7_external/p74_best_stable_summary_a33e7ed.md
# compared 10 cells; 10 comparable / 0 non-comparable
```

compact Prism/vLLM TPOT为 `1.34x-1.40x`，Prism peak allocated约
`17.39-17.50 GiB`，vLLM约 `17.74-17.93 GiB`。这仍是 offline closed-loop，
不形成反超或 online goodput claim。

最终 full regression：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p7_external/p74_full_regression_cc070b3.xml
```

JUnit：`tests=247`、`failures=0`、`errors=0`、`skipped=6`、
`time=264.664s`，即 `241 passed, 6 skipped`。详细 root cause、第一次 regression
失败及被拒绝方案见 `docs/issues/P7-006-LOGITS-FP32-WEIGHT-CAST.md`。

### P7.4-B CUDA Graph Replay、CPU/GPU Timeline 与 Padding

正式 trace summary重算：

```bash
.venv-local/bin/python scripts/summarize_p7_graph.py \
  --trace-analysis \
    data/p7_graph/p74b_single_image_graph_analysis_0fdd4a6.json \
  --semantic-profile \
    data/p7_graph/p74b_single_image_graph_semantic_0fdd4a6.jsonl \
  --padding-records \
    data/p7_graph/p74b_padding_fixed8_matrix_00b1012.jsonl \
  --json-output data/p7_graph/p74b_summary_72f85ba.json \
  --markdown-output data/p7_graph/p74b_summary_72f85ba.md
```

输入证据分别来自 clean `0fdd4a6` trace与 clean `00b1012` fixed-ceiling matrix；
summary工具来自 clean `72f85ba`。合同验证：

- trace为 schema-v2 `nsys_profile_summary`，包含 replay与五个 Graph 外 target
  ranges；八类 kernel partition fraction之和为 1。
- 31 个 replay的 kernel busy median/p90为 `12.920926/12.932637 ms`，每步
  `2,000` kernels；linear/GEMV为 `9.122773 ms`、`70.551%`。
- replay CPU range median为 `1.899233 ms`，CPU/GPU busy overlap median为
  `0.030400 ms`（`1.618%`），CPU返回后的 GPU tail为 `13.088793 ms`。
- engine decode与 replay的 kernel busy中位数差为 `0.768933 ms`；优化后 logits
  direct GPU busy为 `0.761571 ms`。sampler `13.790 ms` CPU range只暴露 stream
  synchronization，其 direct GPU busy为 `0.007 ms`，禁止重复相加。
- fixed `max_num_seqs=8` 的 8-cell matrix精确覆盖
  `1->1, 2->2, 3->4, 4->4, 5..8->8`；padding为 `0,0,1,0,3,2,1,0`。
  每个 cell repeat-stable，所有 replicated request token rows exact且互不污染。

focused回归：

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_p7_graph_summary.py \
  tests/test_nsys_analysis.py \
  tests/test_benchmark_schema.py
# 43 passed in 3.90s
```

`tests/test_p7_graph_summary.py` 会显式拒绝 kernel category partition缺失、bucket/
padding映射错误与 padding row输出污染。`git diff --check` PASS。bucket matrix的每个
cell是单独 process-level run，故本门禁只证明 capture coverage/correctness，不形成
padding性能或 online goodput claim。timeline解释见
`docs/issues/P7-008-CUDAGRAPH-TIMELINE-ACCOUNTING.md`。

### P7.5 Projection Fusion（完整动态门禁 PASS）

packed gate/up合同与低显存回归：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  tests/test_p7_packed_mlp.py \
  tests/test_qwen3_vl.py \
  tests/test_full_model_structure.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_model_runner_vl_prefill.py
# 32 passed in 62.19s
```

门禁覆盖 packed storage/view alias、旧 state-dict strict load、`Module.to()` 后
rebind、supported forward只调用一次 gate_up projection，以及真实
`hidden/intermediate=4096/12288` BF16 CUDA shape。clean component matrix：

```bash
.venv-local/bin/python benchmarks/bench_packed_mlp.py \
  --correctness-only \
  --batch-sizes 1,2,4,8,210,408,988 \
  --output \
    data/p7_optimization/p75_packed_mlp_shape_correctness_01b3625.json
```

七个 case的 packed/legacy MLP outputs均 bitwise exact，max/mean diff为 `0`。早期
clean `01b3625` correctness record因隐藏GPU占用被正确标记`formal_eligible=false`；
恢复后clean `396702d`正式运行：

```bash
.venv-local/bin/python benchmarks/bench_packed_mlp.py \
  --batch-sizes 1,2,4,8,210,408,988 \
  --warmup 20 --repeat 100 \
  --require-formal-environment \
  --output data/p7_optimization/p75_packed_mlp_micro_396702d.json
```

启动baseline为`4 MiB / 0%`，`formal_eligible=true`，七个case仍全部bitwise exact。
decode rows `1/2/4/8`的Graph ratio为`0.9859x/0.9901x/0.9876x/0.9918x`；
prefill rows `210/408/988`为`0.8248x/0.9945x/0.9041x`。这些只作为组件证据。

QKV correctness-first probe：

```bash
.venv-local/bin/python benchmarks/probe_p7_qkv_fusion.py \
  --output data/p7_optimization/p75_qkv_correctness_01b3625.json
```

batch1 exact；batch2/4/8 的 K/V不 exact且 max diff `1.0`，Q exact。record状态为
`rejected_by_strict_correctness`、`performance_measured=false`，所以不会为已失败候选
制造 timing claim。

clean `8293851` 增加`mlp_projection_mode=legacy|packed`，保证同权重、同state-dict、
同commit单变量A/B；benchmark schema-v6显式记录模式。clean `021d4e2`将同一字段
接入online schema-v2，兼容旧记录。

Full HF门禁：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q -s \
  tests/test_vl_logits_distribution.py \
  --junitxml=data/p7_optimization/p75_hf_logits_ppl_8293851.xml
# 1 passed in 17.11s
```

single/multi-image/video各32-token model-precision logits相对HF max/mean diff为`0`，
PPL diff为`0`；750/750 weights loaded，无missing/unexpected。

Offline A/B使用`bench_system.py --modes off_graph --mlp-projection-mode <mode>`，
output32、clean baseline。text、single/multi-image、video、mixed与三个真实COCO cell
共8/8 token exact；packed/legacy decode TPOT ratio为`0.9924x–0.9952x`，即改善
`0.483%–0.762%`。single-image两轮反向process order结果稳定。E2E受vision prefill
双峰影响，不形成latency speedup claim。

Online A/B使用schema-v2记录single-image rate4与mixed-rate10；两组逐请求token exact，
双方SLO goodput fraction均`1.0`，mixed peak active均`5`。由于每个cell无process-level
repeats，online仅作为regression/SLO门禁。

Node-level Systems命令沿用P7.4 CUDA Profiler API capture，并分别传入
`--mlp-projection-mode legacy|packed`。31个replay结果：

```text
legacy: all kernels=2000, linear=253, busy=12.814647 ms, linear=9.086664 ms
packed: all kernels=1964, linear=217, busy=12.721307 ms, linear=8.998766 ms
delta:  all=-36, linear=-36
```

最终完整回归：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p8_delivery/final_full_regression_021d4e2.xml
# 281 passed, 6 skipped in 297.62s
```

JUnit为`287 tests / 0 failures / 0 errors / 6 skipped`。P7.5判定PASS并保留packed
默认；允许声明的只有记录workload上的小幅decode TPOT改善，不包括稳定E2E或online
speedup。机器可读汇总为`data/p7_optimization/p75_summary_021d4e2.json`。

## P8: 项目交付验证

### P8.1 Editable install 与 API smoke

clean `568f7bb` 修复 setuptools/PEP 621 metadata、项目URL和依赖边界。隔离venv复用
宿主CUDA/PyTorch stack，执行：

```bash
python -m venv --system-site-packages /tmp/prism-install-audit
/tmp/prism-install-audit/bin/python -m pip install -e /data/Prism-Infer
/tmp/prism-install-audit/bin/python - <<'PY'
import importlib.metadata as metadata
from prism_infer import LLM, SamplingParams

print(metadata.version("prism-infer"))
print(LLM.__module__, LLM.__name__)
print(SamplingParams(temperature=0.0, max_tokens=1))
PY
```

实测结果：editable wheel build/install PASS，distribution `0.3.0`，
`prism_infer.llm.LLM` import PASS。该venv的全局`pip check`仍报告继承宿主的
`nvidia-dali-cuda120`要求`six<=1.16`而宿主为`1.17.0`；这不是Prism依赖引入，故
记录为环境warning，不冒充完全独立CUDA stack验收。

### P8.2 无权重环境检查与CPU smoke

clean `d547385`：

```bash
python scripts/check_environment.py

python -m pytest -q \
  tests/test_check_environment.py \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_visual_importance_scoring.py \
  tests/test_compression_off.py \
  tests/test_engine_contracts.py
```

结果：

```text
environment check status: PASS
Prism/Torch/Transformers import: PASS
Running 40 items in this shard
40 passed in 5.11s
```

model preflight：

```bash
python scripts/check_environment.py \
  --model "$PRISM_MODEL_PATH" \
  --require-cuda \
  --min-free-gib 18
```

脚本验证Qwen3-VL config、最小tokenizer/processor文件、4个safetensors shards
（`16.330 GiB`）和CUDA free memory，不加载权重。GPU恢复后连续baseline与正式运行
之间稳定为`1–4 MiB used / 0–2% utilization`，KI-001按完整动态证据关闭；未来若再次
出现外部占用，脚本仍会在加载权重前fail closed。

### P8.3 文档与claim一致性

交付物：

- `README.md`；
- `docs/TECHNICAL_REPORT.md`；
- `docs/REPRODUCIBILITY.md`；
- `docs/KNOWN_ISSUES.md`；
- `docs/APPLICATION_MATERIALS.md`。

本地Markdown链接检查遍历README和`docs/*.md`中的相对链接，结果：

```text
checked_local_links=46
local_link_check=PASS
```

比率口径固定为：7-image aggregate physical/active为`0.535x/0.538x`；COCO batch4
性能cell为`0.536x/0.571x`。README、技术报告、投递材料和CLAIMS不得混用两者。

### P8.4 Fresh install、8B demo 与 full regression

重启后重新创建venv，正常安装`pyproject.toml`声明依赖和editable wheel：

```bash
python -m venv --system-site-packages /tmp/prism-install-audit-local
/tmp/prism-install-audit-local/bin/python -m pip install -e /data/Prism-Infer
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/tmp/prism-install-audit-local/bin/python example.py
```

输出：

```text
Token IDs: [32, 6303, 9334, 448, 279, 3409, 330, 64330]
Text: 'A blue square with the word "BLUE'
```

进程正常退出并释放GPU。该venv复用同一宿主CUDA/PyTorch/driver，不等于另一台机器
ABI验收。当前主线full JUnit见P7.5，`281 passed, 6 skipped`。

### P8.5 最终 Gate Review

- [x] package metadata可构建，核心API在隔离venv可导入。
- [x] 无模型CPU/focused smoke PASS。
- [x] README覆盖安装、模型preflight、demo、compression、trace和验证层级。
- [x] 技术报告、复现手册、Known Issues和投递材料存在且本地链接可解析。
- [x] README/ROADMAP/CLAIMS中的关键数字和禁止claim已同步。
- [x] fresh环境完整8B demo与当前主线full regression PASS。
- [x] P7.5 full-engine/performance动态门禁PASS，claim边界已同步。

因此P8静态与动态出口均PASS。这里的“fresh”限定为同一宿主新venv；在P8关闭时，
TP2、NCU hardware counter、标准大规模质量集和网络server仍是明确后续项，不回写为
P8失败。P9开始后NCU权限已恢复并关闭KI-004；当前租约仍只分配 GPU0，TP2 保持
NOT RUN / UNVERIFIED。

交付前必须检查：

```bash
git diff --check
git status --short
```

并确认剩余条件项仍在`docs/KNOWN_ISSUES.md`和`docs/CLAIMS.md`中显式受限。

## P9: 秋招旗舰化验证

### P9-A.1 RFC 与成功标准

设计来源：`docs/P9_ARCHITECTURE_PERFORMANCE_RFC.md`。进入任何结构性 runtime
改动前必须确认：

- 项目定位为 Qwen3-VL 跨层多模态 Runtime，不把 last-query selector 声称为算法创新；
- 最终同时要求 quality–physical-memory Pareto 与一个预注册 long-visual runtime/SLO
  胜出；
- external headline 使用 best-stable，diagnostic matched 只做归因；
- unit-scale FP8 是 rejected baseline，scaled FP8 重新独立过门禁；
- 工程主线在 2026-08-06 冻结，保留 7–10 天学习与复习。

P9-A RFC review：PASS。workload/quality manifest 和正式性能证据仍按后续小节单独判定，
不能因为文档完成而自动 PASS。

### P9-A.2 Workload 与标准质量协议

机器可读 contract：

```text
benchmarks/workloads/p9_headline.json
canonical SHA256: 42d1387320b1b30c3b0afa0bf3113f0dd905a38b38bc583cfe6c6eb3ef4f8656

benchmarks/workloads/p9_quality_protocol.json
canonical SHA256: 85adb4b246ab3fc55bc70e02ad75d97c5aa903e89387e499fc3aea1ac2edb25d
```

runtime manifest 固定：

- H1：8-image 448px、output128、offline batch1/4、5 fresh-process repeats；
- H2：16-frame 448px、output128，跨框架 prompt token 不同则条件跳过；
- H3：主 trace 为 text/single-image/H1 的 `40%/30%/30%`，600 requests，Poisson
  rate `1/2/4 req/s`，三个固定 seed；video-compatible trace 单独为
  `40%/30%/20%/10%`；
- 物理 KV budget `4,294,967,296 bytes`，payload、scale 与 metadata 都计入；
- SLO 由 vLLM best-stable low-load class p50 按固定 `5x TTFT / 2x TPOT` 公式冻结。

quality protocol 固定 DocVQA/MuirBench/MVBench 的 repository revision、确定性 SHA256
选样算法、`1.0 percentage point / 0.01 normalized metric` non-inferiority margin 与
paired bootstrap 95% CI。MVBench 需要手工媒体且有 source-video license 条件，媒体或
prompt 语义不可复现时显式排除，不换题。现有 7-image COCO 只作 preflight。

验证：

```bash
PYTHONPATH=/data/Prism-Infer python -m pytest -q \
  tests/test_p9_protocol.py tests/test_bench_paged_decode.py
# 15 passed in 0.11s
```

判定：协议与 source revision PASS。公开数据媒体尚未物化；selected-ID/media SHA256
是 P9-C 首次标准质量运行前置门禁，不影响 P9-A 的协议冻结。

### P9-A.3 GPU 可见性与资源分配边界

管理员开放 NCU/NSYS 权限后，以下只读命令显示宿主机有 8 张可见设备：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,pci.bus_id \
  --format=csv,noheader,nounits
nvidia-smi topo -m

```

但用户确认当前租约只分配 GPU0。`nvidia-smi` 可见性、空闲快照和拓扑都不是调度分配
证明，也不授权使用其余设备。

P9-A 早期曾误判“可见即有权使用”，并跨 GPU0–1 运行 Prism TP2、Prism-stack
all-reduce 与隔离 vLLM-stack all-reduce。下面这些旧结果全部标记为无效实验：

```text
INVALID: Prism TP2 / Prism all-reduce on GPU0-1 -> cudaErrorInvalidValue
INVALID: isolated vLLM-stack all-reduce on GPU0-1 -> 3.0
```

由于 GPU1 未分配给当前租约，失败不能定位 Torch/CUDA/NCCL/SM120，成功也不能证明
双卡合法可用。相关日志仅作审计轨迹，不进入 capability/root-cause evidence。判定是
TP2 NOT RUN / UNVERIFIED；后续命令均限定 `CUDA_VISIBLE_DEVICES=0`，未来获得明确双卡
分配后再从逐卡 allocation 和最小 collective 重新验证。

### P9-A.4 NCU hardware counter

权限恢复验证：

```bash
ncu --version
# Nsight Compute 2025.1.0.0
```

BF16/Qwen GQA、batch8/context4096、page16/256 clean full-set counter：

| Page | Duration | DRAM throughput | Compute throughput | Achieved occupancy | Waves/SM | Registers/thread |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 449.95 us | 17.48% | 14.16% | 12.49% | 0.19 | 64 |
| 256 | 543.26 us | 14.44% | 11.70% | 12.48% | 0.17 | 56 |

correctness 均 PASS，max diff `4.882812e-4`、mean diff约 `3.0e-5`。NCU 对两个 case
都指出 grid 太小；当前 launch grid 是 `(batch, query_head)=256`。判定：KI-004
CLOSED；counter 只解释该 kernel/case，不外推为 full-engine GPU utilization。
两格 block 都是 128 threads，NCU 规则给出约 `0.2` full waves；不能把低 DRAM/
compute counter 简化为纯 memory-bound/compute-bound。早期权限恢复 diagnostic 的
`445.60/550.46 us` 没有配套 clean raw，已被本节数字取代，不得混用。

report 内嵌的 profiler command 等价于（`PAGE=16` 和 `PAGE=256` 各执行一次）：

```bash
PAGE=16
STEM="data/p9_baseline/ncu_paged_decode_page${PAGE}_b8_c4096_29c0dbe"
ncu --target-processes all --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*paged_decode_attention_kernel.*' \
  --launch-count 1 \
  --export "$STEM" --force-overwrite --log-file "${STEM}.log" \
  .venv-local/bin/python benchmarks/bench_paged_decode.py \
  --page-sizes "$PAGE" --batch-sizes 8 --context-lens 4096 \
  --cache-dtypes bf16 --warmup 1 --repeat 1 --seed 20260717

ncu --import "${STEM}.ncu-rep" --csv --page raw > "${STEM}.csv"
```

正式 artifacts 与 SHA256：

```text
data/p9_baseline/ncu_paged_decode_page16_b8_c4096_29c0dbe.ncu-rep
32101271e93747f087f6a836b991ea95a07e747ee68c5e48c761ef83b9bfed35
data/p9_baseline/ncu_paged_decode_page16_b8_c4096_29c0dbe.csv
b336814d2c2b4969b6e7d42ca030b6c33d3588b13066dedf153a50a3d4e9205a
data/p9_baseline/ncu_paged_decode_page256_b8_c4096_29c0dbe.ncu-rep
96b92cf9e878036f210ff339808c29f02a25358fcf38c4fe1c078db45d3fe478
data/p9_baseline/ncu_paged_decode_page256_b8_c4096_29c0dbe.csv
fd6b45825ecb86cfe390507c485a81a18d67e6205f89ab5927efdf07c53a874c
```

对应 `.log` 同目录保存 profiler stdout/stderr。判定：profiler raw-artifact 子门禁 PASS。

### P9-A.5 结构化 Paged Attention benchmark

`benchmarks/bench_paged_decode.py` 的 P9 contract：

- `--seed` 固定 logical Q/K/V；相同 batch/context/dtype 在不同 page 下使用同一输入；
- `--page-sizes` 支持多 page matrix，旧 `--block-size` 保留为单值 alias；
- 每个 case 同时检查 max/mean absolute difference；
- JSON/JSONL 记录全部 latency samples、median/p90/p99/min/max/token/s；
- 记录 Q/K/V/page-table shape、logical/physical KV bytes、allocator before/after/peak；
- 记录 full commit、dirty state、GPU UUID、Torch/CUDA/Triton、driver/NVML preflight；
- 默认不覆盖已有 artifact。

CPU/focused helper tests：

```bash
PYTHONPATH=/data/Prism-Infer python -m pytest -q tests/test_bench_paged_decode.py
# 10 passed in 0.09s
```

dirty smoke（只验证 harness，不形成性能 claim）：

```bash
PYTHONPATH=/data/Prism-Infer python benchmarks/bench_paged_decode.py \
  --page-sizes 16,256 \
  --batch-sizes 1 \
  --context-lens 256 \
  --cache-dtypes bf16 \
  --warmup 1 --repeat 2 \
  --max-start-memory-used-mib 1024 \
  --max-start-gpu-utilization 5 \
  --output data/p9_baseline/paged_decode_smoke_dirty.jsonl \
  --overwrite
```

结果为 `2/2` correctness PASS；两种 page 的 max/mean diff 都是
`1.953125e-3/1.085676e-4`，证明 page-independent seed/packing 生效。repeat=2 且
`git_dirty=true`，latency 只用于 smoke。

clean formal 命令：

```bash
PYTHONPATH=/data/Prism-Infer python benchmarks/bench_paged_decode.py \
  --page-sizes 16,32,64,128,256 \
  --batch-sizes 1,8 \
  --context-lens 4096,8192 \
  --cache-dtypes bf16 \
  --warmup 10 --repeat 100 \
  --seed 20260717 \
  --max-start-memory-used-mib 1024 \
  --max-start-gpu-utilization 5 \
  --output data/p9_baseline/paged_decode_page_matrix_<commit>.jsonl
```

正式结果来自 clean commit `29c0dbedc6c945637f286dd6ac6916b12dcee5ae`、RTX 5090
UUID `GPU-989db6f6-3273-d1dd-b2b9-56cced4f30a4`。20/20 correctness PASS，每格
100 samples，`git_dirty=false`，GPU preflight PASS。kernel median：

| Batch | Context | P16 | P32 | P64 | P128 | P256 | Best vs P256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 0.2385 ms | 0.2364 ms | 0.2735 ms | 0.2729 ms | 0.2736 ms | 13.6% lower |
| 1 | 8192 | 0.4491 ms | 0.4464 ms | 0.5205 ms | 0.5200 ms | 0.5208 ms | 14.3% lower |
| 8 | 4096 | 0.3711 ms | 0.3695 ms | 0.4570 ms | 0.4571 ms | 0.4578 ms | 19.3% lower |
| 8 | 8192 | 0.7023 ms | 0.7048 ms | 0.8792 ms | 0.8797 ms | 0.8793 ms | 20.1% lower |

全矩阵最大 max diff 为 `4.882812e-4`，最大 observed mean diff 为
`3.035463e-5`。raw evidence：

```text
data/p9_baseline/paged_decode_page_matrix_29c0dbe.jsonl
SHA256: 9460339fdf9bce7a4b8dfb6b4c8b93b0dc973f914b473d38e8516addb3b757b8
```

判定：formal matrix PASS；page16/32 进入下一阶段候选。边界是本矩阵的 context
都能被 page size 整除，尚未覆盖页尾碎片；SDPA reference 包含 Python page gather，
只用于 correctness。上述 kernel microbenchmark 不能直接表述为 full-engine TPOT，
也不足以立即修改默认 page size。

### P9-A.6 Gate Review（当前）

最终 focused regression：

```bash
PYTHONPATH=/data/Prism-Infer .venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py \
  tests/test_paged_decode_kernel.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_p9_protocol.py \
  tests/test_bench_paged_decode.py
# 64 passed in 6.99s

.venv-local/bin/python -m compileall -q prism_infer tests benchmarks scripts
git diff --check
```

本地 Markdown 链接检查覆盖 README 与 `docs/**/*.md`，结果为
`checked_local_links=57 / PASS`；Page Matrix 与四个 NCU report/CSV SHA256 重新计算
全部匹配；结束时已分配 GPU0 回到 `1 MiB / 0%`。其他可见设备不属于当前租约，
不纳入 release gate。本地证据提交后要求 `git status` clean。

- [x] 架构、量化、Graph/compiler、kernel、scheduler/server 与 TP 决策冻结。
- [x] 资源边界已纠正：仅 GPU0 已分配；TP2 与跨 GPU1 的旧实验标记无效。
- [x] NCU counter 权限恢复，代表性 page16/256 指标已采集。
- [x] structured benchmark helper tests 与 dirty smoke PASS。
- [x] H1/H2/H3 与 DocVQA/MuirBench/MVBench manifest/hash/revision 冻结。
- [x] clean 20-cell page matrix。
- [x] NCU raw artifacts 与 SHA256。
- [x] P9-A 最终 focused regression、文档链接和 clean worktree。

当前判定：P9-A PASS；允许进入 P9-B。P9-B 第一批 diff 仍只允许处理 typed config、
`Sequence` 全局 page state 与 execution backend contract，不能混入 scaled FP8 或
kernel 改动。

### P9-B 架构硬化与执行边界（PASS）

P9-B 只修改 config、request/page identity、scheduler/executor contract 和 backend
lifecycle；没有加入 scaled FP8、NVFP4、W/A quantization 或新 kernel。实现门禁：

- frozen typed domains：`ModelConfig/CacheConfig/SchedulerConfig/ExecutionConfig/
  QuantizationConfig/ServingConfig`；strict flat adapter 对 unknown kwargs 和隐式 bool
  coercion fail closed；HF model parsing 在用户配置校验之后；
- runtime `Config` 的 EOS、GPU/CPU KV capacity 由 replacement API 生成，不原地修改；
  CPU KV block ratio 是显式 cache policy，不再散落 `// 2`；
- `Sequence` 必须显式接收 page size 与 request ID；真实 ID 来自 engine-owned
  `MonotonicRequestIdAllocator`，测试可注入；pickle 保留 request FSM/page/layout 并拒绝
  legacy tuple 或类型错误 payload；
- immutable `BatchPlan -> DeviceBatch -> ExecutionResult`；`DeviceBatch` 不包含 mutable
  `Sequence`，phase/context/ID/token-count/batch-cardinality 均有负向门禁；
- `ModelExecutor` 只分发 `run_plan`；backend 统一暴露
  `prepare/warmup/capture/execute/release`，执行异常后 Context 必须释放；
- 显式 `exit()` 注销 atexit handler、断开 backend -> runner ownership、在 runner exit
  失败时仍清理引用，并在退出路径回收 Python cycle/CUDA cache；该逻辑不进入推理热路径；
- `compile_graph`、动态 logical-prune + Graph、超出支持 batch 的 Graph 和未连接的
  W/A quantization/serving 在 startup 失败，不允许运行时 silent fallback。

专项架构回归（含 frozen Config pickle + `multiprocessing spawn`）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/Prism-Infer \
  .venv-local/bin/python -m pytest -q \
  tests/test_p9_architecture_contracts.py
# 16 passed in 8.91s
```

受影响 focused regression：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/Prism-Infer \
  .venv-local/bin/python -m pytest -q \
  tests/test_p9_architecture_contracts.py \
  tests/test_compile_execution_config.py \
  tests/test_sequence_multimodal.py \
  tests/test_engine_contracts.py \
  tests/test_kv_engine_hardening.py \
  tests/test_kv_physical_layout.py \
  tests/test_scheduler_swap_tables.py \
  tests/test_model_runner_context_reset.py
# 65 passed in 11.78s
```

CUDA 隐藏的完整 CPU/contract 回归隔离验证 host contract；原套件中两个直接 CUDA
test 没有 no-CUDA skip，因此正式命令显式 deselect，并由下方独立 GPU gate 覆盖：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/data/Prism-Infer \
  .venv-local/bin/python -m pytest -q tests \
  -k 'not test_prepare_prefill_builds_paged_prefix_context and \
      not test_text_only_generate_greedy_smoke'
# 284 passed, 34 skipped, 2 deselected in 107.59s
```

GPU0 上的 VL input/context/profile 组故意在同一个 pytest 进程连续运行 single-image、
multi-image 和 video 的 HF -> Prism 8B 路径，用于同时验证 correctness 与退出生命周期：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/Prism-Infer \
  .venv-local/bin/python -m pytest -q \
  tests/test_llm_vl_generate.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_context_reset.py \
  tests/test_performance_profile.py
# 21 passed in 46.24s
```

首次组合运行在第一轮 Prism engine 退出后仍看到约 `27.4 GiB` active allocation，后续
HF load 因而 OOM。root cause 是 runner/backend ownership cycle 使模型与 KV tensor 未被
确定性释放；同时 atexit 仍保留已完成 engine。修复后同进程 21 项全部 PASS，并新增
backend-cycle 与 runner-exit-failure 两项 CPU contract test。

真实 Qwen3-VL-8B eager smoke：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/Prism-Infer \
  .venv-local/bin/python -m pytest -q \
  tests/test_text_only_regression.py
# 1 passed in 15.87s
```

真实最小 CUDA Graph smoke 使用 `execution_backend='cuda_graph'`、batch `1`、
`max_model_len=max_num_batched_tokens=128`、2-token greedy。结果：

```text
tokens: [785, 3070]
resolved GPU/CPU KV blocks: 315 / 157
captured batch sizes: [1]
capture scope: decode_model_forward
capture time: 275.993 ms
requested/selected batch: 1 / 1
post-exit active/reserved: 17,039,360 / 41,943,040 bytes
```

这证明 frozen runtime capacity replacement、Graph capture/replay 和 engine release 在真实
8B 单卡路径可用；它不是 latency claim。eager 与 Graph 都固定
`CUDA_VISIBLE_DEVICES=0`。进程内剩余量低于本轮设置的 `64 MiB` CUDA runtime-residue
guard，不再存在模型/KV 级泄漏；进程退出后已分配 GPU0 为 `1 MiB / 0%`。该 guard
用于防回归，不作为性能收益 claim。

TP2 没有运行有效动态门禁：当前租约未分配第二张 GPU。P9-A 期间跨 GPU1 的 TP2 与
all-reduce 尝试已经作废，不得用其中的 `cudaErrorInvalidValue` 推导 NCCL/SM120 blocker，
也不得用隔离环境的成功输出声称双卡可用。TP2 状态为 NOT RUN / UNVERIFIED，不阻塞
单卡 P9-C；只有未来明确获得至少两张卡后才重新执行 `tests/test_llm_vl_tp2.py`。

静态门禁：

```bash
.venv-local/bin/python -m compileall -q prism_infer tests benchmarks scripts
git diff --check
```

当前判定：P9-B PASS；允许进入 P9-C scaled FP8 scale lifecycle。P9-C 必须保持本节
backend/identity contract，不得把 unit-scale direct cast 重新命名为 scaled FP8。

### P9-B.2 代码质量复审与 compile 安全边界（PASS，2026-07-19）

本轮不是新增算法或性能 claim，而是对 P9-B 之后的生产代码做第二次 AI Infra 架构
复审。主要结果：

- scheduler prefill 被拆为 immutable candidate、batch resource builder、running chunk
  收集和 waiting admission；视觉 patch budget 使用 `Config` 的正式默认常量，不再用
  `2**63 - 1` 伪装“无限”；FCFS 顺序、prefix-cache 和 oversized visual request 独占
  batch 的语义保持不变；
- `LLMEngine` 将 text/image/video 的 host prepare 与 scheduler submit 分开；mixed batch
  先完整校验和预处理再发布请求；text/VL/mixed 同步 API 共用一个生成循环，异常时也
  关闭 progress bar；exit 按 runner/backend、control channel、worker process 和 CUDA
  cache 分层释放，并保留首个 cleanup failure；
- online session 将 request value contract、arrival/cancel event、engine step、idle wait 和
  result materialization 分开；NaN/负时间、错误 prompt/media payload 在 event loop 之前
  fail closed；时间换算统一使用命名常量；
- quality artifact、external comparison、profiling、KV trace 和 pruning fidelity 等
  analysis/evidence 路径已按输入合同、运行合同、数据集计分、聚合与渲染职责拆分；
- 整个 `prism_infer` 包（包括 `analysis`、`engine`、`layers`、`models`、`ops`、
  `vision`）通过 `C901/PLR0911/PLR0912/PLR0915`；生产包 runtime `assert` 为 0，
  `PLR2004` 为 0。上述范围已加入 CPU CI，不再只依赖一次性人工扫描。低层 Triton
  kernel ABI 与标准 `nn.Module` 构造签名没有为了 lint 形式主义强行包装。

真实 `torch.compile` 调查同时修复了一个安全性问题。每层 K/V cache 是同一块
monolithic KV allocation 的非零 `storage_offset` view。把 mutable KV store 捕获进
fullgraph 后，AOT functionalization 曾为 V cache 生成 `155,189,248` elements 的 clone，
但 view data pointer 已位于 `storage_offset=150,994,944`；生成代码因此可能从 view
起点越界读取，表现为 token 分叉和 `illegal memory access`。这不是普通 BF16 rounding。

最终 compile state boundary 为：

```text
compiled pure subgraph: qkv_projection_qk_norm_mrope
graph-external mutable boundary: validated_runtime_store_and_paged_decode
```

即 fullgraph 只捕获 QKV projection、QK-Norm 和 M-RoPE；KV store commit 与 paged
attention read 继续使用带完整 contract validation 的 Prism Triton runtime。删除了不安全、
无正式调用方的 compiled KV-store primitive；compile 仍必须由
`allow_unsafe_decode_compile=True` 显式开启，没有 fallback，也没有提升为 supported
backend。

Correctness 边界必须诚实保留：真实 8B batch1/output32 eager 与 compile token exact；
batch2/output32 两行都在第 29 个生成 token 出现后段分叉。FP32 logits 加
`CUDA_LAUNCH_BLOCKING=1` 的诊断已不再触发 illegal access，但不能消除 batch2 数值
分叉。因此本修复只把 compile 从“可能非法访问 aliased KV view”收敛为“内存安全、
边界可审计的 rejected benchmark candidate”。

本轮完整门禁：

```bash
.venv-local/bin/ruff format --check prism_infer tests benchmarks scripts
.venv-local/bin/ruff check .
.venv-local/bin/ruff check --select C901,PLR0911,PLR0912,PLR0915 prism_infer
.venv-local/bin/ruff check --select S101,PLR2004 prism_infer
.venv-local/bin/python -m compileall -q prism_infer tests benchmarks scripts tools
git diff --check
# PASS

.venv-local/bin/python -m pytest -q \
  -m "not model and not gpu and not slow and not distributed"
# 368 passed, 76 deselected in 20.71s

# 在 fresh interpreter 中令 triton/flash_attn 的 find_spec 返回 None，
# 并阻止其直接 import 后重跑同一 CPU marker 集：
# 368 passed, 76 deselected in 20.39s

CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python -m pytest -q \
  -m "gpu and not model and not slow and not distributed"
# 18 passed, 426 deselected in 7.59s

CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python -m pytest -q \
  tests/test_compile_execution_config.py::\
test_compile_qkv_split_handles_nonzero_offset_cache_views
# 1 passed in 6.72s
```

最终真实 Qwen3-VL-8B batch1/output4 smoke 使用同一 single-image workload、BF16、
greedy、`off_eager,off_compile_attention`：两边 token IDs 都是
`[785, 2168, 3897, 374]`。schema-v7 metadata 分别记录
`qkv_projection_qk_norm_mrope` 与 `validated_runtime_store_and_paged_decode`；运行结束后
GPU0 回到 `1 MiB / 0%`。本 smoke 只有 `warmup=0/repeat=1`，仅作为 correctness、
metadata 和 release gate，不形成 latency claim。

本地 raw evidence：

```text
/tmp/prism_compile_split_architecture.jsonl
/tmp/prism_compile_split_long.jsonl
/tmp/prism_compile_split_fp32.jsonl
/tmp/prism_final_compile_smoke.jsonl
```

这些 `/tmp` 文件尚未进入 release artifact；KI-011 的可复现证据风险仍适用。

### P9-C.1 scaled FP8 与标准质量 evaluator（FORMAL QUALITY PASS）

本检查点把 unit-scale `fp8_kv` 与动态 `scaled_fp8_kv` 保持为两个独立模式。scaled
路径使用 per-token-per-KV-head FP32 scale，并在 GPU/CPU swap、copy-on-write、physical
compaction、Triton store、paged decode 和 CUDA Graph replay 中与 FP8 payload 同生命周期。
KV artifact/schema v7 分别记录并交叉校验 payload、scale 和 total bytes。

标准质量协议固定为 DocVQA validation、MuirBench test 与 MVBench test。物化数据被
`data/` gitignore 隔离，未进入仓库；本轮复核结果为 31 个 source 文件、5,196 个唯一
媒体文件、1,212,748,856 bytes，状态 `PASS`。身份如下：

```text
quality protocol canonical SHA256:
85adb4b246ab3fc55bc70e02ad75d97c5aa903e89387e499fc3aea1ac2edb25d
quality evaluator canonical SHA256:
aa00962fd516c08d7a9fb42df33f20929e360b17b97c369757ce4bd46999d91b
materialization manifest SHA256:
dcfe4c82691013cccd7fd58f987919ee83c329caef50dd243c28522d7082a50a
tracked selection SHA256:
c511ab44dca420a5b4ef65ae378104ce046f21f81703203a26a73620f9a9651e
```

评测链路的三项真实缺陷在正式质量结论冻结前被 fail closed：

- interleaved MuirBench 首样本含 4 个 576-token image span；原冻结的 512-token
  chunk 无法原子消费。质量 evaluator 因而明确使用 non-chunked eager prefill，隔离
  KV precision 变量；per-image chunk payload slicing 留给独立 runtime 工作，不在质量
  结论中临时改参数。
- 容器预装 `opencv==4.10.0` 没有 FFmpeg/GStreamer，且 HF processor 会把已选 16 帧
  按默认 24 FPS 二次采样为 4 帧。quality extra 现固定
  `opencv-python-headless==4.10.0.84 / FFMPEG`，artifact 记录 decoder 身份；processor
  使用 `do_sample_frames=false` 并接收原视频 FPS/帧索引。MVBench smoke 最终
  `video_grid_thw.T=8`，对应保留 16 个输入帧。
- MVBench final 的 `86438.webm` 容器元数据报告 41 帧，但 OpenCV/FFmpeg 只能顺序
  解码 25 帧，随机 seek 到 frame 26 必然失败。冻结 decoder contract 现显式记录
  `random_seek_then_sequential_count_and_decode`：仅在 seek/decode 失败时顺序统计实际
  可解码帧数，重算 segment centers 并顺序取帧。完整 final 预检为 190/190 eligible
  PASS，仅该样本触发 fallback；artifact 记录 reported/actual count、触发操作/索引与
  sampled RGB identity，paired comparator 强制 off/scaled 的逐帧身份完全一致。

生成 artifact 同时保存 clean raw prediction、含特殊 token 的 lossless decode 与 token
IDs。官方 scorer 使用 clean prediction；修复前 DocVQA 样本的 `<|im_end|>` 尾缀把
ANLS 错误压到 `0.6`，修复后为 `1.0`。独立 comparator 不信任 artifact 内的 score 或
aggregate：它从 run contract 绑定的物化 manifest/JSONL 按 sample ID 定位 reference，
核对媒体 SHA，独立重算每条 score 及汇总；缺少 reference 的 formal artifact 会 fail
closed。随后检查 run/evaluator/protocol/input 哈希链与 KV 字节，执行 seed `20260717`、
10,000 resamples 的 paired-bootstrap 95% CI；MuirBench strict parser 是必过 guardrail。

历史 GPU0 dirty smoke（每项 1 sample，evaluator
`f1f93d6ae9fede46729056982e10dc3d9a78275f7bb44bf4b531c940f568ea8a`；仅验证链路，
已被当前 fallback-aware evaluator supersede，不是质量 headline）：

| dataset | off artifact SHA256 | scaled artifact SHA256 | comparison SHA256 | token exact | score off/scaled |
|---|---|---|---|---:|---:|
| DocVQA | `1583eb0ced1ff7c6daee6b85bf794de7a4ed57f0c06579ee9516738205d14e77` | `49e13cee1d4d4494bbf50e048a3f6d361642d97f5d787734c665c98ddc960630` | `f3412b99ca78139f6961823faae4c4c8769727a9f187b32d87af3076e9367760` | 1/1 | ANLS `1.0/1.0` |
| MuirBench | `f929cab356f76785f89308ad2e1e6f9a44551780c4c47c489e959bedebc27772` | `540322f4e52f17b2631deb8aeef44049f7c467eb961ec2bc497d036a9a1ba776` | `8b0dad864bed75eaadc9bb0c70a570d66467922bec16e02f0827f753b827205e` | 1/1 | accuracy `0.0/0.0` |
| MVBench | `3a753b87a72a67a734b425cc14d65a68b9878cd686b2ebc04267b6bdc383ac83` | `6ca475ddb4b8acbca8c3a3183c9e07c98a9ac3ebb9fe0d7e75d2a7dbaaa74879` | `bda4e52d21c750044e2717407d1f4459cf7f0bd3e54e8b9d7985efe5ab3af8f4` | 1/1 | accuracy `0.0/0.0` |

三组 scaled KV pool 都是 payload `754,974,720` + scale `23,592,960` =
`778,567,680` bytes；BF16 pool 为 `1,509,949,440` bytes，因此物理总比例为
`0.515625`，节省 `48.4375%`。三组 paired delta 与 CI 均为 `0.0`，但单样本 smoke
只允许标记 `SMOKE_ONLY`，不得用来声称标准质量 non-inferiority。

当前 evaluator 的 clean formal development/final matrix 于 2026-07-18 在 commit
`5ada892cd44a118cbd04f31399bad52e541f71f6`（`git_dirty=false`）完成。development
通过后没有修改代码、evaluator、协议或阈值；decoder 修复产生新 evaluator hash 后，
此前 commit `7e62f34` 的所有 formal 诊断 artifact 均作废，下面 12 个 prediction 与 6 个
comparison artifact 已在 `5ada892` 从头重跑。原始文件保存在 gitignored
`data/p9_quality/results/`。

| dataset | subset | eligible/selected | metric off → scaled FP8 | paired delta / 95% CI | token exact | decision |
|---|---|---:|---|---|---:|---|
| DocVQA | development | 200/200 | ANLS `0.9246396993 → 0.9246396993` | `0.0 / [0.0, 0.0]` | 198/200 | PASS |
| DocVQA | final | 500/500 | ANLS `0.9225580116 → 0.9225580116` | `0.0 / [0.0, 0.0]` | 497/500 | PASS |
| MuirBench | development | 200/200 | official `0.695 → 0.695`; strict `0.690 → 0.690` | 两项均 `0.0 / [0.0, 0.0]` | 199/200 | PASS |
| MuirBench | final | 500/500 | official `0.654 → 0.654`; strict `0.646 → 0.646` | 两项均 `0.0 / [0.0, 0.0]` | 495/500 | PASS |
| MVBench | development | 97/100 | accuracy `0.6082474227 → 0.6082474227` | `0.0 / [0.0, 0.0]` | 97/97 | PASS |
| MVBench | final | 190/200 | accuracy `0.6157894737 → 0.6157894737` | `0.0 / [0.0, 0.0]` | 190/190 | PASS |

所有 comparison 都使用 seed `20260717`、10,000 次 paired bootstrap；bounded accuracy
margin 为 `0.01`，DocVQA normalized margin 为 `0.01`。MuirBench official random
fallback 在 development/final 的 off/scaled 两边分别都是 `4/4` 与 `15/15`，strict
guardrail 同时通过。MVBench 的 3/10 条 development/final exclusion 均来自预冻结协议，
没有因运行失败新增排除；final 的 `86438.webm` 在 off/scaled 两边都得到 RGB identity
`98bfe86f1998e460140c12d33902f855d5dcd1c5cf523c94ac790e434b8cc152`。

正式原始文件 SHA256：

| dataset/subset | off prediction | scaled prediction | comparison |
|---|---|---|---|
| DocVQA development | `ec3e2159b572e45964603f0886792343a847be9aeab2c90d1db8e39911e3e3cf` | `d94dfc94289df5b9748f012267beb27d62c49ecde768e46dfd4c56b29385c891` | `53d5f2f527d8e062f83930acc9654daa463a78be2966e9633b8a1add71443e52` |
| DocVQA final | `9fadb1d8d9c6e9404af90510e66a1d1f1bc259fbdb2ae50817a657883a89bcd7` | `a2f2b74c26bcf95496105a7895240ac453673b19da9e280a0505d235dab06a33` | `3c324538f1eabe7598223ff62e74d12ba52c56d31949e6681c4a9c5066ac24d6` |
| MuirBench development | `59d665bfd4a121e460165ba8d3a13fbf3c45cf4dfefbddca4afe00a675506895` | `cd65561a35ccae3e1646ce0a50b2c8c49b524db869bc034263e3a6d9a4350d45` | `e2f50c9526f850bb022b4a615553c152032a49c79eb83ce0a3ec5350873b3294` |
| MuirBench final | `7f8d68ecd6fad3b3ce37c7935a26e9be6b5930df0d4eb61370e212c07b90a8da` | `b93b0654741593686aa5b0ccb601c696295b34757f0a8733d6b537bf0b7c3ce5` | `65ddac38fcf3301fd29b188647ae1088451bb51d697facd64285f29ec7ce17ae` |
| MVBench development | `e28cd5ce04b04f87ecd2899a3b33efde3dee934d7fb4c53451898c894f9e944d` | `d98f29ac0518275ba02e9695ea2b9ef916c7ceebc00e30983d7b82af6bc700d3` | `fbf2b81759328970349826f4a0a4423ead3f269cba6c72e1f098f41f9811d6ce` |
| MVBench final | `501a8cc013f06e50e8df436bcaf90cc605a1661bf8362b23f2cdd517b5193ff0` | `b8220910f6acb0fa59224f8eaf998313f39c01209e33d69d360587d806b3168e` | `894941189983d374061b0a93f1ac8ec61a73d6ca19e8e047172bfcf4165b791c` |

验证命令：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python -m pytest -q \
  tests/test_p9_*.py tests/test_scaled_fp8_kv_cache.py \
  tests/test_benchmark_schema.py tests/test_processor_pipeline.py \
  tests/test_processor_pipeline_multi_image.py \
  tests/test_processor_pipeline_video.py tests/test_http_range_reader.py \
  tests/test_llm_output_decoding.py
# 127 passed in 14.41s

CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python -m pytest -q
# 377 passed, 6 skipped in 265.78s

.venv-local/bin/python -m compileall -q prism_infer benchmarks scripts tests
git diff --check
```

当前判定：P9-C scaled FP8 相对 Prism BF16 的标准多模态 development/final formal
non-inferiority gate 为 `PASS`，同时物理 KV pool 节省 `48.4375%`。该结论只覆盖
Prism 内部 BF16→scaled FP8 的质量与 cache-pool bytes，不等于完整 P9 Gate A：尚未完成
同一 Qwen3-VL/processor 语义下 vLLM 最强 scaled-FP8 baseline 的质量–物理显存 Pareto，
也不包含 page-table/allocator metadata、外部 runtime 或服务端 headline。

### P9-C.2 vLLM External Gate A（FORMAL MATRIX COMPLETE / MIXED）

External runner使用隔离的 vLLM `0.24.0`（distribution commit `gee0da84ab`）与同一
Qwen3-VL-8B-Instruct revision。质量 cell固定单卡 GPU0、eager、TP1、关闭 prefix
cache/chunked prefill/async scheduling，并设置
`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`。运行时对象审计
确认 language 36层均为 `TRITON_ATTN`，vision 27层均为 `FLASH_ATTN`；不能仅依据请求
参数或启动日志推断实际 backend。

公平容量使用框架原生 page layout：Prism `40 × 256` 与 vLLM `640 × 16` 都是
10,240 logical token slots。vLLM BF16实际36个 KV tensor合计 `1,509,949,440` bytes；
`fp8_per_token_head` payload为 `754,974,720` bytes，inline FP32 K/V scales为
`23,592,960` bytes，总计 `778,567,680` bytes，即 BF16 pool的 `0.515625x`。每个 cell
还读取实际 worker block-table tensor：GPU/CPU各 `2,048` bytes。Python allocator对象图
尚无跨框架统一字节合同，Prism schema-v1 formal quality artifact也未记录 page-table/
allocator metadata，因此 artifact明确设置 `full_physical_comparable=false`；这些 smoke
数字不能包装成完整物理显存 Pareto优势。未测 allocator字段显式写为 `null`，validator
拒绝把不完整计量写成 `0`。环境证据同时强制记录 GPU UUID、driver、compute capability、
total memory，并直接哈希13个实际 vLLM执行文件和3个 Transformers Qwen3-VL/Qwen2-VL
processor文件；完整环境对象进入 `run_contract`，因此跨版本、跨源码或跨GPU的 checkpoint
不能被 `--resume` 混入同一 artifact。validator同时锁定 vLLM `0.24.0`、distribution
commit `gee0da84ab` 与 Transformers `5.13.0`，避免仅凭 package标签推断本地执行代码。

集成门禁发现并修复了三个会破坏公平性的 harness问题：

- `mm_processor_kwargs={"max_pixels": ...}` 对 Transformers 5.13 Qwen3-VL video
  processor是无效参数。runner现传完整 `size={shortest_edge,longest_edge}`，并在启动
  engine前核对 pinned processor有效值；MVBench恢复冻结 grid `[8,10,16]` 和320个
  visual placeholder，而不是未resize输入产生的1,200个 placeholder。
- vLLM 0.24视频 processor替换完整 marker triplet，HF只替换中间 `video_pad`。版本化
  adapter `qwen3_vl_preserve_hf_outer_video_markers_v1` 在送入 vLLM前增加一层 marker；
  CPU重建与真实 `RequestOutput.prompt_token_ids` 两道门禁都证明展开后459个 token与
  Prism/HF逐项相同。adapter、processor size和最终 token hash均写入合同；数量或
  marker变化时 fail closed。
- 对 `.venv-local/bin/python` 使用 `Path.resolve()` 会解引用 symlink并误启动系统
  Python，丢失冻结 OpenCV/FFmpeg。runner现保留 venv entrypoint；视频帧由 Prism环境
  解码成临时 lossless NPZ，vLLM消费后自动删除。首样本 sampled RGB identity为
  `d28fdd49eb3c9859b5568965b46ea0e7ac474f7c0babf8a82b30a9b862549531`。

最终 dirty smoke artifacts如下。每项均通过独立 validator：从 materialized JSONL重算
score、核对媒体/selection/run identity、实际 KV tensor bytes、backend evidence，并与
既有 Prism formal artifact中的同一样本 `input` 对象逐字段相等。MuirBench/MVBench
单样本 accuracy为0只说明各自首样本答错，不是整体质量结论。

| dataset | vLLM mode | sample | prompt tokens / visual tokens | quality | artifact SHA256 |
|---|---|---|---:|---|---|
| DocVQA | BF16 KV | `5437` | `611 / 575` | ANLS `1.0` | `7b9c8448175c0207af7d95db659d3f193dcb36ff12c6a9c8176ee70480efefa9` |
| DocVQA | per-token-head FP8 KV | `5437` | `611 / 575` | ANLS `1.0` | `3dd61d6e466785e0716363999c544b008b9ff0b2647508429d1fea42e07187ef` |
| MuirBench | per-token-head FP8 KV | `63` | `2381 / 2304` | official/strict `0/0` | `567bd80a4465a9dc2db8cefcf51719cfd4c2f53513986b5c94cf52f8753a8927` |
| MVBench | per-token-head FP8 KV | `counterfactual_inference\|video_12359.mp4\|150` | `459 / 320` | accuracy `0` | `e2914414483dc470c95511d771f0a2a20910f3fd8c93a8a6463b1172849cf04a` |

正式 development/final 矩阵于 2026-07-18 在 clean harness commit
`3ec90a504a31f17e58ccf5271411a6933dddf09b`（`git_dirty=false`）完成。12 个 vLLM
prediction artifact 全部来自同一 RTX 5090
`GPU-fa649184-cff6-76d2-cda0-328763ddd1ea`、driver `610.43.02`、vLLM `0.24.0`
distribution commit `gee0da84ab`、wheel RECORD
`a936c81ea72ecd7e1c51e35391d7e3f667595312687dbd6f99cecd57d6724e66` 与
Transformers `5.13.0`。evaluator、materialization manifest、模型 revision 和 16 个
关键执行文件哈希在所有 cell 中完全相同。六组 development sample arrays 与对应
final 前缀的 canonical SHA256 逐组相同；运行期间未修改 tracked file、协议、阈值或
样本选择。

下面每一行都先由独立 validator 从物化 reference 重算 score，再核对 selection、媒体、
prompt token、视频 sampled-RGB、实际 backend、KV tensor 和 run identity，最后运行
seed `20260717`、10,000 resamples 的 paired bootstrap。`validation_status=PASS` 表示
证据结构和重算通过，不等于质量 gate 的 `decision=PASS`；所有行都满足
`formal_evidence=true` 与 `semantic_input_exact=true`。

| dataset | subset | vLLM KV | Prism off → vLLM metric | delta / paired 95% CI | token exact | pool ratio | decision |
|---|---|---|---|---|---:|---:|---|
| DocVQA | development | BF16 | ANLS `0.9246396993 → 0.9248802921` | `+0.0002405927 / [-0.0070670996, +0.0073559774]` | 195/200 | `1.0x` | PASS |
| DocVQA | development | per-token-head FP8 | ANLS `0.9246396993 → 0.9259975082` | `+0.0013578089 / [-0.0058741259, +0.0085489510]` | 192/200 | `0.515625x` | PASS |
| DocVQA | final | BF16 | ANLS `0.9225580116 → 0.9212837223` | `-0.0012742892 / [-0.0082694043, +0.0054985779]` | 485/500 | `1.0x` | PASS |
| DocVQA | final | per-token-head FP8 | ANLS `0.9225580116 → 0.9195411351` | `-0.0030168765 / [-0.0090901632, +0.0018194872]` | 483/500 | `0.515625x` | PASS |
| MuirBench | development | BF16 | official `0.695 → 0.695`; strict `0.690 → 0.690` | official `0 / [-0.015, +0.015]`; strict `0 / [0, 0]` | 198/200 | `1.0x` | **FAIL (official)** |
| MuirBench | development | per-token-head FP8 | official `0.695 → 0.715`; strict `0.690 → 0.715` | official `+0.020 / [0, +0.045]`; strict `+0.025 / [0, +0.050]` | 190/200 | `0.515625x` | PASS |
| MuirBench | final | BF16 | official `0.654 → 0.650`; strict `0.646 → 0.644` | official `-0.004 / [-0.016, +0.008]`; strict `-0.002 / [-0.010, +0.004]` | 486/500 | `1.0x` | **FAIL (official)** |
| MuirBench | final | per-token-head FP8 | official `0.654 → 0.660`; strict `0.646 → 0.650` | official `+0.006 / [-0.006, +0.020]`; strict `+0.004 / [-0.008, +0.016]` | 476/500 | `0.515625x` | PASS |
| MVBench | development | BF16 | accuracy `0.6082474227 → 0.6082474227` | `0 / [0, 0]` | 97/97 | `1.0x` | PASS |
| MVBench | development | per-token-head FP8 | accuracy `0.6082474227 → 0.6185567010` | `+0.0103092784 / [-0.0206185567, +0.0515463918]` | 93/97 | `0.515625x` | **FAIL** |
| MVBench | final | BF16 | accuracy `0.6157894737 → 0.6210526316` | `+0.0052631579 / [0, +0.0157894737]` | 189/190 | `1.0x` | PASS |
| MVBench | final | per-token-head FP8 | accuracy `0.6157894737 → 0.6263157895` | `+0.0105263158 / [-0.0105263158, +0.0315789474]` | 184/190 | `0.515625x` | **FAIL** |

MuirBench BF16 的 formal FAIL 必须保留，但不能误读为 strict accuracy 大幅回退。
development/final 的 strict CI 分别为 `[0, 0]` 与 `[-0.010, +0.004]`，均通过；失败只
来自 official parser 的随机 fallback。两套框架出现不同数量的不可解析输出后，会以
不同节奏消费同一个 seed-42 RNG 序列，甚至使相同 raw output 得到不同随机字母。
development/final 的 fallback 数分别为 Prism/vLLM `4/3` 与 `15/13`。协议已预注册
official 与 strict 都为 required metric，因此没有在看到结果后删除 official gate；对外
叙述同时报告 formal FAIL、strict PASS 与 `97.2%` final token exact。

MVBench FP8 的 FAIL 不含随机 parser。final 共有 6 个 token 翻转，其中相对 Prism off
是 3 个改善、1 个回退、2 个同错，净多 2 个正确样本；但 paired CI 下界
`-1.05263158pp` 仍略低于预注册 `-1pp` margin，所以均值更高也不能改写为
non-inferiority PASS。作为非 headline 的归因诊断，直接以同一 vLLM BF16 为 baseline
时，FP8 final 为 184/190 token exact、净多 1 个正确样本，CI
`[-1.57894737pp, +2.63157895pp]`，同样 FAIL。这证明 formal FAIL 不是 Prism/vLLM
BF16 唯一分歧造成的。

与 P9-C.1 的同容量 Prism scaled-FP8 结果并列后，预注册结论清晰：两者 allocated KV
pool 都是 `778,567,680 / 1,509,949,440 = 0.515625x`，但 Prism scaled-FP8 在
DocVQA、MuirBench、MVBench development/final 六项全部 PASS，MVBench final 更是
190/190 token exact；vLLM per-token-head FP8 在 DocVQA/MuirBench PASS、MVBench
development/final FAIL。该结论是“同 logical capacity、同 allocated-KV-pool 比例下，
Prism 通过了全部预注册质量稳定性门禁”，不是“Prism accuracy 显著高于 vLLM”；vLLM
MVBench FP8 的点估计实际更高。

正式原始文件 SHA256：

| dataset/subset | vLLM BF16 prediction | BF16 comparison | vLLM FP8 prediction | FP8 comparison |
|---|---|---|---|---|
| DocVQA development | `a8d6af7869d3dda69041a3294f7849718d83399d7bc910e8218367bce50a1886` | `205e891cc0ed640afd7ea5ea4665325e770102531463037ee2cad9ebe822e9c5` | `ec832a79191896427a5d9e9de86ffa4df515488e8644b7437cb1eeb5a185d679` | `d63adcd379078dccd587d0d6284546280f60b0996f748ef57bd4912db7250925` |
| DocVQA final | `dfaa5f63a9aaea6ae6246342f0d6e2fc6b0aba43faa868c891540180a9710600` | `5899cb1a4237d75f5c8ebf80938b0e767d0fb9312e6ef3f2f3f0b34cbd59d39c` | `c3e7c57512b39b495df2f5e97fd83800ae17a0838865fb77c15816a2e836eee5` | `8e9b7a098adf554c5c6ab8cd3ad1ce73a724ba1ecf48f4909c33f75ff7848b85` |
| MuirBench development | `8f0473ebfd7f0fa70bd949488888e99251efd1407aabfa282697d4761f85ac08` | `11310c7052b5f8734fc8bb6dbf374a64d4cc3f4b373f2fdf4d092d1271945d6b` | `0c5b59ac512d81e699e985ffed9315040f99377c479d7754ca5b4bc6fc6be8e8` | `2f1bb1b4b362a7183bb2348eeba33492146c1249fca75611f22f66048056de6b` |
| MuirBench final | `2572c13ca0147321b1df69bf0995d635f2e1eb0313b70589f87100d5bdd10353` | `7a606e170bb1b5d59a9af69a3bb6aa55dd76aa308116807bf228616adfd9b66d` | `9695588080803e66cae233fde894c0027a0255fffded20e1d615bb71e10d1d7d` | `d28d46ed8ffe9f5c51d782cf6384bf824f778a00a112c86498440fc7de243475` |
| MVBench development | `b66932eb3c9dc1ef6bbf2cd6481ba3d2bd77a77bd3aab0ed6de9d3a0944811b2` | `7568460d4ef765a0a331d3fcd7dba321bd7340696d98a12387ed18e5b28fb7d4` | `915e39a0dd54bb10434db45c5061f7671217f2eff9fc940ab461b390fb047ba3` | `d63c051bb0fe4a2965ec5528f7e6e277e58fcfcdf41646e30b657bb1da8bc87b` |
| MVBench final | `4ce6b12f775a6f2484d0e82a46d86cf635518b4577b8b2ce58fb1bbed8cdb062` | `3013ebe5844596e3259123cc902ea2f9dd6ce2bc160fe05b0d24d37293fbf5de` | `8ec852db14f34ac0ad685faabdf3d6a74646eeab50c0d77a76ab4138c9661466` | `1b91927c645af04bcc5573017311e94a0ceebe9a1554a17f37ab0a6a8723886f` |

本检查点回归与静态门禁：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python -m pytest -q \
  tests/test_p9_*.py tests/test_scaled_fp8_kv_cache.py \
  tests/test_benchmark_schema.py tests/test_processor_pipeline.py \
  tests/test_processor_pipeline_multi_image.py \
  tests/test_processor_pipeline_video.py tests/test_http_range_reader.py \
  tests/test_llm_output_decoding.py
# 158 passed in 14.56s

/usr/bin/env --chdir=/data/Prism-Infer \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/Prism-Infer \
  /data/Prism-Infer/.venv-local/bin/python -m pytest -q
# 408 passed, 6 skipped in 264.68s

.venv-local/bin/python -m compileall -q prism_infer benchmarks scripts tests
.venv-local/bin/python -m black --check <12 changed Python files>
/data/vllm-omni/.venv/bin/ruff check --ignore E402 <12 changed Python files>
git diff --check
```

当前判定：External Gate A 的 runner/integration 为 `PASS`，冻结 formal matrix 已完成，
质量结果为 `MIXED`，不得汇总伪装成全 PASS。Prism scaled-FP8 的三数据集正式
non-inferiority 为全 PASS；同容量 vLLM per-token-head FP8 为 DocVQA/MuirBench PASS、
MVBench FAIL。完整物理显存 Pareto headline 继续阻塞，直到双方 page-table/allocator
metadata 采用统一可复核字节合同；本节也不包含 TTFT/TPOT、online scheduling、
Torch Compile/CUDA Graph 或 server SLO 结论。TP2/TP4 因当前租约只分配 GPU0，仍为
**NOT RUN / UNVERIFIED**。

### P9-D.0 Compiler/Graph 基线证据链（MECHANISM PASS / FORMAL PENDING，2026-07-20）

本检查点建立 P9-D 正式 profiling 前的 correctness、backend identity 和 fresh-process
证据链，不包含正式性能收益。当前物理设备是
`GPU-662a2fa1-37e4-cc52-0a51-27557dba315b / RTX 5090 32 GiB`；它与历史 P8/P9-A
formal UUID 不同，因此旧 latency/counter 不能与本轮组成 ratio。

本轮关闭的关键问题：

- full-model 测试端 `state_dict()` 浅引用、GPU logits 和未同步读数曾让已删除 Prism 模型
  显示 16.4 GiB allocated；修复 ownership 后纯文本/单图/多图/视频函数内均回到
  `0.0 GiB`，进程退出后 NVML 为 `1 MiB`；
- Transformers 5.13 的 HF reference forward 新增 `mm_token_type_ids` 合同；兼容逻辑集中在
  reference-only adapter，没有扩大 Prism production input；
- vision backend 曾由 flash-attn 包存在性和 segment shape 隐式选择，造成多图 full logits
  max diff `0.484375`；现在 startup 显式选择 `sdpa/flash_attn`，默认 SDPA，缺 capability
  fail closed。SDPA 下单图、多图、视频 full logits bit-exact；
- 原 benchmark 只做到 fresh model，不是 fresh process。新增标准库-only parent，每个
  mode/repeat 独立 child，按同一 UUID 执行 idle/release、ABBA/BAAB、逐字段 comparability
  和 process-level bootstrap CI；输出路径必须 gitignored且禁止覆盖；
- H1 traffic batch4 受 `max_vision_patches_per_batch=8192` 和 prefill/decode interleaving
  影响，短 smoke 的 actual decode histogram 是 `1:2 / 2:2 / 3:2`，不是静态 batch4。
  schema-v9 保存 actual histogram 和 Graph actual→captured 映射
  `1→1:2 / 2→2:2 / 3→4:2`，并要求 eager/Graph actual histogram exact；
- 四个 `test_full_model*.py` 原先只有 main block，组合 pytest 会被其他 item 掩盖成绿灯。
  现在四项都可被 pytest 收集，test 与 direct-script 复用同一 verification runner，任何
  非 PASS 都会失败。

H1 batch4 dirty diagnostic 使用正式 pool 配置但只运行 output4、warmup0、每 mode 1 个
fresh process。BF16 (`113` blocks / `4,265,607,168 B`) 和 scaled-FP8 (`220` blocks /
`4,282,122,240 B`) 的 eager/Graph 两组均为 15/15 comparability PASS、token exact、
prompt/image tokens `6472/6272`、active prompt blocks `28`；所有 child 前后均为
`1 MiB / 0%`。manifest 因 dirty、单 process、无 warmup 和 100 bootstrap resamples正确
标记 `formal_eligible=false`；其 latency ratio 全部丢弃，不进入性能结论。

完整候选 diff 门禁：

```bash
.venv-local/bin/ruff format --check prism_infer tests benchmarks scripts
.venv-local/bin/ruff check prism_infer tests benchmarks scripts
.venv-local/bin/ruff check --select C901,PLR0911,PLR0912,PLR0915 prism_infer
.venv-local/bin/ruff check --select S101,PLR2004 prism_infer
.venv-local/bin/python -m compileall -q prism_infer tests benchmarks scripts tools
git diff --check
# PASS; 188 Python files formatted

.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py tests/test_p9_process_matrix.py
# 60 passed

CUDA_VISIBLE_DEVICES=0 PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python -m pytest -q tests \
  --junitxml=data/p9_baseline/p9_prebaseline_full_regression_dirty.xml
# 466 passed, 1 skipped in 297.11s
# JUnit: 467 tests / 0 failures / 0 errors / 1 skipped
```

四项新可收集 full-logits gate 在同一完整 suite 外也单独串行复验：

```text
pure text:  [1,64,151936], max/mean diff 0 / 0
single:     [1,151936],    max/mean diff 0 / 0
multi:      [1,151936],    max/mean diff 0 / 0
video:      [1,151936],    max/mean diff 0 / 0
result: 4 passed in 39.60s
```

详细问题链、错误假设和两分钟面试讲法见 `docs/ISSUE_LOG.md` 的 P9-001–P9-008；
pipeline capture/compile/NSYS/NCU 执行顺序和止损规则见
`docs/P9_COMPILER_GRAPH_PLAYBOOK.md`。

当前判定：P9-D baseline mechanism `PASS`，正式性能 `PENDING`。只有提交后的 clean
commit 在当前 UUID 上完成 H1 BF16/scaled-FP8 batch1/4、output128、warmup2、每 mode 5
fresh processes、10,000 bootstrap resamples，并且 manifest
`formal_eligible=true`，结果才允许进入 NSYS/NCU 和后续 full-step Graph claim。

### P9-D.1 H1 BF16 batch4 Graph correctness 闭环与 formal 结果（PASS，2026-07-20）

commit `460d21a` 的第一次 clean formal cell 被 correctness gate 正确拒绝：五个 eager
process 与五个 Graph process 各自 deterministic，但 request 0 在生成 index 31 从 token
`2504` 分叉为 `448`。旧 artifact 保留为：

- `data/p9_baseline/h1_bf16_b4.jsonl`；
- `data/p9_baseline/h1_bf16_b4.manifest.json`，`status=failed_comparability`；
- `data/p9_baseline/h1_bf16_b4_runs/`。

fixed-history 逐 step 诊断把首个数值差异定位到 engine step 5：actual batch3 被 padding 到
captured batch4。active input/control 与 padding sentinel 全部正确；晚 admission、只经历
exact batch4 的 request 3 保持 logits exact，直到尾部第一次进入 3→4 才漂移。将 batch1–8
改为 exact capture 后，相同 4×128 trajectory 的 512 个完整 logits row 全部 bit-exact，
max diff 为 0。诊断 artifact 为：

- `data/p9_diagnostics/h1_bf16_b4_graph_fixed_trajectory_v1.json`；
- `data/p9_diagnostics/h1_bf16_b4_graph_fixed_trajectory_exact_small_v2.json`。

修复提交 `40466b693e30c35652a9d2e739c61d5ccf1df0e3` 的 clean formal artifact：

- `data/p9_baseline/h1_bf16_b4_exact_small_40466b6.jsonl`，SHA256
  `700dd64fa9a56602a252f8c39918b65286fb8c0acceeac71e4330f239201fc6d`；
- `data/p9_baseline/h1_bf16_b4_exact_small_40466b6.manifest.json`，SHA256
  `26e7c523fb009a6d95981240439ecf559df4bc37eb689d67661543dca87dbdb4`。

结果：`status=completed`、`formal_eligible=true`、15/15 comparability PASS、10/10 fresh
children PASS，所有 child 前后均为 `1 MiB / 0%`；eager/Graph output SHA256 均为
`a0f0cccd5699d11305c163bbbb20e6a9d50e82536a524cc760734cb7c57816b8`。Graph trajectory
为 `1→1:2 / 2→2:2 / 3→3:2 / 4→4:124`。

| H1 BF16 batch4 指标 | Eager median | Graph median | 改善与 process-bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| Decode step | 32.608 ms | 20.519 ms | 37.07% `[36.62%, 38.34%]` |
| Decode throughput | 119.492 tok/s | 190.432 tok/s | 59.37% `[58.38%, 62.47%]` |
| End-to-end | 5969.98 ms | 4348.14 ms | 27.17% `[25.14%, 28.52%]` |
| Engine output throughput | 98.095 tok/s | 139.585 tok/s | 42.30% `[38.85%, 47.90%]` |

engine TTFT 与 preprocessing-inclusive TTFT 的 CI 都跨零，因此不声明改善。Graph 的代价
是 peak allocated `+8.16 MiB`、reserved `+24 MiB`，五个 fresh-process capture time 为
`967.546–985.300 ms`；capture 是 startup 成本，不混入 synchronized request timing。

本轮完整 suite 为 `468 passed, 1 skipped in 300.37s`，JUnit 469 tests / 0 failures /
0 errors / 1 skipped；随后新增的 text-position/padding-audit CPU test 在 focused suite
通过。ruff、complexity/runtime assert/magic number、compileall、diff check 和 61 个本地
Markdown 链接均 PASS。

当前边界：H1 BF16 batch1 与 batch4 已 formal PASS，P9-008 Verified；batch9–15 等仍使用
sparse bucket，不能外推 token-exact。scaled-FP8 batch1/4 formal matrix 尚未运行，完整 P9-D
仍为 `IN PROGRESS`，但 BF16 correctness 不再阻塞下一 cell。

## 每次任务交付模板

阶段级交付使用 `docs/STAGE_DELIVERY_TEMPLATE.md`，其中包含 requirement mapping、
环境身份、correctness/quality/performance、claim边界和 raw evidence完整门禁。每个
较小任务完成时，在回复或阶段文档中至少使用:

```text
模块:
改动:
验证命令:
验证结果:
PASS/FAIL:
未验证风险:
下一步:
```
