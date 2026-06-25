# Prism-Infer 验证标准

> 修订日期: 2026-06-25
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
- 2026-06-25 完成 P3.7 后，P3 当前门禁已通过: grouped regression `49 passed in 356.34s`，纯文本/单图/多图/视频 full logits 均 strict PASS，VL CUDA Graph decode 与 paged decode kernel 均有 correctness 和 benchmark 基线。
- GPU 不可用时可以降级为“未验证风险”，但不能把缺失验证写成通过。

## PASS 标准

| 类型 | PASS 标准 |
|---|---|
| 语法检查 | `compileall` 无错误 |
| 同精度模块对齐 | max diff `< 1e-5` |
| 跨精度模块对齐 | max diff `< 1e-2` |
| Full logits | max diff `< 1e-2` 且无 NaN；更严格目标按对应测试定义执行 |
| Greedy 端到端 | `temperature=0` 输出 token 完全一致 |
| 采样模式 | logits 分布或 perplexity 对齐，ppl diff `< 0.1` |
| 压缩策略 | compression off 等价 baseline；compression on 给出压缩率、质量退化和性能/显存数据 |
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
.venv-local/bin/python tests/test_full_model_layerwise_debug.py
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python tests/test_attention_micro_debug.py
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
.venv-local/bin/python tests/test_full_model_vl_layerwise_debug.py
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
- 每条请求在 mixed batch 中的 logits 或 greedy token ids 与单请求独立运行一致。
- `slot_mapping/block_tables/context_lens` 不串扰，KV cache 写入/读取 shape 和 max diff 有输出。
- 不支持 prefix-cache/chunked prefill 的组合必须显式报错，不能 silent fallback。

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

- mixed 公开入口:

```text
1 passed in 33.67s
single token_ids: [[11], [785], [785], [785]]
mixed token_ids: [[11], [785], [785], [785]]
LLM.generate_mixed mixed batch single-run equivalence: PASS
```

- 当前 P3.3 不覆盖 prefix-cache/chunked-prefill VL mixed batch；该组合仍作为后续风险，不并入 P3.3 PASS。

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
- greedy `max_tokens=8/16/32` token ids 与 HF 完全一致，或明确记录首个分叉 token、该步 logits max diff/mean diff。
- 分布测试输出 logits shape、mean/std、max diff、mean diff、perplexity 或等价指标，采样模式 ppl diff `< 0.1`。
- 长输出测试必须记录 prompt token 数、image/video token 数、generated token 数、EOS 状态和总耗时。

当前状态:

- 2026-06-25 已验证 `max_tokens=8/16/32` 与 logits/ppl PASS。
- HF greedy 长输出:

```text
1 passed in 26.76s
single-image prompt tokens: 210
single-image prefix@8 match: True
single-image prefix@16 match: True
single-image prefix@32 match: True
single-image first mismatch: None
multi-image prompt tokens: 408
multi-image prefix@8 match: True
multi-image prefix@16 match: True
multi-image prefix@32 match: True
multi-image first mismatch: None
video prompt tokens: 420
video prefix@8 match: True
video prefix@16 match: True
video prefix@32 match: True
video first mismatch: None
```

- mixed batch 长输出:

```text
mixed text prefix@8 match: True
LLM.generate_mixed VL rows mixed batch 32-token equivalence: PASS
```

- logits/ppl 分布:

```text
single-image logits shape HF/Prism: [1, 32, 151936]
single-image logits max diff: 0.000000e+00
single-image logits mean diff: 0.000000e+00
single-image ppl diff: 0.000000e+00
multi-image logits shape HF/Prism: [1, 32, 151936]
multi-image logits max diff: 0.000000e+00
multi-image logits mean diff: 0.000000e+00
multi-image ppl diff: 0.000000e+00
video logits shape HF/Prism: [1, 32, 151936]
video logits max diff: 0.000000e+00
video logits mean diff: 0.000000e+00
video ppl diff: 0.000000e+00
```

- text-only mixed batch numeric sensitivity:

```text
HF duplicate batch max diff: 5.312500e-01
HF duplicate batch mean diff: 1.473503e-01
Prism duplicate batch max diff: 5.312500e-01
Prism duplicate batch mean diff: 1.473503e-01
HF/Prism duplicate batch numeric sensitivity: PASS
```

- 剩余风险: 随机采样文本一致性不作为 PASS 标准；长上下文压力、prefix-cache/chunked-prefill VL mixed batch 仍未完成。

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
  - prefix-cache/chunked-prefill VL mixed batch 仍未支持。
  - P3.6 kernel 是 baseline kernel；batch=1/context=4096 慢于 SDPA reference。
  - 真实视频文件采样策略、多卡 TP、4070/4090 benchmark 和 vLLM/SGLang 同条件对比未在 P3 完成。

## P4: KV Cache 分析验证

计划新增测试:

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_kv_trace_no_output_change.py \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py
```

PASS 标准:

- trace on/off 的 greedy 输出一致。
- trace 文件 schema 稳定，字段完整。
- visual/text token 区间划分正确。
- 分析脚本能生成至少一份可复现报告。

## P5: 压缩策略验证

计划新增测试:

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_compression_off_equals_baseline.py \
  tests/test_visual_token_pruning_shapes.py \
  tests/test_compression_no_silent_fallback.py \
  tests/test_compression_quality_regression.py
```

PASS 标准:

- compression off 与 FP baseline 完全一致。
- compression on 的 KV shape、block mapping 和 decode 状态一致。
- 失败路径显式报错，不 silent fallback。
- 输出压缩率、质量退化、显存、latency 或 throughput 数据。

## P6: Benchmark 验证

每个 benchmark 必须输出:

- 硬件型号、CUDA、torch、transformers、commit hash。
- 输入 shape、batch、seq_len、图像数量、compression config。
- 对比 vLLM/SGLang 时必须记录对方版本或 commit、启动参数、调度参数、dtype、max model len、显存利用率、CUDA Graph 或 equivalent 设置。
- warmup 次数和 repeat 次数。
- `torch.cuda.synchronize()` timing 边界。
- GPU memory allocated/reserved/peak。
- latency median、p90、min、max。
- throughput 或 token/s。

禁止:

- 只报 mean。
- 用估算数字代替实测。
- 混用不同输入条件做优化前后对比。
- 在未验证 correctness 的 kernel 上报告性能收益。
- 在不同输入集合、不同采样配置或不同显存限制下声称吞吐超越。

## P7: 交付验证

交付前必须检查:

```bash
git status --short
```

并完成:

- README 最小 demo 命令可执行。
- 文档中的关键数字都有日志或测试输出来源。
- Known Issues 包含 full logits、端到端、压缩策略、性能测试中的未完成项。
- 新环境复现步骤不依赖口头说明。

## 每次任务交付模板

每个任务完成时，在回复或阶段文档中使用:

```text
模块:
改动:
验证命令:
验证结果:
PASS/FAIL:
未验证风险:
下一步:
```
