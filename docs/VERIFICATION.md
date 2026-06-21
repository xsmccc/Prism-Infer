# Prism-Infer 验证标准

> 修订日期: 2026-06-21  
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
- 当前 strict PASS 仅覆盖纯文本 full logits。图文输入、视觉 token 替换、DeepStack 注入和端到端 generate 仍需在 P2 验证。
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
.venv-local/bin/python -m compileall prism_infer tests
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

计划新增或完善测试:

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_llm_vl_generate.py \
  tests/test_text_only_regression.py
```

PASS 标准:

- processor 输出的 `input_ids`、`pixel_values`、`image_grid_thw` 与参考一致。
- `prepare_prefill` 生成的 position_ids shape 和语义正确。
- 单图 greedy tokens 与 HF 一致。
- 纯文本请求不回归。

在这些测试文件实现前，P2 只能通过手动 smoke 验证，不能标记为完成。

## P3: KV Cache 分析验证

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

## P4: 压缩策略验证

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

## P5: Benchmark 验证

每个 benchmark 必须输出:

- 硬件型号、CUDA、torch、transformers、commit hash。
- 输入 shape、batch、seq_len、图像数量、compression config。
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

## P6: 交付验证

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
