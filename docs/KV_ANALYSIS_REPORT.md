# P4 KV Cache 分析报告

> 日期: 2026-06-26
> 状态: P4 当前门禁已通过
> 模型: `/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

## 结论摘要

P4 已建立 Prism-Infer repo-local KV trace 能力，并在三类多模态输入上完成 trace on/off greedy 一致性验证。
2026-07-04 补充: trace 记录新增 attention entropy 与 visual-token 条件 entropy，用作 P5 pruning scoring 的集中度指标。

当前实测结果:

| case | token ids | layer records | steps | phases | trace on/off |
|---|---|---:|---:|---|---|
| single_image_description | `[32, 6303]` | 72 | 2 | decode, prefill | PASS |
| single_image_detail_qa | `[2518, 151645]` | 72 | 2 | decode, prefill | PASS |
| multi_image_comparison | `[28715, 389]` | 72 | 2 | decode, prefill | PASS |

核心观察:

- 单图描述样例的 visual attention mass 均值约 `0.147857`，范围 `0.025376` 到 `0.317253`。
- 单图细节问答样例的 visual attention mass 均值约 `0.147504`，范围 `0.025493` 到 `0.361367`。
- 多图比较样例的 visual attention mass 均值约 `0.067564`，范围 `0.021200` 到 `0.175849`。
- 三个样例的 visual/text K norm ratio 均值分别约 `0.997039`、`0.987061`、`1.014112`，说明 K norm 本身不能单独作为删除依据。
- 三个样例的相邻层 visual K head cosine 均值约 `0.993`，部分中后层超过 `0.998`，可作为 P5/P6 研究层间冗余信号的候选证据。
- Attention entropy 已进入 trace schema；后续 P5 可结合 visual mass 与 entropy 区分“关注视觉但分散”和“集中关注少量视觉 token”的场景。

## 实现位置

- Trace schema/session: `prism_infer/analysis/kv_trace.py`
- Context metadata: `prism_infer/utils/context.py`
- ModelRunner metadata 注入: `prism_infer/engine/model_runner.py`
- Attention 层采集: `prism_infer/layers/attention.py`
- 离线分析脚本: `scripts/analyze_kv_trace.py`
- 样例运行脚本: `scripts/run_kv_trace_samples.py`
- 设计文档: `docs/P4_KV_TRACE_DESIGN.md`

## Trace Schema

JSONL 第一行是 `trace_header`，包含:

- `schema_version`
- `trace_config`
- `model_config`

后续每行是 `attention_layer`，包含:

- `step_id`
- `phase`
- `layer_id`
- `num_heads`
- `num_kv_heads`
- `head_dim`
- `batch.input_ids_shape`
- `batch.position_ids_shape`
- `batch.sequences[].image_grid_thw`
- `batch.sequences[].video_grid_thw`
- `batch.sequences[].spans`
- `tensor_stats.q/k/v/output`
- `span_stats`
- `head_stats`
- `attention.sequence_stats`
  - `attention_entropy_mean`
  - `attention_entropy_normalized_mean`
  - `visual_attention_entropy_normalized_mean`

## 验证命令

### 轻量验证

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m compileall prism_infer tests scripts && \
.venv-local/bin/python -m pytest -q \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_kv_trace_no_output_change.py -s
```

输出摘要:

```text
4 passed in 1.46s
trace off output shape: [1, 2, 4]
trace on output shape: [1, 2, 4]
trace output max diff: 0.000000e+00
trace output mean diff: 0.000000e+00
trace visual attention mass: 4.361440e-01
```

该结果证明:

- visual span 分组逻辑可用。
- trace metadata schema 可序列化。
- summary 能计算 visual attention、attention entropy、KV norm ratio、head 差异、层间冗余。
- 小张量 decode attention 中，trace on/off 输出完全一致。

### 三类真实样例

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python scripts/run_kv_trace_samples.py \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

输出摘要:

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

原始输出位于 gitignored 目录:

- `data/kv_trace_samples/manifest.json`
- `data/kv_trace_samples/*.jsonl`
- `data/kv_trace_samples/*.summary.json`
- `data/kv_trace_samples/*.summary.md`
- `data/kv_trace_samples/*.summary.svg`

这些文件不入库，原因是 `data/` 属于实验产物目录；复现命令已固定在本报告和 `docs/VERIFICATION.md`。

## 样例统计

| case | visual mass mean | visual mass range | K ratio mean | head mass std mean | adjacent cosine mean |
|---|---:|---:|---:|---:|---:|
| single_image_description | 0.147857 | 0.025376 - 0.317253 | 0.997039 | 0.144045 | 0.993047 |
| single_image_detail_qa | 0.147504 | 0.025493 - 0.361367 | 0.987061 | 0.137523 | 0.992861 |
| multi_image_comparison | 0.067564 | 0.021200 - 0.175849 | 1.014112 | 0.059833 | 0.993233 |

层级观察:

- 单图描述样例 visual mass 最高层为 layer 21，约 `0.317253`；最低层为 layer 7，约 `0.025376`。
- 单图细节问答样例 visual mass 最高层为 layer 18，约 `0.361367`；最低层为 layer 7，约 `0.025493`。
- 多图比较样例 visual mass 最高层为 layer 2，约 `0.175849`；最低层为 layer 7，约 `0.021200`。
- 三个样例均存在多个相邻层 visual K head cosine `>= 0.998` 的位置；这说明 visual KV 的 head-level norm 统计在部分相邻层高度相似，但它只是冗余候选信号，不等价于可以直接删除 KV。

## P5 压缩假设

基于当前 P4 数据，P5 第一版应采用保守、可回退、可对齐的策略:

1. 优先实现 compression off baseline，证明压缩开关关闭时与 FP baseline 完全一致。
2. 第一种 active 策略建议是 visual token importance pruning，而不是直接固定比例裁剪。
3. importance score 初版可组合:
   - 当前 query 对 visual token 的 attention mass。
   - 当前 query 的 attention entropy，以及 visual token 内部条件 entropy。
   - visual token 是否进入 top-k token importance。
   - layer/head 的 visual mass 分布，避免忽略少数 head 的高关注视觉 token。
4. 不建议只用 K norm 做裁剪依据，因为三个样例的 visual/text K norm ratio 均值接近 1，单独用 norm 无法稳定区分重要和冗余 token。
5. 层间冗余可以作为第二阶段研究方向: 当相邻层 visual K head cosine 长期接近 1 时，考虑层间共享 compression mask 或复用 importance score，但不能直接跳过层或删除整层 KV。

P5 的最小可验证出口应包含:

- compression off greedy tokens 与 baseline 完全一致。
- compression on 输出压缩率、token 一致率、logits/ppl diff、显存变化和 latency 变化。
- 失败路径显式报错，不 silent fallback。
- 至少覆盖单图描述、细节问答、多图或长上下文中的三类输入。

## 剩余风险

- Trace 是分析路径，会增加同步和 JSON 序列化开销，不用于性能 benchmark 数字。
- 当前 P4 样例覆盖两类单图和一类多图；视频 trace 未纳入 P4 最小门禁，但 P3 已验证视频 correctness。
- 当前 attention mass 是 trace 模式下重算的分析指标，不参与模型输出。
- P4 只形成压缩假设，不代表 P5 压缩策略已经实现或有收益。
