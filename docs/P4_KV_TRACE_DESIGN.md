# P4 KV Cache Trace 设计

> 日期: 2026-06-26
> 状态: Implemented, verification in progress
> 目标: 在不改变 Prism-Infer 正常推理输出的前提下，捕获视觉 token 的 KV/attention 行为，为 P5 压缩策略提供证据。

## 设计目标

P4 先解决“看得见”的问题，再进入 P5 压缩。trace 系统必须满足:

- 默认关闭，不影响普通推理路径。
- 开启后不改变 greedy 输出。
- trace 文件可复现，包含模型配置、输入 shape、image/video grid、layer/head、token span 和统计量。
- 能区分 text/image/video token span。
- 能输出 visual attention mass、token importance、层间冗余和 head 差异的离线统计。

## 接入位置

### `prism_infer/analysis/kv_trace.py`

新增独立分析模块，负责:

- `TraceConfig`: trace 开关、输出路径、top-k token 等配置。
- `TokenSpan`: text/image/video 连续 token 区间。
- `SequenceTraceInfo`: 单条请求的 prompt 长度、query 区间、block table、grid 和 span。
- `TraceMetadata`: batch 级 metadata，挂到 `Context.trace_metadata`。
- `kv_trace(...)`: 显式开启 trace 的上下文管理器。
- `record_attention_layer(...)`: 记录每层 Q/K/V、span KV norm、attention mass 和 top token。
- `summarize_trace(...)`: 离线汇总 visual/text KV norm、attention mass、head 差异和层间相似性。
- `render_summary_svg(...)`: 无额外依赖生成 SVG 图，展示 layer-wise visual attention mass 和 visual/text K norm ratio。

### `prism_infer/utils/context.py`

`Context` 新增 `trace_metadata` 字段，默认 `None`。普通推理不创建 metadata，attention 层快速返回。

### `prism_infer/engine/model_runner.py`

- `prepare_prefill` 和 `prepare_decode` 在 trace 开启时构造 `TraceMetadata`。
- 从 `Sequence` 读取 `token_ids`、`image_token_id`、`video_token_id`、`image_grid_thw`、`video_grid_thw`。
- `allocate_kv_cache` 给每个 attention module 写入 `layer_idx`，保证记录能定位到层号。
- `register_model_config` 在初始化和 prepare 阶段都调用，保证“先建 LLM，后开 trace”也能写出模型配置。

### `prism_infer/layers/attention.py`

在原 attention 输出 `o` 计算完成后调用 `record_attention_layer`。采集逻辑只读 detached tensor，不参与模型输出计算。

## Trace Schema

JSONL 第一行:

```json
{
  "record_type": "trace_header",
  "schema_version": 1,
  "trace_config": {},
  "model_config": {}
}
```

后续每行:

```json
{
  "record_type": "attention_layer",
  "schema_version": 1,
  "step_id": 0,
  "phase": "prefill",
  "layer_id": 0,
  "num_heads": 8,
  "num_kv_heads": 1,
  "head_dim": 128,
  "batch": {
    "input_ids_shape": [210],
    "position_ids_shape": [3, 210],
    "sequences": [
      {
        "seq_id": 0,
        "prompt_len": 210,
        "total_len": 210,
        "image_grid_thw": [[1, 28, 28]],
        "spans": [
          {"modality": "text", "start": 0, "end": 14},
          {"modality": "image", "start": 14, "end": 210}
        ]
      }
    ]
  },
  "tensor_stats": {
    "q": {"shape": [210, 8, 128], "mean": 0.0, "std": 1.0},
    "k": {"shape": [210, 1, 128], "mean": 0.0, "std": 1.0},
    "v": {"shape": [210, 1, 128], "mean": 0.0, "std": 1.0}
  },
  "span_stats": [],
  "attention": {
    "kind": "prefill_last_query",
    "sequence_stats": []
  }
}
```

## 指标定义

- `visual_attention_mass`: 当前 query 对 image/video span 的 attention probability 之和。
- `token_importance`: 当前 query attention probability 的 token 级 top-k。
- `visual_k_norm_mean`: visual span 中 K 向量的 head/token 平均 norm。
- `visual_text_k_norm_ratio`: visual K norm 与 text K norm 的比值。
- `visual_head_mass_std`: 不同 attention head 对 visual token 关注度的标准差，衡量 head 差异。
- `adjacent_layer_redundancy.visual_k_head_cosine`: 相邻层 visual K head norm 向量 cosine，相似度高说明可作为层间冗余候选信号。

## 当前边界

- 当前 trace 主要用于 eager 分析路径；P3 的 CUDA Graph decode correctness 不回归，但 graph replay 内部不采集逐层 trace。原因是 CUDA Graph 捕获期间嵌入 Python side-effect 会破坏 graph 使用模型。
- Prefill attention mass 记录 last query 对当前 prompt keys 的精确 attention。Decode attention mass 通过 paged KV cache 和 block table 读取历史 keys 后计算当前 query attention。
- trace 统计会带来额外 CPU/GPU 同步和 JSON 序列化开销，不能用于性能 benchmark 数字。
- P4 只输出分析和压缩假设，不直接修改 KV cache 或生成结果。

## 验证门禁

轻量门禁:

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_kv_trace_no_output_change.py -s
```

真实样例门禁:

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python scripts/run_kv_trace_samples.py \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

PASS 标准:

- 轻量测试全部通过。
- 三类样例 trace off/on greedy token ids 完全一致。
- 每个样例生成 JSONL、summary JSON、summary Markdown。
- 每个样例生成 summary SVG 可视化。
- `docs/KV_ANALYSIS_REPORT.md` 引用真实 trace 输出，并明确 P5 压缩假设。
