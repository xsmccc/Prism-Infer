# P5 KV Cache Compression Report

> 修订日期: 2026-07-08
> 当前状态: P5.0 off baseline 和 P5.1 offline importance scoring 已完成；active pruning/quantization 未实现。

## 结论摘要

当前 P5 仍处在 compression strategy 的准备阶段:

- P5.0 已接入 `compression_mode="off"`、per-step `CompressionMetadata` 和 off-only guard。
- P5.1 已实现基于 P4 trace 的离线 visual-token importance scoring。
- 目前没有 active compression/pruning/quantization。
- 当前不能声明压缩率、显存收益、latency/throughput 收益或质量收益。
- 外部评估中提到的 FP8 KV、VScan/PoRe、DeepStack-aware pruning 和 M-RoPE block compaction 当前均未在 runtime 落地，不能作为已完成能力或已验证收益。

## P5.0 Off Baseline

实现位置:

- `prism_infer/engine/compression.py`
- `prism_infer/config.py`
- `prism_infer/utils/context.py`
- `prism_infer/engine/model_runner.py`
- `prism_infer/layers/attention.py`

设计:

- `compression_mode="off"` 是唯一支持模式。
- active mode 在配置校验或 attention guard 中显式失败。
- metadata 记录 step phase、batch size、prompt tokens、image/video visual token counts 和 block size。
- 不改变 K/V 写入、paged decode、SDPA 或 FlashAttention 路径。

验证记录:

```text
.venv-local/bin/python -m pytest -q tests/test_compression_off.py -s
4 passed in 0.10s
```

## P5.1 Visual Importance Scoring

实现位置:

- `prism_infer/analysis/visual_importance.py`
- `scripts/score_visual_tokens.py`
- `tests/test_visual_importance_scoring.py`

输入:

- P4 `kv_trace()` 生成的 JSONL records。
- 主要字段:
  - `attention.sequence_stats[].span_masses`
  - `attention.sequence_stats[].top_visual_tokens`
  - `attention.sequence_stats[].visual_attention_entropy_normalized_mean`
  - `span_stats`

输出:

- visual token ranking。
- visual span ranking。
- keep-ratio simulation。
- JSON report。
- Markdown report。
- limitations。

评分公式:

```text
score = token_attention_mass * (
    attention_mass_weight
    + entropy_focus_weight * (1 - visual_attention_entropy_normalized_mean)
    + k_norm_weight * visual_text_k_norm_ratio
)
```

默认权重:

```text
attention_mass_weight = 1.0
entropy_focus_weight = 0.5
k_norm_weight = 0.1
```

设计决策:

- 选择 attention mass 作为主信号。理由是 P4 trace 已直接记录当前 query 对 visual span/token 的关注度。
- 选择 visual entropy focus 作为放大项。理由是同样的 visual mass 下，集中到少数 visual tokens 的注意力更适合形成 pruning 候选。
- 选择 K norm ratio 作为弱信号。理由是 P4 报告中 visual/text K norm ratio 接近 1，单独用 K norm 不能稳定区分重要和冗余 token。
- 选择 offline scoring 而不是直接接入 runtime pruning。理由是 P5.1 的目标是先形成可审计 ranking，再进入 P5.2 做可回退的 active pruning correctness。

验证记录:

```text
.venv-local/bin/python -m pytest -q tests/test_visual_importance_scoring.py -s
4 passed in 1.38s
```

输出摘要:

```text
importance source layer records: 1
importance total visual tokens: 3
top token row: token_index=2, score_sum=0.7875
keep_ratio=0.5 -> keep_count=2, drop_count=1
P5.1 visual importance ranking: PASS
P5.1 visual importance modalities: PASS
P5.1 text-only importance empty result: PASS
P5.1 visual importance CLI: PASS
```

## 当前限制

- P5.1 不修改 runtime KV cache。
- P5.1 不实现 pruning mask。
- P5.1 不输出 compression ratio。
- P5.1 不输出显存、latency、throughput 或质量收益。
- P4 trace 未保存完整 per-token attention distribution；`top_visual_tokens` 只细化已记录 top-k token，未进入 top-k 的 visual tokens 使用 span mass 剩余量均分。
- P5.1 的 keep-ratio simulation 只是 ranking 层面的离线模拟，不代表真实 KV 删除后的模型质量。
- P5.1 的离线 ranking 与在线 pruning 存在 gap: 在线策略必须明确决策时机、block/page 粒度、M-RoPE position 语义和 DeepStack 注入后的 token 对齐边界。

## 候选路线边界

以下方向可作为 P5/P6 候选设计，但在实现和同条件验证前不得写入完成状态:

- FP8 KV cache: 需要实现 allocate/store/decode dequant 或 FP8 kernel 路径，并给出 round-trip、full decode、显存和 latency/throughput 数据。
- Visual pruning/retention: 需要基于 P5.1 score 或其他可解释信号产生 runtime decision，并与 FP baseline 对比 greedy/logits/ppl 或质量样例。
- Physical KV compaction: 需要证明 block mapping、context length、slot mapping、prefix/swap 状态和 M-RoPE position 语义一致。
- VScan/PoRe/DeepStack-aware pruning: 当前只可作为设计假设；必须先补源码、测试和 benchmark，再进入项目 claim。

## 下一步

P5.2 应实现首个可回退 logical visual token pruning/retention 策略:

- 从 P5.1 ranking 生成 retention mask。
- 保留 compression off FP reference。
- compression on 显式报告 token 保留率和 drop 决策。
- 明确不做或显式拒绝 physical compaction、FP8、VScan/PoRe 等未实现路径。
- 对比 greedy tokens、logits/ppl 或质量样例。
- 记录显存、latency 和 throughput。
- 未支持的压缩模式继续显式失败。
