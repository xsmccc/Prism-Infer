# P5 KV Cache Compression Report

> 修订日期: 2026-07-09
> 当前状态: P5 当前门禁已完成。P5.0 off baseline、P5.1 offline importance scoring、P5.2 active logical visual pruning 和 P5.3/P5.4 FP8 KV baseline 均有实现与验证；physical visual-token compaction 尚未实现。

> 历史快照：本文件冻结 P5 当时的状态，不代表当前主线能力。physical compaction 已在
> P6 完成；P9-C 又新增并正式验证 `scaled_fp8_kv`。当前结论以
> [CLAIMS](CLAIMS.md)、[VERIFICATION](VERIFICATION.md) P9-C 和
> [KNOWN_ISSUES](KNOWN_ISSUES.md) 为准，旧 `fp8_kv` 仍是 unit-scale rejected baseline。

## 结论摘要

当前 P5 已完成当前门禁，核心结论如下:

- P5.0 已接入 `compression_mode="off"`、per-step `CompressionMetadata` 和 off-only guard。
- P5.1 已实现基于 P4 trace 的离线 visual-token importance scoring。
- P5.2-A 已新增 visual-token pruning decision helper 和 runtime shadow metadata，用于生成可审计 keep/drop record。
- P5.2 已新增 `compression_mode="visual_prune"` active logical pruning:
  - prefill 阶段生成 pruning decision 并持久化到 `Sequence`。
  - decode 阶段按 retained visual tokens 构造 compact KV view。
  - 不做 physical KV compaction，不减少已分配 KV cache block。
- 单图 smoke 中 `keep_ratio=0.5` 将 visual tokens 从 `196` 逻辑保留为 `98`，8-token greedy 与 off baseline 完全一致。
- 小型端到端 benchmark 中 active logical pruning 当前更慢: off median `0.292072s`，active median `0.798550s`。因此当前不能声明 latency/throughput 收益。
- 显存数据接近，且没有 physical compaction；当前不能声明物理显存收益。
- P5.3/P5.4 已新增 `compression_mode="fp8_kv"`:
  - 同样 16 个 KV blocks 下，BF16 KV cache bytes `603979776`，FP8 KV cache bytes `301989888`，ratio `0.5`。
  - quality matrix 覆盖 text/single-image/multi-image/video，`fp8_kv` 对 off 为 `32/32` token exact match。
  - single-image benchmark 中 GPU allocated/reserved/peak median 从 `17319.02/19938.00/19698.23 MB` 降到 `17023.57/19626.00/19402.86 MB`。
  - 当前 FP8 decode latency 更慢: off median `0.278173s`，fp8 median `0.704317s`；不能声明吞吐收益。
- 外部评估中提到的 VScan/PoRe、DeepStack-aware pruning 和 M-RoPE block compaction 当前均未在 runtime 落地，不能作为已完成能力或已验证收益。

## P5.0 Off Baseline

实现位置:

- `prism_infer/engine/compression.py`
- `prism_infer/config.py`
- `prism_infer/utils/context.py`
- `prism_infer/engine/model_runner.py`
- `prism_infer/layers/attention.py`

设计:

- `compression_mode="off"` 是 FP baseline。
- `compression_mode="visual_prune"` 是 logical visual token retention mode。
- `compression_mode="fp8_kv"` 是 physical KV storage baseline；其他未实现 mode 在配置校验或 attention guard 中显式失败。
- metadata 记录 step phase、batch size、prompt tokens、image/video visual token counts 和 block size。
- 不改变 K/V 写入、paged decode、SDPA 或 FlashAttention 路径。

验证记录:

```text
.venv-local/bin/python -m pytest -q tests/test_compression_off.py -s
9 passed in 0.10s
```

```text
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
.venv-local/bin/python -m pytest -q tests/test_text_only_regression.py -s
1 passed in 17.02s
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

## P5.2 Decision Preflight

实现位置:

- `prism_infer/engine/visual_pruning.py`
- `tests/test_visual_pruning.py`

当前能力:

- 扫描一条 sequence 内的 image/video visual-token spans，不假设所有 visual tokens 只形成一个连续段。
- 支持 `uniform` retention decision。
- 支持显式传入 per-token score 的 `score` retention decision；缺少 score 时直接报错，不 fallback 到 uniform。
- 输出 `PruningDecision.to_record()`，包含 visual token 总数、保留数、丢弃数、keep ratio、strategy、span 列表、kept/dropped token indices 和 `physical_compaction=False`。
- 可生成实验性的 prefill `slot_mapping` mask，但该 mask 不是完整 active compression 实现。

设计边界:

- P5.2 preflight 阶段先选择 decision helper，而不是直接开放 `compression_mode="visual_prune"`。理由是当时 decode attention 仍按完整 `context_len` 和 block table 读取 KV；没有 retention-aware decode 或 physical compaction 前，开放 runtime mode 会形成 silent no-op 或 KV 空洞。
- 选择显式 `score` 输入而不是在 helper 内读取 P5.1 JSON。理由是 P5.1 ranking 到 runtime token index 的映射仍需单独验证；helper 先固定 decision contract。
- 拒绝 `strategy="importance"` fallback。理由是 importance strategy 尚未和 P5.1 score/runtime span 对齐，fallback 会把未实现策略伪装成可用策略。

验证记录:

```text
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning.py -s
15 passed in 0.17s
```

## P5.2-A Runtime Shadow Mode

实现位置:

- `prism_infer/config.py`
- `prism_infer/engine/compression.py`
- `prism_infer/engine/visual_pruning.py`
- `tests/test_compression_off.py`

当前能力:

- `Config` 新增 `enable_visual_pruning_shadow`、`visual_pruning_keep_ratio`、`visual_pruning_min_keep_tokens` 和 `visual_pruning_strategy`。
- `build_compression_metadata(...)` 在 prefill 阶段生成 visual pruning decision records，并写入 `CompressionMetadata.visual_pruning_decision_records`。
- `CompressionMetadata.enabled` 仍只由 `compression_mode` 决定；shadow mode 下仍为 `False`。
- decode metadata 不重算 pruning decision，避免 decode 阶段只有 last token 或缺少完整 `token_ids` 时产生错误决策。
- attention path 对 off shadow records 保持 no-op；携带 shadow records 不改变 attention 输出。

设计边界:

- P5.2-A 选择 shadow mode 而不是直接启用 `compression_mode="visual_prune"`。理由是当时 decode attention 仍按完整 `context_len` 和 block table 读 KV，尚未支持 retained-token layout。
- 选择 prefill-only decision record。理由是 visual spans 和 prompt visual token index 在 prefill 阶段完整可见；decode 阶段只应消费已验证的状态，不应重新推断 visual spans。
- `score` strategy 在 runtime shadow 中缺少 token score 时显式失败。理由是 P5.1 offline score 到 runtime token index 的映射还未完成，不能 fallback 到 uniform。

验证记录:

```text
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning.py -s
15 passed in 0.17s
```

## P5.2 Active Logical Visual Pruning

实现位置:

- `prism_infer/config.py`
- `prism_infer/engine/compression.py`
- `prism_infer/engine/sequence.py`
- `prism_infer/engine/visual_pruning.py`
- `prism_infer/layers/attention.py`
- `prism_infer/engine/model_runner.py`
- `tests/test_visual_pruning_active.py`
- `tests/test_compression_off.py`

当前能力:

- `compression_mode="visual_prune"` 已通过 config validation。
- prefill 阶段生成 batch-aligned pruning decision records，并把 active record 持久化到 `Sequence.visual_pruning_decision_record`。
- `Sequence.__getstate__/__setstate__` 会保留 pruning record，避免 decode 跨进程序列化时丢失状态。
- decode 阶段从 `Sequence` 恢复 record，写入 `CompressionMetadata.visual_pruning_records_by_batch`。
- active decode 强制走 retained-aware eager path:
  - 不使用当前连续 context 的 Triton paged decode kernel。
  - 不走 CUDA Graph replay，避免 graph 绕过 Python attention 分支形成 silent no-op。
  - 直接从 paged KV cache 收集 retained logical positions，形成 compact K/V view 后运行 SDPA。
- `compression_mode="off"` baseline 保持原路径；keep-all active path 与 off path focused test exact match。
- 缺少 active prefill decision record 时显式失败，不 fallback 到 uncompressed。

设计决策:

- 选择 logical decode pruning，而不是立即做 physical KV compaction。理由是当前 block size 为 `256`，单图 `196` visual tokens 常落在同一物理 block 内；没有重新设计 context length、slot mapping、block table 和 free block 生命周期前，物理 compaction 容易破坏 decode 状态。
- 选择 prefill 全量写 KV。理由是 prompt prefill logits 仍需要完整上下文，先保证 FP baseline 和第一个生成 token 不受影响。
- 选择 decode retained-token view。理由是 decode query 可以在读取历史 KV 时忽略 dropped visual prompt positions，从而先验证真实 pruning 对生成质量的影响。
- 选择 active mode 绕过 CUDA Graph。理由是 CUDA Graph replay 不重新进入 Python attention 分支，不能消费新的 retention metadata。
- 拒绝 `score` runtime fallback。理由是 P5.1 offline score 与 runtime token index 的映射还未验证，当前 active mode 只开放 `uniform` 或显式传 score 的 helper 级能力。

Focused verification:

```text
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_visual_pruning_active.py \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning.py -s
20 passed in 0.19s
```

关键输出:

```text
active prune q shape: [1, 4, 8]
active prune k_cache shape: [2, 4, 2, 8]
active prune output shape: [1, 4, 8]
active prune reference shape: [1, 4, 8]
active prune output mean/std: 1.249826e-01/6.917473e-01
active prune reference mean/std: 1.249826e-01/6.917473e-01
active prune max diff: 0.000000e+00
keep-all max diff: 0.000000e+00
```

真实模型 smoke:

```text
off token_ids: [785, 2168]
active keep-all token_ids: [785, 2168]
token exact match: True
visual_prune keep-all LLM.generate_vl smoke: PASS
```

```text
off token_ids: [785, 2168, 3897, 374, 264, 6437, 11, 13794]
active keep-ratio=0.5 token_ids: [785, 2168, 3897, 374, 264, 6437, 11, 13794]
matched prefix tokens: 8/8
token exact match: True
```

同一 processor 输入的 decision record:

```text
prompt tokens: 210
visual tokens total: 196
visual tokens kept: 98
visual tokens dropped: 98
actual keep ratio: 0.500000
physical compaction: False
```

小型端到端 benchmark:

```text
gpu: NVIDIA GeForce RTX 5090
warmup: 1, repeat: 3
input: single_image 448x448, prompt='Describe this image.', max_tokens=8
```

| mode | latency median/p90/min/max (s) | output token/s median | allocated/reserved/peak median (MB) |
|---|---:|---:|---:|
| off | `0.292072 / 0.294718 / 0.284047 / 0.294718` | `27.390514` | `25743.02 / 28348.00 / 28122.23` |
| visual_prune keep_ratio=0.5 | `0.798550 / 0.810786 / 0.781913 / 0.810786` | `10.018163` | `25707.93 / 28312.00 / 28087.13` |

解释:

- `visual_prune keep_ratio=0.5` 在该样例上有 logical compression ratio: visual tokens `196 -> 98`。
- 质量 smoke: 8-token greedy 与 off baseline 完全一致；这不是完整质量评测。
- active logical pruning 当前更慢，不能声明 latency/throughput 收益。
- 没有 physical KV compaction；显存 allocated/reserved/peak 接近，不能声明物理显存收益。

## P5.3/P5.4 FP8 KV Baseline

实现位置:

- `prism_infer/config.py`
- `prism_infer/engine/compression.py`
- `prism_infer/engine/model_runner.py`
- `prism_infer/layers/attention.py`
- `tests/test_fp8_kv_cache.py`
- `benchmarks/bench_kv_compression.py`

当前能力:

- `compression_mode="fp8_kv"` 已通过 config validation。
- `ModelRunner.allocate_kv_cache()` 在 `fp8_kv` 下把物理 KV cache dtype 设为 `torch.float8_e4m3fn`。
- `num_kvcache_blocks > 0` 时使用固定容量分配；未指定时保持原有“按显存预算最大化 blocks”的语义。
- FP8 KV store 不走当前 Triton store，改走 PyTorch fallback，把 K/V 显式转换为 FP8 后写入 cache。
- FP8 decode 不走当前 BF16 Triton paged decode，也不走 CUDA Graph replay；decode 读取 paged KV 后 dequant 到 query dtype，再运行 SDPA。
- off baseline、visual_prune 和 unsupported-mode hard fail 均保留。

设计决策:

- 选择 FP8 KV baseline，而不是在 P5.3 立即实现 visual-token physical compaction。理由是 visual-token compaction 需要重新定义 compressed context length、slot mapping、block table、free block 回收和 prefix/swap 状态；FP8 KV 可以先在不改变 logical context 的情况下提供真实物理 KV bytes 收益。
- 选择固定 `num_kvcache_blocks` benchmark 来证明显存收益。理由是默认 allocation policy 会用节省出的 bytes 分配更多 blocks；这体现容量收益，但不体现 allocated memory 下降。固定 blocks 能同条件比较每个 mode 的 KV bytes 和 GPU memory。
- 选择 eager dequant + SDPA。理由是先保证 correctness；FP8-aware Triton decode kernel 属于 P6 优化，不在 P5 中伪装成已完成吞吐优化。

Focused verification:

```text
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_fp8_kv_cache.py \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_pruning_active.py -s
16 passed in 0.23s
```

关键输出:

```text
fp8 store key shape: [5, 2, 8]
fp8 store cache shape: [2, 4, 2, 8]
bf16 cache bytes for one tensor: 256
fp8 cache bytes for one tensor: 128
fp8 store k roundtrip max diff: 0.000000e+00
fp8 store v roundtrip max diff: 0.000000e+00
fp8 decode output shape: [1, 4, 8]
fp8 decode reference shape: [1, 4, 8]
fp8 decode max diff: 0.000000e+00
```

受影响窄回归:

```text
PYTHONPATH=/data/Prism-Infer /data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py \
  /data/Prism-Infer/tests/test_kv_engine_hardening.py \
  /data/Prism-Infer/tests/test_kv_trace_no_output_change.py -s
11 passed in 4.15s
```

真实模型 fixed-block smoke:

```text
case: off
kv_cache dtype: torch.bfloat16
kv_cache shape: [2, 36, 16, 256, 8, 128]
kv_cache bytes: 603979776
token_ids: [785, 2168, 3897, 374]
case: fp8_kv
kv_cache dtype: torch.float8_e4m3fn
kv_cache shape: [2, 36, 16, 256, 8, 128]
kv_cache bytes: 301989888
token_ids: [785, 2168, 3897, 374]
token exact match: True
kv byte ratio fp8/off: 0.500000
```

可复现 benchmark:

```text
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

single-image benchmark:

| mode | KV bytes | latency median/p90/min/max (s) | token/s median | allocated/reserved/peak median (MB) |
|---|---:|---:|---:|---:|
| off | `603979776` | `0.278173 / 0.287490 / 0.274921 / 0.287490` | `28.759042` | `17319.02 / 19938.00 / 19698.23` |
| fp8_kv | `301989888` | `0.704317 / 0.704871 / 0.692049 / 0.704871` | `11.358516` | `17023.57 / 19626.00 / 19402.86` |

quality matrix:

```text
text: 8/8 exact
single_image: 8/8 exact
multi_image: 8/8 exact
video: 8/8 exact
aggregate token match: 32/32
kv byte ratio fp8/off: 0.500000
```

解释:

- `fp8_kv` 的物理 KV cache bytes 为 off 的 `0.5x`。
- 固定 16 blocks 下，GPU allocated/reserved/peak median 分别下降约 `295.45/312.00/295.37 MB`。
- 当前质量矩阵没有 token 退化，退化为 `0/32` tokens changed；这只是 P5 fixed input set，不代表所有输入。
- 当前 latency/throughput 退化明显，原因是 P5 使用 eager FP8 dequant + SDPA，未实现 FP8-aware paged decode kernel。吞吐优化进入 P6。

## 当前限制

- P5.2 active logical pruning 不修改 physical KV cache allocation。
- P5.2 active logical pruning 只改变 decode attention 读取的 K/V view，不回收 block，不改变 block table，也不减少 `num_kvcache_blocks`。
- `fp8_kv` 当前只覆盖 fixed synthetic text/image/video input set；还没有覆盖更大真实数据集、多 seed、采样分布或人工质量评测。
- `fp8_kv` 当前比 off 慢，主要风险来自 eager FP8 dequant + SDPA 无 Triton kernel；需要后续优化。
- P4 trace 未保存完整 per-token attention distribution；`top_visual_tokens` 只细化已记录 top-k token，未进入 top-k 的 visual tokens 使用 span mass 剩余量均分。
- P5.1 的 keep-ratio simulation 只是 ranking 层面的离线模拟，不代表真实 KV 删除后的模型质量。
- P5.1 的离线 ranking 与在线 pruning 存在 gap: 在线策略必须明确决策时机、block/page 粒度、M-RoPE position 语义和 DeepStack 注入后的 token 对齐边界。

## 候选路线边界

以下方向可作为 P5/P6 候选设计，但在实现和同条件验证前不得写入完成状态:

- FP8 KV cache: 已完成 allocate/store/decode dequant baseline；FP8-aware Triton decode kernel 属于 P6。
- Visual pruning/retention: 已有 uniform logical runtime baseline；下一步需要接入已验证的 score source、扩大质量评测，并优化 retained-aware decode kernel。
- Physical KV compaction: 需要证明 block mapping、context length、slot mapping、prefix/swap 状态和 M-RoPE position 语义一致。
- VScan/PoRe/DeepStack-aware pruning: 当前只可作为设计假设；必须先补源码、测试和 benchmark，再进入项目 claim。

## 下一步

P5 当前门禁已完成，下一步进入 P6/P7:

- 为 `fp8_kv` 写 FP8-aware paged decode Triton kernel，减少 eager dequant + SDPA 的 latency。
- 为 `visual_prune` 写 retained-aware paged decode kernel，避免 Python gather。
- 扩展质量评测集到更多真实图像/视频问题和更长输出。
- 若继续做 physical visual-token compaction，需要先设计 compressed context length、slot mapping、block table、free block 回收和 prefix/swap 状态。
- 未支持的压缩模式继续显式失败。
