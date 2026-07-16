# P7-006: 逐 decode 的 FP32 lm-head 整权重转换

- 状态: `RESOLVED`
- 首次 region 证据: `b17f933`
- node-level root cause capture: `66e6f9f`
- 修复 commit: `a33e7ed`
- 最终回归 commit: `cc070b3`
- 硬件/软件: RTX 5090，Qwen3-VL-8B，Prism Torch `2.6.0a0+nv25.01`
- 影响: TPOT、临时 peak allocation、与 vLLM 的 Graph residual gap

## 现象

P7.1 best-stable 中 Prism compact Graph TPOT为 vLLM的 `1.65x-1.78x`。semantic
profile显示每步约 `13.4 ms` 在 Graph replay，另有稳定的 `4.068 ms` 位于 Graph
外 `runner.model.compute_logits`。后者约占单步 TPOT的 23%，且不随 visual KV
compaction变化。

## 如何发现

先用 region profile排除 scheduler、Graph input copy、sampler和 compaction copy，
再对 Prism/vLLM 相同 single-image/output32 Graph workload采集 node-level Nsight
Systems。旧 Prism kernel summary出现两条 vLLM没有的显著工作：

- BF16→FP32 `direct_copy` 全 capture合计 `96.141 ms`。
- 32 次 FP32 vocab GEMV合计 `48.282 ms`。

`prism::runner.model.compute_logits` 的 32 个 ranges median为 `4.068 ms`，与这两组
kernel时间吻合。

## 根因

旧实现为追求 greedy tie-break稳定性，在 CUDA FP16/BF16 输入上执行：

```python
F.linear(hidden_states.float(), self.lm_head.weight.float())
```

lm-head shape为 `151,936 × 4,096`。`.weight.float()` 不是一次性转换，而是在每个
prefill/decode step重新物化整张 FP32权重；它既读取约 1.16 GiB BF16权重，又创建
约 2.32 GiB FP32临时 tensor，然后执行 FP32 GEMV。CUDA Graph只覆盖 decoder
forward，因此这条路径一直留在 Graph 外。

## 解决方案

- `Config.logits_precision` 显式支持 `model/fp32`，默认设为 `model`。
- Qwen3-VL 默认直接执行 BF16 lm-head；`fp32` 只保留历史复现，不再是生产默认。
- benchmark record记录 logits precision，避免新旧路径被静默混比。
- vLLM harness增加 `--cuda-profiler-range`，允许排除模型加载/Graph capture后做同
  workload Nsight capture。

## 为什么有效

模型原生路径只执行一次 BF16 lm-head GEMV，不再逐步扫描和扩张整张权重。优化后：

| Metric | FP32 historical | Model precision |
|---|---:|---:|
| logits CUDA median | `4.068 ms` | `0.762 ms` |
| logits kernels/range | `4` | `1` |
| single off-Graph TPOT | `17.887 ms` | `14.151 ms` |
| single peak allocated | `19,708.6 MiB` | `17,391.5 MiB` |

五 workload × off/compact 的 clean 单变量矩阵全部复现 `1.216x-1.280x` TPOT
speedup，并减少 `2,230-2,317 MiB` peak allocation。

## Correctness 与质量验证

model precision不是近似替代：HF Qwen3-VL本身也使用模型 dtype lm-head。固定
single-image、multi-image、video 32-token teacher-forced轨迹上，Prism model
precision与 HF logits/PPL均为 `max diff=0 / mean diff=0 / ppl diff=0`；历史 FP32
路径反而有约 `0.123-0.125` max logit diff。

7-image/35-caption content-aware gate：

- token-F1 `0.318842 -> 0.314482`，drop `0.004360 <= 0.01`。
- ROUGE-L `0.285863 -> 0.289953`，改善 `0.004090`。
- physical tokens `0.535x`，active bytes `0.538x`，task gate PASS。

第一次 full regression为 `2 failures`：旧 mixed-batch测试要求 batch1 GEMV 与
batch4 GEMM 的 greedy tokens跨 shape exact。低精度 GEMM shape sensitivity已在
HF/Prism duplicate-batch证据中存在；model precision对同 shape与 HF逐值一致。
最终测试不删除该现象，而是改为：

- 同一 mixed shape重复运行 token exact。
- text prefix@8、image/multi-image跨 shape prefix至少 16。
- video跨 shape首 token分叉显式记录，不伪装为实现错误。
- HF teacher-forced分布与 reference task gate继续作为独立质量门禁。

clean `cc070b3` full regression：`241 passed, 6 skipped in 264.664s`。

## 复现命令

单变量性能：

```bash
.venv-local/bin/python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 2 --repeat 5 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching --logits-precision model \
  --output data/p7_external/p74_logits_model.jsonl
```

将 `--logits-precision model` 改为 `fp32` 即得到同 commit历史对照。

node-level capture：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --force-overwrite=true \
  --output=data/p7_external/p74_logits_model \
  .venv-local/bin/python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 --modes off_graph \
  --max-tokens 32 --warmup 2 --repeat 1 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching --profile-repeat 1 --cuda-profiler-range \
  --profile-output data/p7_external/p74_logits_model_semantic.jsonl
```

## 被拒绝的方法

- **缓存完整 FP32 lm-head**：可省重复 cast，但常驻增加约 2.32 GiB，直接侵蚀 KV
  capacity，且不如 HF model-precision数值忠实。
- **把 logits强行塞进现有 CUDA Graph**：只能隐藏 launch，不能消除每步整权重
  物化和显存流量。
- **只删除 mixed-batch失败测试**：会丢失真实 shape sensitivity。最终以同-shape
  determinism、HF分布和任务质量替换错误的跨-shape exact合同。
- **声称已经超过 vLLM**：优化后 compact TPOT仍为 vLLM的 `1.34x-1.40x`。

## 剩余限制

- 真实 COCO 的 model-precision greedy token不保证与历史 FP32路径 exact；后者不是
  质量 reference。
- mixed video batch1/batch4可从首 token分叉；同一 shape确定，但跨 shape复现需
  固定 batching/execution contract。
- Graph replay仍约 `12.93 ms`，现在占绝大多数 TPOT；其 kernel-level差距尚未闭环。
- 本阶段仍是 offline closed-loop，不形成 online SLO goodput claim。

## 面试表达

> 我先用语义 region看到 CUDA Graph 外还有稳定的 4 ms，再用 node-level Systems
> 找到每步把 15 万词表的整张 lm-head从 BF16转成 FP32。改回模型原生精度后，
> logits降到 0.76 ms、整步提升 1.22-1.28 倍并少了约 2.3 GiB peak。第一次全回归
> 暴露跨 batch shape的低精度 tie-break，我没有藏掉：改成同-shape determinism、
> HF logits/PPL和任务质量三层门禁。最终与 vLLM的TPOT差距缩到 1.34-1.40 倍，
> 下一步只继续追 Graph replay。
