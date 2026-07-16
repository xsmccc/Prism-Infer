# P7-005: Offline TTFT/vision prefill 双峰

- 状态: `INVESTIGATING`
- commit: `b17f933`
- 硬件/软件: RTX 5090，Qwen3-VL-8B，Prism Torch `2.6.0a0+nv25.01`
- workload: single image 448x448, output32, Graph
- 影响: offline TTFT、E2E latency 和 E2E throughput；decode TPOT不受明显影响

## 现象

正式 `warmup=2/repeat=5` 中，部分 Prism Graph cell 的 engine TTFT 在约
`50-60 ms` 和 `130-140 ms` 两档之间变化，导致 single-image compact/off 的
E2E throughput 中位数看似相差约 13%，与只有约 1.5% 的 TPOT变化不一致。

定向 `warmup=5/repeat=15` 仍复现：

| Mode | TTFT median / p90 / min / max | TPOT median / p90 |
|---|---|---|
| off Graph | `131.468 / 139.074 / 50.040 / 141.531 ms` | `17.922 / 17.957 ms` |
| compact Graph | `58.037 / 132.139 / 51.271 / 136.011 ms` | `17.670 / 17.709 ms` |

## 定位过程

1. 增加 repeat，排除单个偶发样本决定结论。
2. preprocessing 只有约 `6-9 ms`，主要波动在 GPU prefill。
3. semantic CUDA profile 将 prefill分成 vision 和 language model：
   - off Graph `model.vision.image` 为 `13.748..87.165 ms`。
   - off Graph language model 为 `35.834..76.866 ms`。
   - compact Graph 本次 profile 中 vision 为 `79.142..90.964 ms`，language
     model稳定在 `36.404..36.635 ms`。
4. decode Graph replay、logits 和 TPOT分布很窄，说明问题集中在 eager prefill，
   不是 Graph replay。
5. 尝试 `nvidia-smi --lock-gpu-clocks=2400,2400`，当前容器缺少修改 clocks 的
   driver permission，无法用固定频率实验确认动态 P-state 假设。

## 复现命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 5 --repeat 15 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching \
  --visual-pruning-keep-ratio 0.5 \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/p7_external/prism_single_image_graph_stability_b17f933.jsonl
```

## 当前结论

双峰真实存在且主要来自 vision/eager prefill，但根因尚未充分证明。动态 GPU
频率、宿主调度或底层 kernel algorithm 状态都是候选；目前不能仅凭 idle P8
状态写成“GPU 降频导致”。

## 解决方案：隔离不稳定指标

- TPOT 使用稳定的 step 分布，可以作为 P7.1 headline。
- E2E/TTFT保留 median/p90/min/max，但不把 compact/off E2E差异归因于压缩。
- P7.3 online benchmark 将分别报告 idle/cold 和 sustained-load TTFT；持续在线
  负载不能直接使用 offline 单请求 E2E数字外推。
- 后续在允许锁频/采集 GPU metrics 的环境，用相同 workload 复跑并记录时钟、
  power 和 kernel timeline。

## 为什么这种处理有效

限制 claim 比删除“慢样本”更可靠：TPOT的数百个 decode step 已稳定，而 TTFT
只有每 repeat 一个样本且存在两种状态。分别报告两类指标保留事实，也避免用
统计中位数掩盖系统状态变化。

## 验证

- `warmup=5/repeat=15` 在增加样本后仍复现 TTFT 两档，排除单个 outlier。
- semantic CUDA profile 将波动缩小到 eager prefill；Graph replay、logits 和
  `15×31` 个 decode-step 样本保持窄分布。
- P7.1 正式结论改用稳定 TPOT；E2E/TTFT仍保留原始分布，没有删除慢样本。

## 被拒绝的方法

- 删除 `130-140 ms` 样本后只报告快档：没有证据证明这些样本无效。
- 看到 idle P8 就直接写成“GPU 降频”：缺少逐 iteration clock/power counter 和
  锁频对照，因果证据不足。
- 用 compact/off 的 E2E中位数差异声称 13% 压缩收益：它与约 1.5% 的稳定 TPOT
  改善不一致，混入了 prefill 状态变化。

## 剩余限制

- 当前容器不能锁 GPU clocks，也没有逐 iteration 的 clock/power telemetry。
- 尚未采集双峰两档各自的 Nsight Systems kernel timeline，底层 algorithm 或宿主
  调度仍是候选原因。
- online sustained-load 下的频率状态和 TTFT分布可能不同，不能从 offline 单请求
  直接外推。

## 面试表达

> 我看到 E2E 提升和 TPOT改善不一致，没有把它写成压缩收益。增加到 15 次后
> 确认 TTFT 双峰，再用语义 CUDA region 把波动缩小到 vision prefill。因为当前
> 环境不能锁频且 counter不足，我保留为未决问题，只发布稳定 TPOT，并为 online
> benchmark区分 cold 与 sustained-load TTFT。
