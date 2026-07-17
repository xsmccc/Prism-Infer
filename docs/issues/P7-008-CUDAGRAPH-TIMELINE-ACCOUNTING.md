# P7-008: CUDA Graph timeline 的 CPU/GPU 重复计时风险

- 状态: `DOCUMENTED_LIMITATION`
- 首次观察 commit: `0fdd4a6`
- 汇总合同 commit: `72f85ba`
- 硬件/软件: RTX 5090；Prism Torch `2.6.0a0+nv25.01`；Nsight Systems node trace
- workload: synthetic single image 448x448，prompt 210，output 32，BF16，TP1
- 影响: 错把 CPU submission/synchronization range相加，会把同一段 GPU工作重复
  归因给 Graph、logits和 sampler，并产生错误优化优先级。

## 现象

single-image clean trace中，`runner.cudagraph.replay` 的 CPU range median只有
`1.899 ms`，而 replay kernel busy median为 `12.921 ms`。相反，
`runner.sampler` CPU range为 `13.790 ms`，但其 direct GPU busy只有 `0.007 ms`，
semantic CUDA-event elapsed只有 `0.187 ms`。

如果直接把这些 CPU range解释成各模块独立耗时，会得到两个相互矛盾的结论：Graph
似乎只需约 1.9 ms，而 sampler似乎消耗约 13.8 ms。两者都不符合 GPU timeline。

## 如何发现

P7.4-B 扩展 `benchmarks/analyze_nsys.py`，不再只累计落在 NVTX range内启动的
kernel duration，还计算：

- 每个 range 的 GPU busy interval union与完整 activity span；
- CPU range与 GPU busy的时间交集；
- CPU range返回后、其关联 GPU activity仍持续的 tail；
- Graph 外 range直接发起的 kernel/memcpy/memset activity；
- Graph replay的 kernel category partition。

capture使用 CUDA Profiler API排除加载、warmup和 Graph capture，并以
`--cuda-graph-trace=node` 展开 31 个 measured decode steps。

## 定位过程

replay的关键中位数：

| Metric | Value |
|---|---:|
| CPU NVTX range | `1.899233 ms` |
| kernel busy | `12.920926 ms` |
| GPU busy union | `12.922270 ms` |
| GPU activity span | `14.956944 ms` |
| CPU/GPU busy overlap | `0.030400 ms` (`1.618%`) |
| GPU tail after CPU return | `13.088793 ms` |

Graph 外 direct GPU busy中，prepare inputs、prepare sample inputs与 Graph copy分别
只有 `0.0016/0.0003/0.0062 ms`，logits为 `0.7615 ms`，sampler为约
`0.0075 ms`。整个 decode kernel busy `13.689859 ms` 减去 replay
`12.920926 ms` 为 `0.768933 ms`，与 logits直接 activity吻合。

## 根因

CUDA Graph replay与后续 logits是异步提交到同一 CUDA stream。host在 GPU完成
Graph前就能从 `graph.replay()` 返回并继续提交依赖工作；真正需要 host读取采样结果
时，sampler路径触发同步。因此长 sampler CPU range主要是在等待已经提交的 Graph/
logits工作完成，不是 sampler自身执行了同样长的独立 GPU计算。

NVTX CPU nesting与“由该 range发起的 GPU activity”回答的问题不同。只有把 CPU
range、direct GPU activity、stream依赖和 tail放在同一 timeline解释，才能避免重复
计时。

## 解决方案

- analyzer schema-v2同时输出 CPU range、direct GPU busy/span、overlap和 tail。
- P7.4-B summary固定展示 Graph 外 direct activity，并把 sampler同步与 padding
  matrix的 claim boundary写入机器可读输出。
- 性能报告使用 replay kernel busy/category做优化排序；无 profiler TPOT仍是最终
  E2E数字，trace只用于归因。

## 为什么有效

分类后的 replay kernel busy加上 Graph 外约 `0.769 ms`，与完整 decode kernel busy
闭合；Graph 外差值又与 logits `0.762 ms`闭合。这个交叉检查能发现把 sampler wait
重复加入 GPU critical path的错误。

## 验证

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_p7_graph_summary.py \
  tests/test_nsys_analysis.py \
  tests/test_benchmark_schema.py
# 43 passed in 3.90s
```

正式 summary：

```text
data/p7_graph/p74b_summary_72f85ba.{json,md}
```

## 被拒绝的方法

- 把 `graph.replay()` CPU range当作完整 Graph latency：忽略异步执行与 GPU tail。
- 把 sampler CPU range加到 replay/logits上：把前序 stream wait重复计时。
- 把 `gpu_span - gpu_busy`写成 occupancy或“可消除 idle”：node tracing带有
  instrumentation，且 span差不提供 SM active/DRAM throughput counter。
- 用 semantic profiler或 Nsight trace替代无 profiler TPOT：profiling会扰动时序。

## 剩余限制

- RTX 5090环境仍没有可用的 NCU hardware counter权限；本记录不能证明 GEMV是
  memory-bound、compute-bound或 occupancy-bound。
- node trace只覆盖一个模型、单图、output32与 batch1；kernel占比不能外推到长
  context、不同 batch、其他 GPU或 online mixed workload。
- fixed-bucket matrix是进程级 coverage/correctness证据，不是 timeline trace，也不
  能量化 padding开销。

## 面试表达

> CUDA调用是异步的，所以 NVTX里的 host时间不能直接当 GPU模块时间。我同时计算
> replay的 busy union、CPU/GPU overlap和 CPU返回后的 GPU tail，发现 Graph提交只占
> host约1.9 ms，但 GPU继续执行约13.1 ms；sampler的长 CPU range则是在最终读取时
> 等待前序 stream。这样避免把同一段 Graph工作重复算成 sampler开销。
