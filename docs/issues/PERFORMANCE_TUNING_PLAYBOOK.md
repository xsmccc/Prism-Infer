# Prism-Infer 性能调优方法

本文固定 Prism-Infer 后续性能工作的证据链。目标不是让 profiler 截图变多，
而是让每项优化都能回答：瓶颈在哪里、为什么选择这个改动、收益来自哪里、
correctness 和质量是否保持。

## 1. 先定义要优化的 workload

所有性能结论先固定：

- 模型 snapshot、processor/tokenizer 和 dtype。
- GPU、driver、CUDA、Torch 和框架 commit。
- text/image/video 输入、prompt token、输出长度和采样参数。
- batch、concurrency、request rate 和显存/KV pool 预算。
- eager/compile/CUDA Graph、attention backend、compression mode。
- preprocessing、输出 decode 和同步边界是否计时。

如果 workload 或计时口径不同，数字只能并列展示，不能计算 speedup。

## 2. 分离四层证据

### 2.1 无 profiler 的系统基线

使用 warmup + repeat、显式 CUDA synchronize，输出 TTFT、TPOT、E2E、吞吐、
峰值显存、物理 KV token/page/bytes。这是最终性能 claim 的来源。

### 2.2 语义 trace

用 Prism NVTX/semantic region 回答“时间属于 scheduler、vision、prefill、
compaction、decode 还是 sampler”。该路径可能同步和序列化，只用于归因。

### 2.3 Nsight Systems

回答 CPU/GPU 时间线问题：

- CPU launch gap 是否让 GPU 空闲。
- 是否出现同步、D2H/H2D 或 allocator 调用。
- CUDA Graph 内外各有多少 kernel。
- preprocessing、copy、compute 是否重叠。

### 2.4 Nsight Compute

只对 Systems 已确认的 top kernels 采集 counter，判断：

- memory-bound、compute-bound 还是 latency/occupancy-bound。
- DRAM/L2 throughput、arithmetic intensity、tensor core 利用。
- registers、shared memory、occupancy 和 warp stall。

不要一开始对整个模型运行 NCU；重放开销和数据量会掩盖问题。

## 3. 用对照实验定位，不靠名字猜瓶颈

每次只改变一个轴：

```text
off eager -> off Graph                 归因 launch/host overhead
off Graph -> compact Graph             归因物理 context 与 scorer/compaction
BF16 paged -> FP8 paged                归因 KV load/dequant
PyTorch reference -> Triton/TK kernel  归因单 kernel 实现
batch/context/output matrix            判断收益适用区间
```

例如 paged attention microbenchmark 更快，但整步 TPOT 不变，说明它不是当前
end-to-end critical path；此时继续调它不会产生系统收益。

## 4. 建立假设并设置停止条件

优化前记录：

1. 假设：哪个 region 为什么慢。
2. 证据：timeline/counter/ablation 支持什么。
3. 预测：改动应该减少哪个可观察指标。
4. correctness 门禁：token/logits/layout/quality 如何验证。
5. 停止条件：若目标指标不变，删除候选或降级为研究记录。

候选 kernel 至少需要独立 reference correctness 和 shape matrix；如果 kernel
有明显 micro speedup 但代表性 E2E 无可测收益，应如实标记为 kernel-only。

## 5. CUDA Graph 的分析方法

Graph 不是一个布尔开关。至少检查：

- capture scope、capture sizes 和首次 capture 时间。
- requested batch、selected bucket 和 padding。
- replay hit/miss、静默 eager fallback。
- replay 外 input copy、logits、sampler 和 scheduler 成本。
- graph pool 带来的额外峰值显存。

eager→Graph 的收益只证明 launch/host overhead 存在；Graph 后仍有的外部差距
必须继续在模型算子、attention backend、数据准备和框架调度之间分解。

## 6. 正式结果流程

```text
实现 benchmark/schema/tests
  -> focused regression
  -> commit
  -> clean-commit formal benchmark
  -> 生成自动汇总
  -> correctness/质量门禁
  -> 文档记录（此时工作区可再次变 dirty）
  -> full regression
```

正式记录必须保存 commit 和 dirty state。失败、OOM 和异常值原样保留；不得删除
异常样本后重算，也不得把 offline replicated batch 称为 online serving。

## 7. 面试时如何讲一次调优

按“现象—证据—假设—实验—结果—限制”讲：

> 我先在固定 workload 上看到 eager TPOT 落后。Nsight Systems 显示每步大量
> launch，因此用同 attention/KV layout 做 eager/Graph 正交实验，Graph 将
> TPOT 降低约 1.8 倍，证明 host launch 是一部分瓶颈。随后与 vLLM Graph
> 同条件重跑，仍存在明显差距，所以没有继续宣称问题已解决，而是把下一轮
> profiling 转向 Graph 内模型执行和 top kernels。

这种表达同时体现性能结果、归因能力和工程边界意识。
