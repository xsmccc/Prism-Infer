# P7-003: CUDA Graph 后仍存在的 vLLM 差距

- 状态: `INVESTIGATING`
- 首次观察 commit: `c970c61`（preflight）
- workload: synthetic single image 448x448, prompt 210, output 32, BF16
- KV pool: 603,979,776 bytes, block size 256, TP1

## 现象

在相邻的 preflight 中：

| Engine | Decode backend | Median TPOT |
|---|---|---:|
| Prism off | CUDA Graph | `17.915 ms` |
| Prism content-aware compact | CUDA Graph | `17.650 ms` |
| vLLM 0.24.0 | `FULL_AND_PIECEWISE` | `10.097 ms` |

这组 preflight 使用 `warmup=1/repeat=3`，用于确认路径和形成假设，不是 P7.1
正式 claim。正式结果必须在 schema v2、clean harness、`warmup=2/repeat=5` 下重跑。

## 如何发现

P6 先通过 eager/Graph 正交实验发现 launch overhead：Prism Graph 比 eager 快约
1.8 倍。P7 随后把 vLLM 从 eager 切到其默认 Graph，而不是拿旧 external eager
数字继续比较。vLLM 日志确认：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
Capturing CUDA graphs (decode, FULL)
```

因此剩余差距不能继续全部归因于 Python launch。

## 当前排除项

- 两边 prompt token 都是 210。
- 两边 BF16、TP1、block 256、固定相同 KV pool、prefix/MM cache off。
- 两边 temperature 0、ignore EOS、固定输出 32。
- 重复输出各自稳定。

preflight 尚未用 schema-v2 自动检查全部条件，以上结论需要正式 rerun 确认。

## 下一步定位顺序

1. 正式跑 `diagnostic_matched` 和 `best_stable`，量化 eager gap 与 Graph 后 gap。
2. 对 Prism/vLLM Graph 各采一条相同 workload 的 Nsight Systems timeline。
3. 分离 graph replay 内 GPU duration 与 graph 外 input copy/logits/sampler。
4. 对比每 token 的 GEMM、RMSNorm/RoPE/KV store、paged attention kernel 数和总时长。
5. 只对 top kernels 使用 NCU；根据结果决定 Inductor fusion、TK kernel 或调度。

## 为什么这个顺序有效

Systems 先回答“时间花在哪些 kernel/空隙”，NCU 再回答“top kernel 为什么慢”。
若跳过 Systems 直接优化 paged attention，很可能重复 P6.5 的结果：microbenchmark
明显改善，但短 context 整步 TPOT几乎不变。

## 暂不接受的解释

- “vLLM 代码更多所以快”：不可证伪，也不能指导优化。
- “都是 paged attention 慢”：P6 full-engine 数据尚不支持。
- “上 ThunderKittens 就会快”：必须先证明目标 region 是 critical path。

## 面试表达

> CUDA Graph 把 Prism 的 launch overhead 大幅消除后，我重新让 vLLM 也使用其
> Graph 最优路径，发现仍有残差。这个实验把问题从 host launch 缩小到了 Graph
> 内模型执行和 Graph 外固定开销，下一步再用 Systems/NCU决定是否做 compiler
> fusion 或 Blackwell kernel。
