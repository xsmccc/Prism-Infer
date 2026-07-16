# P7-003: CUDA Graph 后仍存在的 vLLM 差距

- 状态: `INVESTIGATING`（差距已确认，kernel root cause 待 Systems/NCU）
- 首次观察 commit: `c970c61`（preflight）
- 正式证据 commit: `b17f933`
- 硬件/软件: RTX 5090；Prism Torch `2.6.0a0+nv25.01`；vLLM
  `0.24.0/ee0da84ab`、Torch `2.11.0+cu130`
- workload: synthetic single image 448x448, prompt 210, output 32, BF16
- KV pool: 603,979,776 bytes, block size 256, TP1

## 现象

在相邻的 preflight 中：

| Engine | Decode backend | Median TPOT |
|---|---|---:|
| Prism off | CUDA Graph | `17.915 ms` |
| Prism content-aware compact | CUDA Graph | `17.650 ms` |
| vLLM 0.24.0 | `FULL_AND_PIECEWISE` | `10.097 ms` |

这组 preflight 使用 `warmup=1/repeat=3`，只用于确认路径和形成假设。

schema-v2、clean commit `b17f933`、`warmup=2/repeat=5` 的正式结果随后确认：

| Workload | Prism off Graph | Prism compact Graph | vLLM Graph |
|---|---:|---:|---:|
| single image | `17.932 ms` | `17.671 ms` | `10.098 ms` |
| two images | `18.501 ms` | `17.959 ms` | `10.119 ms` |
| real COCO | `18.234 ms` | `17.822 ms` | `10.105 ms` |
| fidelity batch A (4) | `18.966 ms` | `18.574 ms` | `10.822 ms` |
| fidelity batch B (3) | `18.536 ms` | `18.120 ms` | `10.970 ms` |

全部 10 个 best-stable comparison cells 通过自动 comparability gates。compact
Graph 的 TPOT 是 vLLM 的约 `1.65x-1.78x`，因此 residual gap 是正式结果。

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

formal rerun 还校验了 model config SHA256、GPU UUID、clean harness/source、
warmup/repeat、timing scope 和 effective `FULL_AND_PIECEWISE` mode。

## Prism 内部时间分解

同一 clean commit 的 single-image semantic CUDA profile 给出：

| Region | Off Graph median | Compact Graph median |
|---|---:|---:|
| Graph replay | `13.394 ms` | `13.124 ms` |
| logits | `4.068 ms` | `4.068 ms` |
| Graph input copy | `0.129 ms` | `0.130 ms` |
| sampler GPU | `0.175 ms` | `0.174 ms` |

Graph replay 约占 Prism TPOT 的四分之三，Graph 外 logits 约占 23%。压缩把 replay
减少约 `0.27 ms`，解释了 TPOT 的小幅改善；compaction copy 本身发生在 prefill，
不是剩余 decode gap 的主因。

semantic profile 会因 CUDA event/synchronize 增加开销，因此只用 region CUDA
duration做归因，最终 TPOT仍来自无 profiler benchmark。

## 复现与验证

正式 cell 由 `docs/P7_OFFLINE_COMPARISON_DESIGN.md` 中的双 profile 命令生成，
以下命令重新执行机器可比性门禁：

```bash
.venv-local/bin/python scripts/summarize_p7_external.py \
  --comparison-profile best_stable \
  --prism data/p7_external/prism_*_formal_b17f933.jsonl \
  --external data/p7_external/vllm_*_best_stable_formal_b17f933.json \
  --prism-modes off_graph visual_compact_graph \
  --prism-keep-ratio 0.5 \
  --json-output /tmp/p7_best_stable.json \
  --markdown-output /tmp/p7_best_stable.md
# compared 10 P7.1 cells under best_stable
```

同一 single-image workload 的 region 归因命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 5 --repeat 5 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching \
  --visual-pruning-keep-ratio 0.5 \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/p7_external/prism_single_image_graph_profile_benchmark_b17f933.jsonl \
  --profile-output data/p7_external/prism_single_image_graph_semantic_profile_b17f933.jsonl \
  --profile-repeat 5
```

## 下一步定位顺序

1. ~~正式跑 `diagnostic_matched` 和 `best_stable`。~~ 已完成。
2. ~~分离 Prism replay 与 Graph 外 input/logits/sampler。~~ semantic profile 已完成。
3. 对 Prism/vLLM Graph 各采一条相同 workload 的 Nsight Systems timeline。
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

## 剩余限制

- 当前只有 Prism 的语义 region 分解，还没有双方同条件的 Nsight Systems kernel
  timeline，不能把 `1.65x-1.78x` 归因给某一个 kernel。
- 结果限定 RTX 5090、Qwen3-VL-8B、短输出 offline closed-loop；不代表 online
  goodput，也不外推到其他模型、GPU 或长输出。
- semantic profiler 会改变整步延迟，因此其 region duration 只用于定位，不替代
  无 profiler 的正式 TPOT。

## 面试表达

> CUDA Graph 把 Prism 的 launch overhead 大幅消除后，我重新让 vLLM 也使用其
> Graph 最优路径，发现仍有残差。这个实验把问题从 host launch 缩小到了 Graph
> 内模型执行和 Graph 外固定开销，下一步再用 Systems/NCU决定是否做 compiler
> fusion 或 Blackwell kernel。
