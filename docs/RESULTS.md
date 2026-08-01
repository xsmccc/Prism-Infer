# 性能结果

## 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GB |
| Driver / CUDA | 580.105.08 / 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.14.1 |
| Model | Qwen3-VL-8B-Instruct |
| Model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| External engines | vLLM 0.25.1 / SGLang 0.5.15.post1 |

TP2 使用同一台机器上的两张 RTX 5090。两张卡之间没有直接 P2P/NVLink，NCCL 经过
PCIe/CPU `NODE` 路径。

## 1. TP1 Decode

设置：batch 1、greedy、output 128、warmup 2、repeat 5。三个引擎使用相同的
Qwen3-VL prompt token。

| 场景 | 引擎 | TPOT | TTFT | E2E |
|---|---|---:|---:|---:|
| 8 张 448×448 图片 | **Prism** | **9.8821 ms** | **245.349 ms** | **1,598.843 ms** |
| | SGLang | 10.3520 ms | 284.844 ms | 1,600.005 ms |
| | vLLM | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| 16 帧 448×448 视频 | **Prism** | **9.8680 ms** | **240.175 ms** | **1,601.801 ms** |
| | SGLang | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| | vLLM | 10.5278 ms | 323.819 ms | 1,673.800 ms |

这两组输入中，Prism 的 TPOT 比 SGLang 低 4.54%–4.83%，比 vLLM 低
6.13%–6.27%。主要收益来自完整 Decode CUDA Graph、Packed Projection、Triton
Fusion 和更轻的 greedy token selection。

Scaled-FP8 KV 下，H1/H2 TPOT 分别为 10.2363 和 10.2588 ms。它比 Prism BF16
略慢，但仍低于同组 vLLM BF16 结果。

## 2. 双卡 TP2

设置：单张 448×448 图片、210 prompt tokens、batch 1、greedy、output 32、
warmup 1、repeat 3、`max_model_len=512`。

| 引擎 | TPOT | TTFT | E2E | 输出 |
|---|---:|---:|---:|---|
| **Prism** | **5.9701 ms** | 86.057 ms | 281.098 ms | 与 vLLM 逐 token 相同 |
| vLLM | 6.1612 ms | 61.251 ms | 252.744 ms | 参考输出 |
| SGLang | 5.9701 ms | 45.452 ms | 230.601 ms | 第 21 个 token 起不同 |

Prism 的 TP2 TPOT 比 vLLM 低 3.10%。TTFT 和 E2E 仍慢于 vLLM，因为 Vision
Encoder 在两个 rank 上重复执行，而当前机器也没有直接 GPU P2P。

### 优化过程

| 版本 | TPOT | 改动 |
|---|---:|---|
| 初始 TP2 CUDA Graph | 8.4720 ms | 正确的权重/KV 分片和 Graph replay |
| Fused kernels + Packed QKV | 6.5958 ms | QK-Norm、M-RoPE、Residual、Paged Attention tile |
| Rank-local torch.compile | 6.5709 ms | 编译 QKV、QK-Norm、M-RoPE |
| Persistent rank buffers | 6.3212 ms | 两个 rank 复用 pinned Graph inputs |
| Compact TP control | 5.9928 ms | 每步只发送 token、position 和 KV 状态 |
| 最终 repeat-3 | **5.9701 ms** | 完整配置复测 |

Rank-local compile 的单独收益约为 0.56%，更大的提升来自 Kernel 组合、Graph 输入
复用和 TP Host Control。混合 text+image+video batch3 的 TPOT 从 11.0999 降到
8.3603 ms，Decode 吞吐从 268.998 提升到 357.712 tok/s。

### Nsight Systems 分析

248 次 CUDA Graph replay 的 GPU 时间中位数为 5.9689 ms。两个 rank 的活跃时间
接近，说明工作切分比较均衡。

| GPU 时间占比 | 比例 |
|---|---:|
| cuBLAS BF16 GEMV | 约 84% |
| NCCL AllReduce | 约 8% |
| Paged Attention | 约 2.6% |
| Fused Add + RMSNorm | 不足 2% |

当前 Decode 的主要成本仍然是读取模型权重并执行 GEMV，而不是 NCCL 或 CUDA Graph
Launch。继续优化需要减少权重流量，例如权重量化或更大 batch 下的 GEMM 化；简单替换
cuBLAS GEMV 很难获得同等级提升。

## 3. KV Cache 容量

| 配置 | Pages / capacity | KV bytes | NVML peak | Torch peak |
|---|---:|---:|---:|---:|
| BF16 | 113 / 28,928 tokens | 4,068.000 MiB | 23,938 MiB | 21,637.368 MiB |
| Scaled-FP8，同容量 | 113 / 28,928 tokens | 2,097.562 MiB | 21,966 MiB | 19,667.298 MiB |
| Scaled-FP8，约 4 GiB | 220 / 56,320 tokens | 4,083.750 MiB | 23,952 MiB | 21,653.298 MiB |

结果：

- 同容量 KV 存储减少 **48.44%**；
- 进程显存峰值减少 **1,972 MiB（8.24%）**；
- 相同 KV 预算下容量从 28,928 提升到 56,320 tokens，增加 **94.69%**。

Scaled-FP8 在 DocVQA、MuirBench 和 MVBench 的测试集上保持了 BF16 的结果。直接
unit-scale cast 的长输出质量较差，没有用于最终配置。

视觉 KV 压实的 batch2 实验中，每个请求的 prompt pages 从 7 页降到 4 页，384 个
Decode steps 中有 378 个可以保持 batch2，请求吞吐提升 58.83%。

## 4. 多模态缓存与在线负载

设置：60 请求，Poisson 4 req/s，output 64，包含 text、image、multi-image 和
video。每次请求都会重新创建媒体对象，相同媒体根据内容命中缓存。

| 媒体重复率 | Prism raw | Prism Goodput | vLLM raw | vLLM Goodput |
|---:|---:|---:|---:|---:|
| 0% | 216.188 | 133.316 | **223.079** | **215.643** |
| 25% | 216.441 | 187.582 | **223.474** | **216.025** |
| 50% | 217.083 | 206.228 | **223.521** | **219.796** |
| 75% | 224.279 | 224.279 | **225.112** | **225.112** |
| 100% | 224.369 | 224.369 | **225.004** | **225.004** |
| 100%，更换问题 | 224.301 | 224.301 | **225.004** | **225.004** |

高重复率下，Prism 可以复用 Processor、Vision/DeepStack 和压实后的 Prefix KV，
与 vLLM 的吞吐差距缩小到 0.28%–0.37%。0% 重复时，冷多模态 Prefill 会明显影响
Decode 请求，仍是在线性能的主要短板。

100% 重复的 600 请求测试得到：

| 指标 | 结果 |
|---|---:|
| Raw throughput / Goodput | 241.428 / 241.428 tok/s |
| 满足 SLO | 600 / 600 |
| TTFT p50 | 146.418 ms |
| TPOT p50 | 13.041 ms |
| Process peak | 24,006 MiB |
| 释放的物理 pages | 480 |

## 5. 结果怎么看

Prism 当前最稳定的优势是 Qwen3-VL batch1 Decode：TP1 和 TP2 都在实测输入上低于
vLLM。Scaled-FP8 和视觉 KV 压实解决的是上下文容量问题，多模态前缀缓存解决的是
重复媒体请求的 Prefill 成本。

还没有解决的是冷 Vision/Prefill、Vision Parallel 和更大规模的分布式推理。对短
输出请求，TTFT 的影响可能大于 TPOT；对长输出或服务端连续生成，Decode 优化才会
更明显。
