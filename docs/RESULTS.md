# 性能与质量结果

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

不同章节使用各自固定协议，不能把 TPOT microbenchmark、工作集 TTFT 和容量结果混成
一次实验。请求级记录、派生表和 Trace 位于
[`artifacts/working_set`](../artifacts/working_set/README.md)。

## 1. 重复视觉上下文

### 1.1 工作集与 KV 预算

三套引擎使用同一份带编号 media-first MuirBench plan。每个工作集先冷建媒体条目，再
运行 600 条 Zipf-1.0 多问题请求；Poisson 4 req/s、greedy、output 16。KV 总预算为
4,282,122,240 bytes，对应 Prism 220 个 Scaled-FP8 pages。

| Workset | Media groups | Dense Prefix pages | Budget ratio |
|---|---:|---:|---:|
| fit | 25 | 151 | 68.6% |
| knee | 38 | 223 | 101.4% |
| pressure | 59 | 333 | 151.4% |

![Working-set pressure](../artifacts/working_set/performance/working_set_summary.png)

### 1.2 Prism 内部对照

`pressure` 工作集：

| 路径 | 驻留媒体 | 淘汰 | Vision miss | 重算 prompt tokens | TTFT p50 / p99 | E2E p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Vision/DeepStack Cache only | 7 | 0 | 394 | 1,052,583 | 578.965 / 3,044.275 ms | 1,832.086 / 4,646.051 ms |
| Dense Prefix | 33 | 110 | 91 | 192,090 | 124.009 / 707.520 ms | 420.122 / 1,179.932 ms |
| Compact Prefix | **48** | **33** | **17** | **83,018** | **102.401 / 270.588 ms** | **350.512 / 774.318 ms** |

Compact 运行的 dense-equivalent / actual Prefix pages 为 4,412 / 2,910（-34.0%）。
相对 Dense Prefix，驻留媒体增加 45.5%，重算 token 减少 56.8%，TTFT p50/p99
降低 17.4%/61.8%。

### 1.3 三引擎比较

| Workset | Engine | TTFT p50 / p99 | E2E p50 / p99 | Process peak | 重算 prompt tokens |
|---|---|---:|---:|---:|---:|
| fit | Prism Compact | **97.693** / 253.682 ms | 330.853 / 690.553 ms | 23,998 MiB | 64,222 |
| fit | vLLM | 112.978 / **220.883 ms** | **291.372 / 449.895 ms** | **23,628 MiB** | **50,020** |
| fit | SGLang | 428.032 / 1,176.561 ms | 739.369 / 1,911.111 ms | 23,944 MiB | 50,020 |
| knee | Prism Compact | **97.837** / 218.424 ms | 333.900 / 701.338 ms | 24,002 MiB | 63,392 |
| knee | vLLM | 109.921 / **200.345 ms** | **286.894 / 419.819 ms** | **23,628 MiB** | **50,547** |
| knee | SGLang | 430.673 / 1,614.508 ms | 704.202 / 2,172.660 ms | 23,944 MiB | 50,547 |
| pressure | Prism Compact | **102.401 / 270.588 ms** | 350.512 / **774.318 ms** | 24,006 MiB | **83,018** |
| pressure | vLLM | 131.483 / 581.976 ms | **323.228** / 886.035 ms | **23,714 MiB** | 165,031 |
| pressure | SGLang | 494.911 / 1,623.782 ms | 855.944 / 2,270.308 ms | 24,284 MiB | 179,111 |

`pressure` 上，Prism 相对 vLLM 的 TTFT p50/p99 低 22.1%/53.5%，E2E p99 低
12.6%，重算 token 少 49.7%；E2E p50 慢 8.4%，进程峰值高 292 MiB。三者输出
吞吐都约 64.1–64.2 tok/s，固定到达率使该指标不具区分度。

### 1.4 质量

| Dataset / cohort | Dense reference | Compact result | 解释 |
|---|---:|---:|---|
| MuirBench，85 条布局对照 | 官方交错 49/85 | labeled media-first 46/85 | Prompt 重排损失 3 题 |
| MuirBench，49 条实际删除样本 | labeled Dense 27/49 | Uniform 20/49 | 净损失 7 题；Attention 两种对照同为 20/49 |
| DocVQA，190 条 | ANLS 0.93335 | ANLS 0.93335 | 0 条发生删除，不能证明压实无损 |
| MVBench，252 条 | 183/252 | 113/252 | 视频删除质量下降，默认关闭 |

Compact 是有损 operating point，不是与 vLLM/SGLang 等质量的绝对排名。MuirBench
四组配对结果、DocVQA 与 MVBench 明细见
[`working_set_quality.md`](../artifacts/working_set/quality/working_set_quality.md)。

### 1.5 Prefix-hit Trace

代表性 Nsight capture 的 Trace 审计 6/6 通过：cold range 出现一次
`prism::model.vision.embedding_cache_miss`；Prefix-hit range 没有 Vision/DeepStack，
`visual_hydration_skips` 增加 1，`stale_probe_fallbacks` 为 0，并复用 145/275 个 prompt
tokens。Profiler 下 cold/hit GPU busy time 为 44.365/19.576 ms。Trace 只验证执行路径，
不替代上面的非 profiler 性能测量。

## 2. TP1 Decode

协议：batch 1、greedy、output 128、warmup 2、repeat 5；三套引擎使用相同的
Qwen3-VL prompt tokens。

| 场景 | 引擎 | TPOT | TTFT | E2E |
|---|---|---:|---:|---:|
| 8 张 448×448 图片 | **Prism** | **9.8821 ms** | **245.349 ms** | **1,598.843 ms** |
| | SGLang | 10.3520 ms | 284.844 ms | 1,600.005 ms |
| | vLLM | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| 16 帧 448×448 视频 | **Prism** | **9.8680 ms** | **240.175 ms** | **1,601.801 ms** |
| | SGLang | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| | vLLM | 10.5278 ms | 323.819 ms | 1,673.800 ms |

Prism TPOT 比 SGLang 低 4.54%–4.83%，比 vLLM 低 6.13%–6.27%。收益来自完整
Decode CUDA Graph、Packed Projection、Triton Fusion 和更轻的 greedy token
selection。Scaled-FP8 KV 下 H1/H2 TPOT 为 10.2363/10.2588 ms；它的主要目标是容量，
不是降低单 token 延迟。

## 3. KV Cache 容量

| 配置 | Pages / capacity | KV bytes | NVML peak | Torch peak |
|---|---:|---:|---:|---:|
| BF16 | 113 / 28,928 tokens | 4,068.000 MiB | 23,938 MiB | 21,637.368 MiB |
| Scaled-FP8，同容量 | 113 / 28,928 tokens | 2,097.562 MiB | 21,966 MiB | 19,667.298 MiB |
| Scaled-FP8，约 4 GiB | 220 / 56,320 tokens | 4,083.750 MiB | 23,952 MiB | 21,653.298 MiB |

Scaled-FP8 在同 token capacity 下把 KV 存储减少 48.44%，进程显存峰值减少
1,972 MiB（8.24%）；相同约 4 GiB KV 预算下容量从 28,928 提升到 56,320 tokens
（+94.69%）。直接 unit-scale cast 的长输出质量不稳定，最终路径使用 per-token、
per-KV-head scale。

## 4. 双卡 TP2

协议：单张 448×448 图片、210 prompt tokens、batch 1、greedy、output 32、warmup 1、
repeat 3、`max_model_len=512`。

| 引擎 | TPOT | TTFT | E2E | 输出 |
|---|---:|---:|---:|---|
| **Prism** | **5.9701 ms** | 86.057 ms | 281.098 ms | 与 vLLM 逐 token 相同 |
| vLLM | 6.1612 ms | 61.251 ms | 252.744 ms | 参考输出 |
| SGLang | 5.9701 ms | 45.452 ms | 230.601 ms | 第 21 个 token 起不同 |

Prism TPOT 比 vLLM 低 3.10%，但 TTFT/E2E 更慢。Vision Encoder 仍在两个 rank
重复执行，且测量机器没有直接 GPU P2P/NVLink。TP2 Nsight 中 cuBLAS BF16 GEMV 约占
84% GPU 时间，NCCL AllReduce 约 8%，Paged Attention 约 2.6%；继续替换小 Kernel
不会带来同量级收益。

## 5. 结果适用范围

- 主要工作集结果只覆盖 Qwen3-VL-8B、RTX 5090、TP1、固定 KV 预算和重复多图提问；
- Compact 图片配置存在明确质量损失，视频删除默认关闭；
- Prefix Cache 不跨 Engine Process 或机器；
- 冷 Vision/Prefill、Vision Parallel、PP、MoE Expert Parallel 和多机推理尚未实现；
- 当前 Serving API 为项目自有格式，不兼容 OpenAI API。
