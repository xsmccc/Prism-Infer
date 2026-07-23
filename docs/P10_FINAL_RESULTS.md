# P10 最终结果：Compiler/Graph 低延迟与 scaled-FP8 KV 容量

> 冻结 benchmark commit：`47793420b6796951a784b436946100011d4f75b1`
>
> 日期：2026-07-23
>
> 结论边界：Qwen3-VL-8B-Instruct、RTX 5090、TP1、batch1、greedy、
> output128、offline closed-loop。本文不外推为跨模型、跨硬件、任意 batch 或
> online serving 的全面排名。

## 1. 两条最终 profile

Prism-Infer 最终保留两条目的不同、都可正确运行的路径：

1. **Latency Profile**：BF16 KV、`torch.compile` 无状态 decode 子图、完整
   CUDA Graph decode replay。目标是在冻结 H1/H2 中给出最低 TPOT。
2. **Memory/Capacity Profile**：per-token/per-KV-head scaled-FP8 KV，保持
   相同约 4 GiB KV budget，同时复用同一 compile/Graph 路径。目标是容量接近翻倍，
   并让 TPOT 仍低于同协议 vLLM/SGLang，而不是宣称量化本身加速。

旧 unit-scale `fp8_kv` 质量失败，不进入最终 profile。content-aware physical
compaction 与 scaled-FP8 的组合尚未完成标准质量矩阵，只保留为研究候选。

## 2. 冻结环境与协议

```text
GPU: NVIDIA GeForce RTX 5090
GPU UUID: GPU-7f63f8b0-1027-d3bf-18b7-5102cbc9f2eb
Driver: 580.105.08
CUDA: 13.0
Python: 3.12.3
PyTorch: 2.11.0+cu130
Prism Transformers: 5.14.1
vLLM: 0.25.1
SGLang: 0.5.15.post1
Model revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Model dtype / tensor parallel: BF16 / TP1
Warmup / measured repeats: 2 / 5
Output: 128 greedy tokens, ignore_eos=true
Profiler in latency runs: off
```

两个冻结 workload：

| ID | 输入 | Prompt tokens | Prompt-token SHA256 |
|---|---|---:|---|
| H1 | 8 张 448×448 合成图片 | 1,618 | `04205e4593a1c294efa78f78462246266c6469d59decbe161973aeba757786b9` |
| H2 | 16 帧 448×448 合成视频，24 fps | 1,667 | `a3241f512bbb1a3e825585d118dc00383119a33302901642137cfd95c16bc5b2` |

八个最终单元都满足：

- harness commit 为 `4779342`，工作树与外部框架 source 均 clean；
- GPU UUID 一致；
- 同一 workload 在 Prism、vLLM、SGLang 间 prompt-token SHA256 exact；
- warmup2/repeat5，greedy 输出在各框架内部跨 repeat exact；
- Prism 与 vLLM 固定约 4 GiB KV pool；SGLang 固定 28,928-token capacity；
- H2 的 SGLang 输入由 FFV1 无损封装，解码后 16 帧 RGB 与原始帧逐字节一致，
  fps 为 24，decoded RGB SHA256 为
  `accd9e13f7dc3c841f385176f681a2f2e016cc694d3f639fbea2948e5946326e`。

vLLM 的 H2 parser 会消费完整视频 marker triplet。benchmark 复用已有质量评测
兼容器，在送入 vLLM 前增加一层外 marker；处理后 prompt IDs 与 Prism exact。修复前
1,665-token H2 只保留为诊断证据，不参与排名。

## 3. Latency Profile：BF16 compile + CUDA Graph

| Case | System | TPOT median | TTFT median | E2E median |
|---|---|---:|---:|---:|
| H1 | **Prism BF16** | **9.8821 ms** | **245.349 ms** | **1,598.843 ms** |
| H1 | SGLang BF16 | 10.3520 ms | 284.844 ms | 1,600.005 ms |
| H1 | vLLM BF16 | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| H2 | **Prism BF16** | **9.8680 ms** | **240.175 ms** | **1,601.801 ms** |
| H2 | SGLang BF16 | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| H2 | vLLM BF16 | 10.5278 ms | 323.819 ms | 1,673.800 ms |

相对外部框架，Prism BF16 的 latency 降幅：

| Case | 对比 | TPOT lower | TTFT lower | E2E lower |
|---|---|---:|---:|---:|
| H1 | SGLang | 4.54% | 13.87% | 0.07% |
| H1 | vLLM | 6.13% | 15.56% | 1.84% |
| H2 | SGLang | 4.83% | 38.44% | 6.17% |
| H2 | vLLM | 6.27% | 25.83% | 4.30% |

因此允许使用的表述是：**在冻结 H1/H2、单卡 batch1 offline CUDA Graph
协议中，Prism BF16 的 TPOT 比 SGLang 低 4.54%–4.83%，比 vLLM 低
6.13%–6.27%。** H1 对 SGLang 的 E2E 只低 0.07%，应按近似持平表达。

### 3.1 compile/Graph 到底覆盖什么

- CUDA Graph capture 覆盖 decode model forward、logits、guarded greedy selection；
- `torch.compile` 使用 Inductor
  `max-autotune-no-cudagraphs` 编译 batch1 无状态热点；
- 当前 compiled region 包含 output projection，以及动态 FP8 LM-head 候选生成；
- 候选阶段只缩小 token 集，最终 token 由精确 rerank 决定；低 margin 时走精确回退；
- Paged KV 的 payload/scale、block table、slot mapping 继续由受审计的 runtime
  store/attention 边界管理，不把有状态 KV 所有权交给 compiler。

这是一条 guarded fast path，不是以近似 token 代替正确输出。正式 artifact 仍保存
跨 repeat token hash、prompt hash、backend 配置和 capture bucket。

## 4. Memory/Capacity Profile：scaled-FP8 KV

存储格式为 E4M3FN K/V payload，加每个 token、每个 KV head 独立的 K/V FP32
scale。完整生命周期覆盖 store、paged decode、CoW、swap、physical compaction 和
CUDA Graph replay。

### 4.1 真实 KV 与进程显存

NVML 采样与 latency 计时分离；开启采样的 artifact 被显式标为不具备 latency
headline 资格。

| Profile | Blocks / capacity | KV bytes | NVML process peak | Torch peak allocated |
|---|---:|---:|---:|---:|
| BF16 | 113 / 28,928 tokens | 4,068.000 MiB | 23,938 MiB | 21,637.368 MiB |
| scaled-FP8，同容量 | 113 / 28,928 tokens | 2,097.562 MiB | 21,966 MiB | 19,667.298 MiB |
| scaled-FP8，同约 4 GiB budget | 220 / 56,320 tokens | 4,083.750 MiB | 23,952 MiB | 21,653.298 MiB |

由此得到两个不同口径的结论：

- **同容量**：allocated KV 下降 48.4375%；NVML 进程峰值下降 1,972 MiB，
  即 8.24%。不能写成“整卡显存减半”。
- **同约 4 GiB KV budget**：capacity 从 28,928 增至 56,320 tokens，
  提升 94.69%；NVML 峰值只增加 14 MiB。

按完整请求 `prompt + 128 output` 的页数估算，KV-limited resident 上限为：

| Case | Blocks / sequence | BF16 113 blocks | scaled-FP8 220 blocks |
|---|---:|---:|---:|
| H1 | 7 | 16 sequences | 31 sequences |
| H2 | 8 | 14 sequences | 27 sequences |

这些是 KV page 容量上限，不是实测 online goodput 或并发 SLO。

### 4.2 标准质量

冻结的 DocVQA、MuirBench、MVBench development/final 共六个 paired
non-inferiority cell 全部 PASS；每个 cell 500 个样本。示例：

- DocVQA ANLS：BF16 与 scaled-FP8 都为 `0.922558`；
- MuirBench accuracy：两者都为 `0.654`；
- 多数单元 exact output match 约 99%，MVBench final 为 190/190 exact。

这只证明冻结 Qwen3-VL-8B 与指定协议下的质量，不外推为任意模型、任意上下文或
“所有 FP8 都无损”。旧 unit-scale FP8 仍为 rejected baseline。

### 4.3 容量 profile 的 latency

scaled-FP8 使用 220 blocks、56,320-token capacity，KV bytes 与外部 BF16 约 4 GiB
budget 接近：

| Case | System | TPOT median | TTFT median | E2E median |
|---|---|---:|---:|---:|
| H1 | **Prism scaled-FP8** | **10.2363 ms** | **248.672 ms** | 1,612.314 ms |
| H1 | SGLang BF16 | 10.3520 ms | 284.844 ms | **1,600.005 ms** |
| H1 | vLLM BF16 | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| H2 | **Prism scaled-FP8** | **10.2588 ms** | **229.821 ms** | 1,686.067 ms |
| H2 | SGLang BF16 | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| H2 | vLLM BF16 | 10.5278 ms | 323.819 ms | **1,673.800 ms** |

TPOT 相对 SGLang 低 `1.06%–1.12%`，相对 vLLM 低 `2.55%–2.77%`。这说明
容量翻近一倍后仍保持受限场景的 TPOT 优势；不说明 FP8 路径比 Prism 自己的 BF16
更快。H1 对 SGLang、H2 对 vLLM 的 E2E 分别慢 0.77% 和 0.73%，因此 E2E 结论为
mixed，不能只报有利单元。

## 5. content-aware physical compaction 的判定

`visual_compact_scaled_fp8` 已接入 compile/Graph 路径，但当前只作为研究候选：

| Case | Logical → physical prompt | Pages | Active prompt bytes | TPOT |
|---|---:|---:|---:|---:|
| H1 | 1,618 → 834 | 7 → 4 | 129.938 → 74.250 MiB | 10.0199 ms |
| H2 | 1,667 → 883 | 7 → 4 | 同为 4 active pages | 10.0244 ms |

固定 pool 下 NVML 峰值与 pure scaled-FP8 相同，说明 page release 带来 allocator
capacity，而不是缩小已分配 pool tensor。组合路径尚无 DocVQA/MuirBench/MVBench
标准质量矩阵，输出 hash 也不同于 pure scaled-FP8，所以不进入外部 headline。

## 6. Profiling 结论与被拒绝候选

H1 scaled-FP8 node trace 的每 token 归因：

| Region | Approx. time / token |
|---|---:|
| 三类 BF16 weight GEMV | 7.734 ms |
| scaled paged attention | 0.888 ms |
| compiled output projection | 0.760 ms |
| compiled dynamic FP8 LM-head candidate | 0.369 ms |
| fused add/RMSNorm | 0.113 ms |
| selective top-k merge | 0.046 ms |
| scaled KV store | 0.041 ms |

主瓶颈已经是近带宽上限的 BF16 weight GEMV。attention 结构候选没有强行合入：

- GQA4 shared-KV split/merge 在 context 1,618/1,667 约慢 `1.88x–1.90x`；
- query-head split-K flash decoding 在同 context 约慢 `1.85x–1.88x`；
- context 4,096 仍约慢 `1.21x`；
- 根因是额外 launch 与 merge 开销，当前 32 个 query-head CTA 已足够。

两个候选代码均已移除，失败数据保留在
`data/p10_memory_profile/kernel_candidates/rejected_gqa4_and_split_k_dirty_79f9208.jsonl`。
这比为了“有 kernel 优化”保留负收益实现更符合最终项目目标。

## 7. Raw evidence

最终同提交 latency 集：

```text
data/p10_final_bounded_4779342/prism/
data/p10_final_bounded_4779342/vllm/
data/p10_final_bounded_4779342/sglang/
```

真实进程显存：

```text
data/p10_memory_profile/final_clean_59bb4ae/
```

content-aware 组合：

```text
data/p10_memory_profile/final_clean_79f9208/
```

NSYS：

```text
data/p10_memory_profile/final_profile_79f9208/
```

标准质量沿用已冻结 P9-C formal matrix，见 `VERIFICATION.md` 的 P9-C.1/P9-C.2。

## 8. 对外允许与禁止的表述

允许：

- “在 RTX 5090、Qwen3-VL-8B、TP1、batch1、output128 的冻结 H1/H2 offline
  CUDA Graph cell 中，Prism BF16 TPOT 比 SGLang 低 4.54%–4.83%，比 vLLM
  低 6.13%–6.27%。”
- “per-token/per-KV-head scaled-FP8 KV 在六个标准质量 cell 全 PASS；同容量
  KV bytes 下降 48.44%、NVML 进程峰值下降 8.24%；同约 4 GiB budget 的 token
  capacity 提升 94.69%，TPOT 仍低于冻结的 vLLM/SGLang baseline。”

禁止：

- “Prism 全面超过 vLLM/SGLang”；
- “KV 量化让整卡显存减半”；
- “容量翻倍已经证明 online 并发或 goodput 翻倍”；
- “content-aware + scaled-FP8 已通过标准质量”；
- “FP8 比 Prism BF16 更快”；
- 使用修复前的 1,665-token vLLM H2 或旧语义错误 output hash 排名。
