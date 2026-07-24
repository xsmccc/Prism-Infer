# P11 结果：Vision Graph、模态自适应视觉 KV 与动态页复用

> 实现提交：`c20fd8d`（Vision tensor CUDA Graph）、
> `4bc2094`（质量 runner 参数化）、`a4a06b3`（视频专用保留下限）
>
> 日期：2026-07-24
>
> 证据环境：Qwen3-VL-8B-Instruct revision
> `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`、RTX 5090
> `GPU-1bf42358-f8ac-c597-c1c9-30289ce22ba7`、driver `580.105.08`、
> PyTorch `2.11.0+cu130`、TP1。

本文补齐 P10 当时未完成的两项：多模态 prefill 的精确 CUDA Graph 归因，以及
`visual_compact_scaled_fp8` 的标准质量与动态页复用。P10 的跨框架 batch1
TPOT 数字仍按其冻结协议解释，本文不重新排名 vLLM/SGLang。

## 1. Vision tensor CUDA Graph

runtime 对重型、固定 shape 的 vision tensor region 建立 shape-safe CUDA Graph
cache。capture key 包含 shape、stride、dtype、device、max sequence length 和
segment ranges；replay 前复制动态输入，replay 后 clone 输出，避免 static tensor
alias。小于 2,048 patches 的输入保留 eager，防止单图场景负优化。

clean H1（8×448 image）、repeat9、output128：

| 路径 | Eager | Vision Graph | 变化 |
|---|---:|---:|---:|
| Engine TTFT median | 244.035 ms | 229.270 ms | -6.05% |
| Full TTFT median | 347.968 ms | 344.056 ms | -1.12% |
| Peak allocated | baseline | +88.266 MiB | Graph 静态缓冲成本 |

两条路径 output token hash exact，结束后 GPU memory 均为 0 MiB。H2 engine TTFT
为 `230.005 -> 232.843 ms`，差异 `+1.23%`，不构成加速证据；单图命中
`small_shape_fallback`，没有 capture。

H1 的 NVTX-only NSYS A/B 进一步说明收益来自 host launch/gap 收缩，而不是减少
计算或降低精度：

| 指标 | Eager | Vision Graph |
|---|---:|---:|
| Vision CPU range | 119.697 ms | 89.702 ms |
| Vision GPU busy | 48.754 ms | 48.579 ms |
| Vision GPU span | 116.687 ms | 101.859 ms |
| CUDA Runtime API calls | 5,025 | 1,185 |
| GPU kernels | 2,312 | 2,312 |

因此允许表述为“固定重型 vision shape 上减少 host launch 与 GPU gap，并使 H1
engine TTFT 下降 6.05%”。不能称为减少了 GPU 算术量，也不能外推为 H2/单图均加速。

## 2. 模态自适应视觉 KV 策略

最终研究策略为：

```text
keep_ratio = 0.6
image / mixed minimum = 768 visual tokens
video-only minimum = 256 visual tokens
selection = uniform
KV storage = per-token/per-KV-head scaled FP8
```

设置 768-token 图像安全线，是因为 DocVQA 单图样本只有 `171–588` 个 visual
tokens；在这些 OCR 请求上不做视觉裁剪，只应用已验证的 scaled-FP8。H1/H2 和
多数 MuirBench 多图请求仍会真实裁剪。MVBench 每条约 `320–360` 个 visual
tokens，单独使用 256-token 视频下限，保留约 `71%–80%`。

`effective_min_keep_tokens` 被写入每个 pruning decision，便于审计实际采用的是
图像还是视频下限。混合 image+video 请求使用更保守的图像下限。

## 3. 正式质量证据

以下都是 paired、clean、formal development 运行；比较器重新计算 reference
score，并使用预注册 non-inferiority margin `-0.01`：

| Dataset | Samples | Baseline | Candidate | Paired delta | 95% bootstrap CI | Exact outputs | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| DocVQA ANLS | 200 | 0.924640 | 0.925308 | +0.000669 | [0, 0.002006] | 198/200 | PASS |
| MuirBench accuracy | 200 | 0.690000 | 0.690000 | 0 | [0, 0] | 198/200 | PASS |
| MVBench accuracy | 97 | 0.608247 | 0.608247 | 0 | [0, 0] | 97/97 | PASS |

DocVQA/MuirBench 在 clean `4bc2094` 上验证 768-token 图像下限；MVBench 在
clean `a4a06b3` 上验证新增的 256-token 视频下限。两次变更之间只新增了
video-only floor 及其透传/审计，图像默认语义不变。

三组 candidate 的 KV pool 均为：

```text
BF16 baseline:       1,509,949,440 bytes
scaled-FP8 payload:    754,974,720 bytes
FP32 scales:            23,592,960 bytes
candidate total:        778,567,680 bytes
saving:                       48.4375%
```

这个 48.4375% 是 allocated KV pool bytes，不是整进程或整卡显存下降比例。

## 4. 动态页回收与真实复用

冻结 cell：H1、batch2、output128、eager、page256、仅 11 个 KV blocks、
warmup1/repeat3。每个 dense H1 prompt 为 1,618 tokens，需要 7 pages；两个 dense
prompt 共需 14 pages，因此无法同时驻留在 11-page pool。

### 4.1 调度与页表证据

未压缩：

```text
dense prompt pages: 14
pool pages: 11
released pages: 0
decode batch-size counts: batch1 = 762
```

两个记录的页表共享 `[4,5,6]`，说明第二个请求在第一个请求之后复用页，而不是同时
驻留。

压缩后，每个 H1 将 `1,568 -> 941` visual tokens、`1,618 -> 991` physical
prompt tokens、`7 -> 4` pages：

```text
final active prompt pages: 8
released prompt pages: 6
request 1 old/new: [6,4,3,2,1,0,5] -> [6,4,3,2]
request 2 old/new: [10,9,8,7,1,0,5] -> [10,9,8,7]
decode batch-size counts: batch1 = 6, batch2 = 378
```

第一页表释放的 `[1,0,5]` 被第二个请求的 prefill 重新分配，随后再次释放。这是
页 ID 级的动态复用证据，不只是“压缩记录里页数变小”。

### 4.2 该容量受限 cell 的结果

| 指标 median | Dense BF16 | Compact scaled-FP8 | 变化 |
|---|---:|---:|---:|
| E2E | 9,251.057 ms | 5,824.340 ms | -37.04% |
| Requests/s | 0.2162 | 0.3434 | +58.83% |
| Decode tokens/s | 29.6708 | 49.2762 | +66.08% |
| Allocated KV pool | 415,236,096 B | 214,106,112 B | -48.44% |

这里的吞吐提升来自“页回收使两个请求并发 decode”与 scaled-FP8 的组合，不应写成
单请求 kernel 加速。合成 H1 的压缩输出与 dense 不 token-exact；质量结论必须引用
上一节真实 DocVQA/MuirBench/MVBench gate。

固定大小的 KV pool tensor 在逐请求 compact 时不会向 CUDA driver 缩容；动态回收
的收益是 block ID 回到 allocator 并被其他请求使用。真实进程显存下降继续引用 P10
同容量 NVML `23,938 -> 21,966 MiB`（-8.24%），而不是把本节页回收误写成 NVML
按页下降。

## 5. Raw evidence

```text
data/p11_multimodal_ttft/vision_graph_clean_c20fd8d_gpu_1bf42358/
data/p11_multimodal_ttft/vision_graph_nsys_nvtx_clean_c20fd8d_gpu_1bf42358/
data/p11_compaction_quality/formal_dev_clean_4bc2094_gpu_1bf42358/
data/p11_compaction_quality/modality_clean_a4a06b3_gpu_1bf42358/
data/p11_dynamic_reclaim/h1_batch2_blocks11_a4a06b3.jsonl
```

## 6. 允许与禁止的表述

允许：

- 固定重型 H1 vision shape 的 Graph replay 保持 token exact，engine TTFT
  下降 6.05%，Runtime API calls 从 5,025 降至 1,185。
- 图像 768 / 视频 256 的模态自适应保留策略与 scaled-FP8 在三项 formal
  development gate 全 PASS；MVBench 97/97 output exact。
- 11-page 容量受限 H1 batch2 中，回收页被第二请求真实复用，并使 decode 从串行
  batch1 转为主要 batch2。

禁止：

- “Vision Graph 对所有图片/视频都加速”。
- “视觉压缩无损”或“任意模型/数据集质量不下降”。
- “页回收让固定 KV pool 的 NVML 显存按页立即下降”。
- 把容量受限 batch2 的 +58.83% requests/s 写成单请求 TPOT 或通用 online
  goodput 提升。
