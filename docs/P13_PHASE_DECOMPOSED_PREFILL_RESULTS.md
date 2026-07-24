# P13 Phase-Decomposed Multimodal Prefill：实现、证伪与删除

> 日期：2026-07-24
> 基线提交：`96f46c4ee624d3fd5df22e9452ad18f285250898`
> 分支：`codex/torch28-p9d`
> 结论：候选没有通过 loaded H3 门禁，运行时代码已全部删除；本文只保留问题、
> 实现、数据和取舍证据。

## 1. 问题与假设

P12 的 600-request rate-4 trace 已经证明：

- Prism raw output throughput 为 `239.607 tok/s`，距 vLLM/SGLang 仅
  `0.78%/0.76%`；
- 但 H1/H2 visual prefill 是约 `180–210 ms` 的原子执行段，而 decode batch
  median 约为 `14.13 ms`；
- FCFS 每个 decode batch 后都可插入一次 prefill，导致 loaded TPOT 和 goodput
  远差于外部框架。

P13 的假设是：先独立完成视觉编码并缓存主视觉特征与 DeepStack 特征，再把语言
prefill 切成可调度 chunk，可能在不改变 token 的前提下缩短 decode 的 head-of-line
blocking。

## 2. 候选实现

候选只支持 TP1，并默认关闭。执行阶段为：

```text
VISION
  └─ 完整 image/video encoder，只执行一次
       ├─ main visual embeddings
       └─ 3-layer DeepStack embeddings

PREFILL chunk 0..N
  └─ 按 prompt 中视觉占位符位置切片并注入缓存 embedding
       └─ 每个 chunk 后允许 scheduler 运行 decode

DECODE
  └─ 继续使用 retained scaled-FP8 KV、torch.compile 与 CUDA Graph
```

实现中额外记录：

- `vision_stages_scheduled`；
- `phase_decomposed_prefill_chunks`；
- `peak_cached_visual_tokens`；
- BF16 prefill-only KV workspace 的 allocation、peak bytes 和最终释放状态。

### 2.1 为什么需要 BF16 prefill workspace

第一版 chunked prefill 从永久 scaled-FP8 KV cache 读取历史，而完整 prefill 的
attention 使用 BF16 K/V。这会把“调度切块”变成“改变模型数值路径”。H2 在相同
prompt/arrival hash 下，第 4 个生成 token 即发生分叉。

候选随后增加临时 BF16 workspace：

```text
[K/V, decoder layer, prompt token, KV head, head dim]
```

每个 chunk 同时写永久 FP8 cache 与临时 BF16 history；最后一个 prefill chunk 完成后
立即释放 workspace，再执行既有 visual KV physical compaction。H1/H2 单请求峰值
workspace 分别为 `238,583,808 / 245,809,152 bytes`。

### 2.2 视频确定性 fallback

即使使用 BF16 history，H2 的 512/1024-token chunk 仍会因为完整 FlashAttention 与
分块 FlashAttention 的计算 shape/舍入不同而在低 margin token 上分叉。因此最终
候选对视频只拆分 VISION，不切语言 prefill；图像请求才使用 language chunk。

这不是低 margin 运行时重算，也不增加第二次完整 prefill，而是基于冻结 H2 证据的
确定性 modality fallback。

## 3. Correctness 与机制门禁

所有完成的单请求 A/B 均使用相同模型、GPU、prompt hash、arrival hash、greedy
64 tokens、scaled-FP8 + visual compaction + compile/Graph 路径。
候选最终 focused regression 为 `23 passed, 2 skipped`，并通过 Ruff 与
`py_compile`；这些只证明实现合同，没有替代下方 8B correctness 和 loaded gate。

| Case | Candidate | Token exact | 非 decode 阶段 | 结论 |
|---|---|:---:|---|---|
| H1 | phase 512 + BF16 workspace | PASS | `113.448 ms vision` + `59.149/59.414/57.711/51.020 ms` prefill | 机制正确，单请求 TTFT `+16.5%` |
| H1 | phase 1024 + BF16 workspace | PASS | `119.412 ms vision` + `88.782/64.154 ms` prefill | 进入 loaded 筛选 |
| H2 | phase 512，直接读 FP8 history | FAIL | 4 个 language chunks | 第 4 个生成 token 分叉 |
| H2 | phase 1024，直接读 FP8 history | FAIL | 2 个 language chunks | 第 4 个生成 token 分叉 |
| H2 | phase 512 + BF16 workspace | FAIL | 4 个 language chunks | 仍在第 4 个 token 分叉 |
| H2 | 独立 VISION + 单块 language | PASS | `101.327 ms vision` + `130.786 ms` prefill | workspace 为 0，保留为 fallback |

H1 512 将单请求原 `247.929 ms` 原子 prefill 拆成最大 `113.448 ms` 的执行段；
H1 1024 的最大执行段为 `119.412 ms`。两者都没有增加 NVML headline 桶，
但 workspace 是真实临时成本，不能因为 NVML 10 ms 采样没有看到更高桶就忽略。

## 4. 混合队列暴露的调度问题

第一轮 H3-primary 运行在 sequence 1 失败：队首 text 与后续 image 被同一个 admission
pass 接纳，image 因而绕过 VISION 直接进入 language prefill。

修复后的 invariant 是：

> phase 模式下，携带视觉 payload 且尚未完成视觉编码的请求不能成为 prefill
> candidate。

新增 mixed-queue test 后，text 可以完成当前 prefill batch，下一次合法调度才发布
image VISION plan。这个问题说明单类 benchmark 不能替代真实混合队列。

## 5. 同 trace loaded 筛选

筛选协议：

- frozen H3 primary：40% text、30% single image、30% H1；
- Poisson rate 4、seed `20260717`、60 requests、warmup 10；
- output 64、max resident 8、约 4 GiB scaled-FP8 KV pool；
- trace SHA256
  `61387d317efc71a86214990abc5dcf719b6a396a3a6fefd657d3fd9710a01d0a`；
- prompt SHA256
  `a25be0679f8f320e28fcb54342466986c681208a22c4fc6469fca0ae9881901f`。

这些是 dirty candidate-selection 运行，不是 formal headline。

| 指标 | Baseline | Phase 512 | 变化 | Phase 1024 | 变化 |
|---|---:|---:|---:|---:|---:|
| Output tok/s | 215.694 | 201.780 | -6.45% | 212.958 | -1.27% |
| Class-aware goodput tok/s（headline） | 21.569 | 0.000 | -100% | 14.197 | -34.18% |
| Generic goodput req/s（辅助） | 2.415 | 0.525 | -78.24% | 2.218 | -8.16% |
| Queue p50 | 159.848 ms | 589.396 ms | +268.7% | 234.339 ms | +46.6% |
| TTFT p50 | 359.999 ms | 751.780 ms | +108.8% | 419.556 ms | +16.5% |
| TPOT p50 | 27.670 ms | 33.858 ms | +22.4% | 28.218 ms | +1.98% |
| E2E p50 | 2,179.148 ms | 2,858.588 ms | +31.2% | 2,164.356 ms | -0.68% |
| Generic good requests | 43/60 | 10/60 | -33 | 40/60 | -3 |

512 产生 `36 vision + 111 prefill` batches；1024 产生
`36 vision + 70 prefill` batches；baseline 只有 49 个 prefill batches。更短的单次
阻塞被更多 stage、workspace copy 和更慢的 prompt completion 抵消。

1024 的确改善部分极端 tail：

- queue max `1200.134 -> 987.092 ms`；
- TTFT max `1381.706 -> 1233.718 ms`；
- E2E max `3064.001 -> 2663.899 ms`；
- mixed-batch prefill max `446.229 -> 119.489 ms`，另有 vision max
  `133.833 ms`。

但项目门槛不是只优化一个 tail 数字。median、goodput 和 raw throughput 均未形成
可交付收益，因此不进入 600-request formal。

loaded A/B 的逐请求 token 序列也不是全 exact：512 为 `36/60`，1024 为 `39/60`。
分叉只出现在 text/single-image，不在 H1，符合不同 decode batch composition
引起 low-margin argmax 分叉的现象；它仍然意味着候选没有通过当前 token-hash
发布门禁。

## 6. 最终决定

P13 候选被全部删除，retained runtime source 精确恢复到 clean `96f46c4`；后续只增加
交付文档。不保留：

- `VISION` BatchPhase；
- Sequence 侧 GPU visual embedding cache；
- BF16 prefill workspace；
- modality fallback 与 phase batching policy；
- benchmark CLI 和运行时 counters。

保留：

- raw JSON/stdout/stderr；
- 本文的根因、失败和取舍；
- 本地控制目录 `remote_edit/p13_phase/new/` 中的候选源码快照。

不把默认关闭的负收益代码留在主线，是最终交付质量的一部分。

## 7. 允许与禁止的表述

允许：

- “实现并验证了 phase-decomposed multimodal prefill 原型；H1 将最大原子执行段从
  约 248 ms 降到约 119 ms，并保持单请求 64-token exact。”
- “profiling 发现永久 FP8 KV 不能直接作为 chunked prefill history，因此增加了
  约 228–234 MiB BF16 临时 workspace；视频仍因 FlashAttention shape 的低 margin
  分叉采用确定性 fallback。”
- “同 trace loaded test 中 1024 chunk 的 class-aware goodput 从 `21.569` 降至
  `14.197 tok/s`（-34.18%），TTFT p50 上升 16.5%，因此删除候选，没有扩跑或
  包装成优化。”

禁止：

- “Prism 已实现或默认启用 phase-decomposed/chunked multimodal prefill。”
- “P13 提升了 online goodput/TPOT。”
- “缩短最大 prefill block 等价于系统吞吐或 SLO 提升。”
- “P13 使 Prism 在 loaded H3 超过 vLLM/SGLang。”

## 8. 面试讲法

最重要的结论不是“chunk 没用”，而是三个系统层次必须一起看：

1. **模型数值合同**：永久 FP8 KV 适合 decode storage，不等于适合构造后续
   prefill hidden states。
2. **调度合同**：VISION 与 PREFILL 分开后，混合队列必须阻止视觉请求绕过编码阶段。
3. **端到端排队**：把 248 ms 拆成多个 50–120 ms block 只改善可抢占粒度；如果总
   GPU work 和 stage 数增加，steady-state queue 仍会变差。

因此实验流程是：单请求 exact → 混合队列 invariant → 同 trace loaded 筛选 →
决定是否进入 600-request formal。P13 在最后一道筛选失败，所以删除实现。

## 9. Evidence ledger

| Artifact | SHA256 | 用途 |
|---|---|---|
| `h1_off_chunk2048_dirty.json` | `2d5e7b625216ce562ced04c3d979366ee0424754db0c23aecc92fe5ca8e79dc7` | H1 exact baseline |
| `h1_phase512_bf16ws_final_dirty.json` | `3f44252d7d23d2cdd6636a706f379f51c19278ba0c593b411f3581235edd22ad` | H1 512 mechanism |
| `h1_phase1024_bf16ws_dirty.json` | `5740baf839a4518d84f1116ec8378ce67c9a21f7ad77a3a38f8706b2bc877752` | H1 1024 selection |
| `h2_off_chunk2048_dirty.json` | `286f571ed271534a9ab333426c85bb893c1576a6d85f923eacf5edd7b2605830` | H2 exact baseline |
| `h2_phase512_dirty.json` | `31a56752afe7062b93892433cfde082a00bb3bcdaf7dd36efbf373d33594a56c` | FP8-history mismatch |
| `h2_phase512_bf16ws_v3_dirty.json` | `615e3f9299657f1dcb02eaeb95801eb29f94ac98ff090e386e6ad413f5a0be6d` | BF16-history mismatch |
| `h2_phase512_video_atomic_no_ws_dirty.json` | `dbe7992c3e470851728a68958ca0a529293bd7b8fda5379a50e817e9381051ca` | video fallback exact |
| `h3_primary_r4_n60_off_dirty.json` | `4d20a23775e5c85d2fc2b1a3926dd81c9e66968fdcd90d55b4081272546ad2b3` | loaded baseline |
| `h3_primary_r4_n60_phase512_v2_dirty.json` | `8d6755a9c49b0dd28bedd3a2b882452166aa4b06fb159374fbc742e1f4fee99f` | rejected 512 |
| `h3_primary_r4_n60_phase1024_dirty.json` | `0ee0540b500207d7bf153105047fd909fb036e190d8c9f882c7bdf754f682b71` | rejected 1024 |
| `h3_primary_r4_n60_phase512_dirty.stderr` | `382bbdbf5ea2a625ba9ce770825e089e7ac5d05514c1539dba7a7dc918f1d8f3` | mixed-queue failure |

服务器路径统一为 `data/p13_phase/tuning/`。候选已从 retained source 删除，所以这些
artifact 用于审计历史，不作为当前 HEAD 可直接重跑的正式 benchmark。
