# Prism-Infer 秋招最终交付

> 冻结日期：2026-07-30
> 冻结分支：`codex/p14-loaded-decode`
> P12 closure 文档点：`96f46c4ee624d3fd5df22e9452ad18f285250898`
> 正式 runtime commits：以 P10/P11/P12 各自 evidence ledger 为准（P12 rate-4
> 为 `921de81/e883de5`）
> 最终文档提交：以本文件所在 clean HEAD 为准

## P17 内容寻址压缩多模态前缀缓存

P17 修复了 P16 最重要的泛化边界：语义相同但每次重新解码得到的新媒体对象也能命中。
缓存 key 不使用对象地址，而是以 length-delimited SHA256 覆盖模型/processor、
媒体像素与布局，并在 compacted prefix KV 层绑定截至最后视觉占位符的精确 prompt
token。它复用 processor 张量、Vision/DeepStack 和物理压缩的 scaled-FP8 KV 页；
同媒体换问题时只重建视觉占位符之后的文本，不复用语言后缀或生成 token。

前缀 KV 缓存与现有 Paged KV/压缩主线共用真实页面：full page 只读共享，partial
tail 首次命中执行有效行 CoW，之后把派生尾页留在同一预算内供后续请求租用；引用计数、
eviction、M-RoPE logical/physical metadata 和退出释放均进入同一生命周期。

fresh-object n60 对比结果：

| Repeat | Prism raw / Goodput | vLLM raw / Goodput | 结论 |
| ---: | ---: | ---: | --- |
| 0% | 216.188 / 133.316 | 223.079 / 215.643 | vLLM 领先 |
| 50% | 217.083 / 206.228 | 223.521 / 219.796 | vLLM 领先 |
| 75% | 224.279 / 224.279 | 225.112 / 225.112 | 相差 0.37% |
| 100% | 224.369 / 224.369 | 225.004 / 225.004 | 相差 0.28% |
| 100% 换问题 | 224.301 / 224.301 | 225.004 / 225.004 | 相差 0.31% |

100% repeat 的 Prism raw/Goodput 相对可用 SGLang cache-on 参考高
`0.82%/4.30%`。正式 n600 结果为 `241.428 tok/s`、600/600 SLO、
TTFT/TPOT p50 `146.418/13.041 ms`，相对 P16 Goodput 提升 `6.68%`；
H1/H2 exact、scaled-FP8 KV `-48.44%`、视觉页回收、零失败和退出显存释放保持。

结论边界很明确：P17 在高重复多模态 workload 中与 vLLM 统计持平并超过当前可用
SGLang 参考；unique/低重复仍由 vLLM 领先。完整证据见
[P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS](P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS.md)。

## P16 稳态多模态 SLO Goodput 更新

P16 将 P12/P15 的“raw throughput 接近、loaded Goodput 落后”推进为一个有明确
场景边界的 Goodput 胜出：

- 在冻结 600-request、Poisson rate-4、seed20260717 workload 上，以 per-request
  TTFT SLO 推导 latest safe start，按 prefill cost tier 预留 `120/250/700 ms`；
- deadline order 只隔离 tight text 与 visual prefill，保留 visual-visual batching；
- 已在执行的 heavy prefill 只允许在 ViT block/语言层语义边界被到期的低成本请求打断；
- 用 128-entry exact CPU processor LRU 与 256 MiB exact GPU
  Vision Encoder/DeepStack LRU 消除同一媒体对象的重复预处理与 ViT；
- language prefill、scaled-FP8 KV store、视觉物理压缩、CUDA Graph decode、精确
  rerank、sampling 与 64-token 生成仍逐请求完整执行。

正式 n600 结果：

| Result | Raw tok/s | Goodput tok/s | Good requests | TTFT p50 | TPOT p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P15 n600 | 239.387 | 67.427 | 169/600 | 892.717 ms | 12.764 ms |
| P16 cache-off | 241.037 | 171.538 | 427/600 | 305.479 ms | 16.270 ms |
| **P16 cache-on** | **241.184** | **226.311** | **563/600** | 275.225 ms | 14.329 ms |
| vLLM | 241.489 | 212.108 | 527/600 | 180.244 ms | 14.283 ms |
| SGLang | 241.447 | 196.779 | 489/600 | 258.192 ms | 14.392 ms |

P16 Goodput 相对 vLLM/SGLang 高 `6.70%/15.01%`，相对 cache-off 高
`31.93%`；600/600 完成、无失败，H1/H2 token hash exact，KV pool 仍为
`4,282,122,240 B`（相对 BF16 `-48.44%`），并释放 480 个视觉物理 block。

该结论只覆盖 warm、单进程、三个 exact media object 重复的 workload。外部系统未
配置等价的 Prism Vision-output cache；因此这是同一 workload 的 feature-enabled
system result，不是 unique-media、冷启动、网络 content-hash cache 或全面 serving
排名。n600 TPOT 比 SGLang 低 `0.44%`、比 vLLM 高 `0.32%`，只能称近似持平。
完整结果见 [P16_STEADY_STATE_GOODPUT_RESULTS](P16_STEADY_STATE_GOODPUT_RESULTS.md)。

## 0. P15 loaded-serving 更新

P15 解决了 P14 “TPOT 达标但 TTFT/吞吐/Goodput 退化”的主要取舍。最终保留：

- 250 ms deadline-aware prefill coalescing：至少三个请求时恢复 atomic batching，
  underfilled/超时批次继续在 ViT block 与语言层边界协作执行；
- 显式 `--online-cpu-intraop-threads=8`，避免单个异步媒体预处理任务使用默认
  104 路 CPU intra-op 并行而饿死 CUDA Graph launch thread；
- P14 的 B1--B8 exact-batch Graph、guarded FP8 candidate + FP32 rerank、
  scaled-FP8 KV 与视觉物理压缩。

冻结 60-request trace 的四次中位数为 `215.628 tok/s`、TTFT
`776.863 ms`、TPOT `12.490 ms`、class-SLO Goodput `75.566 tok/s`。
TPOT 相对 bounded vLLM/SGLang references 低 `8.56%/13.86%`；raw、TTFT 与
Goodput 仍落后，不能写成全面 online 胜出。H1/H2 64-token hash exact，KV pool
相对 BF16 仍节省 `48.44%`。

完整结果、Profiler、失败候选与面试叙事见
[P15_BALANCED_MULTIMODAL_RESULTS](P15_BALANCED_MULTIMODAL_RESULTS.md)。

## 1. 项目定位

Prism-Infer 是一个面向 Qwen3-VL-8B 的单机多模态推理研究引擎。它不是把 vLLM
封装一层；当前已包含原生 HTTP/SSE 服务边界，但不是 OpenAI-compatible 或多机
生产服务。核心价值是从模型语义、Paged KV、GPU kernel、compiler/Graph 到
arrival-driven scheduler 建立完整、可审计的优化闭环。

秋招中最有辨识度的三条主线：

1. **torch.compile + CUDA Graph decode**：只编译无状态热点，把有状态 Paged KV
   留在受审计 runtime 边界；用 Nsight Systems 找到并消除 per-token 完整 LM-head
   FP32 cast。
2. **多模态 KV 容量**：实现 per-token/per-KV-head scaled-FP8 全生命周期，以及
   content-aware visual KV physical compaction、页回收和 logical M-RoPE /
   physical KV position 分离。
3. **真实系统取舍**：用冻结 prompt/arrival/SLO hash 做 vLLM/SGLang 同协议对比；
   对 concurrency cap、cadence、deadline、adaptive guard、phase-decomposed
   prefill 等负收益候选保留数据并删除代码。

## 2. Retained implementation

- 自实现 Qwen3-VL text/vision、M-RoPE、DeepStack 与 decoder 主路径；
- Request FSM、continuous batching、immutable BatchPlan、Executor、Paged KV
  manager、swap/CoW、metrics；
- 原生 HTTP/SSE JSON/streaming、bounded ingress、断连取消与单 engine owner；
- BF16 与 scaled-FP8 paged KV store/decode/Graph；
- visual KV physical compaction、尾页回收、动态页复用；
- batch1 `torch.compile` 无状态 decode 子图；
- decode CUDA Graph fixed buckets；固定 shape Vision tensor Graph 仅保留为受限能力，
  dynamic mixed-shape serving 默认关闭；
- guarded FP8 LM-head candidate + FP32 exact rerank；
- packed gate/up projection；
- schema 化 correctness、quality、TTFT、TPOT、E2E、goodput、NVML 与 raw evidence。

没有保留：

- unit-scale FP8 质量失败路径作为最终 profile；
- GQA4 merge、split-K、QKV packed、phase-decomposed prefill；
- vision-aware scheduler 只保留为显式实验策略，不作为默认 goodput 优化；
- TP2、OpenAI-compatible API、多机 serving、投机解码、PD 分离、megakernel、
  权重/激活量化的未验证 claim。

## 3. 最终结果

### 3.1 冻结 batch1 offline latency

RTX 5090、Qwen3-VL-8B、TP1、greedy output128、warmup2/repeat5、三引擎 prompt
token exact：

| Case | Prism BF16 | SGLang BF16 | vLLM BF16 |
|---|---:|---:|---:|
| H1 TPOT | **9.8821 ms** | 10.3520 ms | 10.5276 ms |
| H2 TPOT | **9.8680 ms** | 10.3689 ms | 10.5278 ms |

Prism TPOT 相对 SGLang 低 `4.54%–4.83%`，相对 vLLM 低
`6.13%–6.27%`。这是受限 batch1 offline Graph 结果，不是全面排名。

### 3.2 scaled-FP8 KV Pareto

| 口径 | 结果 |
|---|---|
| 六项 formal quality | DocVQA/MuirBench/MVBench development/final 全 PASS |
| 同容量 allocated KV | BF16 的 `0.515625x`，节省 `48.4375%` |
| 同容量 process NVML peak | `23,938 -> 21,966 MiB`，下降 `8.24%` |
| 同约 4 GiB KV budget | `28,928 -> 56,320 tokens`，容量 `+94.69%` |
| 容量 profile TPOT | 相对 SGLang 低 `1.06%–1.12%`，相对 vLLM 低 `2.55%–2.77%` |

量化目标是容量，不是加速 Prism 自身 BF16；E2E 结论为 mixed。

### 3.3 visual compaction 与页复用

- 模态自适应策略：keep 0.6，image/mixed floor 768，video floor 256；
- DocVQA/MuirBench/MVBench development formal gate 全 PASS；
- H1 batch2、11-page 压力 cell 中，每请求 prompt `7 -> 4 pages`；
- 第一请求释放的 page IDs 被第二请求 prefill 真实复用；
- 378/384 decode steps 从 batch1 转为 batch2，requests/s `+58.83%`。

这是容量受限 cell，不是通用 TPOT 或 online goodput 提升。

### 3.4 arrival-driven online closure

600-request、rate-4、conditional-video H3：

| System | Output tok/s | SLO goodput tok/s | Good requests | NVML peak | KV tokens |
|---|---:|---:|---:|---:|---:|
| vLLM | 241.489 | 212.108 | 527/600 | 23,874 MiB | ~29,127 |
| SGLang | 241.447 | 196.779 | 489/600 | 23,560 MiB | 28,928 |
| Prism | 239.607 | 65.093 | 163/600 | 24,456 MiB | 56,320 |

Prism raw throughput 距两者不到 `0.8%`，KV-token capacity 为约
`1.93x/1.95x`，但 loaded goodput 明显落后。保留的 bounded 特点是 H1/H2 TTFT
p50 比 SGLang 低 `7.5%/51.0%`；不能扩写成整体 online 胜出。

P13 进一步实现 phase-decomposed prefill 原型。1024 chunk 将 mixed prefill max
`446.229 -> 119.489 ms`，但同 trace class-aware goodput 从 `21.569` 降至
`14.197 tok/s`（-34.18%）、TTFT p50 `+16.5%`、TPOT p50 `+1.98%`，因此候选
被删除。

### 3.5 原生 HTTP/SSE 与视觉调度

新 RTX 5090 上，Prism 60-request network/in-process raw throughput 为
`214.503/214.398 tok/s`，差 `0.049%`，说明当前 loopback HTTP/SSE 不是主要瓶颈。
profile 将 loaded stall 定位到 `204–232 ms` 的 H1/H2 原子视觉 prefill。

600-request frozen H3 中，dynamic Vision Graph on 产生 2 个异常首 token，其中一个
请求 64 个 token 全为 0。默认关闭后异常 `2→0`、显存
`24,456→24,018 MiB`、raw `+0.88%`、TPOT p50 `-1.36%`，但 goodput
`-15.39%`，所以这是 correctness/stability 修复而不是全面加速。

有界 vision-aware 策略把 TTFT p50/p90 改善 `14.51%/3.91%`，但 goodput
下降 `15.79%`，最终默认仍为 FCFS。完整证据与面试复盘见
[NETWORK_SERVING_RESULTS](NETWORK_SERVING_RESULTS.md)。

## 4. 推荐简历 bullets

选择三条即可：

- 面向 600-request Poisson 多模态稳态负载，设计基于请求成本与 TTFT slack 的
  调度器，并实现 256 MiB exact Vision Encoder/DeepStack LRU；在重复媒体
  workload 上将 class-SLO Goodput 从 `171.54` 提升至 `226.31 tok/s`
  （`+31.9%`），同协议结果高于 vLLM `6.70%`、SGLang `15.01%`，同时保持
  H1/H2 token hash exact、scaled-FP8 KV bytes `-48.44%` 与视觉物理页回收。
- 自实现 Qwen3-VL-8B text/vision、M-RoPE、DeepStack、Paged KV 与 continuous
  batching 主路径；用 HF module/logits/PPL、greedy、CUDA Graph、标准多模态任务和
  token/prompt hash 建立分层 correctness gate。
- 设计 per-token/per-KV-head scaled-FP8 KV 与 content-aware visual KV physical
  compaction，覆盖 store、paged decode、CoW、swap、Graph 和页回收；六项冻结质量
  cell 全 PASS，同容量 KV bytes `-48.44%`、进程 NVML peak `-8.24%`，同约
  4 GiB budget 容量 `+94.69%`。
- 用 Nsight Systems 定位 per-token 完整 LM-head BF16→FP32 cast，将 logits CUDA
  median `4.068 -> 0.762 ms`、五类 workload TPOT 提升 `1.216x–1.280x`；冻结
  H1/H2 batch1 cell 中 BF16 TPOT 比 SGLang 低 `4.54%–4.83%`、比 vLLM 低
  `6.13%–6.27%`。

可替换的 serving 版本：

- 构建 600-request Poisson 多模态 H3 benchmark，统一 prompt/arrival/SLO hash、
  TTFT/TPOT/goodput/NVML；定位 loaded 瓶颈为 `180–210 ms` 原子 visual prefill，
  实现并证伪 512/1024 phase prefill，因 goodput/median 退化删除候选。

## 5. 60 秒讲解

我做了一个 Qwen3-VL-8B 的轻量多模态推理引擎，自己实现 Vision、M-RoPE、
DeepStack、decoder、Paged KV 和调度，HF 只作为输入与数值 reference。第一条优化线
是 decode：我把有状态 KV 留在 runtime，只用 torch.compile 编译无状态热点，再把
model forward、精确 logits 选择和 greedy decode 放进 CUDA Graph。通过 Systems
trace 找到每 token 整张 LM-head 转 FP32 的问题，修复后 TPOT 提升
1.216 到 1.280 倍；冻结 H1/H2 中 BF16 TPOT 小幅低于 vLLM/SGLang。

第二条线是多模态 KV：我实现 per-token/per-head scaled-FP8 和视觉 KV 物理压缩，
不是逻辑 mask，而是重排 K/V、更新 page table、释放尾页并让后续请求复用。六项标准
质量 cell 全通过，同预算 KV capacity 提升 94.69%。我也做了 600-request online
对比：早期 raw throughput 已接近两家，但 loaded Goodput 落后。phase-chunk 虽然
缩短单次阻塞，却增加总工作，所以我删除了候选；随后用 class SLO slack 做调度，
只隔离 tight text 与 visual batch。P16 先验证同对象 Vision/DeepStack 复用；
P17 再升级为内容 SHA256，并把复用延伸到 compacted scaled-FP8 prefix KV 的真实
页面、CoW 和尾页池。fresh-object 高重复负载超过当前 SGLang 参考并与 vLLM 相差
0.3% 左右，n600 达到 600/600 SLO；unique/低重复仍落后，我不会把它讲成全面排名。

## 6. 面试主线

推荐按以下顺序讲：

1. 为什么先自实现与做 correctness，而不是先跑 benchmark；
2. logical pruning 为什么不能叫 KV compression；
3. scaled-FP8 的 scale 为什么必须进入 CoW/swap/compaction/Graph 全生命周期；
4. torch.compile 为什么只覆盖无状态区域；
5. CUDA Graph 为什么减少 launch，但不消灭 GEMV；
6. profiler 如何把 LM-head cast 从现象定位到 root cause；
7. capacity、NVML、TPOT、goodput 为什么是四个不同指标；
8. 为什么短 trace 的 guard/phase candidate 会在 steady state 失败；
9. 哪些结果真的超过外部 baseline，哪些明确没有。

## 7. 必须主动说明的边界

- “超过 vLLM/SGLang”的离线结论仍只用于冻结 H1/H2 batch1 TPOT；
- online 高重复结论只用于 P17 单进程 fresh-object frozen trace；必须同时说明
  vLLM 是高重复统计持平、SGLang 单机没有功能完全等价的 multimodal global cache，
  且 0--50% repeat 仍由 vLLM 领先；
- P17 的“持久”是同一 engine 生命周期内跨请求，不是跨进程/跨节点网络缓存；
- `-48.44%` 是 allocated KV pool，不是整卡显存；整进程实测为 `-8.24%`；
- `+58.83% requests/s` 是 11-page 容量受限 batch2；
- Graph speedup 是 Prism internal eager→Graph；
- 不声称 TP2、网络 serving、投机解码、PD 分离或权重量化已完成。

## 8. 证据与复现入口

- `P10_FINAL_RESULTS.md`：compile/Graph 与 scaled-FP8 Pareto；
- `P11_MULTIMODAL_COMPACTION_RESULTS.md`：Vision Graph、标准质量、动态页复用；
- `P12_ONLINE_GOODPUT_RESULTS.md`：600-request external online closure；
- `P13_PHASE_DECOMPOSED_PREFILL_RESULTS.md`：phase 候选实现与否决；
- `P16_STEADY_STATE_GOODPUT_RESULTS.md`：P16 调度、视觉复用、正式对比和边界；
- `P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS.md`：内容指纹、fresh-object 矩阵、
  压缩前缀页生命周期、失败候选和 n600 闭环；
- `CLAIMS.md`：允许/限制/禁止的唯一口径；
- `REPRODUCIBILITY.md`：环境、命令与 artifact contract；
- `APPLICATION_MATERIALS.md`：简历、STAR、面试追问。

## 9. 最终完成标准

- retained source 无已知负收益实验代码；
- 工作树 clean，分支和 commit 明确；
- P10/P11/P12/P16/P17 正式 artifact 与 rejected artifact 均有索引；
- README、claim ledger、简历与面试口径一致；
- 不再为了“多一个优化点”新增未经 profiling 证明的模块。

项目在当前范围内可以作为秋招交付。后续若继续研究，最高价值项不是更多 FCFS
常数，而是真实网络 server 的同协议 external H3，以及只有在 profiler 证明可重叠时
才做的异步 vision/language pipeline。
