# P9 架构与性能旗舰化 RFC

> 状态：P9-A PASS；P9-B ENTRY APPROVED
> 日期：2026-07-17  
> 目标模型：Qwen3-VL-8B-Instruct  
> 目标硬件：单张 NVIDIA GeForce RTX 5090 32 GiB；TP2 为条件分支  
> 时间边界：2026-08-06 前完成工程主线，2026-08-07 起保留 7–10 天学习与面试复习

## 1. 决策摘要

P9 不再把 Prism-Infer 定位成“实现了 Qwen3-VL，再附带一个 KV pruning
实验”的教学项目。项目的最终定位是：

> **面向 Qwen3-VL 的跨层多模态推理 Runtime：把策略无关的视觉 KV 保留、
> 真实 Paged-KV 物理压缩、细粒度页分配、scaled FP8、Compiler/CUDA Graph、
> 多模态调度和优化 Kernel 接成一条可验证的运行时路径。**

P9 的核心价值不是发明另一种 token importance score，而是解决从“逻辑上少保留
token”到“物理显存、执行图、调度和在线 SLO 真正获益”之间的系统鸿沟。当前
last-query attention selector 保留为可替换 baseline，不作为算法创新 claim。

本 RFC 冻结以下决定：

1. 保留 P7.2 已验证的 `BatchPlan`、`ModelExecutor`、`SchedulerPolicy`，按依赖方向
   渐进拆分，不进行推倒重写。
2. 删除 `Sequence.set_block_size()`、类级 block size 和静默配置过滤；配置错误必须
   在模型加载或 CUDA 初始化前失败。
3. KV 主线是 **content-aware physical compaction + per-token-per-KV-head scaled
   FP8**。现有 unit-scale FP8 只作为失败 baseline。
4. 权重/激活量化只做 BF16、FP8 W8A8、W4A16 shootout；最多接入一个经过质量和
   E2E 门禁的后端，不同时实现 AWQ、GPTQ、SmoothQuant 三套系统。
5. NVFP4 暂不进入核心主线：当前 Prism Torch 2.6 栈没有
   `torch.float4_e2m1fn_x2`，不能为了一个 dtype 擅自升级整套运行环境。
6. CUDA Graph 的首要目标是 greedy full-step（decoder + LM head + sampler），
   `torch.compile` 由显式 backend 管理，不能继续依靠散落的装饰器暗中决定执行模式。
7. 服务端只做薄 OpenAI-compatible、streaming、backpressure 验证层；核心项目仍是
   Runtime，但 external SLO goodput claim 必须经过真实网络入口。
8. 不承诺所有 workload 全面超过 vLLM/SGLang。最终硬门槛是下文定义的两个
   headline gate，且必须同时满足质量和公平性条件。

## 2. 为什么这个定位成立

### 2.1 已经拥有的地基

P8 已交付并形成以下可信基础：

- 自实现 Qwen3-VL text、vision、M-RoPE、DeepStack 路径，并与 HF 建立分层和
  full-logits 参考。
- single/multi-image、frame-sequence video、mixed batch、chunked prefill、
  continuous batching、Paged KV、CUDA Graph 和 TP 控制面。
- visual KV 从逻辑 retention 到物理 compaction、page 回收和压缩 decode 的完整路径。
- 固定 7-image COCO lexical preflight 中，BF16 content-aware compaction 的
  physical-token/active-byte ratio 为 `0.535x/0.538x`，但这不是标准任务 accuracy。
- P7.5 clean full regression 为 `281 passed, 6 skipped`；packed gate/up 在限定
  workload 上改善 decode TPOT `0.483%–0.762%`。

### 2.2 当前不能支撑旗舰 claim 的缺口

- 现有 `Config` 混合 model/cache/scheduler/execution/compression 设置，并通过
  `Sequence.set_block_size()` 修改进程级隐式状态。
- `LLMEngine.__init__` 只挑选认识的 kwargs，未知参数被静默丢弃。
- 物理 page 默认 `256`，缺少 page-size Pareto；它同时影响内部碎片、prefix reuse、
  compaction 粒度和 kernel 调度，不能继续作为无依据常量。
- 当前 FP8 仅把 BF16 K/V 直接写入 FP8 tensor，等价于 scale=1；长输出质量 gate
  已失败。
- Paged decode kernel 每个 program 处理一个 `(sequence, query_head)`。在
  batch=8、32 query heads 时 grid 只有 256，NCU 已证明并行度不足。
- Prism best-stable Graph TPOT 仍慢于 vLLM，现有结果没有 external online server
  goodput，也没有可用的 TP2 Prism 动态证据。
- 现有 COCO lexical gate、逻辑 token ratio 或一次进程内 benchmark 都不足以成为
  秋招 headline。

### 2.3 与公开 KV compression 工作的边界

- [AirCache](https://arxiv.org/abs/2503.23956)研究 inter-modal relevancy、observation
  window 和 layer-wise budget。
- [TGV-KV](https://arxiv.org/abs/2606.03075)研究 text-grounded ranking、layer budget
  和 text-prioritized retention。
- [BACON](https://arxiv.org/abs/2606.14782)明确把 last-query attention 作为
  observation-window attention 的补充证据，并做跨层校准。

因此 Prism 不能把“使用 last query 给视觉 token 排序”包装为新算法。P9 的差异化
是将 selector 设计成策略接口，并证明某个质量合格的 selector 能落到真实 page、scale、
allocator、Graph 和 SLO 上。若论文代码不能在同一 Qwen3-VL/processor 语义下运行，
只做文献边界说明，不制造不公平数字。

## 3. 最终成功标准

所有 headline 数字必须来自 clean commit、固定 manifest、同一 GPU UUID、完整原始
记录和预先冻结的阈值。P9 同时要求 Gate A 和 Gate B；只完成其中之一不能称为最终
旗舰交付。

### 3.1 Gate A：KV 质量–物理显存 Pareto

至少存在一个 Prism 配置点，在标准质量集上合格，并相对 vLLM 的强 KV baseline
形成 Pareto 优势。

一个 Prism 点 `P` 对 vLLM 点 `V` 的“质量–显存支配”定义为：

```text
quality(P) >= quality(V) - delta
physical_kv_bytes(P) < physical_kv_bytes(V)
```

其中：

- bounded accuracy 的 non-inferiority margin `delta = 1.0` 个百分点；生成式归一化
  指标的 `delta = 0.01`。阈值写入 workload manifest，不藏在 runtime 常量中。
- 必须报告 paired bootstrap 95% CI；只有 CI 仍落在 non-inferiority margin 内才合格。
- `physical_kv_bytes` 包括 K/V payload、K/V scale、page table 和压缩 metadata，
  不能用逻辑 token 数代替。
- vLLM baseline 至少包括 BF16 KV 和其在当前 Qwen3-VL backend 上可运行的最强
  scaled FP8 方案；不能只与 unit-scale 或 eager 弱配置比较。
- Prism 的 7-image COCO lexical preflight 只作为快速回归，不进入最终 Gate A。

### 3.2 Gate B：长视觉 Runtime/SLO 优势

在第 5 节预注册的 headline workload 中，至少一个 workload 的以下任一指标优于
vLLM 或 SGLang best-stable strong baseline：

- decode TPOT；
- SLO goodput；
- p95 或 p99 TTFT。

性能胜出必须同时满足：

1. 相同模型内容、processor/token 语义、sampling、请求集和物理 GPU；
2. 双方使用各自 best-stable backend，不以 Prism Graph 对比外部 eager；
3. 候选配置先通过质量 gate；
4. 至少 `5%` practical effect，且 process-level bootstrap 95% CI 不跨越无收益边界；
5. 短文本与普通单图 guardrail 的关键延迟/吞吐不得回退超过 `5%`，否则必须作为
   专用 long-visual mode 暴露，不能替换默认模式。

这里的 `5%` 是为了排除时钟波动和一次性微小收益，不是 kernel 内 magic number。

### 3.3 明确不是出口条件的内容

- 所有 batch/context/输入类型全面胜出；
- TP4/TP8 或多机扩展；
- 自创并发表新的 KV importance 算法；
- 为了简历数字降低 correctness 或质量阈值；
- 用 preflight、单次 run、kernel duration、逻辑 retention ratio 代替最终 claim。

## 4. 目标架构

### 4.1 依赖方向

```text
Public API / thin server
          |
Request lifecycle + request-owned multimodal identity
          |
SchedulerPolicy + multimodal cost/admission model
          |
immutable BatchPlan -> immutable DeviceBatch
          |
ExecutionBackend (eager / compile / graph / compile+graph)
          |
ModelExecutor -> AttentionBackend / LinearBackend
          |
KVCacheManager
  | allocator | layout | reservation | compaction | precision/scales |
          |
Kernel layer (paged attention / KV quant-store / compact-copy / sampler)
```

上层只能依赖下层 contract。Kernel 不读取 `Sequence`，scheduler 不直接调用 model，
metrics 只能观察，不能反向驱动调度。

### 4.2 保留的 P7.2 结构

- `BatchPlan`：继续作为 scheduler 到 executor 的不可变 host decision。
- `ModelExecutor`：继续负责 KV transfer、model run 和 compaction commit 的事务边界。
- `SchedulerPolicy`：继续保留策略接口，FCFS 是 baseline。
- `RequestLifecycle` 和 `EngineMetrics`：继续作为 request FSM 与观测合同。

这些结构已有完整回归保护。P9 只消除兼容层泄漏和错误依赖，不重写已证明正确的
边界。

### 4.3 新增或强化的边界

#### 配置域

```text
PrismConfig
  ModelConfig
  CacheConfig
  SchedulerConfig
  ExecutionConfig
  QuantizationConfig
  ServingConfig
```

- 用户输入配置先校验，再解析 HF model，最后生成 frozen `ResolvedConfig`。
- 未知字段、互斥 backend、非法 page/scale granularity 使用明确异常；不用 `assert`。
- 默认值集中为有名称、有注释的 policy default；硬件/模型可推导值不允许写死。
- 兼容旧 flat kwargs 的 adapter 只保留一个版本周期，并对未知参数报错。

#### 请求与 device batch

- `Sequence` 构造时显式接收 block/page contract；删除类级 `block_size`。
- request id 由 engine-owned allocator 产生，测试可注入 deterministic allocator。
- multimodal cache key 必须包含 pixel/content hash、processor revision、grid/temporal
  identity，不能只 hash placeholder token ids。
- `DeviceBatch` 只保存执行所需的 tensor、shape/bucket、KV view 和 scale view，不携带
  可变 `Sequence` 对象，作为 compile/Graph 的稳定输入边界。

#### 执行 backend

统一 contract 至少暴露：

```text
prepare(plan) -> DeviceBatch
warmup(bucket)
capture(bucket)
execute(device_batch) -> ExecutionResult
release()
```

支持矩阵由配置显式选择：`eager`、`compile`、`cuda_graph`、`compile_graph`。
unsupported 组合在启动时失败，不在运行中 silent fallback。

## 5. 冻结 workload 与质量协议

### 5.1 Headline runtime workload

| ID | 场景 | 固定核心 | 用途 |
|---|---|---|---|
| H1 | 8-image long visual | 8 张固定 448px 图片，greedy output=128，batch 1/4 | page/compaction、TPOT、显存 |
| H2 | 16-frame video | 16 固定帧，greedy output=128 | temporal KV；仅在跨框架 prompt token 完全一致时比较 |
| H3 | 32 GiB mixed arrival | text/single-image/H1/H2 的固定 request trace | TTFT、TPOT、goodput、KV pressure |

H1 是必做 headline。H2 如果 vLLM/SGLang 与 Prism 的 timestamp/placeholder token
语义不一致，只保留 Prism regression，不进入胜负统计。H3 使用至少 600 个完成请求，
固定 arrival seed；p99 只有在有效样本数足够时报告。

机器可读协议已冻结为：

- `benchmarks/workloads/p9_headline.json`，canonical SHA256
  `42d1387320b1b30c3b0afa0bf3113f0dd905a38b38bc583cfe6c6eb3ef4f8656`；
- `benchmarks/workloads/p9_quality_protocol.json`，canonical SHA256
  `85adb4b246ab3fc55bc70e02ad75d97c5aa903e89387e499fc3aea1ac2edb25d`。

runtime manifest 使用确定性合成媒体隔离系统性能，不用其生成质量分数；标准质量集使用
固定公开数据 revision。媒体物化后还必须生成 selected-ID 与逐媒体 SHA256，manifest
hash 不能替代数据内容 hash。

H3 的 SLO 不在看到候选结果后手调。先在 vLLM best-stable 低负载下测每个 request
class 的 baseline，再按固定公式冻结：

```text
TTFT SLO = 5 * low-load p50 TTFT
TPOT SLO = 2 * low-load p50 TPOT
```

arrival-rate grid 和 SLO 随 manifest 一起提交，之后 Prism 与 external 框架使用同一
trace。goodput 定义为同时满足该请求 class 的 TTFT 与 TPOT SLO 的 output tokens/s。

### 5.2 Guardrail workload

- text-short、single-image 448px、2-image 448px；
- compression-off 的 HF model-precision logits/PPL；
- mixed text/image/video greedy tokens；
- full regression。

### 5.3 标准质量集

最终质量集使用三个互补类别，数据 revision、样本 ID、媒体 SHA256 和 evaluator 版本
必须落入 manifest：

- DocVQA：高分辨率文档/OCR，使用官方 ANLS；
- MuirBench：多图理解，使用官方 accuracy；
- MVBench：视频理解，使用官方 accuracy。

每类先建立固定开发子集做快速 gate，最终 claim 使用预注册的较大评估集。若授权或
下载条件导致某数据集不可复现，必须在实现压缩策略前用同类别公开集替换并提交
decision record；不能在看到模型结果后换题。

## 6. 公平外部对比协议

### 6.1 固定身份

- 模型：Qwen3-VL-8B-Instruct，revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- Prism：每条记录保存 full commit、dirty state、环境和 GPU UUID。
- vLLM：`0.24.0`，build commit `ee0da84ab`。
- SGLang：`0.5.15`，commit `f63458b5beaceabbd9d749b9fc956370e1b649e6`。
- 硬件：同一 RTX 5090 UUID；每次进程启动前通过空闲显存/utilization gate。

### 6.2 两类 profile，禁止混用

1. `diagnostic_matched`：尽量固定 page、KV bytes、eager/attention backend，用于归因。
2. `best_stable`：每个框架使用其质量合格、可复现的最佳稳定配置，用于 headline。

vLLM 当前默认 block size 为 16；SGLang CUDA 默认 page size 为 1。P9 不再强迫
external 强基线使用 Prism 的 256-token page。Prism 先完成 `16/32/64/128/256`
page matrix，再选择自己的配置。若某个 page/backend 组合不支持，记录为 unsupported，
不能替换成另一配置后仍标 comparable。

### 6.3 必须相同的语义

- processor、chat template、媒体内容、prompt token count/IDs；
- model/weight dtype、sampling、EOS、output length；
- prefix/MM cache 开关和请求顺序；
- KV 的实际可用 bytes，而不是相同“block 数”；
- timing scope：processor、queue、prefill、decode、network 各自独立，E2E 单独报告。

### 6.4 量化公平性

- BF16、scaled FP8、content compaction 是三个独立维度，必须做消融。
- FP8 bytes 包括 scale tensor。Prism unit-scale FP8 不能与 external scaled FP8
  作为同一量化级别比较。
- 权重/激活量化只有双方使用同一模型量化 artifact 和质量 gate 时才进入 runtime
  ratio；否则只做各自能力 shootout。
- NVFP4 只能在双方 runtime 都有可审计 dtype/layout/kernel 支持时进入 headline。

### 6.5 统计与原始证据

- microbenchmark：warmup >= 10、repeat >= 100，保存全部 samples。
- offline full engine：每配置至少 5 个 fresh-process repeats，运行顺序使用 ABBA/BAAB
  降低漂移。
- online：固定 arrival trace，至少 600 个 completed requests，并做 process-level repeats。
- 报告 median、p90、p95、p99、min/max、bootstrap 95% CI、allocated/reserved/peak 和
  NVML process bytes。
- 不手工删除 outlier；若环境 gate 失败，整次 process run 作废并保留失败记录。

## 7. KV Cache 与量化主线

### 7.1 为什么现有 FP8 失败

`prism_infer/layers/attention.py` 当前 store kernel 直接把 BF16 K/V 写入 FP8 cache，
paged decode 再直接 load 为 FP32。没有 quant/dequant scale，所以实际采用固定
scale=1。该路径在短输出偶然 token exact，但长输出最终 quality gate 已失败。

这不是 RTX 5090 缺少 FP8 能力。固定 vLLM 环境提供
`fp8_per_token_head`，其实现对每个 `(token, KV head)` 计算 K/V absmax scale，分别
存储 FP8 payload 与 FP32 scale。P9 的首个公平实现采用相同粒度作为 correctness
baseline：

```text
k_scale = max(abs(K[token, head])) / FP8_MAX
v_scale = max(abs(V[token, head])) / FP8_MAX
K_fp8 = clamp(K / k_scale)
V_fp8 = clamp(V / v_scale)
```

decode attention 在 load 后分别乘回 `k_scale`、`v_scale`。scale 的存储、copy、swap、
compaction 和 page 回收必须与 payload 同生命周期。

### 7.2 Scale granularity 候选

| 方案 | 质量 | metadata/带宽 | 决策 |
|---|---|---|---|
| unit scale | 已失败 | 最低 | 只保留 rejected baseline |
| layer/global K/V | 需 calibration | 最低 | external compatibility diagnostic |
| page × KV-head | 中等 | 较低 | 后续性能候选 |
| token × KV-head | 最强动态范围 | 每 token/head 两个 scale | P9 correctness baseline |

只有 token-head baseline 通过标准质量后，才评估 page-head scale 是否能用更少 metadata
换取相同质量。不能先优化错误量化方案。

### 7.3 Content-aware retention 的定位

- selector 是 `VisualRetentionPolicy`，输入为 request/span/layer evidence，输出为稳定、
  可审计的 retained logical positions 与 budget。
- physical layer 只消费 retention decision，不理解算法名称。
- 默认 last-query/last-layer 是 baseline；P9 至少补 observation window 或 text-grounded
  evidence 的消融，但不承诺复刻三篇论文。
- global、per-span、per-layer budget 必须作为显式 policy；当前被拒绝的 per-span
  ablation 保留，不通过改名重新启用。

### 7.4 权重和激活量化

KV 量化与 W/A 量化正交：前者降低随 context/concurrency 增长的 cache，后者降低
固定权重和 GEMM/GEMV 成本。做了 KV FP8 并不意味着不需要 W/A 量化，但一个月内
不能同时深挖全部算法。

P9 shootout：

| 模式 | 代表路线 | 目的 |
|---|---|---|
| BF16 | 当前基线 | correctness 与最终兜底 |
| FP8 W8A8 | SmoothQuant/ModelOpt 类 backend | 5090 tensor-core throughput 候选 |
| W4A16 | AWQ 或 GPTQ artifact | 权重显存/带宽候选 |

AWQ/GPTQ 都是 weight-only 路线，SmoothQuant 主要把 activation outlier 平滑到权重以
支持 W8A8；它们不是三个应同时“实现”的等价开关。shootout 只决定是否值得接入一个
backend。任何量化 artifact 都要单独记录 revision、校准集和标准质量结果。

## 8. Compiler 与 CUDA Graph

### 8.1 当前判断

- 已验证 CUDA Graph decode 相对 eager 有明显收益，是强 baseline。
- 当前 full decode `torch.compile` 在 32 GiB cold compile OOM；attention-only 曾出现
  batch2/8 长输出不一致，不能作为支持 backend。
- 散落在 RMSNorm/RoPE/activation/sampler 上的 `@torch.compile` 装饰器让 compile
  ownership、cache 和错误边界不可审计，P9 要集中管理。

### 8.2 优化顺序

1. 建立 `ExecutionBackend` 与 `DeviceBatch`，记录 graph break、recompile、cold time、
   generated code cache 和 steady-state。
2. 将当前 decoder Graph 扩成 greedy full-step：decoder、model-precision LM head、
   argmax sampler；消除 Graph 外同步。
3. 固定 batch/context/page/precision bucket，验证 padding row、KV length 和 scale view。
4. 在同一 full-step 边界测试 `compile_graph`；失败则保留明确 rejected evidence。
5. 只有 NCU/NSYS 证明某个子图仍受 launch 或 elementwise overhead 支配，才做新的
   compiler fusion。

## 9. Kernel 主线

### 9.1 新 NCU 证据

RTX 5090、BF16、Qwen GQA、batch=8、context=4096 的真实 NCU counter：

| Page | Duration | DRAM throughput | Compute throughput | Achieved occupancy | Waves/SM | Registers/thread |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 449.95 us | 17.48% | 14.16% | 12.49% | 0.19 | 64 |
| 256 | 543.26 us | 14.44% | 11.70% | 12.48% | 0.17 | 56 |

两者对 SDPA reference correctness 均 PASS，max diff `4.882812e-4`、mean diff 约
`3.0e-5`。该 kernel 既没有接近 DRAM roof，也没有接近 compute roof；首要证据是
grid 过小而非单纯“memory-bound”。表中数字来自 clean commit `29c0dbe` 保存的
NCU full-set raw report；此前 diagnostic 的 `445.60/550.46 us` 已被正式 raw evidence
取代，不能与本表混用。

### 9.2 候选设计

当前 grid 为 `(batch, query_head)=256`。Qwen GQA 中 4 个 query heads 共享一个 KV
head，理论上可在一个 program/CTA 内复用 K/V；但直接合并会把 grid 降到 64，令
并行度更差。因此候选必须组合：

```text
GQA query-head grouping + context split-K/split-attention
grid = batch * kv_heads * context_splits
```

每个 split 产生局部 online-softmax `(m, l, acc)`，第二阶段做稳定 merge。实验矩阵
至少覆盖 split `1/2/4/8`、page `16/32/64/128/256`、batch `1/8`、context
`4096/8192`，并记录寄存器、shared memory、occupancy、waves、DRAM 和 duration。

### 9.3 Triton、CUDA/CUTLASS 的选择

- 先用 Triton 实现 split/context mapping，快速验证算法、layout 和 correctness。
- 若 NCU 证明瓶颈来自寄存器、shared-memory layout、异步 copy 或 Triton codegen，
  再把同一 kernel contract 移到 CUDA/CUTLASS/CuTe DSL。
- 为学习而写 CUDA demo 可以作为附录，但只有替换真实 hot path、通过全矩阵和 E2E
  gate 的 kernel 才进入项目 claim。
- 不把普通 fusion、Graph 或一个较大的 Triton program 称为 megakernel。

## 10. 多模态调度与服务层

### 10.1 Cost model

scheduler admission 不再只看“序列条数/逻辑 token”，而显式估算：

```text
text prefill tokens
image patches / image count
video frames / temporal grid
decode active KV bytes (payload + scales)
vision encoder workspace
Graph bucket padding
```

视觉 payload region 在 chunked prefill 中保持原子语义；decode admission 使用物理 KV
reservation。H3 将比较 FCFS baseline 与最小的 multimodal cost-aware policy，重点是
降低长视觉请求造成的 head-of-line blocking，而不是堆砌 scheduler 名称。

### 10.2 Thin server

服务层提供：

- OpenAI-compatible chat completion；
- image/video request schema；
- streaming、disconnect cancellation、bounded queue 和 backpressure；
- request id 与 engine metrics 关联；
- 外部 load generator 可测 client-observed TTFT/ITL/E2E。

模型执行、KV 管理和调度不进入 HTTP handler。网络 server 只用于证明 Runtime 在真实
请求生命周期中仍保持 SLO，不把项目改造成 Web 后端工程。

## 11. TP 与八卡边界

当前八张 RTX 5090 均可见且空闲，但没有 NVLink；GPU0–3 属 NUMA0，GPU4–7 属
NUMA1。GPU 可见不等于当前 Prism 栈可以做 NCCL TP：

- Prism：Torch `2.6.0a0+...nv25.01`、CUDA 12.8、NCCL 2.25.1；GPU0–1 最小
  all-reduce 在首次 collective 报 `cudaErrorInvalidValue`。NCCL kernel 需要
  `82,240 B` shared memory，而当前设备函数上限为 `79,856 B`。
- 隔离 vLLM 环境：Torch `2.11.0+cu130`、NCCL 2.28.9；同一 GPU0–1
  all-reduce 得到 `3.0`，PASS。

所以 KI-003 是 **Prism software-stack blocker**，不是“只有一张卡”或 GPU 故障。
P9 单卡 headline 不等待 TP。若要恢复 TP2，优先在隔离环境创建 Prism compatibility
probe，并先征得用户同意；不直接升级当前已完成 P8 回归的主环境。TP2 只选同 NUMA
的 GPU0–1，且如实报告无 NVLink 的通信成本。

## 12. P9 日程与止损点

| 日期 | 阶段 | 交付物 | Stop/go gate |
|---|---|---|---|
| 07-17–07-18 | P9-A 规范与正式基线 | 本 RFC、fairness manifest、page/NCU/TP baseline | 协议与证据可审计 |
| 07-19–07-22 | P9-B 架构硬化 | typed config、无全局 block size、DeviceBatch/backend contract | full regression + config failure tests |
| 07-23–07-28 | P9-C KV/量化 | scaled FP8、scale lifecycle、细页 allocator、标准质量 gate | Gate A 候选存在 |
| 07-29–08-02 | P9-D Graph/compiler/kernel | full-step Graph、split-GQA paged attention、NCU/NSYS 闭环 | 至少一个 E2E 可测优化 |
| 08-03–08-05 | P9-E 调度/服务/外部对比 | thin server、H3、vLLM/SGLang strong baseline | Gate B 判定 |
| 08-06 | P9-F 最终交付 | 报告、README、简历故事、复现包 | clean release candidate |
| 08-07 起 | 学习与复习 | 代码走读、知识卡、面试演练 | 不再扩大主线 scope |

止损规则：

- 07-25 前 scaled FP8 component/logits 仍不通过：停止做 page-head 等性能变体，先保留
  BF16 compaction，定位 scale/layout correctness。
- 07-28 前标准质量仍无合格 retention 点：不再发明 selector；Gate A 降为 scaled
  FP8 + BF16 content compaction 的诚实结果，Runtime/Graph 成为唯一性能主线。
- 08-02 前 split-GQA kernel 没有 E2E 收益：保留 rejected NCU evidence，停止切换
  CUDA/CUTLASS，转向 full-step Graph 与 scheduler。
- 08-05 Gate B 未通过：不得挑一次偶然 run；最终材料改为“差距与已定位瓶颈”，不写
  超过 vLLM/SGLang。

## 13. P9-A 出口与 P9-B 进入条件

P9-A 只有满足以下条件才能 PASS：

- [x] 项目定位、非目标、最终 Gate 和时间边界冻结。
- [x] 目标架构、量化、Graph/compiler、kernel、scheduler/server 和 TP 决策冻结。
- [x] H1/H2/H3 与标准质量集生成 versioned manifest，并保存 canonical content hash；
  数据集媒体物化 hash 属 P9-C 质量运行前置门禁。
- [x] `bench_paged_decode.py` 支持固定 seed、多 page matrix、JSON/JSONL 和完整环境记录。
- [x] page `16/32/64/128/256` × batch `1/8` × context `4096/8192` clean BF16
  formal matrix 完成。
- [x] page 16/256 NCU raw report、counter summary 和复现命令落盘。
- [x] KI-003/KI-004、ROADMAP、VERIFICATION 与本 RFC 一致。
- [x] focused tests、`git diff --check` 和文档链接检查 PASS。

P9-B 的第一批改动只允许处理配置、全局 state 和 backend contract。scaled FP8 或
kernel 改动不能与架构迁移混在同一 correctness diff 中。

## 14. 外部实现证据

以下位置来自本轮固定源码/安装环境，作为设计事实，不代表 Prism 已实现同等能力：

- vLLM `vllm/config/cache.py:19-35`：支持 FP8、per-token-head FP8 和 NVFP4；
  `:46` 默认 block size 16；`:110-114` 说明缺少 scale 时可能退回 1.0。
- vLLM `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:143-160,
  198-258,271-309`：per-token-head K/V scale 的计算、存储和 launch grid。
- SGLang `sglang/srt/server_args.py:546-570`：FP8 KV 应提供 scale JSON，否则默认
  1.0 可能产生精度问题；同处列出 FP4 的版本条件。
- SGLang `sglang/srt/arg_groups/overrides.py:1754-1775`：CUDA 默认 page size 1。

这些证据说明 P9 必须同时解决 scale 和 page contract，不能把现有 unit-scale FP8
或 256-token page 当作“业界默认”。
