# Prism-Infer 技术报告

> 版本：P9-C architecture/quality update，2026-07-19
> 目标模型：Qwen3-VL-8B-Instruct
> 冻结质量基线：`p6.12-content-aware-kv` / `c970c61`
> 当前完整性能验证点：P7.4-B / `72f85ba`
> 当前 scaled-FP8 质量点：Prism `5ada892` / vLLM external `3ec90a5`

## 摘要

Prism-Infer 是一个以 Qwen3-VL-8B 为目标的多模态推理和视觉 KV Cache 研究引擎。
工程主线不是调用 Hugging Face、vLLM 或 SGLang 的模型 wrapper，而是实现 vision
encoder、Qwen3 decoder、M-RoPE、DeepStack 注入、Paged KV、scheduler、CUDA Graph
decode、trace、visual-token scoring 和 physical KV compaction，并逐层与独立 reference
对齐。

项目形成了四类结果：

1. 自实现 Qwen3-VL 多模态 forward 与单机推理 engine 的 correctness 基线。
2. 从 KV trace 到 content-aware physical compaction 的质量—显存—性能闭环。
3. 从 node-level Systems trace 到 logits projection 修复、Graph replay 分解和下一轮
   projection packing 的性能归因闭环。
4. 从失败的 unit-scale FP8 到 per-token/per-KV-head scaled FP8、标准多模态质量集和
   external fail-closed comparator 的量化证据闭环。

当前结论并非“全面超过 vLLM”。在同条件 best-stable CUDA Graph 对比中，Prism 的
quality-qualified compact TPOT 仍为 vLLM 的 `1.34x–1.40x`。项目价值主要在可解释的
自实现、严格证据边界、视觉 KV 物理压缩和系统瓶颈定位。

## 1. 研究问题与边界

项目依次回答四个问题：

1. 能否在轻量 engine 中正确实现 Qwen3-VL 的 text、vision、M-RoPE 和 DeepStack？
2. 视觉 token 在 attention/KV 中呈现什么可记录行为，trace 是否不改变输出？
3. 能否真正减少 physical KV token/page/bytes，而不是只做 logical mask？
4. 压缩和执行优化在质量、显存、TPOT、E2E 与 online SLO 上分别有什么效果？

明确排除或尚未完成的范围：网络 server、跨机 PD/EP、投机解码、真实 megakernel、
TP2 动态矩阵、NVFP4、权重/激活量化、scaled-FP8 正式 runtime speedup，以及跨框架
page-table/allocator 全口径物理显存 Pareto。详见 [Known Issues](KNOWN_ISSUES.md)。

## 2. 系统总览

```text
HF tokenizer / processor
          │
          ▼
Prism VL input contract ── image/video spans, grid_thw, 3D positions
          │
          ├──────────────► Vision Encoder ─► main + 3 DeepStack features
          │                                      │
          ▼                                      ▼
Request/Sequence ─► Scheduler ─► BatchPlan ─► Qwen3-VL decoder
                         │               │              │
                         ▼               ▼              ▼
                   BlockManager     Paged KV      logits/sampler
                         │               │              │
                         └── compaction ─┴── trace ─────┘
```

关键边界：

- tokenizer/processor 复用 HF，避免重新实现格式兼容层。
- 模型 forward、3D position ids、DeepStack 注入和 engine execution 由 Prism 实现。
- HF 模型只在测试进程中作为数值 reference，不进入 Prism runtime forward。
- benchmark schema 把 offline、online、correctness、quality 和 performance 分开记录。

## 3. Qwen3-VL 模型自实现

### 3.1 Vision Encoder

[vision_encoder.py](../prism_infer/vision/vision_encoder.py) 实现：

- 3D patch embedding；
- Vision Rotary Embedding；
- ViT attention/MLP/block；
- spatial merge 与 PatchMerger；
- 主视觉特征和多层 DeepStack merger。

ViT 指定层产生 DeepStack features，主 merger 输出与 language hidden size 对齐的视觉
embedding。模块测试覆盖 patch、MLP、attention、RoPE、完整 encoder 与 HF reference。

### 3.2 Text Decoder 与 DeepStack

[qwen3_vl.py](../prism_infer/models/qwen3_vl.py) 实现 RMSNorm、GQA attention、MLP、
decoder layer、text model、Vision+LLM model 和 causal LM head。视觉主特征替换 image/
video placeholder embedding；三路 DeepStack feature 分别在 text decoder layer
`0/1/2` 后加到视觉 token 位置。ViT 侧特征提取层与 LLM 注入层是不同概念，代码中
显式分离。

权重加载保留 HF state-dict key 兼容。P7.5 的 gate/up packing 仍暴露
`gate_proj/up_proj/down_proj` 键；两个逻辑 view 共享一份 packed storage，device/dtype
转换后重新绑定，避免隐式复制权重。

### 3.3 M-RoPE 与位置语义

[qwen3_vl_position.py](../prism_infer/models/qwen3_vl_position.py) 从 expanded multimodal
tokens、`grid_thw` 和 attention mask 生成 `[3,batch,seqlen]` position IDs 与
`rope_delta`。text token 使用一致的一维 progression；image/video token 在 temporal、
height、width 三个轴上编码。

物理 KV compaction 后，decode 的 logical M-RoPE position 继续按原序列增长，而
physical KV position 指向压缩后的 page slot。这个分离是压缩后生成不发生位置漂移的
核心 invariant。

### 3.4 Correctness 层级

模型验证不是单一“能生成”：

| Gate | 目的 |
|---|---|
| module reference | 定位 patch/attention/MLP/RoPE 等局部误差 |
| structure/state dict | 防止 key、shape、共享 storage 破坏加载合同 |
| full logits/layerwise | 验证误差是否在完整模型累积 |
| greedy tokens | 验证实际输出路径 |
| teacher-forced logits/PPL | 分离低 margin argmax 与分布错误 |
| mixed/Graph/long output | 覆盖 batch shape、执行后端和长期稳定性 |

P1–P3 的 text 与单图 full logits 曾达到 max/mean diff `0`；P7.4 model-precision 路径
的 single/multi-image/video teacher-forced logits/PPL 相对 HF 也为逐值 exact。BF16
跨 batch shape 的低 margin argmax 可以分叉，因此最终合同要求同一 shape
deterministic，并用 teacher-forced distribution 和固定 task gate补充，而不是错误地
要求所有 GEMV/GEMM shape 的 token 永远一致。

## 4. Engine、Paged KV 与调度

### 4.1 强类型执行合同

P7.2 将 engine 主循环拆成：

```text
RequestLifecycle
    → SchedulerPolicy / admission
    → immutable BatchPlan
    → ModelExecutor / ExecutionResult
    → scheduler postprocess
    → RequestOutput / metrics
```

[contracts.py](../prism_infer/engine/contracts.py) 定义 batch phase、KV transfer、执行和
metrics 协议；[request.py](../prism_infer/engine/request.py) 管理 request FSM；
[executor.py](../prism_infer/engine/executor.py) 隔离 runner backend。这减少了旧
`step()` 五元组和 scheduler/model-runner 双向隐式状态耦合，同时保留兼容 API。

### 4.2 Paged KV 与 hardening

KV cache 使用 canonical layout：

```text
[K/V, layers, blocks, block_size, kv_heads, head_dim]
```

BlockManager 管理 page allocation、release、swap 和 prefix hash。P4.5 修复了 CPU
fallback 写入、prefix hash 释放清理、swap CPU/GPU page table 混用，并显式拒绝当时
不支持的 prefix-hit prefill；P7.3 后增加 Q<K chunked/prefix paged prefill。VL
prefix hash仍禁用，因为仅比较 placeholder token IDs 无法区分不同像素语义。

### 4.3 Continuous batching 与 online metrics

P7.3 实现 wall-clock arrival、admission/cancel、prefill/decode interleave、chunked
prefill 和请求级 queue/TTFT/TPOT/latency/goodput。clean `e7796e9` 的 9-cell matrix
全部完成且每 cell goodput fraction 为 `1.0`；最高压 mixed case出现动态 active batch
`4–5`。这是 engine-level harness，不包含 HTTP/gRPC、跨进程 server 或外部框架
online 对比。

## 5. KV Trace 与分析

[kv_trace.py](../prism_infer/analysis/kv_trace.py) 提供默认关闭的 session/schema。
trace record 包含：

- model/config 与 request identity；
- text/image/video token span 和 grid metadata；
- layer/head attention statistics；
- K/V norm、entropy 与 visual-token summary；
- compression decision 和 logical/physical KV metadata。

trace on/off greedy tokens 有等价门禁。离线工具把 JSONL 汇总为 JSON、Markdown、SVG，
并可按 attention mass、entropy focus 和 K-norm 生成视觉 token importance。trace 的
作用是形成可审计假设，不把有 instrumentation 的 timing 当正式 latency。

## 6. Visual KV 压缩

### 6.1 从 logical pruning 到 physical compaction

早期 `visual_prune` 只让 decode attention读取 retained slots，KV pool和 page没有释放；
它证明选择逻辑，但不是显存压缩。P6.4 的 physical path执行：

1. prefill 完成并收集 visual token score；
2. 生成 retained original positions；
3. 构造 source/destination physical slots；
4. 原子 gather/scatter K/V；
5. 提交 compressed page table和 physical length；
6. 释放尾部 page；
7. decode append 到压缩后的 physical tail，同时保留 logical M-RoPE position。

layout、append、mixed batch、swap、pickle、keep-all 和 CoW 都有 focused regression。
CUDA FP8 因 PyTorch index copy限制使用自实现两阶段 Triton compaction。

### 6.2 Content-aware scorer

最终 BF16 主线在 prefill 最后一层聚合最后 query 对 visual tokens 的 attention mass，
做 global top-k，keep ratio `0.5`、minimum `32`。score 在 GPU 上跨层/TP聚合，直到
prefill 末尾才 materialize decision；记录中保存 score source、layers、min/max/mean
和每个 visual span 的保留数。

被拒绝方案也保留证据：

- uniform pruning：代表性 output128 和真实质量 gate失败；
- last4/last8 aggregation：固定 batch的 ROUGE-L gate不如 last1；
- 等额 per-span quota：没有改善 COCO/multi-image/video fidelity；
- Python MMR/coverage ablation：质量没有稳定改善且 prefill overhead过高。

### 6.3 质量、物理显存与短 workload 性能

clean `e51c16d`，7 张 COCO val2017 图片、35 captions、output32：

| Metric | Dense | Last1 compact | Delta / ratio | Gate |
|---|---:|---:|---:|:---:|
| token-F1 macro | `0.321635` | `0.318347` | `-0.003288` | PASS |
| ROUGE-L macro | `0.289116` | `0.285406` | `-0.003710` | PASS |
| physical prompt tokens | — | — | `0.535x` | PASS |
| active prompt bytes | — | — | `0.538x` | PASS |

这是相对 dense baseline 的 lexical preflight，不是标准 COCO accuracy。exact requests
只有 `3/7`，最差 stable prefix为 `3/32`；task gate PASS 不等于逐 token完全一致。

同一策略的 COCO batch4/output32 稳定性能：

| Metric | Dense Graph | Compact Graph | Relative |
|---|---:|---:|---:|
| prefill median | `221.874 ms` | `224.179 ms` | `1.010x` cost |
| decode-step median | `18.945 ms` | `18.553 ms` | `1.021x` speedup |
| engine output tok/s | `158.048` | `160.087` | `1.013x` |
| E2E median | `993.238 ms` | `988.486 ms` | `1.005x` speedup |

结论是“物理 KV 明显减少、短上下文 latency收益很小”，不是压缩带来大幅通用加速。

### 6.4 FP8 边界

FP8 storage和 paged decode kernel完成 32-case independent SDPA correctness matrix，
旧 `fp8_kv` 固定 block pool payload bytes可到 BF16 的 `0.5x`，但它没有 scale；
unit-scale FP8 在长输出真实 workload 没有通过最终质量门禁。uniform+FP8 的 `4.016x`
observed peak running capacity也伴随质量 FAIL，只能作为 rejected baseline。

P9-C 新增独立的 `scaled_fp8_kv`：每个 token、每个 KV head 分别保存 K/V FP32 scale，
并让 scales 与 payload 一同经历 Triton store、paged decode、COW、swap、physical
compaction 和 CUDA Graph replay。clean `5ada892` 的冻结标准质量矩阵结果为：

- DocVQA、MuirBench、MVBench development/final 六项相对 Prism BF16 全部
  non-inferiority PASS；
- allocated KV pool 为 `778,567,680 / 1,509,949,440 = 0.515625x`，节省
  `48.4375%`；
- clean `3ec90a5` 的同容量 vLLM per-token-head FP8 external matrix中，
  DocVQA/MuirBench PASS，MVBench development/final FAIL。

最后一项的正确表述是“预注册外部质量稳定性结果为 MIXED”。vLLM 的 MVBench
accuracy 点估计实际更高，不能声称 Prism accuracy 显著领先；双方 page-table/Python
allocator 没有统一字节合同，也不能据此宣称完整物理显存 Pareto 胜出。

## 7. 系统性能优化

### 7.1 CUDA Graph

decode CUDA Graph使用 `[1,2,4,8]` buckets和 padding row隔离。P6.11 physical
compression Graph相对同 compression eager 的 offline decode speedup约
`1.76x–1.94x`。这是 Prism internal eager→Graph，不是对 vLLM 的加速比。

P7.4-B 的 clean node trace：

| Category | Kernels/step | Busy ms/step | Replay fraction |
|---|---:|---:|---:|
| linear/GEMV | `253` | `9.123` | `70.55%` |
| paged attention | `36` | `1.693` | `13.17%` |
| elementwise | `1,157` | `1.165` | `9.02%` |
| 其余 copy/reduction/layout/KV/trig | `554` | `0.939` | `7.26%` |

总计 `2,000` kernels/step、kernel busy median `12.921 ms`。CPU `graph.replay()` range
只有 `1.899 ms`，但返回后 GPU还有 `13.089 ms` tail，说明它是异步提交窗口；不能
把 CPU range误认为完整 Graph 时长。

### 7.2 Trace-driven logits optimization

旧 `compute_logits()` 每个 decode step执行：

```python
F.linear(hidden_states.float(), lm_head.weight.float())
```

`151,936 × 4,096` 的 BF16 LM-head每步临时转 FP32，产生大显存流量和约 2.3 GiB
transient allocation。改为模型原生精度后：

| Metric | FP32 historical | Model precision | Change |
|---|---:|---:|---:|
| logits CUDA median | `4.068 ms` | `0.762 ms` | `5.34x` region speedup |
| logits kernels/range | `4` | `1` | `-3` |
| single-image TPOT | `17.887 ms` | `14.151 ms` | `1.264x` |
| peak allocated | `19,708.6 MiB` | `17,391.5 MiB` | `-2,317.2 MiB` |

五 workload clean matrix的 TPOT提升为 `1.216x–1.280x`。HF logits/PPL和固定质量
gate通过后才形成 claim。

### 7.3 Projection packing preflight

P7.5 按 trace证据检查 projection fusion：

- packed QKV在 batch1 exact，但 batch2/4/8 的 K/V BF16 max diff为 `1.0`，在计时前
  被严格拒绝；
- packed gate/up在 rows `1/2/4/8/210/408/988` 的完整 MLP output bitwise exact，
  focused regression `32 passed`；
- clean node trace实测调用数由每层三次 MLP linear变两次，全 replay linear count
  `253 → 217`、总 kernels `2,000 → 1,964`。

完整HF logits/PPL、text/单图/多图/video/mixed、7-image COCO、online SLO与full
regression均已通过。8个clean offline cell token exact，unprofiled decode TPOT改善
`0.483%–0.762%`。该收益在记录cell中方向一致但很小；vision prefill仍双峰，所以
不声称稳定E2E或online goodput加速。

## 8. 外部框架对比

P7.1/P7.4 固定模型、GPU、KV budget、sampling、prompt tokens、prefix/cache、执行模式、
warmup/repeat和 clean state。自动汇总器对不公平 cell拒绝计算 ratio。

P7.4 model-precision 后 best-stable Graph：

- quality-qualified compact Prism TPOT为 vLLM的 `1.34x–1.40x`；
- Prism allocator peak约 `17.39–17.50 GiB`，vLLM约 `17.74–17.93 GiB`；
- E2E仍受 vision prefill/TTFT双峰影响，vLLM总体更快；
- external comparison是 offline closed-loop，不是 online goodput。

报告劣势是项目结论的一部分。压缩的可见优势主要是 active KV page和容量语义，尚未
转化为当前短/中 context下的 raw TPOT领先。

## 9. 验证与证据治理

项目采用：

```text
Plan → Implement → Verify → Teach → Document → Gate Review
```

每项性能工作要求 correctness-first、单变量 A/B、clean commit、raw JSON/JUnit、环境
身份和 claim边界。主要入口：

- [VERIFICATION](VERIFICATION.md)：命令和 PASS 标准；
- [PERFORMANCE_REPORT](PERFORMANCE_REPORT.md)：正式数字和 raw evidence路径；
- [CLAIMS](CLAIMS.md)：可用、受限、禁止结论；
- [REPRODUCIBILITY](REPRODUCIBILITY.md)：外部用户最小复现路径；
- [STAGE_DELIVERY_TEMPLATE](STAGE_DELIVERY_TEMPLATE.md)：阶段验收模板。

最新完整 formal regression是 clean `021d4e2` 的
`281 passed, 6 skipped in 297.622s`；JUnit为`287 tests / 0 failures / 0 errors / 6 skipped`。

## 10. 局限与下一步

最高优先级：

1. 建立跨框架 page-table/allocator metadata 的统一字节合同，完成或诚实否决
   full-physical Gate A。
2. 用已恢复的 NCU/NSYS 进入 P9-D，验证 full-step Graph 和 split-GQA/context kernel
   是否有 full-engine收益。
3. 建立 thin network server后再做 arrival/SLO external online goodput。
4. 只有获得明确双卡资源后，才运行 TP2 logits/greedy、NCCL、per-GPU memory和 latency。

P8 fresh venv、完整8B demo与full regression，以及 P9 标准质量矩阵均已完成；上述
剩余条件项仍只能由对应物理字节合同、runtime profiling、网络环境或合法双卡证据关闭。

## 结论

Prism-Infer 已形成一套可运行、可追踪、可压缩和可解释的 Qwen3-VL 单机研究 engine。
最重要的工程结果不是单个 speedup，而是把模型数值对齐、视觉 KV 物理语义、质量
门禁、CUDA Graph执行和系统 profiling放进同一证据链。当前压缩主线在受限 lexical
gate下减少约一半 active visual KV，但短 workload性能收益有限；trace-driven logits
修复显著降低了无效转换成本，packed gate/up又以36个更少的linear带来不足1%的decode
TPOT改善。scaled FP8 在冻结标准质量协议下把 allocated KV pool 降低 `48.4375%`，
但 external TPOT仍落后，full-physical Pareto也尚未建立。剩余工作边界清楚，未验证
能力没有被包装成完成结论。
