# Prism-Infer 投递与面试材料

> 更新日期：2026-07-23
> 使用规则：所有数字必须能回到 [CLAIMS](CLAIMS.md) 和对应证据；投递时选择与岗位
> 匹配的 2–3 条，不要把本文件整段复制到一页简历。

## 1. 一句话项目描述

### 中文

自实现 Qwen3-VL-8B 单机多模态推理引擎，在严格 HF 数值门禁上完成 Paged KV、
CUDA Graph、continuous batching、KV trace 与 content-aware visual KV physical
compaction、per-token/per-KV-head scaled FP8，并用 `torch.compile`、CUDA Graph 与
Systems trace 闭环优化 decode；冻结 H1/H2 中 BF16 TPOT 低于同协议 vLLM/SGLang，
scaled-FP8 在约翻倍 KV capacity 下仍保持 TPOT 优势。

### English

Built a single-node Qwen3-VL-8B multimodal inference engine with independently
implemented model/vision execution, Paged KV, CUDA Graph decode, continuous batching,
KV tracing, content-aware physical visual-KV compaction, and dynamically scaled FP8 KV,
backed by layered HF correctness, standard multimodal quality, and systems-profiling gates;
on two frozen batch-1 RTX 5090 workloads, its BF16 TPOT was lower than matched
vLLM/SGLang baselines, while scaled-FP8 nearly doubled KV capacity within the same budget.

## 2. 推荐简历 bullets

### 2.1 ML Systems / Inference 版本

- 自实现 Qwen3-VL-8B text/vision/M-RoPE/DeepStack 与推理 engine 主路径，覆盖
  text、单图、多图、视频和 mixed batch；建立模块、full logits/PPL、greedy、
  CUDA Graph和长输出分层门禁，P9-C clean full gate为 `377 passed / 6 skipped`。
- 设计 content-aware visual KV physical compaction：prefill末层 attention top-k、
  page内重排/尾页回收、logical M-RoPE与physical KV位置分离；7-image/35-caption
  preflight将 physical tokens/active bytes降至 `0.535x/0.538x`，token-F1与
  ROUGE-L drop为 `0.003288/0.003710`。
- 实现 per-token/per-KV-head scaled FP8 KV 全生命周期；冻结的 DocVQA、MuirBench、
  MVBench development/final 六项 formal quality gate 全 PASS，allocated KV pool降至
  BF16的`0.515625x`；同容量 NVML 进程峰值下降 `8.24%`，同约 4 GiB budget 的
  capacity 提升 `94.69%`。
- 将 scaled-FP8 KV 接入 batch1 `torch.compile + CUDA Graph` 正确路径；clean H1/H2
  中 56,320-token capacity profile 的 TPOT 为 `10.2363/10.2588 ms`，相对
  SGLang 低 `1.06%–1.12%`、相对 vLLM 低 `2.55%–2.77%`。
- 用 Nsight Systems node trace定位每 token完整 LM-head BF16→FP32转换，将 logits
  CUDA median从 `4.068 ms`降至 `0.762 ms`；五类 workload TPOT提升
  `1.216x–1.280x`，allocator peak减少 `2,230–2,317 MiB`。

### 2.2 GPU / Performance 版本

- 构建 schema化 benchmark与 profiler闭环，分离 preprocessing、TTFT、TPOT、E2E、
  online goodput、KV bytes和 allocator memory；CUDA Graph replay分解出每步
  `2,000` kernels / `12.921 ms` busy，linear/GEMV占 `70.55%`。
- 实现 BF16、unit-scale FP8与scaled FP8 paged decode/KV compaction Triton kernels；
  32-case Qwen GQA kernel correctness通过 independent SDPA reference，并通过
  tensorized slot mapping
  消除旧 visual gather的 `24,696` 次 async copy/stream sync。
- 建立 H1/H2 三引擎 clean comparability gates；同提交、同 GPU、同 prompt hash、
  warmup2/repeat5 下，Prism BF16 TPOT 为 `9.8821/9.8680 ms`，相对 SGLang
  低 `4.54%–4.83%`、相对 vLLM 低 `6.13%–6.27%`；限定为 batch1 offline cell。
- 实现HF-compatible gate/up packed projection；node trace确认每步linear
  `253→217`、总kernels `2,000→1,964`，8个clean多模态/COCO cell token exact且
  decode TPOT改善`0.483%–0.762%`，明确不扩写为E2E或online加速。

### 2.3 Engine / Serving 版本

- 将 engine重构为 Request FSM、SchedulerPolicy、immutable BatchPlan、Executor、
  KV manager和Metrics contracts，补齐 admission/cancel、资源释放、swap表和可替换
  policy边界。
- 实现进程内 wall-clock arrival、continuous batching、prefill/decode防饥饿、
  Q<K chunked paged prefill及 queue/TTFT/TPOT/goodput schema；clean 9-cell matrix中
  已完成请求全部满足各 cell预声明SLO。
- 实现 canonical Paged KV layout、post-prefill physical compaction、page回收和
  decode append，覆盖 mixed text/image/video、pickle/swap/CoW与keep-all invariants。

### 2.4 English bullets

- Implemented the Qwen3-VL-8B text/vision/M-RoPE/DeepStack execution path and a
  lightweight inference engine, with layered module, full-logit/PPL, greedy,
  CUDA-Graph, mixed-modal, and long-output correctness gates.
- Implemented per-token/per-KV-head scaled FP8 KV with coupled payload/scale lifecycle;
  passed all six frozen DocVQA, MuirBench, and MVBench quality cells while reducing the
  allocated KV pool to `0.515625x` of BF16, cutting same-capacity process NVML peak by
  `8.24%` and increasing capacity by `94.69%` within the same ~4 GiB KV budget.
- Integrated scaled-FP8 KV into a guarded `torch.compile` + CUDA Graph decode path;
  on frozen H1/H2 cells, BF16 TPOT was `4.54%–4.83%` below SGLang and
  `6.13%–6.27%` below vLLM, while the near-2x-capacity profile retained a bounded lead.
- Designed content-aware physical visual-KV compaction with last-layer attention
  scoring, page reclamation, and decoupled logical M-RoPE/physical KV positions;
  reduced physical prompt tokens and active KV bytes to `0.535x/0.538x` while keeping
  7-image lexical metric drops below `0.004`.
- Used node-level Nsight Systems traces to eliminate a per-token full LM-head BF16→FP32
  conversion, reducing logits CUDA median from `4.068 ms` to `0.762 ms`, improving
  TPOT by `1.216x–1.280x`, and cutting peak allocator memory by `2.18–2.26 GiB`.

## 3. 数字的必要限制

| 简历数字 | 面试中必须主动补充 |
|---|---|
| `0.535x/0.538x` | physical prompt tokens/active prompt bytes，不是整张 GPU显存 |
| quality drop `<0.004` | 7-image、35 captions、output32 lexical preflight，不是 COCO CIDEr/SPICE |
| `1.216x–1.280x` | lm-head precision单变量带来的 Prism internal TPOT提升 |
| logits `4.068→0.762 ms` | RTX 5090、Qwen3-VL-8B、node trace对应 region |
| Graph `1.76x–1.94x` | Prism eager→Graph，不是 Prism→vLLM |
| online goodput fraction `1.0` | 单次 engine-level 9-cell正式运行，无网络/外部对比/置信区间 |
| capacity `4.016x` | uniform+FP8质量FAIL，建议不放简历，只在失败复盘中讲 |
| scaled FP8 `0.515625x` | allocated KV payload+scales，不是整卡显存，也不含跨框架统一的allocator/page-table字节 |
| NVML peak `-8.24%` | Prism BF16→scaled 同容量、采样与 latency 分离，不是跨框架显存排名 |
| capacity `+94.69%` | 固定约 4 GiB KV budget 的 token capacity，不是 online goodput |
| BF16 TPOT 低 `4.54%–6.27%` | 只覆盖 RTX 5090、Qwen3-VL-8B、TP1、batch1、H1/H2、output128 offline Graph |
| scaled TPOT 低 `1.06%–2.77%` | 与外部 BF16 baseline 比；不代表 scaled 比 Prism BF16 更快，E2E 有 mixed 单元 |

## 4. 60 秒自我介绍版本

我做了一个 Qwen3-VL-8B 的轻量多模态推理引擎，不是把现成框架包一层，而是自己实现
Vision Encoder、M-RoPE、DeepStack、decoder、Paged KV和调度，再用 HF 做分层数值参考。
系统优化上，我把有状态 KV 留在受审计的 runtime 边界，只用 `torch.compile` 优化
batch1 无状态热点，再把 model forward、guarded LM-head candidate/exact rerank 和
greedy decode 放进 CUDA Graph。最终在同一 RTX 5090、同 prompt hash、warmup2/repeat5
的 H1/H2 中，BF16 TPOT 比 SGLang 低约 4.5%–4.8%，比 vLLM 低约 6.1%–6.3%。
第二条线是 per-token/per-KV-head scaled-FP8 KV：三个标准多模态数据集的六个正式 cell
全部通过质量门禁，同容量 KV bytes下降48.44%、进程NVML峰值下降8.24%；同约4 GiB
budget容量提升94.69%，TPOT仍小幅领先两家外部baseline。我保留了E2E mixed、online
尚未验证和content-aware组合缺少标准质量这些边界。

## 5. 5 分钟项目讲解结构

### 0:00–0:40：为什么做

- 多模态引擎不仅是 text decoder加图片 embedding；Qwen3-VL有 3D positions、
  Vision RoPE、PatchMerger和DeepStack。
- 在不可靠 baseline上做 KV压缩，无法区分模型 bug和压缩退化。

### 0:40–1:40：如何建立可靠 baseline

- 自实现模型和 engine，HF只做 processor/tokenizer/reference。
- 从 module → full logits → greedy → teacher-forced PPL → mixed/Graph/long output。
- BF16跨 batch shape可能因 low-margin argmax分叉，所以不滥用 token exact。

### 1:40–2:40：physical visual KV compaction

- logical pruning只少算 attention，不释放 page。
- prefill后用 attention score选 visual tokens，K/V重排，提交新 page table并释放尾页。
- logical M-RoPE位置不变，physical KV位置压缩，decode append保持两者合同。
- 质量结果约一半 active KV，但 task gate只覆盖受限数据集。

### 2:40–3:40：系统 profiling闭环

- benchmark先分开 TTFT/TPOT/E2E，Graph trace再按 node归因。
- 找到每 token整张 LM-head cast，而不是凭 kernel名字猜。
- 单变量修复、HF quality gate、五 workload、external baseline和full regression闭环。

### 3:40–4:30：两条最终 profile

- BF16 latency profile：H1/H2 TPOT 相对 SGLang 低 `4.54%–4.83%`，相对 vLLM
  低 `6.13%–6.27%`。
- scaled-FP8 capacity profile：同约 4 GiB budget 容量提升 `94.69%`，TPOT
  仍小幅领先，但不比 Prism BF16 更快。
- H1 BF16 对 SGLang E2E 近似持平，scaled E2E 有两个轻微负单元；不只报 TPOT
  之外的有利数字。

### 4:30–5:00：下一步

- 为 content-aware + scaled-FP8 组合补标准多模态质量矩阵；
- 建立真实网络 server后做 external online SLO 与容量/并发 goodput；
- 只在 profile 证明收益时研究 weight-only/outlier-correction kernel，不再扩展已失败的
  GQA4 merge 或 split-K；
- 获得合法双卡资源后再补TP2。

## 6. STAR 深挖故事

### 6.1 LM-head FP32 cast

**Situation**：Prism Graph TPOT仍明显慢于 vLLM，semantic profile显示 Graph外 logits约
`4.068 ms/token`。

**Task**：解释差距并做单变量优化，同时不破坏模型数值合同。

**Action**：

1. 用 Nsight Systems CUDA Graph node trace，把 NVTX range关联到直接 GPU activity。
2. 发现完整 `151,936×4,096` LM-head每步 BF16→FP32 direct copy和FP32 GEMV。
3. 将默认 logits改为模型原生精度，保留显式 FP32开关作历史复现。
4. 运行 HF teacher-forced logits/PPL、7-image quality、五 workload clean matrix、vLLM
   comparability和full regression。

**Result**：logits `4.068→0.762 ms`，TPOT `1.216x–1.280x`，peak减少
`2,230–2,317 MiB`；Prism仍未反超 vLLM。

**Reflection**：CPU range会暴露 stream同步，Graph replay是异步提交；不能把 ranges
简单相加。优化结论必须同时有 kernel root cause、质量与 E2E。

### 6.2 Logical pruning 到 physical compaction

**Situation**：初版视觉 pruning减少 attention输入，但 KV pool bytes不变，不能称显存
压缩。

**Task**：真实减少 physical pages，同时保持 M-RoPE、mixed batch、swap与decode append。

**Action**：引入 layout descriptor与compaction plan；prefill后将 retained K/V gather到
新 slots，原子更新page table并释放尾页；分离 logical/physical lengths；为FP8增加
Triton两阶段copy；补layout/append/swap/pickle/CoW测试。

**Result**：multi-image keep=0.5可从 `408→212` physical tokens、`2→1` active blocks；
最终 content-aware策略在7-image gate保持约`0.535x` physical tokens并通过受限词法质量
门禁。

**Reflection**：显存收益要区分 active KV bytes、预分配 pool、allocator peak和整卡
memory；它们不能互换。

### 6.3 拒绝错误优化

**Situation**：trace显示 linear/GEMV占 Graph replay `70.55%`，QKV packing看似直接。

**Task**：在性能测试前证明数学/数值等价。

**Action**：构建真实 Qwen shape BF16 probe，比较独立 q/k/v与packed输出，覆盖 batch
`1/2/4/8`。

**Result**：batch1 exact，但 batch2/4/8 K/V max diff为`1.0`，立即标记
`rejected_by_strict_correctness`且不计时。gate/up候选通过完整MLP、HF logits/PPL、
offline/online与full regression；Systems确认linear `253→217`，8个clean cell的
decode TPOT改善`0.483%–0.762%`。

**Reflection**：低精度下“数学等价”不保证数值合同；correctness-first可以减少无意义
benchmark和回归风险。不到1%、在记录cell中方向一致的decode收益也必须与不稳定的
vision-prefill E2E分开表达。

## 7. 高频面试问题

### 为什么不用 vLLM直接改？

研究目标需要看到从 visual token span、M-RoPE到physical KV page的完整语义。轻量
自实现便于建立可解释实验边界；vLLM/SGLang作为固定外部 baseline，而不是被包装成
自己的实现。生产落地未必重造框架，但研究阶段需要掌握合同和root cause。

### M-RoPE和普通 RoPE有什么区别？

文本位置可以看作一维 progression；视觉 token按 temporal/height/width三轴生成位置。
Q/K的rotary sections按配置交错应用三轴cos/sin。压缩时不能把physical slot当logical
position，否则删除 token后空间/时间位置会漂移。

### 为什么 physical KV减半，速度只提升约2%？

当前质量 workload上下文较短，decode仍受36层projection和大量elementwise kernel
主导。P7.4 trace中 paged attention仅占 replay约`13.17%`，linear/GEMV占
`70.55%`；减少一半visual KV只能优化其中一部分，而且prefill还需score。压缩首先改善
active page/capacity，不保证短上下文大幅TPOT收益。

### 如何证明不是 logical mask？

记录 physical token count、active/dense blocks、occupied bytes、released block IDs和新
page table；compaction后尾页回到free pool，decode从compressed tail append。测试覆盖
swap/pickle/CoW和mixed batch。

### 为什么质量用 token-F1/ROUGE-L，而不是只看 exact？

压缩允许输出变化，exact过严且不能衡量语义；但主观文本也不可审计。项目用固定
COCO captions做无外部依赖的lexical preflight，并保留stable prefix和逐请求输出。
它仍不是标准任务指标，因此结论严格限定，未来要补CIDEr/SPICE/VQA。

### CUDA Graph解决了什么？

它把大量Python/driver launch合并为一次Graph launch，并用固定bucket覆盖动态batch。
Prism internal decode相对eager约快`1.76x–1.94x`。它不减少Graph内部计算，也不自动
解决linear/GEMV热点；bucket padding还需要输出隔离和coverage测试。

### 你如何做公平外部对比？

固定model config hash、GPU UUID、dtype、TP、KV pool bytes、max lengths、prompt tokens、
sampling、prefix cache、preprocessing scope、execution mode、warmup/repeat和clean state。
汇总器逐字段comparability gate，任一关键字段不一致就不算ratio。双方可用各自稳定
backend，不强迫内部算法相同。

### 为什么旧 FP8 失败，而 scaled FP8 能作为成功结果？

旧 `fp8_kv` 是 scale=1 direct cast；kernel correctness和storage reduction不等于生成
质量，它在长输出发生早期分叉，所以只保留为 rejected baseline。P9-C 的
`scaled_fp8_kv` 对每个token/KV head分别计算K/V scale，并把scale纳入store、decode、
COW、swap、compaction和Graph生命周期；它在冻结的六个标准质量cell中全PASS。
但这仍不是“所有FP8无损”：组合压缩、正式runtime收益和跨框架全物理字节Pareto都要
单独验证。

### 当前最大的技术债是什么？

TP2没有合法双卡动态证据；跨框架page-table/allocator字节尚不可比；scaled FP8还没有
正式runtime speedup；online只有engine harness、没有网络server。每项都有明确恢复命令
和禁止claim。

## 8. 作品集页面建议

推荐页面顺序：

1. 一句话定位和明确边界；
2. 架构图；
3. correctness金字塔；
4. physical compaction前后page示意；
5. 质量—KV—TPOT三列结果；
6. logits trace root cause前后；
7. external对比（明确仍落后）；
8. rejected experiments与下一步；
9. GitHub、复现手册和claim ledger。

避免只放吞吐柱状图。至少展示一次“现象 → trace → root cause → 单变量修复 → quality/
E2E → external”的完整证据链。

## 9. 可公开与不可公开措辞

### 可以使用

- “自实现 Qwen3-VL 主推理路径，并以 HF作为数值 reference。”
- “实现视觉 KV physical compaction和page回收。”
- “在固定7-image lexical preflight中，physical KV约减半且指标drop小于0.004。”
- “通过trace移除per-token完整LM-head FP32转换，TPOT提升1.216x–1.280x。”
- “packed gate/up在8个clean cell中减少36个linear，并小幅改善decode TPOT
  `0.483%–0.762%`。”
- “scaled FP8在冻结三数据集的六个formal cell中全PASS，allocated KV pool为BF16的
  `0.515625x`；external quality matrix为MIXED。”
- “优化后仍慢于vLLM 1.34x–1.40x，并继续定位Graph内热点。”

### 必须避免

- “全面超过 vLLM/SGLang”。
- “显存减半”而不说明只是active prompt KV。
- “COCO accuracy下降不到1%”。
- “unit-scale FP8质量通过”或不带协议边界地写“所有FP8质量无损”。
- “scaled FP8已在全物理显存口径支配vLLM”或“已证明正式runtime加速”。
- “online serving超过vLLM”。
- “TP2已验证”。
- “实现了megakernel/PD分离/投机解码”。
- “packed MLP已提升端到端性能”。

## 10. 证据索引

| 主题 | 入口 |
|---|---|
| 阶段状态 | [ROADMAP](ROADMAP.md) |
| 验证命令与输出 | [VERIFICATION](VERIFICATION.md) |
| 性能数字与环境 | [PERFORMANCE_REPORT](PERFORMANCE_REPORT.md) |
| 压缩研究 | [COMPRESSION_REPORT](COMPRESSION_REPORT.md) |
| KV trace分析 | [KV_ANALYSIS_REPORT](KV_ANALYSIS_REPORT.md) |
| claim边界 | [CLAIMS](CLAIMS.md) |
| 未完成项 | [KNOWN_ISSUES](KNOWN_ISSUES.md) |
| 外部复现 | [REPRODUCIBILITY](REPRODUCIBILITY.md) |

投递前最后检查：数字、commit和边界是否仍与这些权威文档一致；如果后续 P7.5复跑改变
结论，先更新 claim ledger，再更新本材料。
