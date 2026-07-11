# P6 系统优化与视觉 KV 物理压缩设计

> 日期: 2026-07-11
> 状态: Design Gate Complete
> 目标: 在 P1-P5 correctness baseline 上，建立可归因的系统性能基线，实现视觉 token 感知的物理 KV 压缩，并评估 CUDA Graph、`torch.compile`、压缩 Paged Attention 和可选 megakernel 执行模式。

## 1. 阶段定位

P6 不把 Prism-Infer 扩展成通用 serving 平台，也不把“全面超过 vLLM/SGLang”设为完成条件。P6 要解决两个可验证问题:

1. 多模态推理的时间和显存具体消耗在哪些阶段，CUDA Graph、`torch.compile` 和自定义 kernel 分别能优化什么。
2. 在视觉 token 占比高的多图/视频场景中，如何把 P5 的 logical pruning 推进为真实 physical KV compaction，并在质量门槛内改善 KV bytes、最大并发或 decode 性能。

P6 的最终 claim 必须限定硬件、模型、输入、版本和采样配置。允许 Prism 在部分 workload 上不如外部框架，但所有结果必须可复现、可解释。

## 2. 当前基线与已知瓶颈

当前可复用证据:

- P3 已完成 text/image/multi-image/video/mixed batch correctness、VL CUDA Graph decode 和 Triton paged decode baseline。
- P3 mixed VL CUDA Graph benchmark 中，eager decode median `31.5488ms`，graph decode median `16.4468ms`。
- P3 paged decode kernel 在多数 batch/context case 快于 eager reference，但 batch=1/context=4096 时仍慢于 reference。
- P5 `visual_prune` 只构造 retained-token KV view，不回收物理 block；Python gather 和 eager SDPA 使其明显慢于 off。
- P5 `fp8_kv` 把固定 blocks 的 KV bytes 降为 BF16 的 `0.5x`，但 eager dequant + SDPA 使 latency 明显退化。
- P4 importance scoring 与 P5 runtime pruning 尚未形成经过验证的在线 score 映射。

当前 P6 必须先验证的瓶颈假设:

| 假设 | 证据要求 | 未证实时的处理 |
|---|---|---|
| decode 存在明显 CPU/kernel launch overhead | Nsight timeline、eager vs CUDA Graph、不同 batch TPOT | 不推进 megakernel |
| visual-prune 慢在 Python gather/同步和 eager SDPA | 分段计时、kernel timeline、retained length matrix | 先优化 runtime/layout，不盲目重写模型 |
| FP8 慢在 cache dequant 和非 fused attention | FP8 store/dequant/attention 分段计时 | 不宣称 FP8 throughput 收益 |
| visual-heavy 请求受 KV 容量限制 | 固定 32GB 下 max concurrency/OOM boundary | 不把 bytes ratio外推为系统吞吐收益 |
| `torch.compile` 可覆盖稳定模型计算区域 | graph break、recompile、generated kernel 统计 | 缩小 compile boundary，不 compile scheduler |

## 3. 非目标

P6 当前门禁不包含:

- 多机 PD 分离生产实现。
- 投机解码生产实现。
- 多机 EP、RDMA、NVSHMEM 或远端 KV transport。
- 完整 OpenAI-compatible server。
- 同时适配多个新模型。
- 在没有同条件 benchmark 时声称全面超过 vLLM、SGLang 或 vLLM-Omni。
- 在没有可运行实现时用“megakernel”命名普通算子融合。

P6 会为未来 PD 分离保留可序列化 KV layout descriptor，但不让该扩展阻塞单卡主线。

## 4. 目标架构

```text
Workload / Benchmark Adapter
  -> Processor / Qwen3-VL Semantics
  -> Scheduler + Execution Planner
       -> ExecutionMode
            eager | cuda_graph | compile | megakernel(optional)
       -> CompressionPolicy
            off | visual_compact | fp8 | visual_compact_fp8
       -> KVCacheManager
            logical context | physical context | block table | dtype/scales
       -> AttentionBackend
            sdpa_ref | paged | compressed_paged | fp8_paged
  -> Metrics / Trace / Profiler
       correctness | TTFT | TPOT | throughput | memory | capacity
```

实验维度必须正交。改变一个维度时，其余配置保持不变；不允许把新 kernel、新压缩算法和新调度策略同时打开后把总收益归因给其中一个模块。

### 4.1 ExecutionMode

最小接口语义:

```text
eager:
  Python/PyTorch 正常执行，作为系统 correctness 和性能 reference。

cuda_graph:
  replay 已验证的稳定 decode graph；不改变 attention 算法和 KV layout。

compile:
  只编译 Vision Encoder、LLM prefill 或 LLM decode 的稳定 tensor 区域；
  scheduler、BlockManager 和请求生命周期留在 Python runtime。

megakernel:
  仅在 profiler 证明 workload 为 launch-bound 后评估；必须有独立实现和
  相同输入下的 correctness/performance 数据，不作为 P6 核心门禁。
```

`torch.compile` 不直接理解动态 KV allocator。Paged Attention、KV store 或 compact kernel 应通过项目自有 custom op 边界接入，并提供 fake/meta contract；是否进一步实现 Inductor lowering 由 profiling 决定。

### 4.2 KVCacheLayoutDescriptor

physical compaction 后不能继续让一个 `context_len` 同时表示逻辑位置和物理 KV 数量。目标 descriptor 至少包含:

```text
logical_context_len: 原始 prompt + generated token 逻辑长度
physical_kv_len: 当前实际保存的 KV token 数
prompt_logical_len: 原始 prompt 长度
compressed_prompt_kv_len: 压缩后 prompt KV 长度
retained_original_positions: retained KV 对应的原始 token position
block_table: 当前物理 KV pages
kv_dtype: bf16/fp8
kv_scale_metadata: 可选 per-page/per-head scale
compression_record: strategy/ratio/modality/decision version
```

约束:

- M-RoPE query position 继续使用 `logical_context_len`。
- 已写入 cache 的 K/V 保留原始 position 语义；物理搬移不重新应用 RoPE。
- decode append 使用 `physical_kv_len` 选择写入 slot，同时用 `logical_context_len` 生成 position ids。
- block 回收后，旧 block id 不得继续存在于 hash、free-list、swap 或 prefix metadata。
- descriptor 必须能跨 Sequence 序列化，并为未来 KV transport 保留稳定 schema。

### 4.3 CompressionPolicy

P6 支持的评估模式:

| mode | 决策时机 | 物理显存 | prefill 计算 | decode KV |
|---|---|---:|---:|---:|
| off | 无 | baseline | baseline | baseline |
| visual_prune logical | prefill 后 | 不下降 | 不变 | retained view |
| visual_compact | prefill 后 | 下降 | 不变 | compact pages |
| fp8_kv | allocation/store | 下降 | 近似不变 | FP8 cache |
| visual_compact_fp8 | prefill 后 + quantize | 进一步下降 | 不变 | compact FP8 pages |

第一版 physical compaction 使用现有 uniform decision 证明 layout correctness。P4 score 接入必须作为后续独立子任务，避免把 scoring 错误与 block layout 错误混在同一次实现中。

外部 pruning PR 在链接/commit 未提供前只登记为 `pending_external_reference`，不能写入实现结论或对比数字。获得 PR 后必须先判断其发生在 Vision Encoder 前、LLM prefill 前还是 prefill 后 KV 阶段。

### 4.4 AttentionBackend

推进顺序:

1. 保留 PyTorch SDPA/eager gather 作为 correctness reference。
2. physical compaction 先复用现有 dense paged decode kernel，因为 compact pages 对 kernel 应表现为更短的 physical context。
3. 增加 shape-aware `BLOCK_N/num_warps` 调优，修复 batch=1 长上下文风险。
4. FP8 路径在 kernel 内 load/dequant，避免整段 cache eager 转回 BF16。
5. 只有在上述路径稳定后，才评估 persistent/megakernel decode。

## 5. Benchmark 协议

### 5.1 Workload schema

每个 case 必须记录:

```text
case_id
model snapshot/commit
prompt text token count
image/video count and input shape
visual token count
output token count
batch/concurrency/request rate
dtype/TP/block size/max model len
execution/attention/compression mode
preprocessing included or excluded
```

最小 workload matrix:

| workload | 目的 | 最小形态 |
|---|---|---|
| text | 防止 VL 优化破坏 text baseline | 256/1024 text tokens |
| single_image | 基础 VL correctness | 1 x 448x448 |
| multi_image | visual-heavy KV | 2/4/8 images |
| video | 长 visual context | 4/8/16 frames |
| mixed batch | scheduler/graph | text + image + video |
| long decode | TPOT/KV 压力 | output 32/128/256 |

synthetic inputs用于 deterministic smoke；最终质量 claim 必须增加固定的真实图像、OCR、多图和视频样例集。

### 5.2 Metrics

系统 benchmark 至少输出:

- TTFT: first output token 前的端到端时间。
- TPOT/ITL: decode token 间延迟，至少 median/p90/p99。
- end-to-end latency: median/p90/min/max。
- request/s、output token/s；如报告 total token/s，必须单独标出 visual/input tokens。
- GPU memory allocated/reserved/peak。
- physical KV bytes、logical/physical KV token count、block count。
- 固定显存下 max concurrency 和 OOM boundary。
- GPU utilization、kernel count；P6.2 profiling case 记录 CPU launch gap。
- quality: exact token、稳定前缀、teacher-forced logits/ppl 或任务指标。

所有 GPU timing 使用 warmup、repeat 和 `torch.cuda.synchronize()` 边界。online benchmark 另需记录 arrival distribution 和并发策略；在 Prism 没有同等 online server 前，不用 offline 数字声称 online serving 超越。

### 5.3 Baseline hierarchy

| 层级 | baseline | 用途 |
|---|---|---|
| correctness | Hugging Face | 模型/质量 reference |
| kernel | PyTorch SDPA/eager | 自定义 kernel correctness/performance |
| internal system | Prism off + eager + BF16 | ablation 主 baseline |
| execution | Prism eager/graph/compile | 归因 launch/compile 收益 |
| compression | off/logical/compact/fp8/hybrid | 质量-显存-性能 Pareto |
| external system | vLLM/SGLang/vLLM-Omni fixed commit | 同条件系统对比 |
| optional execution | real megakernel implementation | launch-bound decode 实验 |

### 5.4 外部框架公平性

每个外部 adapter 必须固定:

- repo URL、version/commit 和未提交 diff。
- model snapshot、processor/tokenizer version。
- dtype、attention backend、TP size、max model len、block/page size可见配置。
- GPU memory utilization、prefix cache、chunked prefill、CUDA Graph或等价配置。
- 相同 prompt、图像/视频、output length、sampling 和 EOS 行为。
- preprocessing 是否计时。
- warmup、repeat、并发和请求顺序。

不能强制外部框架采用与 Prism 相同的内部算法，但必须记录其真实 backend。外部对比只给系统级结果，不用结果反推对方内部实现。

## 6. P6 子阶段

### P6.0 设计门禁

交付:

- 本文档。
- `ROADMAP.md` 和 `VERIFICATION.md` 同步。
- 明确 P6 核心门禁与 optional 项。

状态: Completed。

### P6.1 统一 benchmark contract

交付:

- 统一 JSONL schema。
- Prism internal baseline runner。
- deterministic workload manifest。
- benchmark schema focused tests。
- 首份 `docs/PERFORMANCE_REPORT.md` baseline section。

第一实现任务: 新增 `benchmarks/bench_system.py` 和对应 schema/test，不做性能优化。

### P6.2 分层 profiling

交付:

- processor、vision、prefill、decode、sample、scheduler 分段时间。
- eager/graph kernel count 和 CPU launch gap。
- visual-prune gather 与 FP8 dequant 的独立成本。
- 每个优化任务绑定一个已测瓶颈。

### P6.3 执行后端门禁

交付:

- eager 与 CUDA Graph 当前结果刷新到统一 harness。
- Vision/prefill/decode compile boundary 调查。
- graph break、recompile、compile time、steady-state 数据。
- compile correctness 与 off baseline 对齐。

### P6.4 Visual KV physical compaction

交付:

- `KVCacheLayoutDescriptor` 或等价稳定 contract。
- logical/physical length 分离。
- post-prefill compact、block 回收、decode append。
- Sequence pickle、swap、CoW、mixed batch、M-RoPE correctness。
- keep-all physical path 与 off exact equality。

### P6.5 Compressed/FP8 paged attention

交付:

- compact pages 复用/扩展 paged decode kernel。
- FP8-aware load/dequant correctness。
- Qwen GQA shape、batch/context matrix。
- 每个 kernel 与 independent reference 的 max diff/mean diff。

### P6.6 质量、容量与 Pareto 评估

交付:

- keep ratio `0.25/0.5/0.75/1.0`。
- text/single-image/multi-image/video/mixed matrix。
- quality-memory-TPOT Pareto 数据。
- 固定 32GB max concurrency/OOM boundary。

### P6.7 外部框架对比

交付:

- vLLM/SGLang/vLLM-Omni adapter 或可复现命令。
- 固定 commit 与配置 manifest。
- Prism off 与 compression-on 都参与比较。
- 不要求全面胜出；报告优势区间、劣势区间和原因。

### P6.8 两卡 TP 验证

交付:

- 1 GPU vs 2 GPU greedy/logits correctness。
- 权重/KV heads shard evidence。
- NCCL collective、显存和 latency 数据。
- 小 workload 通信退化也必须如实记录。

### P6.9 Megakernel 可选实验

启动条件:

- P6.2 证明目标 decode workload 为 launch-bound。
- 有可运行实现或项目内明确 kernel scope。
- 不阻塞 P6.4-P6.8。

交付:

- 相同模型区域和输入下，discrete eager、CUDA Graph、compile 和 megakernel 对比。
- kernel count、CPU launch、TPOT、显存、correctness。
- 明确 megakernel 是否能与 compile/graph 组合，而不是把三者描述为互斥框架。

### P6.10 阶段 Review

交付:

- P1-P5 回归不退化。
- `docs/PERFORMANCE_REPORT.md` 完整。
- 原始 JSONL 和复现命令存在。
- 所有对外 claim 有 commit、配置和日志。

## 7. P6 核心出口标准

P6 核心门禁完成必须同时满足:

- 统一 benchmark 能复现 Prism off/eager baseline。
- 至少一个 physical visual KV compaction mode 真实减少 block/KV bytes。
- compression-on 有质量、TTFT、TPOT、throughput、memory 和 capacity 数据。
- CUDA Graph 与 `torch.compile` 结论来自同条件 internal ablation。
- 至少完成一个自定义 kernel 的 correctness + benchmark 优化闭环。
- 完成 2 卡 TP correctness 和基础性能验证。
- 完成固定版本的外部框架公平对比；不要求所有 workload 胜出。
- P1-P5 full regression 不退化。
- 输出 `docs/PERFORMANCE_REPORT.md`。

megakernel、PD 分离和投机解码不属于 P6 核心出口标准。

## 8. 目标指标与 claim 边界

以下是研究目标，不是当前结果:

- visual-heavy workload 物理 KV bytes 下降目标 `>=40%`。
- 真实任务 accuracy 下降目标 `<1%`，或 teacher-forced ppl diff `<0.1`。
- compression-on TPOT 目标不超过 off 的 `1.05x`，争取实现降低。
- 固定 RTX 5090 32GB 下 max concurrency 提升目标 `>=30%`。
- 至少在一个明确 workload 上形成相对固定版本外部框架的可测优势。

未达到目标不等于实验失败，但必须分析瓶颈并收缩 claim。禁止通过更换输入、减少 output length、降低外部框架显存预算或忽略 preprocessing 来制造优势。

## 9. 关键设计决策

### D1: 先建统一 baseline，再优化

选择: P6.1 先完成 benchmark contract，不修改核心执行路径。

拒绝: 直接添加 `torch.compile`、新 kernel 或 physical compaction 后再补 baseline。没有优化前数据无法归因收益。

### D2: physical compaction 是主线，FP8 是正交维度

选择: visual token 选择与 KV dtype 分开，支持 compact-only、FP8-only 和 hybrid。

拒绝: 把全局 FP8 bytes 减半写成视觉 token 专属优化。P5 FP8 是物理存储 baseline，但不具备 modality awareness。

### D3: compile 只覆盖 tensor compute region

选择: scheduler、BlockManager 和动态 request state 留在 Python；通过 custom op 隔离 KV/attention。

拒绝: compile 整个 engine。动态对象、Python side effect 和 allocator 会让结果难以稳定和解释。

### D4: megakernel 由 profiling 触发

选择: 只有 launch-bound 证据和真实实现后才进入 P6.9。

拒绝: 因概念热门而把普通 fusion 或 CUDA Graph称为 megakernel。

### D5: 外部对比不以全面超过为门禁

选择: 寻找视觉长上下文、显存受限等 Prism 策略真正适用的窄场景，同时完整报告劣势。

拒绝: 用不同版本、输入或显存配置制造“超过 vLLM/SGLang”的总括性结论。

## 10. 立即下一步

进入 P6.1，先实现统一 benchmark contract:

1. 定义 JSONL result schema 和 workload manifest。
2. 新增 `benchmarks/bench_system.py`，只支持 Prism internal modes。
3. 复现 off/eager、off/CUDA Graph、visual_prune 和 fp8_kv 现有 case。
4. 分离 preprocessing、TTFT、decode/TPOT 和 end-to-end timing。
5. 新增 benchmark schema focused tests。
6. 生成 `docs/PERFORMANCE_REPORT.md` 的 baseline section。

在 P6.1 完成前，不开始 physical compaction、`torch.compile` 或 megakernel 实现。
