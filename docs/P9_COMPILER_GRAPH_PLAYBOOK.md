# P9 Compiler / CUDA Graph Pipeline Playbook

> 状态：执行合同已冻结，P9-D 实现与正式测量待完成
> 目标：把 `torch.compile` 和 CUDA Graph 优化推进到完整推理 pipeline 的可证明边界，
> 同时沉淀可复现的教学材料与面试问题链。

## 1. “优化到极致”的验收定义

这里的“极致”不等于把所有 Python 函数都交给 compiler，也不等于把 kernel 数降到最少。
只有同时满足以下条件，候选才能进入 supported 或 performance claim：

1. **Pipeline coverage 可解释**：明确 host、prefill、decode、KV、LM head、sampler、copy
   和同步边界，不能只展示一个子图或一个 kernel。
2. **Correctness 不退化**：固定输入 shape 下 token exact；HF/model-precision logits、
   nonzero-storage-offset KV、BF16/scaled-FP8 和 padding row 均通过门禁。
3. **执行稳定**：没有 silent fallback、隐藏 graph break、非预期 recompile 或 bucket 漂移。
4. **内存安全**：capture/functionalization 不复制错误的 aliased KV view，不制造不可控
   cold-compile peak，也能在失败和退出路径释放 Graph/model/KV ownership。
5. **端到端有效**：报告 CPU launch、GPU span、kernel busy、TPOT、TTFT/E2E 和显存；
   只有 fresh-process repeats 的收益才形成结论。
6. **失败同样可交付**：若 compiler 的正确边界小于 Graph，保留 root cause、rejected
   candidate 和止损依据，不把“捕获更多”误写成“优化更好”。

## 2. Pipeline 边界

```text
request ingress / tokenizer / processor
              │
              ▼
request validation ── scheduler/admission ── BatchPlan
              │
              ▼
host input preparation ── H2D copy ── DeviceBatch
              │
       ┌──────┴────────┐
       ▼               ▼
vision + prefill    decode steady state
                         │
                         ├─ embedding / position update
                         ├─ decoder layers
                         │    ├─ QKV + QK norm + M-RoPE
                         │    ├─ KV quant-store
                         │    ├─ paged attention
                         │    └─ MLP / residual / norm
                         ├─ LM head
                         ├─ greedy argmax sampler
                         └─ minimal result/status copy
```

P9-D 的第一目标是 **greedy full-step CUDA Graph**：steady-state decoder、
model-precision LM head 和 argmax sampler进入同一 replay 边界。动态 vision/prefill 不为追求
覆盖率强行 capture；它们独立 profiling，并只在有稳定 bucket 和收益证据时进入候选。

## 3. 必须采集的证据

| 层级 | 必须记录 | 主要工具 |
|---|---|---|
| Host pipeline | validation、scheduler、prepare、launch、同步与结果物化时间 | NVTX/NSYS、结构化 timer |
| Graph | capture bucket、node 数、CPU replay range、GPU replay span、GPU tail | CUDA events、NSYS |
| Compiler | graph breaks、guards、recompile、cold time、generated code cache、peak memory | Dynamo/Inductor logs、preflight artifact |
| Decoder | attention、MLP、norm、LM head、sampler的GPU region和kernel数量 | NVTX/NSYS |
| Kernel | duration、grid、occupancy、waves/SM、register、shared memory、DRAM/compute | NCU |
| KV | payload/scale view、page size、context length、slot/block table、storage offset | schema + focused tests |
| E2E | TTFT、TPOT、throughput、p50/p95/p99、allocated/reserved/NVML bytes、vision backend | benchmark harness |

CPU Graph replay range只是异步提交时间，不能当作完整 GPU step；GPU span减去kernel busy
也不能直接叫 occupancy。每个时间范围必须声明起止事件和同步语义。

## 4. Backend 候选矩阵

| Backend | 作用 | 当前状态 | P9-D 判定 |
|---|---|---|---|
| eager | correctness与归因基线 | supported | 保留 |
| model-only CUDA Graph | 当前强内部基线 | supported | 与 full-step并列重跑 |
| greedy full-step CUDA Graph | decoder + LM head + argmax | pending | 主候选 |
| pure compile subgraph | QKV/QK-Norm/M-RoPE | memory-safe、batch2分叉 | rejected evidence |
| compile + full-step Graph | 相同DeviceBatch/capture边界 | pending | 只做一次正式候选 |

任何 backend 超出支持的 batch/page/precision bucket 必须 startup fail closed，不能退回 eager
后仍把记录标成 Graph/compile。

Vision attention backend 与 decode execution backend 是两条正交轴。默认 `sdpa` 用于 strict
reference；显式 `flash_attn` 必须单独完成 single-image/H1/H2 质量、vision latency、TTFT
和峰值显存矩阵。不得由可选包存在性或输入 segment 数静默切换，也不得混合两者的 repeats。

## 5. Correctness 与 shape matrix

最低覆盖：

- batch `1/2/4/8`；
- output `4/32/128`；
- page `16/32`，保留 page256历史基线；
- context 包含整页和非整页/ragged tail；
- BF16 与 `scaled_fp8_kv`；
- text、single-image、H1 8-image，H2 只做语义一致时的外部比较；
- vision SDPA 为必测基线，FlashAttention 为显式候选且不得放宽同 bucket 稳定性门禁；
- padding row不写KV、不泄漏logits、不影响真实request；
- monolithic KV中非零`storage_offset`的K/V/scale view；
- capture/replay后COW、compaction、exit与重复engine lifecycle。

固定 shape 必须重复 token exact。跨 batch shape 的低-margin argmax差异只能作为单独数值
边界分析，不能掩盖同一 bucket 的非确定性。

## 6. 优化顺序

1. 在新 GPU UUID 上建立 eager/model-only Graph clean baseline。
2. 用 NSYS 把完整 decode step分解，列出所有Graph外CPU/GPU工作和同步。
3. 把 model-precision LM head纳入稳定 device buffer，验证权重不发生逐步转换/复制。
4. 把 argmax sampler和必要的状态更新纳入Graph，结果只做最小D2H copy。
5. 验证 full-step bucket、padding、KV/scale view和生命周期，再做正式 repeats。
6. 在同一DeviceBatch边界测试compile+Graph，记录break/guard/recompile/cold peak。
7. 只有profile仍指向attention并行度，才实现split-context/stable-softmax kernel。
8. 用NCU解释kernel变化，用H1 full-engine TPOT决定是否合入。

第1步统一通过`benchmarks/run_p9_process_matrix.py`执行：标准库 parent 不导入 torch，
每个 mode/repeat 一个 child process，运行前后按物理 UUID 做 idle/release gate，并保存
ABBA/BAAB 顺序、完整 comparability checks 和 process-level bootstrap 95% CI。具体命令
见`docs/REPRODUCIBILITY.md`第11节；该生命周期缺口及修复记录为 P9-005。

requested traffic batch、scheduler 发布的 actual decode batch 和 Graph captured bucket 是
三个不同层级。H1 batch4 会受视觉 patch admission 与 prefill/decode interleaving 影响，
实际先经历 batch `1/2/3`，其中 actual 3 replay captured bucket 4；output128 才会在后段
形成 batch4 steady state。因此正式 artifact 必须保存每个 actual bucket 的 step count 和
actual→captured 映射，eager/Graph actual histogram 必须 exact，不能用 nominal batch4 或
最后一次 replay 代替。完整问题链记录为 P9-006。

## 7. 止损规则

- compile再次出现OOM、非法alias处理或同bucket token分叉：保持rejected，不扩大捕获面。
- full-step Graph没有降低GPU span或TPOT：先定位同步/尾部工作，不用kernel数量替代收益。
- split-context只改善microbenchmark、H1 TPOT不足约`3%`：不迁移CUDA/CUTLASS。
- 任何候选不能在当前32 GiB上稳定capture并释放：不作为supported backend。
- 不能通过至少5次fresh-process repeats和95% CI：不形成“优化到极致”的最终结论。

## 8. 问题与面试故事记录

每个真实问题追加到`docs/ISSUE_LOG.md`，至少记录：

```text
现象与影响
最初假设
如何证伪
最终根因
修复与为什么这样设计
拒绝的替代方案
correctness / profiler / E2E证据
仍然存在的限制
两分钟面试讲法
```

不只记录成功优化。OOM、graph break、数值分叉、错误同步、无E2E收益和被拒绝的kernel
同样是重要工程结果，只要证据完整且结论诚实。

## 9. 后续教学顺序

1. CUDA Graph的capture/replay、静态地址、stream和同步模型。
2. Dynamo guard、graph break、AOTAutograd functionalization与Inductor codegen。
3. Prism的DeviceBatch、KV aliased view和Graph ownership逐行走读。
4. NSYS时间线：CPU launch、GPU span、kernel busy与尾部同步。
5. NCU：grid、occupancy、waves、寄存器与split-context设计。
6. 用一次成功优化和一次rejected compiler问题完成面试演练。
