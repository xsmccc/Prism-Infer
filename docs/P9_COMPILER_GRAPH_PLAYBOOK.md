# P9 Compiler / CUDA Graph Pipeline Playbook

> 状态：full-step CUDA Graph 已完成；P10 batch1 stateless compile + Graph 候选已在
> RTX 5090 上通过 clean H1 与代表性 token-exact 验证（2026-07-23）
> 目标：把 `torch.compile` 和 CUDA Graph 优化推进到完整推理 pipeline 的可证明边界，
> 同时沉淀可复现的教学材料与面试问题链。

## 0. P10 结论：编译更少，但覆盖完整 decode replay

最终支持的 `compile_graph` 没有编译整个 decoder。NSYS 显示原始 full-step Graph 的
decode kernel busy 中位数为 `10.119693 ms`，其中线性层/GEMV约占 `92.33%`；实验随后证明
QKV和MLP交给Inductor会改变BF16 reduction order，并在短文本上造成逐token分叉。因此最终
边界只包含两个已验证的无状态热点：

- batch1 attention `o_proj` 由Inductor编译，逐bit保持原输出；
- LM head先用动态activation scale、逐行weight scale的FP8 `_scaled_mm` 生成top-64候选，
  再由两个Triton kernel使用原始BF16权重做FP32点积和确定性tie-break；
- decoder、mutable KV、paged attention、QKV和MLP保留原精确路径；完整model forward、
  candidate generation、rerank、greedy token与最小D2H仍处于同一个CUDA Graph replay。

clean commit `6052205fd7e740aa166228155789c2e4ae069929` 在同一张
`GPU-7f63f8b0-1027-d3bf-18b7-5102cbc9f2eb` RTX 5090、Torch 2.11/CUDA 13、同一模型快照和
H1 8-image/128-output协议上的结果如下：

| 系统 | TPOT median (ms) | Prism相对领先 |
|---|---:|---:|
| Prism `compile_graph` top-64 | **9.8510** | — |
| Prism原full-step Graph | 10.3174 | **4.52%** |
| SGLang/Triton | 10.3513 | **4.83%** |
| vLLM | 10.5215 | **6.37%** |

三次clean TPOT为`9.8508 / 9.8510 / 9.8786 ms`，128-token SHA256为
`76ad1fb97daffe7dcbdec4300198350a5dac1341f09a78e312e55ae3376e14c6`，repeat内一致。
短文本、单图、双图、16帧视频和H1均与原Graph基线逐token exact。最终NSYS node trace
覆盖127个decode step：kernel busy中位数`9.670442 ms`、`384` kernels/step；原基线为
`10.119693 ms`、`388` kernels/step。cold compile、一次性FP8 weight量化和Graph capture分别约
`408.2 / 30.2 / 1044.1 ms`，不计入steady-state TPOT。

主要证据：

- `data/p10_compile_graph/stateless_candidate/h1_final_top64_repeat3_clean_6052205.jsonl`
- `data/p10_compile_graph/stateless_candidate/h1_final_top64_semantic_nodes_clean_6052205_analysis.json`
- `data/p10_compile_graph/correctness_matrix/*_compile_graph_final_top64.jsonl`
- 同卡外部基线：`data/p10_compile_graph/current_env_external/`

结论只适用于上述GPU UUID和冻结协议。batch大于1仍走原BF16投影/LM-head路径；top-64 recall
已由代表性矩阵和H1证明，但不能外推为任意模型、采样策略或硬件上的无条件等价。

## 0.1 P10.1：Vision 分段元数据只物化一次

P10 decode 收口后，H1 prefill 的 NSYS 归因发现：SDPA 多图路径在每个 ViT attention
层内对 CUDA `cu_seqlens` 调用两次 `.tolist()`。H1 使用两个 vision microbatch，
因此 27 层会重复产生 `2 × 27 × 2 = 108` 次 device-to-host copy 和 stream sync。

commit `e8eed9c2ba482c9a4ec365f4f10072114d718fda` 在每个 microbatch 的动态准备阶段一次性
生成不可变 `segment_ranges`，并沿 `VisionEncoder -> ViTBlock -> ViTAttention` 传递。
FlashAttention 仍使用原 `cu_seqlens`；直接调用 SDPA attention 且未提供预计算边界时，
保留兼容回退。模型计算、权重、精度和 decode Graph 均未改变。

同一 RTX 5090、同一模型快照和 H1 协议的 clean 证据：

- prefill stream sync：`187 -> 79`，async memcpy：`217 -> 109`，均精确减少 `108`；
- prefill kernel busy：`170.594 -> 170.502 ms`，kernel 数：`5020 -> 5018`，证明收益来自
  删除 host/device 同步，而不是减少主要模型计算；
- clean H1 warmup 2/repeat 3 TTFT 中位数：`271.409 -> 240.054 ms`，改善 `11.55%`；
- clean H1 E2E 中位数：`1595.090 -> 1569.382 ms`，改善 `1.61%`；
- decode TPOT 为 `9.8695 ms`，仍落在原 top-64 clean 波动范围内；128-token SHA256
  保持 `76ad1fb97daffe7dcbdec4300198350a5dac1341f09a78e312e55ae3376e14c6`；
- clean 双图与 16 帧视频 guardrail 也分别保持既有 token SHA256。

主要证据：

- `data/p10_compile_graph/vision_segment_ranges/h1_repeat3_clean_e8eed9c.jsonl`
- `data/p10_compile_graph/vision_segment_ranges/h1_vision_segment_ranges_semantic_clean_e8eed9c_analysis.json`
- `data/p10_compile_graph/vision_segment_ranges/guardrail_two_image_448_clean_e8eed9c.jsonl`
- `data/p10_compile_graph/vision_segment_ranges/h2_video_16x448_clean_e8eed9c.jsonl`

## 0.2 P10.2：单 payload prefill 不做空 concat

H1 的八张图在一个请求内已经是单个连续 `pixel_values` tensor，但旧
`_multimodal_inputs` 仍执行 `torch.cat([payload])`，先在 host 复制完整的
`38,535,168 B`，再 pin memory 和 H2D。commit
`79f631ef5d5260dd8bec416e259d26cb692373e1` 对单 chunk 直接复用原 tensor；
多请求/多 chunk 仍保留原 concat，pinned nonblocking H2D 语义不变。

同一张 RTX 5090 上的定向 staging microbenchmark：

- 旧 `cat + pin + H2D` 中位数 `5.11 ms`；
- 新 `pin + H2D` 中位数 `1.68 ms`；
- pageable H2D 中位数 `2.90 ms`，虽然尾延迟更稳，但 H1 E2E 更差，因此被拒绝。

clean 语义 NSYS 中，`runner.prepare_inputs` 的两个观测 range CPU 中位数
`35.84 -> 15.77 ms`；传输仍为 `11` 次、`38,855,808 B`，stream sync 仍为 `4`，
证明只删除 host copy。完整 prefill kernel busy 保持
`170.502 -> 170.449 ms`。紧邻的 clean H1 warmup 2/repeat 5 对照中，TTFT 中位数
`255.159 -> 246.532 ms`（`3.38%`），token SHA256 不变。E2E 分布受图像预处理
波动影响，没有形成本项的正式收益 claim。

主要证据：

- `data/p10_compile_graph/prefill_staging/h1_baseline_repeat5_clean_35ebecc.jsonl`
- `data/p10_compile_graph/prefill_staging/h1_single_chunk_fastpath_repeat5_clean_79f631e.jsonl`
- `data/p10_compile_graph/prefill_staging/h1_single_chunk_semantic_clean_79f631e_analysis.json`

## 0.3 P10.3：两条“profile 更漂亮”但不能合并的候选

继续清理 vision 元数据时，候选把每个 microbatch 的 grid Python rows 只物化一次，
并在 CPU 构造 cu-seqlens。NSYS 中 prefill stream sync `79 -> 11`、async memcpy
`109 -> 39`，vision 内 sync `71 -> 3`，kernel busy 保持约 `170.4 ms`。但紧邻
repeat9 A/B 的普通 H1 TTFT 中位数是候选 `251.454 ms`、clean 基线 `231.075 ms`，
候选反而慢 `8.82%`。它已完整回退，没有为了 sync 数字而牺牲真实 latency。

packed gate-up 则暴露了一个真实但尚不满足 correctness 的 kernel 窗口。第一层真实
batch1 decode activation/weight 上，编译后的动态 FP8 `_scaled_mm` 中位数
`0.0787 ms`，比 BF16 GEMV `0.1218 ms` 快 `35.4%`；未编译 FP8 路径为
`0.1615 ms`，说明收益依赖 `torch.compile` 融合 activation scale 与量化。单层
SwiGLU 激活 RMSE 为 `4.46e-4`。然而临时替换全部 36 层后，64-token 短文本在第
2 个生成 token 就从基线 token `358` 分叉为 `11`，因此没有进入 supported 路径。

这两项分别证明：

- CUDA API/sync 数下降不是端到端收益的充分条件；
- FP8 gate-up 的速度上限值得继续研究，但必须先有可证明的 outlier/residual
  correction，不能用近似输出污染当前 token-exact 主结果。

保留证据：

- `data/p10_compile_graph/vision_grid_rows/h1_grid_rows_repeat9_dirty.jsonl`
- `data/p10_compile_graph/vision_grid_rows/h1_grid_rows_baseline_repeat9_clean_6d67dfa.jsonl`
- `data/p10_compile_graph/vision_grid_rows/h1_grid_rows_semantic_dirty_analysis.json`
- `data/p10_compile_graph/gate_up_fp8/layer0_real_decode_compile_probe_6d67dfa.json`
- `data/p10_compile_graph/gate_up_fp8/guardrail_text_short_all_layers_probe_6d67dfa.json`

## 0.4 P10.4：batch4 编译、MLP FP8 与视觉 FlashAttention 的止损边界

clean `797c4bc` 的 H1 batch4 基线完整捕获 batch `1/2/3/4`，其中实际 replay 次数为
`6/6/6/372`。128-token 四请求输出 repeat 内一致，SHA256 为
`1e41d0c6b46b59018d634b2172fbfdc1e42637ece16a3e2d0c2ac24cf094e04f`；
decode step 中位数 `11.9959 ms`，四路合计 decode throughput `326.43 token/s`。
NSYS 的稳定 Graph replay CUDA 中位数约 `12.64 ms`（含 profiling overhead），
kernel busy 约 `10.86–11.60 ms`。整个 capture 的 CUDA kernel 时间中，通用 BF16
CUTLASS GEMM 占 `59.3%`、带 epilogue 的 MLP GEMM 占 `19.2%`、FlashAttention
split-K 约占 `5.0%`；因此 batch4 的主导瓶颈是小批矩阵乘，不是 scheduler 或 KV。

随后验证了三条看似自然但没有真实收益的扩展：

1. 把动态 rowwise FP8 LM-head candidate generation 从 batch1 扩到 batch1–4，
   同时保持 BF16-weight/FP32-dot top-64 精确重排。batch4 token SHA256 完全不变，
   但 decode step 仅 `11.9959 -> 11.9916 ms`（`0.04%`），而冷 Graph capture
   `692 -> 7510 ms`。再把 36 层 `o_proj` 的 Inductor 路径扩到 batch1–4 后，
   decode step 为 `12.0040 ms`，反而慢 `0.07%`。两项均已完整回退。
2. gate-up 的 1×128 blockwise FP8 scaling 在当前 Torch 2.11/CUDA 13 后端上不能用于
   `(1,4096) × (4096,24576)`：fast accumulation 被 API 明确禁止，标准累加又在
   `cublasLtMatmulAlgoGetHeuristic` 返回 `CUBLAS_STATUS_NOT_SUPPORTED`。改为只量化
   decoder 尾部 4/8/12 层时，短文本 token hash 均保持；但零量化 eager 对照 TPOT
   `25.838 ms`，最好的尾部 8 层仍为 `26.129 ms`，慢 `1.13%`，因此也不进入 Graph。
3. 独立 `flash-attn` 未安装，但 vLLM bundled interface 提供兼容的 vision varlen
   kernel。临时 fallback 通过 CUDA parity test，H1 128-token hash 也与 SDPA 相同；
   紧邻 repeat3 的 TTFT 却是 vLLM FlashAttention `246.035 ms`、SDPA
   `244.103 ms`，Flash 慢 `0.79%`。导入 fallback 已回退，避免无收益依赖耦合。

这一轮的工程结论是：当前 batch1 `compile_graph` 边界不能机械外推到 batch4；
BF16 CUTLASS 已经很好地覆盖 M=2–4，而 rowwise quantization、rerank 和新增 shape
compile 抵消了 FP8 tensor-core 收益。后续若继续攻 batch4，应进入可度量的 GEMM
kernel/量化算法研究，而不是扩大 Dynamo 捕获面。

保留证据：

- `data/p10_compile_graph/batch4/h1_b4_current_graph_repeat3_clean_797c4bc.jsonl`
- `data/p10_compile_graph/batch4/h1_b4_semantic_v3_clean_797c4bc_analysis.json`
- `data/p10_compile_graph/batch4/h1_b4_compile_graph_fp8_repeat3_v3_dirty.jsonl`
- `data/p10_compile_graph/batch4/h1_b4_compile_graph_fp8_oproj_repeat3_dirty.jsonl`
- `data/p10_compile_graph/gate_up_fp8/layer0_blockwise128_backend_rejection_797c4bc.json`
- `data/p10_compile_graph/gate_up_fp8/guardrail_text_short_zero_layers_probe_797c4bc.json`
- `data/p10_compile_graph/gate_up_fp8/guardrail_text_short_last{4,8,12}_layers_probe_797c4bc.json`
- `data/p10_compile_graph/vision_flash/h1_{vllm_flash,sdpa}_repeat3_dirty.jsonl`

## 0.5 P10.5：H2 归因与 bit-exact prefill SwiGLU

H2 clean `96d7090` 的同卡 compile+Graph repeat3 为 prompt `1667`、output `128`，
TPOT 中位数 `9.8535 ms`、TTFT 中位数 `250.106 ms`，输出 SHA256
`4a61f1adb74d2c774edca95eb18f8f101f5a87e21c9846884045933bf208166f`。
vLLM 0.25.1 虽得到 `10.5243 ms` TPOT，但只生成 `1665` 个 prompt token；首个差异发生
在时间戳/视觉 special-token 顺序，因此按 manifest 合同只保留为诊断，不能写成 H2
外部领先 claim。Prism 与 vLLM adapter 现都记录不泄漏 token 内容的
`prompt_token_ids_sha256`，后续比较必须先通过 prompt identity。

H2 NSYS 将 prefill 定位为 compute-bound：`177.820 ms` kernel busy 中，语言模型占
`128.309 ms`，其中 BF16 线性投影占 `110.586 ms`（`86.2%`）。Vision 两个
microbatch 共 `48.585 ms` kernel time、每个 14 次 stream sync；这些同步与 P10.3
已否决的 grid-row 路径相同，不能因为 sync 数更少而重做。Torch 2.11
Vision tensor-region 在严格精度配置下 steady latency `10.693 -> 7.603 ms`，但输出
最大差异仍为 `0.515625`；进一步收缩到 vision RoPE 后为
`0.0855 -> 0.0905 ms`，仍有 `0.0078125` 差异。两条 compiler 候选均不接入。

真正合入的优化来自语言 prefill 的激活层。原实现只在 batch `1–4` decode 使用
bit-exact Triton SwiGLU，prefill 每层仍分别启动 SiLU 和 BF16 multiply。新路径对大
token 矩阵使用 1024-element/8-warp tile，并显式保留“SiLU 先舍入到 BF16，再乘 up”
的 eager 舍入边界。真实 H2 `[1667, 24576]` 输入上，单层激活中位数
`0.1268 -> 0.0740 ms`，逐元素完全相同；128/452/784/1667/4096 行矩阵也全部
bit-exact。

候选 H2 NSYS 中，语言模型 kernel 数 `2672 -> 2636`，GPU busy
`128.417 -> 126.143 ms`（`-1.77%`）；完整 prefill kernel busy
`177.820 -> 175.653 ms`。36 个原生 SiLU 与 36 个对应 multiply 合并为 36 个
`_fused_swiglu_kernel`，同步与 memcpy 数不变。H1/H2 128-token 全流程分别保持原
SHA256 `76ad1f...14c6` 与 `4a61f1...166f`。H2 TTFT 仍有已知双峰，因此不使用
repeat7 中位数放大 claim；邻接候选/clean 的 min 和 max 均稳定前移约 `1.9 ms`，
与 kernel 归因一致。

主要证据：

- `data/p10_compile_graph/h2_external/h2_prism_compile_graph_repeat3_clean_7b322ef.jsonl`
- `data/p10_compile_graph/h2_external/h2_vllm_repeat3_7b322ef.jsonl`
- `data/p10_compile_graph/h2_profile/h2_semantic_targets_clean_96d7090_analysis.json`
- `data/p10_compile_graph/h2_vision_compile/vision_tensor_torch211_same_precision_clean_96d7090.json`
- `data/p10_compile_graph/h2_vision_compile/vision_rotary_torch211_same_precision_dirty.json`
- `data/p10_compile_graph/h2_vision_compile/h2_prefill_swiglu_{baseline_repeat7_clean_96d7090,repeat7_dirty}.jsonl`
- `data/p10_compile_graph/prefill_swiglu/h2_prefill_swiglu_language_dirty_analysis.json`

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

P9-D 的下一个实现目标是 **greedy full-step CUDA Graph**：steady-state decoder、
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
| model-only CUDA Graph | 当前强内部基线 | supported | 四个 formal cell 已完成；NSYS 归因中 |
| greedy full-step CUDA Graph | decoder + LM head + argmax | pending | 主候选 |
| pure compile subgraph | QKV/QK-Norm/M-RoPE | memory-safe、batch2分叉 | rejected evidence |
| compile + full-step Graph | batch1无状态热点 + 完整Graph replay | supported | P10 clean H1领先外部基线4.83%–6.37% |

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

1. **已完成**：在当前 GPU UUID 上建立 eager/model-only Graph clean baseline。
2. **进行中**：用 NSYS 分解 prefill、完整 decode step、Graph 外 CPU/GPU 工作和同步，
   优先归因 P9-009 的 scaled-FP8 batch1 engine TTFT 回退。
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
实际先经历 batch `1/2/3`，output128 才在后段形成 batch4 steady state。早期 sparse policy
曾把 actual batch3 replay 到 captured batch4，并触发 BF16 轨迹分叉；该失败 artifact 已
保留。commit `40466b6` 起 batch1–8 各自 exact capture，修复后正式 batch4 轨迹为
`1→1:2 / 2→2:2 / 3→3:2 / 4→4:124`，512 个完整 logits row bit-exact。batch9–15 等仍
使用 stride16 sparse bucket，不能外推 token-exact。因此正式 artifact 必须保存每个 actual
bucket 的 step count 和 actual→captured 映射，eager/Graph actual histogram必须 exact，
不能用 nominal batch4 或最后一次 replay代替。完整问题链记录为 P9-006/P9-008。

截至 2026-07-20，BF16/scaled-FP8 × batch1/4 四个 clean formal cells 均为 15/15
comparability PASS、token/text exact。model-only Graph 的 decode step 改善为
`37.07%–44.84%`，E2E 改善为 `27.17%–42.62%`；TTFT 单独判定，其中 scaled-FP8
batch1 engine TTFT 回退 `3.32%`、95% CI `[0.55%, 37.35%]`，已登记 P9-009，不能写成
Graph 改善 TTFT。artifact、SHA256 和完整 CI 见`docs/REPRODUCIBILITY.md`第11节。

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
