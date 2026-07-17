# Prism-Infer 项目路线图

> 修订日期: 2026-07-17
> 目标模型: Qwen3-VL-8B-Instruct
> 项目目标: 面向 Qwen3-VL 的跨层多模态推理 Runtime；把视觉 KV 保留、物理 Paged-KV 压缩、scaled FP8、Compiler/CUDA Graph、调度与优化 Kernel 接成可验证系统。

## 项目总目标

Prism-Infer 的交付目标不是单个 demo，而是一套可验证、可复现、可解释的多模态推理与 KV Cache 压缩研究工程:

- 跑通 Qwen3-VL-8B-Instruct 的自实现推理路径，核心模块不使用 HF/vLLM/SGLang wrapper 替代。
- 建立与独立参考实现的模块级、全模型级、端到端级数值对齐验证。
- 基于可靠 baseline 分析视觉 token 的 KV Cache 行为，再实现压缩策略。
- 用实测 benchmark 证明压缩率、质量退化、显存收益和吞吐/延迟变化。
- 输出可复现实验、技术报告、README 和面试/投递材料。

## 当前真实状态

| 领域 | 状态 | 说明 |
|---|---|---|
| 项目治理 | 已建立 | `CLAUDE.md` 和 `prism-infer-rigor` Codex plugin 已配置。 |
| Vision Encoder | 已实现，图文路径严格对齐 | `prism_infer/vision/vision_encoder.py` 已修复 P2-005；单图 processor 输入下 visual/layerwise/logits 已与 HF exact match。 |
| M-RoPE | 已实现，需持续回归 | `prism_infer/vision/mrope.py` 已存在，已有 M-RoPE 测试。 |
| Qwen3-VL Text Model | 纯文本 full logits 已严格对齐 | `prism_infer/models/qwen3_vl.py` 已存在，组件测试已建立。 |
| 模块对齐套件 | PASS | 2026-06-24 回归: P1 轻量 `10 passed in 74.68s`；P2 Gate + vision 回归 `24 passed in 48.49s`。后续改动必须重新跑。 |
| Full logits | PASS | 纯文本 `tests/test_full_model.py` max diff `0.000000e+00`, mean diff `0.000000e+00`；图文 `tests/test_full_model_vl.py` last logits max diff `0.000000e+00`, mean diff `0.000000e+00`。 |
| Engine 端到端 VL 推理 | 多模态 eager 与 CUDA Graph decode correctness 已完成 | P2 已跑通单图 `LLM.generate_vl` 1-token greedy HF exact match，并补齐单图图文 full logits/layerwise strict PASS；P3.1-P3.4 已补齐多图、视频、mixed batch、长输出稳定前缀和 logits/ppl 分布；P3.5 已补齐 VL CUDA Graph decode；P3.6 已接入自实现 Triton paged decode kernel。 |
| VL Engine 完整性 | 已完成当前 P3 门禁 | P3.0-P3.7 已完成；多图、视频、mixed batch、长输出稳定前缀、logits/ppl 分布、VL CUDA Graph decode、paged decode kernel 和 P1/P2/P3 回归均有验证记录。 |
| KV Cache 分析 | P4 已完成当前门禁 | 已实现 repo 内 KV trace schema/session、attention/KV 采集、entropy 指标、离线 summary 和三类样例报告；trace on/off greedy tokens 已验证一致。 |
| KV Engine Hardening | P4.5 已完成当前门禁 | 已统一 canonical 4D paged KV layout 写入语义，修复 CPU fallback、prefix hash 释放清理、swap CPU/GPU 页表混用，补齐 Sequence/Config block size contract，并把 prefix-cache prefill 提前显式拒绝。 |
| KV Cache 压缩 | P5 当前门禁已完成 | 已保留 `compression_mode="off"` baseline，新增 `visual_prune` logical decode retention，并完成 `fp8_kv` physical KV storage baseline；固定 16 blocks 下 FP8 KV cache bytes 为 BF16 的 `0.5x`，质量矩阵 32/32 token exact match；FP8 当前 latency 更慢，吞吐优化进入 P6。 |
| 系统性能优化 | P6.12-C BF16 content-aware 主线已完成并冻结 | 默认 last-layer attention scorer 在 7 张固定 COCO 图片、35 条 caption reference 上通过当前 lexical task gate：token-F1/ROUGE-L drop 为 `0.003288/0.003710`；7-image aggregate physical token/active-byte ratio 为 `0.535x/0.538x`。COCO batch4稳定性能 cell为 `0.536x/0.571x`。freeze tag 为 `p6.12-content-aware-kv`。这不是标准 COCO accuracy；FP8 quality和TP仍未完成，RTX counter缺口已在P9关闭。 |
| 单机性能与外部对标 | P7.0-P7.5 已完成 | online engine、external protocol、logits优化、Graph replay分析与 packed gate/up 均闭环；8 个 clean offline cell 的 packed decode TPOT 改善 `0.483%–0.762%`，不声称稳定 E2E 加速。 |
| 项目交付 | P8 PASS | README、技术报告、复现手册、Known Issues、投递材料、fresh editable install、完整8B demo与当前主线 full regression 均已验收。 |
| 秋招旗舰化 | P9-A 进行中 | 架构/性能 RFC 已冻结项目定位、双 headline gate、目标架构、量化与 kernel 路线；正式 page-size matrix、workload manifest 和证据同步仍在进行。 |
| 当前硬件 | 单卡主线可用；TP 条件阻断 | 8 张 RTX 5090 均可见且空闲，GPU0–3/4–7 分属两个 NUMA、无 NVLink；Prism Torch 2.6/NCCL 2.25.1 在 SM120 collective 失败，隔离 Torch 2.11/NCCL 2.28.9 同卡 all-reduce PASS。 |

## 阶段门禁总览

| 阶段 | 名称 | 目标 | 出口标准 |
|---|---|---|---|
| P0 | 治理与基线 | 固化工程流程、验证入口和当前真实状态 | 文档、插件、验证命令可用，当前风险清楚 |
| P1 | 模型地基严格对齐 | Vision/M-RoPE/Text/Full logits 对齐 | 纯文本 full logits 已 PASS；图文路径进入 P2 |
| P2 | Engine 单图端到端推理 | 从 `LLM` 层接收图文输入并生成 | 单图 greedy tokens、图文 last logits/layerwise 与 HF 一致，纯文本不回归 |
| P3 | VL Engine 完整性与性能基线 | 补齐真实多模态推理必须具备的输入、batch、长输出和 decode 性能路径 | 多图/视频/batch/长输出/CUDA Graph/paged decode 均有 correctness 验证和明确 benchmark 基线 |
| P4 | KV Cache 分析 | 捕获和量化 visual token KV/attention 行为 | trace 可复现，输出分析报告 |
| P4.5 | KV Engine Hardening | 在压缩前修复 KV layout、block manager、swap 和 prefix-cache prefill 语义债 | KV 子系统 invariant 有测试覆盖，P4/P3 窄回归不退化 |
| P5 | KV Cache 压缩策略 | 实现至少一个视觉 token 压缩策略 | 有压缩率、质量退化、显存/性能实测 |
| P6 | 系统优化与视觉 KV 物理压缩 | 统一 benchmark、profiling、CUDA Graph/compile、physical compaction、压缩 Paged Attention、两卡与外部对比 | correctness 不回归，质量/显存/性能/容量可复现，外部对比条件公平 |
| P7 | 单机性能与外部对标 | clean baseline、vLLM公平对比、online调度、Graph/compile/kernel profiling闭环 | 同条件差距可解释，online SLO和目标优化可复现 |
| P8 | 项目交付 | README、技术报告、复现实验和投递材料 | 外部用户能按文档复现核心结果 |
| P9 | 秋招旗舰化 | 架构硬化、scaled FP8、细页 KV、full-step Graph、优化 kernel、多模态 SLO | KV 质量–物理显存 Pareto 优于 vLLM 强基线，且一个预注册长视觉场景的 TPOT/SLO 指标胜出 |

每个阶段必须遵循:

```text
Plan -> Implement -> Verify -> Teach -> Document -> Gate Review
```

任一阶段的 PASS 声明必须绑定实际验证命令和输出。验证缺失时只能标注为未验证风险，不能写成完成。

## P0: 治理与基线

### 目标

把项目从临时开发状态整理成可持续推进的工程状态，保证后续每个模块都有清晰的入口、出口和验证标准。

### 小任务

- [x] 安装项目专用 Codex plugin: `prism-infer-rigor`。
- [x] 记录当前模型路径和 full logits 风险。
- [x] 将路线图改为阶段门禁结构。
- [x] 建立统一验证文档: `docs/VERIFICATION.md`。
- [x] 历史 `DAY_01.md/DAY_02.md` 已在 `311f055` 删除；`docs/HISTORY.md` 明确其只保留在 Git 历史，不再作为当前计划来源。
- [x] 建立 `docs/STAGE_DELIVERY_TEMPLATE.md`，统一目标、范围、环境、验证、claim、风险与下一步字段。

### 出口标准

- `docs/ROADMAP.md` 能回答“项目现在在哪个阶段、下一步做什么、怎样算完成”。
- `docs/VERIFICATION.md` 能回答“每阶段跑哪些命令、PASS 标准是什么”。
- 新任务开始前先读取 `CLAUDE.md`、`ROADMAP.md`、`VERIFICATION.md` 和 `git status`。

## P1: Qwen3-VL 模型地基严格对齐

### 目标

完成 Qwen3-VL 核心模型地基的自实现和数值对齐，包括 Vision Encoder、M-RoPE、Text Model、DeepStack 注入、权重加载和 full logits。

### 小任务

- [x] 重新跑语法和模块对齐基线，确认当前 `20 passed` 仍成立。
- [x] 固化 Vision Encoder 回归测试，覆盖 PatchEmbed、ViT MLP、ViT Attention、RoPE、完整 VisionEncoder。
- [x] 固化 M-RoPE 回归测试，覆盖 `[batch, seqlen]`、`[3, batch, seqlen]` 和兼容输入形态。
- [x] 固化 Qwen3-VL Text Model 组件测试，覆盖 RMSNorm、MLP、DecoderLayer、权重 key。
- [x] 修复 full logits `MARGINAL` 问题，不允许通过放宽阈值完成。
- [x] 建立分层误差定位工具，记录 layerwise hidden/logits max diff 与 mean diff。
- [x] 整理 P1-001 问题解决记录，解释 RoPE dtype、QK-Norm、attention 路径和验证结果。

### 出口标准

- `compileall` PASS。
- 模块对齐套件 PASS。
- full logits 达到严格门槛。
- 权重加载 missing/unexpected keys 为 0，或每个差异有解释。
- 文档中不得写“Qwen3-VL 全模型严格对齐完成”，除非 full logits 或 greedy tokens 已严格 PASS。

### 当前验证结果

- `compileall`: PASS。
- P1 模块对齐套件: `20 passed in 82.17s`。
- `tests/test_full_model.py`: PASS，logits max diff `0.000000e+00`, mean diff `0.000000e+00`。
- P1-001 已记录在 `docs/ISSUE_LOG.md`，状态为 `Verified`。
- 剩余风险: 以上 P1 PASS 是纯文本 full logits；图文输入、视觉 token 替换、DeepStack 注入和端到端 generate 已在 P2 单图 eager 范围验证。

### 主要验证

详见 `docs/VERIFICATION.md` 的 P1。

## P2: Engine 单图端到端推理

### 目标

把已对齐的模型接入 Prism-Infer engine，使系统能从用户侧接收图文输入，完成 Prefill + Decode，并保持纯文本路径不回归。

### 小任务

- [x] P2.0 设计门禁: 明确单图 VL 数据流、关键风险、验证标准和不做范围，见 `docs/P2_ENGINE_VL_DESIGN.md`。
- [x] P2.1 Processor pipeline: 建立 prompt + image 到 `input_ids` / `pixel_values` / `image_grid_thw` 的稳定入口，并说明 HF processor 作为非核心工具的使用理由。
- [x] P2.2 多模态 `Sequence`: 携带单图预处理结果、3D position ids / rope delta，并保证跨进程序列化不丢失必要字段。
- [x] P2.3 自实现 Qwen3-VL 3D position ids: 对齐 HF `get_rope_index` 的单图逻辑，输出 `[3, batch, seqlen]` position ids 和 rope delta。
- [x] P2.4 KV-aware Qwen3-VL attention + Prefill: 让 Qwen3-VL LLM attention 接入 engine KV cache，并把 VL payload 从 `ModelRunner.prepare_prefill` 传到模型 forward。
- [x] P2.5 Decode eager 对齐: decode 阶段不重复传图像，只用 last token、KV cache 和 rope delta 延续 position ids。
- [x] P2.6 Greedy sampler 和 `LLM.generate_vl`: 支持 deterministic greedy，用单图公开 API 对齐 HF tokens。
- [x] P2.7 P1/P2 回归和阶段 Review: 新增纯文本回归测试，更新问题记录和阶段状态。
- [x] P2.8 图文 full logits 与 layerwise strict 对齐: 修复 VisionEncoder RoPE buffer 与 PatchMerger eps，补充 `tests/test_full_model_vl.py`、`tests/test_full_model_vl_layerwise_debug.py` 和 `tests/test_vision_rope_init.py`。

### 出口标准

- 单图 prompt 能从 `LLM` 层跑通。
- greedy `temperature=0` 输出 tokens 与 HF 一致。
- 单图图文 last logits 与 HF 对齐，max diff `< 1e-2`；当前 strict 结果为 `0.000000e+00`。
- full-model layerwise debug 中 visual、embedding、M-RoPE、LLM layers、final norm、logits 均无非零差异。
- 纯文本 prompt 不回归。
- Qwen3-VL attention 必须在 engine prefill/decode 中正确写入和读取 KV cache；仅把图像字段传入模型 forward 不能算 P2 完成。
- P2 第一版当时以 `enforce_eager=True` 完成 correctness；P3.5 已补齐 VL CUDA Graph decode。
- 多图、视频、batch 混合图文不属于 P2 完成范围；这些能力已在 P3.1-P3.3 补齐。
- 若 P1 full logits 未 strict PASS，P2 不能宣称严格精度完成，只能作为功能 smoke。

### 当前验证结果

- P2 Gate + vision 回归测试: `24 passed in 48.49s`。
- `LLM.generate_vl` 单图 1-token greedy HF 对齐: HF token ids `[785]`，Prism token ids `[785]`。
- 图文 full logits: `tests/test_full_model_vl.py` PASS，HF/Prism shape `[1, 151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- 图文 full-model layerwise: `visual/embed/rope/layer_00...layer_35/final_norm/logits` max diff 均为 `0.000000e+00`。
- Vision RoPE 初始化回归: `tests/test_vision_rope_init.py` `2 passed in 8.71s`，`inv_freq/freq_table/rot_pos_emb` max diff 均为 `0.000000e+00`。
- Qwen3-VL engine attention prefill: output max diff `0.000000e+00`，KV cache max diff `0.000000e+00`。
- Qwen3-VL engine attention decode paged KV fallback: output max diff `0.000000e+00`。
- `ModelRunner.prepare_prefill`: 单图输入 `input_ids=[210]`、`position_ids=[3, 210]`、`pixel_values=[784, 1536]`、`image_grid_thw=[1, 3]`。
- `ModelRunner.prepare_decode`: decode 不传 `pixel_values/image_grid_thw`，使用 `rope_delta` 生成 `[3, 1]` position ids。
- 纯文本 engine greedy smoke: output token ids `[785]`。
- P1 轻量回归: `10 passed in 74.68s`。
- P1 full logits 回归: PASS，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- P2 当前状态: 单图、单请求、`enforce_eager=True` correctness 与 strict 图文对齐门禁已完成。
- P2 阶段外能力已在 P3 补齐多图、视频、mixed batch、长输出稳定前缀、logits/ppl 分布、VL CUDA Graph decode 和 paged decode kernel；P7.3 已补 chunked/paged prefill 与 online mixed-VL，VL token-id prefix hash因像素语义不安全而显式禁用，P3 decode kernel 仍只是 baseline kernel。

## P3: VL Engine 完整性与性能基线

### 目标

P2 只证明单图、单请求、`enforce_eager=True` 的 correctness。P3 要把 Prism-Infer 从“可以跑通单图 forward/generate”推进到“具备真实多模态推理框架的核心能力”，覆盖:

- 多图输入: 一条请求包含多张图片。
- 视频输入: 一条请求包含视频帧/视频 grid。
- batch 混合图文: 同一 prefill/decode batch 中混合 text-only、单图、多图、视频请求。
- 长输出多 token 质量评估: 不只验证 1-token greedy，而是验证多 token greedy、logits 分布和质量指标。
- VL CUDA Graph decode: 在 `enforce_eager=False` 下验证 VL decode graph capture/replay correctness。
- 高性能 paged decode kernel: 替换当前 correctness eager fallback，建立可对比的 paged decode kernel 和 benchmark。

### 当前阻断证据

| 能力 | 当前限制 | 证据 |
|---|---|---|
| 多图输入 | P3.1 已解除单请求多图 eager correctness 阻断；P3.3/P3.5 已补齐 mixed batch 与 CUDA Graph decode correctness。 | `prism_infer/engine/vl_inputs.py`, `prism_infer/vision/vision_encoder.py`, `tests/test_full_model_vl_multi_image.py`, `tests/test_llm_vl_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| 视频输入 | P3.2 已解除单请求 synthetic video eager correctness 阻断；P3.3/P3.5 已补齐 mixed batch 与 CUDA Graph decode correctness；真实视频文件读取/采样策略仍未覆盖。 | `prism_infer/engine/vl_inputs.py`, `prism_infer/models/qwen3_vl_position.py`, `prism_infer/models/qwen3_vl.py`, `tests/test_full_model_vl_video.py`, `tests/test_llm_vl_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| batch 混合图文 | P3.3/P3.5 已覆盖 non-prefix eager/Graph；P7.3 已覆盖 online arrival、mixed-VL与 chunked paged prefill。VL prefix hash显式禁用，避免相同占位 token错误复用不同像素 KV。 | `prism_infer/engine/model_runner.py`, `prism_infer/engine/online.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_online_serving.py` |
| 长输出质量 | P3.4 已覆盖单图/多图/视频 32-token 生成诊断、稳定前缀门禁、teacher-forced logits/ppl 分布和 mixed batch 长输出稳定性；text-only mixed 32-token 分叉已证明为 HF/Prism 共有 batch-size 数值敏感性，视频长输出第 6 token 分叉已定位为 bf16 tie-break。 | `tests/test_llm_vl_long_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_vl_logits_distribution.py`, `tests/test_batch_numeric_sensitivity.py` |
| VL CUDA Graph decode | P3.5 已解除阻断；`enforce_eager=False` 下 single-image/multi-image/video/mixed batch token ids 与 eager 对齐。 | `prism_infer/engine/model_runner.py`, `prism_infer/engine/llm_engine.py`, `tests/test_llm_vl_cuda_graph_decode.py`, `tests/test_model_runner_vl_cudagraph.py` |
| 高性能 paged decode | P3.6 已接入自实现 Triton paged decode kernel；保留 PyTorch SDPA eager reference。 | `prism_infer/ops/paged_decode.py`, `prism_infer/layers/attention.py`, `tests/test_paged_decode_kernel.py`, `benchmarks/bench_paged_decode.py` |

### 小任务

- [x] P3.0 设计门禁: 固化 VL 通用输入数据结构、batch 语义、position ids 语义、decode graph 约束、paged kernel 接口和 benchmark 输入集合。
- [x] P3.1 多图输入 correctness:
  - 将 `SingleImageInputs` 泛化为 image/video 可扩展的 `VLInputs` 或等价结构。
  - 支持一条请求包含多张图片，`image_grid_thw=[num_images,3]`。
  - 验证 processor 输出、image token 数、position_ids/rope_delta、full logits 和 `LLM` 入口 greedy tokens 均与 HF 对齐。
- [x] P3.2 视频输入 correctness:
  - 调查并实现 `video_grid_thw`、video token span、视频 position ids/rope_delta。
  - 支持最小可复现 synthetic video 或本地固定视频样例，不依赖网络下载。
  - 验证 processor、position ids、vision/video embeddings、full logits 和 1-token greedy。
- [x] P3.3 batch 混合图文 correctness:
  - 支持同一批次中混合 text-only、single-image、multi-image、video 请求。
  - 放开 `ModelRunner.prepare_prefill/prepare_decode` 中的单 VL sequence 限制。
  - 验证每条请求的输出与单请求独立运行一致，并验证 batch flatten attention/KV slot mapping 不串扰。
- [x] P3.4 长输出多 token 质量评估:
  - 建立 `max_tokens=32` greedy 生成诊断和稳定前缀 token 对齐。
  - 建立 logits 分布或 perplexity 检查，记录 shape、max diff、mean/std、PASS/FAIL。
  - 为非 greedy 采样模式记录可复现 seed、分布指标和质量样例，不把随机文本一致性作为 PASS。
- [x] P3.5 VL CUDA Graph decode:
  - 让 VL decode 的 3D position ids 和 `rope_delta` 能进入 graph replay。
  - `enforce_eager=False` 下验证单图、多图和 mixed batch decode 输出与 eager 完全一致或达到同精度阈值。
  - 建立 graph capture/replay 的 latency benchmark。
- [x] P3.6 高性能 paged decode kernel:
  - 保留当前 eager fallback 作为 correctness reference。
  - 实现或接入项目自有 paged decode kernel 接口，优先覆盖 Qwen3-VL decode 的 GQA、block table、context_lens。
  - correctness 先对齐 eager fallback，再做 latency/token/s benchmark。
- [x] P3.7 阶段 Review:
  - 跑 P1/P2/P3 回归，更新 `docs/VERIFICATION.md`、`docs/ISSUE_LOG.md` 和性能基线日志。
  - 未完成项必须列为风险，不能并入 “VL engine 完成” 声明。

### 出口标准

- 多图、视频、batch 混合图文均能从公开 `LLM` 入口运行。
- 每类输入至少有 processor/position ids/full logits/greedy generate 四层验证。
- 长输出多 token greedy 满足 `docs/VERIFICATION.md` 中定义的稳定前缀门禁；分布或 perplexity 指标达到对应门槛。
- `enforce_eager=False` 的 VL CUDA Graph decode correctness PASS，并有 eager vs graph benchmark。
- 高性能 paged decode kernel correctness PASS，并有 eager fallback vs kernel benchmark。
- P1/P2 单图和纯文本回归不退化。
- 所有性能数字必须来自同一 commit、同一硬件、同一输入条件的实测日志。

### 当前状态

- P3.0 设计门禁已完成，见 `docs/P3_VL_ENGINE_COMPLETENESS.md`。
- P3.1 多图输入 correctness 已完成:
  - `prepare_image_inputs` 支持单图或多图 list/tuple，`image_grid_thw=[num_images,3]`。
  - 多图 processor 输出、`position_ids/rope_delta` 与 HF exact match。
  - 多图图文 full logits strict PASS，HF/Prism shape `[1, 151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
  - `LLM.generate_vl` 多图 1-token greedy 与 HF token ids 完全一致，当前样例均为 `[785]`。
  - P2/P3.1 组合回归已有 `30 passed` 记录。
- P3.2 视频输入 correctness 已完成:
  - synthetic video processor 输出 `input_ids=[1,420]`、`pixel_values_videos=[1568,1536]`、`video_grid_thw=[[2,28,28]]`、video tokens `392 / 392`。
  - 视频 `position_ids/rope_delta` 与 HF exact match，max diff `0.000000e+00`。
  - 视频 full logits strict PASS，HF/Prism shape `[1,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
  - `LLM.generate_video` 1-token greedy 与 HF token ids 完全一致，当前样例均为 `[785]`。
- P3.3 batch 混合图文 correctness 已完成:
  - `ModelRunner.prepare_prefill` 支持 text-only + single-image + multi-image + video 同批，统一输出 `position_ids=[3,total_tokens]`。
  - mixed prefill 样例输出 `input_ids=[1043]`、`position_ids=[3,1043]`、`pixel_values=[2352,1536]`、`image_grid_thw=[3,3]`、`pixel_values_videos=[1568,1536]`、`video_grid_thw=[1,3]`。
  - mixed decode 样例输出 `input_ids=[3]`、`position_ids=[3,3]`、`context_lens=[6,211,421]`。
  - `LLM.generate_mixed` 在 text/single-image/multi-image/video batch 中 1-token greedy 与 fresh 单请求独立运行完全一致，token ids 为 `[[11], [785], [785], [785]]`。
- P3.4 长输出多 token 质量评估已完成:
  - 单图/多图 `max_tokens=32` 生成中 `prefix@8/16` 与 HF 一致；视频 `prefix@5` 与 HF 一致，首个分叉发生在第 6 token 的 bf16 tie-break。
  - 单图/多图/视频 teacher-forced logits shape `[1,32,151936]`，logits mean diff 约 `4.7e-3` 到 `5.3e-3`，ppl diff `< 0.01`，分布级 PASS。
  - mixed batch VL rows 长输出保持稳定前缀；multi-image/video 当前与 fresh 单请求独立运行一致，single-image 首个分叉在 token 28。
  - mixed batch text-only row 32-token 分叉来自 bf16 batch-size 数值敏感性；HF 与 Prism duplicate batch max diff 同量级，argmax 一致。
- P3.5 VL CUDA Graph decode 已完成:
  - `ModelRunner` graph replay 的 decode `position_ids` 统一规范为 `[3,batch]`。
  - `enforce_eager=False` 下 single-image/multi-image/video `max_tokens=2` 与 eager token ids 完全一致。
  - mixed batch=3 命中非标准 graph batch 档位，token ids 与 eager 完全一致: `[[11, 358], [785, 1378], [785, 2766]]`。
  - mixed VL graph benchmark: commit `45edd3a`，RTX 5090，`max_tokens=8`，warmup=2，repeat=5；eager decode median `31.5488ms`，graph decode median `16.4468ms`，correctness PASS。
- P3.6 高性能 paged decode kernel 已完成当前基线:
  - 新增自实现 Triton kernel `prism_infer/ops/paged_decode.py`，支持 Qwen3-VL GQA、paged KV cache、block table 和 context lens。
  - correctness 覆盖小 GQA shape 与 Qwen shape；Qwen shape `q=[2,8,128]` max diff `7.812500e-03`，mean diff `2.812790e-04`，PASS。
  - benchmark 覆盖 batch `1,2,4,8` 与 context `256,1024,4096`，12 个 case 全部 correctness PASS。
  - 性能风险: batch=1/context=4096 下 kernel median `0.2834ms` 慢于 reference `0.2314ms`；batch>=2 和多数场景 kernel 明显快于 reference。
- P3.7 阶段 Review 已完成:
  - `compileall prism_infer tests benchmarks`: PASS。
  - `git diff --check`: PASS。
  - P1/P2/P3 grouped regression: `49 passed in 356.34s`。
  - 纯文本 full logits: HF/Prism shape `[1,64,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
  - 单图 VL full logits: HF/Prism shape `[1,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
  - 多图 VL full logits: HF/Prism shape `[1,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
  - 视频 VL full logits: HF/Prism shape `[1,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- 在 P3.1-P3.4 完成前，不进入 KV Cache 压缩；否则压缩研究会建立在不完整的多模态 baseline 上。

## P4: KV Cache 分析

### 目标

先建立可观测能力，再讨论压缩。捕获每层 visual token 与 text token 的 attention/KV 行为，量化冗余与重要性。

### 小任务

- [x] 设计 trace 数据 schema，包含模型配置、输入 shape、图像网格、层号、head、token 区间和统计量。
- [x] 实现 attention/KV trace 开关，默认关闭，不影响正常推理。
- [x] 验证 trace on/off 输出一致。
- [x] 计算 visual token attention mass、attention entropy、token importance、层间冗余、head 差异。
- [x] 输出离线可视化脚本和样例报告。

### 出口标准

- trace 文件可复现生成。
- 开启 trace 不改变 greedy 输出。
- 至少覆盖单图描述、细节问答、多图或长上下文中的 3 类输入。
- 输出 `docs/KV_ANALYSIS_REPORT.md`，明确压缩假设。

### 当前状态

- 新增 `prism_infer/analysis/kv_trace.py`:
  - `kv_trace(...)` 上下文管理器显式开启 trace，默认关闭。
  - JSONL schema 包含 `trace_header` 和 `attention_layer` 两类 record。
  - `TokenSpan/SequenceTraceInfo/TraceMetadata` 记录 text/image/video span、grid、block table、shape 和 layer/head 元信息。
  - `summarize_trace` 输出 visual attention mass、attention entropy、visual/text K norm ratio、head 差异和相邻层冗余。
- `ModelRunner.prepare_prefill/prepare_decode` 在 trace 开启时构造 metadata；`Attention.forward` 在输出计算完成后只读采集，不参与输出计算。
- 新增离线分析脚本 `scripts/analyze_kv_trace.py`。
- 新增三类样例脚本 `scripts/run_kv_trace_samples.py`，覆盖:
  - `single_image_description`
  - `single_image_detail_qa`
  - `multi_image_comparison`
- 当前验证:
  - `compileall prism_infer tests scripts`: PASS。
  - P4 轻量测试: `4 passed in 1.46s`。
  - 三类真实样例: `result: PASS`，每类均为 `36` 层、`72` 条 layer records、`2` 个 step、phase 为 `decode/prefill`。
  - token ids:
    - single image description: `[32, 6303]`
    - single image detail QA: `[2518, 151645]`
    - multi image comparison: `[28715, 389]`
- 报告: `docs/KV_ANALYSIS_REPORT.md` 已记录样例结果和 P5 压缩假设。
- 原始 trace 输出位于 gitignored `data/kv_trace_samples/`，可由文档命令复现。

### 剩余风险

- Trace 是分析路径，会引入同步和 JSON 序列化开销，不能作为性能 benchmark 路径。
- 当前 P4 样例覆盖两类单图和一类多图；视频 trace 未作为 P4 最小门禁，但 P3 已验证视频 correctness。
- P4 只形成压缩假设，不代表 P5 压缩策略已经实现或有收益。

## P4.5: KV Engine Hardening

### 目标

在进入 KV Cache 压缩前，把 P4 诊断出的 KV 子系统结构债修到可验证状态。P4.5 不实现压缩算法，也不宣称 prefix-cache prefill 可用；它的目标是让后续 P5 压缩不会建立在含混的 layout、block table 或 hash 生命周期上。

### 小任务

- [x] 固化 canonical KV cache layout contract:
  - KV cache 统一按 `[num_blocks, block_size, num_kv_heads, head_dim]` 理解。
  - `slot_mapping` 统一表示 flat slot: `slot = block_id * block_size + block_offset`。
  - CPU eager fallback 与 GPU Triton store 使用同一 slot 语义。
- [x] 修复 `store_kvcache` CPU fallback:
  - 4D paged cache 下不再把 `slot` 误当第一维 block index。
  - 保留 legacy `[slots, heads, dim]` fallback 只用于小形态单测。
- [x] 修复 `BlockManager` hash/free-list invariant:
  - block 释放或重新分配前清理仍指向该 block 的 `hash_to_block_id`。
  - 增加 `free_block_id_set`，避免 `_allocate_block` 依赖 `deque.remove` 线性删除。
  - 释放 sequence 时同步清理 `cpu_block_table`。
- [x] 拆分 swap CPU/GPU block table 语义:
  - `seq.block_table` 只表示 GPU block id。
  - `seq.cpu_block_table` 只表示 swap 后的 CPU block id。
  - `swap_out()` 清空 GPU table 并写入 CPU table；`swap_in()` 反向恢复。
  - `ModelRunner.prepare_*` 拒绝仍在 CPU swap 状态的 sequence。
- [x] prefix-cache prefill early gate:
  - 当前不实现 paged prefill attention。
  - 一旦 prefill 中出现 `num_cached_tokens > 0` 导致 `cu_seqlens_k > cu_seqlens_q`，在 `ModelRunner.prepare_prefill` 阶段显式报错，避免拖到 attention 层或产生 silent fallback。
- [x] Sequence/Config block size contract:
  - `Config.kvcache_block_size` 在 engine 初始化时同步到 `Sequence.block_size`。
  - `BlockManager` 在 allocate/decode/swap/CoW 入口检查 Sequence 与物理 block size 一致，避免静默页表错配。
- [x] 建立 focused 回归测试:
  - `tests/test_kv_engine_hardening.py`
  - `tests/test_scheduler_swap_tables.py`

### 出口标准

- 4D paged KV cache 写入测试 PASS，输出 shape、slot_mapping、max diff。
- BlockManager hash cleanup 测试 PASS，释放后 `hash_to_block_id` 不指向 free block。
- swap table split 测试 PASS，CPU block id 不再污染 GPU `block_table`。
- scheduler swap-in 容量判断使用 `cpu_block_table`，测试 PASS。
- prefix-cache prefill 未实现路径必须 early fail，不能 silent fallback。
- `compileall prism_infer tests` PASS。
- 受影响窄回归 PASS: sequence、attention KV、model runner VL prefill/decode、KV trace、paged decode kernel。

### 当前状态

- P4.5 focused tests: `5 passed`。
- P4.5 + paged decode + engine attention narrow regression: `10 passed`。
- sequence / model_runner / trace narrow regression: `12 passed`。
- `compileall prism_infer tests`: PASS。
- 2026-07-04 focused environment refresh:
  - `.venv-local` created with system site packages plus `transformers` and `xxhash`.
  - trace/paged kernel focused regression: `7 passed`.
  - KV hardening/scheduler focused regression: `6 passed`.
- 2026-07-05 full local model regression:
  - 本地 Qwen3-VL 权重已下载并校验，pytest 收集用例全量通过: `84 passed, 5 skipped in 250.07s`。
  - 5 个 skipped 均为 manual GPU debug script，不是环境缺失或功能失败。
  - `compileall prism_infer tests`: PASS。
  - `git diff --check`: PASS。
- 2026-07-08 外部评估对照修复:
  - 修复 decode 序列化丢失 sampling params。
  - `swap_in` 不再依赖 decode 反序列化对象保留完整 `token_ids`；改用 `swap_out` 保存的 CPU block hash metadata 恢复 prefix-cache index。
  - `ModelRunner.run()` 异常路径使用 `finally` 清理 `Context`，并恢复 chunked prefill 临时截断状态。
  - `Scheduler.schedule()` 空 decode 分支改为显式 `RuntimeError`，不依赖 `assert`。
  - 补充 engine-style flatten VL DeepStack 注入轻量回归，确认 B1 风险在当前代码路径下未复现。
  - focused verification: `17 passed`, `5 passed`, `11 passed`；详见 `docs/EXTERNAL_REVIEW_2026_07_08.md`。

### 设计决策

- 选择 canonical 4D paged cache，而不是改成全局 2D cache。理由是 `ModelRunner.allocate_kv_cache`、paged decode kernel 和现有 P3/P4 GPU 路径都已经以 `[num_blocks, block_size, heads, dim]` 为真实物理布局；只需要把 flat slot 到 4D index 的解释集中修正。
- 选择释放 free block 时清理 `hash_to_block_id`，而不是继续允许 free block 作为可命中的 prefix cache。理由是当前 BlockManager 没有独立 cached/reserved 状态；让 free list 同时承担可复用 prefix cache 会导致 hash 指向已释放 block 的生命周期不清晰。后续若要持久 prefix cache，应新增独立状态，而不是复用 free block。
- 选择新增 `cpu_block_table`，而不是让 `block_table` 根据 sequence status 改变含义。理由是 `prepare_decode`、trace、paged decode kernel 都默认 `block_table` 是 GPU block id；字段语义稳定比节省一个 list 更重要。
- 选择 early gate prefix-cache prefill，而不是在 P4.5 中实现 paged prefill kernel。理由是 P4.5 的目标是 hardening，不扩大到新的 attention kernel；paged prefill 属于后续 P6/P5 交叉的大任务。

### 剩余风险

- `kvcache_block_size=256` 仍然是 P5 active compression 粒度风险。P5.0 已决定保留当前物理 page 并先接入逻辑 compression metadata；真正 sub-page pruning/compaction 仍是 P5.2+ 风险。
- prefix-cache prefill 仍未实现；当前只是把失败前移并显式化。
- swap 数据搬运仍有全局 `torch.cuda.synchronize()`，这是 P6 性能优化项，不属于 P4.5 correctness hardening。
- paged decode Triton kernel 的 `MAX_CONTEXT_LEN` 冗余迭代和 `BLOCK_N=32` 固定调优仍是 P6 kernel 优化项。
- VScan/PoRe、M-RoPE physical compaction 和竞品性能对比仍未实现或未同条件验证，不能作为当前项目 claim；FP8 KV 已在 P5.3/P5.4 作为本项目 baseline 落地。

## P5: KV Cache 压缩策略

### 目标

在可靠 FP baseline 和 P4.5 稳定 KV 语义上实现视觉 token KV Cache 压缩，并用实测数据说明收益与退化。

### 小任务

- [x] 设计 compression config，P5.0 支持 `off` baseline，active 策略显式拒绝。
- [x] P5.0 压缩粒度设计门禁: 保持当前 256-token 物理 page，先接入逻辑 compression metadata；物理 compaction 延后到 pruning correctness 后。
- [x] 建立 compression off 等价 baseline。
- [x] 实现 P5.1 token-level importance scoring 离线分析。
- [x] P5.2 preflight: 建立 visual-token pruning decision helper 和 focused tests，不接入 runtime compression。
- [x] P5.2-A runtime shadow mode: 在 `CompressionMetadata` 中记录 prefill visual pruning decision，不改变 KV。
- [x] P5.2 实现首个可回退 logical visual token pruning/retention 策略，先证明 correctness 和小样例质量退化。
- [x] P5.3 评估 physical KV compaction 或 FP8 KV baseline，不能早于 P5.2 correctness。
- [x] 保证未实现 compression mode 显式报错，不 silent fallback。
- [x] 扩展评估压缩率、显存、延迟、吞吐、token 一致率和质量退化，覆盖多样例并证明至少一个可测收益。

### 出口标准

- compression off 与 FP baseline 完全一致。
- compression on 有明确压缩率和质量退化数字。
- 至少一个策略在指定场景下显示可测收益。
- 输出 `docs/COMPRESSION_REPORT.md` 和原始 benchmark 日志。

### 当前状态

- P5.0 已新增 `prism_infer.engine.compression`:
  - `compression_mode="off"` 是 FP baseline。
  - `compression_mode="visual_prune"` 是 logical visual token retention mode。
  - `compression_mode="fp8_kv"` 是 physical KV storage baseline；其他未实现模式仍显式失败。
  - `CompressionMetadata` 记录 step phase、batch size、prompt tokens、image/video visual token counts 和 block size。
- `ModelRunner.prepare_prefill/prepare_decode` 每步构造 compression metadata 并写入 `Context`。
- `Attention.forward` 在 off 模式不改变 K/V 写入、paged decode 或 SDPA/FlashAttention 路径。
- P5.0 只声明 no-op baseline；不声明压缩率、显存收益或质量收益。
- P5.1 已新增离线 visual-token importance scoring:
  - `prism_infer.analysis.visual_importance` 从 P4 trace records 计算 visual token/span importance proxy。
  - `scripts/score_visual_tokens.py` 可读取 trace JSONL 并输出 JSON/Markdown 报告。
  - scoring 组合 attention mass、visual entropy focus 和弱 K norm ratio；`top_visual_tokens` 只用于在 span 内分配已记录 top-k token mass。
  - P5.1 不修改 runtime KV cache，不实现 pruning，不声明压缩率、显存收益或质量收益。
- P5.2 preflight 已新增 `prism_infer.engine.visual_pruning`:
  - 支持 image/video 多 visual span 扫描、uniform/score retention decision 和可审计 decision record。
  - P5.2-A 已通过 `enable_visual_pruning_shadow` 把 prefill decision record 写入 `CompressionMetadata`。
  - shadow metadata 不改变 `compression_mode="off"` 的 no-op 语义；decode metadata 不重算 pruning decision；attention 输出保持 exact no-op。
  - P5.2 active logical pruning 已把 prefill pruning decision 持久化到 `Sequence`，decode 阶段使用 retained-token KV view。
  - active decode 强制走 retained-aware eager path，不使用当前连续 context 的 Triton paged decode，也不走 CUDA Graph replay。
  - keep-all active path 与 off decode focused test exact match；单图 `max_tokens=2` keep-all 真实模型 smoke token ids 均为 `[785, 2168]`。
  - 单图 `keep_ratio=0.5` smoke: visual tokens `196 -> 98`，8-token greedy 与 off 完全一致；这只是小样例质量信号，不代表完整质量评测。
  - 单图小 benchmark: off median `0.292072s`，`visual_prune keep_ratio=0.5` median `0.798550s`；active 当前更慢，不能声明吞吐收益。
  - 当前仍不实现 physical compaction，实验性 prefill slot mask helper 不能用于声明物理 KV 压缩完成。
- P5.3/P5.4 已完成 `fp8_kv` baseline:
  - 同样 `num_kvcache_blocks=16` 时，BF16 KV cache bytes `603979776`，FP8 KV cache bytes `301989888`，ratio `0.5`。
  - `tests/test_fp8_kv_cache.py` 验证 FP8 store round-trip 和 decode dequant reference，focused max diff `0.000000e+00`。
  - 真实模型质量矩阵覆盖 text/single-image/multi-image/video，`fp8_kv` 对 off 为 `32/32` token exact match。
  - 单图 benchmark: off median `0.278173s`，`fp8_kv` median `0.704317s`；FP8 有显存/KV 容量收益，但当前吞吐更慢。
  - 原始日志位于 `data/p5_compression/fp8_kv_single_image_benchmark_20260709.jsonl` 和 `data/p5_compression/fp8_kv_quality_matrix_20260709.jsonl`。
- 外部评估中提到的 VScan+PoRe、DeepStack-aware pruning 和 M-RoPE block compaction 当前只作为候选路线；在源码、测试和 benchmark 落地前不得写成完成能力。

## P6: 系统优化与视觉 KV 物理压缩

### 目标

在 P1-P5 correctness baseline 上，建立可归因的系统性能基线，把 P5 logical visual pruning 推进为真实 physical KV compaction，并评估 CUDA Graph、`torch.compile`、压缩 Paged Attention、两卡 TP 和固定版本外部框架。

P6 不把“全面超过 vLLM/SGLang”设为完成条件。项目目标是在视觉 token 占比高、长上下文或显存受限的明确场景下形成可复现优势，同时如实报告不占优场景。完整架构、实验矩阵和设计决策见 `docs/P6_SYSTEM_OPTIMIZATION_DESIGN.md`。

### 小任务

- [x] P6.0 设计门禁: 固定阶段范围、目标架构、baseline hierarchy、benchmark 协议、physical KV layout contract 和 megakernel 启动条件。
- [x] P6.1 统一 benchmark contract:
  - 定义可版本化 JSONL result schema 和 deterministic workload manifest。
  - 新增 Prism internal benchmark runner，先覆盖 off/eager、off/CUDA Graph、`visual_prune` 和 `fp8_kv`。
  - 分离 preprocessing、TTFT、decode/TPOT 和 end-to-end timing。
  - 建立 `docs/PERFORMANCE_REPORT.md` baseline section。
- [ ] P6.2 分层 profiling:
  - 记录 processor、vision、prefill、decode、sample 和 scheduler 时间。
  - 定位 CUDA Graph launch 收益、visual-prune gather 和 FP8 dequant 成本。
  - 只有已测瓶颈才能进入 kernel/compile 优化。
  - [x] P6.2-A: 默认关闭的 CPU/CUDA semantic collector、四模式 RTX 5090 profile、NVTX/Nsight capture 和 structured SQLite analyzer 已完成。
  - [x] P6.2-A: `visual_prune` gather 与 FP8 eager KV store 的同步/launch 病理已定位；batch=1/context=4096 paged decode 已完成 correctness/benchmark。
  - [ ] P6.2-B: RTX 5090 Nsight GPU metrics 当前返回 `Already under profiling`，真实 SM utilization 尚未采集；不得把 kernel busy time 写成 GPU utilization。
  - [x] P6.2-C: visual retained indices 已在每个 decode step 映射为跨 36 层复用的 physical-slot tensor；gather target async memcpy/stream sync 从 `24696/24696` 降为 `0/0`，32/32 quality-matrix tokens exact。
  - [x] P6.2-D: CUDA FP8 KV store 已复用自实现 Triton vectorized kernel；store target kernel/async memcpy/stream sync 从 `15624/23436/7812` 降为 `288/0/0`，FP8 对 eager reference exact，质量矩阵 32/32 token exact。
- [x] P6.3 执行后端门禁:
  - 在统一 harness 中刷新 eager 与 CUDA Graph 对比。
  - 调查 Vision Encoder、LLM prefill、LLM decode 的 `torch.compile` 边界。
  - 记录 graph break、recompile、compile time、steady-state latency 和 correctness。
  - [x] P6.3-A: benchmark schema v2 已记录 graph scope/capture/buckets，完成 batch `1/2/4/8` × output `8/32/128` eager/Graph matrix；12/12 cells token exact，Graph decode speedup `1.68x..1.79x`。
  - [x] P6.3-B: 已完成 decoder layer、language-model decode 和 vision候选 region 的 `torch._dynamo.explain`/compile preflight；完整 decode 在 32GB 上 cold compile OOM，Vision/full-layer 编译数值不通过，attention-only 虽比 eager 快但 batch2/8 长输出 token 不 exact，因此不进入支持后端。
- [x] P6.4 Visual KV physical compaction:
  - 分离 logical context length 与 physical KV length。
  - 实现 post-prefill compact、block 回收、compressed block table 和 decode append。
  - 验证 Sequence pickle、swap、CoW、mixed batch、M-RoPE 和 keep-all exact equality。
  - [x] schema v4 记录 logical/physical prompt tokens、active/dense blocks 与 occupied bytes；multi-image keep=0.5 实测 `408 -> 212` physical tokens、`2 -> 1` active blocks、occupied bytes `0.5x`。
  - [x] focused regression `64 passed`；keep-all 8-token exact，mixed text/image/video 2-token exact；multi-image keep=0.5 前 6 token exact、第 7 token 分叉，质量边界进入 P6.6。
- [x] P6.5 Compressed/FP8 paged attention:
  - compact pages 先复用或扩展现有 paged decode kernel。
  - 实现 FP8-aware load/dequant correctness，避免整段 cache eager dequant。
  - 对 Qwen GQA、batch/context matrix 做 correctness 和 benchmark。
  - [x] BF16/FP8 Qwen GQA batch `1/2/4/8` × context `128/256/1024/4096` 共 32 cases 全部对 independent SDPA reference PASS，max diff `<=0.00390625`。
  - [x] FP8 engine decode 已切换到 kernel 内 load/dequant；single-image output32 中 decode median `32.065 -> 31.960 ms`、32-token exact、KV bytes `0.5x`。
- [x] P6.6 质量、容量与 Pareto 评估:
  - keep ratio 覆盖 `0.25/0.5/0.75/1.0`。
  - 覆盖 text、single-image、multi-image、video 和 mixed batch。
  - 输出 quality-memory-TPOT Pareto、固定 32GB max concurrency 和 OOM boundary。
  - [x] 新增 `visual_compact_fp8` 和 FP8-safe Triton physical compaction；组合模式实测 active prompt bytes 最低为 off 的 `0.25x`。
  - [x] 四类 synthetic VL workload 完成 5 modes × 4 ratios × output `8/32/128` 矩阵；固定 COCO 图片完成 4 physical modes × 4 ratios × output32。
  - [x] 600-request multi-image auto-pool 容量实验观察到 peak running `124/248/249/498`，组合模式为 off 的 `4.016x`；四模式均完成 600 请求且无 swap。
  - [x] Pareto 汇总器从 schema-validated JSONL 自动生成 physical tokens、blocks、active bytes ratio、per-request stable prefix 和 TPOT ratio。
  - [x] 评估结论为质量门禁 FAIL：uniform pruning 在代表性 output128 workload 上发生早期 token 分叉；单个真实样例也不能支持 accuracy drop `<1%`，不降低阈值掩盖失败。
- [x] P6.7 外部框架公平对比:
  - 固定 vLLM/SGLang/vLLM-Omni repo、commit、backend 和启动配置。
  - Prism off 与 compression-on 同时参与。
  - 不要求全面胜出；必须报告优势区间、劣势区间和原因。
  - [x] vLLM `0.24.0` build commit `ee0da84ab` 与 SGLang `v0.5.15` commit `f63458b5` 已固定；image/multi-image/COCO 使用相同 prompt token 数、BF16、eager、4096-token KV capacity、prefix/MM cache off。
  - [x] Prism off 与 `visual_compact_fp8 keep=0.5` 均进入自动汇总；外部 TPOT 相对 Prism eager 为约 `0.41x-0.49x`，Prism 当前没有端到端吞吐优势。
  - [x] video/mixed 因 vLLM Qwen3-VL timestamp placeholder 产生 `422/420`、`638/636` prompt token 差异，自动标为不可比较。
  - [x] vLLM-Omni repo 固定在 clean `73bafd64`，其标准 Qwen3-VL autoregressive 执行来自依赖的 vLLM `0.24.0`；不重复包装同一路径制造第三组独立数字。
- [ ] P6.8 两卡 TP 验证:
  - 验证 1 GPU vs 2 GPU greedy/logits correctness、权重/KV heads shard、NCCL collective、显存和 latency。
  - [x] 静态 shard/collective 审计：vocab/column/QKV/row 分片和 embedding/row all-reduce、LM-head gather 已存在；Vision Encoder 当前每 rank 完整复制。
  - [x] 新增 visible GPU 与 Q/KV heads、hidden/intermediate/vocab 可整除 preflight；单卡请求 TP2 现在启动前显式失败。
  - [x] fixed 1 MiB shared-memory control payload 已替换为每 worker 单向 variable-size Pipe；4,817,396-byte 视觉 payload 向两个接收端广播并逐元素校验通过，损坏/序列化/断连错误显式失败。
  - [x] 新增显式启用的 `tests/test_llm_vl_tp2.py`，固定 TP1/TP2 单图 8-token greedy exact smoke 入口。
  - [ ] 两卡 correctness/performance：当前仅一张 RTX 5090；IPC 实现阻断已解除，但 greedy/logits/NCCL/per-GPU memory/latency 仍待两卡平台实测。
- [x] P6.9 Megakernel 可选实验门禁审查（不启动）:
  - 仅当 P6.2 证明目标 decode workload 为 launch-bound 且存在真实可运行实现时启动。
  - 与 discrete eager、CUDA Graph 和 compile 做同条件对比；不阻塞 P6 核心门禁。
  - [x] 当前无 NCU SM utilization/hardware counter，仓库内也无真实 persistent/megakernel；启动条件不满足。
  - [x] P6.3 已有 CUDA Graph 强基线，batch1-8 decode speedup `1.68x-1.79x`；不得把普通 paged kernel、compile fusion 或 CUDA Graph改名为 megakernel。
- [x] P6.10 可执行阶段 Review:
  - 跑 P1-P5 full regression，完成 `docs/PERFORMANCE_REPORT.md` 和原始 JSONL 复现记录。
  - [x] 最终 full suite：`195 passed, 5 skipped in 245.50s`；首次 `184/11` 与二次 `193/2` 失败均保留日志并完成根因修复。
  - [x] variable-size TP IPC 后全量复跑：`197 passed, 6 skipped in 267.13s`；新增 skip 为显式启用的两卡 integration test。
  - [x] `compileall` 与 `git diff --check` PASS。
  - [x] 性能、容量、质量失败、external 劣势、backend 限制和 TP 阻断均已写入报告。
  - [x] P6.11 已在 clean commit `9e30e55` 重跑关键 performance matrix；更早 P6.1-P6.10 performance records 仍是各自记录的 dirty validation evidence。
- [x] P6.11 Compressed KV CUDA Graph:
  - 为 `off/fp8_kv/visual_compact/visual_compact_fp8` 建立 Graph-safe mode contract；physical KV dtype/layout 由 capture 时绑定的 cache 和 replay 前更新的 `slot_mapping/context_lens/block_tables` 表达。
  - `visual_prune` 的 retained-slot gather 仍依赖动态 metadata，因此 `enforce_eager=False` 在 `Config` 构造阶段显式报错，禁止静默退回 eager。
  - 统一 runner 新增三组 physical compression eager/Graph 配对模式，确保执行后端对比时 compression 与 attention backend 不变。
  - clean commit `9e30e55` single-image output32、warmup/repeat `2/5`：compact、FP8、combo decode speedup分别为 `1.8364x/1.8371x/1.8535x`，每组 eager/Graph token SHA256、physical tokens 和 active bytes exact。
  - clean combo batch `1/2/4/8`、output32：decode speedup `1.9428x/1.8937x/1.8354x/1.7572x`，Graph decode throughput `57.11/112.62/218.47/417.32 tok/s`；这是 replicated-request offline decode，不是 online serving 结果。
  - single/multi-image、video、mixed batch=3 bucket4 smoke 与 combo output128 均 eager/Graph token exact；full regression `208 passed, 6 skipped in 267.79s`。
  - P6.11 已保留 commit `ac6e01d` dirty validation，并新增 commit `9e30e55`、`git_dirty=false` formal matrix；这不改变 P6.6 uniform pruning quality FAIL，也不构成超过 vLLM/SGLang 的新对比。
- [x] P6.12 Content-aware visual pruning:
  - [x] P6.12-A runtime scorer：在最后 `N` 个 decoder layers 聚合 prefill 最后 query 对 visual tokens 的 attention mass；device tensor 延迟到 prefill 结束后一次 materialize，TP 下通过 all-reduce 合并各 rank 的 local Q heads。
  - [x] prefill decision 两阶段化：`attention` strategy 在 forward 前只创建 scorer，forward 后生成可审计 decision，再复用 P6.4 physical compaction 和 P6.11 CUDA Graph decode。
  - [x] decision record 写入 `score_source/score_layers/score_min/max/mean`；缺层、重复层、GQA shape、token count 和 score 完整性均显式失败。
  - [x] 单元 independent reference max diff `0`；focused `79 passed`；single-image BF16/FP8 eager/Graph、mixed text/image/video Graph smoke PASS。
  - [x] keep=0.5 quality preflight：COCO uniform/attention stable prefix `3/21`，multi-image `6/7`，video `14/14`；physical prompt tokens 分别保持 `166/212/226`，没有牺牲压缩率。
  - [x] coverage-aware Python MMR ablation 被拒绝：COCO/multi/video prefix 变为 `7/6/14`，且 prefill 增至约 `236-390 ms`；候选代码已删除，不进入支持策略。
  - [x] post-change full regression：`212 passed, 6 skipped in 299.59s`。
  - [x] commit `c07fa34` clean quality rerun：COCO/multi-image/video 9 条 records 均 `git_dirty=false`，stable-prefix 与 physical-token 结论复现。
  - [x] P6.12-B per-span 审计：decision record 新增 `kept_visual_tokens_by_span`；clean global attention 在 multi-image/video 两个 196-token span 中分别保留 `124/72`、`109/87`。
  - [x] 等额 per-span quota ablation 被拒绝：总 keep 和 physical prompt 不变，双 span 均变为 `98/98`，但 COCO/multi/video stable prefix 仍为 `21/7/14`，没有超过 global attention；`attention_span` 候选代码已删除。
  - [x] dataset fidelity/task 工具：`pruning_fidelity.py` 和 CLI 对 schema-v4+ physical KV 保持兼容；schema-v5 额外校验 decoded text/hash、reference provenance、candidate 全 case coverage、output-length cell 与 span audit。旧 schema-v4 记录继续可汇总，但 task gate 显式 `INELIGIBLE`。
  - [x] 固定真实集从 1 张 COCO 扩展到 7 张，按 `4+3` 组成两个 mixed-request batch；每图绑定 5 条 COCO val2017 caption，共 35 条 reference。官方 annotation package、固定 mirror revision、annotation SHA256、图片 URL/SHA256/尺寸和下载后 materialization 均可审计。
  - [x] 7-image dirty quality preflight：attention/uniform exact request rate `0.429/0`，micro prefix ratio `0.696/0.304`，minimum prefix ratio `0.219/0.094`，physical token ratio同为 `0.535x`、active bytes 同为 `0.538x`。
  - [x] dataset-level reference task-quality harness：多参考 token-F1 与 token-level ROUGE-L 分别取 best caption；两项 macro score 相对 off baseline 的绝对下降都必须 `<=0.01`。attention 的 token-F1 `0.321635 -> 0.315285`（drop `0.006351`）通过，ROUGE-L `0.289116 -> 0.276703`（drop `0.012413`）失败；uniform 的 ROUGE-L drop 为 `0.036365`。
  - [x] P6.12-C layer aggregation ablation：固定 last-query scorer 和 keep=0.5，仅比较 last1/last4/last8；batch A 中 last1 两项 task drop 均为 `0.008275` 并通过，last4/last8 均为 `0.012899` 并失败。
  - [x] P6.12-C 首个质量合格策略：clean commit `a7588d3` 的同条件 off/last1 成对复跑中，7-image token-F1 `0.321635 -> 0.318347`（drop `0.003288`）、ROUGE-L `0.289116 -> 0.285406`（drop `0.003710`），两项均通过 `<=0.01` 门禁；physical token/active-byte ratio 保持 `0.535x/0.538x`。
  - [x] P6.12-C 收尾：commit `e51c16d` 上未传 last-N 参数的 attention 路径继续记录 `last1/[35]` 并复现 task gate PASS；multi-image/video output128 stable prefix 为 `7/14`，mixed text/image/video output32 为 `[32,28,14]`，无 visual span starvation。
  - [x] P6.12-C 稳定性能与回归：COCO batch4/output32、`warmup=2/repeat=5` 中 last1 相对 off 的 prefill ratio `1.010x`、decode-step speedup `1.021x`、engine output throughput `1.013x`、E2E speedup `1.005x`，physical tokens `988 -> 530`、active bytes ratio `0.571x`；last1/last4 性能差异小于 `0.2%`，不声称 scorer 加速。full regression `238 passed, 6 skipped in 232.90s`。

### 出口标准

- 统一 benchmark 能复现 Prism off/eager baseline，包含 TTFT、TPOT、latency、throughput、memory、KV bytes、capacity 和输入参数。
- 至少一个 physical visual KV mode 真实减少 block/KV bytes，并提供质量、显存、性能和容量数据。
- CUDA Graph 与 `torch.compile` 结论来自同条件 internal ablation，不能混入压缩算法变化。
- 至少一个自定义 kernel 完成 independent reference correctness 和 benchmark 优化闭环。
- 完成两卡 TP correctness 和基础性能验证。
- 完成固定版本的 vLLM/SGLang/vLLM-Omni 公平对比；不要求所有 workload 胜出。
- P1-P5 full regression 不退化，输出 `docs/PERFORMANCE_REPORT.md`。
- megakernel、PD 分离和投机解码不属于 P6 核心出口标准。

### 当前状态

- P6.0 设计门禁已完成，见 `docs/P6_SYSTEM_OPTIMIZATION_DESIGN.md`。
- P6.1 已实现 `benchmark_schema.py`、internal runner、五类 deterministic workload 和 focused tests；RTX 5090 四模式 runner validation 见 `docs/PERFORMANCE_REPORT.md`。
- 当前 P6.1 数据记录为 `git_dirty=true`，只能作为 runner validation baseline；提交后仍需 clean-commit formal rerun。
- P6.2-A 已完成 semantic CPU/CUDA region、Nsight kernel/API/synchronization capture 和 context=4096 paged kernel 验证；已定位 `visual_prune` retained gather 与 FP8 eager store 的主要成本，证据见 `docs/PERFORMANCE_REPORT.md`。
- P6.2-C 已完成 visual-prune tensorized slot gather：同轮 single-image decode median 为 visual `33.529 ms`、off eager `30.834 ms`；logical pruning 仍不减少 KV bytes，剩余差距进入 retained-aware paged kernel。
- P6.2-D 已完成 FP8 vectorized KV store：semantic FP8 prefill 36 层 store CUDA 合计从 `373.769 ms` 降至 `0.606 ms`；Nsight target 不再有 async memcpy/stream sync。FP8 single-image decode median 仍为 `35.865 ms`，相对同轮 off `31.077 ms` 为 `1.154x`，瓶颈已转移到 gather/dequant/paged attention。
- P6.3-A 已完成 execution matrix：schema v2 向后兼容 v1，并记录 prefill/decode backend、Graph capture scope/time/buckets/selected batch；RTX 5090 上 12 个 eager/Graph cell 全部 token exact。batch8/output32 Nsight 显示 eager decode `2077` 个显式 kernels，Graph 路径为 `13` 个 graph 外 kernels + `1` 次 graph launch，Graph execution median `14.818 ms`。
- P6.3-B 已完成：benchmark schema 升为 v3 并继续兼容 v1/v2；默认关闭的 semantic profiler 不再切碎 Dynamo graph。decoder/full decode 均可形成单 graph，但 full decode 在 batch1/4 cold compile 均 OOM；完整 VisionEncoder 的动态 geometry 为 `7 graphs/6 breaks`，拆分 tensor region 后为 `1 graph/0 break`，但 27 层输出 max diff 仍为 `0.515625`。attention-only system mode decode 比 eager 快约 `1.43x..1.46x`，仍慢于 CUDA Graph，且 batch2/8 在 token 28 分叉，已作为 benchmark-only unsafe candidate 拒绝。
- P6.4 已完成：新增 `KVCacheLayoutDescriptor/KVCompactionPlan`，在 prefill 后原子 gather retained K/V、提交 compact page table 并释放尾页；decode 使用 physical context/tail，M-RoPE 保持 logical position。compact swap-out/pickle/swap-in 保留 layout，compact pages 始终不注册 prefix hash。
- P6.4 RTX 5090 multi-image keep=0.5（warmup/repeat `2/5`、output=8、dirty worktree）：physical prompt `408 -> 212`、active blocks `2 -> 1`、occupied bytes `75,497,472 -> 37,748,736`；decode median `32.204 -> 32.231 ms`，无可测 TPOT 收益；输出前 6 token exact、第 7 token 分叉。mixed batch 中 text row 为 dense no-op，image/video 分别为 `210 -> 112`、`422 -> 226` physical prompt tokens。
- P6.5 已完成：同一个自实现 online-softmax paged kernel 显式支持 BF16/FP16/FP32 与 E4M3FN cache，FP8 在 load 时转 FP32 累积，不再整段 gather/dequant/GQA expand。unsupported device/dtype/block 参数显式失败。
- P6.5 RTX 5090 kernel matrix（warmup/repeat `10/50`）32/32 cases PASS；FP8 batch8/context4096 为 `0.2602 ms`，旧 eager reference 为 `1.8029 ms`。BF16 batch1/context4096 kernel `0.2701 ms` 慢于 reference `0.2077 ms`，因此不声称所有 shape 占优。full-engine single-image output32 中 FP8/off decode ratio 为 `0.997x`，32-token exact。
- P6.6 已完成工程与测量闭环。multi-image/video keep=0.5 的组合模式 active prompt bytes 都为 off 的 `0.25x`，output128 TPOT ratio 分别为 `0.996x/1.002x`；mixed keep=0.5 为 `0.375x/1.004x`。这些是显存/性能结果，不代表质量通过。
- P6.6 quality gate 未通过：上述组合模式 output128 stable prefix 分别为 `27/14/[7,28,14]`；固定 COCO 样例中 compact keep=`0.25/0.5/0.75` stable prefix 为 `3/3/7`，keep=1 BF16 compact exact，但 FP8 keep=1 仍只保持前 3 token。当前 unit-scale FP8 与 uniform pruning 都需要更严格的质量策略。
- P6.6 32GB observed capacity（multi-image、600 requests、output2、prefix cache off）为 off/compact/FP8/combo peak running `124/248/249/498`。组合容量提升 `4.016x`，但 elapsed `91.323/83.510/95.602/111.411 s`，容量提升不能表述为 batch throughput 提升。
- P6.7 fixed-pool eager external comparison 已完成：token 等价的 single-image/multi-image/COCO 上，vLLM TPOT 为 Prism off 的 `0.492x/0.484x/0.487x`，SGLang Triton 为 `0.432x/0.413x/0.435x`。external E2E throughput 为 Prism off 的 `1.85x-2.43x`。Prism compression-on 仍约慢 `1.93x-2.45x`，且 uniform pruning 质量失败。
- P6.7 SGLang Blackwell FA3 被源码显式拒绝，FA4 在当前 CUTLASS DSL 编译失败；可执行 baseline 使用 text `triton` + vision `triton_attn`。vLLM 使用 `FLASH_ATTN` + PyTorch native sampler（FlashInfer sampler 的 SM capability probe 失败）。这些 backend 限制均保留日志，不能称为各框架理论最优性能。
- P6.8 variable-size IPC 已完成但动态门禁仍 BLOCKED：rank0 通过独立单向 Pipe 向 worker 广播有边界的 pickle 消息，4,817,396-byte 视觉 tensor payload 对两个 worker 等价，focused guards `5 passed`。`tests/test_llm_vl_tp2.py` 已创建；本机只有 GPU0 RTX 5090，因此两卡 greedy/logits/NCCL/memory/latency 未运行。
- P6.10 原阶段 Review 全量为 `195 passed, 5 skipped`；variable-size TP IPC 后复跑为 `197 passed, 6 skipped in 267.13s`。新增 skip 是当前单卡无法运行的显式 TP2 integration；其余既有门禁无回归。
- P6.11 已完成 physical compression CUDA Graph：clean commit `9e30e55` 上同压缩模式 output32 eager/Graph token exact，batch1 decode speedup `1.84x-1.85x`；combo batch1-8 speedup `1.76x-1.94x`。dirty validation 的 output128 仍 token exact；logical `visual_prune` 继续显式限制为 eager。
- P6.11 post-change 全量回归为 `208 passed, 6 skipped in 267.79s`；raw log 为 `data/p6_system/p611_full_regression_20260714.txt`。
- P6.12-A 已完成 runtime attention score 到 physical compaction/CUDA Graph 的工程闭环。最后 4 层 scorer 对 independent GQA reference max diff `0`，mixed text/image/video 的 text row 保持 dense，image/video physical prompt 为 `112/226`。
- P6.12-A 质量仍 FAIL：keep=0.5 在固定 COCO 上将 stable prefix 从 uniform `3` 提升到 `21`，但 multi-image 只从 `6` 到 `7`，video 仍为 `14`；这些是 3 个 preflight workload，不是 dataset accuracy。
- P6.12-A full regression 为 `212 passed, 6 skipped in 299.59s`；raw log 为 `data/p6_system/p612_full_regression_20260714.txt`。关键 quality matrix 已在 clean commit `c07fa34` 复现。
- P6.12-B 首个 per-span 调查已完成：global attention 的 multi-image/video keep 分布为 `124/72`、`109/87`；新审计字段能直接记录每个 span 的保留数。
- 临时 `attention_span` 将双 span 固定为 `98/98`，但 COCO/multi-image/video stable prefix 仍为 `21/7/14`；它没有优于 global attention，候选代码已删除，不是支持策略。
- rejected ablation 后 focused regression 为 `58 passed in 4.11s`；本轮未改变 pruning 执行路径，因此未重跑 full model regression。P6.12-B quality 继续 FAIL。
- P6.12-B fidelity/reference harness 已完成：7 张固定 COCO 图片、2 个 batch case、35 条 caption reference、schema-v4 backward compatibility、schema-v5 task evidence 和 summary schema-v2。新增 focused guards 为 `54 passed in 3.84s`，最终受影响回归为 `100 passed in 7.50s`。主 runner 的 `--disable-prefix-caching` 继续避免 mixed-VL quality batch 进入尚未支持的 prefix-hit prefill。
- 7-image keep=0.5 dirty validation 中，attention 相比 uniform 将 exact request rate 从 `0/7` 提高到 `3/7`，micro prefix ratio 从 `0.304` 提高到 `0.696`；两者 physical token/active-byte ratio 均为 `0.535x/0.538x`。task evidence 已 eligible，但 attention 因 ROUGE-L macro drop `0.012413 > 0.01` FAIL；uniform ROUGE-L drop 为 `0.036365`，同样 FAIL。
- 当前 reference token-F1/ROUGE-L 是相对 off baseline 的 lexical preflight，不是官方 CIDEr/SPICE 或完整 COCO accuracy。output32 与 detailed-description prompt 也限制绝对分数解释；本轮 `warmup/repeat=1/1`、`git_dirty=true`，不形成新性能 claim。
- P6.12-C 已完成：commit `e51c16d` 的默认 attention last1 clean rerun 复现 token-F1/ROUGE-L drop `0.003288/0.003710`，两项 task gate PASS；稳定 batch4 matrix 显示 `0.536x` physical tokens、`0.571x` active bytes、`1.021x` decode-step speedup，完整回归为 `238 passed, 6 skipped`。该结论仍限定为 7-image lexical preflight，不覆盖标准 COCO accuracy、FP8 质量或在线 serving。
- P6.2 尚未无条件 PASS：RTX 5090 SM utilization 没有成功采集，P6.2 profile 仍为 dirty-worktree 定位实验；只有 P6.11 关键 matrix 已升级为 clean formal evidence。
- pruning 外部 PR 尚未提供链接/commit，暂记为 `pending_external_reference`；不阻塞 Prism internal baseline 和 physical compaction contract。
- P6 当前单卡 execution 与 BF16 visual-compaction/content-aware quality 主线已完成，TP variable-size control plane 也已落地；阶段出口仍不是无条件 PASS：FP8 quality 尚未解决，P6.8 两卡动态矩阵 BLOCKED，P6.2-B hardware-counter formal rerun 待外部平台。下一单卡主线转向真实吞吐/调度支持和架构代码质量，不得先写“全面超过外部框架”。

## P7: 单机性能与外部对标

### 目标

在不扩大到多机 PD/EP 的前提下，把 P6 的 KV 容量机制转化为可解释的单机
性能研究：先冻结 claim 和同条件外部 baseline，再实现 online continuous
batching，并用 trace 驱动 CUDA Graph、Inductor 和 Blackwell kernel 优化。

### 小任务

- [x] P7.0 P6.12 freeze:
  - [x] 推送 `e51c16d/c970c61`，创建并推送 `p6.12-content-aware-kv` tag。
  - [x] 修复 ROADMAP 顶部过期质量状态，新增 `docs/CLAIMS.md`。
  - [x] 新建 `docs/issues/` 和性能调优记录规范。
- [x] P7.1 offline external baseline v2:
  - [x] 分离 `diagnostic_matched` 与 `best_stable`，禁止 eager/Graph 跨协议 ratio。
  - [x] external schema-v2 记录 effective cudagraph/compile mode、GPU UUID、model hash、KV pool、sampling 和 clean state。
  - [x] 汇总器按完整 comparability gates 拒绝不公平 cell，并保持 schema-v1兼容。
  - [x] clean `b17f933` 上完成 5 workloads × 2 profiles × Prism off/compact 的 20-row 自动汇总，全部 comparability gates PASS。
  - [x] same-workload semantic CUDA trace 将 Prism TPOT 分解为 Graph replay `13.394 ms`、logits `4.068 ms`、copy `0.129 ms`、sampler `0.175 ms`；下一 profiling 目标为 Graph 内 decoder 与 logits。
- [x] P7.2 Engine 架构边界重构：Request FSM、immutable BatchPlan、SchedulerPolicy、KV manager/executor/metrics contract。
  - [x] `BatchPlan/StepResult` 强类型主路径与旧 `step()`/五元组解包兼容层。
  - [x] FCFS policy、admission、cancel、swapped CPU page回收和 executor资源释放合同。
  - [x] clean `8b27edc` 完整回归 `249 passed, 6 skipped in 239.20s`。
- [x] P7.3 online arrival、continuous batching、mixed/chunked prefill、admission/preemption 和 SLO goodput：
  - [x] deterministic arrival/cancel/admission、prefill/decode防饥饿 interleave和 request FSM terminal accounting。
  - [x] Q<K paged chunk/prefix prefill、精确 slot mapping、视觉 atomic payload region和 text-only concurrent prefix reuse。
  - [x] schema-validated request queue/TTFT/TPOT/latency、p50/p90/p99、throughput/goodput、KV occupancy/preemption记录。
  - [x] clean `e7796e9` 9-cell online matrix全部完成且 per-cell SLO goodput fraction `1.0`；完整回归 `262 passed, 6 skipped`。
- [x] P7.4 CUDA Graph replay coverage、bucket/padding、Graph 外固定成本与 CPU/GPU overlap：
  - [x] P7.4-A node-level trace 定位并移除每 decode 的 FP32 lm-head 整权重转换；logits CUDA `4.068 -> 0.762 ms`。
  - [x] clean 五 workload 单变量矩阵、HF/COCO quality、更新 external baseline 和完整回归闭环。
  - [x] P7.4-B clean 31-step trace将 replay分为八类：`2,000` kernels/step、kernel busy `12.921 ms`，linear/GEMV占 `70.55%`；Graph 外 kernel busy差约 `0.769 ms`。
  - [x] 固定 `max_num_seqs=8` 的 batch1-8 matrix验证 `[1,2,4,8]` bucket/padding、repeat稳定和 padding row输出隔离；该 matrix不用于 padding性能 claim。
  - [x] CPU/GPU timeline确认 replay CPU range只是异步提交窗口；sampler CPU时间暴露 stream同步，不能与 replay重复相加。
- [x] P7.5 profiling触发的 projection候选评估与优化：
  - [x] trace将 linear/GEMV映射为每步 `253` 次 projection；QKV packed候选因 batch2/4/8 K/V BF16 max diff `1.0`在计时前拒绝。
  - [x] gate/up共享 packed storage实现已落地，保持旧 state-dict strict load；batch `1/2/4/8/210/408/988` MLP output bitwise exact，focused `32 passed`。
  - [x] clean `396702d/8293851/021d4e2` 完成 formal micro、完整 HF logits/PPL、8 个 offline cell、2 个 online A/B、fresh demo与 node-level Systems trace；所有 correctness gate PASS。
  - [x] 实测 replay linear `253 -> 217`、总 kernels `2,000 -> 1,964`；8 个 unprofiled offline cell 的 packed/legacy TPOT ratio 为 `0.9924x–0.9952x`，保留 packed 默认。E2E受 vision prefill双峰影响，不形成稳定加速 claim。
- [ ] P7.6 两卡可用时补 TP2 correctness/communication/performance，不阻塞单卡主线。

### 出口标准

- P7.1 双 profile 在 clean commit 和相同硬件/KV budget 上可复现。
- online benchmark 能输出 arrival/queueing、p50/p90/p99 TTFT/TPOT、throughput、goodput、KV occupancy 和 preemption。
- 至少一个优化从 trace 到 root cause、实现、correctness、E2E/SLO 形成完整闭环；无收益候选也保留 rejected evidence。
- 与 vLLM 的结论限定模型、硬件、workload、质量和执行配置，不写“全面超过”。

### 当前状态

- P7.0/P7.1、P7.2、P7.3 与 P7.4 已完成；协议和命令见
  `docs/P7_OFFLINE_COMPARISON_DESIGN.md`，性能结果见 `docs/PERFORMANCE_REPORT.md`
  6.2-6.11。
- P7.1 matched eager 下 Prism TPOT约为 vLLM 的 `1.91x-1.97x`；P7.4-A 后双方 best-stable Graph 下，quality-qualified compact Prism 与 vLLM 的差距从 `1.65x-1.78x` 缩小到 `1.34x-1.40x`，仍未反超。
- Prism compact 相对自身 off Graph 的 TPOT改善约 `1.5%-3.0%`，主要优势仍是跨 page boundary 后的 active KV bytes/capacity。
- node-level trace 将旧 logits `4.068 ms` 定位为整张 lm-head BF16→FP32 转换与 FP32 GEMV；model-precision 默认将其降到 `0.762 ms`，同时 peak allocated 减少约 `2.18-2.26 GiB`。
- P7.4-B 将优化后 single-image Graph replay分解为 `12.921 ms` kernel busy和
  `2,000` kernels/step；linear/GEMV占 `70.55%`，attention占 `13.17%`，小型
  elementwise/copy/reduction合计约 `15.15%`。P7.5只从这些证据选择候选，不再优化
  已退出 critical path 的 logits。
- P7.5已完成 projection闭环：QKV fusion由严格数值证据拒绝；gate/up在组件、HF
  logits/PPL、text/单图/多图/video/mixed、7-image COCO、online SLO与完整回归中通过。
  clean Systems trace确认每 replay少 36 个 linear，8 个 offline cell 的 decode TPOT
  均小幅改善 `0.483%–0.762%`。该收益很小，且不外推到 E2E或online speedup。
- offline TTFT/vision prefill 存在双峰，当前不把 E2E中位数差异归因为压缩；见 `docs/issues/P7-005-TTFT_VISION_BIMODALITY.md`。
- 当前主线 clean `021d4e2` 完整回归 JUnit 为 `281 passed, 6 skipped in 297.622s`，
  `287 tests / 0 failure / 0 error`；P7.5 HF model-precision logits/PPL三类 VL case均 exact。

## P8: 项目交付

### 目标

把工程结果整理成可以展示、复现和讲清楚的项目交付物。

### 小任务

- [x] 重写 README，覆盖安装、模型准备、快速运行、验证和压缩实验。
- [x] 整理 `docs/TECHNICAL_REPORT.md`，覆盖模型自实现、M-RoPE、DeepStack、KV 分析、压缩和性能。
- [x] 在 `docs/REPRODUCIBILITY.md` 固定分层复现实验命令、真实CPU日志样例、GPU恢复门禁和raw evidence规则。
- [x] 整理 `docs/KNOWN_ISSUES.md`，记录隐藏GPU负载、P7.5、TP2、hardware counter、FP8、server和prefix/video边界。
- [x] 准备 `docs/APPLICATION_MATERIALS.md`，所有简历数字绑定 claim限制与证据入口。
- [x] 修复 editable install metadata和依赖边界；隔离venv完成build/import，CPU/focused smoke为 `40 passed in 5.11s`。

### 出口标准

- [x] fresh venv按 README 安装依赖与 editable wheel，并跑通完整8B最小 demo；输出
  8 个 token IDs 与 decoded text，退出后 GPU恢复到 `1 MiB` baseline。
- [x] 所有README与投递材料关键 claim均能追溯到验证输出、报告或源码。
- [x] 项目有清楚的限制说明、恢复门禁和下一步方向。

### 当前状态

- P8静态交付物已齐全，Markdown本地链接检查 `46/46` PASS。
- clean `568f7bb` 修复PEP 621/setuptools metadata、项目URL和核心依赖；隔离venv
  editable wheel构建与`LLM` import PASS。
- clean `d547385` 新增无权重加载的环境/model/CUDA检查器及3个单测。
- P8动态出口已完成：fresh editable venv 的 8B demo PASS；clean `021d4e2` full
  regression为 `281 passed, 6 skipped`；P7.5动态门禁已闭环。宿主 DALI/`six`
  pip-check warning、TP2、NCU counter和网络server仍按 Known Issues限定，但不阻塞P8。

## P9: 秋招旗舰化

### 目标

把 P8 的可交付研究工程推进为 Qwen3-VL 跨层多模态推理 Runtime：先消除配置、
全局状态和 backend 边界债务，再完成 scaled FP8、细粒度 Paged KV、full-step
CUDA Graph、counter-driven kernel、多模态调度和薄 serving 闭环。完整决策见
`docs/P9_ARCHITECTURE_PERFORMANCE_RFC.md`。

### 最终硬门槛

- 至少一个标准质量合格的 Prism 配置点，在计算 scale/metadata 后的真实
  quality–physical-KV-bytes Pareto 上支配 vLLM 强 baseline。
- 至少一个预注册长视觉 workload 的 decode TPOT、SLO goodput 或 p95/p99 TTFT
  相对 vLLM/SGLang best-stable 改善至少 `5%`，且 process-level 95% CI 不跨零收益。
- compression-off correctness、短文本/普通单图 guardrail 和完整回归继续通过。
- 不要求所有 workload 全面胜出，不用单次 run 或逻辑 token ratio 形成 claim。

### P9-A：架构、协议与正式基线

- [x] 冻结项目定位、非目标、目标架构、双 headline gate 和 2026-08-06 工程截止线。
- [x] 冻结 scaled FP8、W/A shootout、full-step Graph、split-GQA/context kernel、
  thin server 和 TP 条件分支决策。
- [x] NCU 2025.1 权限恢复；page16/256 的真实 counter 与小 grid root cause 已确认。
- [x] 8-GPU topology 与 Prism/vLLM 两套 NCCL stack 已完成隔离诊断。
- [x] 冻结 H1/H2/H3、标准质量集与 canonical hash/revision manifest；媒体物化 hash
  在首次标准质量运行前生成。
- [ ] 完成结构化 paged-decode benchmark 的 focused test 与 clean formal matrix。
- [ ] 保存 page16/256 NCU raw report，并同步 Known Issues/Verification。

### P9-B–P9-F 执行顺序

- [ ] P9-B 架构硬化：typed domain config、未知参数拒绝、删除 `Sequence` 全局 page
  state、immutable `DeviceBatch` 和显式 execution backend contract。
- [ ] P9-C KV/量化：per-token-per-KV-head scaled FP8、scale 全生命周期、细页
  allocator 和标准质量 gate；W/A 只做三模式 shootout。
- [ ] P9-D Compiler/Graph/Kernel：greedy full-step Graph、compile+Graph 候选、
  split-GQA/context paged attention 和 NCU/NSYS 闭环。
- [ ] P9-E 调度/服务/外部对比：multimodal cost model、thin OpenAI-compatible
  streaming server、H3 goodput 和 external best-stable matrix。
- [ ] P9-F 2026-08-06 前完成 clean release candidate、技术报告、简历故事和复现包；
  之后保留 7–10 天学习与复习，不再扩大主线 scope。

### 当前状态

- P9 RFC 已创建；保留 `BatchPlan`、`ModelExecutor`、`SchedulerPolicy`，不做无意义重写。
- 当前 unit-scale FP8 quality FAIL 的根因已明确；5090 支持 FP8，下一实现必须携带
  独立 K/V scale，不能继续把直接 cast 当成公平 FP8 baseline。
- 新 NCU 显示 batch8/context4096 的 achieved occupancy 约 `12.5%`、waves/SM
  `0.17–0.19`；page16 duration `445.60 us`，page256 `550.46 us`。简单把四个 GQA
  query heads 合并会进一步缩小 grid，必须与 context split/稳定 softmax merge 组合。
- P9 单卡主线不等待 TP2。任何 Torch/CUDA/NCCL 升级必须先单独决策，不能污染 P8
  已验证环境。

## 下一步执行顺序

当前应优先执行:

1. 完成 P9-A workload/quality manifest、结构化 page matrix 和 NCU raw evidence。
2. 先提交 P9-B config/global-state/backend contract，再启动任何量化或 kernel diff。
3. scaled FP8 依次通过 component、model-precision logits/PPL、标准质量和 E2E；
   未通过前不做 page-head scale 性能变体。
4. 用 clean NCU/NSYS 选择 full-step Graph 与 split-GQA/context kernel，不为学习 DSL
   强行替换没有瓶颈证据的 Triton 路径。
5. 最后接 thin server 与 H3 external SLO，2026-08-06 冻结工程，转入复习。

P3/P4/P4.5/P5 已建立多模态 FP baseline、KV trace、KV 语义硬化、logical pruning 和 FP8 storage baseline。P6 只允许在统一 benchmark 和固定外部版本下形成性能 claim；没有真实 megakernel实现或 launch-bound 证据时，不开展 megakernel 对比。
