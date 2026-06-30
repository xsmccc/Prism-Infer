# Prism-Infer 项目路线图

> 修订日期: 2026-06-30
> 目标模型: Qwen3-VL-8B-Instruct
> 项目目标: 自实现 Qwen3-VL 多模态推理引擎，并在可靠 FP baseline 上完成视觉 token KV Cache 分析与压缩研究。

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
| Engine 端到端 VL 推理 | 多模态 eager 与 CUDA Graph decode correctness 已完成 | P2 已跑通单图 `LLM.generate_vl` 1-token greedy HF exact match，并补齐单图图文 full logits/layerwise strict PASS；P3.1-P3.4 已补齐多图、视频、mixed batch、32-token 长输出和 logits/ppl 分布；P3.5 已补齐 VL CUDA Graph decode；P3.6 已接入自实现 Triton paged decode kernel。 |
| VL Engine 完整性 | 已完成当前 P3 门禁 | P3.0-P3.7 已完成；多图、视频、mixed batch、32-token 长输出、logits/ppl 分布、VL CUDA Graph decode、paged decode kernel 和 P1/P2/P3 回归均有验证记录。 |
| KV Cache 分析 | P4 已完成当前门禁 | 已实现 repo 内 KV trace schema/session、attention/KV 采集、离线 summary 和三类样例报告；trace on/off greedy tokens 已验证一致。 |
| KV Engine Hardening | P4.5 已完成当前门禁 | 已统一 canonical 4D paged KV layout 写入语义，修复 CPU fallback、prefix hash 释放清理、swap CPU/GPU 页表混用，并把 prefix-cache prefill 提前显式拒绝。 |
| KV Cache 压缩 | 未开始 | P5 将基于 P4 trace 报告和 P4.5 稳定 KV 语义实现 compression off baseline 和首个 visual token pruning 策略。 |

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
| P6 | 性能优化与扩展 | torch.compile、Triton、自定义算子、多卡、长序列优化 | correctness 不回归，benchmark 可复现 |
| P7 | 项目交付 | README、技术报告、复现实验和投递材料 | 外部用户能按文档复现核心结果 |

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
- [ ] 将历史 `DAY_*.md` 文档标注为历史记录，避免与主路线图冲突。
- [ ] 建立每阶段交付模板: 目标、改动范围、验证输出、风险、下一步。

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
- P2 阶段外能力已在 P3 补齐多图、视频、mixed batch、32-token 长输出、logits/ppl 分布、VL CUDA Graph decode 和 paged decode kernel；仍未支持 prefix-cache/chunked-prefill VL mixed batch，P3 kernel 也只是 baseline kernel。

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
| batch 混合图文 | P3.3 已解除 non-prefix mixed batch correctness 阻断，P3.5 已覆盖 CUDA Graph mixed batch；当前仍未覆盖 prefix-cache/chunked-prefill VL mixed batch。 | `prism_infer/engine/model_runner.py`, `tests/test_model_runner_vl_mixed_prefill.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| 长输出质量 | P3.4 已覆盖单图/多图/视频 8/16/32-token HF greedy exact、32-token logits/ppl 分布、mixed batch VL rows 32-token 等价；text-only mixed 32-token 分叉已证明为 HF/Prism 共有 batch-size 数值敏感性。 | `tests/test_llm_vl_long_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_vl_logits_distribution.py`, `tests/test_batch_numeric_sensitivity.py` |
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
  - 建立 `max_tokens=8/16/32` greedy exact token 对齐。
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
- 长输出多 token greedy 与 HF token ids 完全一致；分布或 perplexity 指标达到 `docs/VERIFICATION.md` 门槛。
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
  - 单图/多图/视频 `max_tokens=32` HF greedy exact，prefix@8/16/32 全部 match。
  - 单图/多图/视频 teacher-forced logits shape `[1,32,151936]`，max diff `0.000000e+00`，mean diff `0.000000e+00`，ppl diff `0.000000e+00`。
  - mixed batch VL rows 32-token 输出与 fresh 单请求独立运行一致。
  - mixed batch text-only row 32-token 分叉来自 bf16 batch-size 数值敏感性；HF 与 Prism duplicate batch max/mean diff 均为 `5.312500e-01 / 1.473503e-01`。
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
- [x] 计算 visual token attention mass、token importance、层间冗余、head 差异。
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
  - `summarize_trace` 输出 visual attention mass、visual/text K norm ratio、head 差异和相邻层冗余。
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

### 设计决策

- 选择 canonical 4D paged cache，而不是改成全局 2D cache。理由是 `ModelRunner.allocate_kv_cache`、paged decode kernel 和现有 P3/P4 GPU 路径都已经以 `[num_blocks, block_size, heads, dim]` 为真实物理布局；只需要把 flat slot 到 4D index 的解释集中修正。
- 选择释放 free block 时清理 `hash_to_block_id`，而不是继续允许 free block 作为可命中的 prefix cache。理由是当前 BlockManager 没有独立 cached/reserved 状态；让 free list 同时承担可复用 prefix cache 会导致 hash 指向已释放 block 的生命周期不清晰。后续若要持久 prefix cache，应新增独立状态，而不是复用 free block。
- 选择新增 `cpu_block_table`，而不是让 `block_table` 根据 sequence status 改变含义。理由是 `prepare_decode`、trace、paged decode kernel 都默认 `block_table` 是 GPU block id；字段语义稳定比节省一个 list 更重要。
- 选择 early gate prefix-cache prefill，而不是在 P4.5 中实现 paged prefill kernel。理由是 P4.5 的目标是 hardening，不扩大到新的 attention kernel；paged prefill 属于后续 P6/P5 交叉的大任务。

### 剩余风险

- `kvcache_block_size=256` 仍然是 P5 压缩粒度风险。P4.5 不直接修改该值，避免破坏 P3 correctness/benchmark 基线；P5.0 必须先决定是降低 block size，还是在 256-token page 内增加 sub-page 压缩 metadata。
- prefix-cache prefill 仍未实现；当前只是把失败前移并显式化。
- swap 数据搬运仍有全局 `torch.cuda.synchronize()`，这是 P6 性能优化项，不属于 P4.5 correctness hardening。
- paged decode Triton kernel 的 `MAX_CONTEXT_LEN` 冗余迭代和 `BLOCK_N=32` 固定调优仍是 P6 kernel 优化项。

## P5: KV Cache 压缩策略

### 目标

在可靠 FP baseline 和 P4.5 稳定 KV 语义上实现视觉 token KV Cache 压缩，并用实测数据说明收益与退化。

### 小任务

- [ ] 设计 compression config，支持 off 和至少一个 active 策略。
- [ ] P5.0 压缩粒度设计门禁: 决定 `block_size` 降低方案或 256-token page 内 sub-page metadata 方案。
- [ ] 建立 compression off 等价 baseline。
- [ ] 实现 token-level importance scoring。
- [ ] 实现首个 visual token pruning 策略。
- [ ] 保证失败时显式报错，不 silent fallback。
- [ ] 评估压缩率、显存、延迟、吞吐、token 一致率和质量退化。

### 出口标准

- compression off 与 FP baseline 完全一致。
- compression on 有明确压缩率和质量退化数字。
- 至少一个策略在指定场景下显示可测收益。
- 输出 `docs/COMPRESSION_REPORT.md` 和原始 benchmark 日志。

## P6: 性能优化与扩展

### 目标

在 correctness 不回归的前提下推进 torch.compile、Triton kernel、多卡 TP、长序列压力测试和工程性能优化。P3 的 paged decode kernel 属于 decode 核心路径基线；P6 在此基础上继续做全系统优化和与 vLLM/SGLang 的同条件对比。

### 小任务

- [ ] 标准化 `bench.py` 参数和输出格式。
- [ ] 建立 4070 / 4090 / 5090 三档硬件的 benchmark 记录模板。
- [ ] 建立与 vLLM/SGLang 对比前的证据清单: 版本、commit、启动参数、输入集合、warmup/repeat、显存限制和采样配置。
- [ ] 只针对已定位瓶颈实现 Triton/CUDA 优化。
- [ ] 每个自定义 kernel 都有 correctness test 和 benchmark。
- [ ] 评估 `torch.compile` 对 prefill/decode/vision encoder 的收益和 graph break 风险。
- [ ] 设计并验证 2 卡 TP 路径。
- [ ] 跑长序列和多图压力测试。

### 出口标准

- benchmark 包含 warmup、repeat、median、p90、min、max、显存和输入参数。
- 优化前后有同条件对照。
- 单卡/多卡输出一致性验证通过。
- 输出 `docs/PERFORMANCE_REPORT.md`。

## P7: 项目交付

### 目标

把工程结果整理成可以展示、复现和讲清楚的项目交付物。

### 小任务

- [ ] 重写 README，覆盖安装、模型准备、快速运行、验证和压缩实验。
- [ ] 整理技术报告，覆盖模型自实现、M-RoPE、DeepStack、KV 分析、压缩和性能。
- [ ] 固定最小复现实验命令和日志样例。
- [ ] 整理 Known Issues，不把未验证内容写成完成。
- [ ] 准备面试/投递材料。

### 出口标准

- 新环境能按 README 跑通最小 demo。
- 所有关键 claim 都能追溯到验证输出、报告或源码。
- 项目有清楚的限制说明和下一步方向。

## 下一步执行顺序

当前应优先执行:

1. P5.0: 压缩粒度与 metadata 设计门禁，决定 block-size/sub-page 方案。
2. P5.1: 设计 compression config，并建立 compression off 等价 baseline。
3. P5.2: 基于 P4 trace 实现首个 visual-token importance scoring。
4. P5.3: 实现首个 visual token pruning/retention 策略，并输出压缩率、质量退化和显存/延迟数据。

P3/P4/P4.5 已建立可用的多模态 FP baseline、KV trace 能力和 KV 语义硬化基础；进入 P5 后仍不能直接声称超越 vLLM/SGLang。对比前必须另建同条件 benchmark 设计，记录对方版本/commit、启动参数、输入集合、warmup/repeat、显存限制和采样配置。
