# Prism-Infer 项目路线图

> 修订日期: 2026-06-21
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
| Vision Encoder | 已实现，需持续回归 | `prism_infer/vision/vision_encoder.py` 已存在，已有 PatchEmbed/ViT/DeepStack 相关测试。 |
| M-RoPE | 已实现，需持续回归 | `prism_infer/vision/mrope.py` 已存在，已有 M-RoPE 测试。 |
| Qwen3-VL Text Model | 纯文本 full logits 已严格对齐 | `prism_infer/models/qwen3_vl.py` 已存在，组件测试已建立。 |
| 模块对齐套件 | PASS | 2026-06-21 重新验证: `20 passed in 82.17s`。后续改动必须重新跑。 |
| Full logits | PASS | 2026-06-21 修复 P1-001 后，`tests/test_full_model.py` 输出 max diff `0.000000e+00`, mean diff `0.000000e+00`。 |
| Engine 端到端 VL 推理 | 未完成 | 仍需完善图像输入、3D position_ids、Prefill/Decode 和 `LLM.generate` 接口。 |
| KV Cache 分析与压缩 | 未开始 | 必须等图文端到端路径可靠后进入。 |

## 阶段门禁总览

| 阶段 | 名称 | 目标 | 出口标准 |
|---|---|---|---|
| P0 | 治理与基线 | 固化工程流程、验证入口和当前真实状态 | 文档、插件、验证命令可用，当前风险清楚 |
| P1 | 模型地基严格对齐 | Vision/M-RoPE/Text/Full logits 对齐 | 纯文本 full logits 已 PASS；图文路径进入 P2 |
| P2 | Engine 单图端到端推理 | 从 `LLM` 层接收图文输入并生成 | 单图 greedy tokens 与 HF 一致，纯文本不回归 |
| P3 | KV Cache 分析 | 捕获和量化 visual token KV/attention 行为 | trace 可复现，输出分析报告 |
| P4 | KV Cache 压缩策略 | 实现至少一个视觉 token 压缩策略 | 有压缩率、质量退化、显存/性能实测 |
| P5 | 性能优化与扩展 | Triton/多卡/长序列优化 | correctness 不回归，benchmark 可复现 |
| P6 | 项目交付 | README、技术报告、复现实验和投递材料 | 外部用户能按文档复现核心结果 |

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
- 剩余风险: 以上 PASS 是纯文本 full logits；图文输入、视觉 token 替换、DeepStack 注入和端到端 generate 在 P2 验证。

### 主要验证

详见 `docs/VERIFICATION.md` 的 P1。

## P2: Engine 单图端到端推理

### 目标

把已对齐的模型接入 Prism-Infer engine，使系统能从用户侧接收图文输入，完成 Prefill + Decode，并保持纯文本路径不回归。

### 小任务

- [x] P2.0 设计门禁: 明确单图 VL 数据流、关键风险、验证标准和不做范围，见 `docs/P2_ENGINE_VL_DESIGN.md`。
- [x] P2.1 Processor pipeline: 建立 prompt + image 到 `input_ids` / `pixel_values` / `image_grid_thw` 的稳定入口，并说明 HF processor 作为非核心工具的使用理由。
- [ ] P2.2 多模态 `Sequence`: 携带单图预处理结果、3D position ids / rope delta，并保证跨进程序列化不丢失必要字段。
- [ ] P2.3 自实现 Qwen3-VL 3D position ids: 对齐 HF `get_rope_index` 的单图逻辑，输出 `[3, batch, seqlen]` position ids 和 rope delta。
- [ ] P2.4 KV-aware Qwen3-VL attention + Prefill: 让 Qwen3-VL LLM attention 接入 engine KV cache，并把 VL payload 从 `ModelRunner.prepare_prefill` 传到模型 forward。
- [ ] P2.5 Decode eager 对齐: decode 阶段不重复传图像，只用 last token、KV cache 和 rope delta 延续 position ids。
- [ ] P2.6 Greedy sampler 和 `LLM.generate_vl`: 支持 deterministic greedy，用单图公开 API 对齐 HF tokens。
- [ ] P2.7 P1/P2 回归和阶段 Review: 新增纯文本回归测试，更新问题记录和阶段状态。

### 出口标准

- 单图 prompt 能从 `LLM` 层跑通。
- greedy `temperature=0` 输出 tokens 与 HF 一致。
- 纯文本 prompt 不回归。
- Qwen3-VL attention 必须在 engine prefill/decode 中正确写入和读取 KV cache；仅把图像字段传入模型 forward 不能算 P2 完成。
- 当前 P2 第一版以 `enforce_eager=True` 完成 correctness；VL CUDA Graph decode 未验证前必须列为风险。
- 多图、视频、batch 混合图文不属于 P2 完成范围；未实现时必须显式报错或在文档中标为未支持。
- 若 P1 full logits 未 strict PASS，P2 不能宣称严格精度完成，只能作为功能 smoke。

## P3: KV Cache 分析

### 目标

先建立可观测能力，再讨论压缩。捕获每层 visual token 与 text token 的 attention/KV 行为，量化冗余与重要性。

### 小任务

- [ ] 设计 trace 数据 schema，包含模型配置、输入 shape、图像网格、层号、head、token 区间和统计量。
- [ ] 实现 attention/KV trace 开关，默认关闭，不影响正常推理。
- [ ] 验证 trace on/off 输出一致。
- [ ] 计算 visual token attention mass、token importance、层间冗余、head 差异。
- [ ] 输出离线可视化脚本和样例报告。

### 出口标准

- trace 文件可复现生成。
- 开启 trace 不改变 greedy 输出。
- 至少覆盖单图描述、细节问答、多图或长上下文中的 3 类输入。
- 输出 `docs/KV_ANALYSIS_REPORT.md`，明确压缩假设。

## P4: KV Cache 压缩策略

### 目标

在可靠 FP baseline 上实现视觉 token KV Cache 压缩，并用实测数据说明收益与退化。

### 小任务

- [ ] 设计 compression config，支持 off 和至少一个 active 策略。
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

## P5: 性能优化与扩展

### 目标

在 correctness 不回归的前提下推进 Triton kernel、多卡 TP、长序列压力测试和工程性能优化。

### 小任务

- [ ] 标准化 `bench.py` 参数和输出格式。
- [ ] 建立 4070 / 4090 / 5090 三档硬件的 benchmark 记录模板。
- [ ] 只针对已定位瓶颈实现 Triton/CUDA 优化。
- [ ] 每个自定义 kernel 都有 correctness test 和 benchmark。
- [ ] 设计并验证 2 卡 TP 路径。
- [ ] 跑长序列和多图压力测试。

### 出口标准

- benchmark 包含 warmup、repeat、median、p90、min、max、显存和输入参数。
- 优化前后有同条件对照。
- 单卡/多卡输出一致性验证通过。
- 输出 `docs/PERFORMANCE_REPORT.md`。

## P6: 项目交付

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

1. P2.2/P2.3: 扩展 `Sequence` 并实现单图 3D position ids / rope delta。
2. P2.4/P2.5: 接入 KV-aware attention、prefill 和 eager decode。
3. P2.6/P2.7: 打通 `LLM.generate_vl` greedy tokens 对齐，并跑 P1/P2 回归。

在 P2 图文端到端 greedy tokens 未达标前，不进入 KV Cache 压缩实现。
