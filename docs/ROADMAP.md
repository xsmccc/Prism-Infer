# Prism-Infer Roadmap

> 修订 2026-06-14: Vision Encoder 改为自实现（参照 vLLM `qwen3_vl.py`），非 HF wrapper。

## 总览

| 月份 | 阶段 | 目标 |
|------|------|------|
| **6月** | 地基 | Qwen3-VL-8B 完整跑通 (全模块自实现) |
| **7月上** | 分析+压缩 | KV Cache 分析 + 压缩策略 |
| **7月下** | 优化 | Triton kernel + 多卡 TP |
| **8月** | 收尾 | MoE offload(可选) + 技术报告 + 投递 |

---

## 6月 W1 (6/14-6/20)：Vision Encoder + 项目基建

### Day 1-2 (6/14-15): 架构分析 + 改动清单 ✅
- [x] 项目框架搭建, 改名 prism-infer, 配置 CLAUDE.md
- [x] Qwen3-VL-8B 数据流 trace
- [x] CHANGES.md 改动清单
- [x] 知识库: 架构深度解析

### Day 3 (6/16): PatchEmbed + ViTMLP
- [ ] 实现 `PatchEmbed`: Conv3d(kernel=(3,16,16), stride=(2,16,16))
- [ ] 实现 `ViTMLP`: GELU-Tanh + Linear(1152→4304→1152)
- [ ] 验证: 与 HF 对应子模块输出误差 < 1e-5
- 参考: vLLM `qwen3_vl.py` L348-L411

### Day 4 (6/17): ViT Attention + 2D RoPE
- [ ] 实现 `ViTAttention`: 合并 QKV + 16 heads + 2D RoPE
- [ ] 实现 `rot_pos_emb`: ViT 专用 2D 旋转位置编码
- [ ] 验证: 与 HF 对应层输出误差 < 1e-5
- 参考: vLLM `qwen3_vl.py` L430-L465 (block), L660-L675 (RoPE)

### Day 5 (6/18): ViTBlock + PatchMerger
- [ ] 实现 `ViTBlock`: Pre-LayerNorm + 残差连接
- [ ] 实现 `PatchMerger`: 空间合并(4→1) + MLP(4608→4608→4096)
- [ ] 验证: 每模块与 HF 误差 < 1e-5
- 参考: vLLM `qwen3_vl.py` L414-L517

### Day 6 (6/19): Position Embed + Full VisionTransformer
- [ ] 实现 `fast_pos_embed_interpolate`: 动态分辨率位置编码插值
- [ ] 组装 `VisionTransformer`: 含 DeepStack 四路输出
- [ ] 验证: 完整 ViT 输出与 HF 误差 < 1e-5
- 参考: vLLM `qwen3_vl.py` L520-L700+, L678-L693

### Day 7 (6/20): 周总结
- [ ] 完整 VisionEncoder 端到端验证
- [ ] 知识库: ViT 实现笔记
- [x] 输出: `prism_infer/vision/vision_encoder.py`

---

## 6月 W2 (6/21-6/27)：M-RoPE + LLM 模型

### Day 8-9 (6/21-22): M-RoPE
- [ ] 实现 `MRope`: Interleaved M-RoPE for LLM (mrope_section=[24,20,20])
- [ ] 验证: 与 HF 的 M-RoPE 输出误差 < 1e-5
- 参考: vLLM `qwen3_vl.py` M-RoPE 实现

### Day 10-11 (6/23-24): LLM Attention + DecoderLayer
- [ ] 实现 `Qwen3VLTextAttention`: 分开 QKV + QK-Norm (RMSNorm)
- [ ] 实现 `Qwen3VLTextMLP`: SiLU + Gate-Up-Down
- [ ] 实现 `Qwen3VLTextDecoderLayer`
- [ ] 验证: 每模块与 HF 误差 < 1e-5

### Day 12-13 (6/25-26): Full Model Assembly
- [ ] 实现 `Qwen3VLTextModel` (36层)
- [ ] 实现 `Qwen3VLForCausalLM`: 含 DeepStack 注入 (Layer 8/16/24)
- [ ] 验证: 端到端 logits 与 HF 误差 < 1e-5

### Day 14 (6/27): 周总结
- [ ] 知识库: M-RoPE + LLM 架构笔记
- [x] 输出: `prism_infer/models/qwen3_vl.py`

---

## 6月 W3 (6/28-7/4)：Engine 集成 + 端到端推理

### Day 15-16 (6/28-29): Input Pipeline
- [ ] Processor 集成: image → pixel_values + input_ids
- [ ] model_runner.prepare_prefill: 支持图像输入, 生成 3D position_ids
- [ ] 验证: 预处理输出与 HF processor 一致

### Day 17-18 (6/30-7/1): Config + Loader + Engine
- [ ] config.py: 支持 VL 模型配置
- [ ] loader.py: 加载 VL 权重 (分开的 QKV, ViT 权重)
- [ ] llm_engine.py: add_request 支持图像参数

### Day 19-20 (7/2-7/3): End-to-End
- [ ] 端到端单图推理 (Prefill + Decode 完整流程)
- [ ] 验证: generate 输出与 HF 一致
- [ ] Benchmark: 本地 4070 上的延迟/吞吐基线

### Day 21 (7/4): 周总结
- [ ] 知识库: Engine 集成笔记

---

## 6月 W4 (7/5-7/11)：KV Cache 分析

- [ ] 截取每层 attention weights (visual vs text token 分开)
- [ ] 可视化 visual token attention pattern
- [ ] 量化 visual token KV 冗余度
- [ ] 输出单图/多图场景分析报告

---

## 7月 W5-6 (7/12-7/25)：压缩策略 + 优化

- [ ] Token-level importance scoring
- [ ] Visual token pruning (基于 attention score)
- [ ] Benchmark: 压缩率 vs perplexity
- [ ] Triton kernel 优化关键算子
- [ ] 与 vllm/sglang 对比评测

---

## 7月 W7-8 (7/26-8/8)：多卡 + 收尾

- [ ] 2卡 TP 支持
- [ ] 长序列压力测试 (5090 32GB)
- [ ] MoE offload 方案 (可选)

---

## 8月 (8/9-8/31)：投递准备

- [ ] 技术博客 + GitHub README
- [ ] 简历更新
- [ ] 面试准备 + 开始投递
