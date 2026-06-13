# Prism-Infer Roadmap

## 总览

| 月份 | 阶段 | 目标 |
|------|------|------|
| **6月** | 地基 | Qwen3-VL-8B 跑通 + KV Cache 分析 + 压缩策略 |
| **7月上** | 优化 | Triton kernel + 端到端评测 |
| **7月下** | 分布式 | 多卡 TP |
| **8月** | 收尾 | MoE offload(可选) + 技术报告 + 投递 |

---

## 6月 W1 (6/14-6/20)：跑通模型

### Day 1-2 (周末): 理解 + 准备
- [ ] 用 HF 加载 Qwen3-VL-8B，trace 前向，画出数据流图
- [ ] 对比 prism_infer 源码，列出所有需要改的文件
- [ ] 搭建 prism-infer 项目框架

### Day 3-4: Vision Encoder + Projector
- [ ] 实现 `VisionEncoder` (ViT wrapper)
- [ ] 实现 `Projector` (vision → LLM hidden dim)
- [ ] 验证: 与 HF 原版输出误差 < 1e-5

### Day 5-6: M-RoPE + Input 准备
- [ ] 实现 M-RoPE (visual 3D + text 1D position encoding)
- [ ] 改 `prepare_prefill` 支持图像输入
- [ ] 验证: 单图+文本推理输出正确

### Day 7: 周总结
- [ ] 端到端单图推理验证
- [ ] 知识库周报
- [ ] Week 2 计划微调

---

## 6月 W2 (6/21-6/27)：KV Cache 分析

- [ ] 截取每层 attention weights (visual vs text token 分开)
- [ ] 可视化 visual token 的 attention pattern
- [ ] 量化 visual token KV 冗余度
- [ ] 输出单图/多图场景下的分析报告

---

## 6月 W3 (6/28-7/4)：压缩策略

- [ ] 实现 token-level importance scoring
- [ ] 实现 visual token pruning (基于 attention score)
- [ ] benchmark: 压缩率 vs perplexity

---

## 6月 W4 (7/5-7/11)：Kernel 化

- [ ] 用 Triton 写 visual token KV fused-merge kernel
- [ ] profiling + roofline 分析
- [ ] 集成到 prism_infer attention 层

---

## 7月 W5-6 (7/12-7/25)：端到端 + 多卡

- [ ] 与 vllm/sglang 做对比评测
- [ ] 2卡 TP 支持
- [ ] 显存/吞吐/精度完整数据

---

## 8月 (7/26-8/31)：收尾 + 投递

- [ ] MoE offload 方案（可选挑战）
- [ ] 技术博客 + GitHub README
- [ ] 简历更新 + 开始投递
