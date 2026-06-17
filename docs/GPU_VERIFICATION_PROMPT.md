
# GPU 验证任务

你现在在 5090 32GB 远程服务器上。请按以下步骤完成验证。

## 环境信息

- 项目路径: `/data/nano-vllm`
- 模型路径: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`
- GPU: RTX 5090 32GB
- torch 2.6.0, CUDA True

## 项目背景

Prism-Infer：基于 nano-vllm 的多模态推理引擎，正在适配 Qwen3-VL-8B。
已完成的模块全部自实现（非 HF wrapper），本地 WSL 7.8GB 内存无法跑 36 层 LLM 全模型，租 5090 验证。

## 已完成的代码

| 文件 | 内容 | 本地验证 |
|------|------|---------|
| `prism_infer/vision/vision_encoder.py` | VisionEncoder (PatchEmbed/ViTMLP/ViTAttention/ViTBlock/PatchMerger/VisionTransformer) | 单模块 diff=0, 完整 ViT diff<0.02 |
| `prism_infer/vision/mrope.py` | MRope: LLM 3D 位置编码, cos/sin [batch, seqlen, 128] | diff=0 |
| `prism_infer/models/qwen3_vl.py` | LLM backbone (RMSNorm/Attention/MLP/DecoderLayer/TextModel/CausalLM) | 单层 diff~0.03(bf16), 权重key匹配 |
| `prism_infer/layers/attention.py` | flash_attn/triton 可选导入 (CPU fallback) | 本地OK |
| `tests/test_vision_encoder.py` | VisionEncoder 完整测试 | PASS (CPU bf16 ~0.008) |
| `tests/test_mrope.py` | MRope 测试 | PASS diff=0 |
| `tests/test_qwen3_vl.py` | LLM 组件 + 权重key测试 | RMSNorm/MLP/DecoderLayer PASS |

## 在 GPU 上要验证的

### 第1步: 安装依赖
```bash
cd /data/nano-vllm
pip install flash-attn --no-build-isolation 2>/dev/null || pip install flash-attn
# 如果需要: pip install transformers pillow safetensors
```

### 第2步: 跑本地已有测试 (确认 GPU 运行正常)
```bash
python3 tests/test_vision_encoder.py
python3 tests/test_mrope.py
python3 tests/test_qwen3_vl.py
```
预期: 所有测试 PASS，GPU 上 diff 应该接近 0 (之前 CPU bf16 有 ~0.01-0.03 的误差)

### 第3步: 全模型文本前向 vs HF (之前 WSL 跑不了)
写一个新测试脚本 `tests/test_full_model.py`:
- 加载 HF Qwen3VLForConditionalGeneration (device_map='cuda')
- 加载我们的 Qwen3VLForCausalLM (device_map='cuda')
- 加载所有权重
- 用相同 input_ids + position_ids 跑 forward
- 对比 logits: max diff, mean diff
- 预期: GPU bf16 下 diff < 0.01

### 第4步: Vision + LLM 端到端
- 加载完整模型 (VisionEncoder + LLM backbone)
- 输入: 一张测试图 + "描述这张图片"
- 跑 Prefill，对比 HF 的 logits

## 已知问题

1. CPU bf16 LayerNorm 精度 ~0.0002, 单层 Attention ~0.03, 36层累积更大。GPU 上应消失。
2. DeepStack 注入未完整实现 (qwen3_vl.py 中的 Qwen3VLModel.forward 标记了 FIXME)
3. gitignore 会排除 models/ 目录, 但 prism_infer/models/ 已加例外

## CLAUDE.md 规则

本项目的 AI 行为宪法：
- 所有声称必须引用源码行号
- 所有模块必须自己实现 (不能 wrap HF)
- 每次代码变更必须有验证输出
- 没有实测不能声称速度/精度
- 做完每个模块必须教会用户

详细规则见 `/data/nano-vllm/CLAUDE.md`。
