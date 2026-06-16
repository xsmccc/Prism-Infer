# Day 2: VisionEncoder 实现

## 目标
在 prism-infer 中实现 Qwen3-VL 的 Vision Encoder，从 HF 加载权重，输出 4 路特征（主 + 3 DeepStack）。

## 为什么先做 VisionEncoder
- 这是多模态推理的第一步，独立于 LLM，可以单独验证
- 本地 4070 7GB 跑 ViT (~1.5GB) 绰绰有余
- 输出正确后，Day 3-4 拼到 LLM 上就水到渠成

---

## 任务 2.1: 搭建 ViT 骨架 + 加载 HF 权重 (1h)

### 做什么
- 创建 `prism_infer/vision/vision_encoder.py`，定义 `VisionEncoder` 类
- 从 HF 模型提取 `model.visual` 的 state_dict
- 实现权重加载逻辑

### 文件
- `prism_infer/vision/__init__.py` (更新)
- `prism_infer/vision/vision_encoder.py` (新建)

### 验证标准
- `VisionEncoder` 能实例化
- 能成功加载 HF 权重，打印每层参数量

---

## 任务 2.2: 实现 PatchEmbed (Conv3d) (45min)

### 做什么
- 实现 `PatchEmbed` 类，封装 Conv3d
- 输入: pixel_values [N, 1536]
- 输出: patch_features [N, 1152]
- 内部流程: reshape → Conv3d → squeeze

### 验证标准
- 输入 [784, 1536] 测试数据，输出 shape = [784, 1152]
- 与 HF 的 `model.visual.patch_embed(pixel_values)` 输出误差 < 1e-5

---

## 任务 2.3: 实现 ViTBlock (1h)

### 做什么
- 实现 `ViTBlock` 类 (LayerNorm + Attention + LayerNorm + MLP)
- Attention: 合并 QKV, 16 heads, 72 dim/head, 双向(无 causal mask)
- MLP: Linear(1152→4304) + GELU-Tanh + Linear(4304→1152)
- 残差连接

### 验证标准
- 随机输入 [1, 784, 1152]，输出 shape = [1, 784, 1152]
- 单层 Block 与 HF 对应 block 输出误差 < 1e-5

---

## 任务 2.4: 实现 Merger + DeepStack (1h)

### 做什么
- 实现 `PatchMerger` 类: LayerNorm → Linear(1152→4096) → GELU → Linear(4096→4096) + 空间合并
- 1 个主 Merger + 3 个 DeepStack Merger
- 空间合并: 784 patches → 196 tokens (spatial_merge_size=2)

### 验证标准
- 输入 ViT Block 26 的输出 [784, 1152]
- 主 Merger 输出 [196, 4096]，与 HF `model.visual.merger(...)` 误差 < 1e-5
- ds[0] 输入 ViT Block 8 的输出，输出 [196, 4096]，与 HF 对应 merger 误差 < 1e-5

---

## 任务 2.5: 组装 VisionEncoder 完整 forward (1h)

### 做什么
- 把 PatchEmbed + 27 ViTBlocks + 4 Mergers 串起来
- forward 返回 `(main_features, [ds0, ds1, ds2])`
- 处理 DeepStack 在特定层提取特征

### 验证标准
- 输入真实图片的 pixel_values [784, 1536]
- 主输出 [196, 4096] 与 HF `model.visual(pixel_values)[0]` 误差 < 1e-5
- 3 个 DeepStack 输出各 [196, 4096] 与 HF `model.visual(pixel_values)[1][i]` 误差 < 1e-5

---

## 任务 2.6: 写入知识库 (30min)

### 做什么
- 将 VisionEncoder 实现过程中的关键决策写入知识库
- 记录: ViT Block 结构、Merger 空间压缩原理、权重加载方式

---

## Day 2 产出清单
1. `prism_infer/vision/vision_encoder.py` — 完整 VisionEncoder
2. `prism_infer/vision/__init__.py` — 更新导出
3. 验证脚本: 与 HF 输出对比
4. 知识库: 1篇

---

## 开发方式
每个任务:
1. 我先讲要做什么、为什么这样做
2. 你来写代码 (或我写然后你 review)
3. 跑验证脚本，确认通过
4. 通过后学一遍，写知识库
5. 下一个任务
