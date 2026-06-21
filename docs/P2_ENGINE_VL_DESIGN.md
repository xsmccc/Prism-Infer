# P2 Engine 单图端到端推理设计

> 修订日期: 2026-06-21
> 阶段目标: 从 `LLM` 用户入口接收单图图文输入，完成 Prefill + Decode，并让 greedy tokens 与 Hugging Face Qwen3-VL 参考完全一致。

## P2 范围

P2 只解决单图、单请求、非视频的端到端路径:

```text
prompt + image
  -> processor/tokenizer
  -> Sequence 多模态请求
  -> ModelRunner.prepare_prefill
  -> Qwen3-VL vision embedding 替换
  -> engine KV-aware attention prefill
  -> decode 复用 KV cache
  -> greedy token 输出
```

P2 不做 KV Cache 压缩、不做多图 batch、不做视频、不做 CUDA Graph VL decode 性能优化。以上能力必须等单图 greedy tokens 对齐后再进入后续阶段。

## 当前证据

| 主题 | 当前事实 | 证据 |
|---|---|---|
| 请求结构 | `Sequence` 当前只保存 token、采样参数、KV block 状态，没有图像字段。 | `prism_infer/engine/sequence.py:20-33` |
| 用户入口 | `LLMEngine.add_request` 当前只接收 `prompt: str | list[int]`，`generate` 无 images 参数。 | `prism_infer/engine/llm_engine.py:58-104` |
| Prefill positions | `ModelRunner.prepare_prefill` 当前生成一维 positions。 | `prism_infer/engine/model_runner.py:320-387` |
| Decode positions | `ModelRunner.prepare_decode` 当前使用 `len(seq) - 1` 一维位置。 | `prism_infer/engine/model_runner.py:389-418` |
| 模型 VL forward | `Qwen3VLForCausalLM.forward` 已支持 `pixel_values/image_grid_thw/position_ids`。 | `prism_infer/models/qwen3_vl.py:427-438` |
| 视觉 token 替换 | `Qwen3VLModel.forward` 已做 image token mask、数量校验和 `masked_scatter`。 | `prism_infer/models/qwen3_vl.py:355-383` |
| 当前 attention | `Qwen3VLTextAttention` 当前使用全序列 SDPA，没有调用 engine 的 `Attention`/KV cache。 | `prism_infer/models/qwen3_vl.py:74-132` |
| engine KV attention | `prism_infer.layers.attention.Attention` 已有 `store_kvcache`、prefill/decode 上下文和 flash-attn KV cache 路径。 | `prism_infer/layers/attention.py:36-112` |
| 采样 | `SamplingParams` 当前禁止 `temperature≈0`，`Sampler` 只实现随机采样。 | `prism_infer/sampling_params.py:11-13`, `prism_infer/layers/sampler.py:17-35` |
| HF processor | HF processor 返回 `input_ids/pixel_values/image_grid_thw`，并按 image grid 展开 image token。 | `.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:146-194` |
| HF M-RoPE index | HF `get_rope_index` 为图文输入生成 `[3, batch, seqlen]` position ids 和 `rope_deltas`。 | `.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033` |
| HF forward | HF forward 在 prefill 计算 rope index，在 decode 用 `cache_position + rope_deltas` 延续 position ids。 | `.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1177-1221` |

## 设计决策

### D1: HF processor 作为非核心工具使用

选择:

- P2 使用 HF `AutoProcessor` 或等价 processor 调用生成 `input_ids/pixel_values/image_grid_thw`。
- 这不是模型核心、attention、M-RoPE、KV cache 或压缩逻辑，属于成熟预处理工具和 ground truth 对齐入口。
- 代码中必须写明第三方 processor 使用理由和源码参考位置。

拒绝:

- 不在 P2 自实现完整图像 resize、patch packing、chat template 和 tokenizer 细节。这样会把 P2 从 engine 对齐扩大成预处理重写，风险过高。
- 不把 HF model wrapper 放进 Prism-Infer 推理路径。HF 只能作为 processor 或验证参考，不能替代自实现模型。

风险:

- Processor 版本变化会影响 token expansion。P2 测试必须固定本地 transformers 版本输出作为参考，并记录 shape 和 token 数。

### D2: P2 第一版限制为单图单请求

选择:

- 先支持一条请求包含一张图片。
- `Sequence` 可以设计为可扩展结构，但 P2 只验证单图路径。
- 混合 batch、多图、视频如果未实现，必须显式报错。

拒绝:

- 不在第一次接入时同时支持 batch 多图和视频。多样输入会同时影响 processor、position ids、slot mapping、KV cache 和 decode 对齐，不利于定位。

风险:

- P2 PASS 不代表多图或视频能力完成，文档和接口错误信息必须明确。

### D3: 自实现 Qwen3-VL rope index helper

选择:

- 在 Prism-Infer 内实现单图 `get_rope_index` 等价 helper，输出 `position_ids` 和 `rope_delta`。
- 参考 HF `modeling_qwen3_vl.py:916-1033`，但不把 HF model 方法作为运行时 wrapper。

拒绝:

- 不在 `prepare_prefill` 中继续使用一维 `range` 位置伪装图文 M-RoPE。
- 不在核心路径调用 HF model 的 `get_rope_index`。

风险:

- `vision_start_token_id/image_token_id/spatial_merge_size` 必须从 config/processor 获取，不能写死。

### D4: P2 必须补上 KV-aware attention

选择:

- Qwen3-VL LLM attention 要接入 engine 的 KV cache 上下文，或实现等价 KV-aware 路径。
- Prefill 要把 K/V 写入 KV cache；decode 只喂 last token 时必须能读完整历史 K/V。

拒绝:

- 不接受“只把图像参数传进模型 forward”作为 P2 完成。当前 `Qwen3VLTextAttention` 是全序列 SDPA；decode 只喂 last token 时如果没有 KV-aware attention，输出不可信。

风险:

- 改 attention 会影响 P1 full logits 和纯文本路径，必须单独跑 P1 回归。

### D5: Greedy 采样是 P2 门禁的一部分

选择:

- P2 必须支持 `temperature=0` 或显式 greedy 模式，用于与 HF end-to-end tokens 完全对齐。
- `Sampler` 要保留随机采样路径，同时增加 deterministic argmax 路径。

拒绝:

- 不用低温随机采样近似 greedy。当前 `SamplingParams` 禁止 `temperature≈0`，`Sampler` 使用 Gumbel-Max 随机采样；这不能作为 greedy 对齐测试。

风险:

- 采样接口变更会影响现有 benchmark 默认行为，需要纯文本回归测试覆盖。

### D6: 先 eager 对齐，再处理 VL CUDA Graph

选择:

- P2 第一版用 `enforce_eager=True` 完成单图 greedy 对齐。
- CUDA Graph 的 3D position ids 和 VL decode graph 作为 P2 后续风险或 P5 性能优化任务。

拒绝:

- 不在首个 P2 实现同时修改 graph capture、graph replay、VL decode、processor 和 attention。

风险:

- P2 PASS 只证明 correctness，不证明 CUDA Graph VL decode 性能。

## P2 小任务

### P2.0: 设计和文档门禁

目标:

- 固化 P2 数据流、任务拆分、验证标准和不做范围。

完成条件:

- `docs/ROADMAP.md` P2 小任务拆分到可执行粒度。
- `docs/VERIFICATION.md` P2 验证命令和 PASS 标准明确。
- 本文档记录关键设计决策和证据。

### P2.1: Processor pipeline

目标:

- 建立 prompt + image 到 `input_ids/pixel_values/image_grid_thw` 的稳定入口。

计划文件:

- `prism_infer/engine/vl_inputs.py` 或等价模块。
- `tests/test_processor_pipeline.py`。

PASS:

- 与 HF processor 输出一致。
- 输出包含 input ids shape、pixel values shape、image grid shape、image token 数量、PASS/FAIL。

当前状态:

- 2026-06-21 已完成。
- 新增 `prism_infer/engine/vl_inputs.py`，封装单图 processor 边界和 shape/token 数校验。
- 新增 `tests/test_processor_pipeline.py`，验证 processor 输出 exact match、`token_ids` 属性和视觉 token mismatch 显式报错。
- 验证结果: `3 passed in 6.23s`。

### P2.2: 多模态 Sequence

目标:

- `Sequence` 能携带单图预处理结果、3D position ids/rope delta，并支持 runner 进程间传输。

计划文件:

- `prism_infer/engine/sequence.py`。
- `tests/test_sequence_multimodal.py`。

PASS:

- 纯文本 `Sequence` 行为不变。
- 单图 `Sequence` 序列化后保留必要 VL 字段。
- 不支持的多图/视频状态显式报错。

### P2.3: Qwen3-VL 3D position ids

目标:

- 自实现单图 `get_rope_index` 等价逻辑。

计划文件:

- `prism_infer/vision/rope_index.py` 或 `prism_infer/models/qwen3_vl_position.py`。
- `tests/test_vl_rope_index.py`。

PASS:

- `position_ids` shape 为 `[3, 1, seqlen]`。
- `rope_delta` shape 为 `[1, 1]`。
- 与 HF `get_rope_index` max diff `0`。

### P2.4: KV-aware Qwen3-VL attention 与 Prefill 接入

目标:

- Qwen3-VL LLM attention 接入 engine KV cache。
- `ModelRunner.prepare_prefill` 传递 VL payload 到模型 forward。

计划文件:

- `prism_infer/models/qwen3_vl.py`。
- `prism_infer/engine/model_runner.py`。
- `tests/test_qwen3_vl_attention_kv.py`。
- `tests/test_model_runner_vl_prefill.py`。

PASS:

- Prefill 写入 KV cache。
- 单图 prefill next-token logits 与 HF 达到门槛。
- 纯文本 full logits 仍 PASS。

### P2.5: Decode eager 对齐

目标:

- decode 阶段不再传图像，只用 last token、KV cache 和 `rope_delta` 延续 position ids。

计划文件:

- `prism_infer/engine/model_runner.py`。
- `tests/test_vl_decode_position.py`。

PASS:

- decode 第 1/2 步 position ids 与 HF 逻辑一致。
- decode 不重复运行 Vision Encoder。
- 单图逐 token greedy 与 HF 一致。

### P2.6: Greedy sampler 与 `LLM.generate_vl`

目标:

- 提供单图用户入口，并支持 deterministic greedy。

计划文件:

- `prism_infer/sampling_params.py`。
- `prism_infer/layers/sampler.py`。
- `prism_infer/engine/llm_engine.py`。
- `tests/test_sampler_greedy.py`。
- `tests/test_llm_vl_generate.py`。

PASS:

- `temperature=0` 或 `greedy=True` 走 argmax。
- 单图 `LLM.generate_vl` 输出 token ids 与 HF 完全一致。
- 现有随机采样路径不回归。

### P2.7: P1/P2 回归和阶段 Review

目标:

- 证明 P2 没有破坏 P1 纯文本 baseline。

计划文件:

- `tests/test_text_only_regression.py`。
- `docs/ISSUE_LOG.md`。
- `docs/ROADMAP.md`。

PASS:

- `compileall` PASS。
- P1 模块对齐套件 PASS。
- `tests/test_full_model.py` PASS。
- P2 全部新增测试 PASS。
- `docs/ISSUE_LOG.md` 记录 P2 期间真实问题、根因、修复和验证。

## P2 总出口标准

P2 只有在以下条件全部满足时才能标记完成:

- 单图 prompt 能从 `LLM` 层跑通。
- greedy tokens 与 HF 完全一致。
- 纯文本 full logits 不回归。
- P2 新增测试全部 PASS。
- 多图、视频、VL CUDA Graph decode 等未完成能力在文档中列为风险，而不是写成已完成。
