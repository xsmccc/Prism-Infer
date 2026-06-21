# Prism-Infer 问题解决记录

> 目的: 每解决一个真实问题，都记录证据、定位路径、根因、修复和验证结果。禁止把猜测写成结论；未验证内容必须标记为未验证。

## 记录模板

```text
ID:
标题:
状态: Open | Investigating | Fixed | Verified | Won't Fix
发现方式:
影响范围:
证据:
定位过程:
根因:
修复:
验证命令:
验证结果:
经验:
剩余风险:
```

## P1-001: Full logits 对齐为 MARGINAL

状态: Verified

发现方式:

- P1 full logits 验证。

影响范围:

- 阻断 P1 “Qwen3-VL 模型地基严格对齐”出口。
- 在修复前，不能声明 full-model strict PASS，也不能进入 KV Cache 压缩实现阶段。

证据:

```text
命令:
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model.py

输出摘要:
GPU: NVIDIA GeForce RTX 5090 (31.4 GB)
Dtype: torch.bfloat16
权重: 750/750 loaded
Missing: []
Unexpected: []
Shape: HF=[1, 64, 151936], Our=[1, 64, 151936]
NaN: HF=0, Our=0
Max diff:  3.125000e-01
Mean diff: 2.480617e-02
Result: MARGINAL
```

定位过程:

- 已确认语法检查 PASS。
- 已确认 P1 模块对齐套件 `20 passed in 81.83s`。
- 已新增并运行 `tests/test_full_model_layerwise_debug.py`，按 embedding、RoPE、每层 norm/attention/MLP/output、final norm、logits 比较 HF 与 Prism-Infer 激活。
- 分层证据:
  - `embed`: max diff `0.000000e+00`。
  - `rope`: max diff `0.000000e+00`。
  - 第一处非零误差: `layer_00.attn`, max diff `3.906250e-03`, mean diff `7.651032e-05`。
  - `layer_00.mlp`: max diff `6.250000e-02`, mean diff `7.205015e-04`。
  - 误差随层数累积，`layer_35.mlp` max diff `2.000000e+01`, mean diff `6.590960e-02`。
  - final norm 后误差收敛为 max diff `1.500000e+00`, mean diff `7.806452e-03`。
  - logits max diff `2.500000e-01`, mean diff `2.831022e-02`。
- 当前证据指向 attention 路径是首个差异来源；embedding、权重加载、RoPE 不是首个差异来源。
- 进一步微定位脚本 `tests/test_attention_micro_debug.py` 显示:
  - `embed/cos/sin/input_norm/q_norm/k_norm/v` 全部 max diff `0.000000e+00`。
  - 修复前第一处差异在 `q_rope/k_rope`。
  - 修复后 `q_rope/k_rope/sdpa/attn_out/layer0_out` 全部 max diff `0.000000e+00`。

根因:

- `prism_infer/vision/mrope.py::apply_mrope` 在应用 RoPE 时把 `q/k/cos/sin` 转成 float32 运算，再 cast 回 bf16。
- HF 4.57.1 的 `Qwen3VLTextAttention` 使用 `apply_rotary_pos_emb`，源码位置:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:378-381`。
- HF 该函数直接在输入 dtype 上做:
  `q_embed = (q * cos) + (rotate_half(q) * sin)`，
  `k_embed = (k * cos) + (rotate_half(k) * sin)`。
- bf16 下，float32 中间计算再回 cast 会改变舍入路径，导致 layer 0 attention 首次出现小误差，并在 36 层 residual/MLP 中累积为 full logits `MARGINAL`。

修复:

- 修改 `prism_infer/vision/mrope.py::apply_mrope`，移除 RoPE 应用阶段的 `.float()` 中间计算，保持与 HF 相同的输入 dtype 运算顺序。
- 保留 `MRope.forward` 中 cos/sin 生成阶段的 float32 计算，因为 HF rotary embedding 在源码 `modeling_qwen3_vl.py:326-334` 也是禁用 autocast 后生成 float32 freqs，再 cast 到输入 dtype。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_attention_micro_debug.py
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  tests/test_mrope.py \
  tests/test_qwen3_vl.py
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python tests/test_full_model.py
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  tests/test_full_model_structure.py \
  tests/test_patch_embed.py \
  tests/test_vit_mlp.py \
  tests/test_mrope.py \
  tests/test_vit_attention.py \
  tests/test_vit_attention_rope.py \
  tests/test_vision_encoder.py \
  tests/test_qwen3_vl.py
```

验证结果:

- `compileall`: PASS。
- `tests/test_attention_micro_debug.py`: 修复后 `q_rope/k_rope/sdpa_gqa/attn_out/layer0_out` max diff 全部 `0.000000e+00`。
- `tests/test_mrope.py tests/test_qwen3_vl.py`: `6 passed in 74.11s`。
- `tests/test_full_model.py`: `Result: PASS`; logits max diff `0.000000e+00`, mean diff `0.000000e+00`。
- P1 模块对齐套件: `20 passed in 82.17s`。

经验:

- 当前问题不是权重缺失、shape mismatch 或 NaN。
- 分层激活显示第一处误差在 layer 0 attention，而不是 embedding 或 RoPE。
- 后续修复必须先复现并缩小 attention 子模块差异，不能直接修改后层 MLP 或 final norm。
- 对齐 HF 时，“数学上等价”的 dtype 提升不一定数值等价。bf16 模型对齐要求复现参考实现的 cast 和舍入顺序。
- 分层定位比直接看 logits 更有效: full logits max diff `3.125e-01` 的根因被缩小到 layer 0 RoPE 应用阶段。

剩余风险:

- 当前纯文本 full logits 已严格 PASS。
- 仍需在 P2 阶段验证图文输入、视觉 token 替换、DeepStack 注入和端到端 generate tokens。
