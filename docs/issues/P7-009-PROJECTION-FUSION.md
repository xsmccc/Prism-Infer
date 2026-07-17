# P7-009: Projection fusion 的数值门禁与 gate/up packing

- 状态: `RESOLVED`（QKV correctness-first拒绝；gate/up完成full-engine动态闭环并保留）
- profiling commit: `0fdd4a6`
- implementation commit: `8767b7a`
- correctness evidence commit: `01b3625`
- full-engine A/B commit: `8293851`
- online/Systems/final gate commit: `021d4e2`
- 硬件/软件: RTX 5090，Torch `2.6.0a0+nv25.01`，BF16
- 影响: Graph replay的 linear/GEMV占 `70.55%`；错误 fusion会改变 K/V数值，
  未经 full-engine门禁不能进入性能 claim。

## 现象

P7.4-B single-image/output32 trace中，每个 Graph replay有 `253` 个 linear/GEMV
kernels，kernel busy `9.123 ms`，占 replay `70.55%`。源码静态映射为 36 层的：

- attention `q/k/v/o`: `36 * 4 = 144`；
- MLP `gate/up/down`: `36 * 3 = 108`；
- 其余 Graph内 linear: `1`。

这使“合并共享输入的 projection”成为比继续微调 logits或 paged attention更直接的
P7.5候选，但它必须先通过 BF16数值门禁。

## 如何发现

Systems trace先给出 kernel count/time partition，再用单层真实 Qwen shape probe分别
检查 QKV和 gate/up。probe只使用数百 MiB，不加载完整 8B模型，因此可在剩余
`14,869 MiB`下执行；污染环境不影响 correctness判断，但禁止 timing claim。

## 候选一：QKV packing

`benchmarks/probe_p7_qkv_fusion.py` 使用同一输入与三块权重，对比三次
`F.linear`和一次 concatenated-weight `F.linear`后 split。clean `01b3625`结果：

| Batch | Q exact | K exact / max diff | V exact / max diff | 结论 |
|---:|:---:|---:|---:|---|
| 1 | yes | yes / `0` | yes / `0` | 当前 shape通过 |
| 2 | yes | no / `1.0` | no / `1.0` | reject |
| 4 | yes | no / `1.0` | no / `1.0` | reject |
| 8 | yes | no / `1.0` | no / `1.0` | reject |

K/V平均绝对差约 `0.079-0.089`。packed output size改变 cuBLAS算法/舍入；数学等价
不代表 BF16 bitwise等价。由于 K/V会进入 KV cache并影响后续所有 token，该候选在
计时前拒绝，不能通过放宽最终 token门禁掩盖内部差异。

## 候选二：Gate/up packing

gate与 up具有相同 `12288 x 4096` shape。实现使用一段
`24576 x 4096`连续 Parameter作为执行 storage，旧 `gate_proj/up_proj`参数是两段
row view；state-dict隐藏 packed内部 key，只保留 HF-compatible三键。forward执行
一次 packed projection、chunk view、SiLU/mul和原 down projection。

该布局不会复制权重；`Module.to/_apply`后显式 rebind view，防止 device/dtype转换
创建长期副本。显式`legacy|packed`执行模式共享同一storage/state-dict，用于同commit
单变量A/B。clean node trace确认linear count从每层3降到2，整步`253 -> 217`。

## 验证

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  tests/test_p7_packed_mlp.py \
  tests/test_qwen3_vl.py \
  tests/test_full_model_structure.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_model_runner_vl_prefill.py
# 32 passed in 62.19s
```

真实 Qwen shape correctness-only matrix覆盖 batch/rows
`1/2/4/8/210/408/988`，packed/legacy完整 MLP output逐元素 exact，max/mean diff
均为 `0`。HF legacy state-dict strict load、storage alias和 dtype conversion rebind均有
单元合同。

## Full-engine关闭证据

历史不可见GPU占用恢复后，formal baseline稳定为`1–4 MiB / 0–2%`。关闭结果：

- clean `396702d` formal micro七个rows全部bitwise exact；
- clean `8293851/021d4e2`八个offline cells全部token exact，packed decode TPOT改善
  `0.483%–0.762%`；
- single/multi-image/video 32-token HF model-precision logits/PPL diff均为`0`；
- single-image与mixed-rate10 online A/B token exact、双方SLO goodput fraction均`1.0`；
- 31-step node trace：all kernels `2,000 -> 1,964`、linear `253 -> 217`、kernel busy
  `12.815 -> 12.721 ms`；
- clean full regression `281 passed, 6 skipped`，0 failure/error。

最终保留packed默认。E2E包含已知vision prefill双峰，online没有process-level repeats，
所以只声明记录workload的unprofiled decode TPOT小幅收益，不声明稳定E2E或online加速。

## 被拒绝的方法

- 直接合并 QKV并只检查最终一个 token：K/V内部已不 exact，会污染缓存和长输出。
- 在共享/污染 GPU报告 paired timing：交替顺序只能降低 drift，不能消除另一
  namespace workload的干扰；正式结果只使用恢复后的clean baseline。
- 把理论 kernel count下降写成实测 speedup：CUDA Graph已降低 host launch成本，
  GEMV memory/algorithm变化必须实测。

## 面试表达

> Trace显示七成 replay时间在 linear后，我没有直接把所有 projection都拼起来。
> QKV micro门禁发现 batch2以上 K/V会因 cuBLAS shape选择产生 BF16差异，所以在
> 计时前拒绝；gate/up通过组件、HF、E2E与online门禁后，trace确认每步少36个linear，
> 8个clean cell的decode TPOT改善约0.48%–0.76%。收益很小，所以没有包装成E2E加速。
