# P7-009: Projection fusion 的数值门禁与 gate/up packing

- 状态: `INVESTIGATING`（实现与 component correctness完成，full-engine/performance待干净 GPU）
- profiling commit: `0fdd4a6`
- implementation commit: `8767b7a`
- correctness evidence commit: `01b3625`
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
创建长期副本。理论 linear count从每层3降到2，整步 `253 -> 217`，但这只是源码
推导，必须由 clean node trace确认。

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

## 当前 GPU 阻塞

同一物理 UUID在三个重建容器中均保留不可见的 `17,282 MiB`占用与约
`0-35%`外部 SM activity；容器内无对应 CUDA PID，5090不支持 in-place reset。
benchmark formal gate要求启动显存 `<=1024 MiB`、utilization `<=5%`和 clean commit，
因此本轮正确输出 `formal_eligible=false`，没有保存 timing数字。

## 待完成门禁

1. 干净 GPU上运行 packed/legacy交替顺序 eager与 Graph microbenchmark。
2. 完整 8B single/multi-image/video HF teacher-forced logits/PPL。
3. offline五 workload、online arrival/SLO与长输出 greedy correctness。
4. 无 profiler TPOT/E2E，以及 node trace验证 linear `253 -> 217`和真实 kernel time。
5. 若无稳定 E2E收益，保留实现或回退必须依据同条件数据，而不是 micro speedup。

## 被拒绝的方法

- 直接合并 QKV并只检查最终一个 token：K/V内部已不 exact，会污染缓存和长输出。
- 在当前共享/污染 GPU报告 paired timing：交替顺序只能降低 drift，不能消除另一
  namespace workload的干扰。
- 把理论 kernel count下降写成实测 speedup：CUDA Graph已降低 host launch成本，
  GEMV memory/algorithm变化必须实测。

## 面试表达

> Trace显示七成 replay时间在 linear后，我没有直接把所有 projection都拼起来。
> QKV micro门禁发现 batch2以上 K/V会因 cuBLAS shape选择产生 BF16差异，所以在
> 计时前拒绝；gate/up则在 decode和代表性 prefill shape保持 bitwise exact，才进入
> 实现。平台 GPU被其他 namespace占用时，runner也会拒绝生成正式性能结论。
