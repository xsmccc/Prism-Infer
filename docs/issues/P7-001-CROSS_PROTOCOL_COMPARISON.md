# P7-001: 跨执行协议的性能数字不能直接比较

- 状态: `RESOLVED`
- 首次修复基线: P7.1 benchmark schema v2
- 影响: Prism 与 vLLM/SGLang 外部性能结论

## 现象

P6.7 的外部基线固定为 eager，而 P6.11/P6.12 后 Prism 已支持 compressed CUDA
Graph。直接把 P6.7 vLLM eager `15-16 ms` 和后来的 Prism Graph `17-19 ms`
并列，会混入不同 commit、warmup/repeat、执行后端和时间点，不能计算 ratio。

反过来，只让 Prism 使用 Graph、让 vLLM 保持 eager，也不能回答“双方最优稳定
配置谁更快”。

## 如何发现

审计 `PERFORMANCE_REPORT.md` 中每个数字的来源，而不是只比较表格数值：

- P6.7: external eager，Prism eager。
- P6.11: Prism eager/Graph internal ablation，没有 external rerun。

这说明缺失的是实验协议，不是一个新的算子实现。

## 根因

external schema v1 只用 prompt token 是否相同决定 `performance_comparable`。
它没有阻止以下错误组合：

- eager 对 Graph。
- dirty harness 对 clean Prism。
- 不同 warmup/repeat。
- 不同 KV pool、block size、sampling 或计时 scope。
- 相同 GPU 型号但实际不是同一设备。

## 解决方案

P7.1 建立两条互斥 profile：

1. `diagnostic_matched`: Prism eager 对 vLLM eager，用于定位 framework/kernel
   overhead。
2. `best_stable`: Prism CUDA Graph 对 vLLM 有效 CUDA Graph，用于当前最优稳定
   offline 系统对比。

schema v2 将 model config hash、GPU UUID、KV pool bytes、block size、sampling、
warmup/repeat、source/harness dirty state 和有效 cudagraph mode写入 record。
汇总器只有在全部 comparability checks 通过时才生成 TPOT/throughput ratio；失败
时列出具体 gate 名称。

## 为什么有效

将“可比性”变成机器校验后，文档作者无法通过选取两条看似相近的 JSON 手工
制造 speedup。诊断实验和最终系统实验各回答一个问题，也避免把 vLLM 的通用
优化关掉后声称战胜其生产路径。

## 验证

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_external_comparison.py tests/test_benchmark_schema.py -s
```

focused tests 覆盖：完整 v2 gate PASS、dirty harness 拒绝、Graph/eager profile
错配拒绝，以及 schema-v1 历史记录兼容。

## 被拒绝的方法

- 只在文档旁注“配置不同”：仍允许汇总器输出误导 ratio。
- 删除 P6 eager 数据：历史数据对定位 launch overhead 仍有价值。
- 强迫双方使用相同 attention kernel：系统对比应允许各框架使用自己的稳定
  backend，只需另做 matched diagnostic。

## 面试表达

> 我发现性能比较最大的风险不是计时 API，而是实验协议漂移，所以把公平性
> 条件编码进 schema。任何一项不匹配，工具会拒绝生成 speedup，并告诉我具体
> 是 KV pool、执行后端还是 dirty state 不一致。
