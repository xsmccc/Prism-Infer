# Prism-Infer 阶段交付模板

每个 P-stage或可独立验收的子阶段完成时复制本模板。不能填写的字段必须写
`未验证`、`不适用`或具体 blocker，禁止留空后仍标记 PASS。

## 1. 身份与状态

```text
阶段:
标题:
状态: PASS | PARTIAL | BLOCKED | REJECTED
负责人:
日期:
实现 commit:
证据 commit:
工作树: clean | dirty（dirty 只能作诊断）
```

## 2. 目标与需求映射

| Requirement / exit criterion | In scope | 实现或证据位置 | 结果 |
|---|:---:|---|---|
|  | yes/no |  | PASS/FAIL/未验证 |

明确写出 out-of-scope；不得把后续阶段能力隐含到本阶段 PASS中。

## 3. 改动范围

```text
新增:
修改:
删除:
兼容性:
数据/schema迁移:
```

对每个关键设计写明选择理由、被拒绝方案与失败边界。核心实现不得用第三方 wrapper
冒充自实现。

## 4. 验证环境

```text
模型与 revision/hash:
GPU name / UUID:
CPU / RAM（相关时）:
Python / Torch / CUDA / Transformers:
关键依赖 revision:
执行 backend:
KV / batch / context / sampling:
启动前 GPU memory/utilization:
随机种子:
```

性能证据必须来自 clean commit和未受其他 workload污染的设备；否则只能标为诊断。

## 5. Correctness 与质量

| Gate | Reference | Shape / workload | Metric | 阈值 | 实测 | 结果 |
|---|---|---|---|---:|---:|:---:|
|  |  |  | max/mean diff、token exact、PPL或 task metric |  |  |  |

至少包含：

- 模块 independent reference；
- full-model logits/PPL（模型路径改动时）；
- greedy token deterministic/exact；
- multi-shape或边界 case；
- 压缩策略的质量与物理 KV门禁（相关时）。

## 6. 性能与资源

```text
计时 scope:
warmup / repeat:
同步方式:
统计: median / p90 / p99 / min / max
显存: allocated / reserved / peak
before/after单变量:
profiler是否扰动:
```

| Workload | Baseline | Candidate | Ratio / delta | Correctness | Claim eligible |
|---|---:|---:|---:|:---:|:---:|
|  |  |  |  |  |  |

microbenchmark、offline TPOT、E2E、online goodput和容量必须分开命名，不能互相替代。

## 7. 执行命令与输出

```bash
# 可复制的最小 correctness命令

# focused regression

# formal benchmark / summary

# full regression
```

```text
测试总数:
passed / failed / skipped:
耗时:
JUnit或结构化 summary:
```

## 8. Claim ledger

### 可以使用

- 

### 必须带限制

- 

### 禁止使用

- 

同步更新 `docs/CLAIMS.md`；没有 raw evidence的数字不得进入 README或投递材料。

## 9. Raw evidence

```text
路径:
record schema/version:
SHA256或生成命令:
是否 gitignored:
```

保存 raw record、机器可读 summary和人类可读 summary；只保存截图不算可复现证据。

## 10. 风险、blocker 与回退

```text
已知限制:
未验证平台/shape:
外部 blocker及复现证据:
回退开关或 commit:
触发回退的条件:
```

## 11. 下一步与交接

```text
下一条最高优先级任务:
依赖:
恢复命令:
预期出口:
```

## 12. Gate Review

- [ ] requirement逐项映射，无遗漏或偷换范围。
- [ ] `git status --short`符合证据声明。
- [ ] correctness先于性能，失败候选未计时或未形成 claim。
- [ ] benchmark环境、输入、warmup/repeat和统计完整。
- [ ] README/ROADMAP/VERIFICATION/PERFORMANCE_REPORT/CLAIMS状态一致。
- [ ] Known Issues记录未完成项。
- [ ] raw evidence可由保存命令重新生成。
- [ ] full regression已执行，或明确标记阻塞原因与待补命令。
- [ ] PASS只在所有 in-scope出口满足时勾选。
