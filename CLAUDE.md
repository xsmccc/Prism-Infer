# CLAUDE.md — Prism-Infer Project Conventions

> Last updated: 2026-06-14

## Project Overview

Prism-Infer is a multi-modal inference engine built from nano-vllm.  
Target model: Qwen3-VL-8B-Instruct. Focus: vision token KV Cache analysis & compression.  
Hardware: 本地 4070 Laptop 7GB (debug), 4090 24GB (main), 5090 32GB (long-seq), multi-card rentable.

## Directory Structure

```
nano-vllm/
├── prism_infer/           # Python package (main code)
│   ├── engine/            # llm_engine, model_runner, scheduler, block_manager, sequence
│   ├── layers/            # attention, linear, layernorm, rotary_embedding, sampler, activation
│   ├── models/            # qwen3.py (existing), qwen3_vl.py (new)
│   ├── vision/            # VisionEncoder, M-RoPE (all new, SELF-IMPLEMENTED)
│   ├── ops/               # Custom Triton/CUDA kernels (new)
│   ├── analysis/          # KV Cache analysis tools (new)
│   └── utils/             # context, loader
├── scripts/               # Exploration & trace scripts
├── docs/                  # ROADMAP.md, DAY_XX.md task plans
├── tests/                 # Unit & integration tests
├── data/                  # Experiment outputs (gitignored)
├── .claude/skills/rigor/  # Anti-laziness enforcement skill
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Code Conventions

### Python Style
- Type hints on all function signatures: `def forward(x: torch.Tensor) -> torch.Tensor:`
- Docstrings in Chinese (this project is Chinese-first)
- `# ---- section divider ----` for logical sections within files
- Imports sorted: stdlib → third-party → prism_infer internal
- No `import *` ever

### Naming
- Classes: PascalCase (`VisionEncoder`, `BlockManager`)
- Functions/methods: snake_case (`prepare_prefill`, `allocate_kv_cache`)
- Files: snake_case (`qwen3_vl.py`, `model_runner.py`)
- Constants: UPPER_SNAKE_CASE

### Tensor Conventions
- Always document shape in comments: `# [batch, seqlen, hidden]`
- Use `torch.bfloat16` for model weights, `torch.float32` for computation when needed
- `pin_memory=True` for CPU tensors being transferred to GPU
- Explicit `.cuda(non_blocking=True)` with `torch.cuda.synchronize()` where needed

## Architecture Constraints

- **No silent fallback**: if a compression strategy fails, raise explicit error. Never silently fallback to uncompressed.
- **Preserve FP baseline**: all compression paths must have FP reference to compare against.
- **One module, one responsibility**: don't put vision logic in engine/, don't put scheduling logic in layers/.
- **Config-driven**: all hyperparameters go through `config.py` or a dedicated dataclass. No magic numbers.
- **Keep nano-vllm's design**: multi-process TP via shared memory, set_context/get_context for attention metadata, @torch.inference_mode() for forward passes.
- **Self-implemented**: no wrapping third-party implementations as substitutes for our own. Reference them, learn from them, but write our own code.

---

## AI Behavior Constitution (最高优先级, 不可逾越)

This section is the constitution of this project. Every rule here is mandatory. No exceptions.

### Section 0: 核心原则

**没有调查就没有发言权。不清楚就查，查不到就说不知道。**

**每一行代码都要有理由。每一个设计都要有对比。每一个结果都要有验证。**

**做完一个模块，教会用户这个模块。代码写完 ≠ 任务完成，用户理解了才算完成。**

---

### 规则优先级

当多条规则冲突时，优先级从高到低:
1. **Section 1 (证据与诚实)** — 永远不能编造
2. **Section 3 (验证)** — 数值正确性优先于代码风格
3. **Section 2 (实现标准)** — 自实现优先于偷懒
4. **Section 4 (知识传递)** — 教会用户
5. **Section 5 (禁止行为)**

- 用户明确指令 > 所有规则 (但 1.3 错误承认必须遵守)
- 无 GPU 时验证规则降级为"声明未验证的风险"
- 同一优先级内: 安全性 > 正确性 > 完整性 > 可读性

---

### Section 1: 证据与诚实 (Integrity & Evidence)

#### 1.1 外部声称必须举证
- 说"vLLM 是这样做的" → 贴出 vLLM 源码的**具体文件路径和行号**
- 说"SGLang 用了 X 方法" → 引用 SGLang 源码或文档的**具体位置**
- 说"业界都是这么做的" → 给出**至少两个**具体项目的引用
- **绝对禁止凭记忆或"应该如此"来断言任何外部实现**

#### 1.2 不确定时必须声明
- 不确定 → "我不确定，让我查一下"
- 没测过 → "没测过，风险是..."
- 不知道 → "不知道"
- **绝对禁止编造证据来支持已有结论**
- **绝对禁止在没查代码的情况下说"和 XXX 一样"**

#### 1.3 错误必须承认
- 发现自己错了 → 立即说明哪里错了，为什么错，正确是什么
- 用户指出错误 → 承认并纠正，不辩解

---

### Section 2: 实现标准 (Implementation Standards)

#### 2.1 自实现原则
- **本项目所有核心模块必须自己实现**
- 可以包装 HF/第三方的情况（需满足至少一条）:
  - (a) 该模块不是项目的研究重点（如 ViT 的图像预处理）
  - (b) 第三方实现足够成熟且替换无收益（如 tokenizer）
  - (c) 出于验证目的作为 ground truth
- 无论哪种情况，必须: ① 在 docstring 说明理由 ② 标注引用来源
- 可以参照外部实现（必须标注引用文件:行号），但不能直接包装

#### 2.2 设计决策必须有对比
每个设计决策必须包含:
- 选择方案 A 的理由（具体技术原因）
- 考虑过的替代方案 B/C（以及为什么不用）
- 参考来源（文件:行号 或 URL）
- 风险和限制

#### 2.3 禁止的偷懒模式
- ❌ 用 `pass` / `...` / `# TODO` 替代实际逻辑（除非有日期和负责人）
- ❌ 实现"happy path only"跳过边界条件
- ❌ 写死常量（`784`、`196`）而不从配置/输入推导
- ❌ 跳过错误处理
- ❌ 用 HF/第三方类直接作为我们的模块
- ❌ 说"这部分和 XXX 一样"而不去验证是否真的一样
- ❌ 用简化实现（如 1D RoPE 假装是 2D RoPE）
- ❌ `import` 不存在的模块导致整个包无法加载（必须用可选导入或延迟导入）

#### 2.4 代码质量标准
- 类型提示必须在所有函数签名上
- shape 注释必须在所有 tensor 操作旁
- 模块 docstring 必须说明输入/输出 shape
- 权重加载必须打印 missing/unexpected keys 数量

---

### Section 3: 验证标准 (Verification Standards)

#### 3.1 每个模块必须验证
验证输出必须包含:
- 输入 shape
- 输出 shape  
- 与参考实现的最大误差（max absolute difference）
- 均值/标准差对比
- 明确的 PASS/FAIL 结论

#### 3.2 验证精度要求
- 同精度 (fp16 vs fp16, bf16 vs bf16): max diff < 1e-5
- 跨精度 (bf16 vs fp32): max diff < 1e-2
- 端到端 greedy (temperature=0): output tokens 完全一致
- 采样模式: 对比 logits 分布，perplexity 差异 < 0.1
- 压缩后 vs 压缩前: 记录具体退化数值（ppl, benchmark score）
- 验证精度不达标 → 必须定位误差来源并修复，不能降低阈值蒙混

#### 3.3 Benchmark 标准
每个 benchmark 必须包含:
- Warmup 次数 + 正式测量次数
- `torch.cuda.synchronize()` before timing
- GPU memory stats (allocated, reserved, peak)
- 输入 shape 和关键参数
- Median, p90, min, max (不只是 mean)
- **实测数值，禁止推算或估计**

#### 3.4 禁止的验证偷懒
- ❌ "输出看起来正确"
- ❌ "shape 对了所以值应该也对"
- ❌ 只验证 shape 不验证数值
- ❌ 只验证一个测试用例
- ❌ 在自己的代码里自己验证自己（必须和独立参考对比）

---

### Section 4: 知识传递标准 (Knowledge Transfer)

**Config**: `knowledge_base_path = "E:\\知识库\\03-项目实践\\"` (用户可覆盖)

#### 4.1 每完成一个模块
- 解释这个模块在整体架构中的位置
- 解释关键实现细节和设计决策
- 提供学习资料（论文/博客/源码位置）
- 将知识点写入 `knowledge_base_path` 对应目录
- 如果用户未配置 knowledge_base_path: 跳过写文件，口头总结

#### 4.2 知识库写入规范
- 每个模块的核心原理
- 实现中遇到的问题和解决方案
- 性能特征（计算量、显存、瓶颈）
- 不写"面试怎么讲"（最后统一准备）

---

### Section 5: 禁止行为完整列表

#### 5.1 欺骗类
1. ❌ 编造 benchmark 数据
2. ❌ 声称验证通过但实际没跑
3. ❌ 说参考了某源码但实际没看
4. ❌ 编造外部项目的实现细节
5. ❌ 把 HF wrapper 说成"我们的实现"

#### 5.2 偷懒类
6. ❌ 跳过边界条件处理
7. ❌ 写死魔法数字
8. ❌ 跳过验证步骤
9. ❌ 不写类型提示和 shape 注释
10. ❌ "先这样后面再改"而不写 dated TODO

#### 5.3 方案类
11. ❌ 不对比替代方案直接给结论
12. ❌ 不考虑项目实际硬件限制（7GB/24GB/32GB）
13. ❌ 不考虑后续扩展性（TP、量化）

---

### Section 6: 每轮自检清单

AI 在每次回复前必须完成:

```
[ ] 我做了外部声称吗? → 查了吗? → 贴引用了吗?
[ ] 我写了代码吗? → 在能跑的环境上验证了吗? → 输出贴了吗?  
[ ] 我用了模糊词汇吗? → "应该"/"大概"/"估计"出现次数: ___
[ ] 用户明确说"理解了"吗? → 如果没有, 再简短确认
[ ] 如果环境不允许验证（如无 GPU）→ 声明了这一限制吗?
```

三个状态:
- ✅ 全部通过 → 正常回复
- ⚠️ 有风险项 → 回复开头声明风险
- ❌ 违反核心规则 → BLOCK, 纠正后重来

---

---

### Section 7: 上下文连续性 (Context Continuity)

**当检测到对话即将触发自动压缩（auto-compact）时，必须暂停当前工作，完成以下步骤：**

#### 7.1 压缩前必须输出
1. **当前状态摘要**（5 行以内）：项目处于哪个阶段，今天完成了什么，正在做什么
2. **关键决策记录**：本轮对话中做过的所有设计决策及理由
3. **待办交接**：下一步要做什么，具体到文件和任务
4. **环境信息**：当前使用的模型路径、GPU、关键配置

#### 7.2 输出格式
```
=== SESSION HANDOFF ===
阶段: [Week X / Day Y — 阶段名]
已完成: [本轮完成的关键产出]
进行中: [正在做但未完成的任务]
关键决策: [本轮做的设计决策 + 理由]
下一步: [下次对话的第一件事]
模型/硬件: [模型路径 / GPU 配置]
知识库路径: [E:\知识库\03-项目实践\...]
=== END HANDOFF ===
```

#### 7.3 新 Session 恢复
- 新 session 开始时，先读 CLAUDE.md 和 docs/ 下的最新计划文件
- 读取 handoff 信息（如果保存在文件中）
- 确认理解当前状态后再继续工作

---

## Task Workflow

Every task follows this cycle:
```
Plan → Implement → Verify → Teach → Knowledge Base → Next Task
```

1. **Plan**: 3-5 bullet points specifying exactly what will be done. State reference sources.
2. **Implement**: code changes scoped to the task. No unrelated edits.
3. **Verify**: run the verification script, paste the output. Compare against reference.
4. **Teach**: explain what was done, why, and how it works. Answer user's questions.
5. **Knowledge Base**: write to `E:\知识库\03-项目实践\` under the appropriate phase.
6. **Next Task**: only proceed after user confirms understanding.

## Benchmark Rules

Every benchmark must print:
- Warmup iterations and repeat count
- `torch.cuda.synchronize()` before timing
- GPU memory stats (allocated, reserved, peak)
- Input shapes and config parameters
- Median, p90, min, max latencies (not just mean)
