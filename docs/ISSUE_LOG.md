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
错误假设与排除过程:
根因:
修复:
设计权衡与拒绝方案:
验证命令:
验证结果:
经验:
面试故事提炼:
剩余风险:
```

## P9-001: 当前 GPU0 物理身份变化，历史性能数据不可直接混入新基线

状态: Verified

发现方式:

- 在 clean `00982ec` 上启动 P9-C.3/P9-D release baseline 前执行 GPU identity gate。

影响范围:

- 所有要求“同一 GPU UUID”的 Prism/vLLM/SGLang 性能比例与 process-level repeats。
- 不影响旧数据在其原始硬件上的历史有效性，也不影响当前 GPU 的 correctness 测试。

证据:

```text
历史 P9-A/P8 formal GPU UUID:
GPU-989db6f6-3273-d1dd-b2b9-56cced4f30a4

2026-07-20 当前唯一可见 GPU0:
GPU-662a2fa1-37e4-cc52-0a51-27557dba315b
NVIDIA GeForce RTX 5090, 1 MiB used, 32149 MiB free, 0% utilization

environment check:
status=PASS, CUDA 12.8, compute capability 12.0,
free/total=30.901/31.396 GiB, model revision and 4 weight files PASS
```

定位过程:

- 用 `nvidia-smi -L` 确认当前容器只暴露一张 GPU，而不是 CUDA ordinal 重排后的多卡视图。
- 用 `nvidia-smi topo -m` 确认当前只有 GPU0。
- 搜索正式文档中的旧 UUID，确认 P8/P9-A page/NCU 数据绑定另一物理设备。
- 重新运行 `scripts/check_environment.py`，确认当前设备、CUDA backend 和模型文件本身可用。

错误假设与排除过程:

- 不能因为设备名同为 RTX 5090、ordinal 同为 `0`，就假设是同一张物理卡。
- 不能把 UUID 变化归因于 Prism、CUDA 或驱动；当前证据只能确认租赁/容器边界暴露了
  不同物理设备，无法观察云平台的具体重新分配机制。

根因:

- 当前运行环境对应的物理 GPU 与历史 formal 环境不同；更底层的资源重新分配原因不可见。

修复:

- 历史结果继续绑定旧 UUID，不重写历史记录。
- 从 `00982ec` 开始建立新的 clean baseline；同一个性能 claim 的 baseline/candidate、
  Prism/external 和全部 repeats 必须在当前 UUID 上完成。
- benchmark schema 继续保存 UUID，并拒绝跨 UUID 自动聚合。

设计权衡与拒绝方案:

- 不用“同型号 GPU”放宽成可直接计算 speedup ratio；RTX 5090 个体、功耗状态、拓扑和
  云平台策略仍可能造成系统差异。
- 不删除旧 P9-A 证据；它仍可用于解释设计发现，但不能和新设备数据拼成正式比值。

验证命令:

```bash
nvidia-smi -L
nvidia-smi topo -m
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python scripts/check_environment.py \
  --model /data/models/Qwen3-VL-8B-Instruct/\
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --require-cuda --min-free-gib 18
```

验证结果:

- 当前 GPU identity、空闲资源、runtime capability 与模型完整性门禁 PASS。
- 跨 UUID comparability 明确判定为不允许；新 formal matrix 尚未开始。

经验:

- GPU ordinal 是进程局部编号，UUID 才是性能证据的物理身份。
- 每次云环境重启后，即使设备型号、显存和软件栈都相同，也必须先重建 identity gate。

面试故事提炼:

- “我在复现实验前发现同为 GPU0 的 RTX 5090 UUID 已变化，因此没有复用旧 baseline。
  我把 GPU UUID 写进 schema 和聚合 comparability gate，避免跨物理设备制造虚假收益。”

剩余风险:

- 当前设备无法锁定 GPU clocks；正式 benchmark 仍需空闲门禁、ABBA/BAAB 顺序和
  process-level 置信区间控制漂移。

## P9-002: Full-model 释放读数被测试端 tensor ownership 与异步执行污染

状态: Verified

发现方式:

- 在当前 GPU 上重跑纯文本 full-model correctness gate，检查测试打印的进程内
  CUDA allocator 释放证据。

影响范围:

- `tests/test_full_model*.py` 的“模型已释放”结论和串行加载内存安全性。
- 不影响已生成 logits 的数值正确性，也没有证据表明 production Engine 存在 ownership
  泄漏。

证据:

```text
修复前:
HF 已释放:          0.0 GiB allocated
Prism 加载完成:    16.4 GiB allocated
Prism 已释放:      16.4 GiB allocated
进程退出后 NVML:      1 MiB used

仅删除 our_sd 后:
Prism 已释放:       1.2 GiB allocated

最终修复后:
HF 已释放:          0.0 GiB allocated
Prism 已释放:       0.0 GiB allocated
进程退出后 NVML:      1 MiB used, 0% utilization
HF/Prism logits: max diff 0, mean diff 0, Result PASS
```

定位过程:

- 先比较进程内 `torch.cuda.memory_allocated()` 与进程退出后的 NVML；后者恢复到 1 MiB，
  因而现象不支持“跨进程不可见占用”或驱动级常驻。
- 检查测试对象生命周期，发现 `our_sd = our.state_dict()` 在删除 `our` 后仍然存活。
  `state_dict()` 是持有参数/缓冲区 tensor 的浅映射，不是与模型 storage 无关的计数快照。
- 保存参数数量后显式删除 `our_sd`，释放读数从 16.4 GiB 降到 1.2 GiB，验证参数 storage
  的第二所有者是主因。
- 剩余读数发生在对比 logits 仍位于 GPU 且 CUDA 工作未显式同步的观测点。测试改为先把
  correctness 输出复制到 CPU，并在读取释放状态前同步；读数降到 0.0 GiB。由于这两项
  同时修复，不进一步虚构 1.2 GiB 在二者之间的精确拆分。

错误假设与排除过程:

- 错误假设一：`del our` 足以释放全部参数。排除依据是删除 `our_sd` 后立即少了约
  15.2 GiB；Python 对象图中还有 storage owner。
- 错误假设二：`torch.cuda.empty_cache()` 会释放所有 GPU 内存。它只能把 allocator 中
  已无活跃 tensor 的缓存块交还驱动，不能销毁仍被 Python 引用或尚待流完成的 allocation。
- 错误假设三：一条 allocator 读数就能证明 Engine 泄漏。进程内 allocated、reserved、
  异步 stream 状态和 NVML process bytes 的语义不同，必须结合对象生命周期判断。

根因:

- 测试 harness 长时间保留 `state_dict()`，使其参数 tensor 在模型对象删除后继续拥有
  GPU storage。
- correctness 输出仍驻留 GPU，且释放打印前缺少明确同步，使第二阶段读数继续含混。

修复:

- 在四个 full-model 脚本中先保存 `expected_parameters = len(our_sd)`，完成加载后显式
  `del our_sd`。
- 将用于比较的 logits 在模型释放前复制到 CPU，严格保证下一份 8B 模型加载时不保留
  上一模型的 GPU 输出。
- 在删除模型对象后显式 `torch.cuda.synchronize()`，再执行 GC、`empty_cache()` 和
  allocator 读数。

设计权衡与拒绝方案:

- correctness gate 接受一次 device-to-host copy 和同步，因为目标是确定性对齐与可靠
  ownership 证据；这些操作不得进入 latency benchmark 或 production fast path。
- 拒绝为测试误报增加 Engine destructor、全局 cache 清理或每步同步；那会污染真实
  pipeline，并掩盖测试端所有权错误。
- 没有把 1.2 GiB 全部归因于 logits 或异步 delayed free；现有实验只证明组合修复后的
  结果，未提供单因素精确拆分。

验证命令:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/\
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
env -C /data/Prism-Infer .venv-local/bin/python tests/test_full_model.py

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
```

验证结果:

- 纯文本 full-model 权重 `750/750` 加载，missing/unexpected 均为空。
- HF 与 Prism full logits shape 均为 `[1, 64, 151936]`，NaN 为零，max/mean diff 均为零。
- 两个模型函数内释放后均显示 `0.0 GiB` allocated，进程退出后 GPU 为 1 MiB used。
- 单图、多图和视频最后 token logits 均与 HF bit-exact，模型函数内释放后均为
  `0.0 GiB` allocated。
- CUDA Graph decode 的单图、多图、视频和 mixed batch 与 eager token exact；测试进程
  退出后 NVML 为 1 MiB used。

经验:

- `state_dict()` 的生命周期也是模型显存 ownership 的一部分；诊断代码本身可以制造
  看似 production 的泄漏。
- 显存排障要至少区分 live tensor bytes、allocator reserved bytes、异步 delayed free
  和 NVML process bytes，并检查进程退出边界。
- correctness、memory forensics 与 performance benchmark 需要不同的同步策略，不能把
  诊断同步混入被测 fast path。

面试故事提炼:

- “我发现模型删除后 allocator 仍显示 16.4 GiB，但进程退出后 NVML 归零。我没有修改
  引擎，而是追踪 Python tensor ownership，定位到测试保留的 `state_dict()` 浅映射；
  删除它后只剩 1.2 GiB，再通过 CPU 化输出和同步消除观测污染。最终 logits bit-exact，
  函数内 allocated 归零，也避免把测试 harness 的 bug 误修进 production runtime。”

剩余风险:

- 后续 release gate 应同时结构化记录 allocated/reserved/NVML，而不是依赖一位小数的
  console 文本。

## P9-003: Transformers 5.13 要求 HF 多模态参考 forward 显式传 modality ids

状态: Verified

发现方式:

- 完成 P9-002 后重跑单图 full-model reference gate，HF 5.13 在模型 forward 前抛出
  `mm_token_type_ids is missing`。

影响范围:

- 影响由 Prism `ImageInputs`/`VideoInputs` 重建参数并调用 HF 的 correctness tests。
- 不影响 Prism production runtime；Prism 在 host preprocessing 阶段自行生成并冻结
  M-RoPE `position_ids`，不依赖 HF model forward。

证据:

```text
Transformers: 5.13.0
input_ids: [1, 210]
image_grid_thw: [1, 3]

ValueError: Multimodal data was passed (via `image_grid_thw` or
`video_grid_thw`) but `mm_token_type_ids` is missing.
```

定位过程:

- 检查当前 HF `Qwen3VLModel.compute_3d_position_ids`，确认只要传入视觉 grid 和
  `input_ids`，新版 forward 就要求 modality ids。
- 检查 processor 输出与仓库测试辅助代码，发现 `tests/conftest.py` 已有根据展开后的
  image/video pad token 构造 modality ids 的兼容逻辑，但 full-model 脚本未复用。
- 运行单图、多图、视频 M-RoPE 对照，Prism position ids 与当前 HF
  `get_rope_index` max diff、rope delta diff 均为零，排除 production M-RoPE 合同缺失。

错误假设与排除过程:

- 没有因为 HF 报“字段缺失”就把 `mm_token_type_ids` 加进 production sequence/device
  contract；该字段对 Prism 是可由 token ids 推导的 HF reference 输入。
- 没有固定向所有 HF 版本无条件传参；旧版本 forward 不一定接受该字段，因此必须检查
  安装版本的函数签名。

根因:

- 依赖升级改变了 HF reference forward 的显式参数合同；full-model 测试仍按旧版本调用。
- 同一兼容逻辑此前只在 RoPE 和 logits-distribution 测试中局部存在，缺少共享入口。

修复:

- 在 `tests/conftest.py` 增加共享 `with_hf_mm_token_type_ids`：只在当前 HF forward
  签名支持该字段时，从 image/video pad token 构造并附加 modality ids。
- 单图、多图、视频 full-model 脚本统一复用该 helper。
- 删除 `test_vl_logits_distribution.py` 的重复私有实现，避免兼容策略继续分叉。

设计权衡与拒绝方案:

- 兼容 helper 只位于 HF reference boundary，不污染 Prism runtime 数据结构。
- 拒绝 pin 回旧 Transformers 来隐藏问题；release 应能明确知道当前 reference API 的
  语义变化，但正式性能数据仍必须绑定精确依赖版本。
- 拒绝用 `**batch` 绕过 Prism 输入合同进行 full-model 测试；测试必须验证 Prism
  preprocessing 保留下来的数据足以重建正确 reference。

验证命令:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/\
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
env -C /data/Prism-Infer .venv-local/bin/python -m pytest -q \
  tests/test_vl_rope_index.py \
  tests/test_vl_rope_index_multi_image.py \
  tests/test_vl_rope_index_video.py -s

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/\
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
env -C /data/Prism-Infer .venv-local/bin/python tests/test_full_model_vl.py
```

验证结果:

- M-RoPE suite: `7 passed`，单图、多图、视频 position ids 与 rope delta exact。
- 单图 HF/Prism logits shape `[1, 151936]`，max/mean diff 均为零。
- 同一 helper 继续通过多图和视频 full-model reference forward。
- 32-token teacher-forced 单图、多图、视频分布测试 PASS；model-precision logits 和
  perplexity 三类输入均 exact。

经验:

- 外部 reference API 不是 production contract；升级依赖时必须先判断变化属于哪一侧。
- 版本兼容判断应集中在 boundary adapter，不能散落为测试中的条件分支。
- “能从已有语义无损推导的 reference-only 字段”通常不应扩大核心 runtime contract。

面试故事提炼:

- “升级 Transformers 后，HF 多模态 forward 强制要求 `mm_token_type_ids`。我先证明
  Prism 自己的 M-RoPE 与新版 HF exact，再把问题限定为 reference adapter 兼容性，
  用签名感知 helper 从视觉 pad token 重建字段，而没有把 HF 私有参数扩散进 runtime。”

剩余风险:

- 后续 Transformers 升级仍需跑 reference-signature、processor 和 M-RoPE 三层门禁；
  不能仅依赖版本号判断兼容性。

## P9-004: Vision FlashAttention 由包存在性隐式启用，造成 shape-dependent 数值漂移

状态: Verified

发现方式:

- 修复 P9-003 后，单图 full logits exact，但同权重、同预处理的双图 full-model gate
  出现显著 BF16 logits 差异。

影响范围:

- 安装 flash-attn 的环境中，`cu_seqlens` 多于一个 segment 的多图和视频 vision path。
- 单图默认走 SDPA，因此原有单图 strict gate 无法发现该环境和 shape 相关行为。
- 影响可复现性、backend 归因和正式性能/质量 claim；不代表 FlashAttention 算法语义错误。

证据:

```text
隐式 FlashAttention:
single-image max/mean logits diff: 0 / 0
multi-image max logits diff:       0.484375
multi-image mean logits diff:      0.08838902

单因素关闭 vision FlashAttention:
multi-image max/mean logits diff:  0 / 0

显式 SDPA 修复后:
multi-image max/mean logits diff:  0 / 0
video max/mean logits diff:        0 / 0
```

定位过程:

- 先运行 processor 与 M-RoPE 对照；多图 token count、grid、position ids 和 rope delta
  全部 exact，排除输入和位置编码。
- 复查历史 P3-001 的跨图 attention 修复，确认当前 SDPA 路径仍按 `cu_seqlens` 分段，
  没有恢复错误的跨图 attention。
- 检查 `ViTAttention.forward`，发现只有多 segment 且检测到 flash-attn 包时才自动调用
  varlen FlashAttention；HF 当前 reference 配置走分段 SDPA。
- 保持输入、权重、dtype 和软件栈不变，仅在 fresh process 中关闭该 capability flag，
  多图 logits 立即恢复 bit-exact，完成单因素归因。

错误假设与排除过程:

- 最初候选一是 Transformers 5.13 新 M-RoPE 语义；position/delta exact 将其排除。
- 候选二是 P3-001 的跨图 attention 边界回归；关闭 FlashAttention 后不改边界即可 exact，
  将其排除。
- 不把 micro-kernel 的小误差等同于 full-model 可忽略：27 层 ViT、DeepStack 和 36 层
  LLM 会放大 BF16 舍入差异，因此必须看端到端 logits/质量。

根因:

- 可选依赖的“能力探测”被直接当成 backend “选择策略”。同一公开配置会随环境是否安装
  flash-attn、以及输入是否包含多个视觉 segment 而静默改变执行后端。
- Prism varlen FlashAttention 与 HF 分段 SDPA 的 BF16 reduction/kernel shape 不同；
  micro 输出接近，但误差经完整多模态模型传播后不再满足 strict logits gate。

修复:

- 新增 `VisionAttentionBackendName`，只支持显式 `sdpa` / `flash_attn`，拒绝隐式
  `auto`。
- `MultimodalConfig` 默认固定为 `sdpa`，flat/nested config 使用同一类型化字段
  `vision_attention_backend`。
- 从 Config、ModelRunner、Qwen3VL model 到 VisionEncoder 全链路显式传递 backend。
- 请求 `flash_attn` 但依赖、device、dtype 或 packed shape 不满足时 fail closed，不静默
  回退到 SDPA。
- 重写 vision micro parity 测试，显式构造同权重 SDPA/FlashAttention 两个候选，而不是
  monkeypatch 包存在性来选择策略。

设计权衡与拒绝方案:

- 没有删除 FlashAttention；它仍是 P9-D 可测候选，但必须单独报告 backend、质量和性能。
- 默认 SDPA 优先维持 HF strict reference；只有端到端质量和同卡性能证据都成立，才考虑
  更改 release policy。
- 拒绝保留 `auto` 再在日志里事后猜 backend；startup-selected backend 是实验身份的一部分。
- 拒绝简单放宽多图 logits 阈值，因为 observed drift 远高于单 kernel parity，且当时没有
  下游质量集证据支持。

验证命令:

```bash
env -C /data/Prism-Infer .venv-local/bin/python -m pytest -q \
  tests/test_p9_architecture_contracts.py::\
test_nested_config_and_flat_compatibility_adapter_are_equivalent \
  tests/test_p9_architecture_contracts.py::\
test_vision_attention_backend_rejects_implicit_auto_policy \
  tests/test_vision_encoder.py::\
test_vision_attention_backend_is_explicit_and_fails_closed -s

CUDA_VISIBLE_DEVICES=0 env -C /data/Prism-Infer \
  .venv-local/bin/python -m pytest -q \
  tests/test_vision_encoder.py::\
test_vision_varlen_flash_attention_matches_segmented_reference -s

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/\
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
env -C /data/Prism-Infer .venv-local/bin/python \
  tests/test_full_model_vl_multi_image.py
```

验证结果:

- backend/config focused CPU tests: `4 passed`。
- explicit varlen FlashAttention vs segmented SDPA micro parity: PASS，max diff `<= 0.01`、
  mean diff `<= 0.001`。
- 默认 SDPA 下单图、多图、视频 HF/Prism logits 全部 bit-exact，释放后 allocated 为零。
- CUDA Graph decode: 单图 `[785, 2168]`、多图 `[785, 1378]`、视频
  `[1986, 2766]` eager/graph exact；mixed batch 三行 exact，`2 passed`。
- CUDA Graph 进程退出后 GPU 为 1 MiB used、0% utilization。
- 32-token teacher-forced distribution: `1 passed`；单图、多图、视频 model-precision
  logits max/mean diff 与 perplexity diff 均为零，fp32 LM-head 模式通过既有容差。

经验:

- capability detection 只能回答“能不能调用”，不能回答“应该调用哪个”；backend policy
  必须显式、可序列化、可进入 benchmark schema。
- 单图 correctness 不能覆盖多图/视频的 packed-varlen 分支，shape matrix 是 runtime 测试
  的一部分。
- micro parity 不能替代 full-model quality；小的 BF16 kernel 误差可能经深层网络放大。

面试故事提炼:

- “我遇到单图 exact、多图却漂移 0.48 的问题。通过 M-RoPE exact 和单因素禁用实验，
  定位到代码把 flash-attn 是否安装误当成 backend policy，导致多 segment 输入静默换
  kernel。我把 vision backend 做成类型化 startup 配置、默认 SDPA、缺能力 fail closed，
  同时保留 FlashAttention 为独立性能候选；最终 full logits 和 CUDA Graph mixed batch
  恢复 exact。”

剩余风险:

- P9-D 需在同一 GPU UUID 上测量 SDPA/FlashAttention 的 vision latency、TTFT、峰值显存
  和真实多图/视频质量，再决定 release 推荐值。
- 正式 artifact schema 需要记录 `vision_attention_backend`，禁止不同 backend 聚合。

## P9-005: 进程内多 mode benchmark 不能证明 cold/fresh-process 性能

状态: Verified

发现方式:

- 在当前 GPU UUID 上准备 P9-D eager/CUDA Graph 基线时，复查 `bench_system.py` 的执行
  生命周期和旧 artifact 的 `process_scope`。

影响范围:

- torch.compile cold peak、CUDA Graph capture ownership、allocator reserved memory、全局
  CUDA context/cache 和 eager/Graph TPOT 对比。
- 不否定旧 runner 的进程内功能回归价值；但它不能单独支持“5 次 fresh-process repeats”
  或 process-level 95% CI claim。

证据:

```text
旧 runner:
- 模块顶层 import torch
- 同一个 Python 进程内依次 for mode in modes
- 每个 mode 会重建/exit LLM，但不会重建 Python/CUDA process
- artifact protocol.process_scope = fresh_model_per_case_and_mode

新 orchestrator smoke（dirty diagnostic，不能形成性能 claim）:
- parent import 后 sys.modules 不含 torch
- off_eager/off_graph 各一个独立 child process
- 两次 child 前后均为 GPU 1 MiB used / 0% utilization
- 15/15 comparability checks PASS
- token IDs / decoded text / KV layout exact
- git_dirty=true、repeats=1、warmup=0，因此 formal_eligible=false
```

定位过程:

- 先区分“fresh model”与“fresh process”：`llm.exit()`、GC 和 `empty_cache()` 可以释放
  Engine ownership，但 CUDA context、模块级 cache、compiler cache 和 Python import 状态仍
  属于同一进程。
- 检查旧 runner 的 mode 循环，确认所有 mode 共用一个已导入 torch 的父进程，无法在
  mode 之间执行可靠的外部 NVML idle/release gate。
- 审阅首版 orchestrator 后继续发现三个证据链缺口：只比较 output hash、没有完整
  comparability gate；只报 process median、没有 bootstrap CI；输出若位于仓库中非
  ignored 路径，会在首个 child 前后改变 Git dirty identity。
- 用两 child smoke 验证进程边界、GPU 释放和完整字段 gate；用纯 CPU 测试验证非 ignored
  路径拒绝、ABBA/BAAB 截断顺序和 deterministic bootstrap。

错误假设与排除过程:

- 错误假设一：`llm.exit() + empty_cache()` 等价于 fresh process。它只处理可释放的
  runtime/allocator ownership，不能重置 CUDA context 和全部全局 cache。
- 错误假设二：token hash 相同就足以比较性能。即使输出相同，model config、prompt
  token、KV layout、vision backend、软件版本或 batch contract 不同，ratio 仍不公平。
- 错误假设三：先检查 clean Git，随后把结果写到任意仓库路径也安全。非 ignored 输出会
  让后续 child 看到不同的 `git_dirty`，污染同一个 matrix。
- 错误假设四：把所有样本放进一个进程做普通 median 就满足“重复实验”。那只能反映
  warm process 内部波动，不能覆盖 process-level cold/capture/ownership 变化。

根因:

- 旧 benchmark 的生命周期合同是“每 mode 新模型”，而 P9-D 需要“每 mode/repeat 新
  进程”；二者在 CUDA runtime 和统计抽样层级上不是同一个实验。
- runner schema 此前记录执行结果，但没有负责跨进程编排、外部 GPU gate 和 process-level
  聚合。

修复:

- 新增 `benchmarks/run_p9_process_matrix.py`，父进程不导入 torch，每个 mode/repeat 只
  启动一个 `bench_system.py` child。
- 运行顺序按 ABBA/BAAB block 交替并截断为每 mode 精确相同次数；5 repeats 时顺序是
  `ABBABAABAB`。
- 每个 child 前按物理 UUID 检查 NVML memory/utilization，退出后等待同一 idle gate；
  固定 `CUDA_VISIBLE_DEVICES`、offline mode 和 `PYTHONHASHSEED`。
- 强制 clean worktree；dirty 仅在显式 `--allow-dirty` 时作为 diagnostic，且永远不标记
  formal eligible。
- 输出只能写到仓库外或 gitignored 路径，拒绝覆盖既有 artifact；manifest 原子更新并
  在每个 child 后保存进度，失败时保留已完成 run 和错误信息。
- 对 environment、model、workload、traffic、sampling、measurement、非执行 mode 字段、
  非 Graph backend 字段、完整 KV metadata 和 token/text 执行 exact comparability gate。
- 保存各 process 原始值、median/p90/p95/p99/min/max，并用固定 seed 的独立 process
  bootstrap 计算 candidate/baseline median ratio 95% CI。
- formal eligibility 额外要求 clean Git、至少 5 次每 mode、warmup 至少 2、10,000 次
  bootstrap 和冻结 seed；smoke 即使全部 correctness gate 通过也不能升级成正式结论。

设计权衡与拒绝方案:

- 不在 parent 中 import benchmark schema/torch 来复用常量；parent 是否持有 CUDA 相关
  runtime 是被测生命周期的一部分，保持标准库-only 更容易审计。
- 不尝试在同一进程手工清空所有 CUDA/compiler cache；这种清理既不完备，也可能改变
  production 行为。
- 不自动覆盖或续写旧路径。保留每次原始证据比“方便重跑同名文件”更重要。
- ABBA/BAAB 只降低时间漂移，不声称消除时钟、温度或云平台噪声；因此仍需 idle gate、
  原始样本和 CI。

验证命令:

```bash
.venv-local/bin/python -m pytest -q tests/test_p9_process_matrix.py -s

.venv-local/bin/python benchmarks/run_p9_process_matrix.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case guardrail_single_image_448 \
  --mode-a off_eager --mode-b off_graph \
  --expected-gpu-uuid GPU-662a2fa1-37e4-cc52-0a51-27557dba315b \
  --fresh-process-repeats 1 --warmup 0 --max-tokens 4 \
  --batch-size 1 --max-num-seqs 1 --num-kvcache-blocks 16 \
  --vision-attention-backend sdpa --bootstrap-resamples 100 \
  --allow-dirty \
  --output data/p9_baseline/p9_process_orchestrator_v2_smoke_dirty.jsonl
```

验证结果:

- orchestrator contract tests: `11 passed`。
- dry-run 生成 5-repeat `ABBABAABAB` 的 10 条 child 命令，且不创建 artifact 目录。
- dirty smoke 两个 child 均正常退出，前后 GPU 都是 `1 MiB / 0%`；15 项 comparability
  gate 全部 PASS，输出 token exact。
- smoke 的单样本 ratio/CI 只验证汇总器，不解释为性能收益；manifest 正确标记
  `git_dirty=true`、`formal_eligible=false`。

经验:

- benchmark 的“进程边界”与 timing scope、输入和 backend 一样，必须进入实验合同。
- 正确性 hash 是必要条件而非 comparability 的充分条件；公平比较需要逐字段白名单。
- formal/diagnostic 不应靠文档口头区分，应由机器可读 eligibility gates 自动判定。

面试故事提炼:

- “我发现原 benchmark 虽然每个 mode 都重建模型，却共用同一个 CUDA 进程，所以不能
  证明 cold compile、Graph capture 或 allocator 状态公平。我把它拆成标准库父进程加
  单次 child，加入 UUID/idle/release、ABBA/BAAB、逐字段 comparability 和 process-level
  bootstrap CI。还修了一个容易忽略的问题：结果若写到非 ignored 路径，会让后续 run
  自己把 clean Git 变脏。最终 smoke 的两次进程前后显存都回到 1 MiB；只有满足 5 次、
  clean commit 和完整统计门禁的矩阵才会被机器标成 formal。”

剩余风险:

- 当前只完成 dirty 单样本 smoke；H1 output128、每 mode 5 fresh processes 的正式数据必须
  在本轮改动提交后的 clean commit 上重跑。
- 5 个 process 的 CI 仍可能较宽；若 CI 跨越无收益边界，应增加预注册 repeats，而不是
  删除 outlier。
- 当前 `kv_cache.bytes` 主要覆盖 payload 和 scale；page table、allocator metadata、
  unique storage 与 fragmentation 的统一物理 accounting 属于 P9-C.3，不能由本 runner
  的 exact KV metadata gate 替代。

## P9-006: 多模态 admission 使 nominal batch4 变成动态 decode bucket 轨迹

状态: Verified

发现方式:

- 用 H1 八图 workload、batch4 和正式 KV pool 配置运行 P9 eager/CUDA Graph
  fresh-process smoke 时，旧 benchmark 无法在第一次 prefill 后找到四个请求的完整
  prompt KV layout；修复 snapshot 后，Graph 路径又因“最后一次 replay 不是 batch4”失败。

影响范围:

- H1/H2 等高视觉成本 workload 的 CUDA Graph bucket、padding、TPOT 归因和 eager/Graph
  comparability。
- prompt KV physical accounting：不同请求可能在不同 prefill step 才完成，不能只在某个
  全局时刻读取 scheduler 当前状态。
- 不影响 scheduler 的 production 语义；暴露的是 benchmark 对动态 admission 的错误假设。

证据:

```text
workload: 4 requests × 8 images × 448×448
aggregate prompt/image tokens: 6472 / 6272
prefill steps: 4
decode steps: 6

eager actual decode batch histogram:
1:2, 2:2, 3:2

CUDA Graph actual -> captured bucket histogram:
1 -> 1:2, 2 -> 2:2, 3 -> 4:2
captured buckets: [1, 2, 4]

BF16 KV pool:       4,265,607,168 B
scaled-FP8 KV pool: 4,282,122,240 B
active prompt blocks: 28
all four outputs: [6025, 264, 16585, 12313]
output SHA256: 60bc800b87a62c23e5bb7ef1c89732fe52222aa7ce0bedb840c703cfdb6db1a7
each child before/after: 1 MiB used, 0% utilization
```

保留的失败 artifact:

- `data/p9_baseline/p9_h1_b4_bf16_config_smoke_dirty*`：第一次 prefill 后要求四个请求
  同时处于 running，报 `could not capture all post-prefill KV layouts`。
- `data/p9_baseline/p9_h1_b4_bf16_config_smoke_v2_dirty*`：按请求保存 layout 后，仍把
  最后 replay 强制等同 nominal batch4，Graph child 失败。

定位过程:

- 检查 H1 materialized input：每个请求有八张 448 图，视觉 patch 成本大于
  `max_vision_patches_per_batch=8192` 的一半，因此一个 prefill batch 无法同时 admission
  两个请求；batch4 实际需要四个 prefill step。
- 检查 scheduler policy：`max_consecutive_prefill_batches=1` 会在等待中的 prefill 与已
  running 请求的 decode 之间交替，而不是先完成四个 prefill 再进入静态 batch4 decode。
- 用 typed `StepResult.plan` 逐 step 记录 phase 和 batch size，得到确定轨迹：请求逐个加入，
  较早请求也逐个达到 `max_tokens=4` 并退出，所以 measured decode 依次覆盖 batch
  `1/2/3`，本次短 smoke 根本没有 batch4 replay。
- 对照 Graph runner 的运行时 observation，实际 batch3 正确 padding 到 captured bucket4；
  eager 与 Graph 的实际 batch histogram、输出和 KV metadata 一致。

错误假设与排除过程:

- 错误假设一：traffic batch4 等于每个 decode step 都是 batch4。traffic batch 只描述同轮
  提交的请求数；admission、prefill 成本和完成时间会改变运行时 batch。
- 错误假设二：第一次 prefill 后可以一次性 snapshot 全部 prompt KV。只有完成 prefill 的
  sequence 才拥有最终 prompt layout；未 admission 请求还在 waiting。
- 错误假设三：最后一次 Graph replay 最能代表 requested batch。短输出下最后存活的请求
  反而最少，最后 replay 是 batch1；Graph correctness 必须逐 decode step 检查。
- 排除了 Graph 调错 bucket：逐 step observation 显示 actual `1/2/3` 分别 replay
  captured `1/2/4`，满足 padding contract，且 token exact。

根因:

- benchmark 把静态 workload 配置、scheduler 的动态 BatchPlan 和 CUDA Graph capture
  bucket 三个不同层级都压缩成一个 `requested_batch_size=4`。
- prompt KV snapshot 与 Graph metadata 都读取“某一个时刻”的全局状态，无法表达 staged
  prefill 和请求生命周期。

修复:

- `bench_system.py` 改用 typed `step_result()`，不再从 legacy signed token count 推断
  prefill/decode。
- 每个 sequence 在自身 prefill 完成、进入 `DECODING` 时复制 prompt KV snapshot；结束后按
  原始 request 顺序聚合。compressed layout 会去掉“已 sample 但尚未写入 KV”的 pending
  append，只保留真正的 prompt boundary。
- 每个 decode step 记录实际 BatchPlan batch；Graph 路径同时读取 runner 观测到的 actual
  和 captured replay batch，并逐 step fail closed。
- benchmark schema 升级到 v9，新增 `decode_batch_size_counts` 和
  `cuda_graph_replay_counts`；校验 histogram 覆盖全部 measured decode steps、actual 不超过
  traffic batch、captured bucket 已捕获且不小于 actual、Graph 投影与 decode histogram
  完全一致。v8 artifact 保持可读。
- fresh-process comparability 允许 Graph replay 映射作为 backend 差异，但要求 eager/Graph
  的实际 decode histogram exact。

设计权衡与拒绝方案:

- 不提高 vision patch budget 来强行制造静态 batch4；那会改变显存峰值和 production
  admission policy，不再是同一个 baseline。
- 不把所有请求 prefill 完后再测 decode；那会绕开真实 scheduler 的 interleaving，掩盖
  TTFT、KV occupancy 和 Graph bucket 的 pipeline 行为。
- 不只记录最后一次 replay，也不把 actual batch3 伪装为 batch4；保存 actual→captured
  映射才能区分真实并发与 padding 开销。
- schema 当前保存聚合 histogram，而逐 step 原始顺序由 diagnostic 运行时检查保证；如果
  后续引入非确定 admission，正式 online trace 还需保存完整有序事件流。

验证命令:

```bash
.venv-local/bin/python -m pytest -q \
  tests/test_benchmark_schema.py \
  tests/test_p9_process_matrix.py

.venv-local/bin/python benchmarks/run_p9_process_matrix.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case h1_eight_image_448 \
  --mode-a off_eager --mode-b off_graph \
  --expected-gpu-uuid "$P9_GPU_UUID" --cuda-visible-devices 0 \
  --fresh-process-repeats 1 --warmup 0 --max-tokens 4 \
  --batch-size 4 --max-model-len 4096 \
  --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --num-kvcache-blocks 113 --kvcache-block-size 256 \
  --vision-attention-backend sdpa --bootstrap-resamples 100 \
  --allow-dirty \
  --output data/p9_baseline/p9_h1_b4_bf16_config_smoke_v3_dirty.jsonl
```

scaled-FP8 使用同一命令合同，将 mode 改为
`scaled_fp8_kv/scaled_fp8_kv_graph`、blocks 改为 `220`。

验证结果:

- schema/process tests: `60 passed`；architecture/vision focused: `29 passed`。
- BF16 和 scaled-FP8 两个 batch4 smoke 均为 15/15 comparability PASS、token exact，
  actual decode histogram 和 Graph replay 映射均通过 schema-v9 校验。
- 所有 child 前后 GPU 都恢复 `1 MiB / 0%`。
- 两次运行均为 dirty、warmup0、每 mode 1 process、output4，manifest 正确标记
  `formal_eligible=false`；任何单样本 latency ratio 均不形成性能结论。

经验:

- serving batch 是动态状态，不是 CLI 常量；多模态 admission 成本会直接改变 Graph
  bucket 分布和可优化边界。
- benchmark 必须观察 scheduler 发布的 BatchPlan，而不是从请求数或最后一次 runner 状态
  反推执行形态。
- “为了 benchmark 方便而改变 admission”会把 pipeline 问题藏掉；真实 interleaving 本身
  就是后续 Graph、scheduler 和 goodput 优化的重要证据。

面试故事提炼:

- “我原本按 batch4 验证 CUDA Graph，却连续遇到 prompt KV snapshot 缺失和最后 replay
  不是 batch4。逐步检查 scheduler 后发现，每个八图请求的 patch 成本使四个 prefill
  分阶段 admission，而且 policy 会穿插 decode，所以真实 batch 是动态的 1、2、3，
  batch3 再 padding 到 Graph bucket4。我没有提高预算掩盖问题，而是让 benchmark 基于
  typed BatchPlan 逐请求 snapshot、逐 step 校验 actual→captured 映射，并把 schema 升到
  v9。最终 BF16/FP8 eager 与 Graph token exact、显存完整释放；dirty smoke 只作为机制
  证据，正式性能仍坚持 clean、ABBA/BAAB 和 process-level CI。”

剩余风险:

- schema-v9 的 histogram 能验证 bucket 计数和 comparability，但不能替代 online workload
  的完整有序 scheduler trace；P9-E 需要记录 arrival/admission/completion 时间线。
- output128 时后续请求最终会汇聚到 batch4 steady state；正式 H1 matrix 必须报告各 actual
  bucket 占比，不能只给一个平均 TPOT。
- 当前 smoke 只验证 SDPA、单 GPU 和固定 scheduler policy；FlashAttention、不同视觉预算
  或未来 cost model 都需要重新建立轨迹证据。

## P9-007: Full-model 文件名像测试但 pytest 实际未收集 logits gate

状态: Verified

发现方式:

- 在 checkpoint 前按 `docs/REPRODUCIBILITY.md` 复核 full-model 命令，检查四个
  `tests/test_full_model*.py` 的 pytest collection identity。

影响范围:

- 纯文本、单图、多图、视频 HF→Prism full-logits correctness gate。
- 文档中的组合 pytest 命令和历史 full-suite 计数：其他测试通过时，命令整体可以 PASS，
  但四个 script-only 文件的 `if __name__ == "__main__"` 根本不会执行。
- 不推翻本轮此前直接运行脚本得到的 bit-exact 结果；问题在自动收集和失败传播。

证据:

```text
修复前文件结构:
- module pytestmark = model/gpu/integration/slow
- run_hf_*/run_our_*/compare_* helpers
- only `if __name__ == "__main__"` invokes the gate
- no `test_*` callable for pytest collection

额外缺口:
- text script打印 FAIL/MARGINAL 后没有非零退出

修复后 collection:
tests/test_full_model.py::test_full_model_logits_match_hf
tests/test_full_model_vl.py::test_full_model_single_image_logits_match_hf
tests/test_full_model_vl_multi_image.py::test_full_model_multi_image_logits_match_hf
tests/test_full_model_vl_video.py::test_full_model_video_logits_match_hf
```

定位过程:

- 对照文档命令与源码入口，发现 pytest 导入模块时 `__name__ != "__main__"`，因此 main
  block 不会执行。
- 用 `pytest --collect-only` 单独检查四个文件，确认修复前没有对应 full-logits item；组合
  命令之所以不报“no tests”，是同一命令中的 structure/generate tests 提供了可收集 item。
- 检查 direct-script 失败传播，三个 VL 脚本会 `SystemExit(1)`，纯文本脚本只打印结果，
  MARGINAL/FAIL 仍可能以 0 退出。

错误假设与排除过程:

- 错误假设一：文件名以 `test_` 开头就会运行其中的 main。pytest 只收集符合规则的 callable，
  不执行 script main block。
- 错误假设二：组合命令 exit 0 证明每个列出的文件都执行。pytest 的退出状态是所收集 item
  的聚合结果，不是“每个路径至少贡献一个 test”的门禁。
- 没有把问题归因于 marker 过滤；即使显式选择 slow/model/gpu，文件中没有 test callable
  仍然无项可运行。

根因:

- heavyweight correctness 最初作为手工诊断脚本编写，后来移动到 `tests/` 并写入 pytest
  复现命令，但入口形态没有完成从 script 到 test 的迁移。
- 输出字符串被当作人工观察结果，没有统一成为 pytest assertion 和 process exit contract。

修复:

- 四个文件各自抽取唯一 `_run_*_verification()`；pytest test 和 direct-script main 复用同一
  执行函数，避免复制 HF/Prism 加载、释放与 compare 逻辑。
- 新增四个可收集 `test_*_logits_match_hf`，CUDA 不可用时按既有 marker 语义 skip，运行时
  必须断言 decision 为 `PASS`。
- direct-script 继续可单独运行，但四条路径都在非 PASS 时 `SystemExit(1)`。
- 保留 `model/gpu/integration/slow` markers，使 8B gate 不进入 CPU presubmit，也不会被普通
  快速回归意外触发。

设计权衡与拒绝方案:

- 不只修改文档为四条 `python file.py` 命令；那仍会让 full pytest/JUnit 永久漏掉核心 gate。
- 不在 test wrapper 里重新实现一份逻辑；共享 runner 保证手工复现与 CI/pytest 是同一条
  correctness 路径。
- 不把 heavyweight test 降为普通 GPU marker；每个 case 顺序加载 HF 与 Prism 8B，必须
  保持 slow 显式选择，避免占满开发机默认回归。

验证命令:

```bash
CUDA_VISIBLE_DEVICES=0 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python -m pytest -q \
  tests/test_full_model.py \
  tests/test_full_model_vl.py \
  tests/test_full_model_vl_multi_image.py \
  tests/test_full_model_vl_video.py -s
```

验证结果:

- pytest collection：四个文件精确收集 4 个 heavyweight item。
- `4 passed in 39.60s`。
- 纯文本 full logits `[1,64,151936]`，单图/多图/视频 last logits `[1,151936]`；四项
  max/mean diff 均为 `0`，NaN 均为 `0`。
- 每次 HF 与 Prism 模型函数内释放后 allocator 显示 `0.0 GiB`；suite 退出后 GPU 回到
  `1 MiB / 0%`。

经验:

- correctness 文件存在、命令列出路径、pytest exit 0 三件事都不能证明某个 gate 被执行；
  release checklist 必须检查 collection identity 和 item count。
- 人工打印的 PASS/FAIL 必须转换成 assertion 或进程退出合同，否则自动化系统无法消费。
- heavyweight 测试可以通过 marker 控制成本，但不能因此停留在“看起来像 test 的脚本”。

面试故事提炼:

- “我在复核 release 命令时发现四个 full-model 文件虽然叫 `test_*`，却只有 main block。
  组合 pytest 因其他测试通过而显示绿灯，实际没有跑纯文本、单图、多图、视频 logits gate；
  纯文本脚本甚至在 MARGINAL 时仍返回 0。我把每条路径收敛成一个共享 verification runner，
  pytest assertion 和 direct-script exit 复用它，并保留 slow/model/gpu markers。修复后明确
  收集 4 项，四类 HF/Prism logits 都 bit-exact，显存逐项释放。”

剩余风险:

- 历史 full-suite/JUnit 的 test count 不包含这四项，不能拿旧计数证明当前 full-logits
  coverage；下一次 clean release 必须生成新的 JUnit。
- 当前 gate 使用固定 synthetic 输入；标准质量与长生成稳定性仍由 P9 quality matrix 和
  teacher-forced/token-generation tests 单独覆盖。

## P9-008: BF16 batch4 CUDA Graph 在动态 bucket 轨迹上稳定产生 token 分叉

状态: Verified

发现方式:

- 在 clean commit `460d21a` 上执行 H1 BF16 batch4、每 mode 5 个 fresh process 的
  正式 eager/CUDA Graph comparability matrix。

影响范围:

- H1 batch4 CUDA Graph correctness，以及所有依赖该 cell 的 TPOT、吞吐和端到端性能 claim。
- 后续 scaled-FP8 batch1/batch4 formal matrix；在基础 BF16 Graph correctness 未闭环前暂停。
- 不影响已经通过 formal gate 的 H1 BF16 batch1 结论。

证据:

```text
保留 artifact:
data/p9_baseline/h1_bf16_b4.jsonl
data/p9_baseline/h1_bf16_b4.manifest.json
data/p9_baseline/h1_bf16_b4_runs/

manifest status: failed_comparability
fresh children: 10/10 executed and released successfully
failed checks: token_ids_exact, decoded_texts_exact
formal_eligible: false

5 eager processes output SHA256:
a0f0cccd5699d11305c163bbbb20e6a9d50e82536a524cc760734cb7c57816b8

5 Graph processes output SHA256:
1a3f60d65d054ca76f720185e83ab11c849cecb0cc227565db74c80d744b7121

first mismatch:
request=0, generated_token_index=31, eager=2504, graph=448
requests 1-3: token exact

eager actual decode histogram:
batch1: 2, batch2: 2, batch3: 2, batch4: 124

Graph actual -> captured histogram:
1 -> 1: 2, 2 -> 2: 2, 3 -> 4: 2, 4 -> 4: 124

every child before/after:
1 MiB used, 0% utilization

fixed-trajectory diagnostic before fix:
artifact: data/p9_diagnostics/h1_bf16_b4_graph_fixed_trajectory_v1.json
fixed history: exact, 4 requests x 128 sampled rows
first numeric diff: engine step 5, request 0, generation index 3
shape: actual batch3 -> captured batch4
max/mean logit diff at first row: 0.25 / 0.0306588
all three active rows at that step: non-exact logits, unchanged argmax
first argmax diff: engine step 34, request 0, generation index 31
eager top2: 2504=35.25, 448=35.00, margin=0.25
Graph top2: 2504=35.25, 448=35.25, margin=0
all active input/control audits: PASS
all padding slot/context/block-table sentinel audits: PASS

counterfactual after exact-small-batch capture:
artifact: data/p9_diagnostics/h1_bf16_b4_graph_fixed_trajectory_exact_small_v2.json
captured batches: [1, 2, 3, 4]
scheduler/device-input trace exact: true
fixed token history exact: true
all 512 eager/Graph logit rows exact: true
natural argmax exact: true
max logit diff: 0
Graph capture time: 752.845 -> 1084.959 ms (+332.114 ms one-time startup)
process exit: 1 MiB used, 0% utilization

clean formal rerun after fix:
commit: 40466b693e30c35652a9d2e739c61d5ccf1df0e3
artifact: data/p9_baseline/h1_bf16_b4_exact_small_40466b6.jsonl
manifest: data/p9_baseline/h1_bf16_b4_exact_small_40466b6.manifest.json
status: completed
formal_eligible: true
comparability: 15/15 PASS
fresh children: 10/10 PASS and released
eager/Graph output SHA256:
a0f0cccd5699d11305c163bbbb20e6a9d50e82536a524cc760734cb7c57816b8
Graph actual -> captured:
1 -> 1: 2, 2 -> 2: 2, 3 -> 3: 2, 4 -> 4: 124

decode step median:
32.6080 -> 20.5188 ms, -37.07%, 95% CI [-38.34%, -36.62%]
decode throughput median:
119.492 -> 190.432 token/s, +59.37%, 95% CI [+58.38%, +62.47%]
end-to-end median:
5969.98 -> 4348.14 ms, -27.17%, 95% CI [-28.52%, -25.14%]
engine output throughput median:
98.095 -> 139.585 token/s, +42.30%, 95% CI [+38.85%, +47.90%]

TTFT:
engine and preprocessing-inclusive CI both cross zero; no improvement claim

Graph memory delta:
peak allocated +8.16 MiB (+0.039%), reserved +24 MiB (+0.112%)
Graph capture range across 5 fresh processes:
967.546-985.300 ms

aggregate SHA256:
700dd64fa9a56602a252f8c39918b65286fb8c0acceeac71e4330f239201fc6d
manifest SHA256:
26e7c523fb009a6d95981240439ecf559df4bc37eb689d67661543dca87dbdb4
```

定位过程:

- 已确认不是跨进程随机性：同一 backend 的五个 fresh process 分别得到完全相同的 output
  hash，而两个 backend 之间稳定不同。
- 已确认不是 workload 或 scheduler trajectory 漂移：eager/Graph 的 request、sampling、KV
  metadata 和实际 decode batch histogram 均 exact。
- 已确认分叉范围：只有最早 admission、经历 `batch1 -> batch2 -> batch3 -> batch4` 轨迹的
  request 0 在第 32 个生成 token 改变 argmax；其余三个请求 exact。
- 新增 `benchmarks/diagnose_graph_trajectory.py`：eager 生成自然 greedy history，Graph 强制
  使用同一 history；按 request/generation step 比较完整 logits、自然 argmax、top-2 margin，
  同时记录动态 batch 和 Graph static buffers。
- fixed-history 证明第一个 argmax 分叉不是首个误差：首个数值差异精确发生在 engine step 5
  的 actual batch3 -> captured batch4，三个 active row 同时不 exact，但 argmax 仍相同。
- request 3 在该 step 后才 admission；它随后经历的 124 个 exact batch4 decode row 全部
  logits exact，直到尾部其他请求结束、它第一次进入 actual batch3 -> captured batch4 才
  出现数值差异。这构成了同一 run 内的对照组。
- Graph static buffer 审计确认 active `input_ids/position_ids/slot_mapping/context_lens/
  block_tables` 全部与 DeviceBatch exact；padding row 的 slot 为 -1、context 为 0、block table
  全 -1。padding input/position 虽保留零值，但没有有效 KV slot 或 context。
- 把 capture 集合从 `[1,2,4]` 改为 `[1,2,3,4]` 后重跑相同 fixed history，512 个 logit row
  全部 bit-exact，首个数值差异和 argmax 差异都消失。
- 诊断工具第一次尝试在同一进程依次加载两个 8B backend 时 OOM；不是 production 泄漏，
  而是诊断 sampler 与 runner 双向持有。进程退出后 NVML 立即回到 1 MiB；显式切断
  `runner -> sampler` 和 `sampler -> runner` 后，同进程顺序加载和释放成功。

错误假设与排除过程:

- 排除“某一次机器噪声导致 token 随机变化”：每个 mode 内 5/5 hash 完全稳定。
- 排除“Graph 实际执行了不同请求数量”：逐 step histogram 和 actual-to-captured bucket
  计数与调度合同一致。
- 排除 padding row 写入有效 KV：padding 的 slot/context/block table 均使用无效 sentinel，
  active control tensors 也逐项 exact。
- 排除“CUDA Graph replay 本身必然改变数值”：exact batch1、batch2、batch4 在历史尚未被
  padded step 污染时都与 eager bit-exact；补录 exact batch3 后全轨迹 exact。
- 错误假设是第 31 个 token 才开始漂移；固定历史显示误差从 generation index 3 已出现，
  只是当时 top-1 margin 足够大。第 31 个 token 的 eager margin 仅 0.25，Graph 中变成并列。

根因:

- production capture policy 对 max batch4 只录制 `[1,2,4]`，actual batch3 必须 padding 到
  captured batch4。虽然每行数学上独立，但 BF16 model forward 的 GEMM batch shape 从 3
  变为 4，会选择不同的低精度执行形状/内核舍入；首步小误差写入 KV 后沿自回归轨迹累积，
  最终在低 margin token 上改变 greedy argmax。
- 不是 stale control row、scheduler 漂移或跨进程噪声；补录同 shape 的 batch3 graph 后，
  不改变其他输入即可让全轨迹恢复 bit-exact。

修复:

- `ModelRunner._cudagraph_batch_sizes` 对 batch1–8 逐个录制 exact graph；batch16 以上继续用
  stride16 稀疏档位，并始终补录配置的 `max_bs`。
- 更新 metadata/shape contract tests，明确 requested batch3 必须 selected batch3、padding=0，
  max5/8/17 的 capture 集合也有精确期望。
- 新增 CPU-tested fixed-trajectory 诊断器，保留完整 logits drift、top-2 margin、调度输入和
  static-buffer 审计能力；输出路径拒绝覆盖既有 artifact。
- 修复诊断 hook 的 ownership：每个 backend 结束时先断开 sampler/runner 双向引用，再释放
  8B 模型和 CUDA allocator。

设计权衡与拒绝方案:

- 不用“文本语义相近”放宽 token exact；这是同权重、同 BF16 KV、同贪婪采样的执行
  backend 对照，必须先解释数值边界。
- 不删除或覆盖失败 artifact；修复后使用新文件名从零重跑，保留 rejected evidence。
- 不先跑后续性能矩阵再回头处理 correctness；错误输出上的加速不构成有效优化。
- 不为所有 1–512 batch 各录一张 graph；本次证据位于最常见的小动态 batch，1–8 exact
  只增加有限的一次性 capture 成本。更大 sparse bucket 仍必须由对应 workload 的 token/
  quality gate 约束，不能外推本次 exact 结论。
- 不把 eager 也 padding 到 batch4 来制造 comparability；这会改变 baseline 语义并隐藏真正
  的 shape sensitivity，而不是修复 Graph 路径。

验证命令:

```bash
jq '.status, .formal_eligible, .comparability_checks, .summaries' \
  data/p9_baseline/h1_bf16_b4.manifest.json

.venv-local/bin/python benchmarks/diagnose_graph_trajectory.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case h1_eight_image_448 --batch-size 4 --max-tokens 128 \
  --max-model-len 4096 --max-num-batched-tokens 16384 \
  --max-num-seqs 4 --num-kvcache-blocks 113 --kvcache-block-size 256 \
  --vision-attention-backend sdpa \
  --output data/p9_diagnostics/h1_bf16_b4_graph_fixed_trajectory_exact_small_v2.json

.venv-local/bin/python -m pytest -q \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_p9_graph_trajectory_diagnostic.py -s

.venv-local/bin/python benchmarks/run_p9_process_matrix.py \
  --python /data/Prism-Infer/.venv-local/bin/python \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case h1_eight_image_448 \
  --mode-a off_eager --mode-b off_graph \
  --expected-gpu-uuid GPU-662a2fa1-37e4-cc52-0a51-27557dba315b \
  --output data/p9_baseline/h1_bf16_b4_exact_small_40466b6.jsonl \
  --fresh-process-repeats 5 --warmup 2 --max-tokens 128 --batch-size 4 \
  --max-model-len 4096 --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --num-kvcache-blocks 113 --kvcache-block-size 256 \
  --vision-attention-backend sdpa --logits-precision model \
  --mlp-projection-mode packed --bootstrap-seed 20260717 \
  --bootstrap-resamples 10000
```

验证结果:

- 修复前 fixed trajectory：首个 numeric diff 与 batch3 -> graph4 边界精确重合；全部 active
  input/control 和 padding sentinel audit PASS。
- 修复后 fixed trajectory：4 x 128 = 512 个完整词表 logits row bit-exact，natural argmax
  exact，max diff 0；Graph capture 后进程正常释放。
- CPU shape/diagnostic contracts：`9 passed`；额外覆盖 text position 三轴规范化和 padding
  static-buffer sentinel audit。
- CUDA Graph 单图/多图/视频/mixed integration：`2 passed`；mixed metadata 明确断言
  requested batch3 -> selected batch3、padding 0。
- 完整 8B suite：`468 passed, 1 skipped in 300.37s`；JUnit 469 tests / 0 failures /
  0 errors / 1 skipped。随后新增的一项纯 CPU static-buffer audit 已在 focused suite 通过，
  production diff 未再变化。
- ruff format/check、production complexity/runtime-assert/magic-number、compileall、diff check
  和 61 个本地 Markdown 链接全部 PASS。
- 原 formal cell 继续保持 rejected，不重写历史；commit `40466b6` 的新路径 clean formal
  rerun 为 `completed/formal_eligible=true`，15/15 comparability 和 10/10 child PASS，
  eager/Graph token/text exact，因此 P9-008 升级为 Verified。
- decode step 改善 37.07%（95% CI 36.62%–38.34%），E2E 改善 27.17%（95% CI
  25.14%–28.52%）；TTFT CI 跨零，明确不作改善声明。

经验:

- fresh-process 稳定性只能证明分叉可复现，不能自动证明任一 backend 正确。
- CUDA Graph correctness 必须覆盖真实 scheduler 的 bucket 转换轨迹；只测固定 batch1 或
  固定 batch4 会漏掉跨 bucket 状态和 shape-sensitive 数值路径。
- teacher forcing 的价值不是“强行让输出相同”，而是把输入历史固定后观察候选自然 logits；
  这样能区分误差起点、累计传播和第一个可见 argmax 分叉。
- Graph padding 的 sentinel 可以保证内存安全，却不能保证 BF16 数值等价；shape 本身就是
  kernel policy 的一部分。

面试故事提炼:

- “正式 ABBA/BAAB 实验在统计性能前被 token exact 门禁拒绝：五个 eager 与五个 Graph
  进程各自稳定，但 request 0 在第 32 个 token 分叉。我没有放宽成语义相近，而是做了
  teacher-forced 固定轨迹，把首个误差定位到 actual batch3 被 padding 到 graph4；active
  control 和 padding sentinel 都正确，且晚加入、只跑 exact batch4 的 request 是 run 内
  对照组。根因是 BF16 GEMM shape 舍入差异写入 KV 后累积。我将 1–8 改为 exact capture，
  代价是约 332 ms 一次性启动 capture；修复后 512 个完整 logits row bit-exact。过程中还
  发现诊断 hook 的双向引用会让第二个 8B 模型 OOM，因此同步修复了 ownership。”

剩余风险:

- exact capture 目前只覆盖 1–8；actual batch9–15 等大 batch 仍可能 padding 到 stride16
  bucket，并具有同类 BF16 shape sensitivity。未来声称更大并发 token exact 前必须跑对应
  动态轨迹，或基于启动/内存数据决定是否扩大 exact capture 范围。
- 新 policy 的 max4 Graph capture 为约 0.97–0.99 秒，且多占 8.16 MiB peak allocated / 24
  MiB reserved；这是已量化的一次性/常驻代价，不应隐藏在 steady-state TPOT claim 中。
- BF16 batch4 已 Verified，scaled-FP8 formal matrix 可恢复；其 token/quality/comparability
  仍必须独立通过，不能从 BF16 结果外推。

## P9-009: scaled-FP8 batch1 Graph 的 engine TTFT 出现正式回退

状态: Investigating

发现方式:

- 在 commit `d28e68a` 上完成 H1 scaled-FP8 batch1/batch4 eager/model-only Graph 的
  clean fresh-process formal matrix，并逐指标检查 bootstrap CI，而不只查看 decode TPOT。

影响范围:

- 不影响 scaled-FP8 Graph token correctness、KV pool identity、decode/E2E speedup 或既有
  P9-C quality non-inferiority PASS。
- 阻止“model-only Graph 同时改善 TTFT”的表述；后续 NSYS 必须单独解释 prefill/TTFT，
  不能用 decode 收益代替完整 pipeline 归因。

证据:

```text
batch1 artifact:
data/p9_baseline/h1_scaled_fp8_b1_d28e68a.jsonl
data/p9_baseline/h1_scaled_fp8_b1_d28e68a.manifest.json
status=completed, formal_eligible=true, comparability=15/15 PASS
token/text exact, 10/10 children released

batch1 engine TTFT raw (ms):
eager: [275.594, 287.790, 288.199, 281.423, 281.480]
Graph: [290.896, 288.941, 289.786, 386.608, 290.818]
median: 281.480 -> 290.818 ms
regression: 3.32%, 95% CI [0.55%, 37.35%]

batch1 preprocessing-inclusive TTFT:
median: 427.669 -> 376.659 ms
improvement CI: [-26.88%, +24.49%], crosses zero; no claim

batch1 decode step:
34.739 -> 19.162 ms, -44.84%, 95% CI [-47.97%, -44.50%]
batch1 E2E:
4898.21 -> 2810.74 ms, -42.62%, 95% CI [-44.98%, -39.57%]

batch4 engine TTFT:
938.967 -> 923.065 ms, improvement CI [-7.64%, +4.86%], crosses zero
batch4 preprocessing-inclusive TTFT:
1612.117 -> 1602.343 ms, improvement CI [-7.41%, +5.75%], crosses zero

batch4 decode step:
35.669 -> 19.998 ms, -43.93%, 95% CI [-45.17%, -43.65%]
batch4 E2E:
6245.12 -> 4200.98 ms, -32.73%, 95% CI [-34.93%, -30.60%]
```

定位过程:

- 先确认不是 correctness 失败：batch1/batch4 两个 manifest 都是 15/15 comparability PASS，
  每组五个 process output hash exact；scaled payload/scale/total bytes 和 KV metadata exact。
- 先确认不是 Graph replay bucket 漂移：batch1 的 127 个 decode step 都是 `1 -> 1`；batch4
  是 `1 -> 1 / 2 -> 2 / 3 -> 3 / 4 -> 4`，没有 padding。
- `engine_ttft_ms` 口径是同步的 prefill step 总和；当前 model-only Graph 只 capture decode
  model forward，理论上不直接优化 vision/prefill。因此 decode 大幅改善与 TTFT 不改善并不
  矛盾，但 batch1 的显著回退仍需解释。
- 检查 raw process 值后发现 Graph 有一个 `386.608 ms` 高值，但即使其余四个集中在
  `288.9–290.9 ms`，也整体略高于 eager 中位数。按冻结协议保留全部样本，不删除该 run。
- 对照 BF16 batch1：engine TTFT CI 跨零；对照 scaled-FP8 batch4：两种 TTFT CI 都跨零。
  因而现有证据不支持“所有 Graph cell 都系统性降低/提高 prefill”，问题目前只绑定该
  batch1 formal cell。

错误假设与排除过程:

- 不能把 E2E 改善 42.6% 解释为 TTFT 改善；output128 下 decode 主导 E2E。
- 不能因为 Graph backend 只作用于 decode 就丢弃 TTFT；用户体验与完整 pipeline claim 仍
  必须报告所有预注册指标。
- 不能事后删除 `386.608 ms` 或立即重跑直到 CI 好看；若增加 repeats，必须先登记目的、
  样本数和停止条件。
- 目前不能把回退写成 CUDA Graph capture 成本：capture 在 engine 初始化阶段完成，不在
  request TTFT timing scope；需要 NSYS/NVTX 验证 warmup 后的 prefill kernel/host timeline。

根因:

- Investigating。已确认这是 batch1 formal 指标结果，不是 correctness、bucket 或资源释放
  失败；尚未证明是 GPU 状态噪声、prefill kernel/cache 状态、host preprocessing overlap，
  还是 Graph 资源常驻对首个 measured prefill 的间接影响。

修复:

- 尚未修改 production。先把该结果作为 P9-D NSYS 的明确归因问题，不为美化 TTFT 改变
  measurement scope 或 Graph backend。

设计权衡与拒绝方案:

- 保留 model-only Graph 为 supported decode candidate：batch1/batch4 correctness 与 decode/
  E2E CI 均强 PASS，TTFT 问题不构成静默 fallback 或错误输出。
- 不声明 TTFT 收益；batch1 engine TTFT 明确报告回退，其他 TTFT cell 报告 CI 跨零。
- 不立即增加 process count。先用 profile 确认 prefill CPU/GPU timeline 是否存在可解释差异；
  若仍需统计复验，再预注册额外 repeats。

验证命令:

```bash
jq '.status, .formal_eligible, .summaries, .comparison.metrics' \
  data/p9_baseline/h1_scaled_fp8_b1_d28e68a.manifest.json

jq '.status, .formal_eligible, .summaries, .comparison.metrics' \
  data/p9_baseline/h1_scaled_fp8_b4_d28e68a.manifest.json
```

验证结果:

- scaled-FP8 batch1/batch4 formal correctness 与 decode/E2E performance PASS。
- batch1 engine-only TTFT regression 被机器可读 CI 确认并保留，状态 Investigating。

经验:

- pipeline 优化不能只看 TPOT；同一个 candidate 可以显著改善 decode/E2E，同时在 TTFT
  上无收益甚至回退。
- process-level raw values 比单一 median 更重要；它能暴露离群点和稳定偏移，但是否归因
  为系统问题仍需要 profiler，而不是凭肉眼删样本。

面试故事提炼:

- “scaled-FP8 Graph 在 batch1 把 decode step 从 34.74 ms 降到 19.16 ms，E2E 也改善
  42.6%，但 formal gate 同时显示 engine TTFT 回退 3.3%。我没有用 E2E 掩盖它，也没有
  删除一个 386 ms process；我确认 token、KV、bucket 和释放都 exact，把问题登记为
  NSYS 的 prefill timeline 归因项。这个案例说明我优化的是完整推理 pipeline，而不是只挑
  好看的 TPOT。”

剩余风险:

- batch1 只有五个 process，TTFT raw 分布较宽；当前 CI 是正式结果，但根因和跨时段
  可重复性尚未验证。
- model-only Graph 尚未 capture LM head、argmax、状态更新或 D2H；后续 full-step candidate
  可能改变 decode 收益与 host timeline，必须重新跑相同 TTFT/TPOT/E2E gate。

## P9-010: NSYS 缺失 host-only ranges，且 analyzer 对长 trace 退化到逐 range 扫描

状态: Fixed（clean H1 trace 待复验）

发现方式:

- 在 clean commit `5ef051d` 上采集 scaled-FP8 batch1 eager/model-only Graph 的 H1
  output128 node-level NSYS trace，准备分解 P9-009 的 prefill/host timeline。
- 检查 SQLite 的 `NVTX_EVENTS` 后发现 tokenizer/image processor/M-RoPE/scheduler 等
  `cuda=False` region 完全不可见；进一步对 eager 的 4,572 个 decode attention ranges
  运行 analyzer 时，分析超过 60 秒仍未完成。

影响范围:

- 原 semantic JSON 仍有 CPU region 和 phase，但 NSYS 无法把这些 host region 放回 CUDA
  timeline，因此不足以证明 scheduler/preprocessing 与 GPU 的先后和重叠关系。
- 原 analyzer 结果语义正确，但对 H1 output128 长 trace 不具备迭代效率，会阻塞后续
  full-step Graph 与 kernel attribution。
- 不影响已经保存的 raw `.nsys-rep`/SQLite、模型 token correctness 或 formal performance
  数字；P9-009 仍为 Investigating，不能由工具修复直接升级为已归因。

证据:

```text
clean diagnostic inputs, commit 5ef051d:
data/p9_nsys/p9d_scaled_fp8_b1_eager_5ef051d.{nsys-rep,sqlite}
data/p9_nsys/p9d_scaled_fp8_b1_graph_5ef051d.{nsys-rep,sqlite}

eager SQLite events: 591,719
graph SQLite events: 304,729
eager semantic ranges:
  engine.model_runner=128
  attention.decode.fp8_paged_triton=4,572
  attention.kv_store.triton_scaled_fp8=4,608
  preprocess/scheduler/engine.step=absent

old per-range SQL analyzer: >60 s on the full eager target set
rejected TEMP-index candidate: still >60 s
bulk in-memory correlation + refactor:
  eager 5.28 s
  Graph 2.98 s
```

定位过程:

- `performance_profile._enabled_region()` 只在 `cuda=True` 且启用 CUDA Event 时调用
  `torch.cuda.nvtx.range_push()`；因此 `cuda=False` 不仅关闭 Event，也意外关闭了 NVTX。
- engine step 本身没有父 range。即使 scheduler range 可见，也无法可靠按第一个 prefill、
  后续 decode 的完整 schedule→execute→postprocess 周期分组。
- analyzer 对每个 target range 分别查询 runtime、kernel、memcpy、memset 和 graph trace；
  数千 range 造成大量 SQLite 往返。给 raw 表建立 TEMP 副本/索引仍保留逐 range 查询结构，
  实测没有达到停止条件，因此撤销该候选。
- 将 runtime/kernel/memory/graph activity 各读取一次，分别按 start time 二分、按
  `correlationId` 建索引后，同一统计可在 Python 内批量关联。

根因:

- NVTX emission 被错误耦合到 CUDA Event 开关，而 `cuda=False` 的真实含义只应是“不创建
  CUDA Event”，不应是“从 Systems timeline 消失”。
- analyzer 的数据访问模型是 `O(target ranges × SQL scans/queries)`，不适合长输出和逐层
  attention ranges。

修复:

- active CUDA profiling session 对 CPU/GPU region 都 push/pop NVTX；只有 `cuda=True`
  region 创建 CUDA Event。
- `PerformanceProfileSession.begin_step()/end_step()` 增加成对的
  `prism::engine.step` 父 range，且只在 profiling session 中执行，不改变默认 inference
  路径。
- analyzer 以 read-only URI 打开 raw SQLite，一次加载 CUPTI activity，在内存建立时间与
  correlation 索引；增加 engine range 的 `cpu_range_ms`，并保持 schema-v2 既有字段。
- kernel 分类规则、summary/total 字段映射和时间单位改为命名常量；拆分 phase、target、
  GPU timeline 和 kernel breakdown，complexity gate 全部通过。
- CLI 增加 `--quiet`，并默认拒绝覆盖已有 summary artifact。

验证:

- eager/Graph 两份真实 H1 trace 的重构输出分别与重构前逐字节一致：
  SHA256 `5a063ccd0f0c816e8057f8fa079b80f7c2fe9db57e6e234cda9711881d02acd8`
  和 `1414dd5920b9ccdfb4b1cba7d0a970b119e7241ffee38fbbf7d8d931fc61e8be`。
- dirty integration smoke 的 SQLite 出现 `engine.step=4`、
  `engine.scheduler.schedule/postprocess=4/4`、`preprocess.image_processor=1`、
  `preprocess.mrope_positions=1`；CPU-only targets 的 direct kernel/GPU busy 均为 0，
  没有被伪装为 CUDA 工作。
- analyzer 以 `engine.step` 正确划分 1 个 prefill 和 3 个 decode；进程退出后 GPU 为
  `1 MiB / 0%`。
- focused profile/analyzer/P7 summary：`11 passed`；ruff、benchmark complexity、
  compileall 和 diff check PASS。
- 完整测试集：`472 collected`，其中 `471 passed, 1 skipped`，`0 failed, 0 errors`，
  JUnit wall time `303.502 s`；结束后 GPU 再次回到 `1 MiB / 0%`。

设计权衡与拒绝方案:

- 不给每个 host region 创建 CUDA Event；这样会增加无意义的 stream event 和同步语义。
  只补 NVTX，保留 semantic JSON 中 `cuda_ms=None` 的合同。
- 不修改 raw SQLite 建永久索引；artifact 必须保持导出时内容。analyzer 用 read-only
  connection 和进程内索引。
- 不保留无效 TEMP-index 候选，也不以“有索引”作为优化结论；用真实 59 万事件 trace 和
  byte-exact summary 验证算法替换。

面试故事提炼:

- “为了定位一个只有 3.3% 的 TTFT 回退，我先审计 profiler 自身，发现 CPU-only region
  因为和 CUDA Event 开关耦合而没有进入 NVTX；同时长 trace analyzer 对 4,572 个 attention
  range 逐个扫 SQLite。第一次加 TEMP 索引仍超过 60 秒，我按停止条件撤销，改成一次加载
  CUPTI activity、按时间和 correlationId 批量关联，eager trace 降到 5.3 秒且 JSON
  byte-exact。之后才继续根因分析，避免用不完整或被工具扭曲的 timeline 下结论。”

剩余风险:

- 当前 integration smoke 是 dirty diagnostic，不构成性能证据；必须提交后重新采集 clean
  H1 eager/Graph trace，确认完整 host ranges、token hash 和 GPU release。
- analyzer 现在用内存换查询时间；更大 online trace 的 peak RSS 尚未量化，后续需要在
  trace metadata 中记录 event count、analysis wall time 和 max RSS。

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
- 已新增并运行现位于 `tools/debug/full_model_layerwise.py` 的手工诊断，按 embedding、RoPE、每层 norm/attention/MLP/output、final norm、logits 比较 HF 与 Prism-Infer 激活。
- 分层证据:
  - `embed`: max diff `0.000000e+00`。
  - `rope`: max diff `0.000000e+00`。
  - 第一处非零误差: `layer_00.attn`, max diff `3.906250e-03`, mean diff `7.651032e-05`。
  - `layer_00.mlp`: max diff `6.250000e-02`, mean diff `7.205015e-04`。
  - 误差随层数累积，`layer_35.mlp` max diff `2.000000e+01`, mean diff `6.590960e-02`。
  - final norm 后误差收敛为 max diff `1.500000e+00`, mean diff `7.806452e-03`。
  - logits max diff `2.500000e-01`, mean diff `2.831022e-02`。
- 当前证据指向 attention 路径是首个差异来源；embedding、权重加载、RoPE 不是首个差异来源。
- 进一步微定位脚本 `tools/debug/attention_micro.py` 显示:
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
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tools/debug/attention_micro.py
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
- `tools/debug/attention_micro.py`: 修复后 `q_rope/k_rope/sdpa_gqa/attn_out/layer0_out` max diff 全部 `0.000000e+00`。
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
- 该记录完成时仍需在 P2 阶段验证图文输入、视觉 token 替换、DeepStack 注入和端到端 generate tokens；后续 P2-004 已验证单图 1-token greedy，P2-005 已验证单图图文 full logits 和 layerwise strict 对齐。

## P2-001: Processor pipeline 输入边界建立

状态: Verified

发现方式:

- P2.1 阶段任务拆分。

影响范围:

- 为后续 P2.2 多模态 `Sequence` 和 P2.3 3D position ids 提供稳定的单图输入数据结构。
- 当前不改变 engine、scheduler、模型 forward 或 KV cache 行为。

证据:

- HF processor 源码显示 `Qwen3VLProcessor.__call__` 返回 `input_ids`、`pixel_values`、`image_grid_thw`，并按 `image_grid_thw.prod() // merge_size**2` 展开 image token:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:146-194`。
- P2 设计文档已将 HF processor 定义为非核心预处理工具，不能替代 Prism-Infer 核心模型。

定位过程:

- 现有测试中 `tests/test_patch_embed.py` 和 `tests/test_vit_attention_rope.py` 已直接使用 HF `AutoProcessor` 生成 `pixel_values`。
- P2 需要把该逻辑从测试散落用法收敛到 engine 边界模块，避免后续 `Sequence` 和 `ModelRunner` 直接依赖临时脚本代码。

根因:

- 这不是 bug 修复，而是 P2 输入边界建设。此前项目没有统一的单图 VL request 数据结构。

修复:

- 新增 `prism_infer/engine/vl_inputs.py`:
  - `SingleImageInputs` 保存 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`、`image_token_id`、视觉 token 数和展开后的 prompt。
  - `load_vl_processor` 延迟加载 HF processor，避免非 VL 单元测试被第三方依赖阻塞。
  - `prepare_single_image_inputs` 只支持单图单请求，生成 P2 后续模块可消费的数据。
  - `validate_single_image_inputs` 校验 shape、raw patch count、merge 后 image token 数，失败时显式报错。
- 新增 `tests/test_processor_pipeline.py`:
  - 对比 Prism-Infer 边界函数输出与 HF processor reference 是否 exact match。
  - 验证 `token_ids` 属性可用于后续 `Sequence` 构造。
  - 验证视觉 token 数不匹配时抛出 `ValueError`。

验证命令:

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m compileall \
  prism_infer/engine/vl_inputs.py \
  tests/test_processor_pipeline.py
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q tests/test_processor_pipeline.py -s
```

验证结果:

- `compileall`: PASS。
- `tests/test_processor_pipeline.py`: `3 passed in 6.23s`。
- 输出摘要:
  - `input_ids shape: [1, 210]`
  - `pixel_values shape: [784, 1536]`
  - `image_grid_thw shape: [1, 3]`
  - `image tokens: 196 / expected 196`
  - `pixel_values max diff: 0.000000e+00`

经验:

- Processor pipeline 必须验证 token 数，而不能只验证 shape。Qwen3-VL 的 LLM 视觉 token 数是 `image_grid_thw.prod() // merge_size**2`，原始 patch 数 `784` 对应 LLM image token 数 `196`。
- 将 HF processor 限定在预处理边界，可以复用成熟 tokenizer/image processor，同时不污染 Prism-Infer 核心模型、attention 和 KV cache 自实现原则。

剩余风险:

- 当前只支持单图单请求，不支持多图、视频和 batch 混合图文。
- 当前只验证 processor 输出，不代表 `Sequence`、3D position ids、Prefill、Decode 或端到端 greedy tokens 已完成。

## P2-002: 多模态 Sequence 与单图 3D position ids 对齐

状态: Verified

发现方式:

- P2.2/P2.3 阶段任务。

影响范围:

- 为 P2.4 `ModelRunner.prepare_prefill` 接收 VL payload、P2.5 decode 使用 `rope_delta` 延续 3D position ids 建立前置数据结构。
- 当前不改变模型 attention、KV cache 写入/读取、scheduler 或公开 `LLM.generate` 行为。

证据:

- 当前 `Sequence` 原本只保存 token、采样参数和 block table，没有图像字段。
- HF 4.57.1 `Qwen3VLModel.get_rope_index` 在图文 prefill 中生成 `[3, batch, seqlen]` position ids 和 `[batch, 1]` rope delta，源码位置:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033`。
- HF forward 在 prefill 计算 rope index，在 decode 用 `cache_position + rope_deltas` 延续 position ids，源码位置:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1177-1221`。

定位过程:

- P2.1 已证明单图 processor 输出 `input_ids=[1, 210]`、`pixel_values=[784, 1536]`、`image_grid_thw=[1, 3]` 且 image token 数为 `196`。
- P2.2 需要让 `Sequence` 能携带这些字段，并在多进程序列化时保留 prefill 必要信息。
- P2.3 需要在 Prism-Infer 内自实现 rope index helper，而不是在运行时调用 HF model 方法。

根因:

- 这不是 bug 修复，而是 P2 engine 数据流建设。没有 `position_ids/rope_delta` 和 VL payload，后续 prefill/decode 无法严格对齐 Qwen3-VL。

修复:

- 新增 `prism_infer/models/qwen3_vl_position.py`:
  - `get_qwen3_vl_rope_index` 自实现纯文本和单图图文 position ids。
  - `get_qwen3_vl_rope_index_from_config` 从 config 读取 `image_token_id`、`video_token_id`、`vision_start_token_id` 和 `spatial_merge_size`。
  - 当前 video token 显式报错，不 silent fallback。
- 修改 `prism_infer/engine/sequence.py`:
  - 构造函数新增 VL 字段。
  - 新增 `Sequence.from_single_image_inputs`。
  - 新增 `is_multimodal`。
  - Prefill 序列化保留 `pixel_values/image_grid_thw/position_ids/rope_delta`。
  - Decode 序列化省略 `pixel_values/image_grid_thw/position_ids`，保留 `rope_delta`。
- 新增 `tests/test_vl_rope_index.py`:
  - 单图 `position_ids/rope_delta` 与 HF `get_rope_index` exact match。
  - 纯文本分支与 HF text-only 逻辑 exact match。
  - `image_grid_thw` 数量不匹配时显式报错。
- 新增 `tests/test_sequence_multimodal.py`:
  - 纯文本 `Sequence` 行为不回归。
  - 单图 prefill 序列化保留 VL payload。
  - 单图 decode 序列化不重复传 pixel values，但保留 `rope_delta`。

验证命令:

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m compileall \
  prism_infer/engine/sequence.py \
  prism_infer/models/qwen3_vl_position.py \
  tests/test_sequence_multimodal.py \
  tests/test_vl_rope_index.py
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_vl_rope_index.py \
  tests/test_sequence_multimodal.py -s
```

验证结果:

- `compileall`: PASS。
- P2.1-P2.3 组合测试: `9 passed in 9.15s`。
- 输出摘要:
  - `input_ids shape: [1, 210]`
  - `position_ids shape: [3, 1, 210]`
  - `rope_delta shape: [1, 1]`
  - `position_ids max diff: 0.000000e+00`
  - `rope_delta max diff: 0.000000e+00`
  - `prefill position_ids shape: [3, 1, 210]`
  - `decode rope_delta shape: [1, 1]`

经验:

- Qwen3-VL 图文 prefill 不能继续使用一维 position ids；视觉 token 区间必须使用 T/H/W 三维位置。
- `rope_delta` 是 decode 正确延续 position ids 的关键状态，decode 阶段不需要重复传图像像素。
- `Sequence` 序列化必须区分 prefill 和 decode，避免多进程 decode 每步重复传大体积 `pixel_values`。

剩余风险:

- 该记录完成时只覆盖数据结构和 rope index；后续 P2-003/P2-004 已完成 `ModelRunner.prepare_prefill` 消费、engine KV attention 和单图 `generate_vl` 入口。
- 该记录完成时项目总体只支持 P2 单图 eager correctness；后续 P3 已补多图、视频和
  mixed batch，P7.3 又补齐 online mixed-VL 与 chunked paged prefill。

## P2-003: Qwen3-VL engine attention 接入 KV cache 时的 flash-attn API 不兼容

状态: Verified

发现方式:

- P2.4/P2.5 新增 `tests/test_qwen3_vl_attention_kv.py` 后运行失败。

影响范围:

- 阻断 Qwen3-VL LLM attention 接入 engine KV cache。
- 如果不修复，engine prefill 会在本地 flash-attn varlen 调用处直接报错；decode paged KV 也不能依赖 `flash_attn_with_kvcache(block_table=...)`。
- 不影响 P1 full-sequence SDPA 路径，但会影响 P2 engine flatten 路径。

证据:

```text
失败命令:
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py -s

失败摘要:
TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'block_table'
```

本地 flash-attn 函数签名证据:

```text
varlen:
(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
 dropout_p=0.0, softmax_scale=None, causal=False, ...)

kvcache:
(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None,
 cache_seqlens=None, cache_batch_idx=None, softmax_scale=None, causal=False, ...)
```

定位过程:

- 先新增小模型 attention 对齐测试，构造 `hidden=[1, 7, 64]` 和 engine flatten `hidden=[7, 64]`。
- 测试在 `Qwen3VLTextAttention._forward_engine -> Attention.forward -> flash_attn_varlen_func` 报错，证明问题发生在 engine attention 的 flash-attn 调用参数，而不是 Q/K/V shape 或权重加载。
- 用 `inspect.signature` 检查本地 flash-attn API，确认当前版本的 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 均不支持 `block_table` 参数。
- 进一步补 decode paged KV 数值测试，避免只修 prefill 而忽略 decode 读取历史 KV 的正确性。

根因:

- `prism_infer/layers/attention.py` 原实现假设本地 flash-attn varlen/kvcache API 支持 paged `block_table` 参数。
- 当前环境安装的 flash-attn API 是连续 KV cache 版本，不支持 `block_table`。这是依赖 API 能力不匹配，不是 Qwen3-VL 模型权重或 M-RoPE 数值问题。

修复:

- 修改 `prism_infer/layers/attention.py`:
  - prefill 无 prefix-cache 时调用 `flash_attn_varlen_func` 的本地支持签名，不再传 `block_table`。
  - paged prefix-cache prefill 当前显式报错，避免 silent incorrect。
  - decode 在 `context.block_tables is not None` 时走可验证 eager fallback: 根据 `block_tables/context_lens` 从 paged KV cache 收集历史 K/V，再用单步 SDPA 计算。
- 修改 `prism_infer/models/qwen3_vl.py`:
  - `Qwen3VLTextAttention` 增加 engine flatten 路径，投影 Q/K/V 后接入 `Attention`。
  - `Qwen3VLTextModel.forward` 对 engine flatten 的 `[3, num_tokens]` position ids 规范化为 `[3, 1, num_tokens]`，避免 `num_tokens == 3` 时被 `MRope` 误判。
  - `Qwen3VLForCausalLM.compute_logits` 在 prefill flatten 输出中按 `cu_seqlens_q[1:] - 1` 选择每条序列最后 token logits。
- 修改 `prism_infer/engine/model_runner.py`:
  - 引入 `ModelInputs`，统一传递 `input_ids/position_ids/pixel_values/image_grid_thw`。
  - VL prefill 传递单图 payload 和 `[3, seqlen]` position ids。
  - VL decode 不传图像，只用 `rope_delta` 生成 `[3, 1]` position ids。
  - P2 第一版显式拒绝跨多个 chunk 的 VL chunked prefill。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline.py \
  /data/Prism-Infer/tests/test_vl_rope_index.py \
  /data/Prism-Infer/tests/test_sequence_multimodal.py \
  /data/Prism-Infer/tests/test_model_runner_vl_prefill.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_full_model_structure.py \
  /data/Prism-Infer/tests/test_qwen3_vl.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model.py
```

```bash
git -C /data/Prism-Infer diff --check
```

验证结果:

- `compileall`: PASS。
- P2.1-P2.5 组合测试: `15 passed in 12.16s`。
- engine attention prefill:
  - `hidden input shape: [1, 7, 64]`
  - `engine output shape: [7, 64]`
  - `attention output max diff: 0.000000e+00`
  - `attention output mean diff: 0.000000e+00`
  - `k_cache max diff: 0.000000e+00`
  - `v_cache max diff: 0.000000e+00`
- engine attention decode paged KV:
  - `decode q shape: [1, 4, 16]`
  - `decode engine output shape: [1, 4, 16]`
  - `decode output max diff: 0.000000e+00`
  - `decode output mean diff: 0.000000e+00`
- `ModelRunner.prepare_prefill`:
  - `prefill input_ids shape: [210]`
  - `prefill position_ids shape: [3, 210]`
  - `prefill pixel_values shape: [784, 1536]`
  - `prefill image_grid_thw shape: [1, 3]`
- `ModelRunner.prepare_decode`:
  - `decode input_ids shape: [1]`
  - `decode position_ids shape: [3, 1]`
  - `decode actual positions: [28, 28, 28]`
- P1 轻量回归: `10 passed in 72.53s`。
- P1 full logits 回归: `Result: PASS`; max diff `0.000000e+00`，mean diff `0.000000e+00`。
- `git diff --check`: PASS。

经验:

- engine attention 接入不能只看模型 forward shape，必须检查 KV cache 写入和 decode 读取。
- 外部依赖 API 不能按记忆假设。即使函数名相同，不同 flash-attn 版本是否支持 `block_table` 也必须用签名或源码确认。
- 高性能 paged decode kernel 可以后续优化，但 P2 correctness 第一版必须有明确、可验证的 eager fallback。
- VL chunked prefill 不能静默复用完整 `pixel_values` 去跑截断 token chunk；在没有设计跨 chunk 视觉 embedding 传递前必须显式拒绝。

剩余风险:

- 该记录完成时 P2.6 尚未完成；后续 P2-004 已验证 `LLM.generate_vl` 单图 1-token greedy 与 HF 完全一致。
- 当前 paged decode fallback 是 correctness 实现，不代表性能优化完成。
- paged prefix-cache prefill 当前显式不支持。
- 该记录完成时多图、视频、batch 混合图文、VL CUDA Graph decode 均未完成；后续 P3.1/P3.2/P3.3/P3.5 已补齐。

## P2-004: `LLM.generate_vl` 端到端接入中的 VL config 与生命周期问题

状态: Verified

发现方式:

- P2.6/P2.7 真实端到端 smoke: 对比 HF `generate(max_new_tokens=1, do_sample=False)` 与 Prism-Infer `LLM.generate_vl(..., temperature=0.0, max_tokens=1)`。

影响范围:

- 阻断 Qwen3-VL 从 `LLM` 用户入口完成单图图文推理。
- 影响 engine 初始化、KV cache 分配和测试进程退出清理。

证据:

第一次失败:

```text
AttributeError: 'Qwen3VLConfig' object has no attribute 'max_position_embeddings'
```

第二次失败:

```text
TypeError: invalid dtype object: only floating-point types are supported as the default type
```

第三次失败:

```text
AttributeError: 'Qwen3VLConfig' object has no attribute 'num_key_value_heads'
AttributeError: 'Qwen3VLConfig' object has no attribute 'num_hidden_layers'
```

端到端 token 对齐最终证据:

```text
HF token_ids: [785]
Prism token_ids: [785]
LLM.generate_vl one-token greedy HF alignment: PASS
```

定位过程:

- 用 1-token greedy 作为最小端到端验证，避免长输出掩盖 first-token logits/采样问题。
- HF 参考路径使用本地模型、同一张 448x448 合成图、同一个 prompt 和 `do_sample=False`。
- Prism-Infer 路径通过 `LLM(model_path, enforce_eager=True, ...)` 初始化完整 engine，再调用 `generate_vl`。
- 逐个修复初始化失败后，才进入真正的 token 对齐验证。

根因:

- Qwen3-VL 的 LLM 配置字段位于 `hf_config.text_config`，而原 `Config`/`ModelRunner` 仍按纯文本 Qwen3 顶层 config 读取:
  - `max_position_embeddings`
  - `num_key_value_heads`
  - `num_hidden_layers`
  - `hidden_size`
- `hf_config.torch_dtype` 对 Qwen3VLConfig 顶层可能为 `None`，真实 dtype 在 `text_config.torch_dtype`。
- `LLMEngine.exit` 不是幂等的；测试中手动 `llm.exit()` 后，`atexit` 再次调用会访问已删除的 `model_runner`。

修复:

- 修改 `prism_infer/config.py`:
  - `max_model_len` 优先使用顶层 `max_position_embeddings`，缺失时使用 `text_config.max_position_embeddings`。
- 修改 `prism_infer/engine/model_runner.py`:
  - 新增 `_resolve_model_dtype`，从顶层 config 或 `text_config` 解析 `torch.dtype`。
  - 新增 `_text_hf_config`，KV cache 和 graph capture 的 LLM 维度统一从 text config 读取。
  - KV cache 分配后断言分配到的 attention 层数等于 `text_config.num_hidden_layers`。
- 修改 `prism_infer/engine/llm_engine.py`:
  - 新增 `add_vl_request` 和 `generate_vl`。
  - `generate_vl` 当前要求 `enforce_eager=True`，避免未验证的 VL CUDA Graph 路径。
  - `exit` 改为幂等。
- 修改 `prism_infer/sampling_params.py` 和 `prism_infer/layers/sampler.py`:
  - 允许 `temperature=0.0`。
  - `Sampler` 对 `temperature <= 1e-10` 使用 deterministic argmax。

验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_sampler_greedy.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py \
  /data/Prism-Infer/tests/test_text_only_regression.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline.py \
  /data/Prism-Infer/tests/test_sequence_multimodal.py \
  /data/Prism-Infer/tests/test_vl_rope_index.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py \
  /data/Prism-Infer/tests/test_model_runner_vl_prefill.py \
  /data/Prism-Infer/tests/test_sampler_greedy.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py \
  /data/Prism-Infer/tests/test_text_only_regression.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_full_model_structure.py \
  /data/Prism-Infer/tests/test_qwen3_vl.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model.py
```

验证结果:

- P2.6/P2.7 新增测试: `7 passed`，其中:
  - `greedy token_ids: [1, 2]`
  - `HF token_ids: [785]`
  - `Prism token_ids: [785]`
  - `LLM.generate_vl one-token greedy HF alignment: PASS`
  - `text output token_ids: [785]`
- 该记录完成时的 P2 Gate 组合测试: `22 passed in 52.64s`；后续 P2-005 加入 vision strict 回归后为 `24 passed in 48.49s`。
- 该记录完成时的 P1 轻量回归: `10 passed in 73.20s`；后续 P2-005 回归为 `10 passed in 74.68s`。
- P1 full logits: `Result: PASS`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- `compileall`: PASS。
- `git diff --check`: PASS。

经验:

- 端到端验证必须从真实 `LLM` 用户入口开始，单测里直接调用模型 forward 不能覆盖 config、KV cache、processor、scheduler 和生命周期问题。
- 多模态 config 经常把 LLM 子配置放在 `text_config`，engine 层不能假设所有字段在顶层。
- `temperature=0` 不能通过随机采样近似；HF token 对齐必须走 deterministic argmax。
- `exit` 这类生命周期接口要幂等，否则测试和用户手动释放会留下噪音异常。

剩余风险:

- 当前 P2 完成范围是单图、单请求、`enforce_eager=True` correctness。
- 该记录完成时 HF token 对齐只覆盖 1-token greedy exact match；后续 P2-005 已补齐单图图文 last logits 和 full-model layerwise strict PASS。
- 长输出、多轮、吞吐、延迟和显存 benchmark 还未评估。
- 该记录完成时多图、视频、batch 混合图文、VL CUDA Graph decode、高性能 paged decode kernel、paged prefix-cache prefill 仍未完成；后续 P3 已补齐前五项，P7.3 已补 chunked paged prefill与 online mixed-VL。VL token-id prefix hash因像素语义不安全而显式禁用。

## P2-005: 图文 full logits 未严格对齐，根因在 VisionEncoder RoPE 初始化与 PatchMerger eps

状态: Verified

发现方式:

- P2 Gate 后补充 `tests/test_full_model_vl.py`，对比 HF 与 Prism-Infer 单图图文最后 token logits。

影响范围:

- 阻断 P2 “每一层精度与 HF 对齐”的严格出口。
- 修复前虽然 `LLM.generate_vl` 1-token greedy token id 与 HF 一致，但不能声明图文 full logits strict PASS。
- 影响 VisionEncoder 输出、DeepStack visual embeds、后续 LLM hidden states 和图文 logits。

证据:

失败命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model_vl.py
```

失败输出摘要:

```text
input_ids shape: [1, 210]
pixel_values shape: [784, 1536]
image_grid_thw shape: [1, 3]
position_ids shape: [3, 1, 210]
Shape: HF=[1, 151936], Our=[1, 151936]
NaN: HF=0, Our=0
HF mean/std:  -1.756945e+00 / 4.123917e+00
Our mean/std: -1.762104e+00 / 4.144710e+00
Max diff:  3.964844e-01
Mean diff: 6.158398e-02
Result: FAIL
```

分层失败摘要:

```text
visual                       2.109375e-01 4.801622e-03
embed                        0.000000e+00 0.000000e+00
rope                         0.000000e+00 0.000000e+00
layer_00.input               2.109375e-01 4.481514e-03
logits                       2.215869e+01 4.351552e-01
```

定位过程:

- 先用 `tools/debug/full_model_vl_layerwise.py` 证明 embedding 和 LLM M-RoPE 为 exact match，首个差异已经出现在 `model.visual` 输出。
- 再写临时 vision 内部分层检查，用同一个 processor 输入 `pixel_values=[784, 1536]` 和 `grid_thw=[[1, 28, 28]]` 对齐 HF 与 Prism-Infer。
- 第一轮定位结果:
  - `patch_embed`: max diff `0.000000e+00`。
  - `pos_embed`: max diff `0.000000e+00`。
  - `rot_pos_emb`: max diff `1.907349e-06`。
  - `block_00`: max diff `6.250000e-02`。
  - `main_merger`: max diff `2.109375e-01`。
- 继续微定位 block 0:
  - `block0.input/norm1/qkv_raw/q_pre_rope/k_pre_rope/v` 全部 max diff `0.000000e+00`。
  - 第一处差异在 vision RoPE 后: `block0.q_rope` max diff `7.812500e-03`，`block0.k_rope` max diff `1.953125e-03`。
- 检查 HF 源码:
  - HF `Qwen3VLVisionRotaryEmbedding` 在构造时注册 `inv_freq` buffer，源码位置:
    `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:79-90`。
  - HF `Qwen3VLVisionPatchMerger` 的 LayerNorm 使用 `eps=1e-6`，源码位置:
    `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:93-106`。
  - HF `Qwen3VLVisionModel.rot_pos_emb/forward` 使用该 rotary buffer，并在 forward 中构造 position embeddings，源码位置:
    `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:603-753`。
- 临时验证显示，如果使用 HF 保存的 `inv_freq` buffer 构造 vision RoPE，`rot_pos_emb` 与 HF exact match；如果每次在 CUDA 上动态重算，`inv_freq` max diff 只有 `5.960464e-08`，但 bf16 RoPE 后会跨量化边界并逐层放大。
- 第一次修复后，单独构造 `VisionEncoder` 已 exact match；但完整 `Qwen3VLForCausalLM` 路径仍失败。进一步定位发现原因是测试用 `torch.set_default_device("cuda")` 构造 Prism-Infer 全模型，导致我们新加的 buffer 在 CUDA 上初始化；HF `from_pretrained(...).cuda()` 是先在 CPU 初始化再移动到 CUDA。

根因:

- `VisionEncoder.rot_pos_emb` 原实现每次 forward 动态计算 RoPE 频率表，并且在完整模型构造时可能走 CUDA 初始化/计算路径；HF 是在模块构造时用 CPU 数值路径生成 `inv_freq` buffer，再随模型移动到 GPU。
- CPU 与 CUDA 计算 `theta ** (...)` 只有 `~5.96e-08` 级差异，但在 bf16 旋转中会触发舍入差异，layer 0 attention 开始分叉，最终影响图文 logits。
- `PatchMerger` 原实现未显式传 `eps`，PyTorch 默认 `1e-5`；HF patch merger 使用 `eps=1e-6`。这会影响 main visual output 和 deepstack visual embeds 的严格对齐。

修复:

- 修改 `prism_infer/vision/vision_encoder.py`:
  - 新增自实现 `VisionRotaryEmbedding`，在构造期注册 `inv_freq` buffer。
  - `inv_freq` 强制按 CPU 数值路径生成，再移动到当前目标 device，匹配 HF `from_pretrained(...).cuda()` 的初始化语义。
  - `VisionEncoder.rot_pos_emb` 改为使用 `self.rotary_pos_emb(max_hw)`，不再每次 forward 动态重算频率。
  - `PatchMerger` 的 `nn.LayerNorm` 显式设置 `eps=1e-6`。
  - 清理过期注释，删除不再使用的动态 `_compute_rope_freqs`。
- 新增 `tests/test_vision_rope_init.py`:
  - 覆盖默认 device 为 CUDA 时，`VisionEncoder` 的 `inv_freq/freq_table/rot_pos_emb` 仍与 HF exact match。
  - 覆盖 main merger 与 3 个 deepstack merger 的 LayerNorm eps 全部为 `1e-6`。

验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_vision_rope_init.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model_vl.py
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tools/debug/full_model_vl_layerwise.py
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline.py \
  /data/Prism-Infer/tests/test_sequence_multimodal.py \
  /data/Prism-Infer/tests/test_vl_rope_index.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py \
  /data/Prism-Infer/tests/test_model_runner_vl_prefill.py \
  /data/Prism-Infer/tests/test_sampler_greedy.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py \
  /data/Prism-Infer/tests/test_text_only_regression.py \
  /data/Prism-Infer/tests/test_vision_rope_init.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_full_model_structure.py \
  /data/Prism-Infer/tests/test_qwen3_vl.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model.py
```

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

验证结果:

- `tests/test_vision_rope_init.py`: `2 passed in 8.71s`。
  - `inv_freq shape: [18]`
  - `rot_pos_emb shape: [784, 36]`
  - `inv_freq max diff: 0.000000e+00`
  - `freq_table max diff: 0.000000e+00`
  - `rot_pos_emb max diff: 0.000000e+00`
  - `merger eps values: [1e-06, 1e-06, 1e-06, 1e-06]`
- `tests/test_full_model_vl.py`: `Result: PASS`。
  - `Shape: HF=[1, 151936], Our=[1, 151936]`
  - `NaN: HF=0, Our=0`
  - `HF mean/std:  -1.756945e+00 / 4.123917e+00`
  - `Our mean/std: -1.756945e+00 / 4.123917e+00`
  - `Max diff:  0.000000e+00`
  - `Mean diff: 0.000000e+00`
- `tools/debug/full_model_vl_layerwise.py`: 从 `visual`、`embed`、`rope`、36 层 LLM、`final_norm` 到 `logits` 全部 max diff `0.000000e+00`，mean diff `0.000000e+00`。
- P2 Gate + vision 回归: `24 passed in 48.49s`。
  - `HF token_ids: [785]`
  - `Prism token_ids: [785]`
  - `LLM.generate_vl one-token greedy HF alignment: PASS`
- P1 轻量回归: `10 passed in 74.68s`。
- P1 full logits: `Result: PASS`，max diff `0.000000e+00`，mean diff `0.000000e+00`。
- `compileall`: PASS。

经验:

- 端到端 token 一致不能替代 full logits。1-token greedy `[785]` 相同，只说明 argmax 没变，不说明分布严格对齐。
- bf16 对齐中，`1e-7` 量级的频率表差异也可能跨过舍入边界，并在 27 层 ViT 和 36 层 LLM 中放大。
- 默认 device 会改变模块初始化路径；对齐 HF 时要复现“CPU 初始化后搬到 GPU”的 buffer 数值语义，而不是只看最终 device。
- LayerNorm `eps` 是数值语义的一部分，不能依赖 PyTorch 默认值。
- full-model layerwise debug 的价值是先证明首个分叉点，再局部修复；这次从 logits diff 0.396 缩小到 vision RoPE 和 PatchMerger eps。

剩余风险:

- 当前图文 strict PASS 覆盖单图、单请求、`enforce_eager=True`、最后 token logits 和 1-token greedy。
- 该记录完成时多图、视频、batch 混合图文、VL CUDA Graph decode、高性能 paged decode kernel、paged prefix-cache prefill、长输出分布/质量评估和吞吐/延迟 benchmark 仍未完成；后续 P3 已补齐多图、视频、mixed batch、32-token 长输出、logits/ppl 分布、VL CUDA Graph 和 paged decode baseline，P7.3 已补 chunked paged prefill与 online mixed-VL。VL token-id prefix hash保持禁用。

## P3-001: 多图 full logits 不对齐，根因是 VisionEncoder 多图跨图 attention

状态: Verified

发现方式:

- P3.1 新增 `tests/test_full_model_vl_multi_image.py`，对比 HF 与 Prism-Infer 在同一条请求包含两张图片时的最后 token logits。

影响范围:

- 阻断 P3.1 “多图输入 correctness” 的 full logits strict 门禁。
- 单图图文 strict PASS 不暴露该问题，因为单图没有跨图 attention 的边界。
- 若不修复，后续 multi-image generate、mixed batch 和视觉 KV 分析都会建立在错误的视觉 token 表征上。

触发命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model_vl_multi_image.py
```

修复前失败摘要:

```text
input_ids shape: [1, 408]
pixel_values shape: [1568, 1536]
image_grid_thw shape: [2, 3]
image tokens: 392 / expected 392
position_ids shape: [3, 1, 408]
Shape: HF=[1, 151936], Our=[1, 151936]
HF mean/std:  -1.318763e+00 / 4.206440e+00
Our mean/std: -1.337387e+00 / 4.196328e+00
Max diff:  2.125000e+00
Mean diff: 2.093065e-01
Result: FAIL
```

根因定位:

- HF `Qwen3VLVisionModel.forward` 按每张图构造 `cu_seqlens`:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:727-735`。
- HF eager vision attention 在非 FA2 路径中按 `cu_seqlens` split 后逐段计算，不让不同图片 patch 互相 attention:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:223-248`。
- Prism-Infer 原 `ViTAttention.forward` 把两张图片的 patch 拼成一个序列后做全局双向 attention，导致跨图 attention。单图时 `cu_seqlens` 只有 `[0, N]`，所以不会暴露。

修复:

- 修改 `prism_infer/vision/vision_encoder.py`:
  - `VisionEncoder.forward` 根据 `grid_thw` 构造 `cu_seqlens`。
  - `ViTBlock.forward` 透传 `cu_seqlens`。
  - `ViTAttention.forward` 在 `cu_seqlens.numel() > 2` 时按图片分段调用 SDPA，再按 token 维拼回。
- 修改 `prism_infer/engine/vl_inputs.py`、`prism_infer/engine/sequence.py`、`prism_infer/engine/llm_engine.py`:
  - `prepare_image_inputs` 支持单图或多图 list/tuple。
  - 保留 `prepare_single_image_inputs` / `SingleImageInputs` 兼容 P2。
  - `LLMEngine.add_vl_request` 和 `generate_vl` 支持多图输入。
- 新增/扩展测试:
  - `tests/test_processor_pipeline_multi_image.py`
  - `tests/test_vl_rope_index_multi_image.py`
  - `tests/test_full_model_vl_multi_image.py`
  - `tests/test_llm_vl_generate.py::test_generate_vl_multi_image_one_token_matches_hf_greedy`

拒绝的替代方案:

- 不把多图拆成多条请求后在用户侧拼结果；这无法验证一条请求内多图 token span、M-RoPE 和 DeepStack 注入。
- 不用 HF model wrapper 替代 Prism-Infer VisionEncoder；HF 只作为 processor 和 correctness reference。
- 不通过放宽 full logits 阈值完成 P3.1；修复前 max diff `2.125000e+00` 已明显超过 `<1e-2` 门槛。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline_multi_image.py \
  /data/Prism-Infer/tests/test_vl_rope_index_multi_image.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py::test_add_vl_request_builds_multi_image_sequence -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model_vl_multi_image.py
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_generate.py::test_generate_vl_multi_image_one_token_matches_hf_greedy -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline.py \
  /data/Prism-Infer/tests/test_processor_pipeline_multi_image.py \
  /data/Prism-Infer/tests/test_sequence_multimodal.py \
  /data/Prism-Infer/tests/test_vl_rope_index.py \
  /data/Prism-Infer/tests/test_vl_rope_index_multi_image.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py \
  /data/Prism-Infer/tests/test_model_runner_vl_prefill.py \
  /data/Prism-Infer/tests/test_sampler_greedy.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py \
  /data/Prism-Infer/tests/test_text_only_regression.py \
  /data/Prism-Infer/tests/test_vision_rope_init.py -s
```

验证结果:

- `compileall`: PASS。
- P3.1 轻量门禁: `5 passed in 8.01s`。
  - `multi input_ids shape: [1, 408]`
  - `multi pixel_values shape: [1568, 1536]`
  - `multi image_grid_thw shape: [2, 3]`
  - `multi image_grid_thw: [[1, 28, 28], [1, 28, 28]]`
  - `multi image tokens: 392 / expected 392`
  - `multi pixel_values max diff: 0.000000e+00`
  - `multi position_ids shape: [3, 1, 408]`
  - `multi rope_delta shape: [1, 1]`
  - `multi position_ids max diff: 0.000000e+00`
  - `multi rope_delta max diff: 0.000000e+00`
- 多图 full logits: `PASS (max diff < 0.01)`。
  - `Shape: HF=[1, 151936], Our=[1, 151936]`
  - `NaN: HF=0, Our=0`
  - `HF mean/std:  -1.318763e+00 / 4.206440e+00`
  - `Our mean/std: -1.318763e+00 / 4.206440e+00`
  - `Max diff:  0.000000e+00`
  - `Mean diff: 0.000000e+00`
- 多图端到端 generate: `1 passed in 20.75s`。
  - `HF multi-image token_ids: [785]`
  - `Prism multi-image token_ids: [785]`
  - `LLM.generate_vl multi-image one-token greedy HF alignment: PASS`
- P2/P3.1 组合回归: `30 passed in 78.39s`。

经验:

- 多图 Vision Encoder 不能把所有 patch 当作一个完整双向 attention 序列；图片间 attention 边界是模型语义的一部分。
- 单图 strict PASS 不能外推到多图。多图需要单独验证 processor、position ids、vision attention 和端到端 logits。
- 多图问题的首个高价值检查是 full logits；如果只看 1-token greedy，argmax 可能仍然相同而掩盖分布错误。

剩余风险:

- 该记录完成时 P3.1 只覆盖单请求多图、`enforce_eager=True` correctness；后续 P3.3/P3.5 已覆盖 mixed batch 和 CUDA Graph decode。
- 该记录完成时视频输入、batch 混合图文、长输出多 token 质量评估、VL CUDA Graph decode、高性能 paged decode kernel、paged prefix-cache prefill 和吞吐/延迟 benchmark 仍未完成；后续 P3 已补齐前五项和基础 benchmark，P7.3 已补 chunked paged prefill与 online mixed-VL。VL prefix hash因像素语义不安全而禁用。

## P3-002: 视频输入 correctness 建立，核心风险是 video_grid_thw 的帧展开语义

状态: Verified

发现方式:

- P3.2 阶段任务。先调查本地 HF processor/model 视频路径，再实现 Prism-Infer video payload、rope index、模型 forward 和公开 `LLM.generate_video` 入口。

影响范围:

- 补齐一条请求包含视频帧输入的 correctness baseline。
- 为 P3.3 mixed batch、P3.4 长输出和后续视觉 KV 分析提供视频路径基础。
- 不改变当前 P2/P3.1 image 单图/多图路径。

外部参考证据:

- HF processor 声明返回 `pixel_values_videos` 和 `video_grid_thw`:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:146-155`。
- HF processor 对视频 token 按 timestamp 和每帧 `<|vision_start|>...<|vision_end|>` 展开:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:196-234`。
- HF `get_rope_index` 先 `repeat_interleave(video_grid_thw, video_grid_thw[:, 0])`，再把 `T` 置为 1:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033`。
- HF `get_video_features` 复用 `get_image_features`，也就是视频 patch 仍走同一个 vision encoder:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1035-1048`。
- HF forward 中 `pixel_values_videos/video_grid_thw` 会生成 video embeds，并替换 `video_token_id` 占位:
  `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1145-1151`。

processor 探测结果:

```text
prompt_text prefix: <|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe this video.<|im_end|>\n<|im_start|>assistant\n
video_token: <|video_pad|> 151656
input_ids shape: [1, 420]
pixel_values_videos shape: [1568, 1536]
video_grid_thw shape: [1, 3]
video_grid_thw: [[2, 28, 28]]
video token count: 392
expected video tokens: 392
```

根因/设计要点:

- 视频不能用 image-only path 假装支持；processor 会插入 timestamp 文本 token，并把一个视频 grid 在 rope index 中按帧拆成多个 `T=1` visual span。
- VisionEncoder 本身可以复用，因为 HF `get_video_features` 直接复用 image feature path；区别在 text token span 和 `video_token_id` 替换。
- `rope_delta` 必须来自 video-aware 3D position ids，否则 decode 后续 position 会错。

修复:

- 修改 `prism_infer/engine/vl_inputs.py`:
  - 新增 `VideoInputs`、`build_video_prompt`、`prepare_video_inputs`、`validate_video_inputs`。
  - 校验 `pixel_values_videos=[num_patches,patch_dim]`、`video_grid_thw=[num_videos,3]`、video token count 和 grid 推导 token 数一致。
- 修改 `prism_infer/models/qwen3_vl_position.py`:
  - `get_qwen3_vl_rope_index` 增加 `video_grid_thw`。
  - image/video span 按 token 顺序处理。
  - 视频 grid 按 HF 语义 `repeat_interleave` 后 `T=1`。
- 修改 `prism_infer/models/qwen3_vl.py`:
  - `Qwen3VLModel.forward` 增加 `pixel_values_videos/video_grid_thw`。
  - video payload 走自实现 `VisionEncoder`。
  - 使用 `video_token_id` 替换 video token；image/video 同时存在时按 visual mask 合并 DeepStack。
- 修改 `prism_infer/engine/sequence.py`、`prism_infer/engine/model_runner.py`、`prism_infer/engine/llm_engine.py`:
  - `Sequence.from_video_inputs` 携带 video payload。
  - `ModelRunner.prepare_prefill` 和 `_forward_model` 传递 video payload。
  - 新增 `LLMEngine.add_video_request` 和 `generate_video`。
- 新增测试:
  - `tests/test_processor_pipeline_video.py`
  - `tests/test_vl_rope_index_video.py`
  - `tests/test_full_model_vl_video.py`
  - `tests/test_llm_vl_generate.py::test_generate_video_one_token_matches_hf_greedy`

拒绝的替代方案:

- 不把 video frames 当作多张 image 走 image token；这会丢失 timestamp 文本展开和 video-specific rope index 语义。
- 不在模型 forward 中调用 HF `get_video_features`；核心模型和 VisionEncoder 必须自实现。
- 不用 1-token greedy 替代 full logits；P3.2 必须同时验证 processor、rope、full logits 和公开入口 token。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_processor_pipeline_video.py \
  /data/Prism-Infer/tests/test_vl_rope_index_video.py \
  /data/Prism-Infer/tests/test_llm_vl_generate.py::test_add_video_request_builds_video_sequence -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python /data/Prism-Infer/tests/test_full_model_vl_video.py
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_generate.py::test_generate_video_one_token_matches_hf_greedy -s
```

验证结果:

- `compileall`: PASS。
- P3.2 轻量门禁: `5 passed in 8.37s`。
  - `video input_ids shape: [1, 420]`
  - `video pixel_values_videos shape: [1568, 1536]`
  - `video_grid_thw shape: [1, 3]`
  - `video_grid_thw: [[2, 28, 28]]`
  - `video tokens: 392 / expected 392`
  - `video pixel_values max diff: 0.000000e+00`
  - `video position_ids shape: [3, 1, 420]`
  - `video rope_delta shape: [1, 1]`
  - `video position_ids max diff: 0.000000e+00`
  - `video rope_delta max diff: 0.000000e+00`
- 视频 full logits: `PASS (max diff < 0.01)`。
  - `Shape: HF=[1, 151936], Our=[1, 151936]`
  - `NaN: HF=0, Our=0`
  - `HF mean/std:  -1.130621e+00 / 4.290061e+00`
  - `Our mean/std: -1.130621e+00 / 4.290061e+00`
  - `Max diff:  0.000000e+00`
  - `Mean diff: 0.000000e+00`
- 视频端到端 generate: `1 passed in 21.99s`。
  - `HF video token_ids: [785]`
  - `Prism video token_ids: [785]`
  - `LLM.generate_video one-token greedy HF alignment: PASS`

经验:

- 视频路径最容易错的是 position ids，不是 VisionEncoder。HF 用 timestamp 文本表达时间，所以 video grid 在 rope index 里被拆成每帧 `T=1`。
- processor 探测要先做，否则容易把 `videos=frames` 和 `videos=[frames]` 的 batch 语义搞混。
- full logits strict PASS 证明 video embeds 替换、DeepStack 注入和 LLM 后续路径没有引入可见误差。

剩余风险:

- 该记录完成时 P3.2 只覆盖单请求 synthetic video、`enforce_eager=True` correctness；后续 P3.3/P3.5 已覆盖 mixed batch 和 CUDA Graph decode。
- 该记录完成时 batch 混合图文、长输出多 token 质量评估、VL CUDA Graph decode、高性能 paged decode kernel、真实视频文件采样策略、paged prefix-cache prefill 和吞吐/延迟 benchmark 仍未完成；后续 P3 已补齐 mixed batch、32-token 长输出、logits/ppl 分布、CUDA Graph、paged decode baseline 和基础 benchmark，P7.3 已补 chunked paged prefill与 online mixed-VL；真实视频文件采样策略仍未覆盖，VL prefix hash保持禁用。

## P3-003: mixed batch 单序列限制解除，prefix-cache 干扰需要明确排除

状态: Verified

发现方式:

- P3.3 阶段任务。原 `ModelRunner.prepare_prefill/prepare_decode` 在 VL batch 中显式要求 `len(seqs) == 1`，导致 text-only、single-image、multi-image、video 无法同批执行。

影响范围:

- 补齐真实 VL engine 的 batch 混合图文 correctness 基线。
- 涉及 scheduler 形成同批请求后的 flatten position ids、visual payload concat、slot mapping、decode position 延续。
- P3.3 当时不覆盖 prefix-cache/chunked-prefill 的 VL mixed batch，也不覆盖 CUDA Graph mixed decode；后者由 P3.5补齐，前者由 P7.3补齐 correctness路径。

触发命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_mixed_batch_generate.py -s
```

修复前失败摘要:

```text
ValueError: P2 VL prefill currently supports exactly one sequence per batch
```

实现过程中新增测试后还暴露了一个测试设计问题:

```text
ValueError: P3 mixed VL prefill does not support prefix/chunk cache hits yet
```

该错误来自同一个 engine 先跑单请求，再跑完全相同 mixed batch，触发了 prefix cache hit。P3.3 的设计范围是 non-prefix mixed batch correctness；prefix-cache/chunked-prefill VL mixed batch 是后续风险，不能混入本阶段 PASS。

根因:

- `ModelRunner.prepare_prefill` 旧逻辑只支持一个携带 `position_ids` 的 VL sequence；mixed text/VL batch 直接报错。
- `ModelRunner.prepare_decode` 旧逻辑只支持一个携带 `rope_delta` 的 VL sequence；mixed decode 直接报错。
- mixed batch 中 text-only 请求原本是一维 positions，VL 请求是 `[3,seqlen]` positions；需要统一成模型可消费的 `[3,total_tokens]` flatten 形态。

修复:

- 修改 `prism_infer/engine/model_runner.py`:
  - prefill 中只要 batch 内有任意 VL 请求，就把所有 sequence positions 统一为 `[3,seqlen]` chunk。
  - text-only sequence 在 mixed VL batch 中用一维 position 扩展到三轴。
  - image payload 按请求顺序 concat 为 `pixel_values`，image grid concat 为 `image_grid_thw`。
  - video payload 按请求顺序 concat 为 `pixel_values_videos`，video grid concat 为 `video_grid_thw`。
  - decode 中 text-only delta 取 `0`，VL 请求使用各自 `rope_delta`，统一输出 `[3,batch]`。
  - prefix/chunk cache hit 仍显式报错，避免未验证路径 silent fallback。
- 修改 `prism_infer/engine/llm_engine.py`:
  - `add_request` 返回 `seq_id`。
  - 新增 `generate_mixed`，支持 `text/image/images/video` 请求列表。
- 新增测试:
  - `tests/test_model_runner_vl_mixed_prefill.py`
  - `tests/test_llm_vl_mixed_batch_generate.py`

拒绝的替代方案:

- 不把 mixed batch 拆成多次单请求执行后拼结果；这无法验证 scheduler batch、slot mapping、context_lens 和 KV cache 不串扰。
- 不在 prefix-cache/chunked-prefill 未验证时把该组合算作 P3.3 PASS；测试改为 fresh 单请求 engine 与 fresh mixed engine 对比。
- 不为 text-only 保持一维 positions 混入同一个 VL batch；统一 `[3,total_tokens]` 可以复用 Qwen3-VL M-RoPE flatten 路径。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_model_runner_vl_mixed_prefill.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_mixed_batch_generate.py -s
```

验证结果:

- `compileall`: PASS。
- mixed ModelRunner 输入准备: `2 passed in 8.20s`。
  - `mixed prefill input_ids shape: [1043]`
  - `mixed prefill position_ids shape: [3, 1043]`
  - `mixed pixel_values shape: [2352, 1536]`
  - `mixed image_grid_thw shape: [3, 3]`
  - `mixed pixel_values_videos shape: [1568, 1536]`
  - `mixed video_grid_thw shape: [1, 3]`
  - `mixed cu_seqlens_q: [0, 5, 215, 623, 1043]`
  - `mixed slot_mapping shape: [1043]`
  - `mixed decode input_ids shape: [3]`
  - `mixed decode position_ids shape: [3, 3]`
  - `mixed decode positions: [[5, 28, 56], [5, 28, 56], [5, 28, 56]]`
  - `mixed decode context_lens: [6, 211, 421]`
- mixed 公开入口: `1 passed in 33.67s`。
  - `single token_ids: [[11], [785], [785], [785]]`
  - `mixed token_ids: [[11], [785], [785], [785]]`
  - `mixed batch size: 4`
  - `LLM.generate_mixed mixed batch single-run equivalence: PASS`

经验:

- mixed batch correctness 不能只看 API 不报错；必须检查 `cu_seqlens_q`、`slot_mapping`、payload concat shape 和 decode `context_lens`。
- prefix cache 会改变 prefill 语义。测试若复用同一个 engine 跑相同 prompt，就可能把尚未支持的 prefix-cache 路径混进 mixed batch 门禁。
- text-only 请求在 Qwen3-VL mixed batch 中也可以扩展为三轴同值 position ids；这与纯文本分支语义等价，并简化 flatten batch。

剩余风险:

- 该记录完成时 P3.3 只覆盖 non-prefix、non-chunked、`enforce_eager=True` mixed batch 1-token greedy correctness；后续 P3.4/P3.5 已覆盖 mixed batch VL rows 32-token、text row batch 数值敏感性解释和 CUDA Graph。
- mixed batch full logits 对 HF batch reference在该记录中未完成；P3 后续补齐 long greedy、VL CUDA Graph、高性能 paged decode baseline和性能 benchmark，P7.3补齐 chunked paged prefill与 online mixed-VL。VL prefix hash不作为支持能力开放。

## P3-004: 长输出 32-token greedy 与 logits/ppl 分布门槛建立

状态: Verified

发现方式:

- P3.4 阶段任务。P2/P3.1/P3.2/P3.3 之前主要验证 1-token greedy；需要证明多 token decode 中 KV cache、rope_delta 和 visual payload 只在 prefill 使用的逻辑不会在后续 token 中分叉。

影响范围:

- 覆盖 single-image、multi-image、video 的 `max_tokens=8/16/32` HF greedy exact。
- 覆盖 single-image、multi-image、video 32-token teacher-forced logits 分布和 perplexity。
- 覆盖 mixed batch 中 VL rows 的 `max_tokens=32` fresh 单请求等价。
- 记录 text-only row 在 bf16 batch=1 与 batch=4 duplicate forward 中的数值敏感性；HF 与 Prism 具有完全相同的 max/mean diff，因此不把 text-only mixed 32-token exact 作为 VL mixed correctness 门槛。

根因/风险:

- 1-token greedy 只能证明 prefill logits argmax 一致，不能证明 decode 多步 KV cache 读取、rope_delta 延续和 sampler 在后续 token 中不分叉。
- 长输出中一旦某一步 token 分叉，后续上下文不同，不能继续用最终文本“看起来合理”作为 correctness。
- bf16 batch-size 数值敏感性会让 text-only long generation 在 batch=1 与 batch=4 中出现后续 token 分叉。HF 自身对相同 `Hello` duplicate batch 的最后 logits 也有 max diff `5.312500e-01`、mean diff `1.473503e-01`，因此该现象不能归因为 VL mixed batch 串扰。

修复:

- 新增 `tests/test_llm_vl_long_generate.py`:
  - HF 先串行生成 single-image、multi-image、video 的 `max_new_tokens=32` greedy reference。
  - Prism-Infer 通过公开 `generate_vl/generate_images/generate_video` 生成 32 token。
  - 输出 prompt token 数、HF token ids、Prism token ids、first mismatch，以及 prefix@8/16/32。
- 扩展 `tests/test_llm_vl_mixed_batch_generate.py`:
  - `test_generate_mixed_batch_thirty_two_tokens_matches_single_request_outputs` 比较 fresh 单请求独立运行与 fresh mixed batch 的 32-token token ids。
  - VL rows 要求 32-token exact；text-only row 要求 prefix@8 exact，并由 batch numeric sensitivity 测试解释后续分叉。
- 新增 `tests/test_vl_logits_distribution.py`:
  - 对 HF 生成的 32-token 轨迹做 teacher-forced forward。
  - 比较 HF 与 Prism 的 logits shape、mean/std、max diff、mean diff、cross entropy 和 perplexity。
- 新增 `tests/test_batch_numeric_sensitivity.py`:
  - 对 HF 与 Prism 分别比较 `Hello` batch=1 vs batch=4 duplicate forward 的最后 logits。
  - 证明两者数值差异完全一致，作为 mixed text row 长输出分叉的证据。

拒绝的替代方案:

- 不用“文本看起来合理”作为质量门槛。
- 不用 1-token greedy PASS 外推到长输出。
- 不把 mixed batch 与同一个 engine 先后运行相同请求作为 reference；那会混入 prefix-cache 语义。

验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_long_generate.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_mixed_batch_generate.py::test_generate_mixed_batch_thirty_two_tokens_matches_single_request_outputs \
  /data/Prism-Infer/tests/test_vl_logits_distribution.py \
  /data/Prism-Infer/tests/test_batch_numeric_sensitivity.py -s
```

验证结果:

- HF greedy 长输出: `1 passed in 26.76s`。
  - single-image prompt tokens: `210`
  - single-image prefix@8/16/32 match: `True / True / True`
  - single-image first mismatch: `None`
  - multi-image prompt tokens: `408`
  - multi-image prefix@8/16/32 match: `True / True / True`
  - multi-image first mismatch: `None`
  - video prompt tokens: `420`
  - video prefix@8/16/32 match: `True / True / True`
  - video first mismatch: `None`
- mixed batch / logits 分布 / batch 数值敏感性: `2 passed in 57.39s`，扩展关键回归中合计 `13 passed in 217.02s`。
  - mixed text prefix@8 match: `True`
  - `LLM.generate_mixed VL rows mixed batch 32-token equivalence: PASS`
  - single-image/multi-image/video logits shape: `[1,32,151936]`
  - single-image/multi-image/video logits max diff: `0.000000e+00`
  - single-image/multi-image/video logits mean diff: `0.000000e+00`
  - single-image/multi-image/video ppl diff: `0.000000e+00`
  - HF duplicate batch max/mean diff: `5.312500e-01 / 1.473503e-01`
  - Prism duplicate batch max/mean diff: `5.312500e-01 / 1.473503e-01`

经验:

- 长输出验证应该输出首个分叉点；这比最终文本更适合定位 decode/KV/position 问题。
- mixed batch 长输出要用 fresh engine 对比，避免 prefix-cache 改变 prefill 路径。
- 32-token greedy exact 与 teacher-forced logits/ppl exact 是比 8-token 更强的 P3.4 门槛。
- mixed batch 中 text-only row 的 32-token 分叉来自 batch-size 数值敏感性，HF 和 Prism 证据一致；VL rows 仍要求 32-token exact。

剩余风险:

- P3.4 仍不覆盖随机采样输出文本一致性；采样模式只应比较分布或 ppl。
- 该记录完成时长上下文压力和 prefix-cache/chunked-prefill VL mixed batch未完成；
  P7.3后续用 301-token text与646-token image+text建立了 chunked correctness基线，
  但 gather+SDPA长上下文性能仍待专用 kernel。

后续 P7.4-A 更新（2026-07-16）:

- 上述 mixed VL rows跨 batch shape exact是 P3.4 当时 logits路径的历史合同。
  默认改为 model precision后，同一 mixed shape重复生成仍 exact，但 batch1 GEMV与
  batch4 GEMM可在低 margin视频 token上分叉，不能继续把跨 shape exact当作通用
  correctness门槛。
- 当前合同要求 image/multi-image跨 shape长前缀、video分叉点显式记录，并由 HF
  teacher-forced logits/PPL exact与独立 reference-task quality gate共同约束；详见
  `docs/issues/P7-006-LOGITS-FP32-WEIGHT-CAST.md`。

## P3-005: VL CUDA Graph decode 支持 3D position ids 与非标准 batch 档位

状态: Verified

发现方式:

- P3.5 阶段任务。`ModelRunner.capture_cudagraph` 原始 graph `positions` 占位为一维 `[max_bs]`，而 VL decode 需要 `[3,batch]`。
- 新增 mixed benchmark sanity 后触发真实失败: `max_num_seqs=3` 时 `graph_bs=[1,2]`，decode batch=3 在 `next(x for x in self.graph_bs if x >= bs)` 处抛出 `StopIteration`。

影响范围:

- `enforce_eager=False` 的 single-image、multi-image、video 和 mixed batch decode。
- CUDA Graph replay 的 `input_ids/position_ids/slot_mapping/context_lens/block_tables` 占位 shape 与地址稳定性。
- 后续 graph-vs-eager latency benchmark 的可信度。

根因:

- 图文 decode position ids 已在 P2/P3.3 中扩展为三轴 `[3,batch]`，但 CUDA Graph capture/replay 仍按 text-only `[batch]` 写入。
- graph 档位只包含 `[1,2,4,8]` 和 16 的倍数；当 `max_num_seqs` 是 3、5、17 等非标准值时，最大实际 batch 没有对应或更大的 graph。
- paged decode 原 fallback 中有 `.item()`、`.tolist()`、动态 `cat` 等 Python 路径，不能作为稳定 CUDA Graph capture 路径；因此 P3.5 需要和 P3.6 kernel 一起推进。

修复:

- `ModelRunner._as_mrope_decode_positions`:
  - text-only `[batch]` 显式扩展为 `[3,batch]`。
  - VL `[3,batch]` 原样通过。
- `capture_cudagraph`:
  - graph `positions` 占位改为 `[3,max_bs]`。
  - capture/replay 都用 `positions[:, :bs]` 调用模型。
  - replay 前清理并拷贝 `block_tables`，避免上一次较长 table 残留。
- `ModelRunner._cudagraph_batch_sizes(max_bs)`:
  - 保留常用 1/2/4/8/16... 档位。
  - 当 `max_bs` 不是标准档位时，额外加入 `max_bs`。
- `LLMEngine.add_vl_request/add_video_request` 移除 `enforce_eager=True` 硬拒绝；公开 VL 入口只有在 graph correctness 通过后才放开。

拒绝的替代方案:

- 不把 VL decode 降级到 eager 后仍声明 graph PASS。
- 不把 text-only graph PASS 外推成 VL graph PASS。
- 不为 `max_num_seqs=3` 在 benchmark 脚本里绕开 batch=3；修复 graph 档位生成才是根因处理。

验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_model_runner_vl_cudagraph.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_cuda_graph_decode.py::test_vl_cuda_graph_single_multi_video_match_eager -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_llm_vl_cuda_graph_decode.py::test_vl_cuda_graph_mixed_batch_matches_eager -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python benchmarks/bench_vl_cudagraph_decode.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --case mixed \
  --max-tokens 8 \
  --warmup 2 \
  --repeat 5 \
  --kvcache-block-size 1024
```

验证结果:

- `tests/test_model_runner_vl_cudagraph.py`: `2 passed in 3.83s`。
  - `text decode input positions shape: [2]`
  - `text graph positions shape: [3, 2]`
  - `vl decode input positions shape: [3, 3]`
  - `vl graph positions shape: [3, 3]`
  - `max_bs=3, graph_bs=[1, 2, 3]`
  - `max_bs=17, graph_bs=[1, 2, 4, 8, 16, 17]`
- single/multi/video graph-vs-eager: `1 passed in 83.41s`。
  - single-image eager/graph token ids: `[785, 2168]`
  - multi-image eager/graph token ids: `[785, 1378]`
  - video eager/graph token ids: `[785, 2766]`
- mixed graph-vs-eager: `1 passed in 31.26s`。
  - mixed eager token ids: `[[11, 358], [785, 1378], [785, 2766]]`
  - mixed graph token ids: `[[11, 358], [785, 1378], [785, 2766]]`
  - batch=3 覆盖非标准 graph 档位。
- benchmark:
  - commit: `45edd3a`
  - GPU: NVIDIA GeForce RTX 5090
  - torch: `2.6.0a0+ecf3bae40a.nv25.01`
  - case: mixed, `max_tokens=8`, warmup=2, repeat=5, `kvcache_block_size=1024`
  - correctness: PASS
  - eager decode median/p90/min/max: `31.5488ms / 34.2537ms / 30.9992ms / 34.5397ms`
  - graph decode median/p90/min/max: `16.4468ms / 16.5553ms / 16.4189ms / 16.6193ms`
  - eager token/s: `93.96`; graph token/s: `182.14`
  - peak memory: `27995.47MiB`

经验:

- CUDA Graph correctness 不只看 capture 成功，还要覆盖实际 replay batch 与录制档位的关系。
- VL decode 的 3D position ids 必须作为 graph 输入形状的一部分固化；只在 eager path 支持 `[3,batch]` 不够。
- benchmark 脚本本身也是验证工具，能暴露普通 correctness 测试没覆盖的非标准配置。

剩余风险:

- 当前 graph benchmark 只覆盖 RTX 5090、单卡、mixed batch、小 synthetic 输入；还未覆盖 4070/4090、多卡 TP、长上下文和真实视频文件采样策略。
- graph benchmark 数值不能直接用于声称超过 vLLM/SGLang；对比前需要单独固定版本、输入集合、参数和显存限制。

## P3-006: 自实现 Triton paged decode kernel 接入与基线 benchmark

状态: Verified

发现方式:

- P3.6 阶段任务。P2/P3 早期 paged decode 为 correctness fallback: 逐请求从 `block_tables/context_lens` 收集 K/V 后调用 PyTorch SDPA，不是高性能 kernel。
- P3.5 CUDA Graph capture 也需要 graph-safe decode 路径；原 eager fallback 里的 `.item()`、`.tolist()` 和动态 `torch.cat` 不适合作为 CUDA Graph replay 内路径。

影响范围:

- `prism_infer/layers/attention.py` decode + paged KV cache 路径。
- Qwen3-VL GQA: `num_heads != num_kv_heads` 时 query head 到 KV head 的映射。
- P3/P6 性能基线和后续 torch.compile/Triton 优化。

根因:

- 原 `_forward_decode_eager` 是正确性实现，不是 kernel 实现。它每条 sequence 在 Python 中遍历 block table 并拼接历史 K/V，CPU overhead 随 batch 增加。
- 如果在 P3.5 中继续依赖 fallback，即使 token ids 正确，也无法建立可 benchmark 的 graph decode 路径。

修复:

- 新增 `prism_infer/ops/paged_decode.py`:
  - `paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale)`。
  - Triton kernel 每个 program 处理 `(seq_idx, q_head)`，按 block table 遍历 paged KV，使用 online softmax 累积输出。
  - 支持 GQA: `kv_head = q_head // (num_heads // num_kv_heads)`。
  - unsupported shape 显式报错，不 silent fallback。
- `Attention.forward` 在 decode 且 `context.block_tables is not None` 时，CUDA + Triton 环境优先走 `paged_decode_attention`；CPU/无 Triton 时显式保留 eager fallback。
- 新增 `tests/test_paged_decode_kernel.py` 与 `benchmarks/bench_paged_decode.py`。

拒绝的替代方案:

- 不用 flash-attn `flash_attn_with_kvcache` 假装支持 paged block table；本地签名没有 block table 参数。
- 不在 kernel unsupported 时 silent fallback 后报告 kernel PASS。
- 不用性能数字替代 correctness；所有 benchmark case 先输出 max diff/mean diff/PASS。

验证命令:

```bash
PYTHONPATH=/data/Prism-Infer \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_paged_decode_kernel.py \
  /data/Prism-Infer/tests/test_qwen3_vl_attention_kv.py -s
```

```bash
PYTHONPATH=/data/Prism-Infer \
/data/Prism-Infer/.venv-local/bin/python benchmarks/bench_paged_decode.py \
  --batch-sizes 1,2,4,8 \
  --context-lens 256,1024,4096 \
  --warmup 10 \
  --repeat 50
```

验证结果:

- correctness: `5 passed in 2.84s`；合并 shape 测试后为 `6 passed in 4.74s`。
- small GQA:
  - q shape: `[3, 4, 16]`
  - k_cache shape: `[9, 4, 2, 16]`
  - block_tables shape: `[3, 3]`
  - context_lens: `[1, 5, 9]`
  - kernel mean/std: `-1.948629e-02 / 7.964026e-01`
  - reference mean/std: `-1.945430e-02 / 7.964330e-01`
  - max diff: `3.906250e-03`
  - mean diff: `1.447549e-04`
  - PASS
- Qwen shape:
  - q shape: `[2, 8, 128]`
  - k_cache shape: `[6, 16, 2, 128]`
  - context_lens: `[17, 33]`
  - kernel mean/std: `-3.931073e-03 / 3.038969e-01`
  - reference mean/std: `-3.932172e-03 / 3.039157e-01`
  - max diff: `7.812500e-03`
  - mean diff: `2.812790e-04`
  - PASS
- Attention 接入:
  - decode engine output shape: `[1, 4, 16]`
  - decode reference output shape: `[1, 4, 16]`
  - engine mean/std: `1.976967e-02 / 3.161756e-01`
  - reference mean/std: `1.984596e-02 / 3.163016e-01`
  - max diff: `1.953125e-03`
  - mean diff: `3.700256e-04`
  - PASS
- benchmark:
  - commit: `45edd3a`
  - GPU: NVIDIA GeForce RTX 5090
  - torch: `2.6.0a0+ecf3bae40a.nv25.01`
  - dtype: bf16
  - warmup=10, repeat=50
  - num_heads=32, num_kv_heads=8, head_dim=128, block_size=256
  - 12 个 batch/context case 全部 correctness PASS。
  - batch=1/context=256: kernel median `0.0466ms`, reference median `0.1256ms`, max diff `1.953125e-03`
  - batch=1/context=4096: kernel median `0.2839ms`, reference median `0.2265ms`, max diff `4.882812e-04`
  - batch=4/context=1024: kernel median `0.0956ms`, reference median `0.4908ms`, max diff `9.765625e-04`
  - batch=8/context=4096: kernel median `0.4646ms`, reference median `1.8313ms`, max diff `4.882812e-04`

经验:

- Triton baseline kernel 在 batch 增大时明显减少 Python/SDPA reference 的 per-sequence overhead；但单 batch 长上下文不一定更快。
- 高性能 kernel 不能只验证 shape。bf16 下 kernel 与 SDPA reference 的 max diff 在 `1e-3` 到 `8e-3`，必须作为跨实现门槛记录。
- Paged decode kernel 的接口必须显式暴露 block table 和 context lens，不能把调度状态藏在 Python list 里。

剩余风险:

- 当前 kernel 是 P3 baseline，不是最终最优 kernel；batch=1/context=4096 慢于 reference，P6 需要继续优化。
- 当前 benchmark 是 synthetic q/k/v，不代表完整模型端到端吞吐；端到端吞吐已由 P3-005 的 graph benchmark覆盖一部分，但仍需 P6 做系统级 benchmark。

## P3-007: P3 阶段 Review 与完整回归

状态: Verified

发现方式:

- P3.7 阶段门禁。P3.5/P3.6 完成后必须证明 P1/P2/P3 correctness 没有回归，并把 full logits 重型验证串行跑完。

影响范围:

- P1 纯文本 full logits。
- P2 单图图文 full logits 和单图 engine 路径。
- P3 多图、视频、mixed batch、长输出、CUDA Graph、paged decode kernel。
- 文档状态从 “P3.5/P3.6 完成，P3.7 待回归” 更新为 “P3 当前门禁完成”。

验证命令:

```bash
cd /data/Prism-Infer && \
/data/Prism-Infer/.venv-local/bin/python -m compileall prism_infer tests benchmarks
```

```bash
cd /data/Prism-Infer && git diff --check
```

```bash
cd /data/Prism-Infer && \
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  tests/test_processor_pipeline.py \
  tests/test_processor_pipeline_multi_image.py \
  tests/test_processor_pipeline_video.py \
  tests/test_sequence_multimodal.py \
  tests/test_vl_rope_index.py \
  tests/test_vl_rope_index_multi_image.py \
  tests/test_vl_rope_index_video.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_paged_decode_kernel.py \
  tests/test_sampler_greedy.py \
  tests/test_llm_vl_generate.py::test_add_vl_request_builds_single_image_sequence \
  tests/test_llm_vl_generate.py::test_add_vl_request_builds_multi_image_sequence \
  tests/test_llm_vl_generate.py::test_add_video_request_builds_video_sequence \
  tests/test_llm_vl_generate.py::test_add_vl_request_allows_graph_mode_sequence_building \
  tests/test_llm_vl_generate.py::test_generate_vl_one_token_matches_hf_greedy \
  tests/test_llm_vl_generate.py::test_generate_vl_multi_image_one_token_matches_hf_greedy \
  tests/test_llm_vl_generate.py::test_generate_video_one_token_matches_hf_greedy \
  tests/test_llm_vl_mixed_batch_generate.py \
  tests/test_llm_vl_long_generate.py \
  tests/test_llm_vl_cuda_graph_decode.py \
  tests/test_text_only_regression.py \
  tests/test_vision_rope_init.py -s
```

```bash
cd /data/Prism-Infer && \
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python tests/test_full_model.py
```

```bash
cd /data/Prism-Infer && \
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python tests/test_full_model_vl.py
```

```bash
cd /data/Prism-Infer && \
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python tests/test_full_model_vl_multi_image.py
```

```bash
cd /data/Prism-Infer && \
PYTHONPATH=/data/Prism-Infer \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
/data/Prism-Infer/.venv-local/bin/python tests/test_full_model_vl_video.py
```

验证结果:

- `compileall`: PASS。
- `git diff --check`: PASS。
- grouped regression: `49 passed in 356.34s`。
  - 覆盖 processor、sequence、rope、engine attention、ModelRunner、sampler、LLM VL generate、mixed batch、32-token long generate、VL CUDA Graph、teacher-forced logits/ppl、batch numeric sensitivity 和 paged decode kernel。
  - single-image/multi-image/video `max_tokens=32` HF greedy exact，prefix@8/16/32 全部 match。
  - single-image/multi-image/video teacher-forced logits shape `[1,32,151936]`，max diff、mean diff、ppl diff 均为 `0.000000e+00`。
  - mixed batch VL rows 32-token 与 fresh 单请求独立运行一致；text-only row 的 32-token 分叉已由 HF/Prism duplicate batch 数值敏感性解释。
- 纯文本 full logits:
  - shape: HF `[1,64,151936]`, Prism `[1,64,151936]`
  - max diff: `0.000000e+00`
  - mean diff: `0.000000e+00`
  - Result: PASS
- 单图 VL full logits:
  - input_ids shape: `[1,210]`
  - pixel_values shape: `[784,1536]`
  - image_grid_thw shape: `[1,3]`
  - position_ids shape: `[3,1,210]`
  - HF/Prism logits shape: `[1,151936]`
  - HF mean/std: `-1.756945e+00 / 4.123917e+00`
  - Prism mean/std: `-1.756945e+00 / 4.123917e+00`
  - max diff: `0.000000e+00`
  - mean diff: `0.000000e+00`
  - PASS
- 多图 VL full logits:
  - input_ids shape: `[1,408]`
  - pixel_values shape: `[1568,1536]`
  - image_grid_thw shape: `[2,3]`
  - image tokens: `392 / expected 392`
  - position_ids shape: `[3,1,408]`
  - HF mean/std: `-1.318763e+00 / 4.206440e+00`
  - Prism mean/std: `-1.318763e+00 / 4.206440e+00`
  - max diff: `0.000000e+00`
  - mean diff: `0.000000e+00`
  - PASS
- 视频 VL full logits:
  - input_ids shape: `[1,420]`
  - pixel_values_videos shape: `[1568,1536]`
  - video_grid_thw shape: `[1,3]`
  - video tokens: `392 / expected 392`
  - position_ids shape: `[3,1,420]`
  - HF mean/std: `-1.130621e+00 / 4.290061e+00`
  - Prism mean/std: `-1.130621e+00 / 4.290061e+00`
  - max diff: `0.000000e+00`
  - mean diff: `0.000000e+00`
  - PASS

经验:

- P3 回归必须拆分轻量 grouped tests 和重型 full logits；后者串行执行可以避免 8B HF/Prism 同时加载导致 OOM。
- P3.5/P3.6 的性能路径必须和 P1/P2/P3 correctness 一起回归，否则可能引入基础精度退化。
- 阶段完成声明必须等文档、Issue Log 和验证命令都同步后再发布。

剩余风险:

- P3 固定长输出门槛已提升到 `max_tokens=32` greedy 和 teacher-forced logits/ppl；长上下文压力未纳入当前门禁。
- P7.3 已补 chunked paged prefill与 online mixed-VL；VL token-id prefix hash因不同像素
  可共享相同 placeholder ids而显式禁用，不能表述为 VL prefix cache支持。
- 当前 paged decode Triton kernel 是 baseline kernel，batch=1/context=4096 慢于 SDPA reference。
- P3 benchmark 只覆盖本机 RTX 5090；4070/4090、多卡 TP、真实视频文件采样策略、vLLM/SGLang 同条件对比留到 P6/P7。

## P3-008: benchmark 脚本直接运行 import 失败

状态: Verified

发现方式:

- P3 完成审计时按 `docs/VERIFICATION.md` 中的 benchmark 命令直接运行 `benchmarks/bench_paged_decode.py`。

影响范围:

- P3.5/P3.6 的 benchmark 可复现性。
- 不影响模型 correctness，但会导致用户按文档命令复现 benchmark 时直接失败。

证据:

失败命令:

```bash
cd /data/Prism-Infer && \
/data/Prism-Infer/.venv-local/bin/python benchmarks/bench_paged_decode.py \
  --batch-sizes 1,2,4,8 \
  --context-lens 256,1024,4096 \
  --warmup 10 \
  --repeat 50
```

失败输出:

```text
ModuleNotFoundError: No module named 'prism_infer'
```

根因:

- Python 直接执行 `benchmarks/bench_paged_decode.py` 时，`sys.path[0]` 是 `/data/Prism-Infer/benchmarks`，不是 repo root。
- benchmark 脚本位于包目录外，直接 `from prism_infer...` 依赖调用者额外设置 `PYTHONPATH`；这与文档中的直接运行命令不一致。

修复:

- 修改 `benchmarks/bench_paged_decode.py` 和 `benchmarks/bench_vl_cudagraph_decode.py`:
  - 启动时用 `Path(__file__).resolve().parents[1]` 得到 repo root。
  - 若 repo root 不在 `sys.path`，插入到 `sys.path[0]`。
  - 保持 benchmark 核心逻辑不变。

拒绝的替代方案:

- 不只在文档中要求用户手动设置 `PYTHONPATH`；benchmark 脚本应能从 repo root 直接执行。
- 不把 benchmark 移入 `prism_infer` 包内；脚本仍属于项目工具入口，最小修复是入口 bootstrap。

验证命令:

```bash
cd /data/Prism-Infer && \
/data/Prism-Infer/.venv-local/bin/python -m compileall benchmarks
```

```bash
cd /data/Prism-Infer && git diff --check
```

```bash
cd /data/Prism-Infer && \
/data/Prism-Infer/.venv-local/bin/python benchmarks/bench_paged_decode.py \
  --batch-sizes 1,2,4,8 \
  --context-lens 256,1024,4096 \
  --warmup 10 \
  --repeat 50
```

```bash
cd /data/Prism-Infer && \
/data/Prism-Infer/.venv-local/bin/python benchmarks/bench_vl_cudagraph_decode.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --case mixed \
  --max-tokens 4 \
  --warmup 1 \
  --repeat 1
```

验证结果:

- `compileall benchmarks`: PASS。
- `git diff --check`: PASS。
- paged decode benchmark: 12 个 batch/context case 全部 `correctness: PASS`。
  - batch=1/context=256: kernel median `0.0460ms`，reference median `0.1264ms`，max diff `1.953125e-03`。
  - batch=1/context=4096: kernel median `0.2834ms`，reference median `0.2314ms`，max diff `4.882812e-04`。
  - batch=4/context=1024: kernel median `0.0956ms`，reference median `0.4969ms`，max diff `9.765625e-04`。
  - batch=8/context=4096: kernel median `0.4662ms`，reference median `1.8635ms`，max diff `4.882812e-04`。
- VL CUDA Graph benchmark direct-run sanity:
  - case: mixed, `max_tokens=4`, warmup=1, repeat=1, `kvcache_block_size=1024`。
  - eager token ids 与 graph token ids 完全一致。
  - correctness: PASS。
  - eager decode median `48.1290ms`，graph decode median `16.4824ms`，peak memory `27995.47MiB`。

经验:

- benchmark 脚本也是阶段交付的一部分，不能只验证 pytest；文档中的 benchmark 命令必须直接可执行。
- 入口脚本放在 repo 包外时，需要显式处理 repo root import path，避免复现依赖调用者 shell 环境。

剩余风险:

- 这次修复只保证 benchmark 脚本入口可复现；性能优化本身仍按 P3-005/P3-006 的剩余风险处理。

## P4-001: KV trace 接入与样例脚本可复现性问题

状态: Verified

发现方式:

- P4 开发过程中运行新增轻量测试和三类真实样例脚本。

影响范围:

- P4 “trace 文件可复现生成”和“trace on/off 输出一致”门禁。
- 不影响普通推理路径；问题发生在新增 trace 测试和 `scripts/run_kv_trace_samples.py` 入口。

证据:

轻量测试首次失败 1:

```text
ValueError: pixel_values and image_grid_thw must be provided together
```

轻量测试首次失败 2:

```text
RuntimeError: FlashAttention only support fp16 and bf16 data type
```

修复后真实样例首次失败:

```text
Traceback (most recent call last):
  File "/data/Prism-Infer/scripts/run_kv_trace_samples.py", line 23, in <module>
    from prism_infer import LLM
ModuleNotFoundError: No module named 'prism_infer'
```

定位过程:

- 第一个失败来自 `Sequence.__init__` 的成对 VL payload 校验。测试只传了 `image_grid_thw/video_grid_thw`，没有传对应的 `pixel_values/pixel_values_videos`，违反现有输入边界。
- 第二个失败来自 CPU/float32 单测在安装了 flash-attn 的环境中走到 `flash_attn_varlen_func`，而 flash-attn 不支持 fp32。trace on/off 单元测试目标是验证 trace 不改变输出，不应依赖 flash-attn dtype 和 GPU。
- 第三个失败与 P3-008 类似: 直接执行 `scripts/run_kv_trace_samples.py` 时，`sys.path[0]` 是 `scripts/`，repo root 不在 import path。
- 过程中还发现 CPU fallback 的 `store_kvcache` 对 paged cache 形态和 slot 语义与真实 GPU Triton 路径不同；该差异不属于 P4 目标，测试改为 `slot_mapping=-1` 使用预填 paged cache 做只读 decode trace，避免把无关 fallback 问题混入 P4。

根因:

- 测试构造没有完全遵守现有 `Sequence` VL payload invariant。
- P4 trace on/off 测试初版选择了 prefill fp32 路径，错误地暴露给 flash-attn 可选依赖。
- 新增脚本位于包目录外，缺少 repo root bootstrap。

修复:

- `tests/test_analysis_schema.py` 为 schema 构造补齐 `pixel_values` 和 `pixel_values_videos`，保持与 `Sequence` 约束一致。
- `tests/test_kv_trace_no_output_change.py` 改为小张量 decode eager fallback 路径，比较 trace off/on 输出 max diff，并检查 trace record 中 visual attention mass。
- `scripts/analyze_kv_trace.py` 和 `scripts/run_kv_trace_samples.py` 启动时将 repo root 加入 `sys.path`，保证文档命令可直接执行。
- `prism_infer/analysis/kv_trace.py` 避免对整块预分配 KV cache 做 mean/std，只记录 cache 元信息，对当前 q/k/v 和有效 span 做统计，避免扫描未使用 cache 槽。

验证命令:

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m compileall prism_infer tests scripts
```

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m pytest -q \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_kv_trace_no_output_change.py -s
```

```bash
cd /data/Prism-Infer && \
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
.venv-local/bin/python scripts/run_kv_trace_samples.py \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

验证结果:

- `compileall prism_infer tests scripts`: PASS。
- P4 轻量测试: `4 passed in 1.46s`。
- trace on/off 小张量 decode:
  - output shape: `[1, 2, 4]`
  - max diff: `0.000000e+00`
  - mean diff: `0.000000e+00`
  - visual attention mass: `4.361440e-01`
- 三类真实样例:
  - `single_image_description`: token ids `[32, 6303]`，layer records `72`，phases `decode/prefill`。
  - `single_image_detail_qa`: token ids `[2518, 151645]`，layer records `72`，phases `decode/prefill`。
  - `multi_image_comparison`: token ids `[28715, 389]`，layer records `72`，phases `decode/prefill`。
  - manifest result: `PASS`。

经验:

- 分析脚本也是阶段门禁的一部分，必须像 benchmark 脚本一样能从 repo root 直接执行。
- trace on/off 测试应隔离“是否改变输出”这个核心问题，避免被 flash-attn dtype、GPU 可用性或 CPU fallback 形态差异干扰。
- KV trace 不能扫描整块预分配 cache 做统计，否则会把未使用槽位和性能开销引入分析结果。

剩余风险:

- CPU `store_kvcache` fallback 与 paged cache 形态的兼容性不是本次 P4 修复范围；真实 P3/P4 GPU 路径使用 Triton store 和 paged decode。
- P4 trace 是分析路径，会增加同步和 JSON 序列化开销，不能用于 benchmark。
- P4 只形成压缩假设，不代表 P5 compression 已实现或有收益。

## P4.5-001: KV layout、prefix hash 与 swap block_table 语义硬化

状态: Verified

发现方式:

- 用户通过多 agent 专家诊断指出 KV 子系统存在 P0/P1 结构债。
- 本轮按代码证据复核 `attention.py`、`paged_decode.py`、`block_manager.py`、`model_runner.py`、`scheduler.py` 和既有 P3/P4 文档。

影响范围:

- P5 KV Cache 压缩算法的地基。
- Prefix cache、swap、paged decode、KV trace 与后续高性能服务调度。
- 不直接影响 P3/P4 已验证的 non-prefix、non-swap、GPU paged decode baseline，但如果不修，会阻断后续 compression 和生产级 engine。

证据:

- `ModelRunner.allocate_kv_cache` 给每层 attention 分配的 cache 是 4D paged layout: `[num_blocks, block_size, num_kv_heads, head_dim]`。
- `paged_decode_attention` 明确要求 `k_cache/v_cache` 为 `[num_blocks, block_size, num_kv_heads, head_dim]`。
- `store_kvcache` 的 GPU Triton 路径按 `slot * D + arange(D)` 对 contiguous 4D cache 做 flat slot 写入；但 CPU fallback 原先直接 `k_cache[slot] = key[i]`，会把 flat slot 错当第一维 block id。
- `BlockManager._deallocate_block` 原先只把 block 从 used 移到 free，不清理 `hash_to_block_id`。
- `BlockManager.swap_out` 原先把 `seq.block_table` 从 GPU block id 替换成 CPU block id；同一个字段在不同状态下含义不同。
- `ModelRunner.prepare_prefill` 原先在 prefix cache hit 时构造 `block_tables`，最后由 attention 层在 prefill + block_tables 路径报错；失败位置偏晚。

根因:

- KV layout 缺少明确 contract。真实主路径已经是 4D paged cache，但 fallback 和注释中存在 flat 2D 直觉。
- Prefix cache hash index 没有和 block 生命周期绑定；free block 仍可被 stale hash 指向。
- Swap 逻辑把 GPU/CPU 两个地址空间复用在 `seq.block_table` 一个字段上，导致后续 prepare/trace/decode 路径难以静态判断字段语义。
- Prefix-cache prefill 尚无 paged prefill attention kernel，却没有在 ModelRunner 阶段早停。

修复:

- `prism_infer/layers/attention.py`
  - 新增 `_store_kvcache_eager`。
  - CPU fallback 支持 canonical 4D paged cache: `block_id = slot // block_size`，`block_offset = slot % block_size`。
  - 保留 legacy `[slots, heads, dim]` fallback 只用于小形态单测。
- `prism_infer/engine/block_manager.py`
  - 新增 `free_block_id_set`，避免指定 block 分配依赖 `deque.remove` 的 O(n) 删除。
  - 新增 `_remove_hash_index_for_block`，block 释放或重新分配前清理仍指向该 block 的 hash index。
  - `_allocate_free_block` 从空闲队列头取真实 free block，跳过过期项。
  - `deallocate` 同步清理 `seq.cpu_block_table`。
- `prism_infer/engine/sequence.py`
  - 新增 `cpu_block_table` 字段。
  - 序列化/反序列化保留该字段。
  - `block_table` 明确只表示 GPU block id。
- `prism_infer/engine/block_manager.py`
  - `swap_out()` 不再把 CPU block id 写入 `seq.block_table`；改为写入 `seq.cpu_block_table` 并清空 GPU table。
  - `swap_in()` 要求 `seq.block_table` 为空、`seq.cpu_block_table` 非空，换入后恢复 GPU table 并清空 CPU table。
- `prism_infer/engine/scheduler.py`
  - swapped queue 的换入容量判断改用 `block_manager.can_swap_in(seq)`，后者基于 `cpu_block_table`。
- `prism_infer/engine/model_runner.py`
  - `prepare_block_tables/prepare_prefill/prepare_decode` 拒绝仍带 `cpu_block_table` 的 swapped sequence。
  - prefix-cache prefill 在 `prepare_prefill` 阶段显式 `RuntimeError`，不再拖到 attention 层。
- 新增测试:
  - `tests/test_kv_engine_hardening.py`
  - `tests/test_scheduler_swap_tables.py`

拒绝的替代方案:

- 不把主 KV cache 改成全局 2D flat layout。现有 `ModelRunner`、paged decode kernel 和 P3/P4 实测路径都以 4D paged layout 为物理真相；修 fallback 和 contract 风险更小。
- 不继续允许 free block 作为 prefix cache 命中对象。当前没有 cached/reserved 独立状态，保留 stale hash 会让 free list 和 prefix cache 生命周期混在一起。后续如果需要持久 prefix cache，应显式设计 cached block 状态。
- 不在 P4.5 实现 paged prefill kernel。P4.5 目标是 hardening，不是扩展新 kernel；paged prefill 属于后续 P6/P5 交叉任务。
- 不在 P4.5 直接修改 `kvcache_block_size=256`。该改动会牵动 P3 correctness/benchmark 和 FlashAttention 约束；P5.0 会作为压缩粒度设计门禁单独处理。

验证命令:

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m compileall prism_infer tests
```

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m pytest -q \
  tests/test_kv_engine_hardening.py \
  tests/test_scheduler_swap_tables.py -s
```

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m pytest -q \
  tests/test_sequence_multimodal.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_prefill.py \
  tests/test_model_runner_vl_mixed_prefill.py \
  tests/test_kv_trace_no_output_change.py -s
```

```bash
cd /data/Prism-Infer && \
.venv-local/bin/python -m pytest -q \
  tests/test_kv_engine_hardening.py \
  tests/test_scheduler_swap_tables.py \
  tests/test_paged_decode_kernel.py \
  tests/test_qwen3_vl_attention_kv.py -s
```

验证结果:

- `compileall prism_infer tests`: PASS。
- P4.5 focused invariant tests:
  - `tests/test_kv_engine_hardening.py` + `tests/test_scheduler_swap_tables.py`: `5 passed`。
  - 4D paged KV store: input shape `[5,2,3]`，cache shape `[3,4,2,3]`，slot_mapping `[0,3,4,9,-1]`，K/V max diff `0.000000e+00`。
  - BlockManager hash cleanup: deallocate 后 `hash index keys after deallocate: []`。
  - swap table split: `swap_out` 后 GPU `block_table=[]`、CPU `cpu_block_table=[0,1]`；`swap_in` 后 CPU table 清空，GPU table 恢复。
  - scheduler swap-in: 使用 `cpu_block_table` 后换入成功，`seq.cpu_block_table=[]`。
  - prefix-cache prefill early gate: PASS。
- 受影响窄回归:
  - `12 passed in 11.87s`。
  - engine attention prefill KV: output/K/V cache max diff `0.000000e+00`。
  - engine attention decode paged KV: output max diff `1.953125e-03`，mean diff `3.700256e-04`。
  - mixed prefill/decode 和 KV trace on/off 均 PASS。
- Paged decode + attention regression:
  - `10 passed in 4.98s`。
  - paged decode small GQA max diff `3.906250e-03`，mean diff `1.447549e-04`。
  - paged decode Qwen shape max diff `7.812500e-03`，mean diff `2.812790e-04`。

经验:

- “运行时必然崩溃”这类诊断要拆成路径级判断。GPU 主路径的 flat slot 写入 4D contiguous cache 是可工作的；真正错误的是 CPU fallback 和缺少 layout contract。
- BlockManager 不能让 free list、prefix cache、swap 三种状态隐式共享同一字段或索引。后续压缩只会进一步增加状态复杂度，必须先把生命周期写清楚。
- 对 unsupported 功能，越早失败越好。prefix-cache prefill 未实现时，ModelRunner 准备阶段报错比 attention 内部报错更容易定位，也避免 trace/metadata 生成半成品状态。

剩余风险:

- prefix-cache prefill 仍未实现；P4.5 只做 early gate。
- `kvcache_block_size=256` 仍然对 visual-token KV 压缩粒度偏粗；P5.0 必须设计 block-size/sub-page metadata。
- swap 数据搬运仍使用全局 `torch.cuda.synchronize()`；这是 P6 性能问题。
- paged decode kernel 的 `MAX_CONTEXT_LEN` 冗余迭代、`BLOCK_N=32` 固定调优仍未处理；这是 P6 kernel 性能问题。
- 本轮未运行 P1/P2/P3 全量 grouped regression 和 full logits 串行重型验证；合并前如要发布阶段 release，应按 `docs/VERIFICATION.md` 重新跑全量门禁。

## P5-001: 外部评估对照后的 P5 readiness correctness 修复

状态: Verified

发现方式:

- 用户提供 2026-07-06 外部评估文本，指出 P1/PB/C 类问题和 P5/P6 技术路线建议。
- 本轮按 `CLAUDE.md` / `prism-infer-rigor` 要求先复核源码、测试和文档，再决定修复范围。

影响范围:

- P5 active compression 前的 engine correctness。
- Sequence 跨进程序列化、swap/prefix-cache hash 生命周期、ModelRunner context 生命周期和 scheduler 空调度错误路径。
- P5 路线文档，避免把 FP8/VScan/PoRe/DeepStack-aware pruning 等未实现候选方案写成已验证能力。

采纳并修复:

- B2: `Sequence.__setstate__` 原先把 decode 反序列化后的 `temperature/max_tokens/ignore_eos` 重置为默认值。本轮在 `__getstate__/__setstate__` 中保存和恢复 sampling 参数。
- P1-2: `BlockManager.swap_in()` 原先依赖 `seq.block(i)` 重算 hash；decode 反序列化对象可能没有完整 `token_ids`。本轮在 `swap_out` 保存 CPU block hash 和满块 token 副本，`swap_in` 用 metadata 恢复 prefix-cache index。
- B3: `ModelRunner.run()` 原先只在正常路径末尾 `reset_context()`。本轮改为 `try/finally`，异常时也清理 context，并恢复 chunked prefill 临时截断状态。
- B4: `Scheduler.schedule()` 原先用 `assert scheduled_seqs`。本轮改为显式 `RuntimeError`，避免 `python -O` 跳过。
- C-4: `Attention.forward()` 原先总是调用 trace 记录函数。现在先检查 `is_trace_enabled()`，trace 关闭时不构造记录调用。
- C-7: `tests/test_paged_decode_kernel.py` 新增 mean diff `< 1e-3` 门槛。
- P1-1: 在已有 `Sequence.set_block_size()` 和 BlockManager mismatch gate 基础上，本轮让每个 `Sequence` 保存实例级 `block_size` 快照，避免构造后被后续 Config 全局同步污染。
- B1: 当前 engine flatten VL 路径未复现 DeepStack 缺失。`ModelRunner._forward_model()` 会传递 visual payload，`Qwen3VLModel.forward()` 会构造 `visual_pos_masks/deepstack_visual_embeds`；本轮补轻量回归覆盖该路径。

暂缓:

- C-1 `_cfg_get` 抽取、C-2 compression/config 依赖拆分、C-3 端口和 SHM 参数化、C-5 decode trace 开关、C-6 visual importance aggregate 合并均属于非阻断重构或 schema 设计项，本轮不扩大范围。
- FP8 KV、VScan/PoRe、M-RoPE physical compaction、vLLM/SGLang 对比和具体性能数字未做外部查源或本地同条件 benchmark；只能作为候选路线，不能作为项目完成 claim。

验证命令:

```bash
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_kv_engine_hardening.py \
  /data/Prism-Infer/tests/test_scheduler_swap_tables.py \
  /data/Prism-Infer/tests/test_model_runner_context_reset.py \
  /data/Prism-Infer/tests/test_full_model_structure.py -s
```

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_sequence_multimodal.py -s
```

```bash
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_importance_scoring.py \
  /data/Prism-Infer/tests/test_visual_token_stats.py \
  /data/Prism-Infer/tests/test_analysis_schema.py -s
```

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall -q \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests \
  /data/Prism-Infer/scripts
```

验证结果:

- KV/scheduler/context/DeepStack focused: `17 passed in 3.57s`。
- Sequence multimodal roundtrip: `5 passed in 4.35s`。
- P5/P4 analysis focused: `11 passed in 1.57s`。
- `compileall`: PASS，无错误输出。

剩余风险:

- 本轮未运行 P1/P2/P3 全量 grouped regression 和 full logits 重型验证。
- P5.2 active compression 尚未实现；当前仍不能声明 compression ratio、显存收益、latency/throughput 收益或质量收益。
- P5 路线中的 FP8/VScan/PoRe/physical compaction 需要单独设计、实现和 benchmark。
