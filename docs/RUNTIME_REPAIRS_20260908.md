# 容量压力、后端选择与请求准备修复

本轮针对 `f880105` review发现的实际问题修复，不新增推理算法或性能排名。
原来的单请求、固定64页实验不能覆盖所有Swap、Chunked Prefill和自动TP容量组合，
因此保留那些实验的原有含义，用针对性请求检查本轮改动。

## KV搬运必须先读完源，再复用槽位

Scheduler先为swapped请求规划换入，再按Decode追加需求抢占、分配新页或CoW。
BlockManager在规划时便更新引用和空闲表，真正搬运由Executor稍后执行。旧的
CoW→swap-out→swap-in顺序与这个规划次序不匹配。

两条真实Scheduler产生的计划说明了问题：

- CoW的目的GPU页也是待换出请求的源页，先CoW会使CPU收到另一请求的数据。
- 换入释放的CPU槽在同轮又被用于换出，先swap-out会覆盖尚未读回的CPU内容。

现在 [Executor](../prism_infer/engine/executor.py) 执行
**swap-in→swap-out→copy-prefix→CoW→model**。换入目的页在该轮开始时是空闲的；先读完
CPU源，再保存待换出GPU页，最后才能把释放的GPU页用于CoW。保留CoW之间的规划顺序，
没有为每次搬运额外复制整个KV池。

回归使用真实Scheduler、BlockManager和ModelRunner复制方法，仅用CPU张量承载存储。
旧执行器两条均破坏payload，新执行器的FP8 payload字节和FP32 scales都一致。
这验证搬运依赖，不是量化质量测试。

## 满池时让尚未完成的Prefill继续推进

普通Chunked Prefill过去仍可能抢占唯一Decoder，随后因Decode批次为空而抛异常。
已经修过的cooperative分段路径不代表普通schedule也正确。

现在只要有未完成的Prefill，轮到Decode时先尝试不抢占的驻留请求；如果没有可追加
token的Decoder，则继续Prefill。Prefill完成后回到普通调度，不生成带未执行Swap操作的
空Decode批次。[Scheduler](../prism_infer/engine/scheduler.py)保留原有正常抢占路径。

真实模型检查使用4个256-token页：A为254-token Prompt、输出8；B为768-token Prompt、
输出1；chunk为256。两条请求各自能放入页池，混跑时仍需处理跨页压力。修复后两条均完成，
没有recompute preemption。

## TP启动使用共同KV容量

自动容量以前由每张卡按本地可用显存独立决定，但共享页表只采用rank0的容量。
worker可用空间较小时，页表可能引用它不存在的页。

现在 `_select_num_kv_blocks()` 在TP模式下先用NCCL MIN归约本地可用页数，再进行自动
选择或显式容量校验；空间不足的rank也先参加归约，使各rank共同决定失败，避免一边
退出、一边继续分配。TP1不增加collective。

CPU检查覆盖容量不同、显式超限和无可用页；两张5090上的NCCL检查注入10/6页的本地
上限，实际归约为6，显式请求7页时两边均拒绝。这是容量函数和真实collective检查，
注入值不是两张卡实际可用显存，也不是新的TP2模型吞吐测量。

## 默认多图配置与视觉缓存准备

默认关闭token-level Chunked Prefill，与当前已验证的多图路径对齐。显式开启时仍要求
完整视觉区间能放进chunk，本轮没有实现新的多图token切分算法。
[tp1_fp8.json](../configs/tp1_fp8.json)给出TP1、Scaled-FP8 KV、CUDA Graph的使用配置；
KV容量自动计算，`compile_region`仍为none，不是历史FP8 LM-head候选投影的配置。

提交请求现在只做Prefix探测与入队。被选中的Prefill在
`ModelExecutionBackend.prepare()`的`runner.prefill.visual_cache`阶段，调用
`ModelRunner.prepare_prefill_visual_cache()`。完整视觉Prefix命中不恢复视觉特征；
当本次slice覆盖全部视觉payload时才查询或构建Encoder Cache。部分视觉Prefix命中
继续由InputPreparer按完整图片切片，只计算未覆盖媒体，不承诺该分支一定复用Encoder Cache。

逐图缓存全部命中时，不再先搬运原始pixels；只恢复已有视觉特征。缺失图片才需要原始
payload。Encoder Cache miss目前仍在选定Prefill的准备阶段整体编码，不应描述成它也能
在每个Vision block间让出GPU。

实际Qwen3-VL检查通过默认三图、同图换问题的完整Prefix命中、重排图片的Encoder Cache
复用及无缓存输出对照。超长新图片请求被拒绝时，视觉缓存计数保持不变，说明拒绝前没有
运行Vision。

## FlashInfer不再静默切换后端

显式请求FlashInfer但依赖不可用、KV格式不支持，或选择了与其冲突的attention-compile
Decode路径时会报错；运行时Decode异常也不再被吞掉。默认不启用该可选后端。

同时修复了适配器的具体问题：异长batch的有效页必须逐请求打包，不能把带padding的整张
页表flatten后配上packed indptr；上下文跨页后需要重新plan，不能只看batch/width是否
相同；Graph使用捕获桶的wrapper并填充安全dummy行；eager页表宽度增长时重建其wrapper。
metadata staging与plan在图外执行，静态buffer地址不变。这次保证正确更新，没有宣称
每层每步replan是性能最优方案。

FlashInfer0.6.13、BF16、page256的原生Graph检查覆盖异长、同宽跨页和实际batch小于
捕获桶，与SDPA参考的最大绝对差均为0.0009765625。这是数值检查，不是bit-exact声明。

## 文档和学习材料

仓库及本地教学材料纠正了compile范围、不存在的全词表回退、请求执行顺序和剪枝参数。
768是历史工作集显式配置，不是当前默认的最小保留量；最小保留量按请求的视觉token总量
计算，而不是对每张图分别设置。默认不启用Token Pruning。

删除了`layers/paged_attention_reference.py`及其旁边的重复测试，现用的`ops`实现和
`tests/test_paged_prefill_fast.py`保留；旧文件可以从Git历史恢复。历史
[FP8 LM-head证据](../artifacts/fp8_lm_head_20260902/README.md)补入公开入口，不与KV量化混算。

所有验证记录和脚本见[本轮记录](../artifacts/runtime_repairs_20260908/README.md)。
既有TTFT/TPOT数字没有改写；这轮没有重跑vLLM/SGLang或新增性能结论。
