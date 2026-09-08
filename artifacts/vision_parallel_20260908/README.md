# 多图视觉并行与两卡策略记录

实现和结论见 [MULTI_GPU.md](../../docs/MULTI_GPU.md)，机器可读汇总见
[summary.json](summary.json)。这组记录没有替换历史 TP1 Decode 或 600 请求工作集数据。

## 运行环境与来源

Qwen3-VL-8B-Instruct 复用训练营个人目录里的现有模型；两张 RTX 5090，驱动 580.142，
Prism 使用 PyTorch 2.11.0+cu130、Transformers 5.14.1。Prism 为 Scaled-FP8 KV + Decode
CUDA Graph，vLLM 0.25.1 的 TP/PP 方向参照为 BF16 KV + eager、async_scheduling=False。
vLLM 使用自己的 Processor/Encoder/Prefix Cache，Prism 的独立 Vision output cache 关闭。
两套框架的绝对时延不能据此直接排名。

本地工作基于 `104b2bf2ff649bdc75f376878b6c16c7a96a49cb`，分支为
`codex/vision-dp-20260908`。远端 GitHub 拉取发生 TLS 中断，因此将本地源码同步到独立的
`/home/lcpu/87120912/Prism-Infer-vision-dp`。该目录不是 Git checkout，原始 JSON 中
`git.commit=unknown` 如实保留；代码改动和下面的阶段关系由本次本地提交保存。
没有覆盖原远端 `Prism-Infer` 中的未提交文件，也没有重新下载模型或创建新环境。

所有 GPU 任务均经 Slurm，单次限制 15 分钟；实际最长任务 4 分 45 秒。新增源码、缓存、
日志都在 `/home/lcpu/87120912` 内。后续复现可从仓库的 `configs/tp2_vision_data.json`
启动，而不是修改依赖或模型配置。

## 哪个文件回答哪个问题

| 作业 | 记录 | 用途 |
|---|---|---|
| 9759 | [vision-features-9759.json](raw/vision-features-9759.json) | 真实 Vision 权重；等尺寸、不等尺寸、单图；逐 rank 时间和特征误差 |
| 9760 | [失败请求](raw/tp2-replicated-9760.json) | 热 Prefix 尾页复制命令未登记，后续补齐 TPMethod；不作为完整性能运行 |
| 9763 | [vision-partition-9763.json](raw/vision-partition-9763.json) | 单 GPU 移除通信变量，逐层 BF16/FP32 对照定位数值差异 |
| 9764 | [replicated](raw/tp2-replicated-9764.json)、[data](raw/tp2-data-9764.json) | 两种 TP2 模式，已修尾页命令，但热 Prefix 仍序列化完整图片 |
| 9765 | [Muir replicated](raw/muir-replicated-9765.json)、[Muir data](raw/muir-data-9765.json) | 现有 materialization 中前 20 条含 2–8 图片的请求；两模式输入完全相同 |
| 9766 | [双副本](raw/replicas-9766/dp2_summary.json) | 两独立 TP1，各 4 请求，两阶段同步开始；原始记录在同一目录 |
| 9761 / 9767 | [vLLM TP2](raw/vllm-tp2-9761.json)、[成功 PP2](raw/vllm-pp2-9767.json) | 同 vLLM 内的并行策略参照；首次 PP2 启动失败也保留 |
| 9769 | [replicated](raw/tp2-replicated-9769.json)、[data](raw/tp2-data-9769.json) | 省略已覆盖图片的序列化后，按 replicated→data 运行；另存最终 Trace |
| 9770 | [replicated](raw/tp2-replicated-9770.json)、[data](raw/tp2-data-9770.json) | 相同代码交换为 data→replicated 次序，检查顺序影响；不是挑最快一轮 |

9769、9770 都保留。两次的冷 TTFT 变化幅度不同，因此文档给出运行范围，不把视觉模块
1.86×写成端到端 1.86×。完整 Prefix 热命中不执行 Vision，两种 Vision 模式不应从该
路径获得不同的算法收益；这次热路径改善来自少传无用媒体。

## 请求和质量口径

性能 fixture 为八张 448×448 彩色图片，媒体在前、问题在后；greedy、固定输出 16 tokens。
Prism 顺序请求预热一次、测五次；冷重复关闭 APC，另记录 APC-on 首次建缓存，热请求再
复用它。并发为有限的 8 请求批次，不是 Poisson 到达或饱和吞吐。Prism 先提交整个批次
再驱动引擎，CPU 请求准备时间也包含在内。原始记录保留全部 token、到达时间、输入 grid、
Prefix 覆盖与 KV 字节；TTFT 从 add-request 前开始，文件读取和模型启动在计时外。

双副本各持有 1,245,708,288 bytes KV，合计是 TP2 的两倍；按两个副本共同时间跨度计算
总吞吐，不能将局部 tok/s 简单相加，也不声称相同 KV 预算。

Muir 的 20 条检查没有进行 Token Pruning，目的仅为对比新旧 Encoder 路径。脚本当时为
固定输出长度设置 `ignore_eos=True`，旧的 `prediction/strict_correct` 对 EOS 后文本
进行了解析，不能作为正常答题准确率。原始字段不改写，汇总从原始 token IDs 截取首个
EOS（151645）前的输出；没有 EOS 时取已记录的 16-token 上限。结果为该范围 20/20
一致、有效选项 19/19 一致，两模式均 11/20 答对；Sample 1236 同样未输出可解析选项。
强制输出的完整 16 tokens 只有 16/20 一致，差异都在 EOS 后。新请求脚本已按 EOS 截断
再解析，避免继续产生这种统计歧义。这不是官方 MuirBench 全量准确率。

## 如何看流水

[trace.json](trace.json) 可导入 Perfetto 或 Chrome Trace Viewer。
[trace_summary.json](trace_summary.json) 列出冷请求和完整 Prefix 命中各自的 Vision/
聚合次数及 GPU 时间；[原始 Nsight 文件](raw/vision-dp-9769.nsys-rep) 可以在 Nsight Systems
打开。重点看 `trace_cold`、`trace_full_prefix_hit` 与两个 GPU 的 kernel 时间线。
冷请求的 encode/gather 各有两个 rank 调用，表示每个 rank 各一次、同一轮 collective，
不是两轮聚合；完整 Prefix 命中两者均为零。
两张卡的累计 kernel 时间不能相加后当作端到端耗时；带 profiler 的请求时间也不替代
普通运行的性能值。SQL 是从 Nsight 文件导出的中间格式，不需要重复保存副本。

## 复现汇总

```bash
python artifacts/vision_parallel_20260908/summarize.py \
  --raw-dir artifacts/vision_parallel_20260908/raw \
  --output artifacts/vision_parallel_20260908/summary.json \
  --final-replicated tp2-replicated-9770.json --final-data tp2-data-9770.json \
  --replicas-summary replicas-9766/dp2_summary.json --vllm-pp2 vllm-pp2-9767.json
```

直接相关的检查覆盖了图片顺序和 DeepStack、空 rank、完整/部分 Prefix 的媒体序列化、
CoW 命令到 worker 的参数与 ACK，以及真实 A+B→A+C 的输出。没有新增通用 CI。
