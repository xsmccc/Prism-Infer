# 热 Prefix 后缀 Prefill 优化记录

父版本 `cc77b08`。最终改动为短后缀 Q/K RMSNorm-M-RoPE 融合、一次完成 K/V gather与
反量化，以及修复初始化遗留的 PyTorch DeviceContext。机制与完整解释见
[PREFIX_PREFILL_OPTIMIZATION.md](../../docs/PREFIX_PREFILL_OPTIMIZATION.md)。

最终同图换问题 TTFT 为 208.37→144.61 ms，两次配对分别降低 27.72%/33.29%。
这不是 vLLM/SGLang 排名；原始输出存在少数长请求最后一个 token 的跨运行差异，基线
重复运行也有差异。位置详见 [summary.json](summary.json)，不将 fixture 当成质量数据集。

## 结果与文件

| 记录 | 用途 |
|---|---|
| `http-fusion-{baseline-1,candidate-1,candidate-2,baseline-2}-9816.json` | 最终旧→新→新→旧 HTTP 请求级记录，每轮11请求、624 tokens |
| `server-fusion-*-9816.json` | 最终配置、模型路径、GPU/PyTorch与Engine metrics |
| `http-fusion-*-9813.json` | 仅有两项融合时的配对记录；热TTFT未稳定改善，不用它作最终收益 |
| `cpu-{cooperative,prefill-fused}-9815.json` | 发现残留DeviceContext的CPU函数调用统计 |
| `cpu-{cooperative,prefill-fused}-9817.json` | 修复后该设备模式的调用从约13089次变为0 |
| [ablate-prefill-fused-9819.json](raw/ablate-prefill-fused-9819.json) | 修复设备模式后，同进程切换融合：Prefill step 57.82→40.31 ms，输出一致 |
| [summary.json](summary.json) | 原始指标、配对变化、输出差异与早期仅融合结果 |
| [trace.json](trace.json)、[trace_summary.json](trace_summary.json) | 热后缀的CPU区间、GPU kernel及分项计数 |
| [fusion-9818.nsys-rep](raw/fusion-9818.nsys-rep) | 最终原始Nsight记录 |
| [fusion-9814.nsys-rep](raw/fusion-9814.nsys-rep) | 仅融合、未修设备模式时的原始Nsight记录 |

以上请求、CPU统计均在 `raw/`。另保留9807/9809/9811早期单轮记录，分别对应初始参考、
Q/K短行融合、Q/K动态长度版本；9809第一次新后缀长度触发编译，不混入最终汇总。

## 协议与复现

RTX 5090、driver580.142、PyTorch2.11.0+cu130、Qwen3-VL-8B-Instruct，TP1、最大长度4096、
64×256-token Scaled-FP8 KV页，Prefix与共享CPU缓存开启，cooperative及token chunking
关闭。完整参数沿用[engine.json](../shared_preprocessing_20260908/engine.json)。

输入为八张448×448图片，同图不同问题与长Decode期间插入冷图片；长/短输出128/16，
greedy、ignore_eos。HTTP计时包括媒体准备和排队，模型初始化及首条预热不计入收益。
每个统计先取每轮请求中位数，再汇总两轮，不是全局p99。

远端源码分别为 `/home/lcpu/87120912/preprocessing/cooperative`（父版本）和
`prefill-fused`（候选/最终）；模型、环境、编译cache和所有输出均复用该用户目录。
所有GPU命令在15分钟以内的Slurm作业运行。

```bash
sbatch run_fusion_checks.sh tests/test_prefill_qk_rmsnorm.py tests/test_paged_kv_gather.py tests/test_paged_prefill_fast.py
sbatch run_fusion_pair.sh
sbatch run_cpu_profile.sh
sbatch run_fusion_trace.sh
sbatch run_ablate.sh
```

服务器脚本由本目录的 `run_checks.sh`、`run_pair.sh`、`run_cpu_profile.sh`、
`run_trace.sh`、`run_ablate.sh`同步；其中前两项和trace加了`fusion`前缀区分既有实验。
`run_pair.sh`沿用[HTTP服务入口](../../benchmarks/serve_preprocessing_study.py)与
[客户端](../../benchmarks/bench_serving_preprocessing.py)，同一allocation内顺序运行四个
独立进程。[profile_cpu.py](profile_cpu.py)仅做CPU归因，不用于性能排名；
[ablate_fusion.py](ablate_fusion.py)只在基准脚本的单进程中切换分发，不新增产品开关。

将raw拷回本目录后，执行 `python analyze.py` 重建最终HTTP汇总。导出以下SQLite后，
同一脚本还能重建Trace计数和展示JSON：

```bash
nsys export --type sqlite --output raw/fusion-9818.sqlite raw/fusion-9818.nsys-rep
nsys export --type sqlite --output raw/fusion-9814.sqlite raw/fusion-9814.nsys-rep
```

旧代码Trace引用相邻 `cooperative_prefill_20260908/raw/prefill-9802.nsys-rep`，不重复
存储；需要重建旧计数时在其目录导出同名SQLite。SQLite为可重建中间文件，不提交。
额外的初始化作用域和能力检测检查不需要GPU，见对应tests文件。
