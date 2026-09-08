# 共享 CPU 预处理与真实 HTTP 对照

实现和线程流程见 [SHARED_PREPROCESSING.md](../../docs/SHARED_PREPROCESSING.md)。
这组测量对比 Git 提交 `8145043` 与本轮共享缓存/后台准备代码，不修改之前的 TP1、TP2
或 600 请求工作集结果，不做跨引擎排名。

## 测量方式

同一客户端通过原生 `POST /v1/generate` 读取 SSE。模型初始化和 fixture 编码在计时外，
HTTP 发送、服务端图片解码、CPU 预处理、排队和模型执行均计入 TTFT/E2E。一个网络 read
完成的多个 SSE 事件共用该 read 的时间戳，不插值构造不存在的 GPU step 时间。

输入为八张 448×448 纯色图片，Qwen3-VL-8B，TP1、RTX 5090、Scaled-FP8 KV、Decode
CUDA Graph，配置见 [engine.json](engine.json)。Vision output cache 关闭，Prefix KV
保持开启。每次服务启动后先完成一个预热请求，再运行：

- 三个同媒体、不同问题的短请求，每个固定输出 16 tokens。
- 一个单独的 128-token 请求作为无新图到达时的参考。
- 三次长请求，每次观察到至少三个 token 后发送另一组内容不同的冷图片，长/短输出
  分别为 128/16 tokens。原始记录证实，三次注入时长请求都尚未结束。

`max_itl_ms` 是先取每个请求的最大 token 间隔，再对三个长请求取中位数，不是总体 p99。
预热时间不参与效果比较。fixture 是计算/请求时序负载，不是 MuirBench 或任务准确率评估。

## 结果

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 同图换问题 TTFT 中位数 | 317.96 ms | 189.14 ms |
| 同图换问题 E2E 中位数 | 485.11 ms | 352.25 ms |
| 插入冷图时，长请求最大 ITL 的中位数 | 388.49 ms | 216.54 ms |
| 注入的冷图请求 TTFT 中位数 | 419.90 ms | 434.61 ms |

冷图自身并未加速：它仍要完成全部媒体处理和 GPU Prefill，CPU worker 也会与 owner
争用 CPU 资源。此次改善的是重复媒体请求，以及已有 Decode 被 CPU 准备阻塞的部分。
冷 GPU Prefill 造成的停顿仍然存在。11 个请求、每轮 624 个生成 token 完整一致，
每条 SSE token 流也与最终返回 IDs 一致；这不等于所有输入或概率分布都被证明相同。

修改后的 cache 统计为 4 次 miss、7 次 hit，其中 4 次重新绑定问题；对应预热一组媒体
和三组冷图片，以及其余重复媒体请求。raw 保留真实状态，不修改历史字段。

## 文件

| 文件 | 含义 |
|---|---|
| [http-baseline-9788.json](raw/http-baseline-9788.json) | 修改前逐请求、逐网络 read、逐 token 时间 |
| [http-candidate-9790.json](raw/http-candidate-9790.json) | 修改后相同请求流 |
| [server-candidate-9790.json](raw/server-candidate-9790.json) | 服务器配置、缓存计数与 engine 指标 |
| [summary.json](summary.json) | 从上述原始结果与 Trace 导出的摘要 |
| [http-candidate-9792.json](raw/http-candidate-9792.json) | TP2 + Encoder DP 的 HTTP 功能复核；不是 TP1 对照中的另一组性能数据 |
| [trace.json](trace.json) | 可导入 Perfetto/Chrome Trace Viewer 的局部时间线 |
| [preprocessing-9791.nsys-rep](raw/preprocessing-9791.nsys-rep) | 未裁剪的原始 Nsight 记录 |

Trace 使用单独运行，开销不计入性能表。展示窗口围绕首个注入冷图片的
`preprocess.image_processor` 区间：CPU 处理 115.02 ms，其间 GPU 计算活动覆盖约
100.50 ms、owner 发起 10 次 Decode replay；预处理与 replay 属于不同 CPU 线程。
这证明二者有实际重叠，不表示所有 CPU/GPU 时间都重叠。JSON 仅裁剪方便阅读，完整范围
仍可在原始 Nsight 文件中查看。

## 运行位置与复现

模型和环境直接复用 `/home/lcpu/87120912/models/Qwen3-VL-8B-Instruct` 与
`/home/lcpu/87120912/venvs/prism`。两个源码副本在
`/home/lcpu/87120912/preprocessing/baseline`、`candidate`；所有新增缓存、日志、Trace
均在该用户目录内。GPU 经 Slurm 申请，每个脚本最长 15 分钟。

```bash
sbatch run_http.sh baseline
sbatch run_http.sh candidate
sbatch run_trace.sh
sbatch --gpus=2 run_http.sh candidate engine-tp2.json
```

源码选择由服务器进程的 `PYTHONPATH` 决定，客户端不加载模型。服务日志及 JSON 中的
`source_package` 记录实际导入路径，两个副本没有共享进程内缓存。

可从 Nsight 文件重新导出中间 SQLite，再执行 [analyze.py](analyze.py)：

```bash
nsys export --type sqlite --output raw/preprocessing-9791.sqlite raw/preprocessing-9791.nsys-rep
python analyze.py
```

SQLite 只是中间文件，不重复提交。CPU 功能检查覆盖跨入口命中、问题重绑与实际 HF
Processor/M-RoPE 对齐、像素/调色板改变不误命中、缓存锁外处理、取消、容量和关闭顺序。
这里没有新增通用 CI、性能通过阈值或自动恢复框架。
