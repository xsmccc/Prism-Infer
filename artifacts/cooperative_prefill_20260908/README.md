# Prefill 交错执行记录

实现、因果解释和取舍见 [PREFILL_INTERLEAVING.md](../../docs/PREFILL_INTERLEAVING.md)。
主结果为 Prism 自身开关对照，不是 vLLM/SGLang 排名。

## 原始文件

- `raw/http-cooperative-{9798,9799,9800,9801}.json`：最终四轮 HTTP 请求时间戳、SSE 与输出。
  9798/9801 开启每 8 层交错，9799/9800 关闭，执行顺序为开→关→关→开。
- `raw/server-cooperative-*.json`：对应配置、GPU/PyTorch、缓存和调度记录。
- `raw/http-candidate-9794.json`：旧代码 `ae5b921` 直接开启 cooperative，默认 quantum=1，
  带旧 FCFS 凑批等待；9795 为同代码关闭对照。
- `raw/http-cooperative-9796.json`：新 FCFS 路径、quantum=4；9797 为 quantum=8 的首轮。
  探索记录不混入最终四轮汇总。
- [summary.json](summary.json)：两次配对的原始指标、中位数和完整输出差异位置。
- [trace.json](trace.json)、[trace_summary.json](trace_summary.json)：截取交错部分的时间线。
- [prefill-9802.nsys-rep](raw/prefill-9802.nsys-rep)：原始 Nsight 记录；对应 HTTP/服务器记录
  为 `http-prefill-trace-9802.json` 与 `server-prefill-trace-9802.json`。
- [requests-9803.json](raw/requests-9803.json)：真实模型的 3 请求 Prefill 与取消检查。

所有普通 HTTP 运行各生成 624 tokens。最终四轮不同请求中的最后一个长输出 token 存在
少量差异，包括关闭交错的两轮之间；不要将其说成完全一致或质量无损。短请求和分歧前的
长输出一致。`max_itl_ms`、`p95_itl_ms` 是单请求指标的中位数，不是全体 token 的 p99。

服务器的 Prefill batch duration 包含暂停期间插入的 Decode，不能作为净 GPU 时间。
原始文件的 `cooperative_underfilled_batches` 为当时字段名，后续改为更准确的
`cooperative_batches`；原始记录不重写。

## 复现

基于 `ae5b921` 加本目录同提交的 runtime 修改。源码在
`/home/lcpu/87120912/preprocessing/cooperative`，旧代码在 `candidate`，同目录下的模型、
环境、编译 cache 直接复用，不在登录节点跑 GPU。每个作业最长 15 分钟。

沿用[上一轮 HTTP 启动脚本](../shared_preprocessing_20260908/run_http.sh)：

```bash
sbatch run_http.sh cooperative engine.json
sbatch run_http.sh cooperative engine-cooperative.json
sbatch run_prefill_trace.sh
sbatch run_requests.sh
```

其中 `engine.json` 是[关闭交错的配置](../shared_preprocessing_20260908/engine.json)，
另一个为本目录的 [engine-cooperative.json](engine-cooperative.json)。
[run_trace.sh](run_trace.sh) 在服务器保存为 `run_prefill_trace.sh`，
[run_requests.sh](run_requests.sh) 调用 [check_requests.py](check_requests.py)。
请求检查通过实例方法临时关闭分段入口，得到同模型的原子执行参考，不更改权重或 kernel。

复制原始结果到本目录 `raw/` 后，可重建汇总与展示 JSON：

```bash
nsys export --type sqlite --output raw/prefill-9802.sqlite raw/prefill-9802.nsys-rep
python analyze.py
```

SQLite 是可重新导出的中间文件，不提交。Trace 独立运行，不用于延迟汇总。模型版本、
请求流和参数见技术说明与 server JSON；历史性能表不因这次实验而改写。
