# 运行时修复验证

对应[修复说明](../../docs/RUNTIME_REPAIRS_20260908.md)，父版本为 `f880105`。
本目录保留请求和数值检查，不是性能基准。

- `requests-9849.json`：Qwen3-VL-8B的满池Chunked Prefill、多图默认路径、视觉缓存
  命中/重排、超长请求在Vision前拒绝，以及缓存与无缓存输出对照。
- `tp-capacity-9849.json`：真实NCCL collective，使用注入10/6页上限验证共同容量和拒绝。
- `flashinfer-9850.json`：FlashInfer0.6.13原生Graph输出对照，异长/跨页/桶padding。
- `gpu_requests.txt`、`flashinfer.txt`：对应Slurm作业日志。

实际环境为RTX5090、driver580.142、PyTorch2.11.0+cu130；模型和venv复用
`/home/lcpu/87120912`下既有文件。作业9849使用2卡、56秒；9850使用1卡、6秒，均已结束。
TP容量注入值用于测试分支，不能说成机器实际内存测量。

工作副本位于 `/home/lcpu/87120912/review-closure-20260908/source`。将本目录的
`check_*.py` 和 `run_*.sh` 复制到 `/home/lcpu/87120912/review-closure-20260908`，
在该目录下执行：

```bash
sbatch run_gpu_checks.sh
sbatch run_flashinfer_check.sh
```

所有GPU工作通过Slurm，每个作业最多15分钟。CPU回归检查源位于tests：Executor搬运
顺序、resident Decode/普通chunked推进、TP容量、视觉缓存准备、FlashInfer metadata
和配置选择。没有运行全项目模型测试或重新建立竞品基准。
