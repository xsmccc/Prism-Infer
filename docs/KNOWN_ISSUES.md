# Prism-Infer Known Issues

> 更新日期：2026-07-17  
> 本表记录当前主线限制，并保留本轮关闭项作为审计轨迹。历史 root cause 见
> `docs/issues/` 与 [ISSUE_LOG](ISSUE_LOG.md)。任何条目只能由可复现证据关闭。

## 总览

| ID | 状态 | 影响 | 摘要 |
|---|:---:|---|---|
| KI-001 | CLOSED | 历史 GPU 环境事件 | 设备恢复至稳定 `1–4 MiB / 0–2%` baseline，完整动态门禁已完成 |
| KI-002 | CLOSED | packed MLP claim | HF/E2E/online/TPOT/Systems/full regression闭环；只声明小幅 decode TPOT收益 |
| KI-003 | BLOCKED | TP2 | 8 卡可见，但 Prism Torch 2.6/CUDA 12.8/NCCL 2.25.1 的 SM120 collective 失败 |
| KI-004 | CLOSED | GPU counter claim | NCU 2025.1 权限恢复，Paged Attention 的 occupancy/DRAM/compute counter 已实测 |
| KI-005 | FAIL/REJECTED | FP8默认压缩 | FP8 KV最终质量门禁未通过 |
| KI-006 | NOT IMPLEMENTED | serving claim | 无 HTTP/gRPC server和 external online goodput |
| KI-007 | LIMITATION | prefix cache | VL prefix hash禁用；无独立 persistent prefix store |
| KI-008 | LIMITATION | video输入 | 支持 frame sequence，不含通用文件解码/采样策略 |
| KI-009 | REJECTED | torch.compile backend | 当前候选有 OOM或数值失败，不是支持后端 |
| KI-010 | OPEN | E2E归因 | vision prefill/TTFT存在双峰 |
| KI-011 | PROCESS | raw evidence | `data/` gitignored，需要单独保存正式实验产物 |

## KI-001：隐藏外部 GPU workload（CLOSED）

### 现象

物理 GPU UUID：

```text
GPU-989db6f6-3273-d1dd-b2b9-56cced4f30a4
```

该设备在多个重建容器中曾稳定显示约 `17,102–17,282 MiB` 已用、`14.7 GiB` 可用，
并持续有约 `22–35%` utilization、`109–188 W` power；容器内
`nvidia-smi --query-compute-apps` 不返回 owner process。重启/重新登录后也曾复现。

2026-07-17 有一次环境检查读到 `30.901/31.396 GiB` free，但随后五次每 2 秒采样又
全部回到：

```text
memory.used=17102 MiB
memory.free=15049 MiB
utilization.gpu=22–33%
```

因此单次空闲快照不足以认定设备稳定独占。

### 影响

完整 Qwen3-VL-8B 当前 formal配置的 torch allocator peak约 `17.4–17.5 GiB`，剩余
`14.7 GiB` 无法构建 engine。外部利用率也会污染 microbenchmark和 TPOT。

### 关闭证据

- 2026-07-17恢复后连续采样为 `1 MiB used / 32149 MiB free / 0% utilization`，
  formal运行之间为 `1–4 MiB / 0–2%`；
- clean `396702d` microbenchmark通过 `<=1024 MiB / <=5%`启动门禁；
- clean `8293851/021d4e2`完成完整8B HF、offline、online、Nsight、fresh demo和
  full regression，运行后均回到 `1 MiB` baseline；
- 当前主线不存在不可见显存占用。若未来再次出现，仍按下述恢复门禁重新打开本条目。

### 恢复门禁

```bash
python scripts/check_environment.py \
  --model "$PRISM_MODEL_PATH" \
  --require-cuda \
  --min-free-gib 18

nvidia-smi --query-gpu=uuid,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader
```

至少多次采样稳定通过、无外部 utilization，再启动 full 8B。若再次复现，需要宿主机/
hypervisor管理员按 GPU UUID定位进程；容器内没有权限清理不可见 owner。

## KI-002：P7.5 packed gate/up claim（CLOSED）

关闭证据：

- 共享 packed storage与旧 state-dict key兼容；
- `Module.to/_apply` 后 view rebind；
- rows `1/2/4/8/210/408/988` 的 BF16完整 MLP output bitwise exact；
- formal micro的七个 rows均 bitwise exact；
- single/multi-image/video 32-token HF model-precision logits max/mean diff与PPL diff均为`0`；
- text、single/multi-image、video、mixed及7-image COCO共8个 clean offline cell均
  token exact，packed decode TPOT改善`0.483%–0.762%`；
- single-image与mixed-rate10 online A/B逐请求token exact，双方SLO goodput fraction均`1.0`；
- node-level Systems实测 linear `253 -> 217`、总 kernels `2,000 -> 1,964`，
  kernel busy `12.815 -> 12.721 ms`；
- clean `021d4e2` full regression为`281 passed, 6 skipped`，0 failure/error。

结论是保留 packed默认，但 claim仅限同一 RTX 5090/Qwen3-VL-8B/记录 workload的
unprofiled decode TPOT小幅改善。vision prefill仍双峰，online没有process-level repeats，
因此不声称稳定E2E latency或online goodput加速。

## KI-003：Prism TP2 被当前 NCCL/SM120 软件栈阻断

静态 shard/collective审计、dimension/device preflight、variable-size Pipe控制面和显式
TP2 integration入口已实现。2026-07-17 当前机器已能看到 8 张空闲 RTX 5090；因此
旧结论“只有一张卡可见”已失效。

硬件拓扑：

- GPU0–3 属 NUMA0，GPU4–7 属 NUMA1；
- 卡间没有 NVLink；同 NUMA 卡之间也经过 PCIe/host bridge；
- 每张卡约 32 GiB，当前采样均为 `1 MiB used / 0% utilization`。

真实 Prism TP2 smoke 在首次 NCCL barrier 失败：

```text
torch: 2.6.0a0+ecf3bae40a.nv25.01
CUDA: 12.8
NCCL: 2.25.1
error: cudaErrorInvalidValue
```

独立最小 `torch.distributed` all-reduce 在同一 Prism stack、GPU0–1 上同样失败；提前
`torch.cuda.set_device()`、显式 `device_id`、禁用 P2P/SHM 都不能解除。NCCL 日志显示
目标 kernel 请求 `82,240 B` shared memory，而当前设备函数上限为 `79,856 B`。

隔离 vLLM 环境对同一 GPU0–1 的控制实验：

```text
torch: 2.11.0+cu130
CUDA: 13.0
NCCL: 2.28.9
all-reduce result: 3.0
PASS
```

因此 blocker 是当前 Prism Torch/CUDA/NCCL Blackwell 组合，不是 GPU 不可用，也没有
证据把它归因于 Prism scheduler/IPC。以下仍未完成：

- TP1/TP2 logits与greedy；
- per-rank weight/KV memory；
- latency/throughput与无 NVLink 通信成本；
- Vision Encoder replication成本。

不得写“Prism TP2已验证”或“可线性扩展”。保留入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 PRISM_RUN_TP2=1 \
python -m pytest -q tests/test_llm_vl_tp2.py -s
```

恢复优先使用隔离 Prism compatibility environment，并在升级 Torch/CUDA/NCCL 前与
用户单独确认；不能直接破坏 P8 已验证环境。即使 smoke 通过，仍需补正式 TP benchmark。

## KI-004：RTX 5090 hardware counters（CLOSED）

P6 阶段 Nsight Compute 曾返回 `Already under profiling`。管理员恢复权限并重启后，
NCU 2025.1 已能在 RTX 5090 采集真实 counters；旧 blocker 关闭。

BF16 Paged Attention、Qwen GQA、batch8/context4096 的代表性结果：

| Page | Duration | DRAM throughput | Compute throughput | Achieved occupancy | Waves/SM | Registers/thread |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 449.95 us | 17.48% | 14.16% | 12.49% | 0.19 | 64 |
| 256 | 543.26 us | 14.44% | 11.70% | 12.48% | 0.17 | 56 |

两者 correctness 均 PASS，max diff `4.882812e-4`、mean diff约 `3.0e-5`。NCU
明确提示 launch grid 太小；当前 grid 为 `(batch, query_head)=256`。这些数字支持
“该 case 并行度不足”，但仍不允许：

- 用单 kernel counters 声称 full-engine GPU utilization；
- 把低 DRAM/compute 百分比简化成纯 memory-bound 或纯 compute-bound；
- 仅凭 kernel launch 数声称 megakernel 必然获益。

P9 候选是 GQA query-head grouping 与 context split/稳定 softmax merge 的组合；只做
GQA 合并会把 grid 降到 64，不能进入计时。正式 raw report、page matrix 与命令记录在
`VERIFICATION.md` P9-A；以后新的 kernel claim 仍需独立 NCU/NSYS 闭环。
上述 clean raw 数字取代权限恢复后的早期 diagnostic `445.60/550.46 us`，后者不再
作为正式 counter evidence。

## KI-005：unit-scale FP8 KV quality FAIL

已完成 FP8 physical storage、KV store、paged load/dequant和 kernel correctness；固定 pool
payload bytes可为 BF16的 `0.5x`。但 unit-scale FP8在真实长输出上没有通过最终质量门禁，且
uniform+FP8的 `4.016x` observed capacity伴随 uniform quality FAIL。

当前规则：

- 默认质量合格主线只使用 BF16 content-aware physical compaction；
- 不声称 unit-scale FP8已质量合格；
- 不把 capacity observation表述为 throughput或通用并发提升。

当前 store 只是 BF16→FP8 direct cast，没有 K/V quant/dequant scale。P9 将另建
per-token-per-KV-head scaled FP8 模式，并把 payload、scale、compaction、swap 和
paged decode 放入同一生命周期；新模式必须重新通过长输出和标准任务质量，不能继承
旧短输出 token-exact claim。

## KI-006：无生产网络 server与 external online 对比

P7.3实现的是进程内 arrival/continuous-batching harness，记录 queue、TTFT、TPOT、
goodput和 request FSM。当前没有：

- HTTP/gRPC/OpenAI-compatible endpoint；
- 网络序列化、backpressure、auth或多进程 frontend；
- 相同 arrival/SLO配置的 vLLM online record；
- process-level repeats和统计置信区间。

因此不能声称网络 serving性能或相对外部框架的 online goodput优势。

## KI-007：Prefix cache边界

- text-only可复用并发请求仍持有的完整 block；
- 没有独立、持久化的 prefix store；
- VL token-ID prefix hash显式禁用，因为相同 image placeholder IDs不代表相同像素；
- mixed-VL quality benchmark默认关闭 prefix caching。

若未来支持 VL prefix，需要把像素/processor/grid identity纳入 hash和生命周期，而不是
移除当前 guard。

## KI-008：Video输入边界

当前 correctness覆盖 synthetic frame list、M-RoPE、Vision Encoder和mixed batch。
仓库不提供通用视频文件容器解码、FPS/时间戳采样、音频处理或生产上传 pipeline。
不同框架对 Qwen3-VL timestamp placeholder的处理曾造成 prompt token差异，相关 external
cell会自动标为不可比。

## KI-009：torch.compile候选已拒绝

- full decode cold compile在 32 GB上 OOM；
- Vision/full-layer编译有数值失败；
- attention-only虽然局部快于 eager，但 batch2/8长输出 token不符合当前合同，且仍慢于
  CUDA Graph。

`allow_unsafe_decode_compile=True` 只用于复现被拒绝的 benchmark，不能作为支持配置。

## KI-010：Vision prefill/TTFT 双峰

offline TPOT稳定，但部分 single-image vision prefill/TTFT出现约 `50–140 ms` 双峰。
当前环境不能锁 GPU clocks，尚无充分 root cause。E2E中位数差异不能归因给压缩；
headline使用更稳定的 TPOT。详见
[P7-005-TTFT_VISION_BIMODALITY](issues/P7-005-TTFT_VISION_BIMODALITY.md)。

## KI-011：Raw evidence 默认不入 Git

`data/` 包含模型运行日志、JSONL、JUnit和 profiler数据库，默认 gitignored，以避免巨大
二进制和环境数据污染源码提交。风险是外部 clone只有报告，没有原始文件。

正式发布应：

1. 保留生成命令和 schema version；
2. 对 raw artifacts计算 SHA256；
3. 上传 release artifact或长期对象存储；
4. 在报告中给出 commit、路径和下载位置；
5. 不用截图替代机器可读记录。

当前 GitHub release artifact尚未建立，因此仓库内数字主要通过报告、测试、summary
生成器和本地 raw路径审计。

## 关闭规则

关闭 Known Issue必须同时更新：

- 本文件状态和复现证据；
- [ROADMAP](ROADMAP.md) checkbox；
- [VERIFICATION](VERIFICATION.md) 命令与结果；
- [PERFORMANCE_REPORT](PERFORMANCE_REPORT.md)（涉及性能时）；
- [CLAIMS](CLAIMS.md) 允许/禁止边界；
- README或投递材料中的相关措辞。
