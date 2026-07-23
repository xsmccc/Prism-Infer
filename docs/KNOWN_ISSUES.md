# Prism-Infer Known Issues

> 更新日期：2026-07-23
> 本表记录当前主线限制，并保留本轮关闭项作为审计轨迹。历史 root cause 见
> `docs/issues/` 与 [ISSUE_LOG](ISSUE_LOG.md)。任何条目只能由可复现证据关闭。

## 总览

| ID | 状态 | 影响 | 摘要 |
|---|:---:|---|---|
| KI-001 | CLOSED | 历史 GPU 环境事件 | 设备恢复至稳定 `1–4 MiB / 0–2%` baseline，完整动态门禁已完成 |
| KI-002 | CLOSED | packed MLP claim | HF/E2E/online/TPOT/Systems/full regression闭环；只声明小幅 decode TPOT收益 |
| KI-003 | UNVERIFIED | TP2 | 当前租约仅分配 GPU0；额外可见设备不代表可用，TP2 尚无有效动态证据 |
| KI-004 | CLOSED | GPU counter claim | NCU 2025.1 权限恢复，Paged Attention 的 occupancy/DRAM/compute counter 已实测 |
| KI-005 | PARTIAL | FP8 模式边界 | unit-scale 已拒绝；scaled FP8 的 Prism 内部 full-physical Pareto 已完成；跨框架 allocator 字节仍不可比 |
| KI-006 | NOT IMPLEMENTED | serving claim | 无 HTTP/gRPC server和 external online goodput |
| KI-007 | LIMITATION | prefix cache | VL prefix hash禁用；无独立 persistent prefix store |
| KI-008 | LIMITATION | video输入 | 支持 frame sequence，不含通用文件解码/采样策略 |
| KI-009 | CLOSED | torch.compile backend | unsafe full-decode 候选仍拒绝；边界受控的 compile+Graph 已通过 H1/H2 正确性与性能门禁 |
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
- P9-B 修复 runner/backend ownership cycle 后，同一进程连续 single/multi-image/video
  HF -> Prism 8B 为 `21 passed`，Graph exit 后仅保留小于 `64 MiB` 的 CUDA runtime
  residue，进程退出后 GPU0 回到 `1 MiB / 0%`；
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

## KI-003：TP2 当前没有已分配的双卡资源（UNVERIFIED）

静态 shard/collective 审计、dimension/device preflight、variable-size Pipe 控制面和显式
TP2 integration 入口已实现；动态 TP2 尚未完成有效验证。

2026-07-17 管理员为 NCU/NSYS 开放权限后，`nvidia-smi` 能观察到宿主机上的 8 张
RTX 5090，但当前租约只分配了 GPU0。设备可见、空闲快照和拓扑信息都不等于计算资源
已分配，也不授权在额外设备上启动 CUDA 或 collective。

P9-A 期间曾误把可见性当作可用性，并在 GPU0–1 上尝试 Prism TP2、最小
`torch.distributed` all-reduce 和隔离 vLLM-stack all-reduce。由于 GPU1 不在当前租约中，
这些尝试全部撤销为 **invalid experiment**：失败不能归因于 Torch/CUDA/NCCL/SM120，
成功结果也不能证明双卡可用。相关日志只保留为审计轨迹，不作为 root-cause 或能力证据。

当前可以声明的只有单卡 TP1 eager/CUDA Graph 动态结果，以及 TP 控制面的静态与
CPU-focused contract。以下仍未完成：

- TP1/TP2 logits 与 greedy 等价性；
- per-rank weight/KV memory；
- latency/throughput与无 NVLink 通信成本；
- Vision Encoder replication成本。

不得写“Prism TP2 已验证”“多卡可用”或“当前软件栈阻断 TP2”。只有在明确租用或获准
使用至少两张卡后，才能运行保留入口：

```bash
CUDA_VISIBLE_DEVICES=<allocated-gpu-a>,<allocated-gpu-b> PRISM_RUN_TP2=1 \
python -m pytest -q tests/test_llm_vl_tp2.py -s
```

恢复顺序是先确认调度器/租约确实分配双卡，再做逐卡 CUDA allocation、最小 collective、
TP1/TP2 correctness 和正式 benchmark。只有合法双卡上的最小 collective 仍失败，才能
继续定位软件栈；未经单独批准不升级 Torch/CUDA/NCCL，也不能破坏 P8 已验证环境。

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

## KI-005：FP8 模式必须分开陈述（PARTIAL）

已完成 FP8 physical storage、KV store、paged load/dequant和 kernel correctness；固定 pool
payload bytes可为 BF16的 `0.5x`。但 unit-scale FP8在真实长输出上没有通过最终质量门禁，且
uniform+FP8的 `4.016x` observed capacity伴随 uniform quality FAIL。

该失败只属于旧 `fp8_kv`，不能覆盖新的 `scaled_fp8_kv`。P9-C 已实现
per-token/per-KV-head K/V FP32 scale，并让 scale 与 payload 一同经历 Triton store、
paged decode、copy-on-write、swap、physical compaction 和 CUDA Graph replay。clean
`5ada892` 的 DocVQA、MuirBench、MVBench development/final 六项 formal
non-inferiority 均 PASS；allocated KV pool 为 BF16 的 `0.515625x`，节省
`48.4375%`。

P10 在 clean `59bb4ae` 进一步完成 Prism 内部 full-physical profile：相同 `28,928`
token capacity 下，current-process NVML 从 `23,938 MiB` 降至 `21,966 MiB`
（少 `1,972 MiB`）；相同约 `4 GiB` KV 预算下，scaled-FP8 capacity 增至
`56,320` tokens（`+94.69%`），NVML 仅多 `14 MiB`。这些是固定配置的 device
memory/capacity 结果，不等同于已测在线吞吐或并发 goodput。

同容量 vLLM 0.24.0 per-token-head FP8 external quality matrix 在 clean `3ec90a5`
完成：DocVQA/MuirBench PASS，MVBench development/final FAIL。vLLM 的 MVBench
accuracy 点估计更高，但 paired CI 下界未过预注册稳定性 margin，因此不能事后改判。
该矩阵的 `full_physical_comparable=false`：双方尚无统一的 page-table/Python allocator
字节合同，所以不能声称完整物理显存 Pareto 已胜出。

当前规则：

- 默认仍是 compression-off/BF16；BF16 content-aware compaction 与 scaled FP8 是两个
  分别验证的 opt-in 候选；
- 不声称 unit-scale FP8已质量合格；
- 可以在冻结 P9 协议范围内声明 `scaled_fp8_kv` formal quality PASS；
- 不把该结论外推到 `visual_compact_scaled_fp8` 组合、任意模型或任意上下文；
- 不把 capacity observation表述为 throughput或通用并发提升。

scaled-FP8 的正式 H1/H2 runtime matrix 已完成，TPOT 相对 SGLang 低
`1.06%–1.12%`、相对 vLLM 低 `2.55%–2.77%`，但 E2E 并非所有格都领先。
剩余未闭环项是跨框架统一 allocator metadata 字节合同、在线并发 goodput，以及
content compaction + scaled FP8 组合质量；这些项目不能从上述结果推导。

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
P9 标准质量路径另行冻结 `opencv-python-headless==4.10.0.84 / FFMPEG` 解码与 16 帧
采样，不能外推为通用视频服务。vLLM 0.24 会替换完整
`vision_start + video_pad + vision_end` triplet，而 pinned HF processor只替换中间的
`video_pad`，原始行为会丢失最外层 marker；external runner使用显式 versioned prompt
adapter补回外层 marker，并要求最终 request token IDs与 Prism逐项相同。若 vLLM版本、
processor marker或 placeholder数量改变，adapter会 fail closed，不能静默继续比较。

## KI-009：torch.compile 边界已闭环（CLOSED）

- 历史 full decode cold compile 在 32 GB 上 OOM，Vision/full-layer 编译有数值失败；
- 历史 attention fullgraph 曾捕获 mutable KV store。每层 cache 是 monolithic allocation
  的非零 `storage_offset` view；AOT functionalization 对 V view 生成的大 clone 会从 view
  data pointer 越界读取，已定位为 token 异常和 `illegal memory access` 的 root cause；
- 正式支持的边界把 mutable KV store 与 paged decode 留在 validated runtime 内，只编译
  纯 QKV/QK-Norm/M-RoPE、packed projection 与适合的 decode 子图，再由 CUDA Graph
  捕获稳定 device views；compression-off 和 scaled-FP8 均使用同一边界合同；
- clean `4779342` 的 H1/H2、warmup `2` / repeat `5`、output `128` matrix 中，
  correctness、prompt-token hash、greedy repeat 稳定性和 GPU UUID 门禁均通过。
  BF16 TPOT 相对 SGLang 低 `4.54%–4.83%`、相对 vLLM 低 `6.13%–6.27%`。

`allow_unsafe_decode_compile=True` 仍只用于复现被拒绝的历史 benchmark，不能作为支持
配置；“compile+Graph 已支持”只指上述明确边界，不能解读成 whole-model/full-decode
任意捕获。该限定既保留失败证据，也关闭 P9-D 当前主线门禁。

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
