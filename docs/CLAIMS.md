# Prism-Infer Claim Ledger

> P6 冻结基线: `p6.12-content-aware-kv` (`c970c61`)
> 当前 P7.4-B 验证点: `72f85ba`
> 当前 P7.5/P8 验证点: projection mode `8293851`；online/trace/final gate `021d4e2`
> 当前 P9-A baseline 点: `29c0dbe`
> 当前 P9-C Prism quality 点: `5ada892`；vLLM external quality 点: `3ec90a5`
> 更新日期: 2026-07-19

本表区分“已实现”“已验证”和“性能占优”。README、简历和面试中的数字必须能
追溯到本表及对应 raw evidence。

## 可以使用的结论

| 结论 | 范围 | 证据 |
|---|---|---|
| Prism 自实现 Qwen3-VL text/vision/M-RoPE/DeepStack/engine 主路径 | Qwen3-VL-8B-Instruct | `VERIFICATION.md` P1-P3 |
| visual KV 是真实 physical compaction，不只是逻辑 mask | BF16/FP8 paged KV；prefill 后 compact、page 回收、decode append | P6.4 tests 与 `PERFORMANCE_REPORT.md` |
| logical M-RoPE position 与 physical KV position 分离 | compact decode | layout/append/mixed/swap focused regression |
| content-aware last-layer scorer 通过当前 reference task gate | 7 张固定 COCO 图片、35 captions、output32、keep=0.5、BF16 | token-F1 `0.321635 -> 0.318347`，drop `0.003288`；ROUGE-L `0.289116 -> 0.285406`，drop `0.003710` |
| 质量合格策略减少物理 KV | 7-image aggregate | physical token ratio `0.535x`，active prompt bytes ratio `0.538x` |
| 压缩 CUDA Graph 路径有效 | RTX 5090，offline decode，batch1-8 | eager/Graph token exact；decode speedup约 `1.76x-1.94x`，见 P6.11 |
| 当前质量合格压缩的短 workload 性能收益很小 | COCO batch4/output32 | decode-step `1.021x`，engine output throughput `1.013x`，E2E `1.005x` |
| P6.12 后全量回归通过 | 单卡环境 | `238 passed, 6 skipped in 232.90s` |
| P7.1 外部比较协议可自动拒绝不公平 cell | schema-v2 offline closed-loop | 两条 profile 共 20 rows 全部通过 model/GPU/KV/execution/clean-state gates |
| P7.1 初始 Graph baseline 仍慢于 vLLM Graph | RTX 5090、固定五类 workload、output32、commit `b17f933` | quality-qualified compact Graph TPOT 为 vLLM `1.65x-1.78x`；这是 P7.4 优化前基线 |
| content-aware compaction 对当前短/中 visual context只有小幅 TPOT收益 | 同一 P7.1 matrix | compact 相对 Prism off Graph约改善 `1.5%-3.0%` |
| model-precision logits 消除逐 decode 的整权重 FP32 转换 | clean `a33e7ed`，五类 workload，off/compact Graph | TPOT 相对显式 FP32 路径提升 `1.216x-1.280x`；peak allocated 减少 `2,230-2,317 MiB` |
| P7.4 后 Prism/vLLM Graph TPOT 差距明显缩小但尚未反超 | clean `a33e7ed`，同 GPU/KV budget/output32，10/10 comparability PASS | quality-qualified compact Prism 为 vLLM `1.34x-1.40x`；Prism peak allocated 约 `17.39-17.50 GiB`，低于 vLLM `17.74-17.93 GiB` |
| model-precision logits 通过 HF 与项目质量门禁 | single/multi-image/video teacher-forced + 7-image COCO lexical gate | HF logits/PPL max diff `0`；token-F1 drop `0.004360`、ROUGE-L 改善 `0.004090`，task gate PASS |
| P7.4 后全量回归通过 | clean `cc070b3`，单卡环境 | JUnit `241 passed, 6 skipped in 264.664s`，0 failure/error |
| engine-level online arrival 与 continuous batching 已实现 | clean `e7796e9`，单进程 RTX 5090 harness | constant arrival、动态 active batch、admission/cancel、prefill/decode interleave、request FSM 与 queue/TTFT/TPOT/goodput schema，见 `PERFORMANCE_REPORT.md` 6.10 |
| chunked paged prefill 已有 correctness 路径 | text 与 image+text 长输入 | 301-token text 为 `128/128/45`，646-token image+text 为 `512/134`；chunked/unchunked 输出 exact |
| P7.3 online matrix 的已完成请求全部满足各 cell 声明的 SLO | clean `e7796e9`，9 cells | 9/9 cell goodput fraction `1.0`；text-short 20 req/s peak active `5`，mixed 10 req/s peak active `4-5` |
| P7.3 后全量回归通过 | clean `e7796e9`，单卡环境 | JUnit `262 passed, 6 skipped in 245.36s`，0 failure/error |
| P7.4-B 已完成 Graph replay分类与 fixed-bucket correctness | clean `0fdd4a6` trace + clean `00b1012` matrix | replay `2,000` kernels/step、kernel busy median `12.921 ms`；linear/GEMV占 `70.55%`；batch1-8全部命中 `[1,2,4,8]` 预期 bucket且输出 exact |
| Prism editable package可在隔离venv构建并导入 | clean `568f7bb/d547385`，复用宿主CUDA/PyTorch stack | wheel build、`from prism_infer import LLM` PASS；6-file CPU/focused smoke `40 passed in 5.11s` |
| packed gate/up减少Graph内projection并小幅改善decode TPOT | Qwen3-VL-8B、RTX 5090 TP1、clean `8293851/021d4e2`、8个offline cells | Systems linear `253 -> 217`、总kernels `2,000 -> 1,964`；所有cell token exact，packed TPOT改善`0.483%–0.762%` |
| packed gate/up通过完整数值与online回归 | single/multi-image/video HF；text/image/video/mixed/7-image E2E；2个online A/B | HF model-precision logits/PPL diff `0`；offline/online token exact；online双方goodput fraction `1.0` |
| P7.5后当前主线完整回归通过 | clean `021d4e2`，单卡环境 | JUnit `287 tests / 0 failures / 0 errors / 6 skipped`，即`281 passed, 6 skipped in 297.622s` |
| fresh editable环境跑通完整8B最小demo | clean `021d4e2`，同一宿主CUDA/PyTorch stack | 新venv安装声明依赖与wheel；`example.py`输出8个token和decoded text，正常释放GPU |
| 细 page 在限定 paged-decode matrix 中降低 kernel latency | clean `29c0dbe`、RTX 5090、BF16、Qwen GQA、batch `1/8`、context `4096/8192` | page16/32 的最优 kernel median 相对 page256 低 `13.6%–20.1%`；20/20 correctness PASS，见 `PERFORMANCE_REPORT.md` 7.1 |
| P9-A 架构/协议/正式基线门禁通过 | RFC、versioned manifest、clean Page Matrix、NCU raw evidence | focused regression `64 passed in 6.99s`；compile/link/artifact/diff/GPU release gate PASS |
| scaled FP8 KV 是独立于 unit-scale FP8 的完整生命周期 | per-token/per-KV-head K/V FP32 scales | scale 与 payload 一同覆盖 Triton store、paged decode、COW、swap、physical compaction 和 CUDA Graph replay；component/GPU contracts PASS |
| Prism scaled FP8 通过冻结的标准多模态质量门禁 | clean `5ada892`；DocVQA/MuirBench/MVBench development/final | 6/6 formal non-inferiority PASS；allocated KV pool 为 BF16 的 `0.515625x`，节省 `48.4375%` |
| 同容量 vLLM FP8 外部质量矩阵结果为 MIXED | clean `3ec90a5`；vLLM 0.24.0 per-token-head FP8；semantic input exact | DocVQA/MuirBench 4 cell PASS，MVBench development/final FAIL；Prism scaled FP8 同六 cell PASS |

## 必须带限制的结论

| 现象 | 必须同时说明 |
|---|---|
| uniform/unit-scale-FP8 组合曾观察到 `4.016x` peak running capacity | uniform quality FAIL；unit-scale FP8 quality 未通过；不是 online throughput |
| 7-image aggregate active prompt bytes降至 `0.538x`；COCO batch4性能cell为`0.571x` | 都不是整个模型/GPU peak memory按相同比例下降 |
| CUDA Graph 提升约 1.8 倍 | 是 Prism internal eager→Graph，不是对 vLLM speedup |
| P6.12 reference token-F1/ROUGE-L drop 小于 0.004 | 不是标准 COCO CIDEr/SPICE，也不是通用 VQA accuracy |
| external eager baseline 比 Prism eager 快约 2 倍 | 仅为 P6 diagnostic matched eager；P7 重新比较双方 Graph |
| model-precision 相对旧 FP32 输出并非所有真实 case token exact | model precision 与 HF BF16 logits/PPL 逐值 exact；跨 batch shape 的低 margin argmax 允许分叉，同一 shape 必须 deterministic |
| P7.3 的 9-cell goodput fraction 为 `1.0` | 每个 cell 是一次多请求正式运行，SLO 按 workload 预先声明；不是跨进程统计置信区间，也不是网络 server 结果 |
| online off/compact 数字可并列报告 | 当前只能称为 observation；未做 process-level repeats，不能据此声称 compact online speedup |
| text-only prefix reuse 已验证 | 只复用并发请求仍持有的 full block；尚无独立 persistent prefix store，VL token-id prefix hash因不包含像素语义而禁用 |
| Graph replay CPU range只有 `1.899 ms` | 这是异步提交窗口；CPU返回后 GPU tail为 `13.089 ms`，不能把 CPU range当作完整 Graph时长 |
| fixed-bucket matrix列出 batch1-8 TPOT | 每个 cell是一次独立 process-level run；只证明 bucket/padding coverage与输出隔离，不证明 padding加速/减速，也不是 online goodput |
| packed gate/up TPOT改善`0.483%–0.762%` | 只覆盖记录的8个offline cells、RTX 5090 TP1与Qwen3-VL-8B；不是稳定E2E latency或online goodput speedup |
| packed gate/up的online A/B均满足SLO | 每个cell只有一次process-level run；用于regression/SLO，不计算可信speedup区间 |
| P8 fresh-environment完整8B demo已通过 | venv复用了同一宿主CUDA/PyTorch/driver stack；不证明另一台机器的CUDA ABI或性能可复刻 |
| page16/32 相对 page256 的 kernel median 低 `13.6%–20.1%` | 仅为 P9-A paged-decode microbenchmark；context 都能被 page 整除，未覆盖碎片，不是 full-engine TPOT/吞吐，也不是相对 vLLM/SGLang 的优势 |
| NCU page16/page256 的 occupancy 约 `12.5%`、waves/SM `0.17–0.19` | 只解释 batch8/context4096 的单个 kernel launch；不能外推为 full-engine GPU utilization，不能仅凭低 counter 定性为纯 memory-bound/compute-bound |
| scaled FP8 allocated KV pool 节省 `48.4375%` | 只计算 payload 与 FP32 scales；不是整卡/整模型显存，跨框架 page-table/Python allocator 尚无统一字节合同 |
| Prism scaled FP8 的六项 formal gate 全 PASS，vLLM FP8 为四 PASS/两 FAIL | 结论是预注册稳定性门禁结果；vLLM MVBench accuracy 点估计实际更高，不能声称 Prism accuracy 显著领先 |

## 当前禁止的结论

- “Prism 全面超过 vLLM/SGLang”。
- “KV 压缩让整体 GPU 显存减半”。
- “标准 COCO accuracy 下降小于 1%”。
- “unit-scale `fp8_kv` 已通过质量门禁”或“所有 FP8 KV 都已无损”；只有独立的
  `scaled_fp8_kv` 在冻结 P9 协议下通过。
- “Prism 已在全物理显存口径上支配 vLLM”或“P9 Gate A 已完整闭环”；当前跨框架
  page-table/Python allocator 字节不可比，external quality matrix 结论仍为 MIXED。
- “offline batch tok/s 等价于 online serving throughput/goodput”。
- “P7.3 已证明 HTTP/gRPC 服务性能”或“已证明相对 vLLM 的 online goodput 优势”。
- “P7.3 正式矩阵证明了 swap/recompute 性能”；正式 9-cell matrix 未触发 preemption。
- “TP2 已验证”“多卡可扩展”或“当前 NCCL/SM120 软件栈阻断 TP2”；当前租约只分配
  GPU0，管理员开放 NCU/NSYS 后额外设备可见不等于可用。此前跨 GPU1 的失败与成功
  control 都是无效实验，TP2 仍为 NOT RUN / UNVERIFIED。
- “已实现 megakernel/PD 分离/投机解码”。
- “GPU span减去 busy就是 occupancy/可消除 idle”或“sampler的 CPU range可与 Graph
  replay直接相加”；node tracing有 instrumentation，sampler CPU时间暴露前序 stream同步。
- “packed gate/up显著提升端到端性能”或“提升online goodput”；实测只支持小幅
  unprofiled decode TPOT改善，E2E受vision prefill双峰影响，online无process repeats。
- “README已在另一台全新机器完成完整8B验收”；当前fresh venv仍复用同一宿主
  CUDA/PyTorch/driver stack。

## P7.1 历史基线与 P7.4 当前结论

- `diagnostic_matched`: Prism eager TPOT约为 vLLM eager 的 `1.91x-1.97x`。
- `best_stable`: Prism off Graph约为 vLLM Graph 的 `1.69x-1.83x`；quality-qualified compact Graph约为 `1.65x-1.78x`。
- 双方 E2E throughput 当前也是 vLLM 更高，但部分 Prism offline TTFT存在双峰，E2E不作为压缩收益归因。
- 这是 offline closed-loop，不形成 online SLO goodput claim。P7.3 已建立 Prism 内部
  engine-level arrival/queue/SLO goodput 基线，但尚无相同 arrival/SLO 配置的 vLLM
  online record，因此仍不能形成外部 online ratio。
- P7.4 使用 node-level Systems trace 定位到旧 `compute_logits` 每 decode 都执行
  `lm_head.weight.float()`；改用模型原生 BF16 后，该 region 从 `4.068 ms` 降至
  `0.762 ms`，clean 五 workload TPOT提升 `1.216x-1.280x`。
- 更新后的 best-stable 中 compact Prism TPOT为 vLLM的 `1.34x-1.40x`，仍不允许
  声称反超；E2E throughput 仍受 prefill/TTFT影响且 vLLM更高。
- P7.4 默认数值路径与 HF teacher-forced logits/PPL逐值一致；显式 `fp32` 仅保留
  历史复现。mixed video 在 batch1/batch4 的低 margin 首 token 可不同，但同一
  mixed shape重复生成 exact，这一边界记录在 P7-006。
