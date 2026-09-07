# Prefix KV 与 Scaled-FP8 Prefill 修复

本轮基于 main `b7ba461` 修复缓存生命周期和 Prefill 执行路径，不新增调度策略、
token selector 或三引擎性能矩阵。以下说明哪些行为改变了，以及如何确认。

## 页的分配、发布和回收

原来 `_allocate_dense()` 为整个 prompt 分配页时立即注册 hash。两条相同的 13-token
请求、page/chunk size 4 下，第一条本轮只算 4 tokens，第二条却把 12 tokens 当作缓存。
现在分配只保留空间，Scheduler 在执行完成后调用 `publish_computed_blocks()`，仅发布
本轮写完的完整页。Decode 也在最后一个位置的 KV 写完后发布，不在 `may_append()` 时
提前发布。取消未完成请求不会留下未计算页的可命中索引。

视觉边界位于页中间时，旧 tail clone 还携带了上一个问题的完整页 hash。新问题较短会
触发 `partial KV block unexpectedly has a prefix hash`，较长时还可能使索引内容失真。
现在只复制公共前缀行，新尾页不继承旧问题元数据；页写满后按新 token 内容发布。若部分
命中落在图片内部，所有需要重算的重叠页都变为私有页，而不只处理第一块。

缓存回收原来只计入 `ref_count == 1` 的页，漏掉了多个 entry 共享、却没有活跃请求引用
的页。现在统计待淘汰 entry 的引用数，并按唯一物理页判断是否可以回收；活跃请求的引用
仍保留。6 页池中两个缓存条目共享一页的复现，已能回收全部空闲缓存并接纳新的 6 页请求。

这三类问题的回归用例及相邻的 Decode/图片边界情形集中在
[`test_prefix_page_lifecycle.py`](../tests/test_prefix_page_lifecycle.py)。已有页共享、
尾页、Swap、压实和调度测试按“执行后发布”更新，没有增加 CI 或通用 smoke 流程。

真实 Qwen3-VL-8B 请求检查使用 Scaled-FP8、TP1、eager、chunk size 256、32 KV pages。
两条相同长文本在 APC 关闭、冷缓存、热缓存下的输出 token 均一致；同图先问长问题，再
连续两次问短问题，两次均命中 Prefix，输出与 APC 关闭的参考一致。该检查复用了服务器
已有模型，没有重新下载权重。

记录见 [`model_requests.json`](../artifacts/review_fixes_20260907/model_requests.json)。
复现入口为同目录 `check_requests.py`，通过 `PRISM_MODEL_PATH` 指定模型。它检查这几个
真实请求的执行一致性，不是大规模生成质量或性能测量。最终相关 CPU 用例 71 项通过，
原始输出保存在同目录 `cpu_tests.log`；GPU 输出位于 `gpu_tests.log`。

## FP8 压实地址

Qwen3-VL-8B 的 K/V 与 36 层展平为 72 rows。220 pages、page size 256、8 KV heads、
head dim 128 时，跨层地址偏移就能超过 int32 范围。2026-08-13 的 int64 修复曾留在旧
工作目录中，未进入 main；本轮将 gather/scatter 的 row/token 索引改为 int64。

RTX 5090 上使用真实的 220/352-page 分配，复制包含重叠 source/destination 的 slots，
逐字节结果与独立保存的源数据一致。已有 KV payload/scale 的 CoW、Swap 和压实检查也通过。

## Prefix 命中的 Attention

原快路径仍逐层执行长度 `.cpu()`、页号 `.item()`，并用 GPU 布尔索引筛页。现在输入准备
保留 CPU 侧 query/key offsets，按逻辑上下文长度截取有效页表。FP8 页按字节 gather 后
恢复 dtype，避免依赖后端是否支持 float8 `index_select`。

Attention 使用 PyTorch `causal_lower_right` 表达“完整缓存前缀 + 因果后缀”，由支持 GQA
的 SDPA 后端执行，不手工生成 Q×K mask 或复制 K/V heads。已移除捕获所有异常后静默
回退的分支；按已知 KV 格式选择路径，真正的执行错误不再被隐藏。

28 项 GPU/数值相关检查通过，包括 BF16/Scaled-FP8、跨页、多个请求、单 token 后缀、
Chunked Prefill、真实容量压实，以及 FP8 scale 的写入和 Graph replay。
本次 GPU 环境为 RTX 5090、driver 580.142、PyTorch 2.11.0+cu130、Triton 3.6.0。

另外对 Q=[32,32,128]、KV heads=8、context=2048、page size=256 的 Scaled-FP8 Attention
进行了三次调用的 profiler 观察：

| 算子 | 次数 |
|---|---:|
| `aten::item` / `aten::_local_scalar_dense` | 0 / 0 |
| `aten::nonzero` / `aten::repeat_interleave` | 0 / 0 |
| `aten::_scaled_dot_product_flash_attention` | 3 |
| `aten::_scaled_dot_product_attention_math` | 0 |

原始计数见 [`prefill_profile.json`](../artifacts/review_fixes_20260907/prefill_profile.json)，
可用同目录的 `profile_prefill.py` 复现。这确认所测调用没有上述主机读回和 GQA 复制，
不是端到端加速百分比。KV 仍需 gather/反量化到连续临时张量，整个引擎也仍是同步 step。

## 历史数据与文档

旧的 MuirBench 27/49 → 20/49、MVBench 183/252 → 113/252，以及 Compact 的跨引擎
延迟领先数字混入了压实地址错误，不再作为当前质量或性能结论。补入 2026-08-13 修复后
的原始请求记录，并重新按 Sample ID 配对核对：85 题 Dense/Uniform 为 46/85、47/85，
实际删除的 49 题为 27/49、28/49，3 个答案不同。它不证明普遍无损，也不是 FP8 相对
BF16 的量化质量测试。

README、Results、Architecture、重复视觉上下文说明及 JSON 摘要已按这些来源对齐。
旧数据原件保留，但不把它们当作本轮 APC 修复后的重新测量。详情见
[历史质量与容量实验](../artifacts/cache_pressure_20260813/README.md)。
