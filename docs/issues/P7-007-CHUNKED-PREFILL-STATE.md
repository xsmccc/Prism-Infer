# P7-007: Chunked prefill 只有调度外壳且 cache 状态重载

- 状态: `RESOLVED`
- 首次系统化复现: `0060171`
- 修复 commit: `e7796e9`
- 影响: online continuous batching、prefix hit、physical compaction、mixed VL

## 现象

P7.2 将 scheduler token budget传入 ModelRunner后，长 prompt第二个 chunk仍会在
`cu_seqlens_k > cu_seqlens_q` early gate失败。说明旧 `enable_chunked_prefill`
只把 Sequence临时截短，并没有实现读取历史 paged KV的 attention路径。

随后 compact online warmup又触发：

```text
RuntimeError: visual KV compaction does not support prefix-cache prefill
```

该请求没有 prefix hit；只是 ModelRunner把已计算 chunk数永久写进
`num_cached_tokens`，compaction因此把普通 chunk progress误判成 shared prefix。

## 根因

1. prefill attention只支持 Q=K 的 contiguous FlashAttention/SDPA，没有 Q<K 的
   paged history路径。
2. slot mapping按完整 block区间生成；chunk从 block中部开始时会重复覆盖旧 KV。
3. `num_computed_tokens` 与 `num_cached_tokens` 都被用于表示 chunk进度，真实 prefix
   hit与本请求已计算 token没有独立所有权。
4. 视频 processor可为同一个 payload生成两个视觉占位区；按连续 span切 chunk会把
   一次 VisionEncoder输出错误拆成两半。
5. 仅按视觉占位 token ids做 prefix hash不包含像素语义，不同图像可能错误命中。

## 修复

- 新增 paged prefill correctness path：当前 K/V先写 cache，再按 block table收集完整
  history；使用显式 bottom-right causal mask处理 Q<K。
- slot mapping逐 query token计算 `block_id * block_size + offset`。
- immutable `BatchPlan.scheduled_token_counts`成为 runner唯一 chunk budget。
- `num_computed_tokens`独占 chunk progress；runner结束后恢复真实
  `num_cached_tokens`，不再污染 compaction contract。
- 同一请求首个到最后一个视觉占位 token整体作为 atomic region，region过大时
  admission fail closed。
- multimodal request完全禁用 token-id prefix hash；text-only concurrent full-block
  prefix hit继续走 paged prefill。
- physical compaction只在该 Sequence所有 prefill chunks完成后提交。

## 验证

- attention unit reference：chunk `Q=2/K=6` max diff `<1e-5`，当前 K/V写入 exact。
- 301-token text：`128/128/45`，chunked/unchunked 2-token exact。
- 646-token image+text：`512/134`，chunked/unchunked 2-token exact。
- 两条 concurrent 301-token text：第二条复用 256-token full block，输出 exact。
- staggered text/image/video 6 requests：动态 batch peak active `5`，全部完成。
- clean full regression：`262 passed, 6 skipped in 245.36s`。

## 被拒绝的方法

- **继续保留 early gate并声称支持 chunked prefill**：只有 scheduler分块，不具备
  attention correctness。
- **每个 chunk重算完整 prompt**：隐藏 Q<K问题，但重复 vision/decoder计算且破坏
  slot/KV ownership语义。
- **按每段 video placeholder分别编码**：VisionEncoder payload/grid是整体输入，
  不能由 token span猜测 feature切分。
- **允许 VL token hash prefix reuse**：hash不含像素，可能跨不同图像复用错误 KV。

## 剩余限制

- 当前 paged prefill是 gather+SDPA correctness path；长上下文性能需后续专用 kernel。
- prefix hash只在并发请求仍持有 block时复用；尚未实现独立 persistent prefix store。
- online harness是 engine-level arrival loop，不是网络 server，也没有外部框架 online
  ratio。
