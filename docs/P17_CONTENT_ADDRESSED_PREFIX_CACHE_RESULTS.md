# P17 Content-Addressed Multimodal Prefix Cache

## 1. Result

P17 replaces P16's process-local Python-object identity cache with a bounded,
content-addressed reuse path for Qwen3-VL media:

- separately decoded but byte-identical PIL images and video frames hit;
- processor tensors, Vision Encoder output, every DeepStack output, and the
  compacted scaled-FP8 multimodal prefix KV can persist across requests;
- the prefix identity includes the loaded model/processor namespace, exact
  media content, processor layout, and the logical prompt prefix through the
  final visual placeholder;
- a different question can reuse media-invariant work without reusing its text
  suffix;
- cached pages remain reference counted, use copy-on-write for a partial tail,
  and retain logical M-RoPE positions independently from physical compacted KV
  positions; and
- cache residency is bounded and evicted by observed lifetime benefit per
  physical page.

The final 60-request matrix uses fresh decoded media objects on every request.
Repeated cells are byte-identical; unique cells receive a deterministic
content marker that preserves tensor and token-layout shapes.

| Media repeat rate | Prism raw tok/s | Prism class-SLO Goodput | Good | vLLM raw tok/s | vLLM class-SLO Goodput | Good |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 216.188 | 133.316 | 37/60 | 223.079 | 215.643 | 58/60 |
| 25% | 216.441 | 187.582 | 52/60 | 223.474 | 216.025 | 58/60 |
| 50% | 217.083 | 206.228 | 57/60 | 223.521 | 219.796 | 59/60 |
| 75% | 224.279 | 224.279 | 60/60 | 225.112 | 225.112 | 60/60 |
| 100% | 224.369 | 224.369 | 60/60 | 225.004 | 225.004 | 60/60 |
| 100%, different question | 224.301 | 224.301 | 60/60 | 225.004 | 225.004 | 60/60 |

The defensible conclusion is:

- P17 removes P16's object-identity fairness limitation without losing its
  high-locality result;
- at 75--100% reuse, Prism and vLLM are within `0.28--0.37%` raw throughput
  and both attain every class SLO, which is a statistical tie rather than a
  Prism win;
- Prism's 100% fresh-object result is `+0.82%` raw and `+4.30%` Goodput over
  the favorable SGLang cache-on n60 reference (`222.538/215.120 tok/s`);
- at 0--50% reuse, vLLM remains materially ahead, primarily because Prism's
  cold multimodal prefills disturb decode cadence; and
- P17 therefore demonstrates a distinctive compressed multimodal reuse
  design, not universal engine superiority.

The final 600-request, 100%-repeat H3 closure confirms that the n60 result is
not a short-run artifact:

- raw throughput and class-SLO Goodput: `241.428 tok/s`;
- SLO attainment: `600/600`;
- TTFT/TPOT p50: `146.418/13.041 ms`;
- processor, Vision/DeepStack, and compacted-prefix hits: `360/360` each;
- compacted-prefix tail copies/reuses: `2/358`, avoiding
  `5,546,686,464 bytes` of repeated page copying;
- zero rejection, cancellation, or terminal failure; and
- full frozen-H3 conformance with process GPU memory released on exit.

Against the prior P16 n600 result, raw throughput is essentially unchanged
(`+0.10%`) while class-SLO Goodput improves from `226.311` to
`241.428 tok/s` (`+6.68%`). This comparison isolates the value of persistent
compacted-prefix reuse and its tail-page pool on the same repeated-media
workload; it is not a unique-media claim.

## 2. Frozen Protocol and Fair Cache-On Baselines

- GPU: NVIDIA GeForce RTX 5090
- GPU UUID: `GPU-a0340044-fe48-ceca-08e0-a50d9bcdd79a`
- driver: 580.105.08
- PyTorch/CUDA: 2.11.0+cu130 / CUDA 13.0
- model: Qwen3-VL-8B-Instruct snapshot
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- workload: conditional-video H3 mix, Poisson 4 requests/s, seed `20260717`,
  warmup 10, greedy output 64
- n60 trace SHA256:
  `b7948e4deb75e174ca76b2fc3ca1cae4aeb8a4676e163f4cd0f8165a5f0e954b`
- n60 prompt-token SHA256:
  `2449ab2ca89420454a04d3ce460dcbfd0cc3d4eb20bb70cf0fd8891e196e3ec6`
- n600 trace SHA256:
  `105fa73b203c42dc61b96be60d367b7b567cc78ccd4183ca9193280fffdf4235`
- n600 prompt-token SHA256:
  `c65975c1b1a3f97dadf1bf4caa199bf7b9dcdf930f38fbfb1679f7e6b37ba79a`
- final n600 artifact:
  `data/p17_repeat_matrix/p17_repeat100_n600_s20260717_r2.json`
  (SHA256
  `fc974c728223063974d654be8b8db87eb97f050824ddcca85e6a9f07a5997e0f`)

Prism uses 220 scaled-FP8 KV pages of 256 tokens, a maximum of 27 retained
multimodal-prefix pages (`525,533,184 bytes` across ranks), a 256 MiB
Vision/DeepStack cache, and 128 processor-cache entries.

vLLM 0.25.1 was rerun with its official cache paths enabled:

- `enable_prefix_caching=True`;
- `mm_processor_cache_gb=1`;
- block size 256;
- 4 GiB KV cache;
- FlashAttention prefill and compiled/CUDA Graph decode; and
- `VLLM_USE_FLASHINFER_SAMPLER=0`, because the installed FlashInfer sampler
  binary rejects SM120. The failed default-sampler log is retained.

SGLang 0.5.15.post1 was rerun with Radix prefix caching enabled. Its official
multimodal global cache is restricted to the disaggregated encoder path in
this version and cannot be enabled for this single-node run. The reported
SGLang reference is therefore a favorable repeated-object cache-on baseline,
not a byte-identical feature match.

## 3. Safe Content Identity

P17 hashes supported media values with length-delimited SHA256:

- bytes and byte arrays;
- file contents rather than path strings;
- decoded PIL mode, size, and exact pixels;
- NumPy dtype, shape, and contiguous bytes;
- Torch dtype, shape, and exact tensor bytes; and
- recursively ordered lists/tuples for multi-image and video inputs.

Opaque objects fail closed and cannot enter the cache. An object-identity memo
may avoid hashing the same live object twice, but it retains strong references
and verifies `is` identity; it is only an acceleration and never defines
semantic equality.

The namespace fingerprints the loaded model path/config metadata, processor,
image/video processor, tokenizer, and pixel-limit configuration. The final
visual key additionally hashes the exact processed pixel tensor, grid, visual
token ID, and visual token count. The compacted prefix key then combines this
visual identity with the exact logical prompt tokens through the last visual
placeholder.

This separation matters:

```text
media content + processor layout
    -> processor/Vision/DeepStack identity
    -> visual prefix token boundary + exact prefix tokens
    -> compacted scaled-FP8 prefix-KV identity
question suffix
    -> never part of visual identity
    -> never cached as generated or language-suffix KV
```

## 4. Different-Question Reuse

The processor normally combines media preprocessing and prompt tokenization.
P16 consequently missed its processor cache when only the question changed.
P17 keeps two exact layers:

1. an exact prompt+media entry; and
2. a media-layout entry containing immutable pixel/grid tensors.

For media-layout hits, Prism rebuilds the chat-template token span for the new
question while retaining the expanded visual prefix and processed media. It
accepts this path only when the changed text begins after the final visual
placeholder and the unchanged template suffix matches exactly; otherwise it
falls back to the full processor.

The 100%-repeat/different-question run records:

- 36/36 processor hits;
- 36 prompt-rebind hits and zero rebind failures;
- 36/36 Vision/DeepStack hits;
- 36/36 compacted prefix-KV hits;
- identical aggregate prompt-token and per-class prompt-token hashes to the
  pre-optimization run;
- 60/60 SLO attainment and zero terminal failures; and
- 224.301 tok/s versus 222.099 tok/s before media-only prompt rebinding
  (`+0.99%`).

## 5. Compacted Prefix Pages, CoW, and Tail-Page Reuse

A cold multimodal request first executes the visual prefix through the final
placeholder, physically compacts it, then executes the question suffix against
that same compacted context. This two-stage order is required for cache
correctness: an early prototype let the cold suffix attend dense unpruned
visual KV while the hit path attended compacted KV, changing H1 output. Both
failed artifacts are retained.

An admitted entry owns references to its compacted physical pages and records:

- logical prefix length;
- physical compacted length;
- retained original positions for M-RoPE;
- KV dtype and compression evidence;
- exact prompt-prefix tokens;
- lifetime hit count; and
- cache-owned tail clones.

Full pages are shared read-only. If the final prefix page is partial, the first
concurrent hit copies only valid prefix rows into a private page. P17 retains
that derived page after the request completes; later requests can lease it
without copying again. Stale suffix rows are safe because every attended suffix
slot is overwritten before use and context lengths exclude unwritten rows.

On the final n60 100%-repeat run:

- prefix hits/misses: 36/0;
- tail copies/reuses: 2/34;
- cache-owned prefix/tail pages: 12, including 3 tail clones;
- resident compacted-prefix bytes: `233,570,304`;
- actual copied bytes: `33,454,080`;
- avoided copy bytes: `529,486,848`; and
- zero eviction, rejection, or terminal failure.

The isolated H1 sequence exercises all three states in order:

| Request | State | TTFT | Output |
| --- | --- | ---: | --- |
| 1 | cold compact + admit | 573.224 ms | exact |
| 2 | prefix hit + first tail copy | 116.852 ms | exact |
| 3 | prefix hit + tail-clone lease | 87.873 ms | exact |

All three 64-token outputs have SHA256
`3b81c4a3e5ec1c9b9d1a67d06a6ad56ffae3320ccdbb89e0dbfc25ad14082b0d`.

## 6. KV Storage, Physical Reclamation, and Correctness

P17 preserves the established scaled-FP8 KV representation:

- FP8 payload: `4,152,360,960 bytes`;
- FP32 scales: `129,761,280 bytes`;
- total: `4,282,122,240 bytes`; and
- allocated storage reduction versus BF16: `48.44%`.

The n60 100%-repeat run still records:

- 33,252 logical prompt tokens;
- 21,966 physical prompt-KV tokens;
- 11,286 dropped visual tokens;
- 48 released physical pages; and
- a 24,004 MiB process-memory peak.

The n600 closure scales these workload-dependent counters to 332,520 logical
prompt tokens, 219,660 physical prompt-KV tokens, 112,860 dropped visual
tokens, and 480 released pages. Its measured process peak is 24,006 MiB.

The isolated H1 cold/copy/reuse result is token exact. H2 has six of six
different-question occurrences equal to the previous correct output. Loaded
runs can form different BF16 GEMM batch shapes, so the established correctness
contract remains isolated H1/H2 equality, same-shape determinism, exact prompt
hashes, exact FP32 final reranking, zero terminal failures, and valid
logical/physical KV invariants.

## 7. What Failed and Why

| Candidate / failure | Evidence | Decision |
| --- | --- | --- |
| Cache hit before cold-path compaction | H1 cache-off/on outputs differed | Fixed with two-stage cold prefix compaction before question suffix |
| Full partial-page CoW | copied all 256 rows per hit | Replaced with valid-row copy, then retained tail-clone leases |
| Object identity as semantic key | fresh byte-identical objects missed | Replaced by fail-closed content SHA256; identity remains memo-only |
| Prompt-bound processor cache | different-question processor hits 0/36 | Split exact prompt cache from reusable media-layout tensors |
| Cold tier-1 cooperative preference | single-image/H1 TPOT improved, H2 regressed; Goodput 118.954 -> 118.949 | Rejected and reverted |
| Tier-0 immediate prefill | TTFT fell sharply, TPOT rose; raw 224.369 -> 224.208 | Rejected and reverted |
| 100 ms cache-hit coalescing | raw +0.015%, but one request lost TPOT SLO; Goodput 224.369 -> 220.662 | Rejected and reverted |
| vLLM default FlashInfer sampler | installed binary rejected SM120 | Preserved failure; official sampler disabled for all valid vLLM runs |
| Prism n600 `--formal` flag | argument parser exited before model load | Preserved operational failure; rerun without unsupported flag |

These attempts are not independent features. The retained design is the
smallest combination that improved the declared cache workload without
sacrificing Goodput.

## 8. Reproduction

Prism:

```bash
P17_RUN_REVISION=r3 \
  benchmarks/run_p17_repeat_matrix.sh "$MODEL_PATH" data/p17_repeat_matrix
```

vLLM:

```bash
benchmarks/run_p17_vllm_repeat_matrix.sh \
  "$MODEL_PATH" data/p17_repeat_matrix
```

Both scripts materialize the same seed, arrivals, class order, output length,
fresh-object policy, repeat ratios, and class-SLO source. Raw JSON and logs are
kept under `data/p17_repeat_matrix/`.

## 9. Resume and Interview Boundary

A bounded resume bullet:

> 为 Qwen3-VL 推理实现内容寻址的多模态前缀缓存：以模型/processor/媒体像素/布局/
> prompt-prefix SHA256 复用 Vision、DeepStack 与物理压缩的 scaled-FP8 KV 页，
> 设计 M-RoPE 逻辑/物理位置解耦、页引用/CoW/收益每页淘汰及可回池尾页；在每请求
> 重新解码媒体对象的 60-request 负载中实现 36/36 跨对象命中、避免 529.5 MiB
> 尾页复制，100% 重复与同媒体不同问题均 60/60 SLO，吞吐 224.37/224.30 tok/s，
> 超过 SGLang 缓存基线并与 vLLM 相差 0.3% 以内，同时保持 H1/H2 exact、
> scaled-FP8 KV -48.44% 与视觉物理页回收。

The interview boundary should be volunteered:

1. The distinctive result is compressed multimodal reuse with exact ownership
   and position semantics, not another generic Python LRU.
2. vLLM remains stronger on unique and low-reuse media because its cold
   multimodal pipeline has lower TTFT and less decode interference.
3. At high reuse Prism eliminates the expensive work and reaches statistical
   parity with vLLM while beating the available SGLang cache-on reference.
4. The rejected scheduler candidates demonstrate why optimizing TTFT alone can
   reduce SLO Goodput: shorter prefill waiting fragmented decode cadence.
