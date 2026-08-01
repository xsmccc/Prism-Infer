# Optimization History

This document keeps the reasoning that matters after removing phase-by-phase
reports from the public repository.

## 1. Build a correct multimodal engine

The project began with a lightweight nano-vllm-style runtime and added the
Qwen3-VL vision stack, DeepStack injection, 3D position IDs, M-RoPE, multi-image
and video inputs, paged KV, chunked prefill, and continuous batching.

A major correctness failure was found later: flattened token/head tensors had
been passed directly to SDPA in an environment without standalone
FlashAttention, causing incorrect dimension interpretation. Replacing that
path with a compatible varlen backend and a shape-correct per-sequence SDPA
fallback invalidated older output hashes. This is why current claims require
semantic references and exact prompt audits, not repeat hashes alone.

## 2. Profile before optimizing

Node-level profiling exposed an expensive logits path that converted the full
LM-head weight to FP32 on every decode step. Keeping the model-precision
projection and using a guarded exact selection path reduced that region from
4.068 ms to 0.762 ms in the recorded profile and improved TPOT by
1.216×–1.280× across five workloads.

Packed gate/up projection then reduced Graph replay linear kernels from 253 to
217 and total kernels from 2,000 to 1,964. The measured TPOT gain was only
0.483%–0.762%, so it is described as a kernel-count optimization rather than a
large end-to-end speedup.

## 3. Establish compiler/Graph ownership

Early `torch.compile` experiments either graph-broke on mutable KV state or
duplicated work already captured by CUDA Graph. The retained design compiles a
stateless region, keeps page state in runtime-owned replay tensors, and
captures fixed batch buckets.

This produced the final H1/H2 BF16 TPOT of 9.8821/9.8680 ms, lower than the
tested vLLM and SGLang cells.

## 4. Treat KV compression as a representation problem

Direct E4M3FN casting with unit scale reduced bytes but failed quality. The
replacement stores per-token/per-KV-head FP32 K/V scales and carries them
through the full KV lifecycle. It reduced allocated KV bytes by 48.44% and
nearly doubled token capacity at the same budget while passing the frozen
quality matrix.

The scaled path is slightly slower than Prism BF16. The engineering result is
capacity without losing the bounded external TPOT position, not free speed
from quantization.

## 5. Convert visual pruning into physical reclamation

Logical attention pruning did not return memory to the allocator. The next
design compacted retained KV rows, rewrote page tables, preserved logical
M-RoPE positions, and released unused pages.

Uniform and overly aggressive policies failed on longer visual tasks. The
retained modality-aware floors balance image/mixed and video contexts. A
capacity-constrained batch-2 cell confirmed that released pages can change a
later scheduling outcome, rather than merely reducing an accounting number.

## 6. Optimize loaded serving by Goodput

Several candidates improved one local metric and harmed the workload:

| Candidate | Observed outcome | Decision |
|---|---|---|
| Dynamic Vision Tensor Graph | reduced some overhead but emitted incorrect tokens in mixed-shape load | disabled |
| Vision-aware bypass scheduling | better TTFT/E2E medians but Goodput -15.79% | not default |
| Phase-decomposed multimodal prefill | reduced the largest prefill segment but Goodput -34.18% | removed |
| Immediate heavy prefill | improved TTFT but fragmented decode cadence | rejected |
| 100 ms cache-hit coalescing | raw throughput +0.015% but lost one SLO and reduced Goodput | rejected |
| GQA4 and split-K paged attention kernels | 1.85×–1.90× slower at headline contexts | removed |

The retained scheduler uses SLO slack and estimated cost. Limiting CPU
intra-op threads also prevented media preprocessing from starving CUDA launch
submission in the same process.

## 7. Evolve media reuse into compressed prefix reuse

An exact Vision/DeepStack cache first demonstrated strong repeated-media
Goodput, but its key depended on Python object identity. That was not a safe
cross-request semantic cache.

The final design replaced identity with content SHA256 and separated:

- exact prompt+media processor entries;
- reusable media-layout tensors;
- Vision/DeepStack outputs; and
- compacted scaled-FP8 visual prefix pages.

The cold path compacts the prefix before executing the question suffix, so
cold and hit requests attend the same physical context. Full pages are shared;
partial tails use CoW and a reusable tail pool. This removed the fairness flaw
and sustained high-repeat performance with newly decoded media objects.

## 8. Turn TP scaffolding into real Qwen3-VL TP2

The repository already had process groups and generic parallel layers, but the
Qwen3-VL language model still instantiated full local projections. The first
real two-GPU run exposed this immediately: attention produced eight KV heads
while each rank's cache owned four. The fix was architectural rather than a
configuration change:

- shard Q/K/V heads, packed MLP gate/up, vocabulary embedding, LM head, and
  per-rank KV storage;
- use row-parallel output/down projections with NCCL reductions;
- scope packed checkpoint mappings to text attention/MLP names so vision
  weights remain untouched; and
- keep TP1 construction bit-for-bit on its existing path.

Two further correctness failures appeared only after Graph capture. The
parallel LM head selected prefill tokens twice, and worker ranks did not resolve
the same greedy sampling mode as rank 0. Removing the duplicate selection and
making every rank replay the same distributed greedy graph produced exact TP1
and TP2 tokens across repeats.

The initial correct version gathered the full sharded vocabulary every decode
step. Profiling showed that greedy decode needs only each rank's local winner.
Replacing the vocabulary gather with one value/token-ID all-gather improved the
TP2 batch-1 decode attribution cell by 4.47% while preserving exact selection.

With CUDA Graph capturing NCCL, TP2 reduced decode-step latency by 28.64% on
the single-image output-32 cell and 16.68% on the mixed text/image/video batch.
Per-rank Torch peak allocation fell by about 46.5%. Eager TP2 was slower, TP2
TTFT remained much worse, and aggregate allocation did not fall: this host has
no direct GPU P2P path and the vision encoder remains replicated. The retained
claim is therefore a real sharded, decode-oriented TP2 implementation, not
universal dual-card acceleration.

## 9. Final assessment

The project is complete as a focused inference-systems portfolio:

- it has a self-owned multimodal model/runtime path;
- it connects profiler evidence to compiler, kernel, memory, and scheduling
  decisions;
- it includes both retained and rejected candidates;
- it now demonstrates single-node TP2 language/KV sharding, distributed
  greedy selection, Graph-captured collectives, and multimodal serving;
- it has scoped wins against mature systems; and
- it states where vLLM remains stronger.

Further work should be a new research phase centered on cold multimodal
prefill, vision distribution, or faster interconnect-aware TP. Adding more
generic CI gates, smoke tests, or feature breadth would not strengthen the
current technical story.
