# 相关工作与项目边界

Prism-Infer 研究的是一个窄而具体的系统问题：同一组图片被连续提出不同问题时，如何在
固定 KV 显存预算下保留更多可复用的视觉前缀。项目没有提出新的视觉 token 选择算法；
核心工作是把压实后的视觉 Prefix KV 接入在线页式缓存，并处理内容身份、共享页所有权、
回收和跨问题复用。

## 通用推理框架中的前缀复用

[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
按完整 KV block 缓存精确前缀。Block identity 包含父 block、当前 token，以及 LoRA、
多模态输入等附加哈希；当前实现也提供 SHA256 身份。vLLM 的多模态处理还会缓存
Processor 与 Encoder 输出。因而，当媒体及其前面的 prompt token 完全一致时，vLLM
本身能够同时复用多模态输入和语言 Prefix KV。

[SGLang](https://arxiv.org/abs/2312.07104) 的 RadixAttention 将可复用 KV 前缀组织在
radix tree 中，并在多轮对话、few-shot 和结构化程序等负载中复用。Prism-Infer 与这两类
机制的差别不是“能否缓存前缀”，而是缓存对象：Prism 保存经过 Scaled-FP8 量化和物理
视觉 token 压实的页，使同一 KV 字节预算可以容纳更多媒体前缀；代价是视觉 token 删除
带来的质量损失。

[vLLM-Omni Prefix Caching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/)
进一步复用流水线 stage 的 hidden states 与多模态输出，并复用 vLLM 的 block/slot
mapping。该设计说明“KV 前缀命中”和“Encoder/中间张量命中”应作为不同层次处理；
Prism 的 Prefix-first 请求路径也采用这一划分，并在 Prefix 命中时跳过视觉缓存恢复。

## 多模态 token 与 KV 压缩

- [LOOK-M](https://aclanthology.org/2024.findings-emnlp.235/) 针对长多模态上下文，根据
  文本与图像的注意力关系压缩 KV，并用 KV merging 补偿被删除的视觉信息。
- [VL-Cache](https://arxiv.org/abs/2410.23317) 区分 Prefill/Decode 及视觉/文本 token 的
  稀疏模式，使用分层预算和 modality-aware token score。
- [FastV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10478_ECCV_2024_paper.php)
  利用早期层注意力，在后续层删除视觉 token，主要减少模型 Forward 的计算量。
- [VisionZip](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html)
  在进入语言模型前选择和合并视觉 token，重点降低视觉前缀长度与 Prefill 成本。

这些工作提供了更强的视觉 token 选择方法，但目标通常是单次请求或同一问题下的精度—
压缩率权衡。Prism 的重复提问路径需要一份压实结果对后续未知问题都有效，因此使用
query-agnostic Uniform 作为清晰对照。实验中，每题 Attention Top-k 无法直接复用，
沿用第一题的 Top-k 也没有优于 Uniform；这不表示 Uniform 在算法上优于上述方法，只
说明当前固定预算下没有得到可复用且质量更好的 Attention 选择结果。

## Prism-Infer 的工程范围

项目实现并验证了以下组合：

1. 媒体内容、Processor 布局和精确公共 token 共同构成缓存身份；
2. Prefix 查询位于 Scheduler admission 前，命中后跳过 Vision/DeepStack；
3. Scaled-FP8 K/V 与 per-token、per-KV-head scale 一起经过 Store、Paged Attention、
   Copy-on-Write、Swap、物理压实和 CUDA Graph Replay；
4. 压实后的完整页只读共享，未满尾页 clone，活跃请求缺页时从同一 KV Pool 回收缓存；
5. 相同的媒体、prompt token、请求顺序和 KV 字节预算用于 Prism、vLLM 和 SGLang。

因此，合适的项目表述是“压缩感知的多模态 Prefix Cache 运行时”，而不是新的 token
剪枝算法，也不是通用场景下全面优于 vLLM 或 SGLang。性能比较必须与实际发生 token
删除的质量结果一起阅读。
