# FP8 LM-head 历史验证

这份2026-09-02记录原保存在本地项目资料库，本次补到公开仓库，没有重新测量。
原材料记录的实现版本为 `0b91c3c`；原始JSON记录RTX 5090、PyTorch2.11.0+cu130，
模型为Qwen3-VL-8B-Instruct。

测试在HF模型的全词表FP32参考轨迹上采集实际LM-head输入，再逐位置执行Prism的
FP8候选生成、Top-64与FP32重排。它检查LM-head策略，不是整个Prism引擎与HF的等价证明。

| 观察项 | 结果 |
|---|---:|
| 请求／Decode位置 | 51／818 |
| 直接FP8 argmax与全词表FP32一致 | 799／818 |
| 全词表FP32 winner进入Top-64 | 818／818 |
| 候选内FP32重排后的winner一致 | 818／818 |
| FP32 winner在候选中的最差名次 | 3 |

其中有24条真实MuirBench多图题，共48个被测Decode位置，其余覆盖文本、单图、多图和视频。
原生BF16 logits的argmax与全词表FP32在10个位置不同；不能把这10处解释成FP8漏召回，
也不能仅据token变化认定答案质量提高或下降。没有发生漏召回只限于这批样本；现有Top-64
路径没有全词表回退，无法找回候选集合以外的winner。

独立的 `[1,4096]→[1,151936]` CUDA Graph微基准中，BF16投影＋argmax为0.780432 ms，
FP8投影＋Top-64＋重排为0.453283 ms。额外常驻FP8权重及scale合计594.08 MiB，原BF16
权重仍保留。41.92%是LM-head子路径收益，不是系统TPOT收益。

原始文件保持不变：

- [逐请求／逐位置结果](fp8_lm_head_semantics_20260902.json)
- [GPU微基准与额外存储](fp8_lm_head_microbench_20260902.json)
- [原测试脚本](bench_fp8_lm_head_semantics.py)

脚本的 `--model`、`--output` 指定模型与结果；重建带MuirBench的51请求协议还需要
`--muir-plan` 和 `--muir-root`。不提供这两项的默认运行不是上述完整样本集合。
这些记录不能替代BF16 KV与Scaled-FP8 KV的生成质量对照；二者是不同优化。
