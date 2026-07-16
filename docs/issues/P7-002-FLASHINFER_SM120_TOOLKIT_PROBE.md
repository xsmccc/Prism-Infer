# P7-002: FlashInfer SM120 CUDA toolkit capability probe

- 状态: `DOCUMENTED_LIMITATION`
- 环境: vLLM 0.24.0, FlashInfer 0.6.12, Torch `2.11.0+cu130`
- 系统 CUDA link: `/usr/local/cuda-12.8`
- GPU: RTX 5090, compute capability 12.0

## 现象

启动 vLLM Graph preflight 时 stderr 出现：

```text
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
```

但模型加载、PIECEWISE mixed prefill/decode capture、FULL decode capture 和正式
generate 都继续成功。

## 如何定位

先在安装环境中搜索完整错误字符串：

```bash
rg -n "SM 12.x requires CUDA|Failed to get device capability" \
  /data/vllm-omni/.venv/lib/python3.12/site-packages
```

字符串来自 `flashinfer/compilation_context.py`。继续检查发现：

- Torch wheel 自带 CUDA 13.0 runtime，并能识别 capability `(12, 0)`。
- FlashInfer 的 JIT helper 通过系统 CUDA toolkit 判断编译能力。
- `/usr/local/cuda` 指向 CUDA 12.8，且当前容器没有 `nvcc`。

因此不是 GPU capability 获取失败，而是 FlashInfer 无法为 SM120 建立可用的
JIT compilation context。

## 当前处理

P7.1 vLLM baseline 使用：

- attention backend `FLASH_ATTN`。
- `VLLM_USE_FLASHINFER_SAMPLER=0`，使用 PyTorch native sampler。
- 每条 record 保留 Torch/CUDA/driver/backend/cudagraph mode。
- stderr 原样保存，不通过设置伪造的 arch 环境变量隐藏 warning。

## 为什么当前 baseline 仍可用

本次测量路径没有依赖需要 JIT 编译的 FlashInfer sampler；vLLM 日志明确完成
PIECEWISE/FULL Graph capture，重复输出稳定。warning 因此是环境限制证据，不是
本次执行路径失败。

但这不证明 FlashInfer 所有 SM120 kernel 都可用，报告不能把当前配置称为
FlashInfer 最优路径。

## 真正解决方法

如果后续要评估 FlashInfer sampler 或 JIT kernels，应提供 CUDA toolkit 12.9+
和匹配的 `nvcc`，清理 JIT cache 后重新运行 correctness 与 performance。仅设置
`FLASHINFER_CUDA_ARCH_LIST=12.0f` 会绕开 capability normalization，却不会补齐
实际编译工具，因此不作为修复。

## 面试表达

> 我沿完整错误字符串定位到依赖的 capability normalization，区分了 Torch
> wheel runtime 和系统 CUDA toolkit。因为受影响的 JIT 路径不在本次 benchmark
> 中，我没有随意改依赖或隐藏 warning，而是切换源码支持的 sampler、记录边界，
> 并保留未来启用 FlashInfer JIT 时的真正修复条件。
