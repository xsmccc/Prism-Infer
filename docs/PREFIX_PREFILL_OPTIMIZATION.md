# Prefix 命中后的短 Prefill 优化

本轮解决的是重复图片请求命中 KV 后，仍需要较长时间才能返回首 token 的问题。它没有
改变 Prefill/Decode 调度，也没有减少输入或输出 token。RTX 5090、TP1 的同一 HTTP
请求流中，同图换问题 TTFT 从 208.37 ms 降至 144.61 ms，两组配对分别降低 27.72% 和
33.29%。冷请求 TTFT 略有改善，独立 Decode TPOT 基本不变。

## 为什么缓存命中后仍然慢

完整视觉 Prefix 命中只省掉已有前缀的计算，问题后缀还要经过全部语言层。既有 Trace
中的热后缀长度为 21 或 17 tokens，但每次启动 2,111 个 GPU kernel。GPU kernel 时间
合计约 14.6 ms，主机侧模型执行却超过 110 ms，因此首先检查短 Prefill 的提交路径。

最初只扩展 Q/K 融合并融合 KV gather，kernel 数降到 959、GPU 时间降到约 12.2 ms，
但 HTTP TTFT 没有稳定改善。`9813` 的两组配对一快一慢，不能宣布成功。继续用 cProfile
分析后，发现模型初始化留下的 PyTorch 默认设备模式，持续拦截 Python 层的 torch 调用。

## 1. 修正默认设备的作用域

旧的 [_model_initialization_defaults](../prism_infer/engine/model_runner.py) 保存默认设备，
调用 `torch.set_default_device("cuda")` 构造模型，最后再调用
`torch.set_default_device(previous_device)` 恢复。通常 previous_device 是 CPU。

问题在于，显式设为 CPU 与没有安装默认设备模式并不相同。当前 PyTorch 实现通过
`DeviceContext.__torch_function__` 实现默认设备分发；恢复设备值时，又安装了一个 CPU
DeviceContext。它不仅作用于 tensor factory，也参与其他 Python torch API 调用。
PyTorch 的[官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.set_default_device.html)
也指出这项额外开销，并建议用 `with torch.device(device)` 临时切换设备。

现在只在模型构造、权重加载、warmup 和既有 Graph capture 的作用域内使用
`with torch.device("cuda")`。退出时由上下文管理器恢复调用者的模式栈，dtype 仍通过
`finally` 恢复；没有强行清除调用者自己设置的模式，也没有使用 PyTorch 私有 API 修补
运行时状态。相关私有模式栈读取仅出现在针对性的测试中。

CPU profile 中，旧版本每个热 Prefill 约有 13,089 次来自
`torch/utils/_device.py` 的 `__torch_function__` 调用；最终版本中该项为 0。
Attention 的 CausalBias 自己仍有正常的分发调用，不应把它说成“所有 torch override
都被删除”。不再使用的 get/set-default-device 启动能力检查也已移除。

## 2. 让短后缀复用现有 Q/K RMSNorm 与 M-RoPE 融合

原来的 [Q/K 融合](../prism_infer/ops/qk_rmsnorm.py) 仅在 Prefill token 数至少 1,024 时
启用。17/21-token 后缀仍分别运行转型、平方、均值、归一化、weight multiply、旋转和
多个逐元素操作。现在同一融合路径也用于支持的短输入。

融合保留两个原生 PyTorch 均值归约，没有改变归约顺序；BF16 normalize、weight multiply
及 M-RoPE 的乘加仍在原来的位置舍入。GPU 检查覆盖 1、5、17、21、37、255、1,607 tokens，
Q/K 均与原实现逐元素相同。TP1 的 Q32/KV8、head_dim128 路由条件没有扩大到未适配结构。

首次尝试还暴露了长度特化问题：Q_ROWS/K_ROWS 是编译常量，第一次遇到 21-token 后缀
会额外编译，首个换问题请求达到约 585 ms。它们现在是运行时参数，并禁止按值特化，
不同长度复用 kernel；head 数、head dimension 等结构参数仍保持编译常量。

## 3. 一次完成 paged K/V gather 和反量化

原路径分别读取 K、V、K scale、V scale，再转型和相乘，产生多次 kernel 提交和中间
张量。新的 [paged_kv_gather.py](../prism_infer/ops/paged_kv_gather.py) 在一次 Triton
kernel 中读页表、读取 K/V 与各自 scale，直接写出供原 SDPA 使用的连续 K/V。

数值顺序没有改变：FP8 payload 和 FP32 scale 都先转为目标 dtype，再相乘并舍入到
目标 dtype。不能把 FP32 scale 直接乘到 payload 上、最后才转 BF16，那是另一种运算。
页号和地址乘法使用 int64，context_len 为运行时参数；只读取有效 context，不读取尾部
页表 padding。CUDA 之外继续使用原参考路径。

这是内部 kernel，Attention 负责 cache 绑定，Scheduler 负责有效页表，不在每层重复
一套设备、shape 和页内容检查。测试覆盖实际层视图、乱序页、部分尾页、不同 strides、
舍入差异样本和大地址偏移。融合不修改 cache，本次没有改动页引用、CoW 或回收策略。

## 收益如何确认

基线为 `cc77b08`，最终版本包含上面三项修改。job `9816` 在同一次 Slurm 分配内顺序
启动四个独立服务器：旧→新→新→旧。每次初始化后先执行一个预热请求，再运行原来的
HTTP 客户端；预热不计入下面的延迟汇总。

配置：Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，
RTX 5090、driver 580.142、PyTorch 2.11.0+cu130，TP1、Scaled-FP8 KV、64 个 256-token
页、max_model_len4096、max_num_seqs4。Prefix Cache 与共享 CPU 预处理开启，Vision
output cache、cooperative Prefill、token chunking 关闭；Decode 仍使用原 CUDA Graph。

输入为八张 448×448 纯色图片。同图换问题短请求输出 16 tokens；另外测独立 128-token
Decode，以及三次“长 Decode 已输出至少三个 token 后插入一组冷图片”。计时包含 HTTP、
服务端媒体准备、排队和模型计算。每格先对每轮的三个对应请求取中位数，再取两轮中位数。
独立 Decode 每轮只有一条，不能把它当成高并发吞吐结果。

| 指标 | 旧版本 | 最终版本 |
|---|---:|---:|
| 同图换问题 TTFT | 208.37 ms | 144.61 ms |
| 同图换问题 E2E | 375.53 ms | 310.13 ms |
| 独立 Decode TPOT | 10.90 ms | 10.87 ms |
| 有冷请求插入的长请求 TPOT | 12.78 ms | 12.65 ms |
| 注入冷请求 TTFT | 475.68 ms | 453.63 ms |

两次热 TTFT 配对分别为 201.12→145.38 ms、215.61→143.83 ms。这是本轮最明确的
端到端收益；不要把很小的 Decode 差值单独包装为新的 GEMV 优化。

为区分融合与设备模式修复，job `9819` 在已经修正设备模式的同一个模型进程内切换
新旧 Q/K 与 gather 路径。各预热一次后，每种执行四次，Prefill step 中位数为
57.82→40.31 ms，所有输出 token 相同。这个计时不含 HTTP 和 CPU 媒体准备，证明融合
本身仍有收益，但不能与 HTTP 的百分比相加。

最终四轮 HTTP 都完成 11 请求、624 tokens，SSE token 流与最终输出一致。以第一轮
旧版本为参照，新1、新2、旧2分别有1、2、2条长请求的最后一个（第128个）token不同；
之前的输出和全部短请求相同。基线重复运行也有此变化，不能据此证明模型质量下降，
也不能宣称完整输出始终 bit-exact。原始差异位置保留在 summary.json；这不是 MuirBench
准确率测试，本轮没有更新旧的质量结论。

## Trace 与当前剩余问题

原始 Nsight 记录通过 CUDA correlation ID 将 kernel 对应到发起它的 CPU 区间，避免
把“区间里碰巧在运行”的别的请求算进去。热 Prefill 的 kernel 数从 2,111 降到 959，
其中每次明确包含 36 个 fused gather 和 36 个 fused Q/K normalize-M-RoPE 调用。
GPU kernel 时间约从 14.6 ms 降到 12.2 ms。带 profiler 的主机时延不能代替前面的
普通 HTTP 测量，也不能把 CPU 区间与 GPU kernel 时间的差直接当成可兑现的加速。

代码、逐请求 JSON、CPU 函数统计、消融与 Trace 入口在
[实验目录](../artifacts/prefix_prefill_fusion_20260908/README.md)。两个新 kernel 的数值
检查与现有 paged Prefill 检查通过；初始化模式恢复和相关能力检查也通过。

本轮没有新增短 Prefill Graph 缓存或更大范围的 torch.compile：它们还有首次捕获和
形状复用成本，需要另外判断。冷请求的语言 GEMM、Vision 计算和 GPU 上整段 Prefill
打断 Decode 仍然存在；这次没有解决所有调度问题，也没有重跑 vLLM/SGLang 排名。

这项工作的复盘重点是：缓存命中不等于没有 Prefill；kernel 数减少也不必然带来端到端
加速。先通过 Trace 确认碎片化执行，再通过 CPU profile 找到初始化残留模式，用作用域
修复和保持数值语义的融合共同减少实际服务延迟。
