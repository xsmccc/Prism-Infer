# ═══════════════════════════════════════════════════════════════
# model_runner.py —— 模型执行器 (engine 层最核心的文件)
#
# 职责: 把 scheduler 的调度结果 (一批序列) 转换成 GPU tensor,
#       喂给模型前向推理, 采样出下一个 token
#
# 关键概念:
#   - Prefill 准备: 拼接多条序列的 token, 计算位置和 slot_mapping
#   - Decode 准备: 每条序列只取最后一个 token
#   - KV Cache 分配: 一次性预分配 GPU 显存, 分配给各 attention 层
#   - CUDA Graph: 预录制 Decode 的 GPU 操作, 消除 CPU launch 开销
#   - Tensor Parallel: 多 GPU 通过共享内存同步调用
#
# C++ 类比: 整个文件 ≈ inference engine 的 execute() 函数
# ═══════════════════════════════════════════════════════════════

import pickle                                      # 序列化, 用于多 GPU 间传递数据
import torch
import torch.distributed as dist                   # 分布式通信 (NCCL)
from multiprocessing.synchronize import Event       # 进程间事件通知
from multiprocessing.shared_memory import SharedMemory  # 进程间共享内存

from prism_infer.config import Config
from prism_infer.engine.sequence import Sequence
try:
    from prism_infer.models.qwen3 import Qwen3ForCausalLM     # Qwen3 纯文本模型 (legacy)
except ImportError:
    Qwen3ForCausalLM = None  # VL 项目中纯文本模型可能不存在, 用 VL 版替代
from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM
from prism_infer.layers.sampler import Sampler              # 采样器 (温度采样/贪婪)
from prism_infer.utils.context import set_context, get_context, reset_context  # 全局上下文
from prism_infer.utils.loader import load_model             # 权重加载


class ModelRunner:

    # ─────────────────────────────────────────────────────────
    # __init__: 初始化模型、KV Cache、CUDA Graph
    # ─────────────────────────────────────────────────────────
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config                # HuggingFace 模型配置 (层数/head数等)
        self.block_size = config.kvcache_block_size  # KV Cache 块大小 (如 16)
        self.enforce_eager = config.enforce_eager    # True = 禁用 CUDA Graph, 每次都 eager 执行
        self.world_size = config.tensor_parallel_size  # 几张 GPU (Tensor Parallel 并行度)
        self.rank = rank                             # 当前 GPU 编号 (0, 1, 2, ...)
        self.event = event                           # 进程间事件, 用于多 GPU 同步

        # ── 初始化分布式通信 ──
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        # "nccl": NVIDIA 的 GPU 集合通信库
        # "tcp://localhost:2333": rendezvous 地址(汇合点), 所有进程连到这里握手
        # C++ 类比: MPI_Init() + MPI_Comm_rank()

        torch.cuda.set_device(rank)                  # 绑定当前进程到第 rank 号 GPU

        # ── 创建模型并加载权重 ──
        default_dtype = torch.get_default_dtype()     # 保存原始默认类型 (float32)
        torch.set_default_dtype(hf_config.torch_dtype)  # 设为模型精度 (如 bfloat16)
        torch.set_default_device("cuda")              # 后续 torch.empty() 等默认在 GPU 上
        if self._is_vl_config(hf_config) or Qwen3ForCausalLM is None:
            self.model = Qwen3VLForCausalLM(hf_config)  # Qwen3-VL 模型结构
            self.is_vl_model = True
        else:
            self.model = Qwen3ForCausalLM(hf_config)    # legacy Qwen3 纯文本结构
            self.is_vl_model = False
        load_model(self.model, config.model)           # 从文件加载权重到模型
        self.sampler = Sampler()                       # 创建采样器

        # ── 初始化 KV Cache 和 CUDA Graph ──
        self.warmup_model()                            # 热身: 跑一次前向, 触发 CUDA kernel 编译
        self.allocate_kv_cache()                       # 根据剩余显存计算能分配多少 block, 分配 KV Cache
        if not self.enforce_eager:
            self.capture_cudagraph()                    # 录制 CUDA Graph (Decode 加速)
        torch.set_default_device("cpu")                # 还原默认设备为 CPU
        torch.set_default_dtype(default_dtype)         # 还原默认类型

        # ── 多 GPU 同步 (Tensor Parallel) ──
        if self.world_size > 1:
            if rank == 0:
                # rank 0 (主进程): 创建共享内存, 大小 1MB
                self.shm = SharedMemory(name="prism_infer", create=True, size=2**20)
                dist.barrier()                          # 等所有进程到这里
            else:
                # rank > 0 (从进程): 等主进程创建好共享内存, 然后连接
                dist.barrier()
                self.shm = SharedMemory(name="prism_infer")
                self.loop()                             # 从进程进入无限循环, 等待主进程指令
                # 注意: 只有 rank>0 会进入 loop(), rank=0 继续执行正常流程

    # ─────────────────────────────────────────────────────────
    # exit: 清理资源
    # ─────────────────────────────────────────────────────────
    def exit(self):
        if self.world_size > 1:
            self.shm.close()                            # 关闭共享内存映射
            dist.barrier()                              # 等所有进程都关了
            if self.rank == 0:
                self.shm.unlink()                       # 主进程删除共享内存段
        if not self.enforce_eager:
            del self.graphs, self.graph_pool            # 释放 CUDA Graph 对象
        torch.cuda.synchronize()                        # 等所有 GPU 操作完成
        dist.destroy_process_group()                    # 销毁分布式通信组
        # C++ 类比: MPI_Finalize()

    # ─────────────────────────────────────────────────────────
    # 多 GPU 通信: loop / read_shm / write_shm / call
    #
    # 工作原理:
    #   rank 0 调用 call("run", seqs, True)
    #   → call 内部 write_shm: 把方法名+参数 pickle 后写入共享内存
    #   → rank>0 的 loop 里 read_shm: 从共享内存读出方法名+参数
    #   → rank>0 调用 self.run(seqs, True) → 同步执行相同操作
    # ─────────────────────────────────────────────────────────

    def loop(self):
        """从进程的无限循环: 等待主进程发指令, 执行相同的方法"""
        while True:
            method_name, args = self.read_shm()         # 从共享内存读取 (阻塞等待)
            self.call(method_name, *args)               # 执行对应方法
            if method_name == "exit":
                break                                    # 收到 exit 指令则退出循环

    def read_shm(self):
        """从进程: 等待事件 → 读共享内存 → 反序列化方法名和参数"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                               # 阻塞, 直到主进程 set event
        n = int.from_bytes(self.shm.buf[0:4], "little") # 前 4 字节 = 数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        # pickle.loads: bytes → Python 对象
        # *args: 解包剩余元素为列表
        self.event.clear()                              # 清除事件, 准备下一次等待
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程: 序列化方法名和参数 → 写共享内存 → 通知从进程"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])       # 序列化为 bytes
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")    # 前 4 字节存长度
        self.shm.buf[4:n+4] = data                      # 后面存数据
        for event in self.event:
            event.set()                                 # 通知所有从进程 "有活干了"
        # self.event 在 rank 0 是一个 list (每个从进程一个 Event)

    def call(self, method_name, *args):
        """调用自身的方法, 同时通知从进程也调用相同方法"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)          # 先通知从进程
        method = getattr(self, method_name, None)       # 反射: 通过字符串找到方法
        # getattr(self, "run") 等价于 self.run
        # C++ 类比: 类似 std::unordered_map<string, function_ptr>
        return method(*args)                            # 调用方法

    # ═══════════════════════════════════════════════════════════
    # 初始化阶段: warmup + allocate_kv_cache
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _is_vl_config(hf_config) -> bool:
        model_type = getattr(hf_config, "model_type", "")
        return (
            "vl" in str(model_type).lower()
            or hasattr(hf_config, "vision_config")
            or hasattr(hf_config, "image_token_id")
        )

    def _forward_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Call either the new Qwen3-VL interface or the legacy text model."""
        if self.is_vl_model:
            return self.model(input_ids=input_ids, position_ids=positions)
        return self.model(input_ids, positions)

    def warmup_model(self):
        """热身: 用最大尺寸输入跑一次模型, 触发 CUDA kernel 编译/分配"""
        torch.cuda.empty_cache()                        # 清空 GPU 缓存
        torch.cuda.reset_peak_memory_stats()            # 重置峰值内存统计
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # max_num_batched_tokens: 一批最多处理多少 token
        # max_model_len: 单条序列最大长度
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 计算最大并发序列数 (受 token 数和序列数上限约束)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 造假序列: 每条都是 max_model_len 个 0
        self.run(seqs, True)                            # 跑一次 Prefill (前向传播)
        # 目的: 让 PyTorch/CUDA 编译 kernel, 知道峰值内存
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 GPU 剩余显存, 计算能分配多少 KV Cache block, 一次性分配"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()         # GPU 显存: 空闲 / 总量
        used = total - free                             # 已使用
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # PyTorch 峰值占用
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # PyTorch 当前占用

        # ── 计算每个 block 的字节数 ──
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # GQA/MQA: KV head 数可能比 Q head 少, 再除以 TP 并行度
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        # 每个 head 的维度, 如 128

        block_bytes = (2 *                              # K 和 V 两份
                       hf_config.num_hidden_layers *     # 每层都有 KV Cache
                       self.block_size *                 # 每个 block 的 token 数
                       num_kv_heads *                    # KV head 个数
                       head_dim *                        # 每个 head 的维度
                       hf_config.torch_dtype.itemsize)   # 每个元素的字节数 (bf16=2)
        # C++ 类比: sizeof(Block_KV) = 2 * layers * block_size * heads * dim * sizeof(half)

        # ── 计算能分配多少个 block ──
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization        # GPU 总量 × 利用率 (如 0.9)
            - used                                       # 减去已用
            - peak + current                             # 减去峰值瞬时占用
        ) // block_bytes
        # 这就是 "剩余空间能放多少个 block"
        assert config.num_kvcache_blocks > 0

        # ── 一次性分配整个 KV Cache 张量 ──
        self.kv_cache = torch.empty(
            2,                                           # 0=K_cache, 1=V_cache
            hf_config.num_hidden_layers,                 # 每层一份
            config.num_kvcache_blocks,                   # block 个数
            self.block_size,                             # 每 block token 数
            num_kv_heads,                                # KV head 数
            head_dim                                     # head 维度
        )
        # shape 示例: [2, 28, 500, 16, 8, 128]
        # 2 = K/V, 28层, 500个block, 每block 16 token, 8个KV head, 128维
        # 整个 KV Cache 在一个连续张量里! block_id 就是第 2 维的索引

        # ── 把 KV Cache 分配给每一层的 Attention ──
        layer_id = 0
        for module in self.model.modules():              # 遍历模型所有子模块
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 找到 Attention 层 (有 k_cache 和 v_cache 属性的模块)
                module.k_cache = self.kv_cache[0, layer_id]  # shape: [num_blocks, block_size, heads, dim]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        # 这样每一层的 Attention 直接引用这个大张量的一个切片

        # ── 分配 CPU 端 KV Cache (Swap 用, pinned memory 加速 GPU↔CPU 传输) ──
        # C++ 类比: cudaMallocHost (pinned memory, page-locked)
        # 比普通 CPU 内存传输快 2-3 倍, 因为避免了额外的内存拷贝
        num_cpu_blocks = config.num_kvcache_blocks // 2  # CPU block 数 = GPU 的一半
        config.num_cpu_blocks = num_cpu_blocks
        if num_cpu_blocks > 0:
            self.cpu_kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                num_cpu_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=hf_config.torch_dtype,
                device="cpu",
                pin_memory=True  # pinned memory: 加速 GPU↔CPU 传输
            )
        else:
            self.cpu_kv_cache = None

    # ═══════════════════════════════════════════════════════════
    # 准备阶段: 把 Sequence 列表转成 GPU tensor
    # ═══════════════════════════════════════════════════════════

    # ── copy_kv_blocks: CoW (Copy-on-Write) GPU 端 KV 数据复制 ──
    # 当多个序列共享同一个 KV Cache block 时,
    # 写入前需要先复制一份, 避免污染其他序列的数据
    #
    # C++ 类比: memcpy(new_page, old_page, PAGE_SIZE)
    # CUDA 类比: cudaMemcpyDeviceToDevice
    def copy_kv_blocks(self, cow_pairs: list[tuple[int, int]]):
        """复制 KV Cache blocks: 把 src block 的数据复制到 dst block"""
        for src_block_id, dst_block_id in cow_pairs:
            # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            # 复制所有层的 K 和 V
            self.kv_cache[:, :, dst_block_id].copy_(self.kv_cache[:, :, src_block_id])

    # ── swap_blocks: GPU ↔ CPU KV Cache 数据搬运 ──
    # C++ 类比: cudaMemcpyAsync(dst, src, size, direction, stream)
    #   swap_out: DeviceToHost (GPU→CPU)
    #   swap_in:  HostToDevice (CPU→GPU)
    def swap_blocks(self, swap_map: list[tuple[int, int]], direction: str):
        """搬运 KV Cache blocks
        direction='out': GPU→CPU (swap_map = [(gpu_id, cpu_id), ...])
        direction='in':  CPU→GPU (swap_map = [(cpu_id, gpu_id), ...])
        """
        if self.cpu_kv_cache is None or not swap_map:
            return
        for src_id, dst_id in swap_map:
            if direction == "out":
                # GPU → CPU (异步, non_blocking=True)
                self.cpu_kv_cache[:, :, dst_id].copy_(
                    self.kv_cache[:, :, src_id], non_blocking=True)
            else:
                # CPU → GPU (异步)
                self.kv_cache[:, :, dst_id].copy_(
                    self.cpu_kv_cache[:, :, src_id], non_blocking=True)
        # 确保搬运完成后再继续 (类似 cudaStreamSynchronize)
        torch.cuda.synchronize()

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把每条序列的 block_table 对齐并转成 GPU 张量"""
        max_len = max(len(seq.block_table) for seq in seqs)  # 找最长的 block_table
        # 不同序列的 block 数可能不同, 需要 padding 到相同长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 短的用 -1 填充 (padding)
        # 例: [[0,1,2], [0,1]] → [[0,1,2], [0,1,-1]]
        block_tables = torch.tensor(block_tables, dtype=torch.int32,
                                     pin_memory=True).cuda(non_blocking=True)
        # pin_memory=True: 分配在锁页内存 (CPU), 加速 CPU→GPU 传输
        # .cuda(non_blocking=True): 异步拷贝到 GPU, 不等拷贝完成
        # C++ 类比: cudaMemcpyAsync(gpu_ptr, cpu_pinned_ptr, size, cudaMemcpyHostToDevice)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prefill 准备: 拼接多条序列的 token, 计算位置和 slot_mapping"""
        input_ids = []          # 所有序列的 token 拼成一个一维列表
        positions = []          # 每个 token 的位置编号 (RoPE 用)
        cu_seqlens_q = [0]      # cumulative sequence lengths for Q (前缀和)
        cu_seqlens_k = [0]      # cumulative sequence lengths for K
        max_seqlen_q = 0        # 最长的 Q 序列长度 (Flash Attention 需要)
        max_seqlen_k = 0        # 最长的 K 序列长度
        slot_mapping = []       # 每个 token 在 KV Cache 中的全局槽位编号
        block_tables = None     # Prefix Cache 命中时才需要

        for seq in seqs:
            seqlen = len(seq)                           # 序列总长度

            input_ids.extend(seq[seq.num_cached_tokens:])
            # seq[start:] = 从 num_cached_tokens 位置开始的 token
            # Prefix Cache 命中的 token 不用喂给模型 → 跳过前 num_cached_tokens 个
            # 如果没有 cache hit, num_cached_tokens=0 → 喂全部 token

            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 位置编号: [num_cached_tokens, num_cached_tokens+1, ..., seqlen-1]
            # RoPE 需要每个 token 的绝对位置

            seqlen_q = seqlen - seq.num_cached_tokens   # Q 的实际长度 (去掉 cached 部分)
            seqlen_k = seqlen                           # K 的长度 = 整个序列 (包括 cached)
            # Q ≠ K 的情况: Prefix Cache 命中时, Q 只算不在 cache 里的, K 包含全部

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # Q 的累积长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # K 的累积长度
            # Flash Attention 的 varlen API 需要: 告诉它每条序列在哪里开始/结束

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup 时没有 block_table
                continue

            # ── slot_mapping: 计算每个(非 cached) token 存到 KV Cache 的哪个位置 ──
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # 从第一个非 cached block 开始
                start = seq.block_table[i] * self.block_size
                # block_table[i] 是物理 block_id, 乘以 block_size = 全局起始槽位
                # 例: block_id=5, block_size=16 → start=80
                if i != seq.num_blocks - 1:
                    end = start + self.block_size       # 不是最后一个 block → 满的
                else:
                    end = start + seq.last_block_num_tokens  # 最后一个 block → 可能不满
                slot_mapping.extend(list(range(start, end)))
                # slot_mapping 告诉 attention 层: "这个 token 的 KV 写到 cache 的第几号槽"

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:         # K 总长 > Q 总长 → 有 Prefix Cache
            block_tables = self.prepare_block_tables(seqs)
            # Prefix Cache 命中时, attention 需要 block_tables 来找 cached 的 KV

        # ── 全部转成 GPU tensor ──
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # ── 设置全局上下文 (attention 层会读取这些信息) ──
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables)
        # True = is_prefill
        # attention 层通过 get_context() 获取这些信息来决定怎么计算

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Decode 准备: 每条序列只取最后一个 token"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []           # 每条序列的上下文长度 (= 总 token 数)

        for seq in seqs:
            input_ids.append(seq.last_token)            # 只取最后一个 token
            positions.append(len(seq) - 1)              # 位置 = 序列长度 - 1
            context_lens.append(len(seq))               # 上下文长度 = 序列总长度

            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            # 新 token 的 KV 存到: 最后一个 block 的最后一个已占用位置
            # 例: block_table[-1]=5, block_size=16, last_block_num_tokens=3
            # → slot = 5*16 + 3 - 1 = 82 (0-indexed)
            # 注意: last_block_num_tokens 已经包含了刚 append 的新 token

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables)
        # False = is_decode
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数: 收集每条序列的温度"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32,
                                     pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ═══════════════════════════════════════════════════════════
    # 执行阶段: run_model + run (对外入口)
    # ═══════════════════════════════════════════════════════════

    @torch.inference_mode()
    # @torch.inference_mode(): 禁用梯度计算 + 自动求导, 推理更快更省内存
    # 比 torch.no_grad() 更彻底 (连 autograd 元数据都不创建)
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行模型前向推理, 返回 logits"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # ---- Prefill / Eager / batch 太大 → 直接跑模型 ----
            return self.model.compute_logits(self._forward_model(input_ids, positions))
            # self.model(input_ids, positions): 前向传播, 返回 hidden_states
            # self.model.compute_logits(...): 乘以 lm_head 权重 → [batch, vocab_size]
        else:
            # ---- Decode + CUDA Graph → 回放预录制的 Graph ----
            bs = input_ids.size(0)                      # batch size (序列数)
            context = get_context()                     # 获取 set_context 设的全局变量

            # 找到 >= bs 的最小预录制 batch size
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # self.graph_bs = [1, 2, 4, 8, 16, 32, ...]
            # 例: bs=3 → 找到 4 → 用为 batch=4 录制的 graph
            # next + generator: 找第一个满足条件的

            graph_vars = self.graph_vars
            # graph_vars 是录制时的 tensor → 回放前把实际数据拷进去

            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)        # 先全填 -1 (padding)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()           # 先全填 0
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            graph.replay()                               # 回放! GPU 执行预录制的所有 kernel
            # 回放不经过 Python → 无 CPU launch overhead → 极快

            return self.model.compute_logits(graph_vars["outputs"][:bs])
            # graph 输出写到 graph_vars["outputs"], 取前 bs 个

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """对外入口: scheduler 调用, 完成一次推理并返回采样的 token_ids"""
        enable_chunked = getattr(self.config, 'enable_chunked_prefill', False)
        max_chunk = getattr(self.config, 'max_chunk_size', 512)

        # warmup 时 block_table 为空, 不走 chunked prefill
        is_warmup = any(not seq.block_table for seq in seqs)
        if is_prefill and enable_chunked and not is_warmup:
            # ── Chunked Prefill: 限制每条 seq 只暴露 chunk 大小的 token ──
            # 保存原始 num_cached_tokens, 并设临时值使 prepare_prefill 只看到 chunk
            saved_cached = {}
            for seq in seqs:
                saved_cached[seq.seq_id] = seq.num_cached_tokens
                # 当前 chunk 的实际 token 数
                remaining = seq.num_prompt_tokens - seq.num_computed_tokens
                chunk = min(remaining, max_chunk)
                # 设 num_cached_tokens = num_computed_tokens, 让 prepare_prefill 从这里开始
                seq.num_cached_tokens = seq.num_computed_tokens
                # 临时截断: 设 num_tokens = num_computed_tokens + chunk
                # 同时截断 token_ids, 使 seq[num_cached_tokens:] 只取 chunk 部分
                seq._orig_num_tokens = seq.num_tokens
                seq._orig_token_ids = seq.token_ids
                seq.num_tokens = seq.num_computed_tokens + chunk
                seq.token_ids = seq.token_ids[:seq.num_tokens]

        # 1. 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 2. 前向推理
        logits = self.run_model(input_ids, positions, is_prefill)

        # 3. 采样
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        if is_prefill and enable_chunked and not is_warmup:
            # ── 恢复 num_tokens, 更新 num_computed_tokens ──
            for i, seq in enumerate(seqs):
                chunk = seq.num_tokens - seq.num_computed_tokens  # 本次 chunk 大小
                seq.num_computed_tokens += chunk
                seq.num_cached_tokens = seq.num_computed_tokens  # 同步: 下次 prepare 跳过已算的
                seq.num_tokens = seq._orig_num_tokens  # 恢复真实总 token 数
                seq.token_ids = seq._orig_token_ids     # 恢复完整 token 列表
                del seq._orig_num_tokens, seq._orig_token_ids
                if not seq.is_prefill_finished:
                    # 中间 chunk: 不采样, token_id 设为 None
                    if token_ids is not None:
                        token_ids[i] = None

        # 4. 清理
        reset_context()
        return token_ids

    # ═══════════════════════════════════════════════════════════
    # CUDA Graph: 预录制 Decode 操作, 消除 CPU 开销
    # ═══════════════════════════════════════════════════════════

    @torch.inference_mode()
    def capture_cudagraph(self):
        """录制不同 batch size 的 CUDA Graph"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)     # 最大 batch size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 最多需要多少个 block (向上取整)

        # ── 创建"占位"tensor (graph 录制时绑定这些 tensor 的地址) ──
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # ── 要录制的 batch size 列表 ──
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # [1, 2, 4, 8, 16, 32, 48, 64, ...]
        # 不是每个 bs 都录, 只录这些 "档位"
        # 实际 bs 不在列表里时, 向上找最近的 (如 bs=3 → 用 bs=4 的 graph)

        self.graphs = {}
        self.graph_pool = None                           # 共享 GPU 内存池

        for bs in reversed(self.graph_bs):               # 从大到小录
            graph = torch.cuda.CUDAGraph()

            # warmup: 先跑一次, 让 CUDA 编译 kernel
            set_context(False, slot_mapping=slot_mapping[:bs],
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self._forward_model(input_ids[:bs], positions[:bs])

            # capture: 录制!
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self._forward_model(input_ids[:bs], positions[:bs])
            # torch.cuda.graph(graph, pool):
            #   pool: 共享内存池, 不同 bs 的 graph 共享 GPU workspace
            #   with 块内的所有 GPU 操作被录制到 graph 里
            #   不会真正执行, 只是记录 "要执行哪些 kernel"

            if self.graph_pool is None:
                self.graph_pool = graph.pool()           # 第一个 graph 创建池

            self.graphs[bs] = graph                      # 存起来, 按 bs 索引
            torch.cuda.synchronize()
            reset_context()

        # ── 保存占位 tensor 的引用 ──
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        # 回放时: 把实际数据拷到这些 tensor 里 → replay → 读 outputs
        # 这些 tensor 的 GPU 地址在整个生命周期内不变 (CUDA Graph 的要求)
