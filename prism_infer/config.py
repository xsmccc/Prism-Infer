import os                           # os.path.isdir: 检查目录是否存在
from dataclasses import dataclass   # @dataclass: 自动生成__init__的装饰器(类似C++ struct带默认值)
from transformers import AutoConfig # HuggingFace: 根据模型目录自动加载config.json中的模型结构配置

from prism_infer.engine.compression import build_visual_pruning_config, normalize_compression_mode
from prism_infer.engine.sequence import Sequence


@dataclass
class Config:
    """引擎级配置: 整个引擎生命期内不变(创建时确定)"""
    model: str                              # 模型本地路径(必填, 无默认值)
    max_num_batched_tokens: int = 16384     # 一次batch最多处理多少token(控制显存)
    max_num_seqs: int = 512                 # 一次batch最多多少条序列
    max_model_len: int = 4096               # 单条序列最大长度(prompt+生成)
    gpu_memory_utilization: float = 0.9     # GPU显存利用率上限(90%)
    tensor_parallel_size: int = 1           # 张量并行数(几块GPU)
    enforce_eager: bool = False             # True=禁用CUDA Graph, 用eager模式执行
    hf_config: AutoConfig | None = None     # HF模型结构配置(层数/隐藏维度/头数), __post_init__自动填充
    eos: int = -1                           # EOS token id, llm_engine.py中由tokenizer填充
    kvcache_block_size: int = 256           # KV Cache每个物理块存多少token(需256对齐)
    num_kvcache_blocks: int = -1            # KV Cache物理块总数, model_runner.py中根据GPU显存自动计算
    enable_chunked_prefill: bool = True     # 是否启用Chunked Prefill(分块预填充)
    max_chunk_size: int = 512               # 每次Prefill最多处理的token数
    enable_prefix_caching: bool = True       # full-block token prefix hash/reuse
    compression_mode: str = "off"           # off | visual_prune | visual_compact | fp8_kv | visual_compact_fp8
    enable_visual_pruning_shadow: bool = False  # off 模式下只记录 pruning decision, 不改变 KV
    visual_pruning_keep_ratio: float = 0.6       # visual pruning 目标保留比例
    visual_pruning_min_keep_tokens: int = 32     # 最少保留 visual token 数
    visual_pruning_strategy: str = "uniform"     # "uniform" | "score"; runtime 当前不提供 score
    decode_compile_region: str = "none"          # "none" | "attention"; 仅作用于 decode
    decode_compile_mode: str = "default"         # Inductor mode: "default" | "reduce-overhead"
    decode_compile_emulate_precision_casts: bool = True  # 保持 BF16 eager 中间 cast 语义
    decode_compile_force_same_precision: bool = True  # 统一 Triton/CUBLAS matmul 精度选择
    allow_unsafe_decode_compile: bool = False  # 仅 P6.3 benchmark 复现被拒绝的候选

    def __post_init__(self):
        """dataclass专属钩子: __init__自动生成后自动调用, 用于参数校验和衍生值计算"""
        assert os.path.isdir(self.model)                    # 模型路径必须是本地已下载的目录
        assert self.kvcache_block_size % 256 == 0            # block_size需256对齐(FlashAttention kernel要求)
        assert 1 <= self.tensor_parallel_size <= 8           # TP并行数1~8(单机最多8块GPU)
        self.compression_mode = normalize_compression_mode(self.compression_mode)
        if self.decode_compile_region not in ("none", "attention"):
            raise ValueError(
                "decode_compile_region must be 'none' or 'attention', "
                f"got {self.decode_compile_region!r}"
            )
        if self.decode_compile_mode not in ("default", "reduce-overhead"):
            raise ValueError(
                "decode_compile_mode must be 'default' or 'reduce-overhead', "
                f"got {self.decode_compile_mode!r}"
            )
        if self.decode_compile_region != "none" and not self.enforce_eager:
            raise ValueError(
                "decode torch.compile region and CUDA Graph are mutually exclusive"
            )
        if self.decode_compile_region != "none" and self.compression_mode != "off":
            raise ValueError(
                "P6.3 decode compile preflight requires compression_mode='off'"
            )
        if (
            self.decode_compile_region != "none"
            and not self.allow_unsafe_decode_compile
        ):
            raise ValueError(
                "decode torch.compile is a rejected P6.3 preflight candidate; "
                "set allow_unsafe_decode_compile=True only to reproduce benchmark evidence"
            )
        build_visual_pruning_config(self)
        Sequence.set_block_size(self.kvcache_block_size)
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 从模型目录的config.json加载模型结构
        model_max_len = getattr(
            self.hf_config,
            "max_position_embeddings",
            getattr(getattr(self.hf_config, "text_config", None), "max_position_embeddings", self.max_model_len),
        )
        self.max_model_len = min(self.max_model_len, model_max_len)  # 不超过模型支持的最大位置编码
        assert self.max_num_batched_tokens >= self.max_model_len  # batch容量必须>=单条最大长度
