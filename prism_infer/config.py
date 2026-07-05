import os                           # os.path.isdir: 检查目录是否存在
from dataclasses import dataclass   # @dataclass: 自动生成__init__的装饰器(类似C++ struct带默认值)
from transformers import AutoConfig # HuggingFace: 根据模型目录自动加载config.json中的模型结构配置

from prism_infer.engine.compression import normalize_compression_mode
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
    compression_mode: str = "off"           # P5 KV 压缩模式；P5.0 只支持 off baseline

    def __post_init__(self):
        """dataclass专属钩子: __init__自动生成后自动调用, 用于参数校验和衍生值计算"""
        assert os.path.isdir(self.model)                    # 模型路径必须是本地已下载的目录
        assert self.kvcache_block_size % 256 == 0            # block_size需256对齐(FlashAttention kernel要求)
        assert 1 <= self.tensor_parallel_size <= 8           # TP并行数1~8(单机最多8块GPU)
        self.compression_mode = normalize_compression_mode(self.compression_mode)
        Sequence.set_block_size(self.kvcache_block_size)
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 从模型目录的config.json加载模型结构
        model_max_len = getattr(
            self.hf_config,
            "max_position_embeddings",
            getattr(getattr(self.hf_config, "text_config", None), "max_position_embeddings", self.max_model_len),
        )
        self.max_model_len = min(self.max_model_len, model_max_len)  # 不超过模型支持的最大位置编码
        assert self.max_num_batched_tokens >= self.max_model_len  # batch容量必须>=单条最大长度
