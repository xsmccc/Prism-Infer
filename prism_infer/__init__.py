# __init__.py: Python包的"门面"文件
# 当用户写 from prism_infer import LLM, SamplingParams 时, Python执行此文件
# 作用: 把内部模块的类"提升"到包顶层, 用户不需要知道具体在哪个文件里
#
# LLM 会导入 engine/tokenizer 依赖，进而需要 transformers。模型单元测试只需要
# prism_infer.models / prism_infer.vision，不应该被这些运行时依赖卡住，所以这里
# 延迟导入 LLM。
from prism_infer.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from prism_infer.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
