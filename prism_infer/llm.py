from prism_infer.engine.llm_engine import LLMEngine  # 导入真正的引擎实现类


class LLM(LLMEngine):
    """用户API入口: 空壳继承LLMEngine, 所有方法透传给父类
    为什么不直接用LLMEngine?
    1. 用户写LLM("model")比LLMEngine("model")更简洁
    2. 将来可在LLM层添加便利方法(chat/stream)而不改引擎核心
    3. 与真实vLLM的API设计保持一致
    """

    pass
