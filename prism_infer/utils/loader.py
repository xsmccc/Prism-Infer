# ═══════════════════════════════════════════════════════════════
# loader.py —— 模型权重加载
#
# 职责: 从 safetensors 文件加载权重到模型
#       处理"合并线性层" (如 QKV 合并、Gate+Up 合并) 的特殊加载
#
# safetensors 格式: HuggingFace 的安全权重格式 (比 pickle 安全, 支持内存映射)
#
# C++ 类比: 反序列化模型权重 → memcpy 到模型参数
# ═══════════════════════════════════════════════════════════════

import os
from glob import glob  # 文件名模式匹配: glob("*.safetensors") → 列出所有 .safetensors 文件
import torch
from torch import nn
from safetensors import safe_open  # safetensors 库: 安全(不执行代码)的权重加载


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认加载方式: 直接复制权重到参数 (shape 完全匹配的情况)"""
    param.data.copy_(loaded_weight)
    # param.data: 参数的底层张量 (绕过 autograd)
    # .copy_(): 原地复制 (in-place), 不改变 param 的地址
    # C++: memcpy(param_ptr, weight_ptr, sizeof(weight))


def load_model(model: nn.Module, path: str):
    """
    加载模型权重

    path: 模型文件目录 (如 "Qwen/Qwen3-8B/")
    该目录下有多个 .safetensors 文件, 每个文件包含部分权重
    """

    # packed_modules_mapping: 模型定义的"合并层"映射表
    # 例: Qwen3 把 q_proj, k_proj, v_proj 合并成一个 qkv_proj
    # 权重文件里叫 "q_proj", 但模型里叫 "qkv_proj" → 需要映射
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # getattr 带默认值 {}: 如果模型没定义这个属性, 就用空字典
    # 空字典 → 没有合并层 → 所有权重都直接 copy

    for file in glob(os.path.join(path, "*.safetensors")):
        # 遍历目录下所有 .safetensors 文件
        # glob: 通配符匹配, * = 任意字符串
        # os.path.join: 拼接路径 (自动处理 / 或 \)

        with safe_open(file, "pt", "cpu") as f:
            # safe_open: 以 PyTorch ("pt") 格式打开, 加载到 CPU
            # with: 确保用完后关闭文件

            for weight_name in f.keys():
                # f.keys(): 文件中所有权重的名字
                # 例: "model.layers.0.self_attn.q_proj.weight"

                for k in packed_modules_mapping:
                    # 检查这个权重名是否属于"合并层"
                    if k in weight_name:
                        # 命中! 例: k="q_proj", weight_name 里有 "q_proj"
                        v, shard_id = packed_modules_mapping[k]
                        # v="qkv_proj" (合并后的参数名)
                        # shard_id=0 (q 是第 0 片, k 是第 1 片, v 是第 2 片)

                        param_name = weight_name.replace(k, v)
                        # "...q_proj.weight" → "...qkv_proj.weight"
                        # 把文件中的名字替换成模型中的参数名

                        param = model.get_parameter(param_name)
                        # 从模型中找到这个参数 (nn.Parameter 对象)

                        weight_loader = getattr(param, "weight_loader")
                        # 合并层的参数有自定义的 weight_loader 方法
                        # 它知道怎么把 q_proj 的权重拼到 qkv_proj 的第 0 片

                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # f.get_tensor: 从文件中读取权重张量
                        # weight_loader(param, weight, shard_id):
                        #   把 weight 拷贝到 param 的第 shard_id 片
                        break
                        # break: 找到匹配就跳出 packed_modules_mapping 循环
                else:
                    # for-else: 如果上面的 for 循环没有 break (没有匹配到合并层)
                    # → 这是一个普通的非合并权重 → 直接拷贝

                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 如果参数有自定义 weight_loader 就用它 (如 TP 分片的权重)
                    # 否则用 default_weight_loader (直接 copy_)
                    weight_loader(param, f.get_tensor(weight_name))
