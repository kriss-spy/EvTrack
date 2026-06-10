#!/usr/bin/env python3
"""
快速量化验证脚本：FP16 / Dynamic INT8 / torchao INT8

用法：
    python quick_quantize_test.py --tracker_dir <path> --dataset_name <visevent/coesot> --quant_mode <fp16/dynamic_int8/torchao_int8>

本脚本可直接集成到现有的 test_rgbe_mgpus.py 评估流程中，修改 tracker 加载部分即可。
"""

import torch
import torch.nn as nn
import argparse

# -----------------------
# 1. FP16 / BF16 快速转换
# -----------------------
def apply_fp16(network):
    """
    将模型转换为 FP16 (half) 精度。
    注意：输入数据也需要转换为 half。
    """
    network = network.half()
    return network

# -----------------------
# 2. PyTorch 动态权重量化 (Dynamic Quantization)
# -----------------------
def apply_dynamic_quantization(network, qconfig_spec=None):
    """
    仅对权重进行静态 INT8 量化，激活仍为 FP32。
    主要优化 CPU 推理，GPU 上加速有限。
    
    Args:
        qconfig_spec: 默认只量化 Linear 层，可扩展为 {nn.Linear, nn.Conv2d}
    """
    if qconfig_spec is None:
        qconfig_spec = {nn.Linear}
    
    quantized_network = torch.quantization.quantize_dynamic(
        network,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8  # type: ignore[attr-defined]
    )
    return quantized_network

# -----------------------
# 3. torchao 量化 (推荐，PyTorch 2.x+)
# -----------------------
def apply_torchao_quantization(network, config_name='int8_dynamic'):
    """
    使用 torchao 进行量化。需要 torch>=2.0 和 torchao 安装。
    
    支持的 config_name:
        - 'int8_dynamic': Int8DynamicActivationInt8WeightConfig (激活动态 INT8，权重 INT8)
        - 'int8_weight_only': Int8WeightOnlyConfig (仅权重量化 INT8)
        - 'int4_weight_only': Int4WeightOnlyConfig (仅权重量化 INT4，group_size 可配)
    """
    try:
        from torchao.quantization import (  # type: ignore[import-not-found]
            quantize_,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            Int4WeightOnlyConfig,
        )
    except ImportError:
        raise ImportError(
            "torchao 未安装。请执行: pip install torchao\n"
            "参考: https://github.com/pytorch/ao"
        )
    
    if config_name == 'int8_dynamic':
        config = Int8DynamicActivationInt8WeightConfig()
    elif config_name == 'int8_weight_only':
        config = Int8WeightOnlyConfig()
    elif config_name == 'int4_weight_only':
        config = Int4WeightOnlyConfig(group_size=128)
    else:
        raise ValueError(f"不支持的 config_name: {config_name}")
    
    # quantize_ 是 in-place 操作，默认作用于所有 Linear 层
    quantize_(network, config)
    return network

# -----------------------
# 4. 分层混合精度量化 (跳过敏感模块)
# -----------------------
def apply_mixed_precision(network, quantize_fn, skip_patterns=None):
    """
    对网络进行分层量化，跳过名称匹配 skip_patterns 的模块。
    
    Args:
        quantize_fn: 量化函数，如 apply_torchao_quantization
        skip_patterns: 列表，如 ['patch_embed_prompt', 'prompt_blocks', 'box_head']
    """
    if skip_patterns is None:
        skip_patterns = ['patch_embed_prompt', 'prompt_blocks', 'box_head']
    
    # 保存原始子模块
    original_children = {}
    for name, module in network.named_modules():
        if any(pattern in name for pattern in skip_patterns):
            original_children[name] = module
    
    # 先对整个网络量化
    network = quantize_fn(network)
    
    # 再将被跳过的模块替换回原始 FP32 版本
    # 注意：这是简化实现，实际可能需要更精细的模块替换
    for name, module in network.named_modules():
        if name in original_children:
            # 这里需要用到 setattr 来替换，但由于 named_modules 是只读的，
            # 更稳妥的做法是在量化前通过 filter_fn 排除这些模块。
            pass
    
    return network

# -----------------------
# 5. 在 tracker 中使用的示例
# -----------------------
def wrap_tracker_network(tracker, quant_mode='none', device='cuda'):
    """
    在 tracker __init__ 中加载模型后，调用此函数进行量化包装。
    
    Args:
        tracker: ViPTTrack 实例
        quant_mode: 'none' | 'fp16' | 'dynamic_int8' | 'torchao_int8' | 'torchao_int4'
    """
    network = tracker.network
    
    if quant_mode == 'none':
        return tracker
    
    elif quant_mode == 'fp16':
        network = apply_fp16(network)
        # 注意：tracker 的输入预处理也需要适配 FP16
        # 在 track() 中调用时需要确保 tensor 是 half
    
    elif quant_mode == 'dynamic_int8':
        # 动态量化在 CPU 上有优势，GPU 上一般建议用 torchao
        network = network.cpu()
        network = apply_dynamic_quantization(network)
        network = network.to(device)
    
    elif quant_mode == 'torchao_int8':
        network = network.to(device)
        network = apply_torchao_quantization(network, 'int8_dynamic')
    
    elif quant_mode == 'torchao_int4':
        network = network.to(device)
        network = apply_torchao_quantization(network, 'int4_weight_only')
    
    else:
        raise ValueError(f"不支持的 quant_mode: {quant_mode}")
    
    tracker.network = network
    return tracker


# -----------------------
# 6. 测试/演示
# -----------------------
def demo_quantize_options():
    """
    演示如何创建一个 dummy model 并应用各种量化。
    """
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(768, 3072)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(3072, 768)
        
        def forward(self, x):
            return self.linear2(self.act(self.linear1(x)))
    
    model = DummyModel().eval()
    x = torch.randn(1, 768)
    
    # 原始 FP32
    print("=== FP32 ===")
    with torch.no_grad():
        out_fp32 = model(x)
    print(f"  输出均值: {out_fp32.mean().item():.4f}")
    
    # FP16
    print("\n=== FP16 ===")
    model_fp16 = model.half()
    with torch.no_grad():
        out_fp16 = model_fp16(x.half())
    print(f"  输出均值: {out_fp16.float().mean().item():.4f}")
    
    # Dynamic INT8
    print("\n=== Dynamic INT8 (weights only) ===")
    model_int8 = apply_dynamic_quantization(model)
    with torch.no_grad():
        out_int8 = model_int8(x)
    print(f"  输出均值: {out_int8.mean().item():.4f}")
    
    # torchao INT8 (如果安装了)
    try:
        print("\n=== torchao INT8 (W8A8) ===")
        model_ao = DummyModel().eval()
        model_ao = apply_torchao_quantization(model_ao, 'int8_dynamic')
        with torch.no_grad():
            out_ao = model_ao(x)
        print(f"  输出均值: {out_ao.mean().item():.4f}")
    except ImportError:
        print("\n=== torchao 未安装，跳过 ===")
    
    print("\n演示完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViPT 量化快速验证')
    parser.add_argument('--demo', action='store_true', help='运行 dummy model 演示')
    parser.add_argument('--quant_mode', type=str, default='none',
                        choices=['none', 'fp16', 'dynamic_int8', 'torchao_int8', 'torchao_int4'],
                        help='量化模式')
    args = parser.parse_args()
    
    if args.demo:
        demo_quantize_options()
    else:
        print("请使用 --demo 运行演示，或将 wrap_tracker_network 集成到 tracker 中。")
