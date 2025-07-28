# -*- coding:utf-8 -*-
# @Time      : 2025/7/21 22:03
# @Author    : yaomw

# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
import torch

# 检查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 检查 CUDA 是否可用
is_available = torch.cuda.is_available()
print(f"CUDA Available: {is_available}")

if is_available:
    # 获取 GPU 数量
    print(f"Device Count: {torch.cuda.device_count()}")
    # 获取当前 GPU 的索引
    print(f"Current Device: {torch.cuda.current_device()}")
    # 获取当前 GPU 的名称
    print(f"Device Name: {torch.cuda.get_device_name(0)}")