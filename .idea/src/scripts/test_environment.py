#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境测试脚本
验证PyTorch环境和基本功能是否正常
"""

import sys
import torch
import torchvision
import numpy as np
import yaml
import PIL
import matplotlib
import sklearn

def test_imports():
    """测试必要的包导入"""
    print("测试包导入...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision 导入失败: {e}")
        return False

    try:
        import numpy
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False

    try:
        import PIL
        print(f"✓ PIL {PIL.__version__}")
    except ImportError as e:
        print(f"✗ PIL 导入失败: {e}")
        return False

    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML 导入失败: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False

    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn 导入失败: {e}")
        return False

    return True

def test_cuda():
    """测试CUDA可用性"""
    print("\n测试CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        return True
    else:
        print("✗ CUDA不可用，将使用CPU")
        return False

def test_tensor_operations():
    """测试张量操作"""
    print("\n测试张量操作...")
    try:
        # 创建张量
        x = torch.randn(3, 224, 224)
        print(f"✓ 创建张量: {x.shape}")

        # 基本运算
        y = x * 2 + 1
        print(f"✓ 基本运算: {y.shape}")

        # 移动到设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        print(f"✓ 设备转移: {x.device}")

        return True
    except Exception as e:
        print(f"✗ 张量操作失败: {e}")
        return False

def test_custom_imports():
    """测试自定义模块导入"""
    print("\n测试自定义模块...")
    try:
        from core.utils import setup_logger, load_config
        print("✓ 工具函数导入成功")
    except ImportError as e:
        print(f"✗ 工具函数导入失败: {e}")
        return False

    try:
        from core.datasets import FaceDataset
        print("✓ 数据集类导入成功")
    except ImportError as e:
        print(f"✗ 数据集类导入失败: {e}")
        return False

    try:
        from core.models import create_face_model
        print("✓ 模型类导入成功")
    except ImportError as e:
        print(f"✗ 模型类导入失败: {e}")
        return False

    return True

def test_config_loading():
    """测试配置文件加载"""
    print("\n测试配置文件加载...")
    try:
        from core.utils import load_config
        config = load_config('configs/face_config.yaml')
        print("✓ 配置文件加载成功")
        print(f"  - 模型类型: {config['model']['model_type']}")
        print(f"  - 嵌入维度: {config['model']['face_embedding_dim']}")
        print(f"  - 批次大小: {config['training']['batch_size']}")
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("人脸识别环境测试")
    print("=" * 50)

    all_passed = True

    # 测试包导入
    if not test_imports():
        all_passed = False

    # 测试CUDA
    test_cuda()  # 不影响all_passed，因为CPU也可以运行

    # 测试张量操作
    if not test_tensor_operations():
        all_passed = False

    # 测试自定义模块
    if not test_custom_imports():
        all_passed = False

    # 测试配置加载
    if not test_config_loading():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！环境配置正确。")
        print("您可以开始训练人脸识别模型了。")
    else:
        print("✗ 部分测试失败，请检查环境配置。")
        print("建议运行: pip install -r requirements.txt")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())