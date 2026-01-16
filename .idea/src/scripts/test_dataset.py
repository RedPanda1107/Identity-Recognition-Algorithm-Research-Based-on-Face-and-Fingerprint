#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集测试脚本
验证人脸数据集是否能正确加载
"""

import os
import sys
sys.path.append('..')

from core.datasets import FaceDataset
from core.utils import load_config

def main():
    print("=" * 50)
    print("人脸数据集测试")
    print("=" * 50)

    # 加载配置
    config = load_config('../configs/face_config.yaml')
    face_data_dir = config['paths']['face_data_dir']

    print(f"数据目录: {face_data_dir}")
    print(f"目录存在: {os.path.exists(face_data_dir)}")

    if not os.path.exists(face_data_dir):
        print("❌ 数据目录不存在！")
        return 1

    try:
        # 创建数据集
        print("\n创建训练数据集...")
        dataset = FaceDataset(data_dir=face_data_dir, mode='train', image_size=224)

        print("✅ 数据集创建成功！")
        print(f"📊 类别数量: {len(dataset.class_to_idx)}")
        print(f"📸 总样本数: {len(dataset)}")
        print(f"🏷️ 类别列表: {sorted(list(dataset.class_to_idx.keys()))}")

        # 检查类别分布
        from collections import Counter
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        label_counts = Counter(labels)
        print(f"📈 各类别样本数: {dict(sorted(label_counts.items()))}")

        # 检查一个样本
        print("\n🔍 检查第一个样本...")
        sample = dataset[0]
        print(f"🖼️ 图像形状: {sample['image'].shape}")
        print(f"🏷️ 标签: {sample['label']}")
        print(f"📁 路径: {os.path.basename(sample['path'])}")

        # 检查图像格式
        from PIL import Image
        img = Image.open(sample['path'])
        print(f"📷 原始图像格式: {img.format}")
        print(f"📐 原始图像尺寸: {img.size}")
        print(f"🎨 图像模式: {img.mode}")

        print("\n✅ 数据集验证通过！")
        print("您可以开始训练人脸识别模型了。")

        return 0

    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())