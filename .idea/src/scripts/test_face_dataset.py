#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试人脸数据集加载
"""

import os
import sys
sys.path.append('..')

from core.datasets import FaceDataset

def main():
    print("测试FaceDataset加载...")

    # 测试路径
    face_data_dir = '../data/face/face'
    print(f"数据路径: {face_data_dir}")
    print(f"绝对路径: {os.path.abspath(face_data_dir)}")
    print(f"路径存在: {os.path.exists(face_data_dir)}")

    if not os.path.exists(face_data_dir):
        print("❌ 数据路径不存在")
        return

    try:
        # 创建数据集
        dataset = FaceDataset(data_dir=face_data_dir, mode='train', image_size=224)

        print("✅ 数据集创建成功"        print(f"类别数量: {len(dataset.class_to_idx)}")
        print(f"图像数量: {len(dataset.image_paths)}")
        print(f"标签数量: {len(dataset.labels)}")

        if len(dataset.class_to_idx) > 0:
            print(f"类别映射: {list(dataset.class_to_idx.items())[:3]}...")

        if len(dataset.image_paths) > 0:
            print(f"前3个图像路径: {dataset.image_paths[:3]}")

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()