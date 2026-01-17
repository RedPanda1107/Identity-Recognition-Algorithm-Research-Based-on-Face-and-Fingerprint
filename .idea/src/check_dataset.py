#!/usr/bin/env python
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config
from core.datasets.face_dataset import FaceDataset

def check_dataset_distribution():
    """检查数据集分布情况"""
    config = load_config("configs/face_config.yaml")

    # 解析数据目录路径
    face_data_dir = config["paths"]["face_data_dir"]
    if not os.path.isabs(face_data_dir):
        face_data_dir = os.path.join(project_root, face_data_dir.lstrip('./'))

    print("=== 数据集检查 ===")
    print(f"数据目录: {face_data_dir}")

    # 创建训练和验证数据集
    train_dataset = FaceDataset(face_data_dir, mode="train", image_size=config["data"]["face_image_size"])
    val_dataset = FaceDataset(face_data_dir, mode="val", image_size=config["data"]["face_image_size"])

    print("\n=== 训练集统计 ===")
    print(f"总样本数: {len(train_dataset)}")

    # 统计每个类别的样本数
    from collections import Counter
    train_labels = [label for _, label in train_dataset.labels]
    train_class_counts = Counter(train_labels)

    print("各类别样本分布:")
    for class_idx, count in sorted(train_class_counts.items()):
        class_name = [name for name, idx in train_dataset.class_to_idx.items() if idx == class_idx][0]
        print(f"  {class_name} (ID: {class_idx}): {count} 个样本")

    print("\n=== 验证集统计 ===")
    print(f"总样本数: {len(val_dataset)}")

    val_labels = [label for _, label in val_dataset.labels]
    val_class_counts = Counter(val_labels)

    print("各类别样本分布:")
    for class_idx, count in sorted(val_class_counts.items()):
        class_name = [name for name, idx in val_dataset.class_to_idx.items() if idx == class_idx][0]
        print(f"  {class_name} (ID: {class_idx}): {count} 个样本")

    # 检查是否有类别在验证集中只有一个样本
    single_sample_classes = [class_idx for class_idx, count in val_class_counts.items() if count == 1]
    if single_sample_classes:
        print("\n⚠️  警告: 以下类别在验证集中只有一个样本:")
        for class_idx in single_sample_classes:
            class_name = [name for name, idx in val_dataset.class_to_idx.items() if idx == class_idx][0]
            print(f"  {class_name} (ID: {class_idx})")

    print(f"\n总类别数: {len(train_dataset.class_to_idx)}")
    print(f"训练集类别数: {len(train_class_counts)}")
    print(f"验证集类别数: {len(val_class_counts)}")

if __name__ == "__main__":
    check_dataset_distribution()