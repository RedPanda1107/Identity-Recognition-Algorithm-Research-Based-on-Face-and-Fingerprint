#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸识别测试脚本
用于评估训练好的模型在测试集上的性能
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from core.datasets import FaceDataset
from core.models import create_face_model
from core.utils import (
    setup_logger, load_config, load_checkpoint,
    calculate_metrics, plot_confusion_matrix, set_seed, get_device
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='人脸识别测试脚本')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_latest.pth',
                       help='模型检查点路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='结果输出目录（默认使用配置文件中的路径）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='测试批次大小（覆盖配置文件）')
    return parser.parse_args()


def test_model(model, test_loader, device, logger):
    """测试模型性能"""
    model.eval()
    logger.info("开始测试...")

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    progress_bar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            paths = batch['path']

            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            # 收集结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_preds, all_labels, all_probs, all_paths


def analyze_results(predictions, labels, probabilities, paths, class_names, output_dir, logger):
    """分析测试结果"""
    logger.info("分析测试结果...")

    # 计算详细指标
    metrics = calculate_metrics(labels, predictions, num_classes=len(class_names))

    # 打印结果
    logger.info("=" * 60)
    logger.info("测试结果:")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info("=" * 60)

    # 打印每类性能
    logger.info("每类性能指标:")
    logger.info("-" * 60)
    for i, class_name in enumerate(class_names):
        logger.info("2d")

    # 找出错误分类的样本
    incorrect_indices = np.where(predictions != labels)[0]
    logger.info(f"\n错误分类样本数量: {len(incorrect_indices)}")

    # 显示前10个错误分类
    if len(incorrect_indices) > 0:
        logger.info("前10个错误分类样本:")
        for i in range(min(10, len(incorrect_indices))):
            idx = incorrect_indices[i]
            true_label = class_names[labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = probabilities[idx][predictions[idx]]
            logger.info("3d")

    # 生成混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        labels, predictions, class_names,
        save_path=cm_path, figsize=(12, 10)
    )
    logger.info(f"混淆矩阵已保存到: {cm_path}")

    return metrics


def save_results(results, output_dir, logger):
    """保存测试结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为JSON
    import json
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 将numpy数组转换为列表
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_results[key][sub_key] = sub_value.tolist()
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"测试结果已保存到: {results_path}")


def main():
    """主测试函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)
    logger = setup_logger()

    # 设置随机种子
    set_seed(config['misc']['seed'])

    # 获取计算设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    # 创建测试数据集
    logger.info("创建测试数据集...")

    face_data_dir = config['paths'].get('face_data_dir', os.path.join(config['paths']['data_dir'], 'face'))
    if not os.path.exists(face_data_dir):
        logger.error(f"人脸数据目录不存在: {face_data_dir}")
        return

    test_dataset = FaceDataset(
        data_dir=face_data_dir,
        mode='val',  # 使用验证模式（无数据增强）
        image_size=config['data']['face_image_size']
    )

    # 创建数据加载器
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['misc']['num_workers'],
        pin_memory=True
    )

    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"类别数量: {len(test_dataset.class_to_idx)}")

    # 创建模型
    logger.info("创建模型...")
    model = create_face_model(
        model_type=config['model'].get('model_type', 'facenet'),
        num_classes=len(test_dataset.class_to_idx),
        embedding_dim=config['model']['face_embedding_dim'],
        pretrained=False  # 测试时不需要预训练
    )

    # 加载检查点
    if not os.path.exists(args.checkpoint):
        logger.error(f"检查点文件不存在: {args.checkpoint}")
        return

    logger.info(f"加载检查点: {args.checkpoint}")
    epoch, loss, accuracy = load_checkpoint(args.checkpoint, model)
    logger.info(f"检查点信息 - Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # 将模型移到设备
    model = model.to(device)

    # 执行测试
    predictions, labels, probabilities, paths = test_model(
        model, test_loader, device, logger
    )

    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else config['paths'].get('results_dir', './results/face')

    # 分析结果
    class_names = list(test_dataset.class_to_idx.keys())
    metrics = analyze_results(
        predictions, labels, probabilities, paths,
        class_names, output_dir, logger
    )

    # 保存结果
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities,
        'paths': paths,
        'class_names': class_names,
        'checkpoint_info': {
            'path': args.checkpoint,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy
        }
    }

    save_results(results, output_dir, logger)

    logger.info("测试完成！")


if __name__ == "__main__":
    main()