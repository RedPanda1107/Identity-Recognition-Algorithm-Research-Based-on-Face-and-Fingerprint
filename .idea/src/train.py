#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸识别训练脚本
支持单模态人脸识别训练，包含数据增强、模型检查点、日志记录等功能
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from core.datasets import FaceDataset
from core.models import create_face_model
from core.utils import (
    setup_logger, load_config, save_checkpoint, load_checkpoint,
    calculate_metrics, plot_training_curves, set_seed, get_device,
    count_parameters, TensorBoardWriter, AverageMeter
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='人脸识别训练脚本')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')
    parser.add_argument('--experiment_name', type=str, default='face_recognition',
                       help='实验名称')
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, tb_writer=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()

        # 更新统计
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })

        # TensorBoard记录
        if tb_writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            tb_writer.add_scalar('train/batch_loss', loss.item(), global_step)
            tb_writer.add_scalar('train/batch_acc', accuracy, global_step)

    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, epoch, logger, tb_writer=None):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    all_preds = []
    all_labels = []

    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean().item()

            # 更新统计
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))

            # 收集预测结果用于计算其他指标
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })

    # 计算详细指标
    metrics = calculate_metrics(all_labels, all_preds)

    # TensorBoard记录
    if tb_writer:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/accuracy', accuracies.avg, epoch)
        tb_writer.add_scalar('val/precision', metrics['precision'], epoch)
        tb_writer.add_scalar('val/recall', metrics['recall'], epoch)
        tb_writer.add_scalar('val/f1_score', metrics['f1_score'], epoch)

    return losses.avg, accuracies.avg, metrics


def main():
    """主训练函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 获取实验名称（优先使用命令行参数，其次使用配置文件）
    experiment_name = args.experiment_name

    # 设置日志记录器（会自动创建实验名称子目录）
    logger = setup_logger(experiment_name=experiment_name)

    # 设置随机种子
    set_seed(config['misc']['seed'])
    logger.info(f"设置随机种子: {config['misc']['seed']}")

    # 获取计算设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    # 创建数据加载器
    logger.info("创建数据加载器...")

    # 检查数据目录 - 强制使用正确的路径
    face_data_dir = './data/face/face'  # 直接指定正确路径
    logger.info(f"使用数据目录: {face_data_dir}")
    logger.info(f"绝对路径: {os.path.abspath(face_data_dir)}")
    logger.info(f"当前工作目录: {os.getcwd()}")

    if not os.path.exists(face_data_dir):
        logger.error(f"人脸数据目录不存在: {face_data_dir}")
        logger.error(f"绝对路径: {os.path.abspath(face_data_dir)}")
        logger.error("请确保数据文件位于正确的目录中")
        return

    # 创建训练数据集
    train_dataset = FaceDataset(
        data_dir=face_data_dir,
        mode='train',
        image_size=config['data']['face_image_size']
    )

    # 创建验证数据集（这里暂时使用相同的目录，后续可以改进为真正的验证集）
    val_dataset = FaceDataset(
        data_dir=face_data_dir,
        mode='val',
        image_size=config['data']['face_image_size']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['misc']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['misc']['num_workers'],
        pin_memory=True
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"类别数量: {len(train_dataset.class_to_idx)}")

    # 创建模型
    logger.info("创建模型...")
    model = create_face_model(
        model_type=config['model'].get('model_type', 'facenet'),
        num_classes=len(train_dataset.class_to_idx),
        embedding_dim=config['model']['face_embedding_dim'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model'].get('dropout_rate', 0.5)
    )

    # 统计参数
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # 将模型移到设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 确保参数类型正确
    lr = float(config['training']['learning_rate'])
    momentum = float(config['training']['momentum'])
    weight_decay = float(config['training']['weight_decay'])

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config['training']['scheduler_step']),
        gamma=float(config['training']['scheduler_gamma'])
    )

    # TensorBoard写入器
    # TensorBoard写入器（使用与logger相同的目录结构）
    tb_writer = TensorBoardWriter(log_dir=os.path.join('./logs', experiment_name))

    # 恢复训练（如果指定了检查点）
    start_epoch = 0
    best_accuracy = 0.0

    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        start_epoch, _, best_accuracy = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # 训练循环
    logger.info("开始训练...")

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")

        # 训练阶段
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, logger, tb_writer
        )

        # 验证阶段
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, epoch, logger, tb_writer
        )

        # 学习率调度
        scheduler.step()

        # 记录结果
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 打印结果
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1_score']:.4f}")

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc,
                config['paths']['checkpoint_dir']
            )
            logger.info(f"保存最佳模型: {checkpoint_path}")

        # 定期保存检查点
        if (epoch + 1) % config['misc']['save_freq'] == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc,
                config['paths']['checkpoint_dir']
            )
            logger.info(f"保存检查点: {checkpoint_path}")

    # 训练完成
    logger.info("训练完成！")

    # 绘制训练曲线
    experiment_name = config['misc'].get('experiment_name', args.experiment_name)
    plot_path = os.path.join(config['paths']['log_dir'], experiment_name, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
    logger.info(f"训练曲线已保存到: {plot_path}")

    # 关闭TensorBoard写入器
    tb_writer.close()

    # 最终结果
    logger.info("=" * 50)
    logger.info("最终结果:")
    logger.info(".4f")
    logger.info(".4f")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()