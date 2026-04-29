#!/usr/bin/env python
"""
融合模型训练脚本
人脸 + 指纹 多模态特征融合

实验模式：
    --experiment_mode full             训练全部（baseline）
    --experiment_mode fusion_only     冻结backbone，只训练融合层
    --experiment_mode face_ablation   消融：指纹置零，测试单用人脸
    --experiment_mode fp_ablation     消融：人脸置零，测试单用指纹

推荐使用 experiments/fusion_experiments.py 进行完整实验
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, setup_logger
from core.datasets.fusion_dataset import FusionDataset
from core.models import create_model
from core.trainers.fusion_trainer import FusionTrainer

# 实验模式配置
EXPERIMENT_MODES = {
    'full': {'freeze_backbone': False, 'ablate_modality': None},
    'fusion_only': {'freeze_backbone': True, 'ablate_modality': None},
    'face_ablation': {'freeze_backbone': False, 'ablate_modality': 'fingerprint'},
    'fp_ablation': {'freeze_backbone': False, 'ablate_modality': 'face'},
}


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fusion_config.yaml")

    parser = argparse.ArgumentParser(
        description="Train fusion model (face + fingerprint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验模式：
  full             训练全部（baseline）
  fusion_only      冻结backbone，只训练融合层
  face_ablation   消融：指纹置零，测试单用人脸
  fp_ablation     消融：人脸置零，测试单用指纹

推荐使用 experiments/fusion_experiments.py 进行完整实验
"""
    )
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fusion_method", type=str, default="simple",
                       choices=['simple', 'adaptive', 'gated', 'hierarchical'],
                       help="融合方法：simple/simple, adaptive/注意力, gated/门控, hierarchical/层级")
    parser.add_argument("--experiment_mode", type=str, default="full",
                       choices=list(EXPERIMENT_MODES.keys()),
                       help="实验模式")
    parser.add_argument("--face_ckpt", type=str, default=None,
                       help="人脸预训练权重（用于fusion_only模式）")
    parser.add_argument("--fp_ckpt", type=str, default=None,
                       help="指纹预训练权重（用于fusion_only模式）")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    return parser.parse_args()


def _normalize_paths(config, script_dir):
    """统一路径解析逻辑"""
    project_root = os.path.dirname(script_dir)

    # 数据相关路径
    for key in ['face_data_dir', 'fingerprint_data_dir', 'mapping_file']:
        if key in config.get('paths', {}):
            path = config['paths'][key]
            if path and not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path.lstrip('./'))

    # 预训练模型路径
    for key in ['pretrained_face', 'pretrained_fingerprint']:
        if key in config.get('paths', {}):
            path = config['paths'][key]
            if path and not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path.lstrip('./'))

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config = _normalize_paths(config, script_dir)

    # 获取实验模式配置
    exp_config = EXPERIMENT_MODES[args.experiment_mode]
    mode_suffix = f"_{args.experiment_mode}" if args.experiment_mode != 'full' else ""
    experiment_name = args.experiment_name or f"fusion_{args.fusion_method}{mode_suffix}"

    # 目录
    log_dir = os.path.join(script_dir, "logs", experiment_name)
    ckpt_dir = os.path.join(script_dir, "checkpoints", "fusion", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 日志
    logger = setup_logger(experiment_name=experiment_name, log_dir=log_dir,
                         level="INFO", logger_name="FusionTrain")

    # 记录实验模式
    logger.info(f"=" * 60)
    logger.info(f"实验模式: {args.experiment_mode}")
    if args.experiment_mode == 'fusion_only':
        logger.info(f"需要预训练权重: face={args.face_ckpt}, fp={args.fp_ckpt}")
    elif args.experiment_mode == 'face_ablation':
        logger.info(f"消融模式: 指纹置零，测试单用人脸")
    elif args.experiment_mode == 'fp_ablation':
        logger.info(f"消融模式: 人脸置零，测试单用指纹")
    logger.info(f"=" * 60)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir) if config.get('misc', {}).get('use_tensorboard', False) else None

    seed = config.get('misc', {}).get('seed', 42)
    set_seed(seed)
    device = get_device(args.device)

    # 统一的数据集参数
    split_ratio = config["data"].get("split_ratio", 0.8)
    test_split_ratio = config["data"].get("test_split_ratio", 0.5)
    gallery_per_person = config["data"].get("gallery_per_person", 3)

    # 数据集
    train_dataset = FusionDataset(
        face_data_dir=config['paths']['face_data_dir'],
        fingerprint_data_dir=config['paths']['fingerprint_data_dir'],
        mapping_file=config['paths'].get('mapping_file'),
        mode='train',
        face_image_size=int(config['data']['face_image_size']),
        fingerprint_image_size=int(config['data']['fingerprint_image_size']),
        augment=config['data'].get('use_augmentation', True),
        split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        gallery_per_person=gallery_per_person,
        seed=seed
    )
    train_dataset.augmentation_params = config['data'].get('augmentation', {}) or {}

    val_dataset = FusionDataset(
        face_data_dir=config['paths']['face_data_dir'],
        fingerprint_data_dir=config['paths']['fingerprint_data_dir'],
        mapping_file=config['paths'].get('mapping_file'),
        mode='val',
        face_image_size=int(config['data']['face_image_size']),
        fingerprint_image_size=int(config['data']['fingerprint_image_size']),
        augment=False,
        split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        gallery_per_person=gallery_per_person,
        class_to_idx=train_dataset.class_to_idx,
        seed=seed
    )

    # 测试集（用于最终评估，仅在训练完成后使用）
    test_dataset = FusionDataset(
        face_data_dir=config['paths']['face_data_dir'],
        fingerprint_data_dir=config['paths']['fingerprint_data_dir'],
        mapping_file=config['paths'].get('mapping_file'),
        mode='test',
        face_image_size=int(config['data']['face_image_size']),
        fingerprint_image_size=int(config['data']['fingerprint_image_size']),
        augment=False,
        split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        gallery_per_person=gallery_per_person,
        class_to_idx=train_dataset.class_to_idx,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=int(config['training']['batch_size']),
        shuffle=True, num_workers=int(config['misc'].get('num_workers', 4)),
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(config['training']['batch_size']),
        shuffle=False, num_workers=int(config['misc'].get('num_workers', 4)),
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(config['training']['batch_size']),
        shuffle=False, num_workers=int(config['misc'].get('num_workers', 4)),
        pin_memory=True
    )

    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, Classes: {num_classes}")

    # 创建单模态模型
    face_model = create_model(
        'face', num_classes=num_classes,
        embedding_dim=int(config['model'].get('face_embedding_dim', 512)),
        pretrained=True
    ).to(device)

    fp_model = create_model(
        'fingerprint', num_classes=num_classes,
        embedding_dim=int(config['model'].get('fingerprint_embedding_dim', 512)),
        pretrained=True
    ).to(device)

    # 融合模型 (学术标准: 使用 ArcFace s=64, m=0.5)
    use_arcface = config['model'].get('use_arcface', True)
    arc_s = float(config['model'].get('arc_s', 64.0))
    arc_m = float(config['model'].get('arc_m', 0.5))

    fusion_model = create_model(
        'fusion',
        fusion_method=args.fusion_method,
        face_embedding_dim=int(config['model'].get('face_embedding_dim', 512)),
        fingerprint_embedding_dim=int(config['model'].get('fingerprint_embedding_dim', 512)),
        num_classes=num_classes,
        fusion_dim=int(config['model'].get('fusion_dim', 256)),
        dropout_rate=float(config['model'].get('fusion_dropout_rate', 0.3)),
        use_arcface=use_arcface,
        arc_s=arc_s,
        arc_m=arc_m
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    logger.info(f"Fusion model: total={total_params:,}, trainable={trainable:,}")

    # 优化器 - 根据实验模式配置
    lr = float(config['training']['learning_rate'])
    wd = float(config['training'].get('weight_decay', 1e-4))

    all_params = list(fusion_model.parameters())
    if not exp_config['freeze_backbone']:
        all_params.extend(list(face_model.parameters()))
        all_params.extend(list(fp_model.parameters()))
        logger.info("Training: fusion + face + fingerprint backbones")
    else:
        logger.info("Training: fusion only (backbones frozen)")

    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=wd)

    epochs = int(config['training']['epochs'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config['training'].get('label_smoothing', 0.1)))

    logger.info(f"Optimizer: AdamW, lr={lr}, wd={wd}, epochs={epochs}")

    # 预训练权重
    pretrained_ckpts = None
    if args.face_ckpt or args.fp_ckpt:
        pretrained_ckpts = {}
        if args.face_ckpt:
            pretrained_ckpts['face'] = args.face_ckpt
        if args.fp_ckpt:
            pretrained_ckpts['fingerprint'] = args.fp_ckpt

    # 训练器
    trainer = FusionTrainer(
        fusion_model=fusion_model,
        face_model=face_model,
        fingerprint_model=fp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        logger=logger,
        pretrained_ckpts=pretrained_ckpts,
        freeze_backbone=exp_config['freeze_backbone'],
        use_amp=config['training'].get('use_amp', True),
        accumulation_steps=int(config['training'].get('accumulation_steps', 1)),
        seed=seed,
        experiment_mode=args.experiment_mode,
        ablate_modality=exp_config['ablate_modality'],
        label_smoothing=float(config['training'].get('label_smoothing', 0.1)),
        tb_writer=writer,
    )

    # 训练循环
    best_rank1 = 0.0
    no_improve = 0
    patience = int(config.get('misc', {}).get('early_stopping_patience', 15))

    history = {
        'experiment': experiment_name,
        'experiment_mode': args.experiment_mode,
        'fusion_method': args.fusion_method,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': []
    }

    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get('current_epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = trainer.train_epoch(epoch, total_epochs=epochs)
        val_loss, rank1, val_metrics = trainer.validate_epoch(epoch, total_epochs=epochs)
        scheduler.step()

        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_rank1': rank1,
            'val_rank5': val_metrics.get('rank_5', 0),
            'val_rank10': val_metrics.get('rank_10', 0),
            'val_rank20': val_metrics.get('rank_20', 0),
            'val_eer': val_metrics.get('eer', 0),
            'lr': optimizer.param_groups[0]['lr']
        })

        if rank1 > best_rank1:
            best_rank1 = rank1
            trainer.save_checkpoint(os.path.join(ckpt_dir, f"best_{args.fusion_method}.pth"),
                                  is_best=True,
                                  extra={'epoch': epoch+1, 'rank1': rank1})
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # 保存历史
    history_path = os.path.join(log_dir, "history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete! Best val Rank-1: {best_rank1:.4f}")

    # ── 测试集评估（仅在训练完成后使用一次）──────────────────────────────
    if trainer.test_loader is not None:
        logger.info("=" * 60)
        logger.info("在测试集上进行最终评估...")
        logger.info("=" * 60)
        test_metrics = trainer.test_epoch(epoch=-1, total_epochs=epochs, use_amp=config['training'].get('use_amp', True))
        if test_metrics and test_metrics.get('rank_1') is not None:
            logger.info("[测试集] " + " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in [
                    ("Rank-1", test_metrics.get('rank_1', 0)),
                    ("Rank-5", test_metrics.get('rank_5', 0)),
                    ("Rank-10", test_metrics.get('rank_10', 0)),
                    ("Rank-20", test_metrics.get('rank_20', 0)),
                    ("EER", test_metrics.get('eer', 0)),
                ]
            ))
        else:
            logger.info("[测试集] 未检测到测试数据（test_split_ratio=1.0 或测试人员不足）")


if __name__ == "__main__":
    main()
