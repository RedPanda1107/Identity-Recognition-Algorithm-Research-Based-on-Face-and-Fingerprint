#!/usr/bin/env python
"""
融合模型实验模块
提供独立可控的消融实验和对照实验功能

实验模式：
    1. full             - 训练全部（backbone + fusion）基线
    2. fusion_only      - 冻结backbone，只训练融合层
    3. face_ablation    - 消融：指纹置零，测试单用人脸
    4. fp_ablation      - 消融：人脸置零，测试单用指纹

使用方法：
    # 命令行
    python -m scripts.experiments.fusion_experiments --mode full
    python -m scripts.experiments.fusion_experiments --mode fusion_only --face_ckpt path/to/face.pth --fp_ckpt path/to/fp.pth
    python -m scripts.experiments.fusion_experiments --mode face_ablation
    python -m scripts.experiments.fusion_experiments --mode fp_ablation

    # 代码调用
    from scripts.experiments.fusion_experiments import run_experiment
    results = run_experiment(mode='fusion_only', ...)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, setup_logger
from core.datasets.fusion_dataset import FusionDataset
from core.models import create_model
from core.trainers.fusion_trainer import FusionTrainer


# 实验配置模板
EXPERIMENT_CONFIGS = {
    'full': {
        'description': '训练全部（backbone + fusion）基线',
        'freeze_backbone': False,
        'experiment_mode': 'full',
        'ablate_modality': None,
    },
    'fusion_only': {
        'description': '冻结backbone，只训练融合层（需要预训练单模态权重）',
        'freeze_backbone': True,
        'experiment_mode': 'fusion_only',
        'ablate_modality': None,
    },
    'face_ablation': {
        'description': '消融实验：指纹置零，测试单用人脸',
        'freeze_backbone': False,
        'experiment_mode': 'face_ablation',
        'ablate_modality': 'fingerprint',  # 指纹置零
    },
    'fp_ablation': {
        'description': '消融实验：人脸置零，测试单用指纹',
        'freeze_backbone': False,
        'experiment_mode': 'fingerprint_ablation',
        'ablate_modality': 'face',  # 人脸置零
    },
}


def parse_args():
    """解析命令行参数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fusion_config.yaml")

    parser = argparse.ArgumentParser(
        description="融合模型实验模块 - 支持消融实验和对照实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验模式说明：
  full             训练全部（baseline）
  fusion_only      冻结backbone，只训融合层
  face_ablation   消融：指纹置零，测试单用人脸
  fp_ablation     消融：人脸置零，测试单用指纹

示例：
  python fusion_experiments.py --mode full
  python fusion_experiments.py --mode fusion_only --face_ckpt checkpoints/face_best.pth --fp_ckpt checkpoints/fp_best.pth
  python fusion_experiments.py --mode face_ablation
  python fusion_experiments.py --mode fp_ablation
"""
    )

    # 实验配置
    parser.add_argument("--mode", type=str, default="full",
                       choices=list(EXPERIMENT_CONFIGS.keys()),
                       help="实验模式")
    parser.add_argument("--config", type=str, default=default_config,
                       help="配置文件路径")

    # 实验输出
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="实验名称（默认: {mode}_{timestamp}）")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="实验结果输出目录")

    # 预训练权重（用于fusion_only模式）
    parser.add_argument("--face_ckpt", type=str, default=None,
                       help="人脸模型预训练权重路径")
    parser.add_argument("--fp_ckpt", type=str, default=None,
                       help="指纹模型预训练权重路径")

    # 训练参数
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None,
                       help="训练轮数（覆盖配置）")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批大小（覆盖配置）")
    parser.add_argument("--lr", type=float, default=None,
                       help="学习率（覆盖配置）")

    # 融合配置
    parser.add_argument("--fusion_method", type=str, default="simple",
                       choices=['simple', 'adaptive', 'gated', 'hierarchical'],
                       help="融合方法")
    parser.add_argument("--fusion_dim", type=int, default=None,
                       help="融合维度（覆盖配置）")

    # 其他
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练路径")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    return parser.parse_args()


def _normalize_paths(config, script_dir):
    """统一路径解析"""
    project_root = os.path.dirname(script_dir)
    for key in ['face_data_dir', 'fingerprint_data_dir', 'mapping_file']:
        if key in config.get('paths', {}):
            path = config['paths'][key]
            if path and not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path.lstrip('./'))
    return config


def create_experiment_name(mode: str, timestamp: str = None) -> str:
    """创建实验名称"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fusion_{mode}_{timestamp}"


def get_pretrained_ckpts(args) -> Optional[Dict[str, str]]:
    """获取预训练权重路径"""
    ckpts = {}
    if args.face_ckpt:
        ckpts['face'] = args.face_ckpt
    if args.fp_ckpt:
        ckpts['fingerprint'] = args.fp_ckpt
    return ckpts if ckpts else None


def run_experiment(
    mode: str,
    config_path: str,
    face_ckpt: str = None,
    fp_ckpt: str = None,
    fusion_method: str = "simple",
    fusion_dim: int = 256,
    output_dir: str = "experiments",
    experiment_name: str = None,
    device: str = "auto",
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    resume: str = None,
    seed: int = 42,
) -> Dict:
    """运行单个实验

    Args:
        mode: 实验模式 ('full', 'fusion_only', 'face_ablation', 'fp_ablation')
        config_path: 配置文件路径
        face_ckpt: 人脸预训练权重
        fp_ckpt: 指纹预训练权重
        fusion_method: 融合方法 ('simple', 'adaptive')
        fusion_dim: 融合维度
        output_dir: 输出目录
        experiment_name: 实验名称
        device: 设备
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        resume: 恢复路径
        seed: 随机种子

    Returns:
        实验结果字典
    """
    args_obj = argparse.Namespace(
        mode=mode,
        config=config_path,
        face_ckpt=face_ckpt,
        fp_ckpt=fp_ckpt,
        fusion_method=fusion_method,
        fusion_dim=fusion_dim,
        output_dir=output_dir,
        experiment_name=experiment_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resume=resume,
        seed=seed,
    )
    return _run_single_experiment(args_obj)


def _run_single_experiment(args) -> Dict:
    """执行单个实验的内部函数"""
    config = load_config(args.config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = _normalize_paths(config, script_dir)

    # 实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or create_experiment_name(args.mode, timestamp)

    # 输出目录
    exp_dir = os.path.join(script_dir, args.output_dir, exp_name)
    log_dir = os.path.join(exp_dir, "logs")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 日志
    logger = setup_logger(
        experiment_name=exp_name,
        log_dir=log_dir,
        level="INFO",
        logger_name=f"Exp_{args.mode}"
    )

    # 随机种子
    set_seed(args.seed)
    device = get_device(args.device)

    # 获取实验配置
    exp_config = EXPERIMENT_CONFIGS[args.mode]
    logger.info(f"=" * 60)
    logger.info(f"实验模式: {args.mode}")
    logger.info(f"描述: {exp_config['description']}")
    logger.info(f"=" * 60)

    # 覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.fusion_dim:
        config['model']['fusion_dim'] = args.fusion_dim

    # 数据集
    split_ratio = config["data"].get("split_ratio", 0.8)
    gallery_per_person = config["data"].get("gallery_per_person", 3)

    train_dataset = FusionDataset(
        face_data_dir=config['paths']['face_data_dir'],
        fingerprint_data_dir=config['paths']['fingerprint_data_dir'],
        mapping_file=config['paths'].get('mapping_file'),
        mode='train',
        face_image_size=int(config['data']['face_image_size']),
        fingerprint_image_size=int(config['data']['fingerprint_image_size']),
        augment=config['data'].get('use_augmentation', True),
        split_ratio=split_ratio,
        gallery_per_person=gallery_per_person,
        seed=args.seed
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
        gallery_per_person=gallery_per_person,
        seed=args.seed
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

    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {num_classes}")

    # 创建模型
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

    use_arcface = config['model'].get('use_arcface', True)
    fusion_model = create_model(
        'fusion',
        fusion_method=args.fusion_method,
        face_embedding_dim=int(config['model'].get('face_embedding_dim', 512)),
        fingerprint_embedding_dim=int(config['model'].get('fingerprint_embedding_dim', 512)),
        num_classes=num_classes,
        fusion_dim=int(config['model'].get('fusion_dim', 256)),
        dropout_rate=float(config['model'].get('fusion_dropout_rate', 0.3)),
        use_arcface=use_arcface,
        arc_s=float(config['model'].get('arc_s', 64.0)),
        arc_m=float(config['model'].get('arc_m', 0.5))
    ).to(device)

    # 参数量统计
    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    logger.info(f"Fusion model: total={total_params:,}, trainable={trainable:,}")

    # 优化器
    lr = float(config['training']['learning_rate'])
    wd = float(config['training'].get('weight_decay', 1e-4))

    all_params = list(fusion_model.parameters())
    if not exp_config['freeze_backbone']:
        all_params.extend(list(face_model.parameters()))
        all_params.extend(list(fp_model.parameters()))

    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=wd)
    epochs = int(config['training']['epochs'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config['training'].get('label_smoothing', 0.1)))

    # 获取预训练权重
    pretrained_ckpts = get_pretrained_ckpts(args)

    # 创建训练器
    trainer = FusionTrainer(
        fusion_model=fusion_model,
        face_model=face_model,
        fingerprint_model=fp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        logger=logger,
        pretrained_ckpts=pretrained_ckpts,
        freeze_backbone=exp_config['freeze_backbone'],
        use_amp=config['training'].get('use_amp', True),
        accumulation_steps=int(config['training'].get('accumulation_steps', 1)),
        seed=args.seed,
        experiment_mode=exp_config['experiment_mode'],
        ablate_modality=exp_config['ablate_modality'],
    )

    # 训练循环
    best_rank1 = 0.0
    best_metrics = {}
    no_improve = 0
    patience = int(config.get('misc', {}).get('early_stopping_patience', 15))
    history = {
        'experiment': exp_name,
        'mode': args.mode,
        'description': exp_config['description'],
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': []
    }

    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get('current_epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, rank1, metrics = trainer.validate_epoch(epoch, total_epochs=epochs)
        scheduler.step()

        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_rank1': rank1,
            'val_rank5': metrics.get('rank5', 0),
            'val_rank10': metrics.get('rank10', 0),
            'val_eer': metrics.get('eer', 0),
            'lr': optimizer.param_groups[0]['lr']
        })

        if rank1 > best_rank1:
            best_rank1 = rank1
            best_metrics = metrics.copy()
            best_metrics['epoch'] = epoch + 1
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, "best.pth"),
                extra={'epoch': epoch + 1, 'rank1': rank1}
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # 定期保存
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"),
                extra={'epoch': epoch + 1, 'rank1': rank1}
            )

    # 保存历史
    history_path = os.path.join(log_dir, "history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 保存结果摘要
    summary = {
        'experiment': exp_name,
        'mode': args.mode,
        'description': exp_config['description'],
        'best_rank1': best_rank1,
        'best_rank5': best_metrics.get('rank5', 0),
        'best_rank10': best_metrics.get('rank10', 0),
        'best_rank20': best_metrics.get('rank20', 0),
        'best_eer': best_metrics.get('eer', 0),
        'best_epoch': best_metrics.get('epoch', 0),
        'total_epochs': epoch + 1,
        'start_time': history['start_time'],
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"实验完成: {exp_name}")
    logger.info(f"Best Rank-1: {best_rank1:.4f}")
    logger.info(f"Best EER: {best_metrics.get('eer', 0):.4f}")
    logger.info(f"结果保存: {exp_dir}")
    logger.info("=" * 60)

    return summary


def run_all_experiments(
    config_path: str,
    face_ckpt: str = None,
    fp_ckpt: str = None,
    output_dir: str = "experiments",
    **kwargs
) -> List[Dict]:
    """运行所有实验模式

    Args:
        config_path: 配置文件路径
        face_ckpt: 人脸预训练权重
        fp_ckpt: 指纹预训练权重
        output_dir: 输出目录
        **kwargs: 其他参数传递给 run_experiment

    Returns:
        所有实验结果列表
    """
    results = []
    for mode in EXPERIMENT_CONFIGS.keys():
        print(f"\n{'=' * 60}")
        print(f"开始实验: {mode}")
        print(f"{'=' * 60}\n")

        result = run_experiment(
            mode=mode,
            config_path=config_path,
            face_ckpt=face_ckpt,
            fp_ckpt=fp_ckpt,
            output_dir=output_dir,
            **kwargs
        )
        results.append(result)

    return results


def print_comparison_table(results: List[Dict]):
    """打印实验对比表格"""
    print("\n" + "=" * 80)
    print("实验结果对比")
    print("=" * 80)
    print(f"{'Mode':<20} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'EER':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['mode']:<20} {r['best_rank1']:<10.4f} {r['best_rank5']:<10.4f} "
              f"{r['best_rank10']:<10.4f} {r['best_eer']:<10.4f}")
    print("=" * 80)


def main():
    args = parse_args()

    print(f"""
========================================
融合模型实验模块
========================================
实验模式: {args.mode}
描述: {EXPERIMENT_CONFIGS[args.mode]['description']}
========================================
    """)

    result = _run_single_experiment(args)

    print(f"""
========================================
实验完成
========================================
Best Rank-1: {result['best_rank1']:.4f}
Best EER: {result['best_eer']:.4f}
========================================
    """)


if __name__ == "__main__":
    main()
