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
from pathlib import Path

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

def _find_best_checkpoint(ckpt_dir, logger):
    """从 checkpoint 目录中找出 val_acc 最高的文件路径

    优先查找 best.pth（新版单文件），兼容旧版 best_epoch_*.pth。
    """
    if not os.path.isdir(ckpt_dir):
        logger.warning(f"[Checkpoint] 目录不存在: {ckpt_dir}")
        return None

    # 新命名：best.pth（始终只保存一个最优）
    best_pth = os.path.join(ckpt_dir, "best.pth")
    if os.path.isfile(best_pth):
        try:
            ckpt = torch.load(best_pth, map_location='cpu', weights_only=False)
            val_acc = float(ckpt.get('val_acc', ckpt.get('rank1', -1.0)))
            logger.info(f"[Checkpoint] 加载最优权重: {best_pth} (val_acc={val_acc:.4f})")
            return best_pth
        except Exception:
            pass

    # 旧命名兼容：best_epoch_*.pth（扫描找最高）
    files = [f for f in os.listdir(ckpt_dir) if f.startswith('best_epoch_') and f.endswith('.pth')]
    if not files:
        logger.warning(f"[Checkpoint] 目录为空，找不到 best.pth: {ckpt_dir}")
        return None

    best_file = None
    best_acc = -1.0
    for f in files:
        try:
            ckpt = torch.load(os.path.join(ckpt_dir, f), map_location='cpu', weights_only=False)
            val_acc = float(ckpt.get('val_acc', ckpt.get('rank1', -1.0)))
            if val_acc > best_acc:
                best_acc = val_acc
                best_file = os.path.join(ckpt_dir, f)
        except Exception:
            pass

    if best_file:
        logger.info(f"[Checkpoint] 自动选择最优权重: {best_file} (val_acc={best_acc:.4f})")
    return best_file


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
                       choices=['simple', 'adaptive'],
                       help="融合方法：simple/加权融合, adaptive/注意力自适应融合")
    parser.add_argument("--experiment_mode", type=str, default="full",
                       choices=list(EXPERIMENT_MODES.keys()),
                       help="实验模式")
    parser.add_argument("--face_ckpt", type=str, default=None,
                       help="人脸预训练权重（用于fusion_only模式）")
    parser.add_argument("--fp_ckpt", type=str, default=None,
                       help="指纹预训练权重（用于fusion_only模式）")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--face_dropout_prob", type=float, default=None,
                       help="训练时以该概率丢弃人脸特征（0.0=禁用, 0.15=15%%丢弃）")
    parser.add_argument("--fp_corruption_prob", type=float, default=None,
                       help="训练时以该概率腐蚀指纹特征（0.0=禁用, 0.15=15%%腐蚀）")
    parser.add_argument("--modality_drop_strategy", type=str, default=None,
                       choices=['clean', 'face_dropout', 'fp_corruption', 'both'],
                       help="模态腐败策略")
    parser.add_argument("--entropy_penalty_weight", type=float, default=0.0,
                       help="Attention 熵正则系数（0.0=禁用，正值=迫使权重均衡，如0.05）")
    parser.add_argument("--freeze_projection", action="store_true",
                       help="冻结投影层（保持恒等映射）")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="冻结 backbone 和投影层，只训练 fusion_layer + classifier")
    return parser.parse_args()


def _normalize_paths(config, script_dir):
    """统一路径解析逻辑（使用 Path 确保跨平台路径一致性）"""
    project_root = Path(script_dir).parent.resolve()

    for key in ['face_data_dir', 'fingerprint_data_dir', 'mapping_file',
                'pretrained_face', 'pretrained_fingerprint',
                'pretrained_face_dir', 'pretrained_fingerprint_dir',
                'checkpoint_dir', 'log_dir', 'results_dir', 'visualization_dir']:
        raw = config.get('paths', {}).get(key, '')
        if raw:
            p = Path(raw)
            config['paths'][key] = str(p if p.is_absolute() else project_root / p)

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config = _normalize_paths(config, script_dir)

    # 获取实验模式配置
    mode_suffix = f"_{args.experiment_mode}" if args.experiment_mode != 'full' else ""
    experiment_name = args.experiment_name or f"fusion_{args.fusion_method}{mode_suffix}"

    # 目录（已由 _normalize_paths 转为绝对 Path，统一用 Path 处理）
    ckpt_base = config['paths'].get('checkpoint_dir') or str(Path(script_dir).parent / "outputs" / "fusion")
    log_base = config['paths'].get('log_dir') or str(Path(script_dir).parent / "outputs" / "logs")
    ckpt_dir = str(Path(ckpt_base) / experiment_name)
    log_dir = str(Path(log_base) / experiment_name)

    # backbone 预训练权重搜索路径
    face_ckpt_search_dir = config['paths'].get('pretrained_face_dir') or str(Path(script_dir).parent / "checkpoints" / "face")
    fp_ckpt_search_dir = config['paths'].get('pretrained_fingerprint_dir') or str(Path(script_dir).parent / "checkpoints" / "fingerprint")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 日志
    logger = setup_logger(experiment_name=experiment_name, log_dir=log_dir,
                         level="INFO", logger_name="FusionTrain")

    # 记录实验模式
    logger.info(f"=" * 60)
    logger.info(f"实验模式: {args.experiment_mode}")
    if args.experiment_mode == 'fusion_only':
        logger.info(f"冻结backbone，从指定路径加载预训练权重")
    elif args.experiment_mode == 'full':
        logger.info(f"训练全部，自动加载单模态 best checkpoint（backbone 可继续微调）")
    elif args.experiment_mode == 'face_ablation':
        logger.info(f"消融模式：指纹置零，测试单用人脸")
    elif args.experiment_mode == 'fp_ablation':
        logger.info(f"消融模式：人脸置零，测试单用指纹")
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
    use_clahe = config['data'].get('use_clahe', True)

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
        seed=seed,
        use_clahe=use_clahe,
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
        seed=seed,
        use_clahe=use_clahe,
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
        seed=seed,
        use_clahe=use_clahe,
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

    # 创建单模态模型（spatial_attention=False 以匹配预训练 checkpoint）
    face_model = create_model(
        'face', num_classes=num_classes,
        embedding_dim=int(config['model'].get('face_embedding_dim', 512)),
        pretrained=True
    ).to(device)

    fp_model = create_model(
        'fingerprint', num_classes=num_classes,
        embedding_dim=int(config['model'].get('fingerprint_embedding_dim', 512)),
        pretrained=True,
        spatial_attention=False   # 必须与 FP checkpoint 一致
    ).to(device)

    # 融合模型（使用与单模态一致的 ArcFace 参数：s=30, m=0.35）
    use_arcface = config['model'].get('use_arcface', True)
    arc_s = float(config['model'].get('arc_s', 30.0))
    arc_m = float(config['model'].get('arc_m', 0.35))

    fusion_model = create_model(
        'fusion',
        fusion_strategy=args.fusion_method,
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

    # freeze_backbone：命令行参数优先于配置文件
    ablate_from_mode = EXPERIMENT_MODES[args.experiment_mode]['ablate_modality']
    freeze_backbone = args.freeze_backbone or config['training'].get('freeze_backbone', False)
    logger.info(f"[Config] freeze_backbone={freeze_backbone} (cli={args.freeze_backbone}, config={config['training'].get('freeze_backbone', False)})")
    logger.info(f"[Config] Experiment mode: {args.experiment_mode}, ablate_modality={ablate_from_mode}")

    # 优化器 - 差分学习率：backbone 低 LR，fusion head 高 LR
    backbone_lr = float(config['training'].get('backbone_lr', 1e-5))
    # 一阶段 fusion 学习率：优先读 fusion_lr，兼容旧的 stage1_fusion_lr
    fusion_lr = float(config['training'].get('fusion_lr',
                          config['training'].get('stage1_fusion_lr', 5e-4)))
    wd = float(config['training'].get('weight_decay', 5e-4))

    param_groups = []

    # Fusion head: 高 LR
    fusion_params = [p for p in fusion_model.parameters() if p.requires_grad]
    param_groups.append({'params': fusion_params, 'lr': fusion_lr, '_base_lr': fusion_lr})

    if not freeze_backbone:
        # Backbone: 低 LR（冻结时不加入优化器）
        backbone_params = []
        if face_model:
            backbone_params.extend([p for p in face_model.parameters() if p.requires_grad])
        if fp_model:
            backbone_params.extend([p for p in fp_model.parameters() if p.requires_grad])
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr, '_base_lr': backbone_lr})

        logger.info(
            f"Differential LR: backbone={backbone_lr:.0e}, fusion={fusion_lr:.0e}, "
            f"ratio={fusion_lr/backbone_lr:.0f}x"
        )
    else:
        logger.info(f"Training fusion head only (backbones frozen), LR={fusion_lr:.0e}")

    optimizer = optim.AdamW(param_groups, lr=fusion_lr, weight_decay=wd)

    epochs = int(config['training']['epochs'])
    warmup_epochs = int(config['training'].get('warmup_epochs', 0))
    warmup_start_lr = float(config['training'].get('warmup_start_lr', 1e-6))
    initial_lr = fusion_lr  # warmup 目标 LR

    # 统一使用 StepLR
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config['training'].get('scheduler_step', 15)),
        gamma=float(config['training'].get('scheduler_gamma', 0.1))
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config['training'].get('label_smoothing', 0.0)))

    logger.info(f"Optimizer: AdamW, backbone_lr={backbone_lr}, fusion_lr={fusion_lr}, wd={wd}, epochs={epochs}")
    drop_strategy = args.modality_drop_strategy if args.modality_drop_strategy else config['training'].get('modality_drop_strategy', 'both')
    face_dp = args.face_dropout_prob if args.face_dropout_prob is not None else float(config['training'].get('face_dropout_prob', 0.0))
    fp_cp = args.fp_corruption_prob if args.fp_corruption_prob is not None else float(config['training'].get('fp_corruption_prob', 0.0))
    if face_dp > 0 or fp_cp > 0:
        logger.info(f"[ModalityDrop] strategy={drop_strategy}, face_dropout={face_dp}, fp_corruption={fp_cp}")
    if args.entropy_penalty_weight > 0:
        logger.info(f"[EntropyPenalty] weight={args.entropy_penalty_weight}")
    if args.freeze_projection:
        logger.info(f"[Config] freeze_projection=True (projections frozen, identity preserved)")

    # 预训练权重
    # 消融实验（face_ablation / fp_ablation）：只加载对应单模态的预训练权重
    # full / fusion_only：加载 face + fp 两个单模态权重
    pretrained_ckpts = None
    if args.face_ckpt or args.fp_ckpt:
        pretrained_ckpts = {}
        if args.face_ckpt:
            pretrained_ckpts['face'] = args.face_ckpt
        if args.fp_ckpt:
            pretrained_ckpts['fingerprint'] = args.fp_ckpt
    elif args.experiment_mode in ('fusion_only', 'full'):
        pretrained_ckpts = {}
        if face_ckpt_search_dir:
            best_face = _find_best_checkpoint(face_ckpt_search_dir, logger)
            if best_face:
                pretrained_ckpts['face'] = best_face
        if fp_ckpt_search_dir:
            best_fp = _find_best_checkpoint(fp_ckpt_search_dir, logger)
            if best_fp:
                pretrained_ckpts['fingerprint'] = best_fp
        if not pretrained_ckpts:
            logger.warning("[Fusion] 未找到单模态预训练权重，将从头训练 backbone。"
                           "建议先运行 train_face.py 和 train_fingerprint.py 训练单模态模型。"
                           "或通过 --face_ckpt / --fp_ckpt 手动指定。")
    elif args.experiment_mode == 'face_ablation':
        # 消融人脸：只加载 face checkpoint 用于 face-only 实验
        pretrained_ckpts = {}
        if face_ckpt_search_dir:
            best_face = _find_best_checkpoint(face_ckpt_search_dir, logger)
            if best_face:
                pretrained_ckpts['face'] = best_face
                logger.info("[Face Ablation] 已加载 face checkpoint，将从 face 权重 fine-tune")
        if not pretrained_ckpts:
            logger.warning("[Face Ablation] 未找到 face checkpoint，将从头训练（不建议）")
    elif args.experiment_mode == 'fp_ablation':
        # 消融指纹：只加载 fingerprint checkpoint 用于 fp-only 实验
        pretrained_ckpts = {}
        if fp_ckpt_search_dir:
            best_fp = _find_best_checkpoint(fp_ckpt_search_dir, logger)
            if best_fp:
                pretrained_ckpts['fingerprint'] = best_fp
                logger.info("[FP Ablation] 已加载 fingerprint checkpoint，将从 FP 权重 fine-tune")
        if not pretrained_ckpts:
            logger.warning("[FP Ablation] 未找到 fingerprint checkpoint，将从头训练（不建议）")

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
        freeze_backbone=freeze_backbone,
        use_amp=config['training'].get('use_amp', True),
        accumulation_steps=int(config['training'].get('accumulation_steps', 1)),
        seed=seed,
        experiment_mode=args.experiment_mode,
        ablate_modality=ablate_from_mode,
        label_smoothing=float(config['training'].get('label_smoothing', 0.0)),
        tb_writer=writer,
        face_dropout_prob=face_dp,
        fp_corruption_prob=fp_cp,
        modality_drop_strategy=drop_strategy,
        freeze_projection=args.freeze_projection,
        entropy_penalty_weight=args.entropy_penalty_weight,
        balance_lr=float(config['training'].get('balance_lr', 0.1)),
        balance_weight_decay=float(config['training'].get('balance_weight_decay', 2.0)),
    )

    # 训练历史记录
    patience = int(config.get('misc', {}).get('early_stopping_patience', 15))

    # Classifier Warm-Start: ablation experiments load classifier from corresponding modality
    face_pretrained = pretrained_ckpts.get('face') if pretrained_ckpts else None
    fp_pretrained  = pretrained_ckpts.get('fingerprint') if pretrained_ckpts else None

    if args.experiment_mode in ('face_ablation', 'fp_ablation'):
        ckpt_for_clf = face_pretrained if args.experiment_mode == 'face_ablation' else fp_pretrained
        if ckpt_for_clf:
            fusion_model.init_classifier_from_pretrained(ckpt_for_clf, None, device)
            logger.info("[ClassifierWarmStart] Loaded classifier from " + os.path.basename(os.path.dirname(ckpt_for_clf)))

    history = {
        'experiment': experiment_name,
        'experiment_mode': args.experiment_mode,
        'fusion_method': args.fusion_method,
        'face_dropout_prob': args.face_dropout_prob,
        'entropy_penalty_weight': args.entropy_penalty_weight,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': []
    }

    # 单阶段训练循环
    best_rank1 = 0.0
    no_improve = 0

    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # Warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            wf = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['_base_lr'] * wf
            logger.info(
                f"[Train] Warmup {epoch+1}/{warmup_epochs}, "
                f"fusion_lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        train_loss, train_acc = trainer.train_epoch(epoch, total_epochs=epochs)
        val_loss, rank1, val_metrics = trainer.validate_epoch(epoch, total_epochs=epochs)
        scheduler.step()
        trainer.step_balance_scheduler()

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
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, "best.pth"),
                is_best=True,
                extra={'epoch': epoch + 1, 'rank1': rank1})
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

    # ── 自动生成可视化图表 ──────────────────────────────────────────────
    vis_output_dir = config['paths'].get('results_dir') or config['paths'].get('visualization_dir')
    if vis_output_dir:
        try:
            import subprocess
            vis_script = os.path.join(project_root, "scripts", "visualize.py")
            if os.path.exists(vis_script):
                logger.info(f"[Visualization] Generating charts in {vis_output_dir} ...")
                os.makedirs(vis_output_dir, exist_ok=True)
                result = subprocess.run(
                    [sys.executable, vis_script,
                     "--logs_dir", log_dir,
                     "--output_dir", vis_output_dir],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    logger.info("[Visualization] Charts generated successfully.")
                else:
                    logger.warning(f"[Visualization] Chart generation failed: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"[Visualization] Skipped: {e}")

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
