#!/usr/bin/env python
"""
Fingerprint recognition training script.

Two-stage training strategy:
    Stage 1 (freeze_epochs): Freeze backbone, train only classifier + projection head
        → Rapid baseline verification: data pipeline, loss stability, initial convergence
    Stage 2 (warmup_epochs + beyond): Unfreeze backbone, fine-tune with conservative LR
        → Domain adaptation: ResNet50 adapts to fingerprint ridge patterns

Key design decisions:
    - NaN-safe: Skip NaN batches, log warnings, continue training
    - AMP: Mixed precision for memory efficiency
    - 1:N retrieval validation: Cosine similarity (not classification loss)
    - Gradient clipping: max_norm=1.0 to prevent fp16 overflow
    - Metric learning (ArcFace) is OFF by default; enable only after Stage 1 baseline works
"""

import os
import sys
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import subprocess
import json

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, count_parameters, setup_logger
from core.datasets.fingerprint_dataset import FingerprintDataset
from core.models import create_model
from core.trainers.fingerprint_trainer import FingerprintTrainer


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(script_dir)
    default_config = os.path.join(proj_root, "configs", "fingerprint_config.yaml")

    parser = argparse.ArgumentParser(description="Train fingerprint recognition model")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="fingerprint_recognition")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _normalize_paths(config, script_dir):
    """Normalize all paths to be absolute relative to project root."""
    from pathlib import Path
    proj_root = os.path.dirname(script_dir)  # scripts/ 的 parent 就是项目根 src/

    for key in list(config.get("paths", {}).keys()):
        raw = config["paths"][key]
        if raw:
            p = Path(raw)
            config["paths"][key] = str(p if p.is_absolute() else Path(proj_root) / p)


def _log_model_info(logger, model, train_dataset, val_dataset, num_classes):
    """Log model architecture and dataset statistics."""
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Device: {next(model.parameters()).device}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"  Total parameters:    {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} "
                f"({trainable_params / total_params * 100:.1f}%)")
    logger.info(f"  Embedding dim:      {model.get_embedding_dim()}")
    logger.info(f"  Number of classes:  {num_classes}")
    logger.info(f"  Training samples:   {len(train_dataset)} "
                f"({len(set(train_dataset.labels))} persons)")
    logger.info(f"  Validation samples: {len(val_dataset)} "
                f"({len(set(val_dataset.labels))} persons)")


def _log_train_config(logger, config, freeze_epochs, warmup_epochs, warmup_start_lr,
                      freeze_lr, unfreeze_lr,
                      metric_learning, arcface_s, arc_m_start, arc_m_end,
                      arc_m_delay_epochs, arc_m_warmup_epochs,
                      label_smoothing=0.0, tta=False):
    """Log training configuration summary."""
    logger.info("=" * 60)
    logger.info("训练配置摘要")
    logger.info("=" * 60)
    logger.info(f"  总 Epochs:          {config['training']['epochs']}")
    logger.info(f"  Batch size:         {config['training']['batch_size']}")
    logger.info(f"  阶段 1 冻结轮次:    {freeze_epochs} epochs")
    logger.info(f"  Stage 1 LR (head):  {freeze_lr:.0e}")
    logger.info(f"  阶段 2 Warmup:      {warmup_epochs} epochs")
    logger.info(f"  Warmup start LR:    {warmup_start_lr:.0e}")
    logger.info(f"  阶段 2 Unfreeze LR: {unfreeze_lr:.0e}")
    logger.info(f"  Metric learning:    {metric_learning} "
                f"(ArcFace s={arcface_s}, m∈[{arc_m_start}→{arc_m_end}])")
    if metric_learning:
        logger.info(f"  ArcFace delay:      {arc_m_delay_epochs} epochs")
        logger.info(f"  ArcFace warmup:     {arc_m_warmup_epochs} epochs")
    logger.info(f"  Label smoothing:     {label_smoothing} (减少过拟合)")
    logger.info(f"  TTA (val):           {tta} (水平翻转增强)")
    logger.info(f"  Early stopping:      patience={config['misc']['early_stopping_patience']}, "
                f"warmup={config['misc']['early_stopping_warmup']}")
    logger.info("=" * 60)


def _apply_freeze_config(logger, model, optimizer, config, stage_name, freeze_epochs,
                         freeze_learning_rate, unfreeze_lr, head_lr_ratio):
    """Apply freeze/unfreeze configuration to model and optimizer.

    Stage 1 (freeze_epochs > 0):
        - Freeze backbone (layer1-layer4, conv1, bn1)
        - Train only classifier + feature_projection
        - Use larger LR for classifier (freeze_learning_rate)
    Stage 2 (after unfreeze):
        - Unfreeze backbone
        - Conservative LR for backbone, moderate LR for head
        - Use separate param groups for fine-grained LR control
    """
    wd = float(config['training'].get('weight_decay', 5e-4))
    if freeze_epochs > 0:
        # Stage 1: Freeze backbone
        model.freeze_until(model.FREEZE_L4)  # Freeze everything except classifier
        info = model.get_trainable_params_info()
        logger.info(
            f"[{stage_name}] backbone 已冻结，"
            f"可训练参数: {info['trainable_pct']:.1f}% ({info['trainable']:,})"
        )

        # Stage 1 optimizer: only trainable params (classifier + projection)
        opt_name = config['training'].get('optimizer', 'adamw').lower()
        if 'adam' in opt_name:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=freeze_learning_rate,
                weight_decay=wd
            )
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=freeze_learning_rate,
                weight_decay=wd
            )
        logger.info(f"[{stage_name}] optimizer: LR={freeze_learning_rate:.0e}, "
                    f"仅训练可学习参数")
    else:
        # Stage 2: Unfreeze backbone with conservative LR
        model.freeze_until(model.FREEZE_ALL)  # Everything trainable
        info = model.get_trainable_params_info()
        logger.info(
            f"[{stage_name}] backbone 已解冻，"
            f"可训练参数: {info['trainable_pct']:.1f}% ({info['trainable']:,})"
        )

        # Stage 2 optimizer: separate LR for backbone and head
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'feature_projection' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        opt_name = config['training'].get('optimizer', 'adamw').lower()
        if 'adam' in opt_name:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': unfreeze_lr},
                {'params': head_params, 'lr': unfreeze_lr * head_lr_ratio},
            ], weight_decay=wd)
        else:
            optimizer = optim.SGD([
                {'params': backbone_params, 'lr': unfreeze_lr},
                {'params': head_params, 'lr': unfreeze_lr * head_lr_ratio},
            ], weight_decay=wd, momentum=0.9)
        logger.info(
            f"[{stage_name}] 分层学习率: backbone={unfreeze_lr:.0e}, "
            f"head={unfreeze_lr * head_lr_ratio:.0e}"
        )

    return optimizer


def _apply_lr_warmup(optimizer, epoch, warmup_epochs, warmup_start_lr, initial_lr):
    """Apply linear LR warmup for the current epoch."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return

    warmup_factor = (epoch + 1) / warmup_epochs
    current_lr = warmup_start_lr + (initial_lr - warmup_start_lr) * warmup_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def _apply_arcface_margin_schedule(trainer, epoch, metric_learning, arc_m_start,
                                    arc_m_end, arc_m_delay_epochs, arc_m_warmup_epochs):
    """Apply ArcFace margin progressive schedule.

    arc_m_delay_epochs 内 m=0（让 classifier 先稳定）。
    之后在 arc_m_warmup_epochs 内从 arc_m_start 线性爬升到 arc_m_end。
    """
    if not metric_learning:
        return

    epoch_since_delay = epoch - arc_m_delay_epochs
    if epoch_since_delay < 0:
        current_m = arc_m_start  # 延迟期内，margin=0
    elif arc_m_warmup_epochs > 0:
        warmup_progress = min(epoch_since_delay / max(arc_m_warmup_epochs - 1, 1), 1.0)
        current_m = arc_m_start + (arc_m_end - arc_m_start) * warmup_progress
    else:
        current_m = arc_m_end

    current_m = max(arc_m_start, min(arc_m_end, current_m))
    trainer.update_arcface_margin(current_m)


def _is_training_stable(train_loss, train_acc):
    """Check if training is producing reasonable losses and accuracies."""
    if train_loss is None or train_acc is None:
        return False
    if train_loss != train_loss:  # NaN check
        return False
    if train_loss > 1e6 or train_loss < 0:
        return False
    return True


def main():
    args = parse_args()
    config = load_config(args.config)

    # 统一路径 + 输出目录结构
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = config.get("misc", {}).get("experiment_name", args.experiment_name)
    _normalize_paths(config, script_dir)

    ckpt_base = config["paths"]["checkpoint_dir"]
    log_base = config["paths"]["log_dir"]
    fig_base = config["paths"]["results_dir"]
    ckpt_dir = os.path.join(ckpt_base, experiment_name)
    fig_dir = fig_base  # figures go directly under results_dir (no experiment_name nesting)

    logger = setup_logger(
        experiment_name=experiment_name,
        log_dir=log_base,  # setup_logger will append experiment_name internally
        level="INFO",
        logger_name="FingerprintRecognition"
    )
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"配置文件: {args.config}")

    set_seed(config.get("misc", {}).get("seed", 42))
    device = get_device(args.device)

    # ── 数据集 ─────────────────────────────────────────────────────────────────
    data_dir = config["paths"]["modality_data_dir"]
    max_persons = config["data"].get("max_persons", None)
    use_clahe = config["data"].get("use_clahe", True)
    test_split_ratio = config["data"].get("test_split_ratio", 0.5)

    train_dataset = FingerprintDataset(
        data_dir,
        mode="train",
        image_size=config["data"]["image_size"],
        augment=config["data"].get("use_augmentation", True),
        max_persons=max_persons,
        use_clahe=use_clahe
    )
    train_dataset.augmentation_params = config["data"].get("augmentation", {}) or {}

    val_dataset = FingerprintDataset(
        data_dir,
        mode="val",
        image_size=config["data"]["image_size"],
        augment=False,
        max_persons=max_persons,
        class_to_idx=train_dataset.class_to_idx,
        use_clahe=use_clahe,
        test_split_ratio=test_split_ratio
    )

    test_dataset = FingerprintDataset(
        data_dir,
        mode="test",
        image_size=config["data"]["image_size"],
        augment=False,
        max_persons=max_persons,
        class_to_idx=train_dataset.class_to_idx,
        use_clahe=use_clahe,
        test_split_ratio=test_split_ratio
    )

    # Check if test set is available
    has_test_attr = hasattr(test_dataset, 'test_gallery_paths')
    has_test_paths = bool(test_dataset.test_gallery_paths) if has_test_attr else False
    has_test = (test_split_ratio < 1.0 and has_test_attr and has_test_paths)
    logger.info(f"[DEBUG] has_test 检查: test_split_ratio={test_split_ratio}<1.0={test_split_ratio < 1.0}, "
                f"has_attr={has_test_attr}, test_gallery_paths非空={has_test_paths} → has_test={has_test}")
    if has_test:
        logger.info(
            f"数据集划分: 训练 {len(train_dataset)} 张 / "
            f"验证 {len(val_dataset)} 张 / "
            f"测试 {len(test_dataset)} 张"
        )
    else:
        logger.info(
            f"数据集划分: 训练 {len(train_dataset)} 张 / "
            f"验证 {len(val_dataset)} 张（无独立测试集）"
        )

    # ── DataLoader ──────────────────────────────────────────────────────────────
    # 强制 num_workers=0 排除 Windows 多进程阻塞问题
    num_workers = 0
    persistent = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=False
    )

    # ── 模型 ───────────────────────────────────────────────────────────────────
    num_classes = len(train_dataset.class_to_idx)
    model = create_model(
        "fingerprint",
        model_type=config["model"].get("model_type", "fingerprint_net"),
        num_classes=num_classes,
        embedding_dim=config["model"].get("embedding_dim", 512),
        pretrained=config["model"].get("pretrained", False),
        dropout_rate=config["model"].get("dropout_rate", 0.5),
        spatial_attention=config["model"].get("spatial_attention", False)
    )
    model = model.to(device)

    _log_model_info(logger, model, train_dataset, val_dataset, num_classes)

    # ── 损失函数 ───────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Loss: CrossEntropyLoss")

    # ── 训练配置参数 ───────────────────────────────────────────────────────────
    freeze_epochs = int(config["training"].get("freeze_epochs", 15))
    freeze_lr = float(config["training"].get("freeze_learning_rate", "1e-3"))
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    warmup_start_lr = float(config["training"].get("warmup_start_lr", "1e-6"))
    unfreeze_lr = float(config["training"].get("unfreeze_lr", "5e-5"))
    head_lr_ratio = float(config["training"].get("head_lr_ratio", "1.0"))
    initial_lr = unfreeze_lr  # warmup target = unfreeze_lr

    # Academic standard: s=64, m=0.5
    arcface_s = float(config["training"].get("arc_s", 64.0))
    arc_m_start = float(config["training"].get("arc_m_start", 0.0))
    arc_m_end = float(config["training"].get("arc_m_end", 0.5))
    metric_learning = config["training"].get("metric_learning", True)
    arc_m_warmup_epochs = int(config["training"].get("arc_m_warmup_epochs", 5))
    arc_m_delay_epochs = int(config["training"].get("arc_m_delay_epochs", 3))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    tta = config["training"].get("tta", False)
    use_amp = config["training"].get("use_amp", False)

    scheduler_type = config["training"].get("scheduler_type", "step")

    _log_train_config(
        logger, config, freeze_epochs, warmup_epochs, warmup_start_lr,
        freeze_lr, unfreeze_lr,
        metric_learning, arcface_s, arc_m_start, arc_m_end,
        arc_m_delay_epochs, arc_m_warmup_epochs, label_smoothing, tta
    )

    # ── 学习率调度器（延迟初始化，optimizer 创建后再绑定）───────────────
    scheduler = None

    # ── Trainer ─────────────────────────────────────────────────────────────────
    seed = config.get("misc", {}).get("seed", 42)

    trainer = FingerprintTrainer(
        model, train_loader, val_loader, optimizer=None, scheduler=scheduler,
        criterion=criterion, device=device, logger=logger, tb_writer=None,
        arcface_s=arcface_s,
        arcface_m=arc_m_start,
        metric_learning=metric_learning,
        label_smoothing=label_smoothing,
        tta=tta,
        seed=seed,
        use_amp=use_amp,
        test_dataset=test_dataset if has_test else None
    )

    # ── 训练历史 ───────────────────────────────────────────────────────────────
    training_history = {
        "experiment_name": experiment_name,
        "model_type": "fingerprint",
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": {k: v for k, v in config.items() if k != 'paths'},
        "stages": [],
        "epochs": []
    }

    # ── 训练循环 ───────────────────────────────────────────────────────────────
    epochs = int(config["training"]["epochs"])
    early_stopping = config.get("misc", {}).get("early_stopping", False)
    early_stopping_patience = int(config["misc"].get("early_stopping_patience", 20))
    early_stopping_warmup = int(config["misc"].get("early_stopping_warmup", 15))

    start_epoch = 0
    best_acc = 0.0
    no_improve_epochs = 0
    optimizer = None  # 延迟初始化（在阶段切换时创建）
    current_stage = 0

    # 判断当前是否处于冻结阶段
    is_frozen = freeze_epochs > 0

    logger.info(f"开始训练，共 {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        stage_name = f"Epoch {epoch+1}/{epochs}"

        # ── 阶段切换：冻结 → 解冻 ─────────────────────────────────────────────
        if freeze_epochs > 0 and epoch == freeze_epochs and optimizer is not None:
            # 从 Stage 1 切换到 Stage 2
            logger.info("=" * 60)
            logger.info("阶段切换：冻结 backbone → 解冻 backbone")
            logger.info("=" * 60)
            is_frozen = False

        # ── 当前阶段的 optimizer 和学习率设置 ─────────────────────────────────
        if is_frozen:
            # Stage 1: 冻结 backbone
            stage_name = f"Epoch {epoch+1}/{epochs} [Stage1-Freeze]"
            if optimizer is None or current_stage != 1:
                optimizer = _apply_freeze_config(
                    logger, model, optimizer, config,
                    stage_name=f"Stage1-Freeze (epoch {epoch+1})",
                    freeze_epochs=freeze_epochs,
                    freeze_learning_rate=freeze_lr,
                    unfreeze_lr=unfreeze_lr,
                    head_lr_ratio=head_lr_ratio
                )
                # 更新 trainer 的 optimizer
                trainer.optimizer = optimizer
                # 重建调度器（绑定新 optimizer）
                if scheduler_type == 'plateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        patience=int(config["training"].get("scheduler_patience", 8)),
                        factor=float(config["training"].get("scheduler_factor", 0.5)),
                        verbose=True
                    )
                else:
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=int(config["training"].get("scheduler_step", 20)),
                        gamma=float(config["training"].get("scheduler_gamma", 0.5))
                    )
                trainer.scheduler = scheduler
                current_stage = 1
        else:
            # Stage 2: 解冻 backbone
            if current_stage != 2:
                optimizer = _apply_freeze_config(
                    logger, model, optimizer, config,
                    stage_name=f"Stage2-Unfreeze (epoch {epoch+1})",
                    freeze_epochs=0,
                    freeze_learning_rate=freeze_lr,
                    unfreeze_lr=unfreeze_lr,
                    head_lr_ratio=head_lr_ratio
                )
                if scheduler_type == 'plateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        patience=int(config["training"].get("scheduler_patience", 8)),
                        factor=float(config["training"].get("scheduler_factor", 0.5)),
                        verbose=True
                    )
                else:
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=int(config["training"].get("scheduler_step", 20)),
                        gamma=float(config["training"].get("scheduler_gamma", 0.5))
                    )
                trainer.optimizer = optimizer
                trainer.scheduler = scheduler
                current_stage = 2
                logger.info(
                    f"Stage 2 开始，解冻 backbone，学习率已重新配置，"
                    f"backbone={unfreeze_lr:.0e}, head={unfreeze_lr * head_lr_ratio:.0e}"
                )
            stage_name = f"Epoch {epoch+1}/{epochs} [Stage2-Unfreeze]"

        # ── Warmup 学习率调度 ─────────────────────────────────────────────────
        if warmup_epochs > 0 and epoch < freeze_epochs + warmup_epochs:
            # 在冻结阶段和 warmup 阶段应用 warmup
            # Stage 1 warmup: 在 freeze_lr 范围内 warmup
            # Stage 2 warmup: 从 warmup_start_lr 爬升到 unfreeze_lr
            if epoch < freeze_epochs:
                # Stage 1 warmup: 完整覆盖 freeze_epochs 周期
                # 修复: 分母改为 freeze_epochs，确保 8 轮 epoch 0-7 都覆盖
                stage1_warmup_factor = (epoch + 1) / freeze_epochs
                stage1_lr = freeze_lr * stage1_warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = stage1_lr
            else:
                # Stage 2 warmup
                stage2_epoch = epoch - freeze_epochs
                _apply_lr_warmup(
                    optimizer, stage2_epoch, warmup_epochs, warmup_start_lr, unfreeze_lr
                )

            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            logger.info(
                f"Warmup epoch {epoch+1}/{freeze_epochs + warmup_epochs}, "
                f"LR: {[f'{lr:.2e}' for lr in current_lrs]}"
            )

        # ── ArcFace margin 调度 ──────────────────────────────────────────────
        _apply_arcface_margin_schedule(
            trainer, epoch, metric_learning,
            arc_m_start, arc_m_end, arc_m_delay_epochs, arc_m_warmup_epochs
        )

        logger.info(f"Epoch {epoch+1}/{epochs}")

        # ── 训练 ─────────────────────────────────────────────────────────────
        train_loss, train_acc = trainer.train_epoch(epoch, use_amp=use_amp)

        # ── 验证 ─────────────────────────────────────────────────────────────
        val_loss, val_acc, val_metrics = trainer.validate_epoch(epoch, total_epochs=epochs, val_acc=best_acc, use_amp=use_amp)

        # ── 学习率调度 ───────────────────────────────────────────────────────
        if scheduler_type == 'plateau':
            # ReduceLROnPlateau: 根据验证 Rank-1 调整（rank_acc 越大越好，mode='max'）
            # 注意：validate_epoch 返回的 val_loss=0.0（我们不计算分类损失）
            # 这里用 rank1_acc 作为调度信号
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # ── 日志记录 ─────────────────────────────────────────────────────────
        # 统一日志格式（由 trainer.log_train_epoch 和 trainer.log_val_epoch 输出）

        # ── 稳定性检查 ───────────────────────────────────────────────────────
        if not _is_training_stable(train_loss, train_acc):
            logger.warning(
                f"[警告] Epoch {epoch+1} 训练不稳定: "
                f"Loss={train_loss}, Acc={train_acc:.4f}。"
                f"如果连续多轮出现此警告，请降低学习率或检查数据。"
            )

        # ── 保存最佳模型 ─────────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "fingerprint", "best.pth")
            trainer.save_checkpoint(
                ckpt_path, is_best=True, extra={
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "is_frozen": is_frozen,
                    "current_stage": current_stage,
                }
            )
            logger.info(f"[保存] 最佳模型: {ckpt_path} (Rank-1={val_acc:.4f})")
        else:
            no_improve_epochs += 1

        # ── 早停 ─────────────────────────────────────────────────────────────
        if (early_stopping
                and no_improve_epochs >= early_stopping_patience
                and epoch >= early_stopping_warmup):
            logger.info(
                f"连续 {no_improve_epochs} 轮无提升 (patience={early_stopping_patience})，"
                f"触发早停。"
            )
            break

        # ── 记录历史 ─────────────────────────────────────────────────────────
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        epoch_data = {
            "epoch": epoch + 1,
            "stage": current_stage,
            "is_frozen": is_frozen,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_rank1": val_acc,
            "val_rank5": val_metrics.get('rank_5', 0.0),
            "val_rank10": val_metrics.get('rank_10', 0.0),
            "val_eer": val_metrics.get('eer', 0.0),
            "feature_norm": val_metrics.get('feature_norm', 0.0),
            "learning_rates": [float(lr) for lr in current_lrs],
            "arcface_m": float(trainer.arcface_m),
        }
        training_history["epochs"].append(epoch_data)

    # ── 保存训练历史 ────────────────────────────────────────────────────────────
    os.makedirs(log_base, exist_ok=True)
    history_path = os.path.join(log_base, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"训练完成！最佳验证 Rank-1: {best_acc:.4f}")

    # ── 测试集评估 ─────────────────────────────────────────────────────────────
    if has_test:
        logger.info("=" * 60)
        logger.info("开始测试集评估...")
        # 重新加载最佳模型（早停时当前状态不一定是最佳的）
        best_ckpt_path = os.path.join(ckpt_dir, "fingerprint", "best.pth")
        if os.path.exists(best_ckpt_path):
            try:
                checkpoint = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                logger.info(f"[加载] 已加载最佳模型: {best_ckpt_path}")
            except Exception as e:
                logger.warning(f"[警告] 加载最佳模型失败: {e}，使用当前模型进行测试")
        else:
            logger.warning(f"[警告] 最佳模型不存在: {best_ckpt_path}，使用当前模型进行测试")
        try:
            test_metrics = trainer.test_epoch(epoch=epoch, total_epochs=epochs, use_amp=use_amp)
            training_history["test_metrics"] = test_metrics
            logger.info("测试集评估完成")
        except Exception as e:
            logger.error(f"[错误] 测试评估异常: {e}", exc_info=True)
            training_history["test_metrics"] = None
    else:
        logger.info("无独立测试集，跳过测试评估")

    logger.info(f"训练历史已保存: {history_path}")

    # ── 触发可视化 ─────────────────────────────────────────────────────────────
    try:
        os.makedirs(fig_dir, exist_ok=True)
        vis_script = os.path.join(script_dir, "visualize.py")
        subprocess.run(
            [sys.executable, vis_script,
             "--experiment_dir", log_base,
             "--output_dir", fig_dir,
             "--include_run_seq"],
            check=False
        )
        logger.info(f"Charts generated: {fig_dir}")
    except Exception as e:
        logger.warning(f"触发可视化失败: {e}")


if __name__ == "__main__":
    main()
