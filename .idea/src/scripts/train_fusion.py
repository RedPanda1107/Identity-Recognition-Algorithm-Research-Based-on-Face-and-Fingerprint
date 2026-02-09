#!/usr/bin/env python
"""
Fusion training script for face + fingerprint multimodal recognition.
Uses feature concatenation approach.
"""
# Windows UTF-8 encoding fix
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import sys
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, count_parameters, setup_logger, calculate_biometric_metrics, save_biometric_results
from core.datasets.fusion_dataset import FusionDataset
from core.models import create_model
from core.trainers.fusion_trainer import FusionTrainer
from core.losses.arcface import ArcMarginProduct
import json
from datetime import datetime
import subprocess


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fusion_config.yaml")

    parser = argparse.ArgumentParser(description="Train fusion model (face + fingerprint)")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="fusion_recognition")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fusion_method", type=str, default="concat",
                       choices=['concat', 'attention_fusion', 'cross_attention', 'tensor_fusion'])
    parser.add_argument("--face_ckpt", type=str, default=None,
                       help="Path to pretrained face model checkpoint")
    parser.add_argument("--fp_ckpt", type=str, default=None,
                       help="Path to pretrained fingerprint model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(
        experiment_name=args.experiment_name,
        log_dir=config["paths"].get("log_dir", "./logs"),
        level=config.get("logging", {}).get("level", "INFO"),
        logger_name="FusionRecognition"
    )

    # Normalize log_dir and checkpoint_dir to be under scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_log_dir = os.path.join(script_dir, "logs")
    default_ckpt_dir = os.path.join(script_dir, "checkpoints")

    log_dir = config["paths"].get("log_dir", default_log_dir)
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(script_dir, log_dir.lstrip('./'))

    ckpt_dir = config["paths"].get("checkpoint_dir", default_ckpt_dir)
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(script_dir, ckpt_dir.lstrip('./'))

    config["paths"]["log_dir"] = log_dir
    config["paths"]["checkpoint_dir"] = ckpt_dir

    # Re-create logger with normalized path to ensure handlers use script-level logs dir
    logger = setup_logger(
        experiment_name=args.experiment_name,
        log_dir=log_dir,
        level=config.get("logging", {}).get("level", "INFO"),
        logger_name="FusionRecognition"
    )

    set_seed(config.get("misc", {}).get("seed", 42))
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load datasets - convert relative paths to absolute
    face_data_dir = config["paths"]["face_data_dir"]
    if not os.path.isabs(face_data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        face_data_dir = os.path.join(project_root, face_data_dir.lstrip('./'))

    fingerprint_data_dir = config["paths"]["fingerprint_data_dir"]
    if not os.path.isabs(fingerprint_data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fingerprint_data_dir = os.path.join(project_root, fingerprint_data_dir.lstrip('./'))

    # 可选的映射文件
    mapping_file = config["paths"].get("mapping_file")
    if mapping_file and not os.path.isabs(mapping_file):
        mapping_file = os.path.join(project_root, mapping_file.lstrip('./'))

    train_dataset = FusionDataset(
        face_data_dir=face_data_dir,
        fingerprint_data_dir=fingerprint_data_dir,
        mapping_file=mapping_file,
        mode="train",
        face_image_size=config["data"]["face_image_size"],
        fingerprint_image_size=config["data"]["fingerprint_image_size"],
        augment=config["data"].get("use_augmentation", True)
    )
    # attach augmentation params from config so FusionDataset can build modality-specific transforms
    train_dataset.augmentation_params = config["data"].get("augmentation", {}) or {}

    val_dataset = FusionDataset(
        face_data_dir=face_data_dir,
        fingerprint_data_dir=fingerprint_data_dir,
        mapping_file=mapping_file,
        mode="val",
        face_image_size=config["data"]["face_image_size"],
        fingerprint_image_size=config["data"]["fingerprint_image_size"],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["misc"].get("num_workers", 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["misc"].get("num_workers", 0)
    )

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Val set size: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.class_to_idx)}")

    # 🔧 【指令C】确认数据对齐绑定
    logger.info("[Data] ✅ 人脸-指纹数据对齐绑定已确认:")
    logger.info(f"[Data] - 人脸数据目录: {face_data_dir}")
    logger.info(f"[Data] - 指纹数据目录: {fingerprint_data_dir}")
    if train_dataset.samples:
        sample = train_dataset.samples[0]
        logger.info(f"[Data] - 配对示例: face={os.path.basename(sample['face_path'])}, "
                   f"fp={os.path.basename(sample['fingerprint_path'])}, label={sample['label']}")

    # Create models
    # Load pre-trained face model for feature extraction
    face_model = create_model(
        "face",
        model_type=config["model"].get("face_model_type", "facenet"),
        num_classes=len(train_dataset.class_to_idx),
        embedding_dim=config["model"].get("face_embedding_dim", 512),
        pretrained=config["model"].get("face_pretrained", True),
        dropout_rate=config["model"].get("dropout_rate", 0.5),
        spatial_attention=config["model"].get("face_spatial_attention", False)  # 🔧 禁用
    )

    # Load pre-trained fingerprint model for feature extraction
    fingerprint_model = create_model(
        "fingerprint",
        model_type=config["model"].get("fingerprint_model_type", "fingerprint_net"),
        num_classes=len(train_dataset.class_to_idx),
        embedding_dim=config["model"].get("fingerprint_embedding_dim", 256),
        pretrained=config["model"].get("fingerprint_pretrained", False),
        dropout_rate=config["model"].get("dropout_rate", 0.5),
        spatial_attention=config["model"].get("spatial_attention", True)
    )

    # Create fusion model
    fusion_model = create_model(
        "fusion",
        face_embedding_dim=config["model"].get("face_embedding_dim", 512),
        fingerprint_embedding_dim=config["model"].get("fingerprint_embedding_dim", 256),
        num_classes=len(train_dataset.class_to_idx),
        hidden_dim=config["model"].get("fusion_hidden_dim", 512),
        dropout_rate=config["model"].get("fusion_dropout_rate", 0.3),
        fusion_method=args.fusion_method
    )

    # Count parameters
    face_params = count_parameters(face_model)[0]
    fp_params = count_parameters(fingerprint_model)[0]
    fusion_params = count_parameters(fusion_model)[0]

    logger.info(f"Face model params: {face_params:,}")
    logger.info(f"Fingerprint model params: {fp_params:,}")
    logger.info(f"Fusion model params: {fusion_params:,}")
    logger.info(f"Total trainable params: {face_params + fp_params + fusion_params:,}")

    # Move to device
    fusion_model = fusion_model.to(device)
    face_model = face_model.to(device)
    fingerprint_model = fingerprint_model.to(device)

    # 【指令B】设置ArcFace分类器到FusionModel
    num_classes = len(train_dataset.class_to_idx)
    face_embedding_dim = config["model"].get("face_embedding_dim", 512)
    fp_embedding_dim = config["model"].get("fingerprint_embedding_dim", 256)
    fusion_features_dim = face_embedding_dim + fp_embedding_dim  # 768

    # 创建ArcFace分类器并设置到FusionModel
    arc_classifier = ArcMarginProduct(
        in_features=fusion_features_dim,  # 融合特征维度768
        out_features=num_classes,
        s=30.0,
        m=0.3
    ).to(device)
    fusion_model.arc_classifier = arc_classifier
    logger.info(f"[Train] ArcFace classifier set: in={fusion_features_dim}, out={num_classes}, s=30.0, m=0.3")

    # 使用CrossEntropyLoss（因为FusionModel内部已集成ArcFace）
    criterion = nn.CrossEntropyLoss()

    # 🔧 【指令B】使用AdamW优化器 + Warmup
    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 1e-2))

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            fusion_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"[Train] 使用 AdamW 优化器, lr={learning_rate}, weight_decay={weight_decay}")
    else:
        momentum = float(config["training"].get("momentum", 0.9))
        optimizer = optim.SGD(
            fusion_model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        logger.info(f"[Train] 使用 SGD 优化器, lr={learning_rate}, momentum={momentum}")

    # 🔧 【指令B】添加Cosine学习率调度 + Warmup
    scheduler_type = config["training"].get("scheduler_type", "cosine").lower()
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    epochs = int(config["training"]["epochs"])

    if scheduler_type == "cosine":
        # Cosine Annealing with Warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,  # 减去warmup期
            eta_min=1e-6
        )
        # 自定义Warmup调度
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # 从10%开始
            total_iters=warmup_epochs
        )
        logger.info(f"[Train] 使用 Cosine调度 + {warmup_epochs}轮 Warmup")
    else:
        # StepLR作为fallback
        step_size = int(config["training"].get("scheduler_step", 20))
        gamma = float(config["training"].get("scheduler_gamma", 0.1))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        warmup_scheduler = None
        logger.info(f"[Train] 使用 StepLR 调度")

    # Create trainer - pass pretrained checkpoints
    pretrained_ckpts = {
        'face': args.face_ckpt,
        'fingerprint': args.fp_ckpt
    }

    # 🔧 【指令C】解冻策略参数
    unfreeze_epoch = int(config["training"].get("unfreeze_epoch", 10))
    face_lr = float(config["training"].get("face_lr", 1e-5))
    fp_lr = float(config["training"].get("fp_lr", 1e-5))

    logger.info(f"[Train] 解冻策略: Epoch {unfreeze_epoch} 后解冻Backbone")
    logger.info(f"[Train] Backbone学习率: face={face_lr}, fingerprint={fp_lr}")

    trainer = FusionTrainer(
        fusion_model=fusion_model,
        face_model=face_model,
        fingerprint_model=fingerprint_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        logger=logger,
        tb_writer=None,
        pretrained_ckpts=pretrained_ckpts,
        unfreeze_epoch=unfreeze_epoch,
        face_lr=face_lr,
        fp_lr=fp_lr
    )

    # 初始化训练历史记录
    training_history = {
        "experiment_name": args.experiment_name,
        "fusion_method": args.fusion_method,
        "model_type": "fusion",
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "epochs": []
    }

    # 获取类别数量用于生物识别指标计算
    num_classes = len(train_dataset.class_to_idx)

    # Training loop
    start_epoch = 0
    best_acc = 0.0
    no_improve_epochs = 0
    early_stopping = config.get("misc", {}).get("early_stopping", False)
    early_stopping_patience = int(config.get("misc", {}).get("early_stopping_patience", 5))
    epochs = int(config["training"]["epochs"])

    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc, val_metrics = trainer.validate_epoch(epoch)

        # 计算生物识别指标
        biometric_results = None
        if "probabilities" in val_metrics and "labels" in val_metrics:
            biometric_results = calculate_biometric_metrics(
                val_metrics["labels"],
                val_metrics["probabilities"],
                num_classes=num_classes
            )

        scheduler.step()

        # 🔧 【指令B】Warmup调度器处理
        if warmup_scheduler is not None and epoch < warmup_epochs - 1:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"[Warmup] Epoch {epoch+1}: lr = {current_lr:.6f}")

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1_score']:.4f}")

        # 记录当前epoch的历史数据
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "val_metrics": {
                k: v for k, v in val_metrics.items()
                if k not in ["predictions", "labels", "probabilities"]  # 不保存大数据
            }
        }

        if biometric_results:
            epoch_data["biometric_metrics"] = {
                "eer": biometric_results.get("macro_avg", {}).get("eer", 0),
                "auc": biometric_results.get("macro_avg", {}).get("auc", 0)
            }
            # 保存详细的生物识别结果
            biometric_dir = os.path.join(config["paths"].get("log_dir", "./logs"), args.experiment_name, "biometric_results")
            os.makedirs(biometric_dir, exist_ok=True)
            biometric_path = os.path.join(biometric_dir, f"epoch_{epoch+1}_biometric.json")
            save_biometric_results(biometric_results, biometric_path)

        training_history["epochs"].append(epoch_data)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = config["paths"].get("checkpoint_dir", "./checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_{args.fusion_method}_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(ckpt_path, is_best=True, extra={"epoch": epoch + 1, "val_acc": val_acc})
            logger.info(f"[保存] 最佳融合模型: {ckpt_path} (Acc={val_acc:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if early_stopping and no_improve_epochs >= early_stopping_patience:
            logger.info(f"连续{no_improve_epochs}轮无提升 (patience={early_stopping_patience})，触发早停")
            break

    # 保存完整的训练历史
    history_dir = os.path.join(config["paths"].get("log_dir", "./logs"), args.experiment_name)
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"{args.fusion_method}_training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"训练完成! 最佳验证准确率: {best_acc:.4f}")
    logger.info(f"Training history saved to: {history_path}")
    # 自动触发可视化（包含序号），非阻塞
    try:
        output_dir = config["paths"].get("visualization_dir", "./visualization_results")
        vis_script = os.path.join(project_root, "scripts", "visualize.py")
        subprocess.run([sys.executable, vis_script, "--experiment_dir", history_dir, "--output_dir", output_dir, "--include_run_seq"], check=False)
        logger.info(f"Triggered visualization for {args.experiment_name} -> {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to trigger visualization: {e}")


if __name__ == "__main__":
    main()