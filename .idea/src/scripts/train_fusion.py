#!/usr/bin/env python
"""
Fusion training script for face + fingerprint multimodal recognition.
Uses feature concatenation approach.
"""

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
import json
from datetime import datetime


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fusion.yaml")

    parser = argparse.ArgumentParser(description="Train fusion model (face + fingerprint)")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="fusion_recognition")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(
        experiment_name=args.experiment_name,
        log_dir=config["paths"].get("log_dir", "./logs"),
        level=config.get("logging", {}).get("level", "INFO")
    )

    set_seed(config.get("misc", {}).get("seed", 42))
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load datasets - convert relative paths to absolute
    face_data_dir = config["paths"]["face_data_dir"]
    if not os.path.isabs(face_data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        face_data_dir = os.path.join(project_root, face_data_dir.lstrip('./'))

    fingerprint_data_dir = config["paths"].get("fingerprint_data_dir")  # Optional for now
    if fingerprint_data_dir and not os.path.isabs(fingerprint_data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fingerprint_data_dir = os.path.join(project_root, fingerprint_data_dir.lstrip('./'))

    train_dataset = FusionDataset(
        face_data_dir=face_data_dir,
        fingerprint_data_dir=fingerprint_data_dir,
        mode="train",
        face_image_size=config["data"]["face_image_size"],
        fingerprint_image_size=config["data"].get("fingerprint_image_size", 224)
    )

    val_dataset = FusionDataset(
        face_data_dir=face_data_dir,
        fingerprint_data_dir=fingerprint_data_dir,
        mode="val",
        face_image_size=config["data"]["face_image_size"],
        fingerprint_image_size=config["data"].get("fingerprint_image_size", 224)
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

    # Create models
    # Load pre-trained face model for feature extraction
    face_model = create_model(
        "face",
        model_type=config["model"].get("face_model_type", "facenet"),
        num_classes=len(train_dataset.class_to_idx),
        embedding_dim=config["model"].get("face_embedding_dim", 512),
        pretrained=config["model"].get("face_pretrained", True)
    )

    # TODO: Load fingerprint model when available
    fingerprint_model = None  # Placeholder

    # Create fusion model
    fusion_model = create_model(
        "fusion",
        face_embedding_dim=config["model"].get("face_embedding_dim", 512),
        fingerprint_embedding_dim=config["model"].get("fingerprint_embedding_dim", 512),
        num_classes=len(train_dataset.class_to_idx),
        hidden_dim=config["model"].get("fusion_hidden_dim", 256),
        dropout_rate=config["model"].get("fusion_dropout_rate", 0.5)
    )

    # Count parameters
    face_params = count_parameters(face_model)[0]
    fusion_params = count_parameters(fusion_model)[0]
    logger.info(f"Face model params: {face_params:,}")
    logger.info(f"Fusion model params: {fusion_params:,}")

    # Move to device
    fusion_model = fusion_model.to(device)
    face_model = face_model.to(device)

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        fusion_model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"].get("scheduler_step", 10)),
        gamma=float(config["training"].get("scheduler_gamma", 0.1))
    )

    # Create trainer
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
        tb_writer=None
    )

    # 初始化训练历史记录
    training_history = {
        "experiment_name": args.experiment_name,
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
            ckpt_path = os.path.join(ckpt_dir, "fusion", f"best_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(ckpt_path, extra={"epoch": epoch + 1, "val_acc": val_acc})
            logger.info(f"Saved best fusion checkpoint: {ckpt_path}")

    # 保存完整的训练历史
    history_dir = os.path.join(config["paths"].get("log_dir", "./logs"), args.experiment_name)
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Training completed. Best validation accuracy: {best_acc:.4f}")
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()