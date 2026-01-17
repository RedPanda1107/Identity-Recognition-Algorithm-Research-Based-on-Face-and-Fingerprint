#!/usr/bin/env python
"""
Fingerprint recognition training script.
Implements the complete training pipeline for fingerprint identification.
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

from core.utils.utils import load_config, set_seed, get_device, count_parameters, setup_logger, calculate_biometric_metrics, save_biometric_results
from core.datasets.fingerprint_dataset import FingerprintDataset
from core.models import create_model
from core.trainers.fingerprint_trainer import FingerprintTrainer
import json
from datetime import datetime


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fingerprint_config.yaml")

    parser = argparse.ArgumentParser(description="Train fingerprint recognition model")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="fingerprint_recognition")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Use experiment name from config if available, otherwise use command line argument
    experiment_name = config.get("misc", {}).get("experiment_name", args.experiment_name)
    logger = setup_logger(experiment_name=experiment_name, log_dir=config["paths"].get("log_dir", "./logs"), level="INFO")

    set_seed(config.get("misc", {}).get("seed", 42))
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Convert relative path to absolute path relative to project root
    fingerprint_data_dir = config["paths"]["fingerprint_data_dir"]
    if not os.path.isabs(fingerprint_data_dir):
        # If relative path, make it relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fingerprint_data_dir = os.path.join(project_root, fingerprint_data_dir.lstrip('./'))

    # Create datasets
    train_dataset = FingerprintDataset(
        fingerprint_data_dir,
        mode="train",
        image_size=config["data"]["fingerprint_image_size"],
        augment=config["data"].get("use_augmentation", True)
    )
    val_dataset = FingerprintDataset(
        fingerprint_data_dir,
        mode="val",
        image_size=config["data"]["fingerprint_image_size"],
        augment=False
    )

    # Create data loaders
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

    # Create model
    model = create_model(
        "fingerprint",
        model_type=config["model"].get("model_type", "fingerprint_net"),
        num_classes=len(train_dataset.class_to_idx),
        embedding_dim=config["model"].get("embedding_dim", 256),
        pretrained=config["model"].get("pretrained", False),
        dropout_rate=config["model"].get("dropout_rate", 0.5)
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
    logger.info(f"Number of classes: {len(train_dataset.class_to_idx)}")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"].get("scheduler_step", 20)),
        gamma=float(config["training"].get("scheduler_gamma", 0.1))
    )

    # Create trainer
    trainer = FingerprintTrainer(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device, logger, tb_writer=None
    )

    # 初始化训练历史记录
    training_history = {
        "experiment_name": args.experiment_name,
        "model_type": "fingerprint",
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

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Train one epoch
        train_loss, train_acc = trainer.train_epoch(epoch)

        # Validate
        val_loss, val_acc, val_metrics = trainer.validate_epoch(epoch)

        # 计算生物识别指标
        biometric_results = None
        if "probabilities" in val_metrics and "labels" in val_metrics:
            biometric_results = calculate_biometric_metrics(
                val_metrics["labels"],
                val_metrics["probabilities"],
                num_classes=num_classes
            )

        # Step scheduler
        scheduler.step()

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_metrics:
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

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = config["paths"].get("checkpoint_dir", "./checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "fingerprint", f"best_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(ckpt_path, extra={
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_loss": val_loss
            })
            logger.info(f"Saved best checkpoint: {ckpt_path} (Acc: {val_acc:.4f})")

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