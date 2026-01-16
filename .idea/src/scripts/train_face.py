#!/usr/bin/env python
import os
import sys
import argparse
from torch.utils.data import DataLoader
import torch

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, count_parameters, setup_logger
from core.datasets.face_dataset import FaceDataset
from core.models import create_model
from core.trainers.face_trainer import FaceTrainer
import torch.nn as nn
import torch.optim as optim


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "face_config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="face_recognition")
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
    face_data_dir = config["paths"]["data_dir"]
    if not os.path.isabs(face_data_dir):
        # If relative path, make it relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        face_data_dir = os.path.join(project_root, face_data_dir.lstrip('./'))
    train_dataset = FaceDataset(face_data_dir, mode="train", image_size=config["data"]["face_image_size"])
    val_dataset = FaceDataset(face_data_dir, mode="val", image_size=config["data"]["face_image_size"])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["misc"].get("num_workers", 0))
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["misc"].get("num_workers", 0))

    model = create_model("face",
                         model_type=config["model"].get("model_type", "facenet"),
                         num_classes=len(train_dataset.class_to_idx),
                         embedding_dim=config["model"].get("face_embedding_dim", 512),
                         pretrained=config["model"].get("pretrained", True),
                         dropout_rate=config["model"].get("dropout_rate", 0.5))

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(config["training"]["learning_rate"]), momentum=float(config["training"]["momentum"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config["training"].get("scheduler_step", 10)), gamma=float(config["training"].get("scheduler_gamma", 0.1)))

    trainer = FaceTrainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device, logger, tb_writer=None)

    start_epoch = 0
    best_acc = 0.0
    epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc, val_metrics = trainer.validate_epoch(epoch)
        scheduler.step()
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = config["paths"].get("checkpoint_dir", "./checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(ckpt_path, extra={"epoch": epoch + 1, "val_acc": val_acc})
            logger.info(f"Saved best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

