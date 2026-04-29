#!/usr/bin/env python
import os
import sys
import argparse
from torch.utils.data import DataLoader
import torch

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config, set_seed, get_device, count_parameters, setup_logger, calculate_biometric_metrics, save_biometric_results
from core.datasets.face_dataset import FaceDataset
from core.models import create_model
from core.trainers.face_trainer import FaceTrainer
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
import subprocess


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "face_config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def _normalize_paths(config, script_dir, project_root):
    """统一路径解析逻辑"""
    # Log dir
    log_dir = config["paths"].get("log_dir", os.path.join(script_dir, "logs"))
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(script_dir, log_dir.lstrip('./'))

    # Checkpoint dir
    ckpt_dir = config["paths"].get("checkpoint_dir", os.path.join(script_dir, "checkpoints"))
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(script_dir, ckpt_dir.lstrip('./'))

    # Data dir (relative to project root)
    data_dir = config["paths"]["modality_data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir.lstrip('./'))

    config["paths"]["log_dir"] = log_dir
    config["paths"]["checkpoint_dir"] = ckpt_dir
    config["paths"]["data_dir"] = data_dir

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    # Use experiment name from config if available, otherwise use command line argument
    experiment_name = args.experiment_name or config.get("misc", {}).get("experiment_name", "face_recognition")

    # 统一路径解析
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config = _normalize_paths(config, script_dir, project_root)

    logger = setup_logger(experiment_name=experiment_name, log_dir=config["paths"]["log_dir"],
                          level="INFO", logger_name="FaceRecognition")

    seed = config.get("misc", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Pass augmentation params from config to dataset
    aug_params = config["data"].get("augmentation", {}) or {}

    # 统一的数据集参数
    split_ratio = config["data"].get("split_ratio", 0.8)
    test_split_ratio = config["data"].get("test_split_ratio", 0.5)  # val/test 划分比例
    gallery_per_person = config["data"].get("gallery_per_person", 3)

    train_dataset = FaceDataset(
        data_dir=config["paths"]["modality_data_dir"],
        mode="train",
        image_size=config["data"]["image_size"],
        augment=config["data"].get("use_augmentation", True),
        gallery_per_person=gallery_per_person,
        val_split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        seed=seed
    )
    train_dataset.augmentation_params = aug_params

    val_dataset = FaceDataset(
        data_dir=config["paths"]["modality_data_dir"],
        mode="val",
        image_size=config["data"]["image_size"],
        augment=False,
        class_to_idx=train_dataset.class_to_idx,
        gallery_per_person=gallery_per_person,
        val_split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        seed=seed
    )

    # 测试集（用于最终评估，仅在训练完成后使用）
    test_dataset = FaceDataset(
        data_dir=config["paths"]["modality_data_dir"],
        mode="test",
        image_size=config["data"]["image_size"],
        augment=False,
        class_to_idx=train_dataset.class_to_idx,
        gallery_per_person=gallery_per_person,
        val_split_ratio=split_ratio,
        test_split_ratio=test_split_ratio,
        seed=seed
    )

    # 统一的 DataLoader 参数
    num_workers = config["misc"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = create_model("face",
                         model_type=config["model"].get("model_type", "facenet"),
                         num_classes=len(train_dataset.class_to_idx),
                         embedding_dim=config["model"].get("embedding_dim", 512),
                         pretrained=config["model"].get("pretrained", True),
                         dropout_rate=config["model"].get("dropout_rate", 0.5),
                         spatial_attention=config["model"].get("spatial_attention", True))

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
    logger.info(f"Model type: {config['model'].get('model_type', 'facenet')}")
    logger.info(f"Spatial Attention: {config['model'].get('spatial_attention', True)}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Use enhanced FaceTrainer with built-in ArcFace support (same API as FingerprintTrainer)
    arc_s = float(config["model"].get("arc_s", 30.0))
    arc_m = float(config["model"].get("arc_m", 0.5))

    optimizer_name = config["training"].get("optimizer", "adam").lower()
    base_lr = float(config["training"].get("learning_rate", 1e-4))
    weight_decay = float(config["training"].get("weight_decay", 5e-4))

    # Optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        momentum = float(config["training"].get("momentum", 0.9))
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler_type = config["training"].get("scheduler_type", "step")
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.2, verbose=True
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config["training"].get("scheduler_step", 10)),
            gamma=float(config["training"].get("scheduler_gamma", 0.1))
        )

    # Create FaceTrainer with ArcFace (academic standard: s=64, m=0.5)
    arc_s = float(config["model"].get("arc_s", 64.0))
    arc_m = float(config["model"].get("arc_m", 0.5))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    tta = bool(config["training"].get("tta", False))

    trainer = FaceTrainer(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, device, logger, tb_writer=None,
        arcface_s=arc_s, arcface_m=arc_m,
        label_smoothing=label_smoothing, tta=tta,
        seed=seed
    )

    # 将测试集绑定到 trainer（用于最终评估）
    trainer.test_dataset = test_dataset

    # 初始化训练历史记录
    training_history = {
        "experiment_name": experiment_name,
        "model_type": "face",
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "epochs": [],
        "test_results": None  # 将在训练完成后填写
    }

    # 获取类别数量用于生物识别指标计算
    num_classes = len(train_dataset.class_to_idx)

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    early_stopping = config.get("misc", {}).get("early_stopping", False)
    early_stopping_patience = int(config.get("misc", {}).get("early_stopping_patience", 5))
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    initial_lr = float(config["training"]["learning_rate"])

    # 检查是否有测试集
    has_test_set = len(test_dataset.test_query_paths) > 0
    if has_test_set:
        logger.info(f"[数据集] 检测到测试集: {len(test_dataset.test_query_paths)} query / "
                    f"{len(test_dataset.test_gallery_paths)} gallery")
    else:
        logger.info("[数据集] 未检测到测试集（test_split_ratio=1.0 或验证人员不足），跳过测试评估")

    # Training loop with optional warmup (now supported by FaceTrainer)
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Warmup: gradually increase learning rate
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = initial_lr * 0.1 + (initial_lr - initial_lr * 0.1) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            logger.info(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR: {current_lr:.6f}")

        use_amp = config.get("misc", {}).get("use_amp", False)
        train_loss, train_acc = trainer.train_epoch(epoch, use_amp=use_amp)
        val_loss, val_acc, val_metrics = trainer.validate_epoch(epoch, total_epochs=epochs, use_amp=use_amp)

        # 计算生物识别指标（如果验证返回了概率）
        biometric_results = None
        if "probabilities" in val_metrics and "labels" in val_metrics:
            biometric_results = calculate_biometric_metrics(
                val_metrics["labels"],
                val_metrics["probabilities"],
                num_classes=num_classes
            )

        # Step scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

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
            biometric_dir = os.path.join(config["paths"].get("log_dir", "./logs"), experiment_name, "biometric_results")
            os.makedirs(biometric_dir, exist_ok=True)
            biometric_path = os.path.join(biometric_dir, f"epoch_{epoch+1}_biometric.json")
            save_biometric_results(biometric_results, biometric_path)

        training_history["epochs"].append(epoch_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            ckpt_dir = config["paths"].get("checkpoint_dir", "./checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "face", f"best_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(ckpt_path, is_best=True, extra={"epoch": epoch + 1, "val_acc": val_acc})
            logger.info(f"[保存] 最佳模型: {ckpt_path} (Val Acc={val_acc:.4f})")
            # reset early stopping counter when improvement observed
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping check
        if early_stopping and no_improve_epochs >= early_stopping_patience:
            logger.info(f"连续{no_improve_epochs}轮无提升 (patience={early_stopping_patience})，触发早停")
            break

    # ── 最终测试集评估（仅在有测试集时执行）────────────────────────────
    if has_test_set:
        logger.info("=" * 70)
        logger.info(f"【训练完成】开始最终测试集评估（使用最佳模型: Epoch {best_epoch}, Val Acc={best_val_acc:.4f}）")
        logger.info("=" * 70)

        # 使用最佳模型进行测试
        best_ckpt_path = os.path.join(ckpt_dir, "face", f"best_epoch_{best_epoch}.pth")
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            logger.info(f"[加载] 已加载最佳模型: {best_ckpt_path}")
        else:
            logger.warning(f"[警告] 最佳模型文件不存在: {best_ckpt_path}，使用当前模型进行测试")

        # 执行测试集评估
        test_metrics = trainer.test_epoch(epoch=best_epoch, use_amp=use_amp)
        training_history["test_results"] = test_metrics

        # 保存测试结果摘要
        logger.info("=" * 70)
        logger.info(f"【最终结果汇总】")
        logger.info(f"  最佳验证准确率 (Val Rank-1): {best_val_acc:.4f}")
        if test_metrics.get("rank_1") is not None:
            logger.info(f"  测试集准确率   (Test Rank-1): {test_metrics['rank_1']:.4f}")
            logger.info(f"  测试集 EER:                  {test_metrics['eer']:.4f}")
        logger.info(f"  最佳模型: Epoch {best_epoch}")
        logger.info("=" * 70)
    else:
        logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")

    # 保存完整的训练历史
    history_dir = os.path.join(config["paths"].get("log_dir", "./logs"), experiment_name)
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Training history saved to: {history_path}")
    # 自动触发可视化（包含序号），非阻塞
    try:
        output_dir = config["paths"].get("visualization_dir", "./visualization_results")
        vis_script = os.path.join(project_root, "scripts", "visualize.py")
        subprocess.run([sys.executable, vis_script, "--experiment_dir", history_dir, "--output_dir", output_dir, "--include_run_seq"], check=False)
        logger.info(f"Triggered visualization for {experiment_name} -> {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to trigger visualization: {e}")


if __name__ == "__main__":
    main()

