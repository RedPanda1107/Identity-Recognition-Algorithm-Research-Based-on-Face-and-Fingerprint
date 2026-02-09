# 模型评估脚本
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime
import subprocess

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import (
    load_config, set_seed, get_device, calculate_biometric_metrics,
    plot_roc_curves, plot_det_curves, plot_far_frr_curves,
    plot_training_curves, plot_confusion_matrix, save_biometric_results
)
from core.datasets.face_dataset import FaceDataset
from core.datasets.fingerprint_dataset import FingerprintDataset
from core.datasets.fusion_dataset import FusionDataset
from core.models import create_model
from core.trainers.face_trainer import FaceTrainer
from core.trainers.fingerprint_trainer import FingerprintTrainer
from core.trainers.fusion_trainer import FusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models and generate biometric metrics")
    parser.add_argument("--model_type", type=str, required=True, choices=["face", "fingerprint", "fusion"],
                       help="Type of model to evaluate")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--experiment_name", type=str, default="evaluation",
                       help="Name for this evaluation experiment")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for evaluation")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path (optional, will try to infer from model type)")
    return parser.parse_args()


def load_model_and_config(model_type, checkpoint_path, device, config_path=None):
    """Load model and configuration from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to infer config path from model type
    if not config_path or not os.path.exists(config_path):
        config_paths = {
            "face": "configs/face_config.yaml",
            "fingerprint": "configs/fingerprint_config.yaml",
            "fusion": "configs/fusion_config.yaml"
        }
        config_path = config_paths.get(model_type, "configs/face_config.yaml")

    config = load_config(config_path)

    # Create model
    if model_type == "fusion":
        # For fusion model, we need to load both face and fingerprint models
        face_checkpoint = config.get("paths", {}).get("face_checkpoint", "")
        fingerprint_checkpoint = config.get("paths", {}).get("fingerprint_checkpoint", "")

        face_model = create_model("face",
                                model_type=config["model"].get("face_model_type", "facenet"),
                                num_classes=config["model"].get("num_classes", 200),
                                embedding_dim=config["model"].get("face_embedding_dim", 512),
                                pretrained=False)
        fingerprint_model = create_model("fingerprint",
                                       model_type=config["model"].get("fingerprint_model_type", "fingerprint_net"),
                                       embedding_dim=config["model"].get("fingerprint_embedding_dim", 256),
                                       pretrained=False)

        # 加载预训练权重
        if face_checkpoint and os.path.exists(face_checkpoint):
            face_model.load_state_dict(torch.load(face_checkpoint, map_location=device)["model_state"])
        if fingerprint_checkpoint and os.path.exists(fingerprint_checkpoint):
            fingerprint_model.load_state_dict(torch.load(fingerprint_checkpoint, map_location=device)["model_state"])

        model = create_model("fusion",
                           face_embedding_dim=config["model"].get("face_embedding_dim", 512),
                           fingerprint_embedding_dim=config["model"].get("fingerprint_embedding_dim", 256),
                           num_classes=config["model"].get("num_classes", 200),
                           hidden_dim=config["model"].get("fusion_hidden_dim", 512),
                           dropout_rate=config["model"].get("fusion_dropout_rate", 0.3),
                           fusion_method=config["model"].get("fusion_method", "concat"))
    else:
        model = create_model(model_type,
                           model_type=config["model"].get("model_type", "facenet"),
                           num_classes=config["model"].get("num_classes", 200),
                           embedding_dim=config["model"].get("embedding_dim", 512))

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(model, val_loader, device, model_type):
    """Evaluate model and collect predictions/probabilities"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Evaluating {model_type} model...")

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            targets = batch["label"]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_labels, all_probs


def create_evaluation_plots(predictions, labels, probabilities, biometric_results,
                          experiment_name, model_type, save_dir):
    """Create and save evaluation plots"""
    print("Generating evaluation plots...")

    # Create results directory structure
    results_dir = os.path.join(save_dir, model_type, experiment_name, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for this evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot ROC curves
    roc_path = os.path.join(results_dir, f"roc_curves_{timestamp}.png")
    plot_roc_curves(biometric_results, save_path=roc_path)

    # Plot DET curves
    det_path = os.path.join(results_dir, f"det_curves_{timestamp}.png")
    plot_det_curves(biometric_results, save_path=det_path)

    # Plot FAR/FRR curves
    far_frr_path = os.path.join(results_dir, f"far_frr_curves_{timestamp}.png")
    plot_far_frr_curves(biometric_results, save_path=far_frr_path)

    # Save biometric results
    biometric_json_path = os.path.join(results_dir, f"biometric_metrics_{timestamp}.json")
    save_biometric_results(biometric_results, biometric_json_path)

    print(f"Plots saved to: {results_dir}")

    return {
        "roc_curve": roc_path,
        "det_curve": det_path,
        "far_frr_curve": far_frr_path,
        "biometric_json": biometric_json_path
    }


def main():
    args = parse_args()

    # Setup
    set_seed(42)
    device = get_device(args.device)

    print(f"Evaluating {args.model_type} model: {args.checkpoint_path}")

    # Load model and config
    model, config = load_model_and_config(args.model_type, args.checkpoint_path, device, args.config)

    # Create dataset
    data_dir = config["paths"]["data_dir"]
    if args.model_type == "face":
        dataset = FaceDataset(data_dir, mode="val", image_size=config["data"]["face_image_size"])
    elif args.model_type == "fingerprint":
        dataset = FingerprintDataset(data_dir, mode="val", image_size=config["data"]["fingerprint_image_size"])
    elif args.model_type == "fusion":
        dataset = FusionDataset(data_dir, mode="val",
                              face_image_size=config["data"]["face_image_size"],
                              fingerprint_image_size=config["data"]["fingerprint_image_size"])

    val_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                           shuffle=False, num_workers=config["misc"].get("num_workers", 0))

    # Evaluate model
    predictions, labels, probabilities = evaluate_model(model, val_loader, device, args.model_type)

    # Calculate biometric metrics
    num_classes = len(dataset.class_to_idx)
    biometric_results = calculate_biometric_metrics(labels, probabilities, num_classes=num_classes)

    # Print results
    print("\nEvaluation Results:")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"EER (Macro): {biometric_results.get('macro_avg', {}).get('eer', 0):.4f}")
    print(f"AUC (Macro): {biometric_results.get('macro_avg', {}).get('auc', 0):.4f}")
    print(f"EER (Overall): {biometric_results.get('overall', {}).get('eer', 0):.4f}")
    # Create plots and save results
    save_dir = config["paths"].get("log_dir", "./logs")
    plot_paths = create_evaluation_plots(predictions, labels, probabilities, biometric_results,
                                       args.experiment_name, args.model_type, save_dir)

    # Save evaluation summary
    summary = {
        "model_type": args.model_type,
        "experiment_name": args.experiment_name,
        "checkpoint_path": args.checkpoint_path,
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "num_samples": len(labels),
        "num_classes": num_classes,
        "biometric_metrics": {
            "eer_macro": biometric_results.get("macro_avg", {}).get("eer", 0),
            "auc_macro": biometric_results.get("macro_avg", {}).get("auc", 0),
            "eer_overall": biometric_results.get("overall", {}).get("eer", 0),
            "auc_overall": biometric_results.get("overall", {}).get("auc", 0)
        },
        "plot_paths": plot_paths
    }

    summary_path = os.path.join(save_dir, args.model_type, args.experiment_name,
                               "evaluation_results", f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation summary saved to: {summary_path}")
    print("Evaluation completed!")
    # 尝试触发训练可视化（若存在训练历史），包含序号
    try:
        history_dir = os.path.join(save_dir, args.model_type, args.experiment_name)
        output_dir = os.path.join(save_dir, "visualization_results")
        vis_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "visualize.py")
        subprocess.run([sys.executable, vis_script, "--experiment_dir", history_dir, "--output_dir", output_dir, "--include_run_seq"], check=False)
        print(f"Triggered visualization for {history_dir} -> {output_dir}")
    except Exception as e:
        print(f"Failed to trigger visualization: {e}")


if __name__ == "__main__":
    main()