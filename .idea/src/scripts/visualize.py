# 可视化训练历史和生物识别指标脚本
import os
import sys
import argparse
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import (
    plot_training_curves, plot_roc_curves, plot_det_curves, plot_far_frr_curves,
    load_config, set_seed
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training history and biometric metrics")
    parser.add_argument("--logs_dir", type=str, default="./logs",
                       help="Root directory containing training logs")
    parser.add_argument("--modalities", type=str, nargs='+',
                       choices=["face", "fingerprint", "fusion"], default=["face", "fingerprint", "fusion"],
                       help="Modalities to visualize")
    parser.add_argument("--experiment_pattern", type=str, default="*",
                       help="Pattern to match experiment names (supports wildcards)")
    parser.add_argument("--output_dir", type=str, default="./visualization_results",
                       help="Output directory for generated plots")
    parser.add_argument("--generate_comparison", action="store_true",
                       help="Generate comparison plots across modalities")
    return parser.parse_args()


def load_training_history(history_path):
    """Load training history from JSON file"""
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {history_path}: {e}")
        return None


def load_biometric_results(biometric_dir, epoch):
    """Load biometric results for specific epoch"""
    biometric_file = os.path.join(biometric_dir, f"epoch_{epoch}_biometric.json")
    try:
        with open(biometric_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {biometric_file}: {e}")
        return None


def extract_training_curves(history_data):
    """Extract training curves data from history"""
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []
    biometric_eers = []
    biometric_aucs = []

    for epoch_data in history_data["epochs"]:
        epochs.append(epoch_data["epoch"])
        train_losses.append(epoch_data["train_loss"])
        val_losses.append(epoch_data["val_loss"])
        train_accs.append(epoch_data["train_acc"])
        val_accs.append(epoch_data["val_acc"])
        learning_rates.append(epoch_data.get("learning_rate", 0))

        # Extract biometric metrics if available
        biometric = epoch_data.get("biometric_metrics", {})
        biometric_eers.append(biometric.get("eer", 0))
        biometric_aucs.append(biometric.get("auc", 0))

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "learning_rates": learning_rates,
        "biometric_eers": biometric_eers,
        "biometric_aucs": biometric_aucs
    }


def plot_learning_rate_curve(epochs, learning_rates, save_path=None):
    """Plot learning rate schedule"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_biometric_evolution(epochs, eers, aucs, save_path=None):
    """Plot biometric metrics evolution over training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # EER evolution
    ax1.plot(epochs, eers, 'r-o', linewidth=2, markersize=4, label='EER')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Equal Error Rate (EER)')
    ax1.set_title('EER Evolution During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # AUC evolution
    ax2.plot(epochs, aucs, 'g-s', linewidth=2, markersize=4, label='AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Area Under Curve (AUC)')
    ax2.set_title('AUC Evolution During Training')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_experiment_plots(experiment_dir, experiment_name, modality, output_dir):
    """Generate plots for a single experiment"""
    print(f"Processing experiment: {experiment_name} ({modality})")

    # Load training history
    history_path = os.path.join(experiment_dir, "training_history.json")
    history_data = load_training_history(history_path)

    if not history_data:
        print(f"Skipping {experiment_name}: no training history found")
        return

    # Create output directory structure
    exp_output_dir = os.path.join(output_dir, modality, experiment_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Extract training curves
    curves_data = extract_training_curves(history_data)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot training curves
    training_curve_path = os.path.join(exp_output_dir, f"training_curves_{timestamp}.png")
    plot_training_curves(
        curves_data["train_losses"],
        curves_data["val_losses"],
        curves_data["train_accs"],
        curves_data["val_accs"],
        save_path=training_curve_path
    )

    # Plot learning rate curve
    lr_curve_path = os.path.join(exp_output_dir, f"learning_rate_{timestamp}.png")
    plot_learning_rate_curve(
        curves_data["epochs"],
        curves_data["learning_rates"],
        save_path=lr_curve_path
    )

    # Plot biometric evolution if data available
    if any(curves_data["biometric_eers"]) and any(curves_data["biometric_aucs"]):
        biometric_evolution_path = os.path.join(exp_output_dir, f"biometric_evolution_{timestamp}.png")
        plot_biometric_evolution(
            curves_data["epochs"],
            curves_data["biometric_eers"],
            curves_data["biometric_aucs"],
            save_path=biometric_evolution_path
        )

    # Generate biometric plots for best epoch
    biometric_dir = os.path.join(experiment_dir, "biometric_results")
    if os.path.exists(biometric_dir):
        # Find the epoch with best validation accuracy
        best_epoch = max(history_data["epochs"], key=lambda x: x["val_acc"])["epoch"]

        biometric_results = load_biometric_results(biometric_dir, best_epoch)
        if biometric_results:
            # Plot ROC curves
            roc_path = os.path.join(exp_output_dir, f"roc_curves_epoch_{best_epoch}_{timestamp}.png")
            plot_roc_curves(biometric_results, save_path=roc_path)

            # Plot DET curves
            det_path = os.path.join(exp_output_dir, f"det_curves_epoch_{best_epoch}_{timestamp}.png")
            plot_det_curves(biometric_results, save_path=det_path)

            # Plot FAR/FRR curves
            far_frr_path = os.path.join(exp_output_dir, f"far_frr_curves_epoch_{best_epoch}_{timestamp}.png")
            plot_far_frr_curves(biometric_results, save_path=far_frr_path)

    print(f"Plots saved to: {exp_output_dir}")


def generate_comparison_plots(all_experiments, output_dir):
    """Generate comparison plots across modalities"""
    print("Generating comparison plots across modalities...")

    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    # Collect data for comparison
    modality_data = {}

    for modality in ["face", "fingerprint", "fusion"]:
        if modality not in all_experiments:
            continue

        modality_data[modality] = {
            "final_accuracies": [],
            "final_losses": [],
            "best_eers": [],
            "best_aucs": [],
            "experiment_names": []
        }

        for exp_name, exp_data in all_experiments[modality].items():
            if not exp_data["epochs"]:
                continue

            final_epoch = exp_data["epochs"][-1]
            best_epoch = max(exp_data["epochs"], key=lambda x: x["val_acc"])

            modality_data[modality]["final_accuracies"].append(final_epoch["val_acc"])
            modality_data[modality]["final_losses"].append(final_epoch["val_loss"])
            modality_data[modality]["experiment_names"].append(exp_name)

            # Add biometric metrics if available
            biometric = best_epoch.get("biometric_metrics", {})
            modality_data[modality]["best_eers"].append(biometric.get("eer", 0))
            modality_data[modality]["best_aucs"].append(biometric.get("auc", 0))

    # Create comparison plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    modalities = list(modality_data.keys())
    x = np.arange(len(modalities))
    width = 0.35

    for i, modality in enumerate(modalities):
        if modality_data[modality]["final_accuracies"]:
            avg_acc = np.mean(modality_data[modality]["final_accuracies"])
            std_acc = np.std(modality_data[modality]["final_accuracies"])

            plt.bar(i, avg_acc, width, label=f'{modality} (n={len(modality_data[modality]["final_accuracies"])})',
                   alpha=0.7, capsize=5, yerr=std_acc)

    plt.xlabel('Modality')
    plt.ylabel('Final Validation Accuracy')
    plt.title('Accuracy Comparison Across Modalities')
    plt.xticks(x, modalities)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    acc_comparison_path = os.path.join(comparison_dir, f"accuracy_comparison_{timestamp}.png")
    plt.savefig(acc_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Biometric comparison (if data available)
    if any(any(data["best_eers"]) for data in modality_data.values()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # EER comparison
        x = np.arange(len(modalities))
        width = 0.35

        eer_values = [np.mean(data["best_eers"]) if data["best_eers"] else 0 for data in modality_data.values()]
        eer_stds = [np.std(data["best_eers"]) if data["best_eers"] else 0 for data in modality_data.values()]

        bars1 = ax1.bar(x, eer_values, width, alpha=0.7, capsize=5, yerr=eer_stds)
        ax1.set_xlabel('Modality')
        ax1.set_ylabel('Equal Error Rate (EER)')
        ax1.set_title('EER Comparison Across Modalities')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modalities)
        ax1.grid(True, alpha=0.3)

        # AUC comparison
        auc_values = [np.mean(data["best_aucs"]) if data["best_aucs"] else 0 for data in modality_data.values()]
        auc_stds = [np.std(data["best_aucs"]) if data["best_aucs"] else 0 for data in modality_data.values()]

        bars2 = ax2.bar(x, auc_values, width, alpha=0.7, capsize=5, yerr=auc_stds)
        ax2.set_xlabel('Modality')
        ax2.set_ylabel('Area Under Curve (AUC)')
        ax2.set_title('AUC Comparison Across Modalities')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modalities)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        biometric_comparison_path = os.path.join(comparison_dir, f"biometric_comparison_{timestamp}.png")
        plt.savefig(biometric_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Comparison plots saved to: {comparison_dir}")


def main():
    args = parse_args()

    print("Starting visualization generation...")
    print(f"Logs directory: {args.logs_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Modalities: {args.modalities}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all experiments
    all_experiments = {}

    for modality in args.modalities:
        modality_dir = os.path.join(args.logs_dir, modality)
        if not os.path.exists(modality_dir):
            print(f"Warning: Modality directory not found: {modality_dir}")
            continue

        # Find all experiment directories
        experiment_dirs = glob.glob(os.path.join(modality_dir, args.experiment_pattern))

        all_experiments[modality] = {}

        for exp_dir in experiment_dirs:
            if not os.path.isdir(exp_dir):
                continue

            exp_name = os.path.basename(exp_dir)
            print(f"Found experiment: {modality}/{exp_name}")

            # Load training history for this experiment
            history_path = os.path.join(exp_dir, "training_history.json")
            history_data = load_training_history(history_path)

            if history_data:
                all_experiments[modality][exp_name] = history_data

                # Generate plots for this experiment
                generate_experiment_plots(exp_dir, exp_name, modality, args.output_dir)
            else:
                print(f"Warning: No valid training history for {modality}/{exp_name}")

    # Generate comparison plots if requested
    if args.generate_comparison and len(all_experiments) > 1:
        generate_comparison_plots(all_experiments, args.output_dir)

    # Save summary
    summary = {
        "generation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "logs_directory": args.logs_dir,
        "output_directory": args.output_dir,
        "modalities_processed": list(all_experiments.keys()),
        "experiments_processed": {
            modality: list(experiments.keys())
            for modality, experiments in all_experiments.items()
        }
    }

    summary_path = os.path.join(args.output_dir, f"visualization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nVisualization generation completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()