#!/usr/bin/env python
"""
实验可视化管理器
统一管理所有实验的图表生成、数据汇总与横向对比

功能：
    1. 从 history.json 提取训练曲线
    2. 从 checkpoint 加载模型在测试集上生成 ROC/DET/FAR-FRR
    3. 生成所有实验的横向对比汇总表（JSON + PNG）
    4. 按实验名称归档，无需手动管理目录

用法：
    # 生成单个实验的图表
    python visualization_manager.py --exp_dir experiments/F0_fusion_full_simple

    # 生成所有实验的横向对比
    python visualization_manager.py --mode comparison

    # 生成特定实验的对比组
    python visualization_manager.py --mode comparison --exp_names F0_fusion_full_simple F1_fusion_face_only F2_fusion_fp_only

    # 仅提取指标（不生成图表）
    python visualization_manager.py --mode extract --output_json results/metrics.json

    # 一键全部生成
    python visualization_manager.py --mode all
"""

import os
import sys
import json
import glob
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import (
    plot_training_curves, plot_roc_curves, plot_det_curves, plot_far_frr_curves,
    load_config, set_seed, get_device, calculate_biometric_metrics,
    save_biometric_results
)
from core.datasets.fusion_dataset import FusionDataset
from core.models import create_model
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量（集中管理）
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
VIS_DIR = EXPERIMENTS_DIR / "visualizations"


# ─────────────────────────────────────────────────────────────────────────────
# 指标提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_history_metrics(history_path: str) -> Optional[dict]:
    """从 history.json 提取关键指标"""
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    if "epochs" not in data or not data["epochs"]:
        return None

    epochs = data["epochs"]

    def safe_mean(values):
        vals = [v for v in values if v is not None and v != 0]
        return round(float(np.mean(vals)), 4) if vals else None

    best_epoch = max(epochs, key=lambda x: x.get("val_rank1", 0))

    return {
        "experiment": data.get("experiment", "unknown"),
        "experiment_mode": data.get("experiment_mode", "unknown"),
        "fusion_method": data.get("fusion_method", "unknown"),
        "total_epochs": len(epochs),
        "best_epoch": best_epoch["epoch"],
        "best_val_rank1": best_epoch.get("val_rank1", 0),
        "best_val_rank5": best_epoch.get("val_rank5", 0),
        "best_val_rank10": best_epoch.get("val_rank10", 0),
        "best_val_rank20": best_epoch.get("val_rank20", 0),
        "best_val_eer": best_epoch.get("val_eer", 0),
        "final_train_loss": epochs[-1].get("train_loss"),
        "final_val_loss": epochs[-1].get("val_loss"),
        "final_val_rank1": epochs[-1].get("val_rank1", 0),
        "final_val_eer": epochs[-1].get("val_eer", 0),
        "train_loss_curve": [e.get("train_loss") for e in epochs],
        "val_loss_curve": [e.get("val_loss") for e in epochs],
        "val_rank1_curve": [e.get("val_rank1", 0) for e in epochs],
        "val_eer_curve": [e.get("val_eer", 0) for e in epochs],
        "lr_curve": [e.get("lr", 0) for e in epochs],
        "epochs_data": epochs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 单实验图表生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_experiment_vis(exp_dir: str, overwrite: bool = False):
    """为单个实验生成所有图表"""
    exp_name = Path(exp_dir).name
    out_dir = VIS_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(exp_dir) / "history.json"
    if not history_path.exists():
        print(f"[Skip] No history.json in {exp_dir}")
        return None

    print(f"[{exp_name}] Extracting metrics...")
    metrics = extract_history_metrics(str(history_path))
    if not metrics:
        return None

    # 保存提取后的 metrics
    metrics_out = out_dir / "metrics_extracted.json"
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 1. 训练曲线
    fig_path = out_dir / "training_curves.png"
    if overwrite or not fig_path.exists():
        epochs_range = list(range(1, len(metrics["train_loss_curve"]) + 1))
        plot_training_curves(
            metrics["train_loss_curve"],
            metrics["val_loss_curve"],
            metrics.get("train_acc_curve", [0] * len(epochs_range)),
            [e.get("train_acc", 0) for e in metrics["epochs_data"]],
            save_path=str(fig_path)
        )
        print(f"  saved: {fig_path.name}")

    # 2. 学习率曲线
    lr_path = out_dir / "learning_rate.png"
    if overwrite or not lr_path.exists():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs_range, metrics["lr_curve"], 'b-o', linewidth=2, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate Schedule — {exp_name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(lr_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: {lr_path.name}")

    # 3. Rank-1 / EER 演化曲线
    rank1_path = out_dir / "rank1_eer_evolution.png"
    if overwrite or not rank1_path.exists():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.plot(epochs_range, metrics["val_rank1_curve"], 'g-o', linewidth=2, markersize=3, label='Rank-1')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Rank-1 Accuracy')
        ax1.set_title('Validation Rank-1 Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(epochs_range, metrics["val_eer_curve"], 'r-o', linewidth=2, markersize=3, label='EER')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Equal Error Rate')
        ax2.set_title('Validation EER Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(rank1_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: {rank1_path.name}")

    # 4. 查找 checkpoint 生成生物特征图表
    ckpt_dir = Path(exp_dir) / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("best_*.pth")) if ckpt_dir.exists() else []
    if ckpt_files:
        try:
            _generate_biometric_plots(str(ckpt_files[0]), exp_dir, str(out_dir), exp_name)
        except Exception as e:
            print(f"  [Warning] Biometric plots failed: {e}")

    # 5. 生成摘要卡片
    _generate_summary_card(metrics, str(out_dir), exp_name)

    print(f"[{exp_name}] Visualization complete -> {out_dir}")
    return metrics


def _generate_biometric_plots(ckpt_path: str, exp_dir: str, out_dir: str, exp_name: str):
    """加载 checkpoint 并生成 ROC/DET/FAR-FRR 图表"""
    import torch

    config_paths = glob.glob(os.path.join(os.path.dirname(os.path.dirname(exp_dir)), "configs", "*.yaml"))
    config_paths += [os.path.join(project_root, "configs", "fusion_config.yaml")]
    config_path = next((p for p in config_paths if os.path.exists(p)), None)

    if not config_path:
        return

    try:
        config = load_config(config_path)
    except Exception:
        return

    device = get_device("auto")
    set_seed(42)

    # 创建数据集
    try:
        dataset = FusionDataset(
            config["paths"]["data_dir"],
            mode="val",
            face_image_size=config["data"].get("face_image_size", 224),
            fingerprint_image_size=config["data"].get("fingerprint_image_size", 224),
            mapping_file=config["paths"].get("mapping_file"),
        )
        val_loader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 16),
                                shuffle=False, num_workers=0)
    except Exception:
        return

    # 创建模型（简化：直接加载融合模型）
    try:
        model = create_model(
            "fusion",
            fusion_method="simple",
            face_embedding_dim=config["model"].get("face_embedding_dim", 512),
            fingerprint_embedding_dim=config["model"].get("fingerprint_embedding_dim", 512),
            num_classes=config["model"].get("num_classes", 500),
            fusion_dim=config["model"].get("fusion_dim", 256),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()
    except Exception:
        return

    # 推理收集结果
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            face = batch["face_image"].to(device)
            fp = batch["fingerprint_image"].to(device)
            labels = batch["label"].cpu().numpy()
            face_emb = model.face_proj(model.face_backbone(face))
            fp_emb = model.fp_proj(model.fp_backbone(fp))
            fused = model.extract_fused_features(face_emb, fp_emb)
            logits = model.classifier(F.normalize(fused, p=2, dim=1))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels)
            all_probs.extend(probs)

    if not all_labels:
        return

    biometric = calculate_biometric_metrics(all_labels, all_probs, num_classes=len(dataset.class_to_idx))

    # 保存生物特征指标
    bio_out = os.path.join(out_dir, "biometric_metrics.json")
    save_biometric_results(biometric, bio_out)

    # 生成图表
    roc_path = os.path.join(out_dir, "roc_curves.png")
    plot_roc_curves(biometric, save_path=roc_path)
    print(f"  saved: roc_curves.png")

    det_path = os.path.join(out_dir, "det_curves.png")
    plot_det_curves(biometric, save_path=det_path)
    print(f"  saved: det_curves.png")

    far_frr_path = os.path.join(out_dir, "far_frr_curves.png")
    plot_far_frr_curves(biometric, save_path=far_frr_path)
    print(f"  saved: far_frr_curves.png")


def _generate_summary_card(metrics: dict, out_dir: str, exp_name: str):
    """生成实验摘要卡片图"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    rows = [
        ["Experiment", exp_name],
        ["Mode", metrics.get("experiment_mode", "N/A")],
        ["Fusion Method", metrics.get("fusion_method", "N/A")],
        ["Best Epoch", f"{metrics['best_epoch']} / {metrics['total_epochs']}"],
        ["Best Val Rank-1", f"{metrics['best_val_rank1']:.4f}"],
        ["Best Val Rank-5", f"{metrics['best_val_rank5']:.4f}"],
        ["Best Val EER", f"{metrics['best_val_eer']:.4f}"],
        ["Final Val Rank-1", f"{metrics['final_val_rank1']:.4f}"],
        ["Final Train Loss", f"{metrics['final_train_loss']:.4f}"],
        ["Final Val Loss", f"{metrics['final_val_loss']:.4f}"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Key", "Value"],
        cellLoc='left',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F5F5F5')

    fig.savefig(os.path.join(out_dir, "summary_card.png"), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 横向对比
# ─────────────────────────────────────────────────────────────────────────────

def discover_experiments(base_dir: str = None) -> list:
    """自动发现所有实验目录"""
    if base_dir is None:
        base_dir = str(EXPERIMENTS_DIR)
    dirs = glob.glob(os.path.join(base_dir, "S1_*")) + glob.glob(os.path.join(base_dir, "F*_*"))
    return sorted(d for d in dirs if os.path.isdir(d) and (Path(d) / "history.json").exists())


def load_all_metrics(exp_dirs: list) -> dict:
    """加载所有实验的 metrics"""
    results = {}
    for exp_dir in exp_dirs:
        exp_name = Path(exp_dir).name
        history_path = Path(exp_dir) / "history.json"
        metrics = extract_history_metrics(str(history_path))
        if metrics:
            results[exp_name] = metrics
            print(f"  loaded: {exp_name}")
        else:
            print(f"  [Skip] {exp_name}: no valid history.json")
    return results


def generate_comparison(all_metrics: dict, output_dir: str = None):
    """生成横向对比图表"""
    if output_dir is None:
        output_dir = str(RESULTS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_tag = datetime.now().strftime("%Y%m%d")
    exp_names = sorted(all_metrics.keys())

    # ── 1. 汇总 JSON ─────────────────────────────────────────────────────────
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {}
    }
    for name, m in all_metrics.items():
        summary["experiments"][name] = {
            "experiment_mode": m.get("experiment_mode"),
            "fusion_method": m.get("fusion_method"),
            "best_epoch": m["best_epoch"],
            "total_epochs": m["total_epochs"],
            "best_val_rank1": round(m["best_val_rank1"], 4),
            "best_val_rank5": round(m["best_val_rank5"], 4),
            "best_val_rank10": round(m["best_val_rank10"], 4),
            "best_val_rank20": round(m["best_val_rank20"], 4),
            "best_val_eer": round(m["best_val_eer"], 4),
            "final_val_rank1": round(m["final_val_rank1"], 4),
            "final_val_eer": round(m["final_val_eer"], 4),
            "final_train_loss": round(m["final_train_loss"], 4) if m["final_train_loss"] else None,
            "final_val_loss": round(m["final_val_loss"], 4) if m["final_val_loss"] else None,
        }

    json_path = os.path.join(output_dir, f"comparison_summary_{date_tag}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Summary] saved: {json_path}")

    # ── 2. 数值对比表 (CSV) ──────────────────────────────────────────────────
    rows = []
    for name in exp_names:
        m = all_metrics[name]
        rows.append({
            "Experiment": name,
            "Mode": m.get("experiment_mode", ""),
            "Fusion": m.get("fusion_method", ""),
            "Best Epoch": m["best_epoch"],
            "Val Rank-1": round(m["best_val_rank1"], 4),
            "Val Rank-5": round(m["best_val_rank5"], 4),
            "Val Rank-10": round(m["best_val_rank10"], 4),
            "Val EER": round(m["best_val_eer"], 4),
            "Final Loss": round(m["final_val_loss"], 4) if m["final_val_loss"] else "",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"comparison_table_{date_tag}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[CSV] saved: {csv_path}")

    # ── 3. Val Rank-1 柱状对比图 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    rank1_vals = [all_metrics[n]["best_val_rank1"] for n in exp_names]
    eer_vals = [all_metrics[n]["best_val_eer"] for n in exp_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(exp_names)))

    bars1 = axes[0].bar(exp_names, rank1_vals, color=colors, edgecolor='white', linewidth=1)
    axes[0].set_ylabel('Rank-1 Accuracy')
    axes[0].set_title('Validation Rank-1 Accuracy Comparison')
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, rank1_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    bars2 = axes[1].bar(exp_names, eer_vals, color=colors, edgecolor='white', linewidth=1)
    axes[1].set_ylabel('Equal Error Rate (EER)')
    axes[1].set_title('Validation EER Comparison (Lower is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, eer_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"rank1_eer_comparison_{date_tag}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] saved: {fig_path}")

    # ── 4. 训练曲线对比 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors_line = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))

    for i, (name, m) in enumerate(all_metrics.items()):
        epochs = list(range(1, len(m["epochs_data"]) + 1))
        axes[0, 0].plot(epochs, m["train_loss_curve"], color=colors_line[i], label=name, linewidth=1.5)
        axes[0, 1].plot(epochs, m["val_loss_curve"], color=colors_line[i], label=name, linewidth=1.5)
        axes[1, 0].plot(epochs, m["val_rank1_curve"], color=colors_line[i], label=name, linewidth=1.5)
        axes[1, 1].plot(epochs, m["val_eer_curve"], color=colors_line[i], label=name, linewidth=1.5)

    axes[0, 0].set_title('Train Loss')
    axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_title('Val Rank-1')
    axes[1, 1].set_title('Val EER')

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    plt.suptitle('Training Curves Comparison Across Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    curves_path = os.path.join(output_dir, f"curves_comparison_{date_tag}.png")
    fig.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] saved: {curves_path}")

    # ── 5. 热力图对比 ─────────────────────────────────────────────────────────
    metric_names = ["Rank-1", "Rank-5", "Rank-10", "EER"]
    heat_data = []
    for name in exp_names:
        m = all_metrics[name]
        # EER 取负（越小越好，图表中越大表示越好）
        heat_data.append([
            m["best_val_rank1"],
            m["best_val_rank5"],
            m["best_val_rank10"],
            1 - m["best_val_eer"],
        ])

    heat_np = np.array(heat_data)
    fig, ax = plt.subplots(figsize=(8, max(4, len(exp_names) * 0.6 + 1)))
    im = ax.imshow(heat_np, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names)
    ax.set_title('Experiment Metrics Heatmap (Green = Better)')
    plt.colorbar(im, ax=ax, label='Score')

    for i in range(len(exp_names)):
        for j in range(len(metric_names)):
            val = heat_np[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color='white' if val > 0.8 else 'black', fontsize=9)

    plt.tight_layout()
    heat_path = os.path.join(output_dir, f"metrics_heatmap_{date_tag}.png")
    fig.savefig(heat_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] saved: {heat_path}")

    print(f"\n[Done] Comparison saved to: {output_dir}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="实验可视化管理器",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, default="all",
                       choices=["single", "comparison", "extract", "all"],
                       help="single: 生成单个实验图表 | comparison: 横向对比 | extract: 仅提取指标 | all: 全部")
    parser.add_argument("--exp_dir", type=str, default=None,
                       help="单个实验目录（mode=single 时使用）")
    parser.add_argument("--exp_names", type=str, nargs='*', default=None,
                       help="指定实验名称列表（mode=comparison 时使用）")
    parser.add_argument("--output_json", type=str, default=None,
                       help="extract 模式的输出 JSON 路径")
    parser.add_argument("--overwrite", action='store_true',
                       help="覆盖已存在的图表")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  实验可视化管理器")
    print("=" * 60)

    if args.mode in ("single", "all"):
        if args.exp_dir:
            dirs = [args.exp_dir]
        else:
            dirs = discover_experiments()
        for d in dirs:
            generate_single_experiment_vis(d, overwrite=args.overwrite)

    if args.mode in ("comparison", "all"):
        if args.exp_names:
            base = str(EXPERIMENTS_DIR)
            dirs = [os.path.join(base, n) for n in args.exp_names]
        else:
            dirs = discover_experiments()
        print(f"\n[Comparison] Loading {len(dirs)} experiments...")
        metrics = load_all_metrics(dirs)
        if metrics:
            generate_comparison(metrics)

    if args.mode == "extract":
        dirs = discover_experiments()
        print(f"[Extract] Loading {len(dirs)} experiments...")
        metrics = load_all_metrics(dirs)
        out_path = args.output_json or os.path.join(str(RESULTS_DIR),
                        f"extracted_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({k: {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, list) or kk in ("epochs_data",)}
                      for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)
        print(f"[Extract] saved: {out_path}")

    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
