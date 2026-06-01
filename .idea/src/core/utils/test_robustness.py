#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态融合模型增强鲁棒性测试

5个实验 + 1个干净基准，评估模型在查询端增强干扰下的1:N检索性能。

实验配置：
  0. baseline       - Gallery干净, Query干净（等效于训练时 test_epoch）
  1. fusion_face_aug    - 融合：仅人脸适度增强
  2. fusion_fp_aug      - 融合：仅指纹适度增强
  3. fusion_both_aug    - 融合：人脸+指纹同时适度增强
  4. fp_only_aug        - 单模态指纹：指纹适度增强
  5. face_only_aug      - 单模态人脸：人脸适度增强

Gallery: 干净图像（无增强）
Query:   增强干扰图像

数据集: test split（250 query, 150 gallery, 50人）
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils import set_seed, setup_logger, get_device
from core.datasets.fusion_dataset import FusionDataset
from inference.model_loader import ModelLoader

# ─── 增强参数 ────────────────────────────────────────────────────────────────

MODERATE_AUG = {
    "face": {
        "gaussian_blur_prob": 0.2,
        "gaussian_blur_kernel": 3,
        "random_erasing_prob": 0.1,
        "erasing_scale": (0.05, 0.15),
        "erasing_ratio": (0.3, 3.3),
    },
    "fp": {
        "gaussian_blur_prob": 0.15,
        "gaussian_blur_kernel": 3,
        "random_erasing_prob": 0.05,
        "erasing_scale": (0.02, 0.08),
        "erasing_ratio": (0.3, 3.3),
    }
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


# ─── Transform 组件 ─────────────────────────────────────────────────────────

class GaussianBlurPIL:
    """概率触发的高斯模糊（作用于 PIL Image）。"""
    def __init__(self, prob: float, kernel_size: int):
        import cv2
        self.prob = prob
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self._rnd = __import__('random')

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        import random
        if random.random() < self.prob:
            img_np = np.array(img)
            blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), 0)
            return Image.fromarray(blurred)
        return img


class ClahePIL:
    """CLAHE 作用于 PIL Image（返回 PIL Image）。"""
    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        import cv2
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        enhanced = self._clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


# ─── Transform 构建函数 ──────────────────────────────────────────────────────

def build_face_clean_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_face_aug_transform(image_size: int = 224) -> transforms.Compose:
    aug = MODERATE_AUG["face"]
    t = [transforms.Resize((image_size, image_size))]
    if aug.get("gaussian_blur_prob", 0) > 0:
        t.append(GaussianBlurPIL(
            prob=aug["gaussian_blur_prob"],
            kernel_size=aug["gaussian_blur_kernel"],
        ))
    t.append(transforms.ToTensor())
    re_prob = float(aug.get("random_erasing_prob", 0.0) or 0.0)
    if re_prob > 0:
        t.append(transforms.RandomErasing(
            p=re_prob,
            scale=aug.get("erasing_scale", (0.05, 0.15)),
            ratio=aug.get("erasing_ratio", (0.3, 3.3)),
        ))
    t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(t)


def build_fp_clean_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        ClahePIL(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_fp_aug_transform(image_size: int = 224) -> transforms.Compose:
    aug = MODERATE_AUG["fp"]
    t = [
        transforms.Resize((image_size, image_size)),
        ClahePIL(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE),
    ]
    if aug.get("gaussian_blur_prob", 0) > 0:
        t.append(GaussianBlurPIL(
            prob=aug["gaussian_blur_prob"],
            kernel_size=aug["gaussian_blur_kernel"],
        ))
    t.append(transforms.ToTensor())
    re_prob = float(aug.get("random_erasing_prob", 0.0) or 0.0)
    if re_prob > 0:
        t.append(transforms.RandomErasing(
            p=re_prob,
            scale=aug.get("erasing_scale", (0.02, 0.08)),
            ratio=aug.get("erasing_ratio", (0.3, 3.3)),
        ))
    t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(t)


# ─── 数据集包装 ──────────────────────────────────────────────────────────────

class FusionPairsDataset(Dataset):
    """从 FusionDataset 提取 test split 的 gallery/query 路径。"""

    def __init__(self,
                 fusion_dataset: FusionDataset,
                 split: Literal['gallery', 'query'] = 'query',
                 face_transform=None,
                 fp_transform=None):
        self.split = split

        g_paths = getattr(fusion_dataset, 'test_gallery_paths', None)
        g_labels = getattr(fusion_dataset, 'test_gallery_labels', None)
        q_paths = getattr(fusion_dataset, 'test_query_paths', None)
        q_labels = getattr(fusion_dataset, 'test_query_labels', None)

        if split == 'gallery':
            self.samples = [(fp[0], fp[1], lbl) for (fp, lbl) in zip(g_paths, g_labels)]
        else:
            self.samples = [(fp[0], fp[1], lbl) for (fp, lbl) in zip(q_paths, q_labels)]

        self.face_transform = face_transform
        self.fp_transform = fp_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, fp_path, label = self.samples[idx]

        try:
            face_img = Image.open(face_path).convert('RGB')
        except Exception:
            face_img = Image.new('RGB', (224, 224), (128, 128, 128))

        try:
            fp_img = Image.open(fp_path).convert('RGB')
        except Exception:
            fp_img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.face_transform:
            face_img = self.face_transform(face_img)
        if self.fp_transform:
            fp_img = self.fp_transform(fp_img)

        if isinstance(face_img, Image.Image):
            face_img = transforms.ToTensor()(face_img)
        if isinstance(fp_img, Image.Image):
            fp_img = transforms.ToTensor()(fp_img)

        return {
            'face_image': face_img,
            'fp_image': fp_img,
            'label': label,
        }


# ─── 评估函数 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_experiment(
    exp_name: str,
    exp_config: dict,
    face_model,
    fp_model,
    fusion_model,
    test_dataset: FusionDataset,
    device: torch.device,
    batch_size: int,
    logger,
) -> dict:
    """
    对单个实验进行 1:N 检索评估。

    exp_config 字段：
      model_type  : 'face_only' | 'fp_only' | 'fusion'
      face_aug_q : bool  Query时人脸是否增强
      fp_aug_q   : bool  Query时指纹是否增强
      use_face   : bool  融合时是否使用人脸模态
      use_fp     : bool  融合时是否使用指纹模态
    """
    face_aug_q = exp_config.get("face_aug_q", False)
    fp_aug_q = exp_config.get("fp_aug_q", False)
    model_type = exp_config.get("model_type", "fusion")
    use_face = exp_config.get("use_face", True)
    use_fp = exp_config.get("use_fp", True)
    description = exp_config.get("description", "")

    log = logger.info if logger else print
    log(f"\n[{'='*60}]")
    log(f"[{exp_name}] {description}")
    log(f"  face_aug={face_aug_q}, fp_aug={fp_aug_q}, model={model_type}")

    # ── Transform 组合 ──────────────────────────────────────────────────
    face_t_gallery = build_face_clean_transform()
    fp_t_gallery = build_fp_clean_transform()
    face_t_query = build_face_aug_transform() if face_aug_q else build_face_clean_transform()
    fp_t_query = build_fp_aug_transform() if fp_aug_q else build_fp_clean_transform()

    # ── DataLoader ──────────────────────────────────────────────────────
    gallery_ds = FusionPairsDataset(test_dataset, split='gallery',
                                    face_transform=face_t_gallery,
                                    fp_transform=fp_t_gallery)
    query_ds = FusionPairsDataset(test_dataset, split='query',
                                  face_transform=face_t_query,
                                  fp_transform=fp_t_query)

    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)
    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    log(f"  Gallery={len(gallery_ds)}, Query={len(query_ds)}")

    # ── 提取 Gallery 特征 ────────────────────────────────────────────────
    gallery_feats = []
    gallery_labels = []

    for batch in tqdm(gallery_loader, desc=f"[{exp_name}] Gallery", leave=False):
        face = batch['face_image'].to(device)
        fp_img = batch['fp_image'].to(device)

        # backbone 输出原始特征（不做归一化，与 FusionTrainer 一致）
        face_feat = face_model.extract_features(face).detach()
        fp_feat = fp_model.extract_features(fp_img).detach()

        if model_type == 'fusion':
            # 传入投影层前归一化（与 fusion_model.extract_fused_features 内部一致）
            face_feat = F.normalize(face_feat, p=2, dim=1)
            fp_feat = F.normalize(fp_feat, p=2, dim=1)

            if not use_face:
                face_feat = torch.zeros_like(face_feat)
            if not use_fp:
                fp_feat = torch.zeros_like(fp_feat)

            feat = fusion_model.extract_fused_features(face_feat, fp_feat)
        elif model_type == 'face_only':
            feat = F.normalize(face_feat, p=2, dim=1)
        else:
            feat = F.normalize(fp_feat, p=2, dim=1)

        gallery_feats.append(feat.cpu().numpy())
        gallery_labels.extend(batch['label'].numpy().tolist())

    gallery_feats = np.concatenate(gallery_feats, axis=0)
    gallery_labels = np.array(gallery_labels)

    # ── 提取 Query 特征 ─────────────────────────────────────────────────
    query_feats = []
    query_labels = []

    for batch in tqdm(query_loader, desc=f"[{exp_name}] Query", leave=False):
        face = batch['face_image'].to(device)
        fp_img = batch['fp_image'].to(device)

        face_feat = face_model.extract_features(face).detach()
        fp_feat = fp_model.extract_features(fp_img).detach()

        if model_type == 'fusion':
            face_feat = F.normalize(face_feat, p=2, dim=1)
            fp_feat = F.normalize(fp_feat, p=2, dim=1)

            if not use_face:
                face_feat = torch.zeros_like(face_feat)
            if not use_fp:
                fp_feat = torch.zeros_like(fp_feat)

            feat = fusion_model.extract_fused_features(face_feat, fp_feat)
        elif model_type == 'face_only':
            feat = F.normalize(face_feat, p=2, dim=1)
        else:
            feat = F.normalize(fp_feat, p=2, dim=1)

        query_feats.append(feat.cpu().numpy())
        query_labels.extend(batch['label'].numpy().tolist())

    query_feats = np.concatenate(query_feats, axis=0)
    query_labels = np.array(query_labels)

    # ── 1:N 检索 ───────────────────────────────────────────────────────
    similarities = np.dot(query_feats, gallery_feats.T)
    ranks = np.argsort(-similarities, axis=1)

    rank1 = rank5 = rank10 = rank20 = 0
    for i, label in enumerate(query_labels):
        top = gallery_labels[ranks[i]]
        if top[0] == label: rank1 += 1
        if label in top[:5]: rank5 += 1
        if label in top[:10]: rank10 += 1
        if label in top[:20]: rank20 += 1

    n = len(query_labels)
    rank1 /= n; rank5 /= n; rank10 /= n; rank20 /= n

    # ── EER & FAR@0.1% ──────────────────────────────────────────────────
    pos_scores, neg_scores = [], []
    for i, label in enumerate(query_labels):
        row = similarities[i]
        same = row[gallery_labels == label]
        diff = row[gallery_labels != label]
        if len(same) > 0: pos_scores.append(same.max())
        if len(diff) > 0: neg_scores.append(diff.max())

    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    all_s = np.concatenate([pos, neg])
    all_l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, thresholds = roc_curve(all_l, all_s)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2
    eer_th = thresholds[eer_idx]

    far_t_idx = np.searchsorted(fpr_arr, 0.001)
    far_t_idx = min(far_t_idx, len(thresholds) - 1)
    frr_at_far001 = float(fnr_arr[far_t_idx])

    log(f"  Rank-1={rank1:.4f} ({int(rank1*n)}/{n})")
    log(f"  Rank-5={rank5:.4f}  Rank-10={rank10:.4f}  Rank-20={rank20:.4f}")
    log(f"  EER={eer:.4f} (th={eer_th:.4f})")
    log(f"  FAR@0.1% FRR={frr_at_far001:.4f}")

    return {
        "experiment": exp_name,
        "description": description,
        "model_type": model_type,
        "face_aug_q": face_aug_q,
        "fp_aug_q": fp_aug_q,
        "use_face": use_face,
        "use_fp": use_fp,
        "rank1": float(rank1),
        "rank5": float(rank5),
        "rank10": float(rank10),
        "rank20": float(rank20),
        "eer": float(eer),
        "eer_th": float(eer_th),
        "far_001_frr": frr_at_far001,
        "n_gallery": len(gallery_feats),
        "n_query": n,
        "augmentation": "moderate" if (face_aug_q or fp_aug_q) else "clean",
    }


# ─── 可视化 ─────────────────────────────────────────────────────────────────

def plot_summary_table(results: list, baseline: dict, output_path: Path):
    """生成汇总表格图（PNG）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # 计算 delta
    for r in results:
        r['delta_rank1'] = r['rank1'] - baseline['rank1']
        r['delta_eer'] = r['eer'] - baseline['eer']

    fig, ax = plt.subplots(figsize=(16, max(4, len(results) * 0.6 + 2)))
    ax.axis('off')

    # 列定义
    col_labels = [
        'Experiment', 'Rank-1', 'Delta R1', 'Rank-5', 'Rank-10',
        'Rank-20', 'EER', 'Delta EER', 'FAR@0.1%FRR',
    ]

    # 颜色方案
    header_color = '#2C3E50'
    baseline_color = '#E8F5E9'
    row_colors = ['#FFFFFF', '#F5F5F5']

    table_data = []
    for i, r in enumerate(results):
        delta_r1 = r['delta_rank1']
        delta_e = r['delta_eer']
        delta_r1_str = f"{delta_r1:+.4f}" if delta_r1 is not None else "—"
        delta_e_str = f"{delta_e:+.4f}" if delta_e is not None else "—"

        table_data.append([
            r['experiment'],
            f"{r['rank1']:.4f}",
            delta_r1_str,
            f"{r['rank5']:.4f}",
            f"{r['rank10']:.4f}",
            f"{r['rank20']:.4f}",
            f"{r['eer']:.4f}",
            delta_e_str,
            f"{r['far_001_frr']:.4f}" if r.get('far_001_frr') is not None else "—",
        ])

    col_widths = [0.22, 0.08, 0.09, 0.08, 0.09, 0.08, 0.08, 0.09, 0.12]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # 表头样式
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')

    # 行样式 + 高亮
    for i in range(len(results)):
        bg = baseline_color if 'baseline' in results[i]['experiment'] else row_colors[i % 2]
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)

            # delta 列着色
            if j == 2 and i > 0:  # Δ Rank-1
                v = results[i]['delta_rank1']
                if v is not None:
                    if v >= -0.01:
                        cell.set_facecolor('#C8E6C9')
                    elif v >= -0.03:
                        cell.set_facecolor('#FFF9C4')
                    else:
                        cell.set_facecolor('#FFCDD2')
            if j == 7 and i > 0:  # Δ EER
                v = results[i]['delta_eer']
                if v is not None:
                    if v <= 0.01:
                        cell.set_facecolor('#C8E6C9')
                    elif v <= 0.03:
                        cell.set_facecolor('#FFF9C4')
                    else:
                        cell.set_facecolor('#FFCDD2')

    # Title
    fig.suptitle('Fusion Model Robustness Test Results (test split)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[Saved] Summary table: {output_path}")


def plot_comparison_charts(results: list, baseline: dict, output_dir: Path):
    """Generate comparison bar charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for r in results:
        r['delta_rank1'] = r['rank1'] - baseline['rank1']
        r['delta_eer'] = r['eer'] - baseline['eer']

    names = [r['experiment'] for r in results]
    rank1s = [r['rank1'] * 100 for r in results]
    deltas_r1 = [r['delta_rank1'] * 100 for r in results]
    eers = [r['eer'] * 100 for r in results]
    deltas_eer = [r['delta_eer'] * 100 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Rank-1
    colors = ['#4CAF50' if 'baseline' in n else '#2196F3' for n in names]
    bars1 = axes[0, 0].bar(names, rank1s, color=colors, edgecolor='white')
    axes[0, 0].set_title('Rank-1 Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(80, 102)
    axes[0, 0].set_ylabel('%')
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].axhline(y=baseline['rank1'] * 100, color='red', linestyle='--', alpha=0.6, label='baseline')
    for bar, v in zip(bars1, rank1s):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # Delta Rank-1
    colors2 = ['#C8E6C9' if v >= -1 else ('#FFF9C4' if v >= -3 else '#FFCDD2') for v in deltas_r1]
    bars2 = axes[0, 1].bar(names, deltas_r1, color=colors2, edgecolor='white')
    axes[0, 1].set_title('Rank-1 Drop (Delta %)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Delta %')
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, v in zip(bars2, deltas_r1):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, v - 0.3, f'{v:.1f}', ha='center', va='top', fontsize=9)

    # EER
    axes[1, 0].bar(names, eers, color='#FF9800', edgecolor='white')
    axes[1, 0].set_title('EER (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, max(eers) * 1.5)
    axes[1, 0].set_ylabel('%')
    axes[1, 0].tick_params(axis='x', rotation=30)
    axes[1, 0].axhline(y=baseline['eer'] * 100, color='red', linestyle='--', alpha=0.6, label='baseline')
    for i, (name, v) in enumerate(zip(names, eers)):
        axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # Delta EER
    axes[1, 1].bar(names, deltas_eer, color='#CE93D8', edgecolor='white')
    axes[1, 1].set_title('EER Change (Delta %)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Delta %')
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate(deltas_eer):
        axes[1, 1].text(i, v + 0.1 * (1 if v >= 0 else -1), f'{v:+.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

    fig.suptitle('Robustness Test: Baseline vs Augmentation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'comparison_chart.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Comparison chart: {out}")


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多模态融合模型增强鲁棒性测试")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_visualization", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir) if args.output_dir else \
        project_root / "outputs" / "robustness_test" / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        experiment_name=f"robustness_test_{timestamp}",
        log_dir=str(output_dir),
        level=logging.INFO,
        logger_name="RobustnessTest"
    )

    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Timestamp: {timestamp}")

    # ── 配置 ────────────────────────────────────────────────────────────
    face_ckpt = project_root / "outputs" / "face" / "checkpoints" / "face_recognition" / "face" / "best.pth"
    fp_ckpt = project_root / "outputs" / "fingerprint" / "checkpoints" / "fingerprint_recognition" / "fingerprint" / "best.pth"
    fusion_ckpt = project_root / "outputs" / "fusion" / "fusion_adaptive" / "best.pth"

    # ── 加载 test split ─────────────────────────────────────────────────
    from core.utils import load_config
    config_path = project_root / "configs" / "fusion_config.yaml"
    config = load_config(str(config_path))

    logger.info("Loading test split...")
    test_dataset = FusionDataset(
        face_data_dir=str(config['paths']['face_data_dir']),
        fingerprint_data_dir=str(config['paths']['fingerprint_data_dir']),
        mapping_file=str(config['paths'].get('mapping_file')),
        mode='test',
        face_image_size=int(config['data']['face_image_size']),
        fingerprint_image_size=int(config['data']['fingerprint_image_size']),
        augment=False,
        split_ratio=config["data"].get("split_ratio", 0.8),
        gallery_per_person=config["data"].get("gallery_per_person", 3),
        seed=args.seed,
    )

    g_n = len(getattr(test_dataset, 'test_gallery_paths', []))
    q_n = len(getattr(test_dataset, 'test_query_paths', []))
    logger.info(f"Test split: Gallery={g_n}, Query={q_n}")

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("Loading models...")

    loader = ModelLoader(
        device=str(device),
        num_classes=500,
        embedding_dim=512,
        fusion_dim=256,
    )
    face_model = loader.load_face_model(str(face_ckpt))
    fp_model = loader.load_fingerprint_model(str(fp_ckpt))

    from core.models import FusionModel
    fusion_model = FusionModel(
        fusion_strategy='adaptive',
        face_embedding_dim=512,
        fingerprint_embedding_dim=512,
        num_classes=500,
        fusion_dim=256,
        dropout_rate=0.3,
        use_arcface=False,
    ).to(device)
    fusion_model.eval()

    if fusion_ckpt.exists():
        ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        fusion_model.load_state_dict(state, strict=False)
        logger.info(f"Fusion model loaded: {fusion_ckpt}")
    else:
        logger.error(f"Fusion checkpoint not found: {fusion_ckpt}")
        return

    # ── 实验配置 ─────────────────────────────────────────────────────────
    EXPERIMENTS = {
        "0_baseline": {
            "model_type": "fusion",
            "face_aug_q": False,
            "fp_aug_q": False,
            "use_face": True,
            "use_fp": True,
            "description": "Gallery=clean, Query=clean (baseline)",
        },
        "1_fusion_face_aug": {
            "model_type": "fusion",
            "face_aug_q": True,
            "fp_aug_q": False,
            "use_face": True,
            "use_fp": True,
            "description": "Fusion: face augmentation only",
        },
        "2_fusion_fp_aug": {
            "model_type": "fusion",
            "face_aug_q": False,
            "fp_aug_q": True,
            "use_face": True,
            "use_fp": True,
            "description": "Fusion: fingerprint augmentation only",
        },
        "3_fusion_both_aug": {
            "model_type": "fusion",
            "face_aug_q": True,
            "fp_aug_q": True,
            "use_face": True,
            "use_fp": True,
            "description": "Fusion: both face+fp augmentation",
        },
        "4_fp_only_aug": {
            "model_type": "fp_only",
            "face_aug_q": False,
            "fp_aug_q": True,
            "use_face": False,
            "use_fp": True,
            "description": "Fingerprint only: fp augmentation",
        },
        "5_face_only_aug": {
            "model_type": "face_only",
            "face_aug_q": True,
            "fp_aug_q": False,
            "use_face": True,
            "use_fp": False,
            "description": "Face only: face augmentation",
        },
    }

    # ── 执行实验 ─────────────────────────────────────────────────────────
    results = []
    for exp_name, exp_cfg in EXPERIMENTS.items():
        result = evaluate_experiment(
            exp_name=exp_name,
            exp_config=exp_cfg,
            face_model=face_model,
            fp_model=fp_model,
            fusion_model=fusion_model,
            test_dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
            logger=logger,
        )
        results.append(result)

    # ── 基准值 ───────────────────────────────────────────────────────────
    baseline = next(r for r in results if 'baseline' in r['experiment'])
    logger.info(f"\n{'='*70}")
    logger.info(f"基准（Gallery干净, Query干净）: Rank-1={baseline['rank1']:.4f}  EER={baseline['eer']:.4f}")
    logger.info(f"{'='*70}")

    # ── 打印汇总表 ───────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"{'Fusion Model Robustness Test Results (val split)':^110}")
    print(f"{'='*110}")
    hdr = f"{'Experiment':<26} {'Rank-1':>8} {'Delta R1':>10} {'Rank-5':>8} {'Rank-10':>9} {'Rank-20':>9} {'EER':>8} {'Delta EER':>10} {'FAR@0.1%FRR':>12}"
    print(hdr)
    print("-" * 110)
    for r in results:
        delta_r1 = r['rank1'] - baseline['rank1']
        delta_e = r['eer'] - baseline['eer']
        far = f"{r['far_001_frr']:.4f}" if r.get('far_001_frr') is not None else "N/A"
        print(f"{r['experiment']:<26} {r['rank1']:>8.4f} {delta_r1:>+10.4f} "
              f"{r['rank5']:>8.4f} {r['rank10']:>9.4f} {r['rank20']:>9.4f} "
              f"{r['eer']:>8.4f} {delta_e:>+10.4f} {far:>12}")
    print("=" * 110)

    # ── 分析结论 ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("Analysis")
    logger.info(f"{'='*70}")

    by_name = {r['experiment']: r for r in results}

    def delta(name):
        return by_name[name]['rank1'] - baseline['rank1']

    logger.info(f"  Baseline (fusion clean):        Rank-1={baseline['rank1']:.4f}  EER={baseline['eer']:.4f}")
    logger.info(f"  Fusion+face aug:               {by_name['1_fusion_face_aug']['rank1']:.4f}  delta={delta('1_fusion_face_aug'):+.4f}")
    logger.info(f"  Fusion+fp aug:                 {by_name['2_fusion_fp_aug']['rank1']:.4f}  delta={delta('2_fusion_fp_aug'):+.4f}")
    logger.info(f"  Fusion+both aug:               {by_name['3_fusion_both_aug']['rank1']:.4f}  delta={delta('3_fusion_both_aug'):+.4f}")
    logger.info(f"  FP only+aug:                   {by_name['4_fp_only_aug']['rank1']:.4f}  delta={delta('4_fp_only_aug'):+.4f}")
    logger.info(f"  Face only+aug:                 {by_name['5_face_only_aug']['rank1']:.4f}  delta={delta('5_face_only_aug'):+.4f}")
    logger.info(f"")
    logger.info(f"  Fusion(both) vs FP only(both): {by_name['3_fusion_both_aug']['rank1']:.4f} - {by_name['4_fp_only_aug']['rank1']:.4f} = {delta('3_fusion_both_aug')-delta('4_fp_only_aug'):+.4f}")
    logger.info(f"  Fusion(both) vs Face only(both): {by_name['3_fusion_both_aug']['rank1']:.4f} - {by_name['5_face_only_aug']['rank1']:.4f} = {delta('3_fusion_both_aug')-delta('5_face_only_aug'):+.4f}")
    logger.info(f"")

    if delta('1_fusion_face_aug') > delta('2_fusion_fp_aug'):
        logger.info(f"  Fingerprint contributes more (face aug: fusion loses only {delta('1_fusion_face_aug'):+.4f})")
    else:
        logger.info(f"  Face contributes more (fp aug: fusion loses only {delta('2_fusion_fp_aug'):+.4f})")

    # ── 保存结果 ─────────────────────────────────────────────────────────
    results_data = {
        "timestamp": timestamp,
        "test_split_info": {
            "gallery_size": g_n,
            "query_size": q_n,
        },
        "baseline": baseline,
        "experiments": results,
        "augmentation_params": MODERATE_AUG,
    }

    results_path = output_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {results_path}")

    # ── 可视化 ──────────────────────────────────────────────────────────
    if not args.skip_visualization:
        plot_summary_table(results, baseline, output_dir / "results_table.png")
        plot_comparison_charts(results, baseline, output_dir)

    logger.info(f"\n输出目录: {output_dir}")
    logger.info("Done.")

    return output_dir


if __name__ == "__main__":
    main()
