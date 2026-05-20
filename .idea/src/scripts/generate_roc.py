#!/usr/bin/env python
"""
生成 ROC 曲线图（人脸 / 指纹 / 融合消融实验通用）
加载 best.pth → 提取 gallery/query 特征 → 计算余弦相似度 → 绘制 ROC 曲线。

用法:
    python scripts/generate_roc.py --modality face
    python scripts/generate_roc.py --modality fingerprint
    python scripts/generate_roc.py --modality fusion_full
    python scripts/generate_roc.py --modality fusion_fp_ablation
    python scripts/generate_roc.py --modality fusion_face_ablation
    python scripts/generate_roc.py --all
"""
import os, sys, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root  = os.path.dirname(os.path.dirname(script_dir))   # .idea/
src_root   = os.path.join(proj_root, "src")
sys.path.insert(0, src_root)

from core.utils import load_config, set_seed, get_device
from core.models import create_model

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']

# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt(model, path, device):
    """加载 checkpoint，处理 model_state 嵌套和 _classifier → classifier 重命名。"""
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get('model_state', ckpt)
    renamed = {}
    for k, v in state.items():
        new_k = k.replace('_classifier.', 'classifier.')
        renamed[new_k] = v
    missing, _ = model.load_state_dict(renamed, strict=False)
    non_cls = [m for m in missing if 'classifier' not in m]
    if non_cls:
        print(f"  Missing (non-classifier): {non_cls[:3]}")


def build_transform(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 特征提取（返回 (embeddings, valid_indices)）
# ─────────────────────────────────────────────────────────────────────────────

def extract_single(model, paths, transform, device, batch_size=32):
    """单模态特征提取，返回 (torch.Tensor, list)。"""
    model.eval()
    embs, valid = [], []

    batch_imgs, batch_idx = [], []
    for i, p in enumerate(tqdm(paths, desc="  Extract")):
        try:
            img = transform(Image.open(p).convert('RGB'))
        except Exception:
            img = transform(Image.new('RGB', (224, 224), (128, 128, 128)))
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) == batch_size:
            batch_t = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                emb = model.extract_features(batch_t)
            ok = ~(torch.isnan(emb).any(dim=1) | torch.isinf(emb).any(dim=1))
            for j, good in enumerate(ok.tolist()):
                if good:
                    embs.append(emb[j].cpu())
                    valid.append(batch_idx[j])
            batch_imgs, batch_idx = [], []

    if batch_imgs:
        batch_t = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = model.extract_features(batch_t)
        ok = ~(torch.isnan(emb).any(dim=1) | torch.isinf(emb).any(dim=1))
        for j, good in enumerate(ok.tolist()):
            if good:
                embs.append(emb[j].cpu())
                valid.append(batch_idx[j])

    return (torch.stack(embs) if embs else torch.empty(0, 512)), valid


def extract_fusion(face_m, fp_m, fusion_m, face_ps, fp_ps,
                   face_t, fp_t, device, batch_size=32, abl=None):
    """融合特征提取，返回 (torch.Tensor, list)。"""
    face_m.eval(); fp_m.eval(); fusion_m.eval()
    if hasattr(fusion_m, 'set_ablation'):
        fusion_m.set_ablation(abl)

    embs, valid = [], []
    batch_fp, batch_face, batch_idx = [], [], []

    n = len(face_ps)
    for i in tqdm(range(n), desc="  Extract fusion"):
        try:
            fi = face_t(Image.open(face_ps[i]).convert('RGB'))
        except Exception:
            fi = face_t(Image.new('RGB', (224, 224), (128, 128, 128)))
        try:
            pi = fp_t(Image.open(fp_ps[i]).convert('RGB'))
        except Exception:
            pi = fp_t(Image.new('RGB', (224, 224), (128, 128, 128)))
        batch_face.append(fi); batch_fp.append(pi); batch_idx.append(i)
        if len(batch_face) == batch_size:
            ft = torch.stack(batch_face).to(device)
            pt = torch.stack(batch_fp).to(device)
            with torch.no_grad():
                ff = face_m.extract_features(ft)
                pf = fp_m.extract_features(pt)
                fu = fusion_m.extract_fused_features(ff, pf)
                fu = F.normalize(fu.float(), p=2, dim=1)
            ok = ~(torch.isnan(fu).any(dim=1) | torch.isinf(fu).any(dim=1))
            for j, good in enumerate(ok.tolist()):
                if good:
                    embs.append(fu[j].cpu())
                    valid.append(batch_idx[j])
            batch_face, batch_fp, batch_idx = [], [], []

    if batch_face:
        ft = torch.stack(batch_face).to(device)
        pt = torch.stack(batch_fp).to(device)
        with torch.no_grad():
            ff = face_m.extract_features(ft)
            pf = fp_m.extract_features(pt)
            fu = fusion_m.extract_fused_features(ff, pf)
            fu = F.normalize(fu.float(), p=2, dim=1)
        ok = ~(torch.isnan(fu).any(dim=1) | torch.isinf(fu).any(dim=1))
        for j, good in enumerate(ok.tolist()):
            if good:
                embs.append(fu[j].cpu())
                valid.append(batch_idx[j])

    return (torch.stack(embs) if embs else torch.empty(0, 512)), valid


# ─────────────────────────────────────────────────────────────────────────────
# 分数收集 & 绘图
# ─────────────────────────────────────────────────────────────────────────────

def collect_scores(q_emb, g_emb, q_lbl, g_lbl, seed=42):
    rng = np.random.RandomState(seed)
    pos, neg = [], []
    sim = q_emb @ g_emb.T
    for i in range(len(q_lbl)):
        same = (g_lbl == q_lbl[i])
        diff = ~same
        if same.sum() > 0:
            pos.extend(np.sort(sim[i, same])[-5:].tolist())
        didx = np.where(diff)[0]
        n_neg = min(3, len(didx))
        if n_neg > 0:
            neg.extend(sim[i, rng.choice(didx, n_neg, replace=False)].tolist())
    return np.array(pos), np.array(neg)


def _eer(fpr, tpr, th):
    fnr = 1 - tpr
    return (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2


def plot_single(pos, neg, label, color, out_path):
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    fpr, tpr, th = roc_curve(y, s)
    auc_v = auc(fpr, tpr)
    eer_v = _eer(fpr, tpr, th)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{label}  (AUC={auc_v:.4f}, EER={eer_v:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title(f'ROC Curve — {label}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  → {Path(out_path).name}  AUC={auc_v:.4f}  EER={eer_v:.4f}")
    return auc_v, eer_v


def plot_combined(all_res, out_path):
    plt.figure(figsize=(9, 7))
    for label, pos, neg, color in all_res:
        y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        s = np.concatenate([pos, neg])
        fpr, tpr, th = roc_curve(y, s)
        auc_v = auc(fpr, tpr)
        eer_v = _eer(fpr, tpr, th)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{label}  (AUC={auc_v:.4f}, EER={eer_v:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random')
    plt.xlabel('FPR', fontsize=13); plt.ylabel('TPR', fontsize=13)
    plt.title('ROC Curves — All Modalities', fontsize=14)
    plt.legend(loc='lower right', fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  → {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# 单模态（face / fingerprint）
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_ckpt(search_dir, dev):
    """Recursively find the best .pth checkpoint in search_dir."""
    if not os.path.isdir(search_dir):
        return None
    best_file, best_acc = None, -1.0
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if not f.endswith('.pth') or 'best' not in f.lower():
                continue
            try:
                ckpt = torch.load(os.path.join(root, f), map_location=dev, weights_only=False)
                acc = float(ckpt.get('val_acc', ckpt.get('rank1', -1)))
                if acc > best_acc:
                    best_acc = acc
                    best_file = os.path.join(root, f)
            except Exception:
                pass
    return best_file, best_acc if best_file else (None, -1.0)


def run_single(mod):
    print(f"\n{'='*50}  {mod}  {'='*50}")
    dev = get_device()
    cfg_dir = Path(src_root) / "configs"

    if mod == "face":
        from core.datasets.face_dataset import FaceDataset
        cfg_path = cfg_dir / "face_config.yaml"
        name = "Face Recognition"
        search_dir = Path(src_root) / "outputs" / "face"
    else:
        from core.datasets.fingerprint_dataset import FingerprintDataset
        cfg_path = cfg_dir / "fingerprint_config.yaml"
        name = "Fingerprint Recognition"
        search_dir = Path(src_root) / "outputs" / "fingerprint"

    ckpt_path, best_acc = _find_best_ckpt(str(search_dir), dev)
    if ckpt_path:
        print(f"  Found best: {ckpt_path}  (val_acc={best_acc:.4f})")
    else:
        ckpt_path = str(Path(src_root) / "outputs" / "face" / "checkpoints" / "face_recognition" / "face" / "best.pth")
        print(f"  [WARN] No checkpoint found, using fallback: {ckpt_path}")

    cfg = load_config(str(cfg_path))
    set_seed(cfg["misc"]["seed"])

    # 模型
    mc = cfg["model"]
    model = create_model(
        modality=mod, model_type="facenet" if mod == "face" else "fingerprint_net",
        num_classes=mc["num_classes"],
        embedding_dim=mc.get("embedding_dim", 512),
        pretrained=True, dropout_rate=mc.get("dropout_rate", 0.5),
        spatial_attention=mc.get("spatial_attention", False),
    ).to(dev)

    if not os.path.exists(ckpt_path):
        print(f"  [WARN] Not found: {ckpt_path}"); return None
    _load_ckpt(model, ckpt_path, dev)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    # 数据集
    seed = cfg["misc"]["seed"]
    sr   = cfg["data"]["split_ratio"]
    tr   = cfg["data"]["test_split_ratio"]
    gp   = cfg["data"].get("gallery_per_person", 3)

    if mod == "face":
        ds = FaceDataset(
            data_dir=str(Path(src_root) / cfg["paths"]["modality_data_dir"]),
            mode='val', image_size=cfg["data"]["image_size"],
            augment=False, gallery_per_person=gp,
            val_split_ratio=sr, test_split_ratio=tr, seed=seed,
        )
        sz = cfg["data"]["image_size"]
    else:
        sz = cfg["data"].get("fingerprint_image_size", cfg["data"].get("image_size", 224))
        ds = FingerprintDataset(
            data_dir=str(Path(src_root) / cfg["paths"]["modality_data_dir"]),
            mode='val', image_size=sz,
            augment=False, gallery_per_person=gp,
            split_ratio=sr, test_split_ratio=tr, seed=seed,
        )

    g_ps   = ds.val_gallery_paths
    g_lbl  = np.array(ds.val_gallery_labels)
    q_ps   = ds.val_query_paths
    q_lbl  = np.array(ds.val_query_labels)
    trans  = build_transform(sz)
    print(f"  Gallery: {len(g_ps)} | Query: {len(q_ps)}")

    # 特征
    g_emb, g_v = extract_single(model, g_ps, trans, dev)
    q_emb, q_v = extract_single(model, q_ps, trans, dev)
    g_lbl = g_lbl[g_v]; q_lbl = q_lbl[q_v]
    print(f"  Gallery emb: {g_emb.shape} | Query emb: {q_emb.shape}")
    if g_emb.shape[0] == 0 or q_emb.shape[0] == 0:
        print("  [WARN] All embeddings are NaN — check model loading"); return None

    # 分数 & 绘图
    pos, neg = collect_scores(q_emb.numpy(), g_emb.numpy(), q_lbl, g_lbl)
    print(f"  Positive: {len(pos)} | Negative: {len(neg)}")
    out_dir = Path(src_root) / "results" / "roc_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    auc_v, eer_v = plot_single(pos, neg, name, COLORS[0], str(out_dir / f"roc_{mod}.png"))
    return pos, neg, name, COLORS[0]


# ─────────────────────────────────────────────────────────────────────────────
# 融合模态
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion(mod):
    print(f"\n{'='*50}  {mod}  {'='*50}")
    dev = get_device()
    cfg_dir = Path(src_root) / "configs"
    cfg_path = cfg_dir / "fusion_config.yaml"

    exp_map = {
        "fusion_full":          ("fusion_adaptive",                               "Fusion (Full)"),
        "fusion_fp_ablation":   ("fusion_adaptive_fp_ablation",                   "FP Ablation (Face Zeroed)"),
        "fusion_face_ablation": ("fusion_adaptive_face_ablation",                 "Face Ablation (FP Zeroed)"),
    }
    exp_name, name = exp_map[mod]
    fusion_ckpt = Path(src_root) / "outputs" / "fusion" / exp_name / "best.pth"

    cfg = load_config(str(cfg_path))
    set_seed(cfg["misc"]["seed"])

    # 融合模型
    mc = cfg["model"]
    fusion_m = create_model(
        modality="fusion", num_classes=mc["num_classes"],
        face_embedding_dim=mc["face_embedding_dim"],
        fingerprint_embedding_dim=mc["fingerprint_embedding_dim"],
        fusion_dim=mc["fusion_dim"], fusion_method="adaptive",
    ).to(dev)
    if not fusion_ckpt.exists():
        print(f"  [WARN] Not found: {fusion_ckpt}"); return None
    _load_ckpt(fusion_m, fusion_ckpt, dev)
    print(f"  Fusion checkpoint: {fusion_ckpt.name}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 辅助：尝试从 fusion checkpoint 加载 backbones（jointly trained 版本）
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_backbones_from_fusion_ckpt(face_m, fp_m, fusion_ckpt_path, dev):
        """从 fusion checkpoint 提取 backbones（jointly trained 版本）"""
        try:
            ckpt = torch.load(fusion_ckpt_path, map_location=dev, weights_only=False)
            model_state = ckpt.get('model_state', ckpt)

            face_state = ckpt.get('face_model_state', {})
            if not face_state:
                face_state = {k:v for k,v in model_state.items()
                              if 'face' in k.lower() or 'backbone' in k.lower()}
            fp_state = ckpt.get('fp_model_state', {})
            if not fp_state:
                fp_state = {k:v for k,v in model_state.items()
                             if 'fp' in k.lower() or 'fingerprint' in k.lower() or 'backbone' in k.lower()}

            def _remap(src, dst):
                matched = {}
                for k, v in src.items():
                    if k in dst and dst[k].shape == v.shape:
                        matched[k] = v
                    else:
                        for p in ['face_','face.','backbone.','face_net.']:
                            if k.startswith(p):
                                nk = k.replace(p,'',1)
                                if nk in dst and dst[nk].shape == v.shape:
                                    matched[nk] = v; break
                        for p in ['fingerprint_','fingerprint.','backbone.','fp_','fp.']:
                            if k.startswith(p):
                                nk = k.replace(p,'',1)
                                if nk in dst and dst[nk].shape == v.shape:
                                    matched[nk] = v; break
                return matched

            fm = _remap(face_state, face_m.state_dict())
            if fm:
                face_m.load_state_dict(fm, strict=False)
                print(f"  Face backbone loaded from fusion ckpt ({len(fm)} params)")

            fpm = _remap(fp_state, fp_m.state_dict())
            if fpm:
                fp_m.load_state_dict(fpm, strict=False)
                print(f"  FP backbone loaded from fusion ckpt ({len(fpm)} params)")

            return len(fm) > 0, len(fpm) > 0
        except Exception as e:
            print(f"  [WARN] Failed to load backbones from fusion ckpt: {e}")
            return False, False

    # Face backbone
    fc = load_config(str(cfg_dir / "face_config.yaml"))
    face_m = create_model(
        modality="face", model_type="facenet",
        num_classes=fc["model"]["num_classes"],
        embedding_dim=fc["model"].get("embedding_dim", 512),
        pretrained=True,
        dropout_rate=fc["model"].get("dropout_rate", 0.5),
        spatial_attention=fc["model"].get("spatial_attention", False),
    ).to(dev)

    # 尝试从 fusion checkpoint 加载 jointly trained backbone
    loaded_from_fusion, _ = _load_backbones_from_fusion_ckpt(face_m, fp_m, fusion_ckpt, dev)
    if not loaded_from_fusion:
        face_out = str(Path(src_root) / "outputs" / "face")
        bp, ba = _find_best_ckpt(face_out, dev)
        if bp:
            _load_ckpt(face_m, bp, dev)
            print(f"  Face checkpoint: {Path(bp).name} (val_acc={ba:.4f})")

    # FP backbone
    fpc = load_config(str(cfg_dir / "fingerprint_config.yaml"))
    fp_m = create_model(
        modality="fingerprint", model_type="fingerprint_net",
        num_classes=fpc["model"]["num_classes"],
        embedding_dim=fpc["model"].get("embedding_dim", 512),
        pretrained=True,
        dropout_rate=fpc["model"].get("dropout_rate", 0.5),
    ).to(dev)

    if not loaded_from_fusion:
        fp_out = str(Path(src_root) / "outputs" / "fingerprint")
        bp, ba = _find_best_ckpt(fp_out, dev)
        if bp:
            _load_ckpt(fp_m, bp, dev)
            print(f"  FP checkpoint: {Path(bp).name} (val_acc={ba:.4f})")

    # 消融模式
    abl = {"fusion_fp_ablation": "face", "fusion_face_ablation": "fingerprint"}.get(mod)

    # 融合数据集
    from core.datasets.fusion_dataset import FusionDataset
    seed = cfg["misc"]["seed"]
    sr = cfg["data"]["split_ratio"]; tr = cfg["data"]["test_split_ratio"]
    gp = cfg["data"].get("gallery_per_person", 3)
    fsz = cfg["data"]["face_image_size"]; fpsz = cfg["data"]["fingerprint_image_size"]

    ds = FusionDataset(
        face_data_dir=str(Path(src_root) / cfg["paths"]["face_data_dir"]),
        fingerprint_data_dir=str(Path(src_root) / cfg["paths"]["fingerprint_data_dir"]),
        mapping_file=str(Path(src_root) / cfg["paths"]["mapping_file"]),
        split_ratio=sr, test_split_ratio=tr, gallery_per_person=gp,
        face_image_size=fsz, fingerprint_image_size=fpsz,
        augment=False, seed=seed,
    )

    g_fp = [p[0] for p in ds.val_gallery_paths]
    g_fi = [p[1] for p in ds.val_gallery_paths]
    g_lbl = np.array(ds.val_gallery_labels)
    q_fp = [p[0] for p in ds.val_query_paths]
    q_fi = [p[1] for p in ds.val_query_paths]
    q_lbl = np.array(ds.val_query_labels)
    print(f"  Gallery: {len(g_fp)} | Query: {len(q_fp)}")

    ft = build_transform(fsz); fpt = build_transform(fpsz)

    g_emb, g_v = extract_fusion(face_m, fp_m, fusion_m, g_fp, g_fi, ft, fpt, dev, abl=abl)
    q_emb, q_v = extract_fusion(face_m, fp_m, fusion_m, q_fp, q_fi, ft, fpt, dev, abl=abl)
    g_lbl = g_lbl[g_v]; q_lbl = q_lbl[q_v]
    print(f"  Gallery emb: {g_emb.shape} | Query emb: {q_emb.shape}")
    if g_emb.shape[0] == 0 or q_emb.shape[0] == 0:
        print("  [WARN] All embeddings are NaN"); return None

    pos, neg = collect_scores(q_emb.numpy(), g_emb.numpy(), q_lbl, g_lbl)
    print(f"  Positive: {len(pos)} | Negative: {len(neg)}")
    out_dir = Path(src_root) / "results" / "roc_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    auc_v, eer_v = plot_single(pos, neg, name, COLORS[0], str(out_dir / f"roc_{mod}.png"))
    return pos, neg, name, COLORS[0]


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "face", "fingerprint",
                                 "fusion_full", "fusion_fp_ablation", "fusion_face_ablation"])
    args = parser.parse_args()

    mods = ["face", "fingerprint", "fusion_full", "fusion_fp_ablation", "fusion_face_ablation"] \
           if args.modality == "all" else [args.modality]

    results = []
    for m in mods:
        r = run_fusion(m) if "fusion" in m else run_single(m)
        if r: results.append(r)

    if len(results) > 1:
        out_dir = Path(src_root) / "results" / "roc_curves"
        plot_combined(results, str(out_dir / "roc_all_modalities.png"))

    print(f"\nDone → {Path(src_root)/'results'/'roc_curves'}")

if __name__ == "__main__":
    main()
