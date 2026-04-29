"""
融合模型训练器
支持人脸+指纹多模态训练
支持消融实验（单模态缺失测试）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms

from .base_trainer import BaseTrainer, AverageMeter


class FusionTrainer(BaseTrainer):
    """多模态融合训练器 - 支持完整训练和消融实验

    支持：
    - 特征级融合（人脸 + 指纹）
    - ArcFace 度量学习
    - 开放集检索验证（Gallery/Query）
    - 单模态 vs 融合对比评估
    - 消融实验（单模态缺失测试）
    """

    MODALITY = 'fusion'

    def __init__(self, fusion_model, face_model, fingerprint_model,
                 train_loader, val_loader, test_loader=None,
                 optimizer=None, scheduler=None, criterion=None,
                 device='cuda', logger=None, pretrained_ckpts=None, freeze_backbone=False,
                 use_amp=True, accumulation_steps=1, seed=42,
                 experiment_mode='full', ablate_modality=None,
                 label_smoothing=0.0, tb_writer=None):
        super().__init__(
            fusion_model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )

        self.face_model = face_model.to(device) if face_model else None
        self.fingerprint_model = fingerprint_model.to(device) if fingerprint_model else None
        self.freeze_backbone = freeze_backbone
        self.current_epoch = 0
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.ablate_modality = ablate_modality
        self.label_smoothing = label_smoothing
        self.test_loader = test_loader
        self.test_dataset = test_loader.dataset if test_loader else None

        # AMP 配置
        self.use_amp = use_amp and device.type == 'cuda'
        self.accumulation_steps = max(1, accumulation_steps)
        self._scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # 带 label_smoothing 的 CrossEntropyLoss（避免每 batch 重建）
        self._criterion_ls = nn.CrossEntropyLoss(label_smoothing=label_smoothing) \
            if label_smoothing > 0 else None

        if self.use_amp:
            self.logger.info(f"[AMP] Mixed precision training enabled (accumulation_steps={self.accumulation_steps})")

        # 实验模式日志
        self._log_experiment_mode()

        # 加载预训练权重
        self._load_pretrained_weights(pretrained_ckpts)

        # 冻结backbone（如需要）
        if freeze_backbone or experiment_mode == 'fusion_only':
            self._freeze_feature_extractors()

        # 设置可训练参数
        self._setup_trainable_params()

        # 初始化验证集图像变换
        self._init_val_transforms()

        # Gallery 缓存
        self._gallery_embeddings_cache = None
        self._gallery_labels_cache = None
        self._gallery_dirty = True
        self._last_best_acc = -1.0

    def _log_experiment_mode(self):
        """记录实验模式配置"""
        mode_descriptions = {
            'full': '训练全部（backbone + fusion）',
            'fusion_only': '冻结backbone，只训练融合层',
            'face_ablation': '消融实验：指纹置零，测试单用人脸',
            'fingerprint_ablation': '消融实验：人脸置零，测试单用指纹',
        }
        desc = mode_descriptions.get(self.experiment_mode, '未知模式')
        self.logger.info(f"[Experiment] Mode: {self.experiment_mode} - {desc}")
        if self.ablate_modality:
            self.logger.info(f"[Ablation] Modality disabled: {self.ablate_modality}")

    def _init_val_transforms(self):
        """初始化验证时的图像变换（用于Gallery特征提取）"""
        self.val_face_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_fp_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _setup_trainable_params(self):
        """设置可训练参数"""
        if self.freeze_backbone or self.experiment_mode == 'fusion_only':
            for p in self.model.parameters():
                p.requires_grad = True
            if self.face_model:
                for p in self.face_model.parameters():
                    p.requires_grad = False
            if self.fingerprint_model:
                for p in self.fingerprint_model.parameters():
                    p.requires_grad = False
            self.logger.info("[Config] Training fusion only (backbones frozen)")
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            if self.face_model:
                for p in self.face_model.parameters():
                    p.requires_grad = True
            if self.fingerprint_model:
                for p in self.fingerprint_model.parameters():
                    p.requires_grad = True
            self.logger.info("[Config] Training fusion + all backbones")

    def _load_single_modality_weights(self, model, ckpt_path, modality_name):
        """加载单个模态的预训练权重"""
        if not model or not os.path.exists(ckpt_path):
            self.logger.warning(f"[{modality_name}] Checkpoint not found")
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state', ckpt)

            model_state = model.state_dict()
            matched = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    matched[k] = v

            if matched:
                model.load_state_dict(matched, strict=False)
                self.logger.info(f"[{modality_name}] Loaded {len(matched)} params")
                return True
        except Exception as e:
            self.logger.warning(f"[{modality_name}] Load failed: {e}")
        return False

    def _load_pretrained_weights(self, pretrained_ckpts):
        """加载预训练权重"""
        if not pretrained_ckpts:
            return

        if self.face_model and pretrained_ckpts.get('face'):
            self._load_single_modality_weights(self.face_model, pretrained_ckpts['face'], "Face")

        if self.fingerprint_model and pretrained_ckpts.get('fingerprint'):
            self._load_single_modality_weights(self.fingerprint_model, pretrained_ckpts['fingerprint'], "FP")

    def _freeze_feature_extractors(self):
        """冻结特征提取器"""
        if self.face_model:
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.fingerprint_model:
            for param in self.fingerprint_model.parameters():
                param.requires_grad = False

    def _apply_ablation(self, face_features, fp_features):
        """应用消融：将指定模态置零"""
        if self.ablate_modality == 'face':
            face_features = torch.zeros_like(face_features)
            self.logger.debug("[Ablation] Face features zeroed out")
        elif self.ablate_modality == 'fingerprint':
            fp_features = torch.zeros_like(fp_features)
            self.logger.debug("[Ablation] Fingerprint features zeroed out")
        return face_features, fp_features

    def _extract_features_train(self, face_images, fp_images):
        """训练时提取特征（允许梯度）"""
        if self.face_model:
            self.face_model.train()
            face_features = self.face_model(face_images)
        else:
            face_features = torch.randn(face_images.size(0), 512, device=self.device)

        if self.fingerprint_model:
            self.fingerprint_model.train()
            fp_features = self.fingerprint_model(fp_images)
        else:
            fp_features = torch.randn(fp_images.size(0), 512, device=self.device)

        face_features, fp_features = self._apply_ablation(face_features, fp_features)

        return face_features, fp_features

    def _extract_features_eval(self, face_images, fp_images):
        """验证时提取特征（无梯度）"""
        if self.face_model:
            self.face_model.eval()
            with torch.no_grad():
                face_features = self.face_model(face_images)
        else:
            face_features = torch.randn(face_images.size(0), 512, device=self.device)

        if self.fingerprint_model:
            self.fingerprint_model.eval()
            with torch.no_grad():
                fp_features = self.fingerprint_model(fp_images)
        else:
            fp_features = torch.randn(fp_images.size(0), 512, device=self.device)

        face_features, fp_features = self._apply_ablation(face_features, fp_features)

        return face_features, fp_features

    def _load_gallery_batch(self, batch_pairs, batch_size):
        """加载一个Gallery批次的图像并提取特征"""
        face_imgs = []
        fp_imgs = []

        for pair in batch_pairs:
            face_path, fp_path = pair
            try:
                face_img = Image.open(face_path).convert('RGB')
                face_img = self.val_face_transform(face_img)
                face_imgs.append(face_img)
            except Exception as e:
                self.logger.warning(f"[Gallery] Failed to load face: {face_path}, {e}")
                face_imgs.append(torch.zeros(3, 224, 224))

            try:
                fp_img = Image.open(fp_path).convert('RGB')
                fp_img = self.val_fp_transform(fp_img)
                fp_imgs.append(fp_img)
            except Exception as e:
                self.logger.warning(f"[Gallery] Failed to load fingerprint: {fp_path}, {e}")
                fp_imgs.append(torch.zeros(3, 224, 224))

        face_batch = torch.stack(face_imgs).to(self.device)
        fp_batch = torch.stack(fp_imgs).to(self.device)

        return face_batch, fp_batch

    # ─────────────────────────────────────────────────────────────────────────
    # EER 计算工具
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_eer(labels, scores):
        from sklearn.metrics import roc_curve
        if len(np.unique(labels)) < 2:
            return 0.0, 0.0
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        threshold = float(thresholds[eer_idx])
        return eer, threshold

    # ─────────────────────────────────────────────────────────────────────────
    # 单步训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, batch, scaler=None, use_amp=False):
        """单步训练。

        流程：特征提取 → 融合 → 分类 logits → CrossEntropyLoss → AMP backward
        - NaN 防护：输入/特征/logits/loss 四层检测
        - AMP：前向 fp16 + loss scale + loss 转 fp32（防 exp 溢出）
        """
        face_images = batch['face_image'].to(self.device)
        fp_images = batch['fingerprint_image'].to(self.device)
        targets = batch['label'].to(self.device)

        if torch.isnan(face_images).any() or torch.isinf(face_images).any():
            return None, None, 0.0, "input_nan"
        if torch.isnan(fp_images).any() or torch.isinf(fp_images).any():
            return None, None, 0.0, "input_nan"

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                face_features, fp_features = self._extract_features_train(face_images, fp_images)
                if torch.isnan(face_features).any() or torch.isinf(face_features).any():
                    return None, None, 0.0, "face_feature_nan"
                if torch.isnan(fp_features).any() or torch.isinf(fp_features).any():
                    return None, None, 0.0, "fp_feature_nan"

                # 清理残留 NaN
                face_features = torch.where(torch.isnan(face_features),
                    torch.zeros_like(face_features), face_features)
                fp_features = torch.where(torch.isnan(fp_features),
                    torch.zeros_like(fp_features), fp_features)

                outputs = self.model(face_features, fp_features)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    return None, outputs, 0.0, "logits_nan"

                if self._criterion_ls is not None:
                    loss = self._criterion_ls(outputs.float(), targets)
                else:
                    loss = self.criterion(outputs.float(), targets)
        else:
            face_features, fp_features = self._extract_features_train(face_images, fp_images)
            if torch.isnan(face_features).any() or torch.isinf(face_features).any():
                return None, None, 0.0, "face_feature_nan"
            if torch.isnan(fp_features).any() or torch.isinf(fp_features).any():
                return None, None, 0.0, "fp_feature_nan"

            face_features = torch.where(torch.isnan(face_features),
                torch.zeros_like(face_features), face_features)
            fp_features = torch.where(torch.isnan(fp_features),
                torch.zeros_like(fp_features), fp_features)

            outputs = self.model(face_features, fp_features)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return None, outputs, 0.0, "logits_nan"

            if self._criterion_ls is not None:
                loss = self._criterion_ls(outputs.float(), targets)
            else:
                loss = self.criterion(outputs.float(), targets)

        if torch.isnan(loss):
            return None, outputs, 0.0, "loss_nan"

        with torch.no_grad():
            preds = outputs.float().argmax(dim=1)
            acc = (preds == targets).float().mean().item()

        return loss, outputs, acc, "valid"

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch, total_epochs=None, use_amp=False):
        """训练一轮。

        NaN-safe：遇到 NaN batch 跳过，但继续处理后续 batch。
        包含特征范数监控（应约等于 1.0）。
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        feat_norm_meter = AverageMeter()
        scaler = self._scaler
        nan_count = 0
        nan_reasons = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            result = self.train_step(batch, scaler, use_amp=use_amp)
            if result[0] is None:
                nan_count += 1
                reason = result[3]
                nan_reasons[reason] = nan_reasons.get(reason, 0) + 1
                continue

            loss, outputs, acc, _ = result

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积步数到达时更新
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            batch_size = outputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            # 特征范数监控
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)
            with torch.no_grad():
                face_feat, fp_feat = self._extract_features_eval(face_images, fp_images)
                fused = self.model.extract_fused_features(face_feat, fp_feat)
                fused = F.normalize(fused, p=2, dim=1)
                feat_norm = fused.norm(dim=1).mean().item()
                if not np.isnan(feat_norm):
                    feat_norm_meter.update(feat_norm, batch_size)

                if feat_norm < 0.1 or feat_norm > 2.0:
                    tqdm.write(f"[调试] 特征范数异常: feat_norm={feat_norm:.4f}, 预期约等于1.0")

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # 处理剩余未更新的梯度
        if (batch_idx + 1) % self.accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

        if nan_count > 0:
            nan_detail = ", ".join(f"{k}={v}" for k, v in nan_reasons.items())
            self.logger.warning(
                f"[训练] Epoch {epoch+1}: 共 {nan_count} 个 NaN batch 已跳过 "
                f"({nan_detail})，有效 batch 数={loss_meter.count}"
            )

        # 统一日志格式
        current_lr = self.optimizer.param_groups[0]['lr']
        gallery_size = len(self.val_loader.dataset.val_gallery_paths) if hasattr(self.val_loader.dataset, 'val_gallery_paths') else None
        query_size = len(self.val_loader.dataset.val_query_paths) if hasattr(self.val_loader.dataset, 'val_query_paths') else None

        self.log_train_epoch(
            epoch=epoch + 1,
            total_epochs=total_epochs,
            lr=current_lr,
            loss=loss_meter.avg,
            acc=acc_meter.avg,
            gallery_size=gallery_size,
            query_size=query_size
        )

        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', loss_meter.avg, epoch)
            self.tb_writer.add_scalar('train/accuracy', acc_meter.avg, epoch)
            self.tb_writer.add_scalar('train/feature_norm', feat_norm_meter.avg, epoch)

        return loss_meter.avg, acc_meter.avg

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮验证（Gallery/Query 1:N 余弦相似度检索）
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate_epoch(self, epoch, total_epochs=50, use_amp=False):
        """1:N 开集检索验证（Gallery/Query 余弦相似度，完全脱离分类头）。

        流程：
            1. 提取 Gallery 特征（L2 归一化）
            2. 提取 Query 特征（L2 归一化）
            3. 计算余弦相似度矩阵
            4. 计算 Rank-1/5/10/20 准确率
            5. 计算 EER（同人匹配 vs 异人拒绝）
        """
        self.model.eval()
        val_dataset = self.val_loader.dataset

        # ── 检查 Gallery/Query 数据是否就绪 ───────────────────────────────
        if not hasattr(val_dataset, 'val_gallery_paths') or not val_dataset.val_gallery_paths:
            self.logger.error("[验证] val_gallery_paths 不存在！检查 FusionDataset 是否已更新。")
            return 0.0, 0.0, {}

        gallery_paths = val_dataset.val_gallery_paths
        gallery_labels_arr = np.array(val_dataset.val_gallery_labels)
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征 ───────────────────────────────────
        gallery_embeddings_list = []
        feat_norm_g_meter = AverageMeter()

        self.logger.info("[Gallery] 正在提取特征...")
        n_gallery_batches = (len(gallery_paths) + batch_size - 1) // batch_size
        pbar_g = tqdm(total=n_gallery_batches, desc="Gallery", leave=False)
        for start_idx in range(0, len(gallery_paths), batch_size):
            end = min(start_idx + batch_size, len(gallery_paths))
            batch_pairs = gallery_paths[start_idx:end]
            face_batch, fp_batch = self._load_gallery_batch(batch_pairs, batch_size)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)
            else:
                face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_g_meter.update(feat_norm, fused_features.size(0))

            gallery_embeddings_list.append(fused_features.cpu())
            pbar_g.update()
        pbar_g.close()

        if not gallery_embeddings_list:
            self.logger.error("[验证] Gallery 特征为空！")
            return 0.0, 0.0, {}

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)  # [G, fusion_dim]
        self.logger.info(
            f"[Gallery] {len(gallery_paths)} 对（验证集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征 ─────────────────────────────────────
        query_embeddings_list = []
        query_labels_list = []
        feat_norm_q_meter = AverageMeter()
        loss_meter = AverageMeter()

        self.logger.info("[验证] 正在提取 Query 特征...")
        pbar_q = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch in pbar_q:
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)
            targets = batch['label']

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_images, fp_images)
            else:
                face_features, fp_features = self._extract_features_eval(face_images, fp_images)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features_q = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_q_meter.update(feat_norm, face_images.size(0))

            query_embeddings_list.append(fused_features_q.cpu())
            query_labels_list.extend(targets.tolist())

            # 计算分类 loss（仅用于监控，与检索指标无关）
            with torch.no_grad():
                outputs = self.model(face_features, fp_features)
                if self._criterion_ls is not None:
                    l = self._criterion_ls(outputs.float(), targets.to(self.device))
                else:
                    l = self.criterion(outputs.float(), targets.to(self.device))
                if not torch.isnan(l):
                    loss_meter.update(l.item(), face_images.size(0))

        if not query_embeddings_list:
            self.logger.error("[验证] 没有 Query 样本！")
            return 0.0, 0.0, {}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)  # [Q, fusion_dim]
        query_labels = np.array(query_labels_list)

        if torch.isnan(query_embeddings).any() or torch.isnan(gallery_embeddings).any():
            self.logger.error("[验证] 特征包含 NaN！模型特征提取器崩溃。")
            return float('nan'), 0.0, {"feature_norm": 0.0}

        self.logger.info(
            f"[验证] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个验证人, "
            f"特征范数={feat_norm_q_meter.avg:.4f}"
        )

        # ── 步骤 3：计算余弦相似度矩阵 ────────────────────────────────
        self.logger.info("[验证] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())  # [Q, G]

        # ── 步骤 4：Rank-K 准确率 ────────────────────────────────────
        top_k = min(20, gallery_embeddings.size(0))
        _, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        top_k_indices = top_k_indices.numpy()
        top_k_labels = gallery_labels_arr[top_k_indices]

        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k:
                correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] in top_k_labels[i, :k]
                )
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        # ── 步骤 5：EER（同人匹配 vs 异人拒绝）─────────────────────
        eer = 0.0
        try:
            positive_scores = []
            negative_scores = []
            rng = np.random.RandomState(self.seed)

            for q_idx in range(len(query_labels)):
                q_label = query_labels[q_idx]
                q_emb = query_embeddings[q_idx]

                same_idx = np.where(gallery_labels_arr == q_label)[0]
                if len(same_idx) > 0:
                    sims = (q_emb @ gallery_embeddings[same_idx].t()).numpy()
                    topk_k = min(5, len(sims))
                    vals = np.sort(sims)[-topk_k:]
                    positive_scores.extend(vals.tolist())

                diff_idx = np.where(gallery_labels_arr != q_label)[0]
                if len(diff_idx) > 0:
                    n_neg = min(3, len(diff_idx))
                    selected = rng.choice(diff_idx, n_neg, replace=False)
                    sims = (q_emb @ gallery_embeddings[selected].t()).numpy()
                    negative_scores.extend(sims.tolist())

            n_pos, n_neg = len(positive_scores), len(negative_scores)
            if n_pos >= 50 and n_neg >= 50:
                scores = np.array(positive_scores + negative_scores)
                eer_labels = np.array([1] * n_pos + [0] * n_neg)
                eer, eer_th = self.calculate_eer(eer_labels, scores)
                self.logger.info(
                    f"[EER] 正样本={n_pos}, 负样本={n_neg}, "
                    f"EER={eer:.4f} (阈值={eer_th:.4f})"
                )
            else:
                self.logger.warning(
                    f"[EER] 样本不足（正={n_pos}/需50，负={n_neg}/需50），跳过 EER"
                )
        except Exception as e:
            self.logger.warning(f"[EER计算] 失败: {e}")

        # ── 步骤 6：汇总指标 ─────────────────────────────────────
        metrics = {
            "rank_1": rank1_acc,
            "rank_5": rank5_acc,
            "rank_10": rank10_acc,
            "rank_20": rank20_acc,
            "eer": eer,
            "feature_norm_gallery": feat_norm_g_meter.avg,
            "feature_norm_query": feat_norm_q_meter.avg,
            "query_count": len(query_labels),
            "gallery_count": len(gallery_paths),
            "gallery_persons": int(len(np.unique(gallery_labels_arr))),
            "query_persons": int(len(np.unique(query_labels))),
        }

        # 统一日志格式
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log_val_epoch(
            epoch=epoch + 1,
            total_epochs=total_epochs,
            lr=current_lr,
            loss=loss_meter.avg,
            rank1=rank1_acc,
            eer=eer if not np.isnan(eer) else None,
            gallery_size=len(gallery_paths),
            query_size=len(query_labels)
        )

        # 详细指标日志
        self.logger.info(
            f"[Metrics] Rank-1: {rank1_acc:.4f} | Rank-5: {rank5_acc:.4f} | "
            f"Rank-10: {rank10_acc:.4f} | Rank-20: {rank20_acc:.4f} | EER: {eer:.4f}"
        )

        # 打印样本匹配详情（按人均匀抽取前 5 个不同人的样本）
        unique_persons = np.unique(query_labels)
        selected_persons = unique_persons[:min(5, len(unique_persons))]
        sample_indices = [np.where(query_labels == p)[0][0] for p in selected_persons]
        self.logger.info("[验证样本] Query vs Top-3 Gallery 匹配（按人均匀抽取前 5 人）:")
        for idx in sample_indices:
            true_label = query_labels[idx]
            top3_pred_labels = top_k_labels[idx, :3]
            top3_sims = similarity_matrix[idx, top_k_indices[idx, :3]].numpy()
            match_str = "[O]" if true_label == top3_pred_labels[0] else "[X]"
            top3_str = ", ".join(
                f"{l}({s:.3f})" for l, s in zip(top3_pred_labels, top3_sims)
            )
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        if self.tb_writer:
            self.tb_writer.add_scalar('val/rank_1', rank1_acc, epoch)
            self.tb_writer.add_scalar('val/rank_5', rank5_acc, epoch)
            self.tb_writer.add_scalar('val/rank_10', rank10_acc, epoch)
            self.tb_writer.add_scalar('val/rank_20', rank20_acc, epoch)
            self.tb_writer.add_scalar('val/eer', eer, epoch)
            self.tb_writer.add_scalar('val/feature_norm_query', feat_norm_q_meter.avg, epoch)

        return loss_meter.avg, rank1_acc, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 测试集评估
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_epoch(self, epoch=None, total_epochs=None, use_amp=False):
        """在测试集上进行评估（与 validate_epoch 逻辑相同，但使用测试集）。"""
        self.model.eval()

        test_dataset = getattr(self, 'test_dataset', None)
        if test_dataset is None:
            self.logger.warning("[测试] 测试集未设置，跳过测试评估。")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        if not hasattr(test_dataset, 'test_gallery_paths') or not test_dataset.test_gallery_paths:
            self.logger.warning("[测试] 测试集不存在，跳过测试评估。")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        gallery_paths = test_dataset.test_gallery_paths
        gallery_labels_arr = np.array(test_dataset.test_gallery_labels)
        batch_size = self.val_loader.batch_size

        # ── Gallery ──────────────────────────────────────────────────────
        gallery_embeddings_list = []
        feat_norm_g_meter = AverageMeter()

        self.logger.info("[Test Gallery] 正在提取特征...")
        n_gallery_batches = (len(gallery_paths) + batch_size - 1) // batch_size
        pbar_g = tqdm(total=n_gallery_batches, desc="Test Gallery", leave=False)
        for start_idx in range(0, len(gallery_paths), batch_size):
            end = min(start_idx + batch_size, len(gallery_paths))
            batch_pairs = gallery_paths[start_idx:end]
            face_batch, fp_batch = self._load_gallery_batch(batch_pairs, batch_size)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)
            else:
                face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_g_meter.update(feat_norm, fused_features.size(0))
            gallery_embeddings_list.append(fused_features.cpu())
            pbar_g.update()
        pbar_g.close()

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
        self.logger.info(
            f"[Test Gallery] {len(gallery_paths)} 对（测试集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── Query ─────────────────────────────────────────────────────────
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )

        query_embeddings_list = []
        query_labels_list = []
        feat_norm_q_meter = AverageMeter()

        self.logger.info("[测试] 正在提取 Query 特征...")
        pbar_q = tqdm(test_loader, desc="Test Query", leave=False)
        for batch in pbar_q:
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_images, fp_images)
            else:
                face_features, fp_features = self._extract_features_eval(face_images, fp_images)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features_q = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_q_meter.update(feat_norm, face_images.size(0))

            query_embeddings_list.append(fused_features_q.cpu())
            query_labels_list.extend(batch['label'].tolist())

        if not query_embeddings_list:
            self.logger.error("[测试] 没有 Query 样本！")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)
        query_labels = np.array(query_labels_list)

        self.logger.info(
            f"[测试] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个测试人"
        )

        # ── 相似度 & 指标 ────────────────────────────────────────────────
        self.logger.info("[测试] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

        top_k = min(20, gallery_embeddings.size(0))
        _, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        top_k_indices = top_k_indices.numpy()
        top_k_labels = gallery_labels_arr[top_k_indices]

        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k:
                correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] in top_k_labels[i, :k]
                )
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        eer = 0.0
        try:
            positive_scores, negative_scores = [], []
            rng = np.random.RandomState(self.seed)
            for q_idx in range(len(query_labels)):
                q_label = query_labels[q_idx]
                q_emb = query_embeddings[q_idx]
                same_idx = np.where(gallery_labels_arr == q_label)[0]
                if len(same_idx) > 0:
                    sims = (q_emb @ gallery_embeddings[same_idx].t()).numpy()
                    topk_k = min(5, len(sims))
                    positive_scores.extend(np.sort(sims)[-topk_k:].tolist())
                diff_idx = np.where(gallery_labels_arr != q_label)[0]
                if len(diff_idx) > 0:
                    n_neg = min(3, len(diff_idx))
                    selected = rng.choice(diff_idx, n_neg, replace=False)
                    sims = (q_emb @ gallery_embeddings[selected].t()).numpy()
                    negative_scores.extend(sims.tolist())

            n_pos, n_neg = len(positive_scores), len(negative_scores)
            if n_pos >= 50 and n_neg >= 50:
                scores = np.array(positive_scores + negative_scores)
                eer_labels = np.array([1] * n_pos + [0] * n_neg)
                eer, eer_th = self.calculate_eer(eer_labels, scores)
                self.logger.info(
                    f"[Test EER] 正样本={n_pos}, 负样本={n_neg}, "
                    f"EER={eer:.4f} (阈值={eer_th:.4f})"
                )
        except Exception as e:
            self.logger.warning(f"[Test EER计算] 失败: {e}")

        # ── 测试结果汇总 ────────────────────────────────────────────────
        metrics = {
            "rank_1": rank1_acc, "rank_5": rank5_acc,
            "rank_10": rank10_acc, "rank_20": rank20_acc,
            "eer": eer, "feature_norm_gallery": feat_norm_g_meter.avg,
            "feature_norm_query": feat_norm_q_meter.avg,
            "query_count": len(query_labels),
            "gallery_count": len(gallery_paths),
            "gallery_persons": int(len(np.unique(gallery_labels_arr))),
            "query_persons": int(len(np.unique(query_labels))),
        }

        self.logger.info("=" * 60)
        self.logger.info("【测试集评估】")
        self.logger.info(f"  Rank-1:  {rank1_acc:.4f} ({int(rank1_acc * len(query_labels))}/{len(query_labels)})")
        self.logger.info(f"  Rank-5:  {rank5_acc:.4f}")
        self.logger.info(f"  Rank-10: {rank10_acc:.4f}")
        self.logger.info(f"  Rank-20: {rank20_acc:.4f}")
        self.logger.info(f"  EER:     {eer:.4f}" if eer > 0 else "  EER:     N/A")
        self.logger.info(f"  Gallery: {len(gallery_paths)} 对 / {len(np.unique(gallery_labels_arr))} 人")
        self.logger.info(f"  Query:   {len(query_labels)} 样本 / {len(np.unique(query_labels))} 人")
        self.logger.info("=" * 60)

        # 样本匹配详情（按人均匀抽取）
        unique_persons = np.unique(query_labels)
        selected_persons = unique_persons[:min(5, len(unique_persons))]
        sample_indices = [np.where(query_labels == p)[0][0] for p in selected_persons]
        self.logger.info("[测试样本] Query vs Top-3 Gallery 匹配（按人均匀抽取前 5 人）:")
        for idx in sample_indices:
            true_label = query_labels[idx]
            top3_pred_labels = top_k_labels[idx, :3]
            top3_sims = similarity_matrix[idx, top_k_indices[idx, :3]].numpy()
            match_str = "[O]" if true_label == top3_pred_labels[0] else "[X]"
            top3_str = ", ".join(f"{l}({s:.3f})" for l, s in zip(top3_pred_labels, top3_sims))
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 保存检查点
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch=None, is_best=False, extra=None):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch or 0,
        }
        if self.face_model:
            state["face_model_state"] = self.face_model.state_dict()
        if self.fingerprint_model:
            state["fp_model_state"] = self.fingerprint_model.state_dict()
        if extra:
            state.update(extra)

        torch.save(state, path)
        self.logger.info(f"[保存] Checkpoint: {path}" if not is_best else f"[保存] 最佳模型: {path}")
        return path

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if checkpoint.get('face_model_state') and self.face_model:
            self.face_model.load_state_dict(checkpoint['face_model_state'])
        if checkpoint.get('fp_model_state') and self.fingerprint_model:
            self.fingerprint_model.load_state_dict(checkpoint['fp_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"[Load] Checkpoint: {path}")
        return checkpoint
