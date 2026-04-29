import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
import numpy as np

from .base_trainer import BaseTrainer, AverageMeter


class FaceTrainer(BaseTrainer):
    """Face trainer: closed-set ArcFace training + open-set Gallery/Query retrieval validation.

    训练：
        - ArcFace loss（闭集分类）
        - AMP + NaN 防护
        - 特征范数监控

    验证：
        - 完全脱离分类头，Gallery/Query 纯余弦相似度匹配
        - Rank-1/5/10/20 准确率 + EER
        - 等效于 FingerprintTrainer 的开集验证逻辑
    """

    MODALITY = 'face'

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 criterion, device, logger, tb_writer=None,
                 arcface_s=64.0, arcface_m=0.5, label_smoothing=0.0, tta=False,
                 seed=42):
        super(FaceTrainer, self).__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m
        self.label_smoothing = label_smoothing
        self.tta = tta
        self.seed = seed
        self._setup_arcface()

        self._gallery_embeddings_cache = None
        self._gallery_labels_cache = None
        self._gallery_dirty = True
        self._last_best_acc = -1.0

    def _setup_arcface(self):
        if self.model._classifier is None:
            from ..losses.arcface import ArcMarginProduct
            num_classes = self.model.num_classes
            embedding_dim = self.model.get_embedding_dim()

            self.model._classifier = ArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                s=self.arcface_s,
                m=self.arcface_m
            ).to(self.device)

            with torch.no_grad():
                w_norm = self.model._classifier.weight.norm(p=2, dim=1, keepdim=True)
                self.model._classifier.weight.div_(w_norm)

            self.logger.info(
                f"[初始化] ArcFace: s={self.arcface_s}, m={self.arcface_m}, "
                f"类别数={num_classes}，权重已归一化"
            )

    def update_arcface_margin(self, new_m):
        self.arcface_m = new_m
        if self.model._classifier is not None:
            self.model._classifier.m = new_m
            self.model._classifier.cos_m = torch.cos(torch.tensor(new_m)).to(self.device)
            self.model._classifier.sin_m = torch.sin(torch.tensor(new_m)).to(self.device)
            self.model._classifier.th = torch.cos(torch.tensor(np.pi - new_m)).to(self.device)
            self.model._classifier.mm = torch.sin(torch.tensor(np.pi - new_m)) * new_m
            self.logger.info(f"[ArcFace] margin 更新: m={new_m:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # EER 计算工具（与 FingerprintTrainer 保持一致）
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_eer(labels, scores):
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
        inputs = batch.get("image", batch.get("input"))
        if inputs is None:
            raise ValueError("Batch must contain 'image' or 'input' key")

        labels = batch["label"].to(self.device)
        inputs = inputs.to(self.device)

        # 检测输入是否包含 NaN/Inf
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            return None, None, 0.0, "input_nan"

        # 训练：只对 forward 使用 autocast
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features = self.model._extract_features(inputs)
                embeddings = F.normalize(features, p=2, dim=1)
                if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                    return None, None, 0.0, "feature_nan"
                logits = self.model._classifier(embeddings, labels)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    return None, None, 0.0, "logits_nan"
        else:
            features = self.model._extract_features(inputs)
            embeddings = F.normalize(features, p=2, dim=1)
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                return None, None, 0.0, "feature_nan"
            logits = self.model._classifier(embeddings, labels)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return None, None, 0.0, "logits_nan"

        if self.label_smoothing > 0:
            criterion_ls = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = criterion_ls(logits.float(), labels)
        else:
            loss = self.criterion(logits.float(), labels)

        if torch.isnan(loss):
            return None, logits, 0.0, "loss_nan"

        if scaler is not None:
            scaled_loss = scaler.scale(loss)
        else:
            scaled_loss = loss

        with torch.no_grad():
            preds = logits.float().argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        return loss, logits, acc, "valid"

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch, use_amp=False):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        feat_norm_meter = AverageMeter()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        nan_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for batch in pbar:
            result = self.train_step(batch, scaler, use_amp=use_amp)
            if result[0] is None:
                nan_count += 1
                continue

            loss, logits, acc, _ = result

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                embeddings = F.normalize(self.model._extract_features(inputs), p=2, dim=1)
                feat_norm = embeddings.norm(dim=1).mean().item()
                logits_max = logits.max().item()
                logits_min = logits.min().item()

                if logits_max > 100 or logits_min < -100:
                    self.logger.warning(
                        f"[调试] Logits异常: max={logits_max:.2f}, min={logits_min:.2f}, "
                        f"s={self.arcface_s}可能过大!"
                    )
                if feat_norm < 0.1 or feat_norm > 2.0:
                    self.logger.warning(
                        f"[调试] 特征范数异常: feat_norm={feat_norm:.4f}, 预期约等于1.0"
                    )

            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            feat_norm_meter.update(feat_norm, batch_size)

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        if nan_count > 0:
            self.logger.warning(
                f"[训练] Epoch {epoch+1}: 共 {nan_count} 个 NaN batch 已跳过，"
                f"有效 batch 数={loss_meter.count}"
            )

        # 统一日志格式
        current_lr = self.optimizer.param_groups[0]['lr']
        gallery_size = len(self.val_loader.dataset.val_gallery_paths) if hasattr(self.val_loader.dataset, 'val_gallery_paths') else None
        query_size = len(self.val_loader.dataset.val_query_paths) if hasattr(self.val_loader.dataset, 'val_query_paths') else None

        self.log_train_epoch(
            epoch=epoch + 1,
            total_epochs=None,  # Will be set by caller
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
    def validate_epoch(self, epoch, total_epochs=50, val_acc=None, use_amp=False):
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
            self.logger.error("[验证] val_gallery_paths 不存在！检查 FaceDataset 是否已更新。")
            return 0.0, 0.0, {}

        gallery_paths = val_dataset.val_gallery_paths
        gallery_labels_arr = np.array(val_dataset.val_gallery_labels)

        excluded_paths = set(gallery_paths)
        path_to_idx = {p: i for i, p in enumerate(val_dataset.image_paths)}
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征 ───────────────────────────────────
        gallery_embeddings_list = []
        pbar_g = tqdm(range(0, len(gallery_paths), batch_size), desc="Gallery", leave=False)
        for start in pbar_g:
            end = min(start + batch_size, len(gallery_paths))
            batch_paths = gallery_paths[start:end]
            images = [
                val_dataset[path_to_idx[p]]['image']
                for p in batch_paths if p in path_to_idx
            ]
            if not images:
                continue
            batch_tensor = torch.stack(images).to(self.device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(batch_tensor)
            else:
                emb_orig = self.model.extract_features(batch_tensor)

            if self.tta:
                emb_flip = self.model.extract_features(batch_tensor.flip(dims=[3]))
                embeddings = F.normalize(emb_orig + emb_flip, p=2, dim=1)
            else:
                embeddings = emb_orig
            gallery_embeddings_list.append(embeddings.cpu())

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
        self.logger.info(
            f"[Gallery] {len(gallery_paths)} 张（验证集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征 ─────────────────────────────────────
        query_embeddings_list = []
        query_labels_list = []
        feat_norm_meter = AverageMeter()

        self.logger.info("[验证] 正在提取 Query 特征...")
        pbar_q = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch in pbar_q:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            inputs = inputs.to(self.device)
            labels = batch["label"]
            paths = batch.get("path", [None] * inputs.size(0))

            mask_gallery = torch.tensor([
                p in excluded_paths if p is not None else True
                for p in paths
            ])
            mask_query = ~mask_gallery

            if mask_query.sum() == 0:
                continue

            inputs_q = inputs[mask_query]
            labels_q = labels[mask_query]

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(inputs_q)
            else:
                emb_orig = self.model.extract_features(inputs_q)

            if self.tta:
                emb_flip = self.model.extract_features(inputs_q.flip(dims=[3]))
                embeddings_q = F.normalize(emb_orig + emb_flip, p=2, dim=1)
            else:
                embeddings_q = emb_orig

            feat_norm = embeddings_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_meter.update(feat_norm, inputs_q.size(0))

            query_embeddings_list.append(embeddings_q.cpu())
            query_labels_list.extend(labels_q.tolist())

        if not query_embeddings_list:
            self.logger.error("[验证] 没有 Query 样本！检查 val_gallery_paths 是否覆盖了所有验证图。")
            return 0.0, 0.0, {}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)
        query_labels = np.array(query_labels_list)

        if torch.isnan(query_embeddings).any() or torch.isnan(gallery_embeddings).any():
            self.logger.error("[验证] 特征包含 NaN！模型特征提取器崩溃。")
            return float('nan'), 0.0, {"feature_norm": 0.0}

        self.logger.info(
            f"[验证] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个验证人, "
            f"特征范数={feat_norm_meter.avg:.4f}"
        )

        # ── 步骤 3：计算余弦相似度矩阵 ────────────────────────────────
        self.logger.info("[验证] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

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

            # 使用固定种子保证 EER 可复现
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
            "feature_norm": feat_norm_meter.avg,
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
            loss=0.0,
            rank1=rank1_acc,
            eer=eer if not np.isnan(eer) else None,
            gallery_size=len(gallery_paths),
            query_size=len(query_labels)
        )

        # 打印样本匹配详情（前 5 个）
        self.logger.info("[验证样本] Query vs Top-3 Gallery 匹配（显示前 5 个）:")
        sample_indices = list(range(min(5, len(query_labels))))
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
            self.tb_writer.add_scalar('val/feature_norm', feat_norm_meter.avg, epoch)

        return 0.0, rank1_acc, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 测试集评估（与 validate_epoch 逻辑相同，但独立于验证过程）
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_epoch(self, epoch=None, total_epochs=None, use_amp=False):
        """在测试集上进行评估。

        与 validate_epoch 的区别：
        - 仅用于最终模型评估，不参与 early stopping
        - 日志使用 "Test" 标识
        """
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

        excluded_paths = set(gallery_paths)
        path_to_idx = {p: i for i, p in enumerate(test_dataset.image_paths)}
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征 ───────────────────────────────────
        gallery_embeddings_list = []
        pbar_g = tqdm(range(0, len(gallery_paths), batch_size), desc="Test Gallery", leave=False)
        for start in pbar_g:
            end = min(start + batch_size, len(gallery_paths))
            batch_paths = gallery_paths[start:end]
            images = [
                test_dataset[path_to_idx[p]]['image']
                for p in batch_paths if p in path_to_idx
            ]
            if not images:
                continue
            batch_tensor = torch.stack(images).to(self.device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(batch_tensor)
            else:
                emb_orig = self.model.extract_features(batch_tensor)
            embeddings = F.normalize(emb_orig, p=2, dim=1)
            gallery_embeddings_list.append(embeddings.cpu())

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
        self.logger.info(
            f"[Test Gallery] {len(gallery_paths)} 张（测试集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征 ─────────────────────────────────────
        # 创建测试集 DataLoader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        query_embeddings_list = []
        query_labels_list = []
        feat_norm_meter = AverageMeter()

        self.logger.info("[测试] 正在提取 Query 特征...")
        pbar_q = tqdm(test_loader, desc="Test Query", leave=False)
        for batch in pbar_q:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            inputs = inputs.to(self.device)
            labels = batch["label"]
            paths = batch.get("path", [None] * inputs.size(0))

            mask_gallery = torch.tensor([
                p in excluded_paths if p is not None else True
                for p in paths
            ])
            mask_query = ~mask_gallery

            if mask_query.sum() == 0:
                continue

            inputs_q = inputs[mask_query]
            labels_q = labels[mask_query]

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(inputs_q)
            else:
                emb_orig = self.model.extract_features(inputs_q)

            embeddings_q = F.normalize(emb_orig, p=2, dim=1)

            feat_norm = embeddings_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_meter.update(feat_norm, inputs_q.size(0))

            query_embeddings_list.append(embeddings_q.cpu())
            query_labels_list.extend(labels_q.tolist())

        if not query_embeddings_list:
            self.logger.error("[测试] 没有 Query 样本！")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)
        query_labels = np.array(query_labels_list)

        self.logger.info(
            f"[测试] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个测试人, "
            f"特征范数={feat_norm_meter.avg:.4f}"
        )

        # ── 步骤 3：计算余弦相似度矩阵 ────────────────────────────────
        self.logger.info("[测试] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

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

        # ── 步骤 5：EER ────────────────────────────────────────────
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
                    f"[Test EER] 正样本={n_pos}, 负样本={n_neg}, "
                    f"EER={eer:.4f} (阈值={eer_th:.4f})"
                )
        except Exception as e:
            self.logger.warning(f"[Test EER计算] 失败: {e}")

        # ── 步骤 6：汇总指标 ─────────────────────────────────────
        metrics = {
            "rank_1": rank1_acc,
            "rank_5": rank5_acc,
            "rank_10": rank10_acc,
            "rank_20": rank20_acc,
            "eer": eer,
            "feature_norm": feat_norm_meter.avg,
            "query_count": len(query_labels),
            "gallery_count": len(gallery_paths),
            "gallery_persons": int(len(np.unique(gallery_labels_arr))),
            "query_persons": int(len(np.unique(query_labels))),
        }

        # 打印测试结果
        self.logger.info("=" * 60)
        self.logger.info(f"【测试集评估】")
        self.logger.info(f"  Rank-1:  {rank1_acc:.4f} ({int(rank1_acc * len(query_labels))}/{len(query_labels)})")
        self.logger.info(f"  Rank-5:  {rank5_acc:.4f}")
        self.logger.info(f"  Rank-10: {rank10_acc:.4f}")
        self.logger.info(f"  Rank-20: {rank20_acc:.4f}")
        self.logger.info(f"  EER:     {eer:.4f}" if eer > 0 else f"  EER:     N/A")
        self.logger.info(f"  Gallery: {len(gallery_paths)} 张 / {len(np.unique(gallery_labels_arr))} 人")
        self.logger.info(f"  Query:   {len(query_labels)} 张 / {len(np.unique(query_labels))} 人")
        self.logger.info("=" * 60)

        # 打印样本匹配详情（前 5 个）
        self.logger.info("[测试样本] Query vs Top-3 Gallery 匹配（显示前 5 个）:")
        sample_indices = list(range(min(5, len(query_labels))))
        for idx in sample_indices:
            true_label = query_labels[idx]
            top3_pred_labels = top_k_labels[idx, :3]
            top3_sims = similarity_matrix[idx, top_k_indices[idx, :3]].numpy()
            match_str = "[O]" if true_label == top3_pred_labels[0] else "[X]"
            top3_str = ", ".join(
                f"{l}({s:.3f})" for l, s in zip(top3_pred_labels, top3_sims)
            )
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 保存检查点
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch=None, is_best=False, extra=None):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "arc_classifier_state": self.model.classifier.state_dict() if self.model.classifier else None,
            "epoch": epoch or 0,
            "arcface_s": self.arcface_s,
            "arcface_m": self.arcface_m,
        }
        if extra:
            state.update(extra)

        if is_best:
            torch.save(state, path)
            self.logger.info(f"[保存] 最佳模型: {path}")
        else:
            latest_path = path.replace(".pth", "_latest.pth")
            torch.save(state, latest_path)

        return path
