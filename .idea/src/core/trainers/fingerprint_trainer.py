import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
import numpy as np
from .base_trainer import BaseTrainer, AverageMeter


class FingerprintTrainer(BaseTrainer):
    """Fingerprint trainer for 1:N retrieval-based verification.

    验证评估策略（与训练解耦）：
        - 训练：用 Softmax（标准分类）或 ArcFace（度量学习）
        - 验证：用纯余弦相似度做 1:N 检索，模拟真实使用场景
          （Gallery 提取特征，Query 与 Gallery 做余弦匹配）

    这样做的好处：
        1. 验证指标直接反映实际使用效果
        2. 验证不依赖分类头，避免分类损失和检索质量的misalignment
        3. 兼容 metric_learning=false（纯分类）和 metric_learning=true（度量学习）
    """

    MODALITY = 'fingerprint'

    # ─────────────────────────────────────────────────────────────────────────
    # 初始化
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 criterion, device, logger, tb_writer=None,
                 arcface_s=64.0, arcface_m=0.5, metric_learning=False,
                 label_smoothing=0.0, tta=False, seed=42, use_amp=False,
                 test_dataset=None):
        super(FingerprintTrainer, self).__init__(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m
        self.metric_learning = metric_learning
        self.label_smoothing = label_smoothing
        self.tta = tta
        self.seed = seed
        self.use_amp = use_amp
        self._setup_classifier()

        # GradScaler 只创建一次（避免每轮重建开销）
        self._scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # 带 label_smoothing 的 CrossEntropyLoss（避免每 batch 重建）
        self._criterion_ls = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 else None

        # Gallery 缓存
        self._gallery_embeddings_cache = None
        self._gallery_labels_cache = None
        self._gallery_dirty = True
        self._last_best_acc = -1.0
        self.test_dataset = test_dataset

    def _setup_classifier(self):
        """根据 metric_learning 配置选择分类器。

        - metric_learning=False：普通 nn.Linear（标准 Softmax）
        - metric_learning=True：ArcFace（m>0 时生效）
        """
        num_classes = self.model.num_classes
        embedding_dim = self.model.get_embedding_dim()

        if self.model._classifier is None:
            if self.metric_learning:
                from ..losses.arcface import ArcMarginProduct
                self.model._classifier = ArcMarginProduct(
                    in_features=embedding_dim,
                    out_features=num_classes,
                    s=self.arcface_s,
                    m=self.arcface_m
                ).to(self.device)
                # 权重 L2 归一化，防止 logit 爆炸
                with torch.no_grad():
                    w_norm = self.model._classifier.weight.norm(p=2, dim=1, keepdim=True)
                    self.model._classifier.weight.div_(w_norm)
                self.logger.info(
                    f"[初始化] ArcFace: s={self.arcface_s}, m={self.arcface_m}, "
                    f"类别数={num_classes}，权重已归一化"
                )
            else:
                # 普通 Softmax 分类头
                self.model._classifier = nn.Linear(
                    embedding_dim, num_classes, bias=False
                ).to(self.device)
                nn.init.xavier_normal_(self.model._classifier.weight)
                self.logger.info(
                    f"[初始化] 普通 Softmax 分类头: embedding={embedding_dim}, "
                    f"类别数={num_classes}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # ArcFace margin 动态更新（仅在 metric_learning=True 时使用）
    # ─────────────────────────────────────────────────────────────────────────
    def update_arcface_margin(self, new_m):
        """动态更新 ArcFace margin（仅在 metric_learning=True 时有效）。"""
        self.arcface_m = new_m
        if self.metric_learning and self.model._classifier is not None:
            self.model._classifier.m = new_m
            self.model._classifier.cos_m = torch.cos(torch.tensor(new_m)).to(self.device)
            self.model._classifier.sin_m = torch.sin(torch.tensor(new_m)).to(self.device)
            self.model._classifier.th = torch.cos(torch.tensor(np.pi - new_m)).to(self.device)
            self.model._classifier.mm = torch.sin(torch.tensor(np.pi - new_m)) * new_m
            self.logger.info(f"[ArcFace] margin 更新: m={new_m:.4f}")

    def switch_classifier(self, metric_learning):
        """训练中途切换分类器（从 Softmax 切换到 ArcFace，或反向）。"""
        if metric_learning == self.metric_learning:
            return  # 无需切换

        self.metric_learning = metric_learning
        self.logger.info(f"[切换] metric_learning={metric_learning}，重建分类头...")
        # 保留 embedding_dim 等配置
        self._setup_classifier()

    # ─────────────────────────────────────────────────────────────────────────
    # 单步训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, batch, scaler=None, use_amp=False):
        """单步训练。

        流程：特征提取 → 分类 logits → CrossEntropyLoss → AMP backward
        - NaN 防护：输入/特征/logits/loss 四层检测
        - AMP：前向 fp16 + loss scale + loss 转 fp32（防 exp 溢出）
        """
        inputs = batch.get("image", batch.get("input"))
        if inputs is None:
            raise ValueError("Batch must contain 'image' or 'input' key")

        labels = batch["label"].to(self.device)
        inputs = inputs.to(self.device)

        # ── NaN/Inf 输入检测 ─────────────────────────────────────────────────
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            return None, None, 0.0, "input_nan"

        # ── 前向传播（统一 AMP 包裹）──────────────────────────────────────────
        # metric_learning=False 时 classifier 是 nn.Linear（只接受 input）
        # metric_learning=True 时 classifier 是 ArcMarginProduct（接受 input, labels）
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features = self.model._extract_features(inputs)
                if torch.isnan(features).any() or torch.isinf(features).any():
                    return None, None, 0.0, "feature_nan"
                if self.metric_learning:
                    logits = self.model._classifier(features, labels)
                else:
                    logits = self.model._classifier(features)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    return None, None, 0.0, "logits_nan"
        else:
            features = self.model._extract_features(inputs)
            if torch.isnan(features).any() or torch.isinf(features).any():
                return None, None, 0.0, "feature_nan"
            if self.metric_learning:
                logits = self.model._classifier(features, labels)
            else:
                logits = self.model._classifier(features)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return None, None, 0.0, "logits_nan"

        # CrossEntropyLoss 需要 fp32（否则 exp 在 fp16 下容易溢出）
        if self._criterion_ls is not None:
            loss = self._criterion_ls(logits.float(), labels)
        else:
            loss = self.criterion(logits.float(), labels)

        # ── NaN 防护 ───────────────────────────────────────────────────────────
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
        """训练一轮。

        NaN-safe：遇到 NaN batch 跳过，但继续处理后续 batch。
        返回 loss_meter.avg（排除 NaN batch 的贡献），acc 正常平均。
        包含特征范数监控（应约等于 1.0）。
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        scaler = self._scaler  # 复用 __init__ 中创建的 scaler
        nan_count = 0
        nan_reasons = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in pbar:
            result = self.train_step(batch, scaler, use_amp=use_amp)
            if result[0] is None:
                nan_count += 1
                reason = result[3]
                nan_reasons[reason] = nan_reasons.get(reason, 0) + 1
                continue  # 跳过 NaN batch

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

            batch_size = logits.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

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
            total_epochs=None,
            lr=current_lr,
            loss=loss_meter.avg,
            acc=acc_meter.avg,
            gallery_size=gallery_size,
            query_size=query_size
        )

        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', loss_meter.avg, epoch)
            self.tb_writer.add_scalar('train/accuracy', acc_meter.avg, epoch)

        return loss_meter.avg, acc_meter.avg

    # ─────────────────────────────────────────────────────────────────────────
    # EER 计算工具
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_eer(labels, scores):
        """计算等错误率（EER）。

        EER 是 FAR（误接受率）= FRR（误拒绝率）的点。
        用于评估同人/异人的区分能力，越低越好。
        """
        if len(np.unique(labels)) < 2:
            return 0.0, 0.0

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        threshold = float(thresholds[eer_idx])
        return eer, threshold

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮验证（1:N 余弦相似度检索）
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate_epoch(self, epoch, total_epochs=55, val_acc=None, use_amp=False):
        """1:N 检索验证（余弦相似度，不依赖分类头）。

        流程：
            1. 用专用 DataLoader 提取 Gallery 特征（L2 归一化）
            2. 用 val_loader 提取 Query 特征（L2 归一化）
            3. 计算余弦相似度矩阵
            4. 计算 Rank-1/5/10 准确率
            5. 计算 EER（同人匹配 vs 异人拒绝）
        """
        self.model.eval()

        # ── 准备数据集 ────────────────────────────────────────────────────────
        val_dataset = self.val_loader.dataset

        # Gallery：从 val_gallery_paths 取（每人 3 张）
        if hasattr(val_dataset, 'val_gallery_paths') and val_dataset.val_gallery_paths:
            gallery_paths = val_dataset.val_gallery_paths
            gallery_labels_arr = np.array(val_dataset.val_gallery_labels)
        else:
            self.logger.error("[验证] val_gallery_paths 不存在！检查数据集划分逻辑。")
            return 0.0, 0.0, {}

        excluded_paths = set(gallery_paths)
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征（用专用 DataLoader）────────────────────
        from torch.utils.data import DataLoader

        # 构建 path → val_dataset index 的映射
        path_to_idx = {p: i for i, p in enumerate(val_dataset.image_paths)}
        gallery_indices = [path_to_idx[p] for p in gallery_paths if p in path_to_idx]

        # 轻量级子集包装器，避免复制整个数据集
        class _IndexedSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

        gallery_subset = _IndexedSubset(val_dataset, gallery_indices)
        gallery_loader = DataLoader(
            gallery_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        gallery_embeddings_list = []
        feat_norm_meter = AverageMeter()

        pbar_g = tqdm(gallery_loader, desc="Gallery", leave=False)
        for batch in pbar_g:
            images = batch['image'].to(self.device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(images)
            else:
                emb_orig = self.model.extract_features(images)

            if self.tta:
                emb_flip = self.model.extract_features(images.flip(dims=[3]))
                embeddings = F.normalize(emb_orig + emb_flip, p=2, dim=1)
            else:
                embeddings = F.normalize(emb_orig, p=2, dim=1)

            feat_norm = embeddings.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_meter.update(feat_norm, embeddings.size(0))
            gallery_embeddings_list.append(embeddings.cpu())

        if not gallery_embeddings_list:
            self.logger.error("[验证] Gallery 特征为空！")
            return 0.0, 0.0, {}

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)  # [G, 512]
        self.logger.info(
            f"[Gallery] {len(gallery_paths)} 张（验证集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征（复用 val_loader）────────────────────────
        query_embeddings_list = []
        query_labels_list = []
        # 用于计算验证时的分类 loss（如果分类头可用）
        loss_meter = AverageMeter()

        self.logger.info("[验证] 正在提取 Query 特征...")
        pbar_q = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch in pbar_q:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            inputs = inputs.to(self.device)
            labels = batch["label"]
            paths = batch.get("path", [None] * inputs.size(0))

            # 跳过 Gallery 图
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
                embeddings_q = F.normalize(emb_orig, p=2, dim=1)

            feat_norm = embeddings_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_meter.update(feat_norm, inputs_q.size(0))

            # 计算分类 loss（仅在存在分类头时）
            try:
                if hasattr(self.model, '_classifier') and self.model._classifier is not None:
                    lbls = labels_q.to(self.device)
                    lbls = lbls.long()
                    # 检查标签范围
                    if lbls.numel() > 0:
                        if lbls.min().item() < 0 or lbls.max().item() >= self.model.num_classes:
                            self.logger.warning(f"[验证] labels 范围异常: min={lbls.min().item()}, max={lbls.max().item()}, num_classes={self.model.num_classes}")
                    if self.metric_learning:
                        logits_q = self.model._classifier(embeddings_q, lbls)
                    else:
                        logits_q = self.model._classifier(embeddings_q)
                    if self._criterion_ls is not None:
                        l = self._criterion_ls(logits_q.float(), lbls)
                    else:
                        l = self.criterion(logits_q.float(), lbls)
                    if not torch.isnan(l):
                        loss_meter.update(l.item(), lbls.size(0))
            except Exception as e:
                self.logger.warning(f"[验证 loss] 计算失败: {e}")

            query_embeddings_list.append(embeddings_q.cpu())
            query_labels_list.extend(labels_q.tolist())

        if not query_embeddings_list:
            self.logger.error("[验证] 没有 Query 样本！")
            return 0.0, 0.0, {}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)  # [Q, 512]
        query_labels = np.array(query_labels_list)
        val_loss = loss_meter.avg

        # NaN 检查
        if torch.isnan(query_embeddings).any() or torch.isnan(gallery_embeddings).any():
            self.logger.error(
                "[验证] 特征包含 NaN！模型特征提取器崩溃。"
                "建议：降低学习率、检查数据预处理或减少增强强度。"
            )
            return float('nan'), 0.0, {"feature_norm": 0.0}

        self.logger.info(
            f"[验证] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个验证人, "
            f"特征范数={feat_norm_meter.avg:.4f}"
        )

        # ── 步骤 3：计算余弦相似度矩阵 ───────────────────────────────────────
        self.logger.info("[验证] 正在计算相似度矩阵...")
        # similarity_matrix: [Q, G]
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())  # [Q, G]

        # 按人级别聚合：先对每个 gallery person 取该人所有图像的最大相似度，再在 person 级别排序
        unique_persons, person_inverse = np.unique(gallery_labels_arr, return_inverse=True)
        person_indices = {p: np.where(gallery_labels_arr == p)[0] for p in unique_persons}

        # 构建 person-level similarity 矩阵 [Q, P]
        person_sims_list = []
        sim_np = similarity_matrix.numpy()
        for p in unique_persons:
            idxs = person_indices[p]
            # 取每个 query 对该人的最大相似度
            max_sims = sim_np[:, idxs].max(axis=1)
            person_sims_list.append(max_sims)
        person_sims = np.stack(person_sims_list, axis=1)  # [Q, P]

        # 计算 Rank-K（基于 person）
        P = person_sims.shape[1]
        top_k_person = min(20, P)
        top_k_person_indices = np.argsort(-person_sims, axis=1)[:, :top_k_person]  # [Q, top_k_person]
        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k_person:
                correct = 0
                for i in range(len(query_labels)):
                    predicted_persons = unique_persons[top_k_person_indices[i, :k]]
                    if query_labels[i] in predicted_persons:
                        correct += 1
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        # ── 步骤 5：EER（同人匹配 vs 异人拒绝）────────────────────────────────
        eer = 0.0
        try:
            positive_scores = []
            negative_scores = []

            # 使用固定种子保证 EER 可复现
            rng = np.random.RandomState(self.seed)

            for q_idx in range(len(query_labels)):
                q_label = query_labels[q_idx]
                q_emb = query_embeddings[q_idx]

                # 同人 Gallery：取 top-5 正样本
                same_idx = np.where(gallery_labels_arr == q_label)[0]
                if len(same_idx) > 0:
                    sims = (q_emb @ gallery_embeddings[same_idx].t()).numpy()
                    topk_k = min(5, len(sims))
                    vals = np.sort(sims)[-topk_k:]
                    positive_scores.extend(vals.tolist())

                # 异人 Gallery：随机采样 3 个
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

        # ── 步骤 6：汇总指标 ─────────────────────────────────────────────────
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
            loss=val_loss,
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
            # 构造 person-level top3
            person_top3 = unique_persons[top_k_person_indices[idx, :3]]
            person_top3_sims = person_sims[idx, :3]
            match_str = "[O]" if true_label == person_top3[0] else "[X]"
            top3_str = ", ".join(
                f"{int(l)}({s:.3f})" for l, s in zip(person_top3, person_top3_sims)
            )
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        if self.tb_writer:
            self.tb_writer.add_scalar('val/rank_1', rank1_acc, epoch)
            self.tb_writer.add_scalar('val/rank_5', rank5_acc, epoch)
            self.tb_writer.add_scalar('val/rank_10', rank10_acc, epoch)
            self.tb_writer.add_scalar('val/rank_20', rank20_acc, epoch)
            self.tb_writer.add_scalar('val/eer', eer, epoch)
            self.tb_writer.add_scalar('val/feature_norm', feat_norm_meter.avg, epoch)

        return float(val_loss), rank1_acc, metrics

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
        from torch.utils.data import DataLoader

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
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征 ───────────────────────────────────
        path_to_idx = {p: i for i, p in enumerate(test_dataset.image_paths)}
        gallery_indices = [path_to_idx[p] for p in gallery_paths if p in path_to_idx]

        class _IndexedSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

        gallery_subset = _IndexedSubset(test_dataset, gallery_indices)
        gallery_loader = DataLoader(
            gallery_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        gallery_embeddings_list = []
        feat_norm_meter = AverageMeter()

        pbar_g = tqdm(gallery_loader, desc="Test Gallery", leave=False)
        for batch in pbar_g:
            images = batch['image'].to(self.device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb_orig = self.model.extract_features(images)
            else:
                emb_orig = self.model.extract_features(images)
            embeddings = F.normalize(emb_orig, p=2, dim=1)

            feat_norm = embeddings.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_meter.update(feat_norm, embeddings.size(0))
            gallery_embeddings_list.append(embeddings.cpu())

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
        self.logger.info(
            f"[Test Gallery] {len(gallery_paths)} 张（测试集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征 ─────────────────────────────────────
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

        # ── 步骤 3：计算余弦相似度矩阵 ───────────────────────────────
        self.logger.info("[测试] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

        # 按人级别聚合并计算 Rank-K
        unique_persons, person_inverse = np.unique(gallery_labels_arr, return_inverse=True)
        person_indices = {p: np.where(gallery_labels_arr == p)[0] for p in unique_persons}
        sim_np = similarity_matrix.numpy()
        person_sims_list = []
        for p in unique_persons:
            idxs = person_indices[p]
            max_sims = sim_np[:, idxs].max(axis=1)
            person_sims_list.append(max_sims)
        person_sims = np.stack(person_sims_list, axis=1)  # [Q, P]

        P = person_sims.shape[1]
        top_k_person = min(20, P)
        top_k_person_indices = np.argsort(-person_sims, axis=1)[:, :top_k_person]
        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k_person:
                correct = 0
                for i in range(len(query_labels)):
                    predicted_persons = unique_persons[top_k_person_indices[i, :k]]
                    if query_labels[i] in predicted_persons:
                        correct += 1
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        # ── 步骤 5：EER ───────────────────────────────────────────
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
        self.logger.info("【测试集评估】")
        self.logger.info(f"  Rank-1:  {rank1_acc:.4f} ({int(rank1_acc * len(query_labels))}/{len(query_labels)})")
        self.logger.info(f"  Rank-5:  {rank5_acc:.4f}")
        self.logger.info(f"  Rank-10: {rank10_acc:.4f}")
        self.logger.info(f"  Rank-20: {rank20_acc:.4f}")
        self.logger.info(f"  EER:     {eer:.4f}" if eer > 0 else "  EER:     N/A")
        self.logger.info(f"  Gallery: {len(gallery_paths)} 张 / {len(np.unique(gallery_labels_arr))} 人")
        self.logger.info(f"  Query:   {len(query_labels)} 张 / {len(np.unique(query_labels))} 人")
        self.logger.info("=" * 60)

        # 打印样本匹配详情（前 5 个） — 使用 person-level Top-3
        self.logger.info("[测试样本] Query vs Top-3 Gallery 匹配（显示前 5 个）:")
        sample_indices = list(range(min(5, len(query_labels))))
        for idx in sample_indices:
            true_label = query_labels[idx]
            person_top3 = unique_persons[top_k_person_indices[idx, :3]]
            person_top3_sims = person_sims[idx, :3]
            match_str = "[O]" if true_label == person_top3[0] else "[X]"
            top3_str = ", ".join(
                f"{int(l)}({s:.3f})" for l, s in zip(person_top3, person_top3_sims)
            )
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 保存检查点
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch=None, is_best=False, extra=None):
        """保存检查点（仅最佳模型 + 最新模型）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "classifier_state": self.model._classifier.state_dict(),
            "epoch": epoch or 0,
            "metric_learning": self.metric_learning,
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
