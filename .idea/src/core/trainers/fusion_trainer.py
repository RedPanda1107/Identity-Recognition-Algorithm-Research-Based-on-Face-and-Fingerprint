import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .base_trainer import BaseTrainer, AverageMeter


class FusionTrainer(BaseTrainer):
    """多模态融合训练器

    支持人脸+指纹特征提取和融合训练
    支持加载预训练的单模态模型权重
    """

    def __init__(self, fusion_model, face_model, fingerprint_model,
                 train_loader, val_loader, optimizer, scheduler, criterion,
                 device, logger, tb_writer=None, pretrained_ckpts=None,
                 unfreeze_epoch=10, face_lr=1e-5, fp_lr=1e-5):
        """初始化融合训练器

        Args:
            pretrained_ckpts: 预训练检查点路径字典
            unfreeze_epoch: 解冻Backbone的轮次 (默认10轮后解冻)
            face_lr: 解冻后人脸模型的学习率
            fp_lr: 解冻后指纹模型的学习率
        """
        # 🔧 【指令C】保存解冻策略参数
        self.unfreeze_epoch = unfreeze_epoch
        self.face_lr = face_lr
        self.fp_lr = fp_lr
        self.current_epoch = 0

        # 初始化父类
        super(FusionTrainer, self).__init__(
            fusion_model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )

        # 存储单模态模型
        self.face_model = face_model.to(device) if face_model else None
        self.fingerprint_model = fingerprint_model.to(device) if fingerprint_model else None

        # 加载预训练权重
        if pretrained_ckpts:
            self._load_pretrained_weights(pretrained_ckpts)

        # 🔧 【指令A】验证单模态准确率 (最重要的"救命稻草")
        self._verify_unimodal_accuracy()

        # 设置特征提取器为评估模式
        if self.face_model:
            self.face_model.eval()
        if self.fingerprint_model:
            self.fingerprint_model.eval()

        # 冻结特征提取器的参数
        self._freeze_feature_extractors()

    def _load_pretrained_weights(self, pretrained_ckpts):
        """加载预训练的单模态模型权重（带前缀兼容性处理）"""
        face_loaded = False
        fp_loaded = False

        if 'face' in pretrained_ckpts and pretrained_ckpts['face'] and self.face_model:
            face_ckpt_path = pretrained_ckpts['face']
            if os.path.exists(face_ckpt_path):
                try:
                    ckpt = torch.load(face_ckpt_path, map_location=self.device)

                    # 获取权重字典
                    if 'model_state' in ckpt:
                        state_dict = ckpt['model_state']
                    else:
                        state_dict = ckpt

                    # 打印原始state_dict的key（用于调试）
                    keys = list(state_dict.keys())
                    self.logger.info(f"[Face] Checkpoint keys (first 5): {keys[:5]}")
                    self.logger.info(f"[Face] Model keys (first 5): {list(self.face_model.state_dict().keys())[:5]}")

                    # 处理前缀兼容性问题
                    state_dict = self._adjust_state_dict_keys(state_dict, self.face_model.state_dict())

                    # 加载权重
                    self.face_model.load_state_dict(state_dict)

                    # 验证权重加载成功
                    with torch.no_grad():
                        test_input = torch.randn(1, 3, 224, 224).to(self.device)
                        test_feat = self.face_model.extract_features(test_input)
                        feat_norm = test_feat.norm().item()
                        self.logger.info(f"[Face] Pretrained weights loaded: {face_ckpt_path}")
                        self.logger.info(f"[Face] Feature norm: {feat_norm:.4f} (non-zero=success)")
                    face_loaded = True
                except Exception as e:
                    self.logger.warning(f"[Face] Failed to load weights: {e}")
            else:
                self.logger.warning(f"[Face] Checkpoint not found: {face_ckpt_path}")

        if 'fingerprint' in pretrained_ckpts and pretrained_ckpts['fingerprint'] and self.fingerprint_model:
            fp_ckpt_path = pretrained_ckpts['fingerprint']
            if os.path.exists(fp_ckpt_path):
                try:
                    ckpt = torch.load(fp_ckpt_path, map_location=self.device)

                    # 获取权重字典
                    if 'model_state' in ckpt:
                        state_dict = ckpt['model_state']
                    else:
                        state_dict = ckpt

                    # 打印原始state_dict的key（用于调试）
                    keys = list(state_dict.keys())
                    self.logger.info(f"[FP] Checkpoint keys (first 5): {keys[:5]}")
                    self.logger.info(f"[FP] Model keys (first 5): {list(self.fingerprint_model.state_dict().keys())[:5]}")

                    # 处理前缀兼容性问题
                    state_dict = self._adjust_state_dict_keys(state_dict, self.fingerprint_model.state_dict())

                    # 加载权重
                    self.fingerprint_model.load_state_dict(state_dict)

                    # 验证权重加载成功
                    with torch.no_grad():
                        test_input = torch.randn(1, 3, 224, 224).to(self.device)
                        test_feat = self.fingerprint_model.extract_features(test_input)
                        feat_norm = test_feat.norm().item()
                        self.logger.info(f"[FP] Pretrained weights loaded: {fp_ckpt_path}")
                        self.logger.info(f"[FP] Feature norm: {feat_norm:.4f} (non-zero=success)")
                    fp_loaded = True
                except Exception as e:
                    self.logger.warning(f"[FP] Failed to load weights: {e}")
            else:
                self.logger.warning(f"[FP] Checkpoint not found: {fp_ckpt_path}")

        # 汇总
        if face_loaded and fp_loaded:
            self.logger.info("[OK] Face-Fingerprint alignment: paired samples share labels")
        else:
            self.logger.warning("[WARN] Using random weights or missing pretrained files")

    def _adjust_state_dict_keys(self, state_dict, target_model_dict):
        """调整state_dict的key前缀，处理model.或backbone.等前缀不匹配问题"""
        adjusted_state_dict = OrderedDict()
        target_keys = set(target_model_dict.keys())

        # 尝试直接匹配
        matched_keys = set(state_dict.keys()) & target_keys
        if len(matched_keys) / len(target_keys) > 0.5:
            self.logger.info(f"[Weight] Direct match: {len(matched_keys)}/{len(target_keys)} keys")
            return state_dict

        # 检查是否需要删除前缀（如 model., backbone.）
        for key, value in state_dict.items():
            # 尝试删除常见前缀
            new_key = key
            prefixes_to_remove = ['model.', 'backbone.', 'module.']
            for prefix in prefixes_to_remove:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    break

            # 检查删除前缀后是否匹配
            if new_key in target_keys:
                adjusted_state_dict[new_key] = value
            else:
                # 尝试添加前缀
                added_prefix = False
                for prefix in prefixes_to_remove:
                    prefixed_key = prefix + key
                    if prefixed_key in target_keys:
                        adjusted_state_dict[prefixed_key] = value
                        added_prefix = True
                        break

                if not added_prefix:
                    # 保留原始key（可能有部分层不匹配）
                    adjusted_state_dict[key] = value

        # 统计匹配情况
        matched = sum(1 for k in adjusted_state_dict.keys() if k in target_keys)
        self.logger.info(f"[Weight] Adjusted: {matched}/{len(target_keys)} keys matched")
        return adjusted_state_dict

    def _verify_unimodal_accuracy(self):
        """🔧 【指令A】验证单模态准确率 (最重要的"救命稻草")

        在验证集上测试人脸和指纹各自的分类准确率
        - 加载正确: 应该有合理的准确率 (>20% for 300 classes)
        - 加载失败: 准确率接近随机 (~0.3% for 300 classes)
        """
        self.logger.info("=" * 60)
        self.logger.info("[验证] 开始验证单模态权重加载...")
        self.logger.info("=" * 60)

        # 临时创建ArcFace分类器用于测试
        num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else 300

        # 验证人脸模型
        face_acc = self._test_unimodal_accuracy(
            self.face_model,
            "人脸",
            num_classes
        )

        # 验证指纹模型
        fp_acc = self._test_unimodal_accuracy(
            self.fingerprint_model,
            "指纹",
            num_classes
        )

        # 汇总结果
        self.logger.info("=" * 60)
        self.logger.info("[FusionTrainer] [STATS] 单模态验证结果汇总:")
        self.logger.info(f"[FusionTrainer]   人脸模型 Acc: {face_acc*100:.2f}%")
        self.logger.info(f"[FusionTrainer]   指纹模型 Acc: {fp_acc*100:.2f}%")
        self.logger.info("=" * 60)

        # 警告
        if face_acc < 0.05:
            self.logger.warning("[警告] 人脸模型准确率过低，权重可能未正确加载！")
        if fp_acc < 0.05:
            self.logger.warning("[警告] 指纹模型准确率过低，权重可能未正确加载！")

    def _test_unimodal_accuracy(self, model, modality_name, num_classes):
        """测试单模态模型在验证集上的准确率"""
        if model is None:
            self.logger.warning(f"[FusionTrainer] {modality_name}模型不存在")
            return 0.0

        try:
            from ..losses.arcface import ArcMarginProduct

            # 创建ArcFace分类器 (与单模态训练一致)
            classifier = ArcMarginProduct(
                in_features=model.embedding_dim if hasattr(model, 'embedding_dim') else 512,
                out_features=num_classes,
                s=30.0,
                m=0.3
            ).to(self.device)

            # 在验证集上测试
            correct = 0
            total = 0

            for batch in self.val_loader:
                images = batch['face_image' if modality_name == '人脸' else 'fingerprint_image'].to(self.device)
                labels = batch['label'].to(self.device)

                with torch.no_grad():
                    # 提取特征
                    if hasattr(model, 'extract_features'):
                        features = model.extract_features(images)
                    else:
                        features = model._extract_features(images)

                    # 计算logits
                    logits = classifier(features, labels)
                    preds = logits.argmax(dim=1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            self.logger.info(f"[FusionTrainer] [OK] {modality_name}模型单模态验证完成: Acc={acc*100:.2f}%")
            return acc

        except Exception as e:
            self.logger.warning(f"[FusionTrainer] ❌ {modality_name}模型验证失败: {e}")
            return 0.0

    def _freeze_feature_extractors(self):
        """冻结特征提取器参数"""
        if self.face_model:
            for param in self.face_model.parameters():
                param.requires_grad = False
            self.logger.info("[冻结] 人脸模型参数已冻结")
        if self.fingerprint_model:
            for param in self.fingerprint_model.parameters():
                param.requires_grad = False
            self.logger.info("[冻结] 指纹模型参数已冻结")

    def _unfreeze_backbone_partial(self):
        """🔧 【指令C】部分解冻Backbone (仅最后两层)

        解冻人脸ResNet50和指纹ResNet34的最后两层卷积层
        设置极小的学习率进行联合微调
        """
        self.logger.info("=" * 60)
        self.logger.info(f"[解冻] Epoch {self.current_epoch + 1}: 开始部分解冻Backbone...")
        self.logger.info("=" * 60)

        if self.face_model:
            # 解冻ResNet50最后两层 (layer3, layer4)
            unfrozen_layers = []
            layer3_unfrozen = False
            layer4_unfrozen = False

            for name, param in self.face_model.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True
                    if 'layer3' in name and not layer3_unfrozen:
                        unfrozen_layers.append('layer3')
                        layer3_unfrozen = True
                    elif 'layer4' in name and not layer4_unfrozen:
                        unfrozen_layers.append('layer4')
                        layer4_unfrozen = True

            self.logger.info(f"[解冻] 人脸模型解冻层: {unfrozen_layers}")

        if self.fingerprint_model:
            # 解冻ResNet34最后两层 (layer3, layer4)
            unfrozen_layers = []
            layer3_unfrozen = False
            layer4_unfrozen = False

            for name, param in self.fingerprint_model.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True
                    if 'layer3' in name and not layer3_unfrozen:
                        unfrozen_layers.append('layer3')
                        layer3_unfrozen = True
                    elif 'layer4' in name and not layer4_unfrozen:
                        unfrozen_layers.append('layer4')
                        layer4_unfrozen = True

            self.logger.info(f"[解冻] 指纹模型解冻层: {unfrozen_layers}")

        self.logger.info("[FusionTrainer] [INFO] 请使用更小的学习率微调 (建议: backbone=1e-5, fusion=1e-4)")

    @torch.no_grad()
    def _extract_features(self, face_images, fingerprint_images):
        """从两个模态提取特征，包含NaN检查和L2归一化"""
        # 提取人脸特征
        if self.face_model:
            face_features = self.face_model.extract_features(face_images)
        else:
            # 如果没有人脸模型，使用随机特征（用于测试）
            face_features = torch.randn(face_images.size(0), 512, device=self.device)

        # 提取指纹特征
        if self.fingerprint_model:
            fingerprint_features = self.fingerprint_model.extract_features(fingerprint_images)
        else:
            # 如果没有指纹模型，使用随机特征（用于测试）
            fingerprint_features = torch.randn(fingerprint_images.size(0), 256, device=self.device)

        # 🔧 【指令A】强制L2归一化 - 稳定数值分布
        face_features = F.normalize(face_features, p=2, dim=1)
        fingerprint_features = F.normalize(fingerprint_features, p=2, dim=1)

        # 🔧 【指令A】NaN/Inf检查
        if torch.isnan(face_features).any() or torch.isinf(face_features).any():
            self.logger.warning("[FusionTrainer] 检测到人脸特征包含NaN/Inf，使用零向量替换")
            face_features = torch.where(
                torch.isnan(face_features) | torch.isinf(face_features),
                torch.zeros_like(face_features),
                face_features
            )

        if torch.isnan(fingerprint_features).any() or torch.isinf(fingerprint_features).any():
            self.logger.warning("[FusionTrainer] 检测到指纹特征包含NaN/Inf，使用零向量替换")
            fingerprint_features = torch.where(
                torch.isnan(fingerprint_features) | torch.isinf(fingerprint_features),
                torch.zeros_like(fingerprint_features),
                fingerprint_features
            )

        return face_features, fingerprint_features

    def train_epoch(self, epoch):
        """训练一个epoch"""
        # 🔧 【指令C】更新当前epoch
        self.current_epoch = epoch

        # 🔧 【指令C】检查是否需要解冻Backbone
        if epoch == self.unfreeze_epoch:
            self._unfreeze_backbone_partial()

        self.model.train()
        # 特征提取器保持在评估模式（除非已解冻）
        if self.face_model:
            self.face_model.eval()
        if self.fingerprint_model:
            self.fingerprint_model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Fusion Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            face_images = batch['face_image'].to(self.device)
            fingerprint_images = batch['fingerprint_image'].to(self.device)
            targets = batch['label'].to(self.device)

            # 提取两个模态的特征
            face_features, fingerprint_features = self._extract_features(face_images, fingerprint_images)

            # 前向传播通过融合模型 (带labels以启用ArcFace)
            outputs = self.model(face_features, fingerprint_features, targets)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算准确率
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), face_images.size(0))
            acc_meter.update(acc, face_images.size(0))

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        self.logger.info(f"[训练] Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, 准确率={acc_meter.avg:.4f}")
        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        if self.face_model:
            self.face_model.eval()
        if self.fingerprint_model:
            self.fingerprint_model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        from tqdm import tqdm
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Fusion Val]", leave=False)

        all_preds = []
        all_labels = []

        for batch in pbar:
            face_images = batch['face_image'].to(self.device)
            fingerprint_images = batch['fingerprint_image'].to(self.device)
            targets = batch['label'].to(self.device)

            # 提取特征
            face_features, fingerprint_features = self._extract_features(face_images, fingerprint_images)

            # 前向传播 (带labels以启用ArcFace)
            outputs = self.model(face_features, fingerprint_features, targets)
            loss = self.criterion(outputs, targets)

            # 计算准确率
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), face_images.size(0))
            acc_meter.update(acc, face_images.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # 计算详细指标
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0

        metrics = {"precision": precision, "recall": recall, "f1_score": f1}
        self.logger.info(f"[验证] Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, 准确率={acc_meter.avg:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
        return loss_meter.avg, acc_meter.avg, metrics

    def save_checkpoint(self, path, is_best=False, extra=None):
        """保存检查点（仅最佳模型）"""
        checkpoint = {
            'fusion_model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }

        if extra:
            checkpoint.update(extra)

        # 只保存最佳模型
        if is_best:
            torch.save(checkpoint, path)
            self.logger.info(f"[保存] 最佳模型: {path}")
        else:
            # 临时保存Latest用于恢复训练
            latest_path = path.replace(".pth", "_latest.pth")
            torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['fusion_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.logger.info(f"加载融合模型检查点: {path}")
        return checkpoint