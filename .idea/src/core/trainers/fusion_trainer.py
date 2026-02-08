import os
import torch
from .base_trainer import BaseTrainer, AverageMeter


class FusionTrainer(BaseTrainer):
    """多模态融合训练器

    支持人脸+指纹特征提取和融合训练
    支持加载预训练的单模态模型权重
    """

    def __init__(self, fusion_model, face_model, fingerprint_model,
                 train_loader, val_loader, optimizer, scheduler, criterion,
                 device, logger, tb_writer=None, pretrained_ckpts=None):
        """初始化融合训练器

        Args:
            pretrained_ckpts: 预训练检查点路径字典，例如:
                {
                    'face': 'checkpoints/face/best_model.pth',
                    'fingerprint': 'checkpoints/fingerprint/best_model.pth'
                }
        """
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

        # 设置特征提取器为评估模式
        if self.face_model:
            self.face_model.eval()
        if self.fingerprint_model:
            self.fingerprint_model.eval()

        # 冻结特征提取器的参数
        self._freeze_feature_extractors()

    def _load_pretrained_weights(self, pretrained_ckpts):
        """加载预训练的单模态模型权重"""
        if 'face' in pretrained_ckpts and pretrained_ckpts['face'] and self.face_model:
            face_ckpt_path = pretrained_ckpts['face']
            if os.path.exists(face_ckpt_path):
                try:
                    ckpt = torch.load(face_ckpt_path, map_location=self.device)
                    # 尝试加载模型权重
                    if 'model_state' in ckpt:
                        self.face_model.load_state_dict(ckpt['model_state'])
                    else:
                        self.face_model.load_state_dict(ckpt)
                    self.logger.info(f"[FusionTrainer] 加载人脸预训练权重: {face_ckpt_path}")
                except Exception as e:
                    self.logger.warning(f"[FusionTrainer] 加载人脸权重失败: {e}")

        if 'fingerprint' in pretrained_ckpts and pretrained_ckpts['fingerprint'] and self.fingerprint_model:
            fp_ckpt_path = pretrained_ckpts['fingerprint']
            if os.path.exists(fp_ckpt_path):
                try:
                    ckpt = torch.load(fp_ckpt_path, map_location=self.device)
                    if 'model_state' in ckpt:
                        self.fingerprint_model.load_state_dict(ckpt['model_state'])
                    else:
                        self.fingerprint_model.load_state_dict(ckpt)
                    self.logger.info(f"[FusionTrainer] 加载指纹预训练权重: {fp_ckpt_path}")
                except Exception as e:
                    self.logger.warning(f"[FusionTrainer] 加载指纹权重失败: {e}")

    def _freeze_feature_extractors(self):
        """冻结特征提取器参数"""
        if self.face_model:
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.fingerprint_model:
            for param in self.fingerprint_model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _extract_features(self, face_images, fingerprint_images):
        """从两个模态提取特征"""
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

        return face_features, fingerprint_features

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        # 特征提取器保持在评估模式
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

            # 前向传播通过融合模型
            outputs = self.model(face_features, fingerprint_features)
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

            # 前向传播
            outputs = self.model(face_features, fingerprint_features)
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
        return loss_meter.avg, acc_meter.avg, metrics

    def save_checkpoint(self, path, extra=None):
        """保存检查点，包含所有模型"""
        checkpoint = {
            'fusion_model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        self.logger.info(f"保存融合模型检查点: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['fusion_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.logger.info(f"加载融合模型检查点: {path}")
        return checkpoint