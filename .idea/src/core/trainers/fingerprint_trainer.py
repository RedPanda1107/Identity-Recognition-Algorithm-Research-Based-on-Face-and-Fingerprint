import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from .base_trainer import BaseTrainer, AverageMeter


class FingerprintTrainer(BaseTrainer):
    """Fingerprint trainer with ArcFace support and feature norm monitoring."""

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 criterion, device, logger, tb_writer=None,
                 arcface_s=30.0, arcface_m=0.50):
        super(FingerprintTrainer, self).__init__(
            model, train_loader, val_loader, optimizer, scheduler, 
            criterion, device, logger, tb_writer
        )
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m
        self._setup_arcface()

    def _setup_arcface(self):
        from ..models.fingerprint_net import FingerprintNet
        
        num_classes = self.model.num_classes
        embedding_dim = self.model.get_embedding_dim()
        
        # Use existing classifier or create new ArcFace
        if self.model._classifier is None:
            from ..losses.arcface import ArcMarginProduct
            self.model._classifier = ArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                s=self.arcface_s,
                m=self.arcface_m
            ).to(self.device)
        
        self.logger.info(f"[初始化] ArcFace: s={self.arcface_s}, m={self.arcface_m}, 类别数={num_classes}")

    def train_step(self, batch):
        """Single training step with ArcFace loss."""
        inputs = batch.get("image", batch.get("input"))
        if inputs is None:
            raise ValueError("Batch must contain 'image' or 'input' key")
        
        labels = batch["label"]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        logits = self.model(inputs, labels=labels)
        
        # ArcFace returns logits when labels provided, use CrossEntropyLoss
        loss = self.criterion(logits, labels)
        
        return loss, logits

    def train_epoch(self, epoch):
        """Training epoch with feature norm tracking."""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        feat_norm_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch in pbar:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            
            loss, logits = self.train_step(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            preds = logits.argmax(dim=1)
            acc = (preds == batch["label"].to(self.device)).float().mean().item()
            
            with torch.no_grad():
                embeddings = F.normalize(self.model._extract_features(inputs.to(self.device)), p=2, dim=1)
                feat_norm = embeddings.norm(dim=1).mean().item()
            
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            feat_norm_meter.update(feat_norm, batch_size)
            
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})
        
        self.logger.info(f"[训练] Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, 准确率={acc_meter.avg:.4f}, 特征范数={feat_norm_meter.avg:.4f}")
        
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', loss_meter.avg, epoch)
            self.tb_writer.add_scalar('train/accuracy', acc_meter.avg, epoch)
            self.tb_writer.add_scalar('train/feature_norm', feat_norm_meter.avg, epoch)
        
        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Validation with Top-1 Accuracy and feature norm tracking."""
        self.model.eval()
        loss_meter = AverageMeter()
        top1_acc_meter = AverageMeter()
        feat_norm_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        all_preds, all_labels = [], []
        
        for batch in pbar:
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            inputs = inputs.to(self.device)
            labels = batch["label"].to(self.device)
            
            # Eval: use model without labels, get embeddings then compute logits via classifier
            features = self.model._extract_features(inputs)
            embeddings = F.normalize(features, p=2, dim=1)
            logits = self.model._classifier(embeddings, labels)
            loss = self.criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            top1_acc = (preds == labels).float().mean().item()
            
            feat_norm = embeddings.norm(dim=1).mean().item()
            
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_acc_meter.update(top1_acc, batch_size)
            feat_norm_meter.update(feat_norm, batch_size)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "top1": f"{top1_acc_meter.avg:.4f}"})
        
        try:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0
        
        metrics = {
            "top1_accuracy": top1_acc_meter.avg,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "feature_norm": feat_norm_meter.avg
        }
        
        self.logger.info(f"[验证] Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, 准确率={top1_acc_meter.avg:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
        
        if self.tb_writer:
            self.tb_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            self.tb_writer.add_scalar('val/top1_accuracy', top1_acc_meter.avg, epoch)
            self.tb_writer.add_scalar('val/feature_norm', feat_norm_meter.avg, epoch)
        
        return loss_meter.avg, top1_acc_meter.avg, metrics

    def save_checkpoint(self, path, epoch=None, is_best=False, extra=None):
        """保存检查点（仅最佳模型）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "arc_classifier_state": self.model.classifier.state_dict(),
            "epoch": epoch or 0,
        }
        if extra:
            state.update(extra)

        # 只保存最佳模型
        if is_best:
            torch.save(state, path)
            self.logger.info(f"[保存] 最佳模型: {path}")
        else:
            # 临时保存Latest用于恢复训练
            latest_path = path.replace(".pth", "_latest.pth")
            torch.save(state, latest_path)

        return path
