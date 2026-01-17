import os
import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


class AverageMeter:
    """Simple average meter used inside trainer (local to avoid extra deps)."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


class BaseTrainer:
    """Minimal, reusable trainer with concise logging and TB optional hooks."""

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, logger, tb_writer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.tb_writer = tb_writer

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Get inputs - prefer 'image' key, fallback to 'input'
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            targets = batch["label"]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # scheduler step is expected to be called externally or by subclass
        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        all_preds = []
        all_labels = []
        all_probs = []  # 新增：收集预测概率
        for batch in pbar:
            # Get inputs - prefer 'image' key, fallback to 'input'
            inputs = batch.get("image", batch.get("input"))
            if inputs is None:
                raise ValueError("Batch must contain 'image' or 'input' key")
            targets = batch["label"]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 计算预测概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())  # 保存预测概率

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # Calculate metrics
        try:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0  # fallback for single-class cases

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs  # 新增：返回预测概率
        }
        return loss_meter.avg, acc_meter.avg, metrics

    def save_checkpoint(self, path, extra=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

