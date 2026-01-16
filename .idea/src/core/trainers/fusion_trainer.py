import torch
from .base_trainer import BaseTrainer


class FusionTrainer(BaseTrainer):
    """Trainer for fusion model that handles face + fingerprint modalities.

    Expects face_model and fingerprint_model to be provided for feature extraction.
    """

    def __init__(self, fusion_model, face_model, fingerprint_model,
                 train_loader, val_loader, optimizer, scheduler, criterion,
                 device, logger, tb_writer=None):
        # Initialize with fusion model as main model
        super(FusionTrainer, self).__init__(
            fusion_model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )

        # Store modality-specific models
        self.face_model = face_model.to(device)
        self.fingerprint_model = fingerprint_model.to(device)

        # Set to eval mode for feature extraction
        self.face_model.eval()
        self.fingerprint_model.eval()

        # Disable gradients for feature extractors
        for param in self.face_model.parameters():
            param.requires_grad = False
        for param in self.fingerprint_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _extract_features(self, face_images, fingerprint_images):
        """Extract features from both modalities."""
        # Extract face features
        face_features = self.face_model.extract_features(face_images)

        # Extract fingerprint features (placeholder for now)
        # TODO: Replace with actual fingerprint feature extraction
        fingerprint_features = self.fingerprint_model.extract_features(fingerprint_images) \
            if self.fingerprint_model is not None else face_features  # Placeholder

        return face_features, fingerprint_features

    def train_epoch(self, epoch):
        self.model.train()
        self.face_model.eval()  # Keep feature extractors in eval mode
        self.fingerprint_model.eval()

        loss_meter = self.__class__.__bases__[0].__dict__['AverageMeter']()
        acc_meter = self.__class__.__bases__[0].__dict__['AverageMeter']()

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Fusion Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            face_images = batch.get('face_image', batch.get('image')).to(self.device)
            fingerprint_images = batch.get('fingerprint_image', batch.get('image')).to(self.device)
            targets = batch['label'].to(self.device)

            # Extract features from both modalities
            face_features, fingerprint_features = self._extract_features(face_images, fingerprint_images)

            # Forward through fusion model
            outputs = self.model(face_features, fingerprint_features)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), face_images.size(0))
            acc_meter.update(acc, face_images.size(0))

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        self.face_model.eval()
        self.fingerprint_model.eval()

        loss_meter = self.__class__.__bases__[0].__dict__['AverageMeter']()
        acc_meter = self.__class__.__bases__[0].__dict__['AverageMeter']()

        from tqdm import tqdm
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Fusion Val]", leave=False)

        all_preds = []
        all_labels = []

        for batch in pbar:
            face_images = batch.get('face_image', batch.get('image')).to(self.device)
            fingerprint_images = batch.get('fingerprint_image', batch.get('image')).to(self.device)
            targets = batch['label'].to(self.device)

            # Extract features from both modalities
            face_features, fingerprint_features = self._extract_features(face_images, fingerprint_images)

            # Forward through fusion model
            outputs = self.model(face_features, fingerprint_features)
            loss = self.criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), face_images.size(0))
            acc_meter.update(acc, face_images.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # Calculate metrics
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0

        metrics = {"precision": precision, "recall": recall, "f1_score": f1}
        return loss_meter.avg, acc_meter.avg, metrics