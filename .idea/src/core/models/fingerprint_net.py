import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class FingerprintNet(nn.Module):
    """Fingerprint feature extraction network.

    Architecture:
        1. ResNet50 backbone (ImageNet pretrained, with modifications for fingerprints)
        2. Feature projection head (Linear → BatchNorm → ReLU → Linear → BatchNorm → Dropout)
        3. Optional classification head (Linear, set via setter)

    Staged training support:
        - Stage 1: freeze backbone, train only classifier + projection head
        - Stage 2: unfreeze backbone, fine-tune with small learning rate
    """

    # 冻结层级定义：数字越大，解冻范围越广
    FREEZE_CONV1 = 0   # 只解冻 conv1 + bn1
    FREEZE_L1 = 1      # 解冻到 layer1
    FREEZE_L2 = 2      # 解冻到 layer2
    FREEZE_L3 = 3      # 解冻到 layer3
    FREEZE_L4 = 4      # 解冻到 layer4（全解冻）
    FREEZE_ALL = 5     # 全部解冻（含 spatial_attn, global_pool, projection）

    def __init__(self, num_classes=6000, embedding_dim=512, pretrained=False,
                 dropout_rate=0.5, spatial_attention=True):
        super(FingerprintNet, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.spatial_attention_enabled = spatial_attention

        # ── Backbone ──────────────────────────────────────────────────────────
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)
        self.backbone_out_channels = 2048  # ResNet50 output channels

        # Use standard ResNet50 first conv (7x7 stride=2) + maxpool
        # Fingerprint ridge details are preserved through augmentation (rotation, translate, scale),
        # not through architectural changes. The 3x3/5x5 + no-maxpool design was the root cause
        # of the 5x slowdown: it produced 15x15 vs face's 7x7 feature maps in layer4,
        # causing CUDA OOM thrashing on 6GB GPU.
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and weights is not None:
            # Use pretrained 7x7 weights directly (no cropping needed)
            old_w = self.backbone.conv1.weight.detach()
            new_w = old_w  # keep full 7x7
            self.backbone.conv1.weight.data = new_w
            del old_w

        # Restore standard ResNet50 maxpool: 112x112 -> 56x56
        # With 7x7 conv1, maxpool output = ceil(224/2/2) = 56x56
        # Layer4 then produces 7x7 (same as face), avoiding CUDA memory thrashing
        self.backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Remove the final FC layer (replaced by feature_projection)
        self.backbone.fc = nn.Identity()

        # ── Backbone 冻结状态标记（用于追踪训练阶段）──────────────────────────────
        self._freeze_level = self.FREEZE_L4  # 默认冻结到 layer4（仅 backbone 可训练）

        # ── Feature extraction components ───────────────────────────────────────
        if self.spatial_attention_enabled:
            from ..modules.attention import SpatialAttention
            self.spatial_attn = SpatialAttention(self.backbone_out_channels, reduction_ratio=16)
        else:
            self.spatial_attn = None

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer4_dropout = nn.Dropout(0.3)

        # Feature projection: Linear → BatchNorm → ReLU → Linear → BatchNorm → Dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_out_channels, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout_rate)
        )

        # Classification head (set via setter, not part of feature extraction)
        self._classifier = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize newly added layers with Kaiming normal."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                if module.weight.requires_grad and module.in_channels != 3:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # 分级冻结 / 解冻工具
    # ─────────────────────────────────────────────────────────────────────────
    def freeze_until(self, level):
        """Freeze backbone up to a specified level.

        Args:
            level: One of FREEZE_* constants
                FREEZE_CONV1: Only conv1 + bn1 trainable (backbone mostly frozen)
                FREEZE_L1:    Up to layer1 trainable
                FREEZE_L2:    Up to layer2 trainable
                FREEZE_L3:    Up to layer3 trainable
                FREEZE_L4:    Up to layer4 trainable (backbone full, projection frozen)
                FREEZE_ALL:   Everything trainable (backbone + projection + classifier)
        """
        self._freeze_level = level

        # Helper: set requires_grad for a module's parameters
        def _set_requires_grad(module, trainable):
            for param in module.parameters():
                param.requires_grad = trainable

        # Freeze all backbone first
        _set_requires_grad(self.backbone, False)

        # Backbone freeze levels
        if level >= self.FREEZE_CONV1:
            _set_requires_grad(self.backbone.conv1, True)
            _set_requires_grad(self.backbone.bn1, True)

        if level >= self.FREEZE_L1:
            _set_requires_grad(self.backbone.layer1, True)

        if level >= self.FREEZE_L2:
            _set_requires_grad(self.backbone.layer2, True)

        if level >= self.FREEZE_L3:
            _set_requires_grad(self.backbone.layer3, True)

        if level >= self.FREEZE_L4:
            _set_requires_grad(self.backbone.layer4, True)

        # Feature extraction heads (projector, attention, pool)
        if level >= self.FREEZE_ALL:
            if self.spatial_attn is not None:
                _set_requires_grad(self.spatial_attn, True)
            _set_requires_grad(self.global_pool, True)
            _set_requires_grad(self.layer4_dropout, True)
            _set_requires_grad(self.feature_projection, True)

        # Classifier (always trainable if present)
        if self._classifier is not None:
            _set_requires_grad(self._classifier, True)

    def get_trainable_params_info(self):
        """Return info about trainable vs frozen parameters."""
        total = 0
        trainable = 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        trainable_names = []
        frozen_names = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": trainable / total * 100,
            "trainable_names": trainable_names,
            "frozen_names": frozen_names[:10],  # 前 10 个冻结层（避免太长）
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 特征提取（前向核心）
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_features(self, x):
        """Extract L2-normalized feature embeddings."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.layer4_dropout(x)

        if self.spatial_attn is not None:
            x = self.spatial_attn(x)

        pooled = self.global_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        features = self.feature_projection(pooled)

        # L2 normalize for metric consistency
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings

    # ─────────────────────────────────────────────────────────────────────────
    # 前向传播（统一接口）
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x, labels=None, return_features=False):
        """Unified forward pass.

        Args:
            x: Input images [B, 3, H, W]
            labels: Ignored (kept for API compatibility with ArcFace path)
            return_features: If True, return (logits_or_features, embeddings)

        Returns:
            - return_features=True: (logits, embeddings) or (embeddings, embeddings) if no classifier
            - return_features=False: logits or embeddings or (no classifier → embeddings)
        """
        embeddings = self._extract_features(x)

        if self._classifier is None:
            if return_features:
                return embeddings, embeddings
            return embeddings

        # Classification with optional ArcFace
        if self.training and labels is not None:
            # ArcFace 等自定义分类器支持 labels 参数做 margin
            # nn.Linear 只接受 (input,) 一个参数；其他自定义分类器（如 ArcFace）接受 (input, labels)
            if isinstance(self._classifier, nn.Linear):
                logits = self._classifier(embeddings)
            else:
                logits = self._classifier(embeddings, labels)
        else:
            # Eval mode or no labels: raw classifier output (no margin)
            logits = self._classifier(embeddings)

        if return_features:
            return logits, embeddings
        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # 分类器设置（API 兼容）
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value
        if value is not None:
            for p in value.parameters():
                p.requires_grad = True

    def extract_features(self, x):
        """Extract L2-normalized features for inference.

        Note: Does NOT use no_grad to allow gradient flow during fusion training.
        """
        return self._extract_features(x)

    def get_embedding_dim(self):
        return self.embedding_dim


def create_fingerprint_model(model_type='fingerprint_net', **kwargs):
    """Factory function: Create fingerprint recognition model."""
    if model_type.lower() == 'fingerprint_net':
        return FingerprintNet(**kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


def get_fingerprint_embedding_dim():
    """Get standard fingerprint embedding dimension."""
    return 512
