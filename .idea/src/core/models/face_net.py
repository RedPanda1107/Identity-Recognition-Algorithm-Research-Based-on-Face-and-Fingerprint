import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from ..losses.arcface import ArcMarginProduct
from ..modules.attention import SpatialAttention


class FaceNet(nn.Module):
    """Enhanced Face Feature Extraction Network with Attention Mechanisms.

    Features:
    - ResNet50 backbone
    - Spatial Attention for local feature enhancement
    - Channel Attention for feature weighting
    - L2-normalized embeddings for better fusion
    - Unified feature extraction API (compatible with FingerprintNet)
    """

    def __init__(self, num_classes=100, embedding_dim=512, pretrained=True,
                 dropout_rate=0.5, spatial_attention=True):
        """
        Args:
            num_classes: Number of identity classes
            embedding_dim: Feature embedding dimension (512 for face, 256 for fingerprint)
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout probability
            spatial_attention: Enable spatial attention module
        """
        super(FaceNet, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.spatial_attention_enabled = spatial_attention

        # ResNet50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)
        self.backbone_out_channels = 2048  # ResNet50 output channels

        # Remove final FC layer
        self.backbone.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Spatial Attention (same as fingerprint)
        if self.spatial_attention_enabled:
            self.spatial_attn = SpatialAttention(self.backbone_out_channels, reduction_ratio=16)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection (移除ReLU，避免特征"死锁")
        # 注意：Projection后接 L2归一化，特征可正可负，ReLU会破坏这一点
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout_rate)
        )

        # Channel Attention (same as fingerprint)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.Sigmoid()
        )

        # Placeholder for ArcFace classifier (set later via setter)
        self._classifier = None

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重（只初始化新增的层，不影响预训练backbone）"""
        for module in self.modules():
            # 只初始化新增的层（不在backbone中）
            if isinstance(module, nn.Linear) and module.in_features not in [2048, 512, 256]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d) and module.in_channels not in [3, 64, 128, 256, 512]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm权重初始化已被预训练权重覆盖，跳过
                pass

    def _extract_features(self, x):
        """Unified feature extraction pipeline (same API as FingerprintNet).

        This enables easy feature fusion between face and fingerprint modalities.
        """
        # Backbone feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        backbone_features = self.backbone.layer4(x)

        # Spatial Attention
        if self.spatial_attention_enabled:
            backbone_features = self.spatial_attn(backbone_features)

        # Global pooling
        pooled = self.global_pool(backbone_features)
        pooled = pooled.view(pooled.size(0), -1)

        # Feature projection
        features = self.feature_projection(pooled)

        # Channel Attention
        attention_weights = self.attention(features)
        features = features * attention_weights

        return features

    def forward(self, x, labels=None, return_features=False):
        """Forward pass with ArcFace support during training.

        Args:
            x: Input image tensor [batch, 3, 224, 224]
            labels: Ground truth labels for ArcFace (optional)
            return_features: Return features along with logits

        Returns:
            Training mode: logits (if classifier set) or embeddings
            Eval mode: logits or embeddings
        """
        features = self._extract_features(x)
        embeddings = F.normalize(features, p=2, dim=1)

        if self.training:
            if labels is not None and self._classifier is not None:
                return self._classifier(embeddings, labels)
            return embeddings

        # Eval mode - ArcFace classifier needs labels too
        if self._classifier is not None:
            if return_features:
                logits = self._classifier(embeddings, labels) if labels is not None else self._classifier(embeddings)
                return logits, embeddings
            return self._classifier(embeddings, labels) if labels is not None else self._classifier(embeddings)

        # No classifier, return embeddings
        if return_features:
            return embeddings, embeddings
        return embeddings

    @property
    def classifier(self):
        """Get classifier (for compatibility with ArcFace trainer)."""
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        """Set classifier (for ArcFace integration)."""
        self._classifier = value

    def extract_features(self, x):
        """Extract L2-normalized feature vectors for inference.

        Compatible with FingerprintNet API for unified feature extraction.
        """
        with torch.no_grad():
            features = self._extract_features(x)
            embeddings = F.normalize(features, p=2, dim=1)
        return embeddings

    def get_embedding_dim(self):
        """Get feature embedding dimension (for fusion alignment)."""
        return self.embedding_dim


def create_face_model(model_type='facenet', **kwargs):
    """Factory function: Create face recognition model.

    Model types:
    - 'facenet': Standard face net with attention mechanisms

    Compatible with create_fingerprint_model() API.
    """
    if model_type.lower() == 'facenet':
        return FaceNet(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_face_embedding_dim():
    """Get standard face embedding dimension."""
    return 512
