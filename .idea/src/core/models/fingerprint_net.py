import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from ..losses.arcface import ArcMarginProduct


class SpatialAttention(nn.Module):
    """Spatial Attention Module for fingerprint local detail enhancement."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialAttention, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        spatial_features = self.conv1x1(x)
        pooled = self.global_pool(spatial_features)
        attention = self.fusion(pooled)
        return x * attention


class FingerprintNet(nn.Module):
    """Fingerprint feature extraction network with Spatial Attention and ArcFace support."""

    def __init__(self, num_classes=6000, embedding_dim=256, pretrained=False, 
                 dropout_rate=0.5, spatial_attention=True):
        super(FingerprintNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.spatial_attention_enabled = spatial_attention

        # ResNet34 backbone
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet34(weights=weights)
        self.backbone_out_channels = 512

        # Remove final FC layer
        self.backbone.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Spatial Attention
        if self.spatial_attention_enabled:
            self.spatial_attn = SpatialAttention(self.backbone_out_channels, reduction_ratio=16)

        # Global pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Channel Attention
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _extract_features(self, x):
        """Unified feature extraction pipeline."""
        # Backbone
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
        """Forward pass with ArcFace support during training."""
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
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    def extract_features(self, x):
        """Extract L2-normalized feature vectors for inference."""
        with torch.no_grad():
            features = self._extract_features(x)
            embeddings = F.normalize(features, p=2, dim=1)
        return embeddings

    def get_embedding_dim(self):
        return self.embedding_dim


def create_fingerprint_model(model_type='fingerprint_net', **kwargs):
    if model_type.lower() == 'fingerprint_net':
        return FingerprintNet(**kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    model = FingerprintNet(num_classes=6000, embedding_dim=256)
    model.classifier = ArcMarginProduct(256, 6000, s=30.0, m=0.50).to(torch.device('cpu'))

    x = torch.randn(4, 3, 224, 224)
    model.train()
    labels = torch.tensor([0, 1, 2, 3])
    logits = model(x, labels=labels)
    print(f"Input: {x.shape}, Logits: {logits.shape}")

    model.eval()
    embeddings = model(x)
    print(f"Embeddings: {embeddings.shape}, Norm: {embeddings.norm(dim=1)}")

    features = model.extract_features(x)
    print(f"Extracted features: {features.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
