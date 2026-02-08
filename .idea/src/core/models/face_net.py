import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from ..losses.arcface import ArcMarginProduct


class SpatialAttention(nn.Module):
    """Spatial Attention Module for local detail enhancement.
    
    Applies to both face (enhances facial features) and fingerprint (enhances ridge details).
    """

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

        # Feature projection (same structure as FingerprintNet)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
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


class FaceNetWithArcFace(FaceNet):
    """Face recognition network with integrated ArcFace loss.
    
    Note: ArcFace is now handled by FingerprintMetricTrainer through the 
    classifier property. This class is kept for backward compatibility.
    """

    def __init__(self, num_classes=100, embedding_dim=512, pretrained=True, 
                 dropout_rate=0.5, spatial_attention=True, 
                 arc_s=30.0, arc_m=0.5):
        # Initialize parent with all parameters
        super(FaceNetWithArcFace, self).__init__(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            spatial_attention=spatial_attention
        )

        # Set up ArcFace classifier directly (legacy approach)
        self._classifier = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=arc_s,
            m=arc_m
        )


def create_face_model(model_type='facenet', **kwargs):
    """Factory function: Create face recognition model.
    
    Model types:
    - 'facenet': Standard face net with attention mechanisms
    - 'facenet_arcface': Face net with ArcFace integrated
    
    Compatible with create_fingerprint_model() API.
    """
    if model_type.lower() == 'facenet':
        return FaceNet(**kwargs)
    elif model_type.lower() == 'facenet_arcface':
        return FaceNetWithArcFace(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# ============================================================
# API Compatibility Utilities for Face-Fingerprint Fusion
# ============================================================

def get_face_embedding_dim():
    """Get standard face embedding dimension."""
    return 512


def get_fingerprint_embedding_dim():
    """Get standard fingerprint embedding dimension."""
    return 256


def align_embedding_dims(face_features, fingerprint_features):
    """Align face (512D) and fingerprint (256D) embeddings for fusion.
    
    Args:
        face_features: Face embeddings [batch, 512]
        fingerprint_features: Fingerprint embeddings [batch, 256]
        
    Returns:
        Tuple of aligned embeddings (both 256D)
    """
    # Project face features from 512D to 256D
    projection = nn.Linear(512, 256)
    face_aligned = projection(face_features)
    face_aligned = F.normalize(face_aligned, p=2, dim=1)
    return face_aligned, fingerprint_features


# ============================================================
# Test Code
# ============================================================

if __name__ == "__main__":
    # Test enhanced FaceNet
    print("=" * 60)
    print("Enhanced FaceNet Test")
    print("=" * 60)
    
    model = FaceNet(num_classes=100, embedding_dim=512)
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    
    # Training mode
    model.train()
    embeddings = model(x)
    print(f"Training mode - Embeddings shape: {embeddings.shape}")
    print(f"Training mode - Embedding norm: {embeddings.norm(dim=1).mean():.4f}")
    
    # Eval mode
    model.eval()
    embeddings = model(x)
    print(f"Eval mode - Embeddings shape: {embeddings.shape}")
    
    # With classifier
    model.classifier = ArcMarginProduct(512, 100, s=30.0, m=0.5).to(x.device)
    labels = torch.tensor([0, 1, 2, 3])
    logits = model(x, labels=labels)
    print(f"With ArcFace - Logits shape: {logits.shape}")
    
    # Feature extraction API (same as FingerprintNet)
    features = model.extract_features(x)
    print(f"Extract features - Shape: {features.shape}")
    
    # Parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: total={total:,}, trainable={trainable:,}")
    
    # Compare with FingerprintNet
    print("\n" + "=" * 60)
    print("API Compatibility Check")
    print("=" * 60)
    print(f"FaceNet embedding_dim: {model.get_embedding_dim()}")
    print("API methods available:")
    print("  - _extract_features(x): Unified feature extraction")
    print("  - extract_features(x): L2-normalized extraction")
    print("  - classifier property: Get/set ArcFace classifier")
    print("  - get_embedding_dim(): Get embedding dimension")
    print("\nFaceNet is now compatible with FingerprintNet for fusion!")
