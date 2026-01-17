import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class FingerprintNet(nn.Module):
    """Fingerprint feature extraction and recognition network."""

    def __init__(self, num_classes=600, embedding_dim=256, pretrained=False, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of classes (person IDs)
            embedding_dim: Feature embedding dimension
            pretrained: Whether to use pretrained weights (usually False for fingerprints)
            dropout_rate: Dropout probability
        """
        super(FingerprintNet, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Use ResNet34 for fingerprint recognition (lighter than ResNet50 for face)
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet34(weights=weights)

        # Get the original fully connected layer input features
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Feature projection layer with attention mechanism for fingerprint details
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Fingerprint-specific attention layer
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, return_features=False):
        """
        Forward pass

        Args:
            x: Input image tensor
            return_features: Whether to return feature vectors

        Returns:
            If return_features=True, return (logits, features)
            Otherwise return logits only
        """
        # Feature extraction
        features = self.backbone(x)  # [batch_size, 512]

        # Feature projection
        features = self.feature_projection(features)  # [batch_size, embedding_dim]

        # Apply attention mechanism
        attention_weights = self.attention(features)
        features = features * attention_weights  # [batch_size, embedding_dim]

        # Classification
        logits = self.classifier(features)  # [batch_size, num_classes]

        if return_features:
            return logits, features
        else:
            return logits

    def extract_features(self, x):
        """Extract feature vectors only (for inference)"""
        with torch.no_grad():
            features = self.backbone(x)
            features = self.feature_projection(features)
            attention_weights = self.attention(features)
            features = features * attention_weights
        return features

    def get_embedding_dim(self):
        """Get feature embedding dimension"""
        return self.embedding_dim


def create_fingerprint_model(model_type='fingerprint_net', **kwargs):
    """Factory function: Create fingerprint recognition model"""
    if model_type.lower() == 'fingerprint_net':
        return FingerprintNet(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Test code
if __name__ == "__main__":
    # Create model
    model = FingerprintNet(num_classes=600, embedding_dim=256)

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)  # batch_size=4, channels=3, height=224, width=224
    logits, features = model(x, return_features=True)

    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {logits.shape}")

    # Test feature extraction
    features_only = model.extract_features(x)
    print(f"Extracted features shape: {features_only.shape}")