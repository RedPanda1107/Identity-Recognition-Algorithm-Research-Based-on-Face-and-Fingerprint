import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Simple fusion model that concatenates embeddings and classifies.

    Designed for face + fingerprint fusion via feature concatenation.
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=100, hidden_dim=256, dropout_rate=0.5):
        super(FusionModel, self).__init__()

        self.face_embedding_dim = face_embedding_dim
        self.fingerprint_embedding_dim = fingerprint_embedding_dim
        self.num_classes = num_classes

        # Fusion layer: concatenate embeddings
        fused_dim = face_embedding_dim + fingerprint_embedding_dim

        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, face_features, fingerprint_features):
        """Forward pass with concatenated features.

        Args:
            face_features: [batch_size, face_embedding_dim]
            fingerprint_features: [batch_size, fingerprint_embedding_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Concatenate along feature dimension
        fused_features = torch.cat([face_features, fingerprint_features], dim=1)
        return self.classifier(fused_features)

    def extract_features(self, face_features, fingerprint_features):
        """Extract fused features (before final classification layer)."""
        fused_features = torch.cat([face_features, fingerprint_features], dim=1)
        # Return features after first linear layer (before final classification)
        x = self.classifier[0](fused_features)
        x = self.classifier[1](x)  # BatchNorm
        x = self.classifier[2](x)  # ReLU
        return x


def create_fusion_model(**kwargs):
    """Factory function for fusion model."""
    return FusionModel(**kwargs)