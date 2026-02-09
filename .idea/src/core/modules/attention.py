import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial Attention Module for local detail enhancement.

    Applies to both face (enhances facial features) and fingerprint (enhances ridge details).

    Architecture:
    - Global average pooling to get channel-wise statistics
    - 1x1 convolution for channel reduction
    - Sigmoid activation to generate attention weights
    - Element-wise multiplication to reweight input features
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

    def forward(self, x):
        """Apply spatial attention weighting.

        Args:
            x: Input feature tensor [B, C, H, W]

        Returns:
            Attention-weighted features [B, C, H, W]
        """
        spatial_features = self.conv1x1(x)
        pooled = self.global_pool(spatial_features)
        # Expand pooled tensor to match spatial dimensions
        attention = self.fusion(pooled)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel Attention Module for adaptive feature re-weighting.

    Enhances discriminative features by emphasizing important channels.

    Architecture:
    - Global average pooling (mean)
    - Global max pooling (max)
    - Shared MLP to compute attention weights
    - Sigmoid activation
    - Element-wise multiplication
    """

    def __init__(self, embedding_dim, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // reduction_ratio, embedding_dim, bias=False)
        )

    def forward(self, x):
        """Apply channel attention weighting.

        Args:
            x: Input feature tensor [B, D]

        Returns:
            Attention-weighted features [B, D]
        """
        avg_out = self.mlp(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
        max_out = self.mlp(self.max_pool(x.unsqueeze(-1)).squeeze(-1))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention
