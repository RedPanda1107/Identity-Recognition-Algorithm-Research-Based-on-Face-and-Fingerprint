import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace margin product for metric learning.

    Standard academic configuration: s=64.0, m=0.5
    - s: Scale factor (controls the hardness of softmax)
    - m: Angular margin penalty (pushes embeddings of different classes apart)

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
              (InsightFace, CVPR 2019)

    Args:
        in_features: size of each input sample (embedding dim)
        out_features: number of classes
        s: norm of input feature (default: 64.0)
        m: margin (default: 0.5)
        easy_margin: whether to use easy margin
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        # Parameter validation
        if s <= 0:
            raise ValueError(f"Scale factor s must be positive, got {s}")
        if m < 0 or m > 1:
            raise ValueError(f"Margin m must be in [0, 1], got {m}")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label=None):
        # input is features: (batch_size, embedding_dim)
        # label is (batch_size,) - optional, if None, return raw cosine similarity
        #
        # ARCHITECTURE: L2 norm → cos(θ) → angular margin → scale
        #   Step 1: F.normalize both input and weight
        #   Step 2: cos(θ) = <W, x> on unit hypersphere
        #   Step 3: Apply ArcFace additive angular margin cos(θ + m)
        #   Step 4: Scale logits by s (controls peakiness)
        #
        # SATURATION PREVENTION:
        #   - s=30 (not 64) to keep logits in manageable range
        #   - m=0.35 (not 0.5) to avoid over-compressing同人 angles
        #   - cosine clamped to [-1+eps, 1-eps] to prevent exp overflow
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [bs, out_features]

        # If no label provided, return raw cosine similarity (m=0 state)
        if label is None:
            return cosine * self.s

        # Compute sine with numerical stability: clamp to prevent sqrt of negative
        cosine_clamped = torch.clamp(cosine, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine_clamped.pow(2), min=0.0))

        # Apply ArcFace margin
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # apply margin to the true class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

