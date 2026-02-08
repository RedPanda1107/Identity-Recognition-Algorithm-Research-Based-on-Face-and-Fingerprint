import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance:
    Args:
        in_features: size of each input sample (embedding dim)
        out_features: number of classes
        s: norm of input feature
        m: margin
        easy_margin: whether to use easy margin
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
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

    def forward(self, input, label):
        # input is features: (batch_size, embedding_dim)
        # label is (batch_size,)
        # normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [bs, out_features]
        sine = torch.sqrt(1.0 - torch.pow(torch.clamp(cosine, min=0.0, max=1.0), 2))
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

