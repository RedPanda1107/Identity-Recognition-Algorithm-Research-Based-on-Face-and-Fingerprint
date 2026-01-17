import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class FaceNet(nn.Module):
    """人脸特征提取和识别网络"""

    def __init__(self, num_classes=100, embedding_dim=512, pretrained=True, dropout_rate=0.5):
        """
        Args:
            num_classes: 类别数量
            embedding_dim: 特征嵌入维度
            pretrained: 是否使用预训练权重
            dropout_rate: dropout概率
        """
        super(FaceNet, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # 加载预训练的ResNet模型
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)

        # 获取原始的全连接层输入特征数
        num_features = self.backbone.fc.in_features

        # 替换最后的全连接层为特征提取器
        self.backbone.fc = nn.Identity()  # 移除原始的fc层

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # 分类器
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入图像张量
            return_features: 是否返回特征向量（用于测试和特征提取）

        Returns:
            如果return_features=True，返回(logits, features)
            否则只返回logits
        """
        # 特征提取
        features = self.backbone(x)  # [batch_size, 2048]

        # 特征投影
        features = self.feature_projection(features)  # [batch_size, embedding_dim]

        # 分类
        logits = self.classifier(features)  # [batch_size, num_classes]

        if return_features:
            return logits, features
        else:
            return logits

    def extract_features(self, x):
        """仅提取特征向量（用于推理）"""
        with torch.no_grad():
            features = self.backbone(x)
            features = self.feature_projection(features)
        return features

    def get_embedding_dim(self):
        """获取特征嵌入维度"""
        return self.embedding_dim


class FaceNetWithArcFace(FaceNet):
    """集成ArcFace损失的人脸识别网络"""

    def __init__(self, num_classes=100, embedding_dim=512, pretrained=True, margin=0.5, scale=64):
        super(FaceNetWithArcFace, self).__init__(num_classes, embedding_dim, pretrained)

        # ArcFace参数
        self.margin = margin
        self.scale = scale

        # L2归一化
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward_arcface(self, features, labels):
        """
        ArcFace前向传播

        Args:
            features: 特征向量 [batch_size, embedding_dim]
            labels: 标签 [batch_size]

        Returns:
            ArcFace logits
        """
        # L2归一化
        features_norm = self.l2_norm(features)
        weights_norm = self.l2_norm(self.classifier.weight)

        # 计算余弦相似度
        cos_theta = torch.matmul(features_norm, weights_norm.t())

        # 添加角度margin
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
        target_theta = theta + self.margin

        # 转换回余弦值
        cos_target_theta = torch.cos(target_theta)

        # 只对正确类别应用margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        output = cos_theta * (1 - one_hot) + cos_target_theta * one_hot
        output *= self.scale

        return output


def create_face_model(model_type='facenet', **kwargs):
    """工厂函数：创建人脸识别模型"""
    if model_type.lower() == 'facenet':
        return FaceNet(**kwargs)
    elif model_type.lower() == 'facenet_arcface':
        return FaceNetWithArcFace(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = FaceNet(num_classes=100, embedding_dim=512)

    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)  # batch_size=4, channels=3, height=224, width=224
    logits, features = model(x, return_features=True)

    print(f"输入形状: {x.shape}")
    print(f"特征形状: {features.shape}")
    print(f"输出形状: {logits.shape}")

    # 测试特征提取
    features_only = model.extract_features(x)
    print(f"提取特征形状: {features_only.shape}")