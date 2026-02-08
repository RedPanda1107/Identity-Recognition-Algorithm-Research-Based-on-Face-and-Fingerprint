import torch
import torch.nn as nn
from torch.nn import functional as F


class EnhancedFusionModel(nn.Module):
    """增强版多模态融合模型

    支持多种融合策略：concat, attention_fusion, cross_attention, tensor_fusion
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=256,
                 num_classes=100, hidden_dim=512, dropout_rate=0.5,
                 fusion_method='concat'):
        super(EnhancedFusionModel, self).__init__()

        self.face_embedding_dim = face_embedding_dim
        self.fingerprint_embedding_dim = fingerprint_embedding_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        if fusion_method == 'concat':
            # 简单拼接 + MLP
            fused_dim = face_embedding_dim + fingerprint_embedding_dim
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

        elif fusion_method == 'attention_fusion':
            # 注意力融合 - 分别计算每个模态的注意力权重
            self.face_attention = nn.Sequential(
                nn.Linear(face_embedding_dim, face_embedding_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(face_embedding_dim // 4, 1),
                nn.Sigmoid()
            )

            self.fp_attention = nn.Sequential(
                nn.Linear(fingerprint_embedding_dim, fingerprint_embedding_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(fingerprint_embedding_dim // 4, 1),
                nn.Sigmoid()
            )

            self.classifier = nn.Sequential(
                nn.Linear(face_embedding_dim + fingerprint_embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

        elif fusion_method == 'cross_attention':
            # 跨模态注意力 - 先投影到相同维度
            self.cross_proj = nn.Linear(fingerprint_embedding_dim, face_embedding_dim)
            self.cross_attention = CrossModalAttention(
                embed_dim=face_embedding_dim,
                num_heads=8
            )

            self.classifier = nn.Sequential(
                nn.Linear(face_embedding_dim * 2, hidden_dim),  # face + attended_fingerprint
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

        elif fusion_method == 'tensor_fusion':
            # 张量融合 (更复杂的交互)
            self.tensor_fusion = TensorFusion(
                face_dim=face_embedding_dim,
                fingerprint_dim=fingerprint_embedding_dim,
                hidden_dim=hidden_dim
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, face_features, fingerprint_features):
        if self.fusion_method == 'concat':
            fused = torch.cat([face_features, fingerprint_features], dim=1)
            return self.classifier(fused)

        elif self.fusion_method == 'attention_fusion':
            # 分别计算注意力权重
            face_weight = self.face_attention(face_features)  # [batch, 1]
            fp_weight = self.fp_attention(fingerprint_features)  # [batch, 1]

            # 应用注意力权重
            attended_face = face_features * face_weight
            attended_fp = fingerprint_features * fp_weight

            # 拼接后分类
            combined = torch.cat([attended_face, attended_fp], dim=1)
            return self.classifier(combined)

        elif self.fusion_method == 'cross_attention':
            # 先将指纹特征投影到相同维度
            fp_proj = self.cross_proj(fingerprint_features)
            # 人脸特征作为query，指纹特征作为key/value
            attended_features = self.cross_attention(face_features, fp_proj)
            combined = torch.cat([face_features, attended_features], dim=1)
            return self.classifier(combined)

        elif self.fusion_method == 'tensor_fusion':
            fused = self.tensor_fusion(face_features, fingerprint_features)
            return self.classifier(fused)

    def extract_features(self, face_features, fingerprint_features):
        """提取融合特征（用于特征分析）"""
        with torch.no_grad():
            if self.fusion_method == 'concat':
                fused = torch.cat([face_features, fingerprint_features], dim=1)
                # 返回第一层特征
                x = self.classifier[0](fused)
                x = self.classifier[1](x)  # BatchNorm
                x = self.classifier[2](x)  # ReLU
                return x

            elif self.fusion_method == 'attention_fusion':
                # 分别计算注意力权重
                face_weight = self.face_attention(face_features)
                fp_weight = self.fp_attention(fingerprint_features)

                # 应用注意力权重
                attended_face = face_features * face_weight
                attended_fp = fingerprint_features * fp_weight

                # 拼接后分类
                combined = torch.cat([attended_face, attended_fp], dim=1)
                x = self.classifier[0](combined)
                x = self.classifier[1](x)
                x = self.classifier[2](x)
                return x

            elif self.fusion_method == 'cross_attention':
                fp_proj = self.cross_proj(fingerprint_features)
                attended_features = self.cross_attention(face_features, fp_proj)
                combined = torch.cat([face_features, attended_features], dim=1)
                x = self.classifier[0](combined)
                x = self.classifier[1](x)
                x = self.classifier[2](x)
                return x

            elif self.fusion_method == 'tensor_fusion':
                fused = self.tensor_fusion(face_features, fingerprint_features)
                x = self.classifier[0](fused)
                x = self.classifier[1](x)
                x = self.classifier[2](x)
                return x


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        # query: face_features, key_value: fingerprint_features
        # 需要扩展维度以适应多头注意力
        query = query.unsqueeze(1)  # [batch, 1, dim]
        key_value = key_value.unsqueeze(1)  # [batch, 1, dim]

        attended, _ = self.attention(query, key_value, key_value)
        attended = attended.squeeze(1)  # [batch, dim]

        return self.norm(attended + query.squeeze(1))


class TensorFusion(nn.Module):
    """张量融合模块 - 学习模态间的复杂交互"""

    def __init__(self, face_dim, fingerprint_dim, hidden_dim):
        super().__init__()

        # 外积张量融合
        self.tensor_weight = nn.Parameter(torch.randn(face_dim, fingerprint_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # 残差连接
        self.residual_proj = nn.Linear(face_dim + fingerprint_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, face_feat, fingerprint_feat):
        # 张量融合: face_feat ⊗ fingerprint_feat
        # [batch, face_dim] -> [batch, face_dim, 1]
        face_expanded = face_feat.unsqueeze(2)
        # [batch, fingerprint_dim] -> [batch, 1, fingerprint_dim]
        fp_expanded = fingerprint_feat.unsqueeze(1)

        # 外积: [batch, face_dim, fingerprint_dim]
        outer_product = torch.matmul(face_expanded, fp_expanded)

        # 与张量权重相乘: [batch, face_dim, fingerprint_dim] @ [face_dim, fingerprint_dim, hidden_dim]
        # -> [batch, hidden_dim]
        fused = torch.einsum('bij,ijk->bk', outer_product, self.tensor_weight)

        # 残差连接
        residual = self.residual_proj(torch.cat([face_feat, fingerprint_feat], dim=1))

        output = fused + residual + self.bias
        return self.norm(self.dropout(output))


# 保持向后兼容性
class FusionModel(EnhancedFusionModel):
    """向后兼容的简单融合模型"""

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=100, hidden_dim=256, dropout_rate=0.5):
        super().__init__(
            face_embedding_dim=face_embedding_dim,
            fingerprint_embedding_dim=fingerprint_embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            fusion_method='concat'
        )


def create_fusion_model(fusion_method='concat', **kwargs):
    """工厂函数：创建融合模型"""
    return EnhancedFusionModel(fusion_method=fusion_method, **kwargs)


def create_enhanced_fusion_model(**kwargs):
    """创建增强版融合模型（别名）"""
    return EnhancedFusionModel(**kwargs)