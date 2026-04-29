"""
多模态特征融合模型
支持人脸+指纹的特征级融合

提供多种融合策略：
    1. SimpleFusionModel      - 简化加权融合
    2. AdaptiveFusionModel    - 注意力自适应融合
    3. GatedFusionModel       - 门控融合（论文常用）
    4. HierarchicalFusionModel - 层级融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjection(nn.Module):
    """模态投影层 - 将不同模态特征映射到统一子空间"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.projection(x)


class SimpleFusionModel(nn.Module):
    """简化版多模态特征融合模型

    架构：
    1. 模态投影 - 统一特征空间
    2. 加权融合 - 学习模态权重
    3. 分类器 - ArcFace 度量学习头

    特征维度验证：
    - face_embedding_dim: 512 (标准)
    - fingerprint_embedding_dim: 512 (标准)
    - fusion_dim: 可配置 (默认 256)
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=300, fusion_dim=256, dropout_rate=0.3, use_arcface=True,
                 arc_s=64.0, arc_m=0.5):
        super().__init__()

        # 特征维度验证
        assert face_embedding_dim == 512, f"Face embedding dim must be 512, got {face_embedding_dim}"
        assert fingerprint_embedding_dim == 512, f"Fingerprint embedding dim must be 512, got {fingerprint_embedding_dim}"

        self.face_dim = face_embedding_dim
        self.fp_dim = fingerprint_embedding_dim
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        self.use_arcface = use_arcface

        # 模态投影
        self.face_proj = ModalityProjection(face_embedding_dim, fusion_dim)
        self.fp_proj = ModalityProjection(fingerprint_embedding_dim, fusion_dim)

        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # 分类器
        if use_arcface:
            from ..losses.arcface import ArcMarginProduct
            self.classifier = ArcMarginProduct(fusion_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            self._init_classifier()

    def _init_classifier(self):
        """Xavier初始化分类器"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, face_features, fp_features, labels=None):
        # 投影到统一空间
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)

        # 加权融合
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * face_proj + weights[1] * fp_proj
        fused = self.dropout(fused)

        # 分类 (ArcFace 或普通分类)
        return self.classifier(fused, labels)

    def extract_fused_features(self, face_features, fp_features):
        """提取融合特征（不含分类）"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * face_proj + weights[1] * fp_proj
        return fused

    def init_classifier_from_pretrained(self, face_ckpt_path, fp_ckpt_path, device):
        """从预训练单模态模型加载分类器权重"""
        import os

        face_loaded = False
        fp_loaded = False

        if face_ckpt_path and os.path.exists(face_ckpt_path):
            try:
                ckpt = torch.load(face_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if 'classifier.weight' in state:
                    self.classifier.weight.data = state['classifier.weight'].clone()
                    self.classifier.bias.data = state.get('classifier.bias',
                        torch.zeros(self.num_classes)).clone()
                    face_loaded = True
            except Exception as e:
                print(f"[Fusion] Face classifier load failed: {e}")

        if fp_ckpt_path and os.path.exists(fp_ckpt_path):
            try:
                ckpt = torch.load(fp_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if 'classifier.weight' in state:
                    if face_loaded:
                        self.classifier.weight.data = (
                            self.classifier.weight.data + state['classifier.weight'].clone()
                        ) / 2
                        self.classifier.bias.data = (
                            self.classifier.bias.data + state.get('classifier.bias',
                                torch.zeros(self.num_classes)).clone()
                        ) / 2
                    else:
                        self.classifier.weight.data = state['classifier.weight'].clone()
                        self.classifier.bias.data = state.get('classifier.bias',
                            torch.zeros(self.num_classes)).clone()
                    fp_loaded = True
            except Exception as e:
                print(f"[Fusion] FP classifier load failed: {e}")

        return face_loaded, fp_loaded


class AdaptiveFusionModel(nn.Module):
    """自适应融合模型 - 使用注意力机制学习模态权重

    特征维度验证：
    - face_embedding_dim: 512 (标准)
    - fingerprint_embedding_dim: 512 (标准)
    - fusion_dim: 可配置 (默认 256)
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=300, fusion_dim=256, dropout_rate=0.3, use_arcface=True,
                 arc_s=64.0, arc_m=0.5):
        super().__init__()

        # 特征维度验证
        assert face_embedding_dim == 512, f"Face embedding dim must be 512, got {face_embedding_dim}"
        assert fingerprint_embedding_dim == 512, f"Fingerprint embedding dim must be 512, got {fingerprint_embedding_dim}"

        self.face_dim = face_embedding_dim
        self.fp_dim = fingerprint_embedding_dim
        self.num_classes = num_classes
        self.use_arcface = use_arcface

        # 模态投影
        self.face_proj = ModalityProjection(face_embedding_dim, fusion_dim)
        self.fp_proj = ModalityProjection(fingerprint_embedding_dim, fusion_dim)

        # 注意力权重网络
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # 特征增强
        self.enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        # 分类器
        if use_arcface:
            from ..losses.arcface import ArcMarginProduct
            self.classifier = ArcMarginProduct(fusion_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            self._init_classifier()

    def _init_classifier(self):
        """Xavier初始化分类器"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, face_features, fp_features, labels=None):
        # 投影
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)

        # 注意力加权
        concat = torch.stack([face_proj, fp_proj], dim=1)
        weights = self.attention(concat.mean(dim=1))
        fused = (concat * weights.unsqueeze(-1)).sum(dim=1)

        # 特征增强
        fused = fused + self.enhancer(fused)

        # 分类 (ArcFace 或普通分类)
        return self.classifier(fused, labels)

    def extract_fused_features(self, face_features, fp_features):
        """提取融合特征"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        concat = torch.stack([face_proj, fp_proj], dim=1)
        weights = self.attention(concat.mean(dim=1))
        fused = (concat * weights.unsqueeze(-1)).sum(dim=1)
        return fused

    def init_classifier_from_pretrained(self, face_ckpt_path, fp_ckpt_path, device):
        """从预训练单模态模型加载分类器权重"""
        import os

        face_loaded = False
        fp_loaded = False

        if face_ckpt_path and os.path.exists(face_ckpt_path):
            try:
                ckpt = torch.load(face_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if 'classifier.weight' in state:
                    self.classifier.weight.data = state['classifier.weight'].clone()
                    self.classifier.bias.data = state.get('classifier.bias',
                        torch.zeros(self.num_classes)).clone()
                    face_loaded = True
            except Exception:
                pass

        if fp_ckpt_path and os.path.exists(fp_ckpt_path):
            try:
                ckpt = torch.load(fp_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if 'classifier.weight' in state:
                    if face_loaded:
                        self.classifier.weight.data = (
                            self.classifier.weight.data + state['classifier.weight'].clone()
                        ) / 2
                        self.classifier.bias.data = (
                            self.classifier.bias.data + state.get('classifier.bias',
                                torch.zeros(self.num_classes)).clone()
                        ) / 2
                    else:
                        self.classifier.weight.data = state['classifier.weight'].clone()
                        self.classifier.bias.data = state.get('classifier.bias',
                            torch.zeros(self.num_classes)).clone()
                    fp_loaded = True
            except Exception:
                pass

        return face_loaded, fp_loaded


class GatedFusionModel(nn.Module):
    """门控融合模型 - 论文常用方案 (Gated Multimodal Fusion)

    通过门控网络学习模态间的动态权重，
    类似 ResNet 的残差门控机制。

    特点：
    - 门控值由两个模态的联合表示生成
    - 可解释性强（门控值反映模态贡献度）
    - 训练稳定

    参考：Are You Acceptable? - Multimodal Gated Fusion
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=300, fusion_dim=256, dropout_rate=0.3, use_arcface=True,
                 arc_s=64.0, arc_m=0.5, gate_hidden_dim=128):
        super().__init__()

        self.face_dim = face_embedding_dim
        self.fp_dim = fingerprint_embedding_dim
        self.num_classes = num_classes
        self.use_arcface = use_arcface

        # 模态投影
        self.face_proj = ModalityProjection(face_embedding_dim, fusion_dim)
        self.fp_proj = ModalityProjection(fingerprint_embedding_dim, fusion_dim)

        # 门控网络 - 基于两个模态的联合表示
        self.gate_fc = nn.Sequential(
            nn.Linear(fusion_dim * 2, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )

        # 特征增强（可选的残差连接）
        self.enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # 分类器
        if use_arcface:
            from ..losses.arcface import ArcMarginProduct
            self.classifier = ArcMarginProduct(fusion_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            self._init_classifier()

    def _init_classifier(self):
        """Xavier初始化分类器"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, face_features, fp_features, labels=None):
        # 投影到统一空间
        face_proj = self.face_proj(face_features)  # [B, fusion_dim]
        fp_proj = self.fp_proj(fp_features)        # [B, fusion_dim]

        # 门控计算：基于联合表示
        concat_features = torch.cat([face_proj, fp_proj], dim=1)  # [B, 2*fusion_dim]
        gate = self.gate_fc(concat_features)  # [B, 1], 0~1 之间

        # 门控融合
        fused = gate * face_proj + (1 - gate) * fp_proj  # [B, fusion_dim]

        # 特征增强（残差）
        fused = fused + self.dropout(self.enhancer(fused))

        # 分类
        return self.classifier(fused, labels)

    def extract_fused_features(self, face_features, fp_features):
        """提取融合特征"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        concat_features = torch.cat([face_proj, fp_proj], dim=1)
        gate = self.gate_fc(concat_features)
        fused = gate * face_proj + (1 - gate) * fp_proj
        return fused + self.enhancer(fused)

    def get_gate_values(self, face_features, fp_features):
        """获取门控值（用于分析模态贡献度）"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        concat_features = torch.cat([face_proj, fp_proj], dim=1)
        gate = self.gate_fc(concat_features)
        return gate  # 1 = 全用人脸, 0 = 全用指纹


class HierarchicalFusionModel(nn.Module):
    """层级融合模型 - 粗粒度+细粒度融合

    多层级特征交互：
    Level 1: 模态投影
    Level 2: 跨模态注意力交互
    Level 3: 融合输出

    特点：
    - 更丰富的模态交互
    - 可捕获细粒度对应关系
    - 适合异构模态（人脸+指纹差异大）
    """

    def __init__(self, face_embedding_dim=512, fingerprint_embedding_dim=512,
                 num_classes=300, fusion_dim=256, dropout_rate=0.3, use_arcface=True,
                 arc_s=64.0, arc_m=0.5, num_heads=4):
        super().__init__()

        self.face_dim = face_embedding_dim
        self.fp_dim = fingerprint_embedding_dim
        self.num_classes = num_classes
        self.use_arcface = use_arcface

        # Level 1: 模态投影
        self.face_proj = ModalityProjection(face_embedding_dim, fusion_dim)
        self.fp_proj = ModalityProjection(fingerprint_embedding_dim, fusion_dim)

        # Level 2: 跨模态注意力交互
        self.cross_attention = CrossModalAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # Level 3: 融合特征压缩
        self.fusion_encoder = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),  # 原始 + 交叉注意力
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        # 分类器
        if use_arcface:
            from ..losses.arcface import ArcMarginProduct
            self.classifier = ArcMarginProduct(fusion_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            self._init_classifier()

    def _init_classifier(self):
        """Xavier初始化分类器"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, face_features, fp_features, labels=None):
        # Level 1: 投影
        face_proj = self.face_proj(face_features)  # [B, D]
        fp_proj = self.fp_proj(fp_features)        # [B, D]

        # Level 2: 跨模态注意力
        face_to_fp, fp_to_face = self.cross_attention(face_proj, fp_proj)

        # Level 3: 融合
        fused = self.fusion_encoder(
            torch.cat([face_proj, fp_proj, face_to_fp, fp_to_face], dim=1)
        )

        # 分类
        return self.classifier(fused, labels)

    def extract_fused_features(self, face_features, fp_features):
        """提取融合特征"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        face_to_fp, fp_to_face = self.cross_attention(face_proj, fp_proj)
        fused = self.fusion_encoder(
            torch.cat([face_proj, fp_proj, face_to_fp, fp_to_face], dim=1)
        )
        return fused


class CrossModalAttention(nn.Module):
    """跨模态注意力模块

    计算两个模态之间的相互注意力：
    - face_to_fp: 人脸对指纹的注意力
    - fp_to_face: 指纹对人脸的注意力
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, D] 模态1特征
            x2: [B, D] 模态2特征
        Returns:
            x1_to_x2: [B, D] x1对x2的注意力加权
            x2_to_x1: [B, D] x2对x1的注意力加权
        """
        B, D = x1.shape

        # x1 -> x2 的注意力
        q1 = self.q_proj(x1).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        k2 = self.k_proj(x2).view(B, self.num_heads, self.head_dim)
        v2 = self.v_proj(x2).view(B, self.num_heads, self.head_dim)

        attn1 = torch.einsum('bhd,bhd->bh', q1, k2) * self.scale  # [B, H]
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout(attn1)

        x1_to_x2 = torch.einsum('bh,bhd->bhd', attn1, v2).reshape(B, D)  # [B, D]
        x1_to_x2 = self.out_proj(x1_to_x2)

        # x2 -> x1 的注意力
        q2 = self.q_proj(x2).view(B, self.num_heads, self.head_dim)
        k1 = self.k_proj(x1).view(B, self.num_heads, self.head_dim)
        v1 = self.v_proj(x1).view(B, self.num_heads, self.head_dim)

        attn2 = torch.einsum('bhd,bhd->bh', q2, k1) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)

        x2_to_x1 = torch.einsum('bh,bhd->bhd', attn2, v1).reshape(B, D)
        x2_to_x1 = self.out_proj(x2_to_x1)

        return x1_to_x2, x2_to_x1


def create_fusion_model(fusion_method='simple', **kwargs):
    """工厂函数：创建融合模型

    Args:
        fusion_method: 'simple', 'adaptive', 'gated', 'hierarchical'
        **kwargs: 传递给模型的其他参数

    Returns:
        融合模型实例
    """
    fusion_methods = {
        'simple': SimpleFusionModel,
        'adaptive': AdaptiveFusionModel,
        'gated': GatedFusionModel,
        'hierarchical': HierarchicalFusionModel,
    }

    if fusion_method not in fusion_methods:
        raise ValueError(
            f"Unknown fusion method: {fusion_method}. "
            f"Available: {list(fusion_methods.keys())}"
        )

    return fusion_methods[fusion_method](**kwargs)
