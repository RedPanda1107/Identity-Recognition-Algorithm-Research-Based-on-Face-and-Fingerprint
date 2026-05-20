"""
通用多模态融合模型。

由三个部分组成：
  1. face_proj / fp_proj — 投影层（始终初始化）
  2. fusion_strategy — 可插拔的融合策略（由 fusion_strategy 参数决定）
  3. classifier — 分类头（ArcFace 或 Linear）

实验模式通过 config 决定使用哪些组件：
  - 融合实验（simple / adaptive）：FusionModel + 对应策略
  - 消融实验：FusionModel + AblationStrategy 包装

向后兼容：
  - 保留 set_ablation(ablate_modality) 方法，兼容 FusionTrainer 的调用方式
  - 保留 get_fusion_weights() 和 get_attention_weights() 方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_strategy import (
    FusionStrategy, WeightedSumStrategy, AttentionStrategy,
    AblationStrategy, create_fusion_strategy
)


class ModalityProjection(nn.Module):
    """模态投影层 — 将不同模态特征映射到统一子空间。

    初始化为恒等映射（当 input_dim == output_dim 时）或部分恒等
    （input_dim > output_dim 时取左上 block），使训练从保真状态起步，
    投影层逐步学习最优映射而非从随机起点。
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self._init_as_identity(input_dim, output_dim)

    def _init_as_identity(self, input_dim, output_dim):
        linear = self.projection[0]
        with torch.no_grad():
            if input_dim == output_dim:
                linear.weight.copy_(torch.eye(output_dim))
            else:
                min_d = min(input_dim, output_dim)
                weight = torch.zeros_like(linear.weight)
                weight[:min_d, :min_d] = torch.eye(min_d)
                linear.weight.copy_(weight)
            linear.bias.zero_()

    def forward(self, x):
        return self.projection(x)


class FusionModel(nn.Module):
    """
    通用多模态融合容器。

    Args:
        fusion_strategy: 融合策略类型，"simple" | "adaptive"
        face_embedding_dim: 人脸 backbone 输出的特征维度
        fingerprint_embedding_dim: 指纹 backbone 输出的特征维度
        num_classes: 类别数量
        fusion_dim: 融合特征维度
        dropout_rate: Dropout 概率
        use_arcface: 是否使用 ArcFace 分类头
        arc_s: ArcFace 的 scale 参数
        arc_m: ArcFace 的 margin 参数
        ablate_face: 是否消融人脸模态（训练单用指纹）
        ablate_fp: 是否消融指纹模态（训练单用人脸）
        attention_hidden_dim: attention 策略的隐藏层维度
    """

    def __init__(self,
                 fusion_strategy: str = "simple",
                 face_embedding_dim: int = 512,
                 fingerprint_embedding_dim: int = 512,
                 num_classes: int = 300,
                 fusion_dim: int = 256,
                 dropout_rate: float = 0.3,
                 use_arcface: bool = True,
                 arc_s: float = 30.0,
                 arc_m: float = 0.35,
                 ablate_face: bool = False,
                 ablate_fp: bool = False,
                 attention_hidden_dim: int = 64):
        super().__init__()

        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.use_arcface = use_arcface
        self._ablate_modality = None

        # 投影层
        self.face_proj = ModalityProjection(face_embedding_dim, fusion_dim)
        self.fp_proj = ModalityProjection(fingerprint_embedding_dim, fusion_dim)

        # 创建基础策略
        base_strategy: FusionStrategy
        if fusion_strategy == "simple":
            base_strategy = WeightedSumStrategy(fusion_dim)
        elif fusion_strategy == "adaptive":
            base_strategy = AttentionStrategy(fusion_dim, hidden_dim=attention_hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

        # 消融包装（如果需要）
        if ablate_face or ablate_fp:
            self.fusion_strategy = AblationStrategy(
                base_strategy, fusion_dim,
                ablating_face=ablate_face, ablating_fp=ablate_fp
            )
        else:
            self.fusion_strategy = base_strategy

        self.dropout = nn.Dropout(dropout_rate)

        # 分类头
        if use_arcface:
            from ..losses.arcface import ArcMarginProduct
            self.classifier = ArcMarginProduct(fusion_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            self._init_classifier()

    def _init_classifier(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ── 向后兼容接口 ───────────────────────────────────────────────────────────

    def set_ablation(self, ablate_modality: str):
        """设置消融模式（兼容 FusionTrainer 的调用方式）。

        将旧的字符串格式映射到新的硬截断策略：
          'face' → ablate_face=True
          'fingerprint' → ablate_fp=True
          None → 不消融
        """
        self._ablate_modality = ablate_modality
        if ablate_modality == 'face':
            self._rebuild_strategy_with_ablation(ablate_face=True, ablate_fp=False)
        elif ablate_modality == 'fingerprint':
            self._rebuild_strategy_with_ablation(ablate_face=False, ablate_fp=True)
        else:
            self._rebuild_strategy_with_ablation(ablate_face=False, ablate_fp=False)

    def _rebuild_strategy_with_ablation(self, ablate_face: bool, ablate_fp: bool):
        """重建 fusion_strategy，应用硬截断消融"""
        inner = self.fusion_strategy
        if hasattr(inner, 'strategy'):
            inner = inner.strategy
        if ablate_face or ablate_fp:
            self.fusion_strategy = AblationStrategy(
                inner, self.fusion_dim,
                ablating_face=ablate_face, ablating_fp=ablate_fp
            )
        else:
            self.fusion_strategy = inner

    def get_fusion_weights(self):
        """返回当前融合权重的诊断摘要。

        对 Adaptive：返回 logits_bias 经过 softmax 的初始分布（不跑前向，避免 zero-ablation 干扰）。
        真实权重由 get_cached_weights() 在 forward 后获取。
        """
        info = self.get_ablation_info()
        if info.get("type") == "weighted_sum":
            return info.get("face_weight", 0.5), info.get("fp_weight", 0.5)
        elif info.get("type") == "ablation":
            face_w = 0.0 if info.get("face") == "zero" else 1.0
            fp_w = 0.0 if info.get("fp") == "zero" else 1.0
            return face_w, fp_w
        elif info.get("type") == "attention":
            # 返回 logits_bias 的 softmax 分布（初始化时就是 0.5/0.5）
            # 这是初始参考值，真实 per-batch 权重由 get_cached_weights() 提供
            strategy = self.fusion_strategy
            if hasattr(strategy, 'strategy'):
                strategy = strategy.strategy
            if hasattr(strategy, 'logits_bias'):
                w = F.softmax(strategy.logits_bias.data, dim=0)
                return float(w[0].item()), float(w[1].item())
            return 0.5, 0.5
        return 0.5, 0.5

    def get_cached_attention_weights(self):
        """返回上次 forward 缓存的 attention softmax 权重均值。

        返回 (attn_face_mean, attn_fp_mean)，若无缓存则返回 None。
        优先从 fusion_strategy 获取（绕过 AblationStrategy wrapper）。
        """
        strategy = self.fusion_strategy
        if hasattr(strategy, 'get_cached_weights'):
            cached = strategy.get_cached_weights()
            if cached is not None and cached.shape[0] > 0:
                mean_face = float(cached[:, 0].mean().item())
                mean_fp = float(cached[:, 1].mean().item())
                return mean_face, mean_fp
        return None

    def get_ablation_info(self):
        """返回融合层诊断信息"""
        strategy = self.fusion_strategy
        if hasattr(strategy, 'get_ablation_info'):
            return strategy.get_ablation_info()
        return {}

    # ── 前向接口 ───────────────────────────────────────────────────────────────

    def forward(self, face_features, fp_features, labels=None):
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        fused = self.fusion_strategy(face_proj, fp_proj)
        fused = self.dropout(fused)
        return self.classifier(fused, labels)

    def extract_fused_features(self, face_features, fp_features):
        """特征提取（不含分类头，用于 Gallery/Query 检索）"""
        face_proj = self.face_proj(face_features)
        fp_proj = self.fp_proj(fp_features)
        return self.fusion_strategy(face_proj, fp_proj)

    def init_classifier_from_pretrained(self, face_ckpt_path, fp_ckpt_path, device):
        """从单模态预训练 checkpoint 加载分类器权重。

        处理两种 key 名（_classifier.weight / classifier.weight）和维度不匹配（512 vs 256）：
          - Standalone checkpoint: classifier weight shape = [num_classes, backbone_embedding=512]
          - FusionModel classifier:   classifier weight shape = [num_classes, fusion_dim=256]
          - 加载时取前 fusion_dim 列（截断策略）
          - 截断后重新 L2 归一化（与 standalone 训练一致）
        """
        import os
        face_loaded = False
        fp_loaded = False

        def _find_clf_key(state):
            for key in ('_classifier.weight', 'classifier.weight',
                        '_arc_classifier.weight', 'arc_classifier.weight'):
                if key in state:
                    return key
            return None

        def _load_clf(state, is_first):
            key = _find_clf_key(state)
            if key is None:
                return False
            w = state[key]
            w_aligned = w
            if w.shape[1] > self.classifier.in_features:
                w_aligned = w[:, :self.classifier.in_features]
            elif w.shape[1] < self.classifier.in_features:
                pad = torch.zeros(self.num_classes,
                                 self.classifier.in_features - w.shape[1], device=device)
                w_aligned = torch.cat([w, pad], dim=1)
            # 截断后重新 L2 归一化（standalone 训练时会归一化，这里保持一致）
            w_aligned = F.normalize(w_aligned, p=2, dim=1)
            if is_first:
                self.classifier.weight.data = w_aligned.clone()
            else:
                self.classifier.weight.data = (self.classifier.weight.data + w_aligned) / 2.0
            return True

        if face_ckpt_path and os.path.exists(face_ckpt_path):
            try:
                ckpt = torch.load(face_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if _load_clf(state, is_first=True):
                    face_loaded = True
            except Exception:
                pass

        if fp_ckpt_path and os.path.exists(fp_ckpt_path):
            try:
                ckpt = torch.load(fp_ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                if _load_clf(state, is_first=(not face_loaded)):
                    fp_loaded = True
            except Exception:
                pass

        return face_loaded, fp_loaded
