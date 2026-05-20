"""
融合策略抽象层 — FusionModel 的可插拔融合核心。

提供了三种融合策略：
  WeightedSumStrategy  — 简单加权求和（对应 simple 融合）
  AttentionStrategy    — 注意力自适应融合（对应 adaptive 融合）
  AblationStrategy    — 消融包装（将任意策略包装为单模态消融）

设计原则：
  - 策略是可插拔的：通过 create_fusion_strategy() 工厂函数创建
  - 消融通过 AblationStrategy 包装器实现，而非在策略内部硬截断
  - 所有策略继承 FusionStrategy 基类，保证接口一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class FusionStrategy(ABC, nn.Module):
    """融合策略基类 — 定义所有策略的统一接口"""

    @abstractmethod
    def forward(self, face_feat: torch.Tensor, fp_feat: torch.Tensor) -> torch.Tensor:
        """融合两个模态的特征，返回融合特征"""
        pass

    def get_ablation_info(self) -> dict:
        """返回消融诊断信息（基类返回空字典）"""
        return {}

    def get_cached_weights(self):
        """返回上次 forward 时缓存的 softmax 权重（基类不支持）"""
        return None


class WeightedSumStrategy(FusionStrategy):
    """简单加权求和策略（对应 simple 融合）。

    通过可学习的 softmax 权重融合两个模态：
        fused = w_face * face_feat + w_fp * fp_feat
    """

    def __init__(self, fusion_dim: int):
        super().__init__()
        self.weight = torch.tensor([0.5, 0.5])

    def forward(self, face_feat, fp_feat):
        w = F.softmax(self.weight, dim=0)
        return w[0] * face_feat + w[1] * fp_feat

    def get_ablation_info(self):
        w = F.softmax(self.weight, dim=0)
        return {"type": "weighted_sum", "face_weight": float(w[0].item()), "fp_weight": float(w[1].item())}


class AttentionStrategy(FusionStrategy):
    """注意力自适应融合策略（对应 adaptive 融合）。

    通过可学习的 softmax 权重融合两个模态，权重由 MLP 基于两模态的
    均值、差异向量和余弦相似度动态计算。

    融合公式：
        weights = softmax(MLP([mean; diff; cos_sim]))
        fused  = weighted_sum(weights, face_feat, fp_feat)

    配合 entropy_penalty_weight > 0 的训练正则化，迫使权重趋向均衡。
    """

    def __init__(self, fusion_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Attention 输入：[mean=256, diff=256, cos_sim=1] → 513 维
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        # 零初始化输出层，使 MLP 初始输出 ≈ 0 → softmax(0,0) = (0.5, 0.5)
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)

        # 可学习的偏移项（初始为 0.5/0.5）
        self.logits_bias = nn.Parameter(torch.zeros(2))

        # 缓存：每次 forward 时记录 softmax 权重（用于诊断打印）
        self._cached_weights = None

    def forward(self, face_feat, fp_feat):
        # ── 模态差异信号 ──────────────────────────────────
        diff = face_feat - fp_feat
        cos_sim = F.cosine_similarity(face_feat, fp_feat, dim=1, eps=1e-8).unsqueeze(1)

        # 拼接：[B, 256] + [B, 256] + [B, 1] → [B, 513]
        pooled = (face_feat + fp_feat) / 2.0
        attn_input = torch.cat([pooled, diff, cos_sim], dim=1)

        # ── 模态权重 ───────────────────────────────────
        logits = self.attention(attn_input) + self.logits_bias
        weights = F.softmax(logits, dim=1)

        # 缓存最新 batch 的权重均值（用于诊断）
        self._cached_weights = weights.detach().cpu()

        # ── 加权融合 ───────────────────────────────────
        concat = torch.stack([face_feat, fp_feat], dim=1)
        fused = (concat * weights.unsqueeze(-1)).sum(dim=1)

        return fused

    def get_cached_weights(self):
        """返回上次 forward 时缓存的 softmax 权重（[B, 2] tensor on cpu）"""
        return self._cached_weights

    def get_ablation_info(self):
        return {"type": "attention"}


class AblationStrategy(FusionStrategy):
    """消融包装策略 — 将任意 FusionStrategy 包装为消融版本。

    ablating_face=True  → face_feat → 0（硬截断，保证消融彻底）
    ablating_fp=True    → fp_feat → 0（硬截断）

    当单个模态被消融时，只截断对应输入，保留另一个模态和增强器的贡献。
    当两个模态都被消融时，额外截断增强器残差（enhancer(0) 也归零），
    保证融合特征严格为零，从而得到真实的单模态消融性能。
    """

    def __init__(self, strategy: FusionStrategy, fusion_dim: int,
                 ablating_face: bool = False, ablating_fp: bool = False):
        super().__init__()
        self.strategy = strategy
        self.ablating_face = ablating_face
        self.ablating_fp = ablating_fp

    def forward(self, face_feat, fp_feat):
        if self.ablating_face:
            face_feat = torch.zeros_like(face_feat)
        if self.ablating_fp:
            fp_feat = torch.zeros_like(fp_feat)
        fused = self.strategy(face_feat, fp_feat)
        # 当两个模态都被消融时，enhancer 的残差不应贡献（否则 0+enhancer(0) != 0）
        if self.ablating_face and self.ablating_fp:
            fused = torch.zeros_like(fused)
        return fused

    def get_ablation_info(self):
        info = {"type": "ablation", "hard_zero": True}
        if self.ablating_face:
            info["face"] = "zero"
        if self.ablating_fp:
            info["fp"] = "zero"
        return info

    def get_cached_weights(self):
        """委托给内部策略"""
        return self.strategy.get_cached_weights()


# ── 工厂函数 ────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    "simple": WeightedSumStrategy,
    "adaptive": AttentionStrategy,
}


def create_fusion_strategy(strategy_name: str, fusion_dim: int, **kwargs) -> FusionStrategy:
    """工厂函数：创建融合策略实例

    Args:
        strategy_name: "simple" | "adaptive"
        fusion_dim: 融合特征维度
        **kwargs: 传递给策略的额外参数（如 attention 的 hidden_dim）

    Returns:
        FusionStrategy 实例
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown fusion strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name](fusion_dim, **kwargs)
