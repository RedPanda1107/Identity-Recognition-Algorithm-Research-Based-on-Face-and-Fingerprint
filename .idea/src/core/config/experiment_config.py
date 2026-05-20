"""
集中化实验配置系统。

消除硬编码，建立可审计的实验参数体系。
所有参数集中管理，确保训练、推理、实验框架使用完全一致的值。

包含：
  - ExperimentMode 枚举
  - EXPERIMENT_CONFIGS 每种模式的标准化超参数
  - AblationConfig 消融实验配置
  - PreprocessingConfig 预处理参数（归一化、CLAHE 等）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── 实验模式枚举 ──────────────────────────────────────────────────────────────

class ExperimentMode(Enum):
    SINGLE_FACE = "single_face"
    SINGLE_FINGER = "single_finger"
    FUSION_SIMPLE = "fusion_simple"
    FUSION_ADAPTIVE = "fusion_adaptive"
    FUSION_ONLY = "fusion_only"

    @classmethod
    def is_fusion(cls, mode: "ExperimentMode") -> bool:
        return mode in (cls.FUSION_SIMPLE, cls.FUSION_ADAPTIVE, cls.FUSION_ONLY)


# ── 消融实验配置 ─────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    enabled: bool = False
    ablate_face: bool = False      # True = 禁用 face（测试 fingerprint-only）
    ablate_fp: bool = False        # True = 禁用 fingerprint（测试 face-only）
    soft_ablation: bool = True     # True = 软门控消融，False = 硬截断
    ablation_start_epoch: int = 0  # 从第几 epoch 开始消融


# ── 预处理配置 ────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    # 归一化参数（统一使用 ImageNet 标准）
    face_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    face_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    fp_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])   # 指纹也统一用 ImageNet
    fp_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # CLAHE 参数（指纹用，人脸不用）
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = field(default_factory=lambda: (8, 8))


# ── 每种实验模式的标准化超参数 ──────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    ExperimentMode.SINGLE_FACE: {
        # 数据
        "batch_size": 32,
        "epochs": 50,
        "base_lr": 1e-4,
        "backbone_lr": 1e-4,       # 全部一起训练
        "weight_decay": 5e-4,
        "warmup_epochs": 5,
        "scheduler_step": 15,
        "scheduler_gamma": 0.1,
        # ArcFace（人脸标准配置）
        "arc_s": 30.0,
        "arc_m": 0.35,
        "label_smoothing": 0.0,
        "use_amp": False,
        "use_clahe": False,
    },
    ExperimentMode.SINGLE_FINGER: {
        # 数据
        "batch_size": 32,
        "epochs": 50,
        "freeze_epochs": 0,        # 0 = 不冻结，直接训练全部
        "unfreeze_lr": 1e-4,
        "head_lr_ratio": 5.0,      # head = backbone * 5.0
        # ArcFace（指纹标准配置）
        "arc_s": 30.0,
        "arc_m_warmup_epochs": 25,
        "arc_m_start": 0.0,
        "arc_m_end": 0.35,
        "arc_m_delay_epochs": 3,
        "use_clahe": True,
    },
    ExperimentMode.FUSION_SIMPLE: {
        # 数据
        "batch_size": 16,
        "epochs": 50,
        "backbone_lr": 1e-5,
        "fusion_lr": 1e-4,
        "weight_decay": 5e-4,
        "warmup_epochs": 5,
        "scheduler_step": 15,
        "scheduler_gamma": 0.1,
        # 融合
        "fusion_dim": 256,
        "dropout_rate": 0.3,
        "use_arcface": True,
        "arc_s": 30.0,            # 修正：与单模态一致
        "arc_m": 0.35,            # 修正：与单模态一致
        "use_amp": True,
        "face_dropout_prob": 0.0,  # 消融时由实验控制
        "entropy_penalty_weight": 0.0,
        "use_clahe": True,
        # 两阶段训练
        "stage1_epochs": 35,
        "stage1_fusion_lr": 1e-3,
        "stage2_fusion_lr": 1e-4,
        "stage2_backbone_lr": 1e-5,
    },
    ExperimentMode.FUSION_ADAPTIVE: {
        # 同 FUSION_SIMPLE，但额外有：
        "entropy_penalty_weight": 0.01,   # 鼓励 attention 平衡
        "attention_hidden_dim": 64,        # attention 中间层维度
    },
}


# ── 消融实验矩阵 ─────────────────────────────────────────────────────────────

ABLATION_EXPERIMENTS = {
    # 实验名称 → 实验配置
    "fusion_simple_face_only": {
        "base_mode": ExperimentMode.FUSION_SIMPLE,
        "ablation": AblationConfig(
            enabled=True,
            ablate_fp=True,          # 禁用 fingerprint
            ablate_face=False,
        ),
        "pretrained_ckpts": ["face"],   # 只加载 face checkpoint
        "output_suffix": "_face_only",
    },
    "fusion_simple_fp_only": {
        "base_mode": ExperimentMode.FUSION_SIMPLE,
        "ablation": AblationConfig(
            enabled=True,
            ablate_face=True,       # 禁用 face
            ablate_fp=False,
        ),
        "pretrained_ckpts": ["fingerprint"],  # 只加载 fingerprint checkpoint
        "output_suffix": "_fp_only",
    },
    "fusion_adaptive_face_only": {
        "base_mode": ExperimentMode.FUSION_ADAPTIVE,
        "ablation": AblationConfig(
            enabled=True,
            ablate_fp=True,
            ablate_face=False,
        ),
        "pretrained_ckpts": ["face"],
        "output_suffix": "_face_only",
    },
    "fusion_adaptive_fp_only": {
        "base_mode": ExperimentMode.FUSION_ADAPTIVE,
        "ablation": AblationConfig(
            enabled=True,
            ablate_face=True,
            ablate_fp=False,
        ),
        "pretrained_ckpts": ["fingerprint"],
        "output_suffix": "_fp_only",
    },
}


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def get_config(mode: ExperimentMode) -> dict:
    """获取指定实验模式的标准化配置"""
    if mode not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment mode: {mode}")
    return EXPERIMENT_CONFIGS[mode].copy()


def get_ablation_config(ablation_name: str) -> Optional[dict]:
    """获取消融实验配置"""
    return ABLATION_EXPERIMENTS.get(ablation_name)


def get_preprocessing_config() -> PreprocessingConfig:
    """获取预处理配置（所有实验共享）"""
    return PreprocessingConfig()
