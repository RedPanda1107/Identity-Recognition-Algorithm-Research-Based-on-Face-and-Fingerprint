"""
融合模型训练器
支持人脸+指纹多模态训练
支持消融实验（单模态缺失测试）
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from .base_trainer import BaseTrainer, AverageMeter


class FusionTrainer(BaseTrainer):
    """多模态融合训练器 - 支持完整训练和消融实验

    支持：
    - 特征级融合（人脸 + 指纹）
    - ArcFace 度量学习
    - 开放集检索验证（Gallery/Query）
    - 单模态 vs 融合对比评估
    - 消融实验（单模态缺失测试）
    """

    MODALITY = 'fusion'

    def __init__(self, fusion_model, face_model, fingerprint_model,
                 train_loader, val_loader, test_loader=None,
                 optimizer=None, scheduler=None, criterion=None,
                 device='cuda', logger=None, pretrained_ckpts=None, freeze_backbone=False,
                 use_amp=True, accumulation_steps=1, seed=42,
                 experiment_mode='full', ablate_modality=None,
                 label_smoothing=0.0, tb_writer=None,
                 face_dropout_prob=0.0, fp_corruption_prob=0.0,
                 modality_drop_strategy='both',
                 freeze_projection=False,
                 entropy_penalty_weight=0.0,
                 balance_lr=0.1, balance_weight_decay=2.0,
                 zero_face_input=False, zero_fp_input=False):
        super().__init__(
            fusion_model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, logger, tb_writer
        )

        self.face_model = face_model.to(device) if face_model else None
        self.fingerprint_model = fingerprint_model.to(device) if fingerprint_model else None
        self.freeze_backbone = freeze_backbone
        self.current_epoch = 0
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.ablate_modality = ablate_modality
        # 消融训练模式：将指定模态的 backbone 特征直接置零
        # 与旧的 GatedIdentityAblation 硬截断不同，这里在特征提取层面直接零化
        # 使得投影层和分类头能从对应单模态预训练权重开始 fine-tune
        self.zero_face_input = zero_face_input
        self.zero_fp_input = zero_fp_input
        self.label_smoothing = label_smoothing
        self.test_loader = test_loader
        self.test_dataset = test_loader.dataset if test_loader else None

        # 模态平衡策略
        self.face_dropout_prob = face_dropout_prob
        self.fp_corruption_prob = fp_corruption_prob
        # 'clean' | 'face_dropout' | 'fp_corruption' | 'both'
        self.modality_drop_strategy = modality_drop_strategy
        self.freeze_projection = freeze_projection
        self.entropy_penalty_weight = entropy_penalty_weight
        self._balance_lr = balance_lr
        self._balance_weight_decay = balance_weight_decay
        self._last_attention_weights = None

        # AMP 配置
        self.use_amp = use_amp and device.type == 'cuda'
        self.accumulation_steps = max(1, accumulation_steps)
        self._scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # 带 label_smoothing 的 CrossEntropyLoss（避免每 batch 重建）
        self._criterion_ls = nn.CrossEntropyLoss(label_smoothing=label_smoothing) \
            if label_smoothing > 0 else None

        if self.use_amp:
            self.logger.info(f"[AMP] Mixed precision training enabled (accumulation_steps={self.accumulation_steps})")

        # 实验模式日志
        self._log_experiment_mode()

        # 加载预训练权重
        self._load_pretrained_weights(pretrained_ckpts)

        # 冻结backbone（如需要）
        if freeze_backbone or experiment_mode == 'fusion_only':
            self._freeze_feature_extractors()

        # 设置消融模式：
        # 旧的 GatedIdentityAblation 硬截断已被移除。
        # 消融现在通过 zero_face_input / zero_fp_input 在特征提取层面完成，
        # 使得投影层能从单模态预训练权重开始 fine-tune。
        if self.ablate_modality:
            if self.ablate_modality == 'fingerprint':
                self.zero_face_input = False
                self.zero_fp_input = True   # 指纹置零，只用人脸
                self.logger.info(f"[Ablation] fingerprint 输入置零（从 face checkpoint fine-tune）")
            elif self.ablate_modality == 'face':
                self.zero_fp_input = False
                self.zero_face_input = True   # 人脸置零，只用指纹
                self.logger.info(f"[Ablation] face 输入置零（从 fingerprint checkpoint fine-tune）")

        # 设置可训练参数
        self._setup_trainable_params()

        # 初始化验证集图像变换
        self._init_val_transforms()

        # 注册 attention hook（捕获 per-batch 权重用于 entropy penalty）
        self._attn_outputs = []
        self._attn_hook_handle = None
        if self.entropy_penalty_weight > 0:
            self._register_attention_hook()

        # 独立优化器：专门管理 logits_bias，快速响应均衡压力
        self._balance_optimizer = None
        self._balance_scheduler = None
        if self.entropy_penalty_weight > 0:
            self._setup_balance_optimizer()

        # 模态腐败日志
        if self.modality_drop_strategy in ('face_dropout', 'both') and self.face_dropout_prob > 0:
            self.logger.info(f"[ModalityDrop] Face dropout: prob={self.face_dropout_prob}")
        if self.modality_drop_strategy in ('fp_corruption', 'both') and self.fp_corruption_prob > 0:
            self.logger.info(f"[ModalityDrop] FP corruption: prob={self.fp_corruption_prob}")

        # Gallery 缓存
        self._gallery_embeddings_cache = None
        self._gallery_labels_cache = None
        self._gallery_dirty = True
        self._last_best_acc = -1.0
        self._ablation_verification_done = False  # 每个实验只做一次指纹独立验证

    def _register_attention_hook(self):
        """注册 attention hook 到 fusion_strategy 内部的 attention 层（用于 entropy penalty）"""
        strategy = self.model.fusion_strategy
        if hasattr(strategy, 'strategy'):
            strategy = strategy.strategy
        if hasattr(strategy, 'attention') and hasattr(strategy.attention, 'register_forward_hook'):
            def _hook(module, inp, out):
                # out 是 raw logits [B, 2]，需要 softmax 后再缓存
                weights = F.softmax(out, dim=1)
                self._attn_outputs.append(weights.detach())
            self._attn_hook_handle = strategy.attention.register_forward_hook(_hook)
            self.logger.info(
                f"[EntropyPenalty] Registered attention hook "
                f"(weight={self.entropy_penalty_weight})"
            )

    def _setup_balance_optimizer(self):
        """为 logits_bias 创建独立的优化器和学习率调度器。

        策略：
        - lr: 高学习率（默认 0.1），比主优化器大 100 倍，快速响应均衡压力
        - weight_decay: 强 L2 正则（默认 2.0），强制 logits_bias 趋向 0 → softmax → 0.5/0.5
        - CosineAnnealingLR: 前期强制均衡，后期逐渐放松约束
        """
        strategy = self.model.fusion_strategy
        if hasattr(strategy, 'strategy'):
            strategy = strategy.strategy

        bias_params = []
        if hasattr(strategy, 'logits_bias') and strategy.logits_bias.requires_grad:
            bias_params.append(strategy.logits_bias)
            self.logger.info(f"[BalanceOpt] logits_bias found (shape={strategy.logits_bias.shape})")
        else:
            self.logger.warning("[BalanceOpt] logits_bias not found or frozen, skipping")
            return

        self._balance_optimizer = torch.optim.AdamW(
            bias_params,
            lr=self._balance_lr,
            weight_decay=self._balance_weight_decay
        )
        self._balance_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._balance_optimizer,
            T_max=30,
            eta_min=1e-4
        )
        self.logger.info(
            f"[BalanceOpt] Separate optimizer for logits_bias: "
            f"lr={self._balance_lr}, weight_decay={self._balance_weight_decay}, "
            f"CosineAnnealing(T_max=30, eta_min=0.0001)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 融合权重诊断工具
    # ─────────────────────────────────────────────────────────────────────────
    def _log_fusion_weights(self, phase: str, with_attention: bool = False):
        """打印当前模型的融合权重状态（诊断用）

        - Simple:  softmax(fusion_weight)
        - Adaptive: logits_bias softmax 初始分布 + 真实 cached softmax 权重
        - with_attention=True（仅 eval 阶段）: 使用 forward 后缓存的真实权重
        """
        model = self.model
        if not hasattr(model, 'get_fusion_weights'):
            return
        w = model.get_fusion_weights()

        if hasattr(model, 'get_cached_attention_weights') and with_attention:
            gate_f, gate_fp = w
            cached = model.get_cached_attention_weights()
            if cached is not None:
                attn_face, attn_fp = cached
                attn_str = f"attn_face={attn_face:.4f} attn_fp={attn_fp:.4f}"
            else:
                # 调试：诊断缓存为空的原因
                strategy = model.fusion_strategy
                raw = None
                if hasattr(strategy, 'get_cached_weights'):
                    raw = strategy.get_cached_weights()
                self.logger.warning(
                    f"[DEBUG:attn] phase={phase} | raw_cached={raw} | "
                    f"strategy_type={type(strategy).__name__}"
                )
                attn_str = "attn=N/A(cold)"
            self.logger.info(
                f"[{phase}权重] gate_face={gate_f:.4f} gate_fp={gate_fp:.4f} | {attn_str}"
            )
        elif hasattr(model, 'get_attention_weights'):
            gate_f, gate_fp = w
            self.logger.info(
                f"[{phase}权重] gate_face={gate_f:.4f} gate_fp={gate_fp:.4f} | attn=N/A"
            )
        else:
            # Simple
            self.logger.info(
                f"[{phase}权重] simple_face={w[0]:.4f} simple_fp={w[1]:.4f}"
            )

        # ── 消融验证：显式检查硬截断效果 ─────────────────────────────
        if self.ablate_modality:
            w = model.get_fusion_weights()
            self.logger.info(
                f"[AblationCheck] disabled={self.ablate_modality} | "
                f"face_w={w[0]:.4f} fp_w={w[1]:.4f}"
            )

    def _log_experiment_mode(self):
        """记录实验模式配置"""
        mode_descriptions = {
            'full': '训练全部（backbone + fusion）',
            'fusion_only': '冻结backbone，只训练融合层',
            'face_ablation': '消融实验：从 face checkpoint fine-tune，指纹输入置零',
            'fp_ablation': '消融实验：从 fingerprint checkpoint fine-tune，人脸输入置零',
        }
        desc = mode_descriptions.get(self.experiment_mode, '未知模式')
        self.logger.info(f"[Experiment] Mode: {self.experiment_mode} - {desc}")
        if self.zero_face_input:
            self.logger.info(f"[Ablation] face 输入置零（zero_face_input=True）")
        if self.zero_fp_input:
            self.logger.info(f"[Ablation] fingerprint 输入置零（zero_fp_input=True）")

    def _init_val_transforms(self):
        """初始化验证时的图像变换（用于Gallery特征提取）

        关键：指纹必须应用 CLAHE，与 FusionDataset.__getitem__ 中的处理保持一致。
        FusionDataset 对指纹始终应用 CLAHE（use_clahe=True），所以验证时的 Gallery 特征
        提取也必须应用 CLAHE，否则 Gallery 和 Query 的指纹预处理不对称。
        """
        # 指纹 CLAHE Transform（与 FusionDataset._apply_clahe 和 FingerprintDataset 保持一致）
        class ClahePIL:
            def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
                self._clahe = cv2.createCLAHE(
                    clipLimit=clip_limit, tileGridSize=tile_size
                )
            def __call__(self, img: Image.Image) -> Image.Image:
                img_np = np.array(img)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                enhanced = self._clahe.apply(gray)
                return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

        self.val_face_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 指纹：Resize → CLAHE → ToTensor → Normalize（与 FusionDataset pipeline 完全一致）
        self.val_fp_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ClahePIL(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _setup_trainable_params(self):
        """设置可训练参数。

        策略：
          - Backbone：是否冻结由 freeze_backbone 控制
          - Projection layers：默认解冻，freeze_projection=True 时冻结（仅 fusion_only 模式推荐）
        """
        if self.freeze_backbone or self.experiment_mode == 'fusion_only':
            for p in self.model.parameters():
                p.requires_grad = True
            if self.face_model:
                for p in self.face_model.parameters():
                    p.requires_grad = False
            if self.fingerprint_model:
                for p in self.fingerprint_model.parameters():
                    p.requires_grad = False
            # 投影层：默认解冻；freeze_projection=True 时冻结
            if self.freeze_projection:
                for name, param in self.model.named_parameters():
                    if 'proj' in name:
                        param.requires_grad = False
                self.logger.info("[Config] Training fusion head + classifier only (backbones + projections frozen)")
            else:
                self.logger.info("[Config] Training fusion head + projections (backbones frozen)")
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            if self.face_model:
                for p in self.face_model.parameters():
                    p.requires_grad = True
            if self.fingerprint_model:
                for p in self.fingerprint_model.parameters():
                    p.requires_grad = True
            self.logger.info("[Config] Training fusion + all backbones")

        # 统计可训练参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"[Params] Fusion model: {trainable:,}/{total:,} trainable "
                         f"({100*trainable/total:.1f}%)")

    def _load_single_modality_weights(self, model, ckpt_path, modality_name):
        """加载单个模态的预训练权重"""
        if not model or not os.path.exists(ckpt_path):
            self.logger.warning(f"[{modality_name}] Checkpoint not found")
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state', ckpt)

            model_state = model.state_dict()
            matched = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    matched[k] = v

            if matched:
                model.load_state_dict(matched, strict=False)
                self.logger.info(f"[{modality_name}] Loaded {len(matched)} params")
                return True
        except Exception as e:
            self.logger.warning(f"[{modality_name}] Load failed: {e}")
        return False

    def _load_pretrained_weights(self, pretrained_ckpts):
        """加载预训练权重。

        消融实验（face_ablation / fp_ablation）：
          从对应单模态预训练权重开始 fine-tune，而非从头训练 ImageNet 权重。
          这是消融实验公平对比的关键：消融模型的性能应该与单模态 baseline 对比，
          两者都应该使用相同的单模态预训练初始化。

        规则：
          - full / fusion_only：从 face + fp 两个单模态 checkpoint 加载（与旧逻辑相同）
          - face_ablation：从 face checkpoint 加载（用于 face-only 实验）
          - fp_ablation：从 fingerprint checkpoint 加载（用于 fp-only 实验）
        """
        if not pretrained_ckpts:
            # 未指定预训练路径：尝试自动搜索
            # face_ablation 只搜索 face checkpoint
            # fp_ablation 只搜索 fingerprint checkpoint
            # full / fusion_only 搜索两者
            if self.experiment_mode == 'face_ablation':
                self.logger.info("[Pretrained] face_ablation 模式：从 face checkpoint 加载")
            elif self.experiment_mode == 'fp_ablation':
                self.logger.info("[Pretrained] fp_ablation 模式：从 fingerprint checkpoint 加载")
            else:
                self.logger.info("[Pretrained] 未指定预训练路径，backbone 使用 ImageNet 预训练权重")
            return

        if self.face_model and pretrained_ckpts.get('face'):
            self._load_single_modality_weights(self.face_model, pretrained_ckpts['face'], "Face")

        if self.fingerprint_model and pretrained_ckpts.get('fingerprint'):
            self._load_single_modality_weights(self.fingerprint_model, pretrained_ckpts['fingerprint'], "FP")

    def _freeze_feature_extractors(self):
        """冻结特征提取器（standalone backbone）。投影层由 _setup_trainable_params 单独控制。"""
        if self.face_model:
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.fingerprint_model:
            for param in self.fingerprint_model.parameters():
                param.requires_grad = False
        # 投影层冻结由 _setup_trainable_params 中的 freeze_projection 控制

    def _apply_ablation(self, face_features, fp_features):
        """已废弃。消融逻辑现已移入 fusion_model 内部（identity mask），
        保留此方法是为了向后兼容，但不再做零截断。"""
        return face_features, fp_features

    def _extract_features_train(self, face_images, fp_images):
        """训练时提取特征（允许梯度），支持批次级模态腐败策略。

        模态腐败策略（modality_drop_strategy）：
          - 'clean': 不做腐败，两个模态始终存在
          - 'face_dropout': face 有 dropout_prob 概率整个批次置零
          - 'fp_corruption': fp 有 corruption_prob 概率整个批次被腐蚀
          - 'both': 两者都生效（推荐；face dropout + fp corruption）
        """
        if self.face_model:
            self.face_model.train()
            face_out = self.face_model(face_images, return_features=True)
            face_features = face_out[1] if isinstance(face_out, tuple) else face_out
        else:
            face_features = torch.randn(face_images.size(0), 512, device=self.device)

        if self.fingerprint_model:
            self.fingerprint_model.train()
            fp_out = self.fingerprint_model(fp_images, return_features=True)
            fp_features = fp_out[1] if isinstance(fp_out, tuple) else fp_out
        else:
            fp_features = torch.randn(fp_images.size(0), 512, device=self.device)

        # 消融模式：将指定模态特征置零
        if self.zero_face_input:
            face_features = torch.zeros_like(face_features)
        if self.zero_fp_input:
            fp_features = torch.zeros_like(fp_features)

        # ── 模态腐败策略 ──────────────────────────────────────
        if self.modality_drop_strategy in ('face_dropout', 'both'):
            if self.face_dropout_prob > 0 and random.random() < self.face_dropout_prob:
                face_features = torch.zeros_like(face_features)

        if self.modality_drop_strategy in ('fp_corruption', 'both'):
            if self.fp_corruption_prob > 0 and random.random() < self.fp_corruption_prob:
                noise = torch.randn_like(fp_features) * 0.1
                fp_features = fp_features * 0.7 + noise

        return face_features, fp_features

    def _extract_features_eval(self, face_images, fp_images):
        """验证时提取特征（无梯度）

        使用 return_features=True 确保始终返回嵌入向量。
        消融模式（zero_face_input / zero_fp_input）：与训练一致，将指定模态置零。
        """
        if self.face_model:
            self.face_model.eval()
            with torch.no_grad():
                face_out = self.face_model(face_images, return_features=True)
                face_features = face_out[1] if isinstance(face_out, tuple) else face_out
        else:
            face_features = torch.randn(face_images.size(0), 512, device=self.device)

        if self.fingerprint_model:
            self.fingerprint_model.eval()
            with torch.no_grad():
                fp_out = self.fingerprint_model(fp_images, return_features=True)
                fp_features = fp_out[1] if isinstance(fp_out, tuple) else fp_out
        else:
            fp_features = torch.randn(fp_images.size(0), 512, device=self.device)

        # 消融模式：与训练保持一致
        if self.zero_face_input:
            face_features = torch.zeros_like(face_features)
        if self.zero_fp_input:
            fp_features = torch.zeros_like(fp_features)

        return face_features, fp_features

    def _load_gallery_batch(self, batch_pairs, batch_size):
        """加载一个Gallery批次的图像并提取特征"""
        face_imgs = []
        fp_imgs = []

        for pair in batch_pairs:
            face_path, fp_path = pair
            try:
                face_img = Image.open(face_path).convert('RGB')
                face_img = self.val_face_transform(face_img)
                face_imgs.append(face_img)
            except Exception as e:
                self.logger.warning(f"[Gallery] Failed to load face: {face_path}, {e}")
                face_imgs.append(torch.zeros(3, 224, 224))

            try:
                fp_img = Image.open(fp_path).convert('RGB')
                fp_img = self.val_fp_transform(fp_img)
                fp_imgs.append(fp_img)
            except Exception as e:
                self.logger.warning(f"[Gallery] Failed to load fingerprint: {fp_path}, {e}")
                fp_imgs.append(torch.zeros(3, 224, 224))

        face_batch = torch.stack(face_imgs).to(self.device)
        fp_batch = torch.stack(fp_imgs).to(self.device)

        return face_batch, fp_batch

    # ─────────────────────────────────────────────────────────────────────────
    # EER 计算工具
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_eer(labels, scores):
        from sklearn.metrics import roc_curve
        if len(np.unique(labels)) < 2:
            return 0.0, 0.0
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        threshold = float(thresholds[eer_idx])
        return eer, threshold

    # ─────────────────────────────────────────────────────────────────────────
    # 单步训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, batch, scaler=None, use_amp=False):
        """单步训练。

        流程：特征提取 → 融合 → 分类 logits → CrossEntropyLoss → AMP backward
        - NaN 防护：输入/特征/logits/loss 四层检测
        - AMP：前向 fp16 + loss scale + loss 转 fp32（防 exp 溢出）
        """
        face_images = batch['face_image'].to(self.device)
        fp_images = batch['fingerprint_image'].to(self.device)
        targets = batch['label'].to(self.device)

        if torch.isnan(face_images).any() or torch.isinf(face_images).any():
            return None, None, 0.0, "input_nan"
        if torch.isnan(fp_images).any() or torch.isinf(fp_images).any():
            return None, None, 0.0, "input_nan"

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                face_features, fp_features = self._extract_features_train(face_images, fp_images)
                if torch.isnan(face_features).any() or torch.isinf(face_features).any():
                    return None, None, 0.0, "face_feature_nan"
                if torch.isnan(fp_features).any() or torch.isinf(fp_features).any():
                    return None, None, 0.0, "fp_feature_nan"

                # 清理残留 NaN
                face_features = torch.where(torch.isnan(face_features),
                    torch.zeros_like(face_features), face_features)
                fp_features = torch.where(torch.isnan(fp_features),
                    torch.zeros_like(fp_features), fp_features)

                # 传入 labels 以激活 ArcFace margin（无 labels 时 ArcFace 仅返回 cos*s，无 margin）
                outputs = self.model(face_features, fp_features, labels=targets)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    return None, outputs, 0.0, "logits_nan"

                if self._criterion_ls is not None:
                    loss = self._criterion_ls(outputs.float(), targets)
                else:
                    loss = self.criterion(outputs.float(), targets)
        else:
            face_features, fp_features = self._extract_features_train(face_images, fp_images)
            if torch.isnan(face_features).any() or torch.isinf(face_features).any():
                return None, None, 0.0, "face_feature_nan"
            if torch.isnan(fp_features).any() or torch.isinf(fp_features).any():
                return None, None, 0.0, "fp_feature_nan"

            face_features = torch.where(torch.isnan(face_features),
                torch.zeros_like(face_features), face_features)
            fp_features = torch.where(torch.isnan(fp_features),
                torch.zeros_like(fp_features), fp_features)

            outputs = self.model(face_features, fp_features, labels=targets)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return None, outputs, 0.0, "logits_nan"

            if self._criterion_ls is not None:
                loss = self._criterion_ls(outputs.float(), targets)
            else:
                loss = self.criterion(outputs.float(), targets)

        # ── Attention 均衡正则化：直接惩罚权重偏离 0.5 ──────────────────
        # 目标：使 attention weights 趋向 (0.5, 0.5)，即 logits 趋向 (0, 0)
        # 对 logits 本身做 L2 正则，等价于推动 softmax 权重趋向均匀分布
        balance_loss = 0.0
        if self.entropy_penalty_weight > 0 and self._attn_outputs:
            attn = self._attn_outputs[-1]   # [B, 2] softmax weights
            self._attn_outputs.clear()
            # 惩罚偏离均匀分布的程度：deviation = ||w - 0.5||_2
            # 熵正则的变体：直接对 softmax 权重做 L2 损失趋向 0.5
            deviation = ((attn - 0.5) ** 2).mean()  # scalar
            balance_loss = deviation
            loss = loss + self.entropy_penalty_weight * balance_loss

        if torch.isnan(loss):
            return None, outputs, 0.0, "loss_nan"

        with torch.no_grad():
            preds = outputs.float().argmax(dim=1)
            acc = (preds == targets).float().mean().item()

        return loss, outputs, acc, "valid"

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch, total_epochs=None, use_amp=False):
        """训练一轮。

        NaN-safe：遇到 NaN batch 跳过，但继续处理后续 batch。
        包含特征范数监控（应约等于 1.0）。
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        feat_norm_meter = AverageMeter()
        scaler = self._scaler
        nan_count = 0
        nan_reasons = {}

        # ── 打印融合权重诊断（每 epoch 一次，使用上一 batch 缓存的 attention 权重）──
        self._log_fusion_weights("Trn", with_attention=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            result = self.train_step(batch, scaler, use_amp=use_amp)
            if result[0] is None:
                nan_count += 1
                reason = result[3]
                nan_reasons[reason] = nan_reasons.get(reason, 0) + 1
                continue

            loss, outputs, acc, _ = result

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积步数到达时更新
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # 独立优化 logits_bias（高 lr + 强 weight_decay）
                if self._balance_optimizer is not None:
                    self._balance_optimizer.step()
                    self._balance_optimizer.zero_grad()

            batch_size = outputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            # 特征范数监控
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)
            with torch.no_grad():
                face_feat, fp_feat = self._extract_features_eval(face_images, fp_images)

                # ── 断点监控：验证硬置零效果 ─────────────────────────────
                _train_debug_printed = getattr(self, '_train_debug_printed', 0)
                if _train_debug_printed < 3:
                    face_nan = torch.isnan(face_feat).sum().item()
                    fp_nan = torch.isnan(fp_feat).sum().item()
                    self.logger.info(
                        f"[DEBUG:train:{_train_debug_printed}] "
                        f"face_feat: mean={face_feat.mean().item():.6f} "
                        f"std={face_feat.std().item():.6f} nan={face_nan} "
                        f"| fp_feat: mean={fp_feat.mean().item():.6f} "
                        f"std={fp_feat.std().item():.6f} nan={fp_nan}"
                    )
                    self._train_debug_printed = _train_debug_printed + 1

                fused = self.model.extract_fused_features(face_feat, fp_feat)
                fused = F.normalize(fused, p=2, dim=1)
                feat_norm = fused.norm(dim=1).mean().item()
                if not np.isnan(feat_norm):
                    feat_norm_meter.update(feat_norm, batch_size)

                if feat_norm < 0.1 or feat_norm > 2.0:
                    tqdm.write(f"[调试] 特征范数异常: feat_norm={feat_norm:.4f}, 预期约等于1.0")

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

        # 处理剩余未更新的梯度
        if (batch_idx + 1) % self.accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self._balance_optimizer is not None:
                self._balance_optimizer.step()
                self._balance_optimizer.zero_grad()

        if nan_count > 0:
            nan_detail = ", ".join(f"{k}={v}" for k, v in nan_reasons.items())
            self.logger.warning(
                f"[训练] Epoch {epoch+1}: 共 {nan_count} 个 NaN batch 已跳过 "
                f"({nan_detail})，有效 batch 数={loss_meter.count}"
            )

        # 统一日志格式
        current_lr = self.optimizer.param_groups[0]['lr']
        gallery_size = len(self.val_loader.dataset.val_gallery_paths) if hasattr(self.val_loader.dataset, 'val_gallery_paths') else None
        query_size = len(self.val_loader.dataset.val_query_paths) if hasattr(self.val_loader.dataset, 'val_query_paths') else None

        self.log_train_epoch(
            epoch=epoch + 1,
            total_epochs=total_epochs,
            lr=current_lr,
            loss=loss_meter.avg,
            acc=acc_meter.avg,
            gallery_size=gallery_size,
            query_size=query_size
        )

        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', loss_meter.avg, epoch)
            self.tb_writer.add_scalar('train/accuracy', acc_meter.avg, epoch)
            self.tb_writer.add_scalar('train/feature_norm', feat_norm_meter.avg, epoch)

        return loss_meter.avg, acc_meter.avg

    def step_balance_scheduler(self):
        """每个 epoch 结束后步进 balance 学习率调度器（由 train_fusion.py 调用）"""
        if self._balance_scheduler is not None:
            self._balance_scheduler.step()

    # ─────────────────────────────────────────────────────────────────────────
    # 单轮验证（Gallery/Query 1:N 余弦相似度检索）
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate_epoch(self, epoch, total_epochs=50, use_amp=False):
        """1:N 开集检索验证（Gallery/Query 余弦相似度，完全脱离分类头）。

        流程：
            1. 提取 Gallery 特征（L2 归一化）
            2. 提取 Query 特征（L2 归一化）
            3. 计算余弦相似度矩阵
            4. 计算 Rank-1/5/10/20 准确率
            5. 计算 EER（同人匹配 vs 异人拒绝）
        """
        self.model.eval()
        val_dataset = self.val_loader.dataset

        # ── 检查 Gallery/Query 数据是否就绪
        if not hasattr(val_dataset, 'val_gallery_paths') or not val_dataset.val_gallery_paths:
            self.logger.error("[验证] val_gallery_paths 不存在！检查 FusionDataset 是否已更新。")
            return 0.0, 0.0, {}

        gallery_paths = val_dataset.val_gallery_paths
        gallery_labels_arr = np.array(val_dataset.val_gallery_labels)
        batch_size = self.val_loader.batch_size

        # ── 步骤 1：提取 Gallery 特征 ───────────────────────────────────
        gallery_embeddings_list = []
        feat_norm_g_meter = AverageMeter()

        self.logger.info("[Gallery] 正在提取特征...")
        n_gallery_batches = (len(gallery_paths) + batch_size - 1) // batch_size
        pbar_g = tqdm(total=n_gallery_batches, desc="Gallery", leave=False)
        for start_idx in range(0, len(gallery_paths), batch_size):
            end = min(start_idx + batch_size, len(gallery_paths))
            batch_pairs = gallery_paths[start_idx:end]
            face_batch, fp_batch = self._load_gallery_batch(batch_pairs, batch_size)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)
            else:
                face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features = F.normalize(fused_features, p=2, dim=1)

            # ── 断点监控：打印特征统计（仅前 3 个 batch）─────────────────
            _gallery_debug_printed = getattr(self, '_gallery_debug_printed', 0)
            if _gallery_debug_printed < 3:
                face_nan = torch.isnan(face_features).sum().item()
                fp_nan = torch.isnan(fp_features).sum().item()
                self.logger.info(
                    f"[DEBUG:gallery:{start_idx//batch_size}] "
                    f"face_feat: mean={face_features.mean().item():.6f} "
                    f"std={face_features.std().item():.6f} nan={face_nan} "
                    f"| fp_feat: mean={fp_features.mean().item():.6f} "
                    f"std={fp_features.std().item():.6f} nan={fp_nan}"
                )
                self._gallery_debug_printed = _gallery_debug_printed + 1

            feat_norm = fused_features.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_g_meter.update(feat_norm, fused_features.size(0))

            gallery_embeddings_list.append(fused_features.cpu())
            pbar_g.update()
        pbar_g.close()

        if not gallery_embeddings_list:
            self.logger.error("[验证] Gallery 特征为空！")
            return 0.0, 0.0, {}

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)  # [G, fusion_dim]
        self.logger.info(
            f"[Gallery] {len(gallery_paths)} 对（验证集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── 步骤 2：提取 Query 特征 ─────────────────────────────────────
        query_embeddings_list = []
        query_labels_list = []
        feat_norm_q_meter = AverageMeter()
        loss_meter = AverageMeter()
        # Validation loss is not computed here (ArcFace without labels → meaningless).
        # The real evaluation metric is the Gallery/Query retrieval accuracy.

        self.logger.info("[验证] 正在提取 Query 特征...")
        pbar_q = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        # 保留原始 backbone 特征用于消融验证（直接在各模态置零后重新融合）
        self._raw_query_face_list = []
        self._raw_query_fp_list = []
        for batch in pbar_q:
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)
            targets = batch['label']

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_images, fp_images)
            else:
                face_features, fp_features = self._extract_features_eval(face_images, fp_images)

            # 保留原始 backbone 特征（消融验证直接复用，不重新跑 backbone）
            self._raw_query_face_list.append(face_features.detach().cpu())
            self._raw_query_fp_list.append(fp_features.detach().cpu())

            # ── 断点监控：打印特征统计（仅前 3 个 batch）─────────────────
            _query_debug_printed = getattr(self, '_query_debug_printed', 0)
            if _query_debug_printed < 3:
                face_nan = torch.isnan(face_features).sum().item()
                fp_nan = torch.isnan(fp_features).sum().item()
                self.logger.info(
                    f"[DEBUG:query:{_query_debug_printed}] "
                    f"face_feat: mean={face_features.mean().item():.6f} "
                    f"std={face_features.std().item():.6f} nan={face_nan} "
                    f"| fp_feat: mean={fp_features.mean().item():.6f} "
                    f"std={fp_features.std().item():.6f} nan={fp_nan}"
                )
                self._query_debug_printed = _query_debug_printed + 1

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features_q = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_q_meter.update(feat_norm, face_images.size(0))

            query_embeddings_list.append(fused_features_q.cpu())
            query_labels_list.extend(targets.tolist())

        if not query_embeddings_list:
            self.logger.error("[验证] 没有 Query 样本！")
            return 0.0, 0.0, {}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)  # [Q, fusion_dim]
        query_labels = np.array(query_labels_list)

        if torch.isnan(query_embeddings).any() or torch.isnan(gallery_embeddings).any():
            self.logger.error("[验证] 特征包含 NaN！模型特征提取器崩溃。")
            return float('nan'), 0.0, {"feature_norm": 0.0}

        self.logger.info(
            f"[验证] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个验证人, "
            f"特征范数={feat_norm_q_meter.avg:.4f}"
        )

        # ── 打印融合权重诊断（使用 Query 循环中 forward 缓存的 attention 权重）────
        self._log_fusion_weights("Val", with_attention=True)

        # ── 步骤 3：计算余弦相似度矩阵 ────────────────────────────────
        self.logger.info("[验证] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())  # [Q, G]

        # ── 步骤 4：Rank-K 准确率 ────────────────────────────────────
        top_k = min(20, gallery_embeddings.size(0))
        _, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        top_k_indices = top_k_indices.numpy()
        top_k_labels = gallery_labels_arr[top_k_indices]

        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k:
                correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] in top_k_labels[i, :k]
                )
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        # ── 步骤 5：EER（同人匹配 vs 异人拒绝）─────────────────────
        eer = 0.0
        try:
            positive_scores = []
            negative_scores = []
            rng = np.random.RandomState(self.seed)

            for q_idx in range(len(query_labels)):
                q_label = query_labels[q_idx]
                q_emb = query_embeddings[q_idx]

                same_idx = np.where(gallery_labels_arr == q_label)[0]
                if len(same_idx) > 0:
                    sims = (q_emb @ gallery_embeddings[same_idx].t()).numpy()
                    topk_k = min(5, len(sims))
                    vals = np.sort(sims)[-topk_k:]
                    positive_scores.extend(vals.tolist())

                diff_idx = np.where(gallery_labels_arr != q_label)[0]
                if len(diff_idx) > 0:
                    n_neg = min(3, len(diff_idx))
                    selected = rng.choice(diff_idx, n_neg, replace=False)
                    sims = (q_emb @ gallery_embeddings[selected].t()).numpy()
                    negative_scores.extend(sims.tolist())

            n_pos, n_neg = len(positive_scores), len(negative_scores)
            if n_pos >= 50 and n_neg >= 50:
                scores = np.array(positive_scores + negative_scores)
                eer_labels = np.array([1] * n_pos + [0] * n_neg)
                eer, eer_th = self.calculate_eer(eer_labels, scores)
                self.logger.info(
                    f"[EER] 正样本={n_pos}, 负样本={n_neg}, "
                    f"EER={eer:.4f} (阈值={eer_th:.4f})"
                )
            else:
                self.logger.warning(
                    f"[EER] 样本不足（正={n_pos}/需50，负={n_neg}/需50），跳过 EER"
                )
        except Exception as e:
            self.logger.warning(f"[EER计算] 失败: {e}")

        # ── 模态独立性验证（两种消融模式各执行一次）──────────────────────────────────
        # query_embeddings: 250 样本（50 人，每人 5 剩余样本）
        # gallery_embeddings: 150 对（50 人，每人 3 gallery 样本）
        # 消融验证在原始 backbone 特征上重新融合（不重新跑 backbone）
        if self.ablate_modality and not self._ablation_verification_done:
            self._ablation_verification_done = True
            self.logger.info("=" * 60)

            raw_face = torch.cat(self._raw_query_face_list, dim=0).to(self.device)
            raw_fp = torch.cat(self._raw_query_fp_list, dim=0).to(self.device)

            if self.ablate_modality == 'fingerprint':
                self.logger.info("[AblationVerification:face_ablation] 禁用指纹，仅用人脸特征检索...")
                orig_ablate_modality = self.model._ablate_modality
                self.model.set_ablation(None)

                face_zeros = torch.zeros_like(raw_face)
                with torch.no_grad():
                    fused = self.model.extract_fused_features(face_zeros, raw_fp)
                    fp_only_query_emb = F.normalize(fused, p=2, dim=1)

                fp_only_sim = torch.mm(fp_only_query_emb, gallery_embeddings.to(fp_only_query_emb.device).t())
                _, fp_only_top1 = torch.topk(fp_only_sim, k=1, dim=1)
                fp_only_top1_labels = gallery_labels_arr[fp_only_top1.cpu().numpy()]
                fp_only_correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] == fp_only_top1_labels[i, 0]
                )
                fp_only_rank1 = fp_only_correct / len(query_labels)
                self.logger.info(
                    f"[AblationVerification:face_ablation] "
                    f"纯指纹 Rank-1 = {fp_only_rank1:.4f} (期望约等于纯指纹单模态基线)"
                )
                self.logger.info(
                    f"[AblationVerification:face_ablation] "
                    f"融合 Rank-1 = {rank1_acc:.4f}"
                )
                if fp_only_rank1 < 0.50:
                    self.logger.info("[AblationVerification] PASS: 纯指纹 Rank-1 < 50%")
                else:
                    self.logger.warning(
                        f"[AblationVerification] WARN: 纯指纹 Rank-1 = {fp_only_rank1:.4f} 异常高"
                    )

                self.model.set_ablation(orig_ablate_modality)
                self.logger.info("[AblationVerification] 已恢复原始 gate 状态")
            elif self.ablate_modality == 'face':
                self.logger.info("[AblationVerification:fp_ablation] 禁用人脸，仅用指纹特征检索...")
                orig_ablate_modality = self.model._ablate_modality
                self.model.set_ablation(None)

                fp_zeros = torch.zeros_like(raw_fp)
                with torch.no_grad():
                    fused = self.model.extract_fused_features(raw_face, fp_zeros)
                    face_only_query_emb = F.normalize(fused, p=2, dim=1)

                face_only_sim = torch.mm(face_only_query_emb, gallery_embeddings.to(face_only_query_emb.device).t())
                _, face_only_top1 = torch.topk(face_only_sim, k=1, dim=1)
                face_only_top1_labels = gallery_labels_arr[face_only_top1.cpu().numpy()]
                face_only_correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] == face_only_top1_labels[i, 0]
                )
                face_only_rank1 = face_only_correct / len(query_labels)
                self.logger.info(
                    f"[AblationVerification:fp_ablation] "
                    f"纯人脸 Rank-1 = {face_only_rank1:.4f}"
                )
                self.logger.info(
                    f"[AblationVerification:fp_ablation] "
                    f"融合 Rank-1 = {rank1_acc:.4f}"
                )
                if face_only_rank1 < 0.50:
                    self.logger.info("[AblationVerification] PASS: 纯人脸 Rank-1 < 50%")
                else:
                    self.logger.warning(
                        f"[AblationVerification] WARN: 纯人脸 Rank-1 = {face_only_rank1:.4f} 异常高"
                    )

                self.model.set_ablation(orig_ablate_modality)
                self.logger.info("[AblationVerification] 已恢复原始 gate 状态")

            self.logger.info("=" * 60)

        # ── 步骤 6：汇总指标 ─────────────────────────────────────
        metrics = {
            "rank_1": rank1_acc,
            "rank_5": rank5_acc,
            "rank_10": rank10_acc,
            "rank_20": rank20_acc,
            "eer": eer,
            "feature_norm_gallery": feat_norm_g_meter.avg,
            "feature_norm_query": feat_norm_q_meter.avg,
            "query_count": len(query_labels),
            "gallery_count": len(gallery_paths),
            "gallery_persons": int(len(np.unique(gallery_labels_arr))),
            "query_persons": int(len(np.unique(query_labels))),
        }

        # 统一日志格式
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log_val_epoch(
            epoch=epoch + 1,
            total_epochs=total_epochs,
            lr=current_lr,
            loss=loss_meter.avg,
            rank1=rank1_acc,
            eer=eer if not np.isnan(eer) else None,
            gallery_size=len(gallery_paths),
            query_size=len(query_labels)
        )

        # 详细指标日志
        self.logger.info(
            f"[Metrics] Rank-1: {rank1_acc:.4f} | Rank-5: {rank5_acc:.4f} | "
            f"Rank-10: {rank10_acc:.4f} | Rank-20: {rank20_acc:.4f} | EER: {eer:.4f}"
        )

        # 打印样本匹配详情（按人均匀抽取前 5 个不同人的样本）
        unique_persons = np.unique(query_labels)
        selected_persons = unique_persons[:min(5, len(unique_persons))]
        sample_indices = [np.where(query_labels == p)[0][0] for p in selected_persons]
        self.logger.info("[验证样本] Query vs Top-3 Gallery 匹配（按人均匀抽取前 5 人）:")
        for idx in sample_indices:
            true_label = query_labels[idx]
            top3_pred_labels = top_k_labels[idx, :3]
            top3_sims = similarity_matrix[idx, top_k_indices[idx, :3]].numpy()
            match_str = "[O]" if true_label == top3_pred_labels[0] else "[X]"
            top3_str = ", ".join(
                f"{l}({s:.3f})" for l, s in zip(top3_pred_labels, top3_sims)
            )
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        if self.tb_writer:
            self.tb_writer.add_scalar('val/rank_1', rank1_acc, epoch)
            self.tb_writer.add_scalar('val/rank_5', rank5_acc, epoch)
            self.tb_writer.add_scalar('val/rank_10', rank10_acc, epoch)
            self.tb_writer.add_scalar('val/rank_20', rank20_acc, epoch)
            self.tb_writer.add_scalar('val/eer', eer, epoch)
            self.tb_writer.add_scalar('val/feature_norm_query', feat_norm_q_meter.avg, epoch)

        return loss_meter.avg, rank1_acc, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 测试集评估
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_epoch(self, epoch=None, total_epochs=None, use_amp=False):
        """在测试集上进行评估（与 validate_epoch 逻辑相同，但使用测试集）。"""
        self.model.eval()

        # ── 打印融合权重诊断（使用 Query 循环中 forward 缓存的 attention 权重）────
        self._log_fusion_weights("Test", with_attention=True)

        test_dataset = getattr(self, 'test_dataset', None)
        if test_dataset is None:
            self.logger.warning("[测试] 测试集未设置，跳过测试评估。")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        if not hasattr(test_dataset, 'test_gallery_paths') or not test_dataset.test_gallery_paths:
            self.logger.warning("[测试] 测试集不存在，跳过测试评估。")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        gallery_paths = test_dataset.test_gallery_paths
        gallery_labels_arr = np.array(test_dataset.test_gallery_labels)
        batch_size = self.val_loader.batch_size

        # ── Gallery ──────────────────────────────────────────────────────
        gallery_embeddings_list = []
        feat_norm_g_meter = AverageMeter()

        self.logger.info("[Test Gallery] 正在提取特征...")
        n_gallery_batches = (len(gallery_paths) + batch_size - 1) // batch_size
        pbar_g = tqdm(total=n_gallery_batches, desc="Test Gallery", leave=False)
        for start_idx in range(0, len(gallery_paths), batch_size):
            end = min(start_idx + batch_size, len(gallery_paths))
            batch_pairs = gallery_paths[start_idx:end]
            face_batch, fp_batch = self._load_gallery_batch(batch_pairs, batch_size)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)
            else:
                face_features, fp_features = self._extract_features_eval(face_batch, fp_batch)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_g_meter.update(feat_norm, fused_features.size(0))
            gallery_embeddings_list.append(fused_features.cpu())
            pbar_g.update()
        pbar_g.close()

        gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
        self.logger.info(
            f"[Test Gallery] {len(gallery_paths)} 对（测试集 {len(np.unique(gallery_labels_arr))} 人）"
        )

        # ── Query ─────────────────────────────────────────────────────────
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )

        query_embeddings_list = []
        query_labels_list = []
        feat_norm_q_meter = AverageMeter()

        self.logger.info("[测试] 正在提取 Query 特征...")
        pbar_q = tqdm(test_loader, desc="Test Query", leave=False)
        for batch in pbar_q:
            face_images = batch['face_image'].to(self.device)
            fp_images = batch['fingerprint_image'].to(self.device)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    face_features, fp_features = self._extract_features_eval(face_images, fp_images)
            else:
                face_features, fp_features = self._extract_features_eval(face_images, fp_images)

            fused_features = self.model.extract_fused_features(face_features, fp_features)
            fused_features_q = F.normalize(fused_features, p=2, dim=1)

            feat_norm = fused_features_q.norm(dim=1).mean().item()
            if not np.isnan(feat_norm):
                feat_norm_q_meter.update(feat_norm, face_images.size(0))

            query_embeddings_list.append(fused_features_q.cpu())
            query_labels_list.extend(batch['label'].tolist())

        if not query_embeddings_list:
            self.logger.error("[测试] 没有 Query 样本！")
            return {"rank_1": None, "rank_5": None, "rank_10": None, "rank_20": None, "eer": None}

        query_embeddings = torch.cat(query_embeddings_list, dim=0)
        query_labels = np.array(query_labels_list)

        self.logger.info(
            f"[测试] Query: {len(query_labels)} 样本, "
            f"{len(np.unique(query_labels))} 个测试人"
        )

        # ── 相似度 & 指标 ────────────────────────────────────────────────
        self.logger.info("[测试] 正在计算相似度矩阵...")
        similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

        top_k = min(20, gallery_embeddings.size(0))
        _, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        top_k_indices = top_k_indices.numpy()
        top_k_labels = gallery_labels_arr[top_k_indices]

        rank_metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= top_k:
                correct = sum(
                    1 for i in range(len(query_labels))
                    if query_labels[i] in top_k_labels[i, :k]
                )
                rank_metrics[f"rank_{k}"] = correct / len(query_labels)

        rank1_acc = rank_metrics.get("rank_1", 0.0)
        rank5_acc = rank_metrics.get("rank_5", 0.0)
        rank10_acc = rank_metrics.get("rank_10", 0.0)
        rank20_acc = rank_metrics.get("rank_20", 0.0)

        eer = 0.0
        try:
            positive_scores, negative_scores = [], []
            rng = np.random.RandomState(self.seed)
            for q_idx in range(len(query_labels)):
                q_label = query_labels[q_idx]
                q_emb = query_embeddings[q_idx]
                same_idx = np.where(gallery_labels_arr == q_label)[0]
                if len(same_idx) > 0:
                    sims = (q_emb @ gallery_embeddings[same_idx].t()).numpy()
                    topk_k = min(5, len(sims))
                    positive_scores.extend(np.sort(sims)[-topk_k:].tolist())
                diff_idx = np.where(gallery_labels_arr != q_label)[0]
                if len(diff_idx) > 0:
                    n_neg = min(3, len(diff_idx))
                    selected = rng.choice(diff_idx, n_neg, replace=False)
                    sims = (q_emb @ gallery_embeddings[selected].t()).numpy()
                    negative_scores.extend(sims.tolist())

            n_pos, n_neg = len(positive_scores), len(negative_scores)
            if n_pos >= 50 and n_neg >= 50:
                scores = np.array(positive_scores + negative_scores)
                eer_labels = np.array([1] * n_pos + [0] * n_neg)
                eer, eer_th = self.calculate_eer(eer_labels, scores)
                self.logger.info(
                    f"[Test EER] 正样本={n_pos}, 负样本={n_neg}, "
                    f"EER={eer:.4f} (阈值={eer_th:.4f})"
                )
        except Exception as e:
            self.logger.warning(f"[Test EER计算] 失败: {e}")

        # ── 测试结果汇总 ────────────────────────────────────────────────
        metrics = {
            "rank_1": rank1_acc, "rank_5": rank5_acc,
            "rank_10": rank10_acc, "rank_20": rank20_acc,
            "eer": eer, "feature_norm_gallery": feat_norm_g_meter.avg,
            "feature_norm_query": feat_norm_q_meter.avg,
            "query_count": len(query_labels),
            "gallery_count": len(gallery_paths),
            "gallery_persons": int(len(np.unique(gallery_labels_arr))),
            "query_persons": int(len(np.unique(query_labels))),
        }

        self.logger.info("=" * 60)
        self.logger.info("【测试集评估】")
        self.logger.info(f"  Rank-1:  {rank1_acc:.4f} ({int(rank1_acc * len(query_labels))}/{len(query_labels)})")
        self.logger.info(f"  Rank-5:  {rank5_acc:.4f}")
        self.logger.info(f"  Rank-10: {rank10_acc:.4f}")
        self.logger.info(f"  Rank-20: {rank20_acc:.4f}")
        self.logger.info(f"  EER:     {eer:.4f}" if eer > 0 else "  EER:     N/A")
        self.logger.info(f"  Gallery: {len(gallery_paths)} 对 / {len(np.unique(gallery_labels_arr))} 人")
        self.logger.info(f"  Query:   {len(query_labels)} 样本 / {len(np.unique(query_labels))} 人")
        self.logger.info("=" * 60)

        # 样本匹配详情（按人均匀抽取）
        unique_persons = np.unique(query_labels)
        selected_persons = unique_persons[:min(5, len(unique_persons))]
        sample_indices = [np.where(query_labels == p)[0][0] for p in selected_persons]
        self.logger.info("[测试样本] Query vs Top-3 Gallery 匹配（按人均匀抽取前 5 人）:")
        for idx in sample_indices:
            true_label = query_labels[idx]
            top3_pred_labels = top_k_labels[idx, :3]
            top3_sims = similarity_matrix[idx, top_k_indices[idx, :3]].numpy()
            match_str = "[O]" if true_label == top3_pred_labels[0] else "[X]"
            top3_str = ", ".join(f"{l}({s:.3f})" for l, s in zip(top3_pred_labels, top3_sims))
            self.logger.info(f"  Query同人={true_label}, Top3预测=[{top3_str}] {match_str}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # 保存检查点
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch=None, is_best=False, extra=None):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "balance_optimizer_state": (
                self._balance_optimizer.state_dict() if self._balance_optimizer else None
            ),
            "balance_scheduler_state": (
                self._balance_scheduler.state_dict() if self._balance_scheduler else None
            ),
            "epoch": epoch or 0,
        }
        if self.face_model:
            state["face_model_state"] = self.face_model.state_dict()
        if self.fingerprint_model:
            state["fp_model_state"] = self.fingerprint_model.state_dict()
        if extra:
            state.update(extra)

        torch.save(state, path)
        self.logger.info(f"[保存] Checkpoint: {path}" if not is_best else f"[保存] 最佳模型: {path}")
        return path

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if checkpoint.get('face_model_state') and self.face_model:
            self.face_model.load_state_dict(checkpoint['face_model_state'])
        if checkpoint.get('fp_model_state') and self.fingerprint_model:
            self.fingerprint_model.load_state_dict(checkpoint['fp_model_state'])
        if checkpoint.get('optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        if self._balance_optimizer and checkpoint.get('balance_optimizer_state'):
            self._balance_optimizer.load_state_dict(checkpoint['balance_optimizer_state'])
        if self._balance_scheduler and checkpoint.get('balance_scheduler_state'):
            self._balance_scheduler.load_state_dict(checkpoint['balance_scheduler_state'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"[Load] Checkpoint: {path}")
        return checkpoint
