# 多模态融合识别实验计划

> 本文档定义完整实验流程，涵盖数据准备、实验矩阵、训练、评估与结果管理。
> 制定日期：2026-04-29

---

## 目录

1. [实验目标](#1-实验目标)
2. [数据准备](#2-数据准备)
3. [实验矩阵](#3-实验矩阵)
4. [训练流程](#4-训练流程)
5. [评估指标](#5-评估指标)
6. [结果管理](#6-结果管理)
7. [执行时间表](#7-执行时间表)

---

## 1. 实验目标

### 1.1 核心目标

验证人脸+指纹多模态融合识别相比单模态的有效性，通过消融实验量化各模态贡献。

### 1.2 研究问题

| 编号 | 问题 | 对应实验 |
|------|------|---------|
| RQ1 | 融合识别是否优于单模态？ | `fusion_full` vs `face_only` / `fp_only` |
| RQ2 | 各模态对融合的贡献比例？ | `face_only` / `fp_only` 的绝对性能 |
| RQ3 | 冻结 backbone 再训融合层是否可行？ | `fusion_only` vs `fusion_full` |
| RQ4 | 不同融合策略（simple/adpative/gated/hierarchical）哪个更好？ | `fusion_simple` / `fusion_adaptive` / ... |

---

## 2. 数据准备

### 2.1 数据集信息

| 模态 | 数据集 | 人数 | 来源配置 |
|------|--------|------|---------|
| 人脸 | CASIA-WebFace（子集） | 500 人 | `configs/face_config.yaml` |
| 指纹 | CASIA-FingerprintV5 | 500 人 | `configs/fingerprint_config.yaml` |
| 映射 | `data/face_casia_mapping.json` | 500 对 | `scripts/generate_casia_mapping.py` |

### 2.2 数据划分（Person-wise）

> **关键原则**：同一人的数据不会同时出现在训练集和测试集，防止信息泄漏。

| 集合 | 比例 | 人数 | 用途 |
|------|------|------|------|
| 训练集 | 70% | 350 人 | 模型参数学习 |
| 验证集 | 10% | 50 人 | 超参数调优、早停 |
| 测试集 | 20% | 100 人 | 最终指标报告 |

```
总计 500 人
├── Train (350人): images/*/train/  + fingerprints/*/
├── Val   (50人): images/*/val/    + fingerprints/*/
└── Test  (100人): images/*/test/   + fingerprints/*/
```

**Gallery 和 Query 划分**（测试集内部）：

- **Gallery**：每人随机选 3 张图片（`configs/fusion_config.yaml` → `gallery_per_person: 3`）
- **Query**：Gallery 外的剩余图片

### 2.3 随机种子

所有数据划分使用统一随机种子：`seed = 42`

```bash
# 验证数据划分无重叠
python scripts/check_dataset.py --check_splits
```

### 2.4 数据检查清单

- [ ] 人脸和指纹数据集目录存在且可读
- [ ] `face_casia_mapping.json` 完整（500 人均有映射）
- [ ] 每人至少有人脸图片和指纹图片各 1 张
- [ ] 训练/验证/测试集人员无交集
- [ ] 测试集 Gallery + Query 覆盖全部测试人员

---

## 3. 实验矩阵

### 3.1 两阶段训练

```
阶段一：单模态预训练（可选但推荐）
    └── 目的：得到高质量的单模态 backbone，为融合提供更好的初始化

阶段二：融合训练（全部实验）
    └── 目的：训练融合层 + 验证消融假设
```

### 3.2 阶段一：单模态预训练

| 实验编号 | 模态 | 实验名称 | 输出 checkpoint |
|---------|------|---------|--------------|
| S1-FACE | Face | `face_baseline` | `checkpoints/face/best_face.pth` |
| S1-FP | Fingerprint | `fp_baseline` | `checkpoints/fingerprint/best_fp.pth` |

**训练命令**：

```bash
# 人脸预训练
python scripts/train_face.py --config configs/face_config.yaml --experiment_name face_baseline

# 指纹预训练
python scripts/train_fingerprint.py --config configs/fingerprint_config.yaml --experiment_name fp_baseline
```

### 3.3 阶段二：融合实验

| 实验编号 | 实验名称 | 实验模式 | 消融模态 | 训练参数 | 对应研究问题 |
|---------|---------|---------|---------|---------|------------|
| F0 | `fusion_full_simple` | `full` | 无 | backbone + fusion 联合训练 | RQ1（基线） |
| F1 | `fusion_face_only` | `face_ablation` | `fingerprint` | backbone + fusion 联合训练 | RQ1, RQ2 |
| F2 | `fusion_fp_only` | `fp_ablation` | `face` | backbone + fusion 联合训练 | RQ1, RQ2 |
| F3 | `fusion_frozen_simple` | `fusion_only` | 无 | 冻结 backbone，只训融合层 | RQ3 |
| F4 | `fusion_full_adaptive` | `full` | 无 | backbone + fusion，策略=adaptive | RQ4 |
| F5 | `fusion_full_gated` | `full` | 无 | backbone + fusion，策略=gated | RQ4 |
| F6 | `fusion_full_hierarchical` | `full` | 无 | backbone + fusion，策略=hierarchical | RQ4 |

> **注意**：`fusion_frozen_simple`（F3）需要传入单模态预训练权重：
> ```bash
> --face_ckpt checkpoints/face/best_face.pth
> --fp_ckpt checkpoints/fingerprint/best_fp.pth
> ```

**训练命令示例**：

```bash
# 基线融合
python scripts/train_fusion.py \
    --experiment_name fusion_full_simple \
    --experiment_mode full \
    --fusion_method simple

# 消融：单用人脸
python scripts/train_fusion.py \
    --experiment_name fusion_face_only \
    --experiment_mode face_ablation \
    --fusion_method simple

# 冻结 backbone 融合
python scripts/train_fusion.py \
    --experiment_name fusion_frozen_simple \
    --experiment_mode fusion_only \
    --face_ckpt checkpoints/face/best_face.pth \
    --fp_ckpt checkpoints/fingerprint/best_fp.pth \
    --fusion_method simple
```

### 3.4 实验输出目录结构

所有实验结果统一归入 `scripts/experiments/`：

```
scripts/experiments/
├── S1_face_baseline/           # 阶段一：人脸预训练
│   ├── checkpoints/best_face.pth
│   ├── logs/training.log
│   └── history.json
├── S1_fp_baseline/             # 阶段一：指纹预训练
│   └── ...
├── F0_fusion_full_simple/      # 阶段二：基线融合
│   ├── checkpoints/best_simple.pth
│   ├── logs/training.log
│   └── history.json
├── F1_fusion_face_only/
│   └── ...
├── F2_fusion_fp_only/
│   └── ...
├── F3_fusion_frozen_simple/
│   └── ...
├── F4_fusion_full_adaptive/
│   └── ...
├── F5_fusion_full_gated/
│   └── ...
├── F6_fusion_full_hierarchical/
│   └── ...
└── results/                    # 自动生成：汇总对比表
    ├── YYYYMMDD_summary.json   # 所有实验指标汇总
    ├── comparison_table.png     # 横向对比图
    ├── comparison_table.csv     # 数值对比表
    ├── roc_comparison.png       # ROC 曲线对比
    └── eer_comparison.png      # EER 柱状图对比
```

---

## 4. 训练流程

### 4.1 统一配置

`configs/fusion_config.yaml` 中的关键参数：

```yaml
training:
  epochs: 50              # 根据验证集收敛情况可调整
  batch_size: 16
  accumulation_steps: 2   # 实际 batch = 16 × 2 = 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  use_amp: true           # 混合精度
  label_smoothing: 0.1

model:
  num_classes: 500        # 与数据集人数一致
  face_embedding_dim: 512
  fingerprint_embedding_dim: 512
  fusion_dim: 256
  use_arcface: true      # 启用度量学习
  arc_s: 64.0
  arc_m: 0.5

data:
  split_ratio: 0.8       # 80% 训练
  test_split_ratio: 0.5  # 剩余 20% 中 50% 验证 50% 测试
  gallery_per_person: 3   # Gallery 每人身图片数

misc:
  seed: 42               # 固定种子
  early_stopping_patience: 15
```

### 4.2 早停与模型保存

- **早停**：验证集 Rank-1 连续 15 个 epoch 无提升则停止
- **最佳模型**：以验证集 Rank-1 准确率为唯一标准保存 `best_*.pth`
- **训练历史**：每个 epoch 记录 `history.json`

### 4.3 训练监控

训练过程中关注以下指标：

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| Train Loss | 单调下降 | NaN、震荡 |
| Val Loss | 先降后升（过拟合） | 一直上升 |
| Val Rank-1 | 最终 > 85% 为良好 | 低于 70% 需检查 |
| Feature Norm | 约等于 1.0（L2 归一化） | 偏离 1.0 太多 |
| LR | Cosine 衰减 | 提前降到 0 |

---

## 5. 评估指标

### 5.1 主要指标

| 指标 | 含义 | 评估位置 |
|------|------|---------|
| **Rank-1 Accuracy** | Top-1 匹配率（最重要） | 验证集 + 测试集 |
| **Rank-5/10/20 Accuracy** | Top-K 召回率 | 验证集 |
| **EER (Equal Error Rate)** | FAR = FRR 的交叉点，越低越好 | 测试集 |
| **AUC (Area Under ROC)** | ROC 曲线下面积，越高越好 | 测试集 |
| **FAR@0.1%** | FAR=0.1% 时的 FRR | 测试集 |

### 5.2 评估命令

```bash
# 评估单个实验
python scripts/evaluate.py \
    --model_type fusion \
    --checkpoint_path scripts/experiments/F0_fusion_full_simple/checkpoints/best_simple.pth \
    --experiment_name fusion_full_simple

# 批量生成对比图（实验完成后运行）
python scripts/experiments/visualization_manager.py --mode comparison
```

### 5.3 消融分析

实验完成后，对比以下数据：

```
融合 vs 单人脸：
    ΔRank-1 = fusion_full - face_only
    ΔEER = eer_fusion - eer_face  （应为负值）

融合 vs 单指纹：
    ΔRank-1 = fusion_full - fp_only
    ΔEER = eer_fusion - eer_fp  （应为负值）
```

ΔRank-1 > 0 且 ΔEER < 0 → 融合有效

---

## 6. 结果管理

### 6.1 每次实验自动记录

每轮训练结束后，`history.json` 包含：

```json
{
  "experiment": "fusion_full_simple",
  "experiment_mode": "full",
  "fusion_method": "simple",
  "start_time": "2026-04-29 10:00:00",
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.35,
      "train_acc": 0.12,
      "val_loss": 1.89,
      "val_rank1": 0.34,
      "val_rank5": 0.61,
      "val_rank10": 0.73,
      "val_rank20": 0.84,
      "val_eer": 0.31,
      "lr": 0.0001
    }
  ]
}
```

### 6.2 最终汇总表

每次实验完成后，运行 `visualization_manager.py` 自动生成：

```json
// results/YYYYMMDD_summary.json
{
  "generated_at": "2026-04-29 22:00:00",
  "experiments": {
    "F0_fusion_full_simple": {
      "test_rank1": 0.912,
      "test_rank5": 0.961,
      "test_eer": 0.048,
      "test_auc": 0.995,
      "best_epoch": 38
    },
    "F1_fusion_face_only": {
      "test_rank1": 0.873,
      "test_rank5": 0.941,
      "test_eer": 0.072,
      "test_auc": 0.989
    }
  }
}
```

---

## 7. 执行时间表

### 阶段一：单模态预训练（预计 2-4 小时）

```
[Day 1]
├─ 10:00  检查数据集完整性
│         python scripts/check_dataset.py --check_splits
├─ 10:30  训练人脸模型
│         python scripts/train_face.py --experiment_name face_baseline
└─ 12:30  训练指纹模型
          python scripts/train_fingerprint.py --experiment_name fp_baseline
```

### 阶段二：融合实验（预计 8-16 小时）

```
[Day 1 下午 - Day 2]
├─ F0 fusion_full_simple      ~2小时
├─ F1 fusion_face_only       ~2小时
├─ F2 fusion_fp_only         ~2小时
├─ F3 fusion_frozen_simple  ~2小时  (需传入 --face_ckpt --fp_ckpt)
├─ F4 fusion_full_adaptive   ~2小时
├─ F5 fusion_full_gated      ~2小时
└─ F6 fusion_full_hierarchical ~2小时
```

### 阶段三：评估与对比（预计 1-2 小时）

```
[Day 3]
├─ 批量评估所有实验
│   python scripts/evaluate.py ...  (每个实验跑一次)
├─ 生成横向对比
│   python scripts/experiments/visualization_manager.py --mode comparison
└─ 整理最终结果
```

---

## 附录：常见问题

### Q1: 显存不够怎么办？

```yaml
# fusion_config.yaml 调低
training:
  batch_size: 8        # 从 16 降到 8
  accumulation_steps: 4  # 累积 4 步补偿
```

### Q2: 模型不收敛怎么办？

1. 检查数据映射是否正确（有无 NaN/异常图片）
2. 降低学习率：`--lr 5e-5`
3. 减小 ArcFace margin：`arc_m: 0.3`
4. 确认 `gallery_per_person` 不要太小（< 2）

### Q3: 消融实验发现单模态效果比融合好？

检查：
1. 消融模态是否真的被置零（看日志中 `[Ablation]` 记录）
2. 融合权重是否偏向消融模态
3. 数据质量是否两个模态均衡

### Q4: 如何复现已有实验？

```bash
# 完全相同配置重新训练
python scripts/train_fusion.py \
    --experiment_name fusion_full_simple_v2 \
    --experiment_mode full \
    --fusion_method simple
    # --resume scripts/experiments/F0_fusion_full_simple/checkpoints/best_simple.pth  # 从上次断点继续
```
