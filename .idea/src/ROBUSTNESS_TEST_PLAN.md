# 融合模型干扰鲁棒性测试计划

> 文档版本: v1.0
> 创建日期: 2026-05-20
> 目标: 对 `fusion_adaptive` 模型进行干扰鲁棒性评估，使用与训练时完全相同的 test split，保证结果可比性。

---

## 1. 背景与动机

### 1.1 问题描述

当前增强干扰测试存在以下问题，导致结果不可比：

| 问题 | 说明 |
|------|------|
| **数据集不一致** | 之前测试用的是 `FusionDataset` 的 val split（150 gallery / 100 query），而训练日志中 97.6% 是 test split（250 query）上的结果，两者不是同一批数据 |
| **样本量差异** | val split 仅 100 query 样本，随机波动大（±3%），无法可靠反映模型真实鲁棒性 |
| **环境不一致** | 训练时的 test split 走的是 `fusion_trainer.evaluate_on_test()`，用的是干净的 `test_dataset`；而干扰测试是独立脚本，走的是另一套 pipeline，CLAHE 对称性难以保证 |

### 1.2 解决思路

> **核心原则**: 用训练时一模一样的 test split，只改变 Query 端的预处理策略（干净 vs 增强），其余所有条件完全固定。

---

## 2. 数据集规格（来自训练配置）

来源: `.idea/src/configs/fusion_config.yaml`

```
split_ratio: 0.8          # 80% 人员训练，20% 剩余人员
test_split_ratio: 0.5     # 剩余人员中 50% 验证 / 50% 测试
最终划分: 70% 训练 / 10% 验证 / 20% 测试
gallery_per_person: 3     # 验证/测试集每人注册 3 张图片
总人数: 500 人
```

| 数据集 | Gallery 大小 | Query 大小 | 人员数 |
|--------|------------|-----------|--------|
| val (验证) | ~50人 × 3 = ~150 | ~50人 × 2 = ~100 | 50 人 |
| **test (测试)** | **~50人 × 3 = ~150** | **~50人 × 5 = ~250** | **50 人** |

> **注意**: 增强干扰测试统一使用 **test split**（250 query），而非 val split（100 query），以获得统计上更可靠的结果。

---

## 3. 测试方案设计

### 3.1 测试管线架构

```
                    ┌─────────────────────────────────────┐
                    │           test split                │
                    │   (gallery + query, 同一批人)        │
                    └──────────┬───────────────┬───────────┘
                               │               │
                    ┌──────────▼──┐    ┌──────▼──────────┐
                    │   Gallery   │    │     Query       │
                    │  (干净, 同  │    │  (实验变量)     │
                    │  训练一致)  │    │                 │
                    │             │    │  策略 A: 干净   │
                    │ Resize      │    │  策略 B: 适度增强│
                    │ CLAHE(FP)   │    │  策略 C: 强增强  │
                    │ ToTensor    │    │  ...            │
                    │ Normalize   │    │                 │
                    └──────┬──────┘    └──────┬──────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────────────────────────────┐
                    │       特征提取 (same model)          │
                    │   face_model + fp_model + fusion    │
                    └─────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────────────────────────────┐
                    │          1:N 检索评估               │
                    │   Rank-1/5/10, EER, FAR@0.1% FRR   │
                    └─────────────────────────────────────┘
```

### 3.2 两种测试入口

#### 入口一：离线脚本测试（优先实现）

独立脚本 `test_inference_robustness.py`，直接在 test split 上做干扰测试。

优势：调试方便，不依赖推理服务，可快速迭代增强策略。

#### 入口二：推理 API 测试（进阶）

通过 `inference/` 路径，用实际推理流程（Gallery 注册 → Query 检索）做干扰测试。

优势：验证实际部署场景下的表现，与训练日志完全对应。

---

### 3.3 实验配置矩阵

| 实验编号 | Gallery | Query | 说明 |
|---------|---------|-------|------|
| **基准 (Baseline)** | 干净 | 干净 | 等同于 `fusion_trainer.evaluate_on_test()`，用于对照 |
| **A1** | 干净 | 适度人脸增强（模糊+遮挡） | 仅人脸受干扰 |
| **A2** | 干净 | 适度指纹增强（模糊+轻微擦除） | 仅指纹受干扰 |
| **A3** | 干净 | 适度双模态增强 | 人脸+指纹同时受干扰 |
| **B1** | 干净 | 强人脸增强（旋转+模糊+遮挡+ColorJitter） | 人脸严重干扰 |
| **B2** | 干净 | 强指纹增强（旋转+模糊+擦除） | 指纹严重干扰 |
| **B3** | 干净 | 强双模态增强 | 双模态严重干扰 |

---

## 4. 增强策略参数

### 4.1 适度增强（Moderate）

对应实际场景：用户光线稍差 / 手指轻微污渍

| 模态 | 操作 | 参数 |
|------|------|------|
| 人脸 | 高斯模糊 | p=0.2, kernel=3 |
| | 随机遮挡 | p=0.1, scale=(0.05, 0.15), ratio=(0.3, 3.3) |
| 指纹 | 高斯模糊 | p=0.15, kernel=3 |
| | 随机擦除 | p=0.05, scale=(0.02, 0.08) |

### 4.2 强增强（Strong）

对应实际场景：用户拍摄角度大 / 手指干燥/潮湿

| 模态 | 操作 | 参数 |
|------|------|------|
| 人脸 | 高斯模糊 | p=0.4, kernel=5 |
| | 随机遮挡 | p=0.3, scale=(0.05, 0.2) |
| | 随机旋转 | ±10° |
| | ColorJitter | brightness=0.2, contrast=0.2 |
| | 水平翻转 | p=0.5 |
| 指纹 | 高斯模糊 | p=0.3, kernel=5 |
| | 随机擦除 | p=0.2, scale=(0.05, 0.15) |
| | 随机旋转 | ±8° |

### 4.3 CLAHE 对称性要求（关键）

所有 Gallery 和 Query 的指纹在 ToTensor 之前**必须应用 CLAHE**，与 `FusionDataset` 的 `_apply_clahe()` 和训练时的 `fusion_trainer.val_fp_transform` 保持一致。

```
指纹 Pipeline: Resize → CLAHE → ToTensor → Normalize
人脸 Pipeline: Resize → ToTensor → Normalize
```

---

## 5. 评估指标

| 指标 | 说明 |
|------|------|
| Rank-1 | 检索结果第一名正确的比例（主要指标） |
| Rank-5 | 前 5 名中有正确结果的比例 |
| Rank-10 | 前 10 名中有正确结果的比例 |
| EER | 等错误率（Equal Error Rate），越低越好 |
| FAR@0.1% FRR | FAR=0.1% 时的 FRR 值，安全场景关键指标 |

---

## 6. 预期基准（来自训练日志）

| 模型配置 | 数据集 | Rank-1 | EER |
|---------|--------|--------|-----|
| face 单模态 | test split | 94.00% | 3.50% |
| fingerprint 单模态 | test split | 96.86% | 2.28% |
| fusion (完整) | test split | **97.60%** | **3.53%** |
| fusion (人脸消融) | test split | 96.40% | 3.60% |
| fusion (指纹消融) | test split | 90.40% | 6.00% |

> 增强后各实验的 Rank-1 相对于基准的衰减幅度即为鲁棒性指标。

---

## 7. 输出要求

### 7.1 文件结构

```
outputs/
  robustness_test/
    test_<timestamp>/
      results.json          # 所有实验原始数据（JSON）
      comparison_table.png  # Rank-K 对比柱状图
      eer_comparison.png    # EER 对比图
      summary_report.txt    # 文字分析报告
```

### 7.2 结果格式（results.json）

```json
{
  "timestamp": "20260520_xxxxxx",
  "test_split_info": {
    "gallery_size": 150,
    "query_size": 250,
    "num_people": 50
  },
  "baseline": {
    "experiment": "baseline",
    "description": "Gallery干净, Query干净",
    "rank1": 0.9760,
    "rank5": 1.0000,
    "rank10": 1.0000,
    "eer": 0.0353,
    "far_001_frr": null
  },
  "experiments": [
    {
      "experiment": "A1_moderate_face_aug",
      "description": "Gallery干净, Query适度人脸增强",
      "augmentation": { "face": {...}, "fp": {} },
      "rank1": 0.xxxx,
      "rank5": 0.xxxx,
      ...
    }
  ],
  "delta_analysis": {
    "A1_vs_baseline": -0.0xxx,
    "A2_vs_baseline": -0.0xxx,
    ...
  }
}
```

---

## 8. 实现步骤

- [ ] **Step 1**: 新建 `test_inference_robustness.py` 脚本，复用 `FusionDataset` 的 test split
- [ ] **Step 2**: 实现 Gallery 干净 Transform（Resize → CLAHE → ToTensor → Normalize，与训练一致）
- [ ] **Step 3**: 实现适度增强 Query Transform（高斯模糊 + 随机遮挡）
- [ ] **Step 4**: 实现强增强 Query Transform（旋转 + 模糊 + 遮挡 + ColorJitter）
- [ ] **Step 5**: 实现评估函数（Gallery 特征 → Query 特征 → 1:N 检索 → 指标计算）
- [ ] **Step 6**: 运行基准实验（Gallery干净/Query干净），验证与训练日志一致（97.6%）
- [ ] **Step 7**: 运行所有增强实验（A1-A3, B1-B3）
- [ ] **Step 8**: 生成对比可视化图表和分析报告
- [ ] **Step 9** (进阶): 通过推理 API 路径验证（Gallery 注册 → Query 检索）

---

## 9. 关键约束

1. **Gallery 和 Query 必须用同一个 test split**，不可与 val split 混用
2. **Gallery Transform 必须与训练时的 val_fp_transform 完全一致**（Resize → CLAHE → ToTensor → Normalize）
3. **指纹 CLAHE 在增强之后仍然必须保留**（增强操作在 CLAHE 之后应用）
4. **所有实验使用同一随机种子**（seed=42），保证 Gallery 完全一致
5. **Gallery 永远不增强**，只有 Query 端可变

---

## 10. 风险与备选

| 风险 | 应对 |
|------|------|
| test split 样本量仍不够稳定 | 增加多随机种子取平均（如 seed=42, 2026, 7 个种子） |
| 增强 transform 与训练不一致 | 对比 `FusionDataset` 的 train transform，确认每一层 |
| GPU 显存不足（多实验） | 每个实验独立加载模型，不一次性缓存在显存中 |
