# 多模态特征融合实验流程 (300人版本)

## 实验前提
✅ 人脸识别准确率: 98%
✅ 指纹识别准确率: 80%
✅ 数据集: 300人 (人脸000-299, 指纹001-300)

---

## 步骤1: 重新绑定数据（300人）

```bash
cd scripts
python bind_fusion_data.py
```

输出:
- `data/face_fingerprint_mapping.json` - 新的映射文件
- `data/face_fingerprint_stats.json` - 统计信息
- `data/face_fingerprint_mapping.csv` - CSV格式映射表

预期结果:
```
绑定完成!
  总人数: 300
  每人脸图片数: 5
  指纹图片数: 10
```

---

## 步骤2: 训练融合模型

### 基础融合实验 (Concat)
```bash
cd scripts
python train_fusion.py \
    --experiment_name fusion_concat_300 \
    --fusion_method concat
```

### 注意力融合实验
```bash
python train_fusion.py \
    --experiment_name fusion_attention_300 \
    --fusion_method attention_fusion
```

### 跨模态注意力实验
```bash
python train_fusion.py \
    --experiment_name fusion_cross_attn_300 \
    --fusion_method cross_attention
```

### 张量融合实验
```bash
python train_fusion.py \
    --experiment_name fusion_tensor_300 \
    --fusion_method tensor_fusion
```

---

## 步骤3: 加载预训练单模态模型

如果你已经有人脸和指纹的预训练权重，可以使用 `--face_ckpt` 和 `--fp_ckpt` 参数:

```bash
python train_fusion.py \
    --experiment_name fusion_pretrained_300 \
    --fusion_method concat \
    --face_ckpt ../checkpoints/face/best_model.pth \
    --fp_ckpt ../checkpoints/fingerprint/best_model.pth
```

---

## 步骤4: 模态消融实验

修改 `configs/fusion_config.yaml`:
```yaml
experiment:
  modality_ablation: true
  face_only: true   # 只用人脸
  fingerprint_only: true  # 只用指纹
```

运行:
```bash
# 只用人脸
python train_fusion.py --experiment_name face_only_300

# 只用指纹
python train_fusion.py --experiment_name fp_only_300
```

---

## 实验设计对比

| 实验名称 | 融合方法 | 预训练权重 | 目的 |
|---------|---------|-----------|------|
| baseline_concat | concat | ImageNet | 基准性能 |
| baseline_attention | attention_fusion | ImageNet | 对比注意力机制 |
| pretrained_concat | concat | 单模态模型 | 预训练效果 |
| pretrained_attention | attention_fusion | 单模态模型 | 预训练+注意力 |
| face_only | - | 单模态模型 | 人脸单独性能 |
| fingerprint_only | - | 单模态模型 | 指纹单独性能 |

---

## 预期性能提升

| 方法 | 预期准确率 | 说明 |
|-----|----------|------|
| 人脸单独 | ~98% | 上限参考 |
| 指纹单独 | ~80% | 上限参考 |
| 简单拼接 | 95-98% | 可能略低于人脸 |
| 注意力融合 | 97-99% | 模态自适应加权 |
| 跨模态注意力 | 97-99% | 模态间交互 |

---

## 结果分析

实验结果将保存在:
- `scripts/logs/fusion_{experiment_name}/`
- `scripts/visualization_results/`

关键指标:
1. 融合准确率 vs 单模态准确率
2. 不同融合方法的对比
3. 模态消融实验（人脸/指纹贡献度）
