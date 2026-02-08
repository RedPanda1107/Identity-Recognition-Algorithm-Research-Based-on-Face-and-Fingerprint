# Scripts Directory Structure

## 核心训练脚本 (根目录)

| 文件 | 说明 |
|------|------|
| `train_face.py` | 人脸识别模型训练 |
| `train_fingerprint.py` | 指纹识别模型训练 |
| `train_fusion.py` | 多模态融合模型训练 |
| `evaluate.py` | 模型评估脚本 |
| `visualize.py` | 训练结果可视化 |

## 工具脚本 (`tools/`)

| 文件 | 说明 |
|------|------|
| `bind_face_fingerprint_ids.py` | 完整数据绑定（全部人） |
| `bind_fusion_data.py` | 融合实验数据绑定（限定300人） |
| `verify_binding.py` | 验证数据绑定结果 |
| `import_socofing.py` | 导入SOCOFing指纹数据集 |

## 子目录

| 目录 | 说明 |
|------|------|
| `checkpoints/` | 模型检查点 |
| `logs/` | 训练日志 |
| `visualization_results/` | 可视化结果 |

---

## 快速使用

### 1. 绑定融合数据（300人）
```bash
cd tools
python bind_fusion_data.py
python verify_binding.py
```

### 2. 训练融合模型
```bash
cd ..
python train_fusion.py --experiment_name fusion_test --fusion_method concat
```

### 3. 完整数据绑定（全部人）
```bash
cd tools
python bind_face_fingerprint_ids.py --mapping-strategy offset
```
