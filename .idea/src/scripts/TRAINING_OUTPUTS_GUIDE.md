# 训练输出文件说明

运行 `train_face.py` 后会生成以下文件结构：

```
scripts/
├── 📁 checkpoints/           # 模型权重文件
│   └── face_recognition/
│       ├── best_model.pth          # ⭐ 最佳模型（准确率最高）
│       ├── last_model.pth         # 最后一次保存的模型
│       └── epoch_10_model.pth     # 第10轮的模型
│
└── 📁 visualization_results/  # 可视化结果
    └── face_recognition/
        ├── accuracy_curve.png      # 准确率曲线
        ├── loss_curve.png          # 损失曲线
        └── confusion_matrix.png     # 混淆矩阵
```

---

## 📁 文件作用详解

### 1. checkpoints/ 模型权重（最重要！）

| 文件 | 作用 | 何时使用 |
|------|------|---------|
| `best_model.pth` | 验证集准确率最高的模型 | ⭐ 正式使用时加载这个 |
| `last_model.pth` | 最后一个epoch的模型 | 恢复训练时用 |
| `epoch_XX_model.pth` | 某一轮的模型 | 实验分析用 |

**pth文件里保存了什么：**
```python
{
    'epoch': 10,                    # 训练到第10轮
    'model_state_dict': {...},      # 模型参数（权重）
    'optimizer_state_dict': {...}, # 优化器状态
    'accuracy': 0.98,              # 验证集准确率
    'loss': 0.05,                  # 验证集损失
    'class_to_idx': {...}           # 类别映射表
}
```

---

### 2. visualization_results/ 可视化图表

| 文件 | 作用 |
|------|------|
| `accuracy_curve.png` | 📈 准确率变化曲线（训练 vs 验证） |
| `loss_curve.png` | 📉 损失变化曲线 |
| `confusion_matrix.png` | 🔢 混淆矩阵（看哪些类别容易分错） |

---

## 🔄 文件生命周期

```
开始训练
    │
    ▼
生成 .log  ──► 查看训练过程
    │
    ▼
每 epoch 保存 ◄── 断点续训
    │
    ▼
发现 best.pth  ◄── ⭐ 保存最佳模型
    │
    ▼
训练完成
    │
    ▼
可视化图表 ──► 分析结果
```

---

## 📌 使用建议

**1. 训练时：** 关注 `logs/training.log` 看实时进度

**2. 训练后：**
- 用 `checkpoints/face_recognition/best_model.pth` 做测试/推理
- 看 `visualization_results/` 的图表分析是否过拟合

**3. 想复现实验：**
- 记录 `experiment_name` 和 `best_model.pth` 路径
- config文件记录所有参数

---

## ❓ 常见问题

**Q: 可以删除旧文件吗？**
A: 可以删除 `visualization_results/`，但 `checkpoints/` 建议保留。`logs/` 为训练临时产物，重新训练即可生成。

**Q: pth文件能直接打开看吗？**
A: 需要用Python加载：
```python
import torch
ckpt = torch.load('best_model.pth')
print(ckpt.keys())
```

**Q: 训练中断能继续吗？**
A: 可以！加载 `last_model.pth` 的 `model_state_dict` 和 `optimizer_state_dict`
