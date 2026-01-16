# 基于人脸和指纹的多模态身份识别算法研究

## 项目简介

本项目实现了一个基于深度学习的双模态生物特征识别系统，结合人脸和指纹特征进行身份识别验证。该项目专为毕业论文设计，采用模块化架构，便于学术研究和实验对比。

## 项目特性

- **双模态融合**: 结合人脸和指纹两种生物特征
- **模块化设计**: 清晰的代码结构，便于维护和扩展
- **学术规范**: 支持消融实验和性能指标计算
- **可配置化**: 所有参数通过配置文件管理

## 目录结构

```
├── data/                    # 数据集目录（不进入版本控制）
│   ├── face/               # 人脸图像
│   └── fingerprint/        # 指纹图像
├── checkpoints/            # 模型权重保存
├── configs/               # 配置文件
├── core/                  # 核心代码
│   ├── models/            # 模型定义
│   ├── datasets.py        # 数据加载
│   ├── loss.py           # 损失函数
│   └── utils.py          # 工具函数
├── scripts/               # 脚本工具
├── logs/                  # 训练日志
├── train.py              # 训练入口
├── test.py               # 测试入口
└── requirements.txt      # 依赖包
```

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据准备

将人脸和指纹图像分别放入`data/face/`和`data/fingerprint/`目录。

### 训练模型

```bash
python train.py
```

### 测试模型

```bash
python test.py
```

## 配置说明

项目通过`configs/config.yaml`文件管理所有参数，包括：
- 训练超参数（学习率、批次大小等）
- 模型架构参数
- 数据处理配置
- 路径配置

## 学术指标

项目支持计算以下身份识别标准指标：
- Top-1/Top-5 准确率
- EER (Equal Error Rate)
- ROC曲线
- FAR/FRR (False Acceptance/Rejection Rate)

## 消融实验

支持以下实验设置：
- 仅人脸识别
- 仅指纹识别
- 双模态融合

## 许可证

本项目仅用于学术研究目的。