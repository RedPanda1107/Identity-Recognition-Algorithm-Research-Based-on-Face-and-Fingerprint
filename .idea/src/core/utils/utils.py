# 辅助函数库
import logging
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


def setup_logger(log_dir='./logs', log_file='training.log', level=logging.INFO, experiment_name=None, logger_name='Training'):
    """设置日志记录器"""
    # 如果提供了实验名称，在日志目录下创建子目录
    if experiment_name:
        log_dir = os.path.join(log_dir, experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path, merge_shared=True):
    """加载YAML配置文件，并自动合并 shared 参数

    Args:
        config_path: 配置文件路径
        merge_shared: 是否从 unified_config.yaml 合并共享参数

    Returns:
        合并后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if merge_shared and config is not None:
        # 尝试加载统一配置并合并共享参数
        config_dir = os.path.dirname(os.path.abspath(config_path))
        unified_path = os.path.join(config_dir, 'unified_config.yaml')
        if os.path.exists(unified_path):
            with open(unified_path, 'r', encoding='utf-8') as f:
                unified = yaml.safe_load(f)
            if unified and 'shared' in unified:
                # 深度合并：配置文件的参数优先
                config = _deep_merge(unified['shared'], config)

    return config


def _deep_merge(shared, specific):
    """深度合并两个字典，specific 优先"""
    result = shared.copy()
    for key, value in specific.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config, save_path):
    """保存配置到YAML文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, checkpoint_dir='./checkpoints'):
    """保存模型检查点"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # 保存最新的检查点
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    accuracy = checkpoint.get('accuracy', 0.0)

    return epoch, loss, accuracy


def calculate_metrics(y_true, y_pred, num_classes=None):
    """计算分类指标"""
    # 准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division='warn'
    )

    # 每类精确率、召回率、F1分数
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division='warn'
    )

    # 确保类型正确
    precision_per_class = np.array(precision_per_class)
    recall_per_class = np.array(recall_per_class)
    f1_per_class = np.array(f1_per_class)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': list(precision_per_class) if num_classes else None,
        'recall_per_class': list(recall_per_class) if num_classes else None,
        'f1_per_class': list(f1_per_class) if num_classes else None
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(10, 8)):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_biometric_metrics(y_true, y_prob, num_classes=None):
    """
    计算生物识别指标 (FAR, FRR, EER等)

    Args:
        y_true: 真实标签
        y_prob: 预测概率 [n_samples, n_classes]
        num_classes: 类别数量

    Returns:
        dict: 包含各种生物识别指标
    """
    import numpy as np

    if num_classes is None:
        num_classes = len(np.unique(y_true))

    results = {}

    # 对于每个类别，计算该类别vs其他类别的FAR/FRR
    all_fars = []
    all_frrs = []
    all_eers = []
    all_auc_scores = []

    for class_idx in range(num_classes):
        # 将当前类别设为正类，其他类别设为负类
        y_binary = (np.array(y_true) == class_idx).astype(int)
        y_score = np.array(y_prob)[:, class_idx] if len(np.array(y_prob).shape) > 1 else np.array(y_prob)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        auc_score = auc(fpr, tpr)

        # 计算FAR (FPR) 和 FRR (1 - TPR)
        far = fpr  # False Acceptance Rate
        frr = 1 - tpr  # False Rejection Rate

        # 计算EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2

        all_fars.append(far)
        all_frrs.append(frr)
        all_eers.append(eer)
        all_auc_scores.append(auc_score)

        results[f'class_{class_idx}'] = {
            'far': far.tolist(),
            'frr': frr.tolist(),
            'thresholds': thresholds.tolist(),
            'eer': eer,
            'auc': auc_score
        }

    # 计算平均指标
    results['macro_avg'] = {
        'eer': np.mean(all_eers),
        'auc': np.mean(all_auc_scores)
    }

    # 计算整体EER (所有类别一起考虑)
    # 使用余弦相似度风格的计算：同人 vs 异人
    # 这里用最大预测概率作为相似度分数
    y_score_flat = np.max(y_prob, axis=1) if len(np.array(y_prob).shape) > 1 else np.array(y_prob)

    # 正确的 EER 计算：需要正样本和负样本的分数分布
    # 同人样本（同 ground truth 类别的样本）→ 正样本
    # 异人样本（不同类别的样本）→ 负样本
    positive_scores = []
    negative_scores = []

    for i, true_label in enumerate(y_true):
        score = y_score_flat[i]
        if np.array(y_prob).ndim > 1:
            # 同类别的其他样本的平均分数作为正样本分数
            same_class_mask = np.array(y_true) == true_label
            same_class_scores = y_score_flat[same_class_mask]
            if len(same_class_scores) > 1:
                positive_scores.append(np.mean(same_class_scores))
            # 异类别的平均分数作为负样本分数
            diff_class_mask = np.array(y_true) != true_label
            diff_class_scores = y_score_flat[diff_class_mask]
            if len(diff_class_scores) > 0:
                negative_scores.append(np.mean(diff_class_scores))

    if len(positive_scores) >= 10 and len(negative_scores) >= 10:
        y_binary_overall = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
        y_scores_combined = np.array(positive_scores + negative_scores)
        fpr_overall, tpr_overall, _ = roc_curve(y_binary_overall, y_scores_combined)
        fnr_overall = 1 - tpr_overall
        eer_idx = np.nanargmin(np.abs(fpr_overall - fnr_overall))
        eer_overall = (fpr_overall[eer_idx] + fnr_overall[eer_idx]) / 2
        auc_overall = auc(fpr_overall, tpr_overall)
    else:
        eer_overall = np.mean(all_eers) if all_eers else 0.0
        auc_overall = np.mean(all_auc_scores) if all_auc_scores else 0.0

    results['overall'] = {
        'eer': eer_overall,
        'auc': auc_overall
    }

    return results


def plot_roc_curves(biometric_results, save_path=None, figsize=(10, 8)):
    """
    绘制ROC曲线

    Args:
        biometric_results: calculate_biometric_metrics的返回值
        save_path: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))

    for i, (class_name, metrics) in enumerate(biometric_results.items()):
        if class_name in ['macro_avg', 'overall']:
            continue

        if 'far' in metrics and 'frr' in metrics:
            far = np.array(metrics['far'])
            frr = np.array(metrics['frr'])

            # ROC曲线: TPR vs FPR, 而FRR = 1 - TPR, 所以TPR = 1 - FRR
            tpr = 1 - frr
            fpr = far

            auc_score = metrics.get('auc', 0)
            plt.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f'{class_name} (AUC={auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_det_curves(biometric_results, save_path=None, figsize=(10, 8)):
    """
    绘制检测误差权衡曲线 (DET曲线)

    Args:
        biometric_results: calculate_biometric_metrics的返回值
        save_path: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))

    for i, (class_name, metrics) in enumerate(biometric_results.items()):
        if class_name in ['macro_avg', 'overall']:
            continue

        if 'far' in metrics and 'frr' in metrics:
            far = np.array(metrics['far'])
            frr = np.array(metrics['frr'])

            # DET曲线在双对数坐标系中绘制FAR vs FRR
            # 过滤掉0值以避免log(0)
            valid_idx = (far > 0) & (frr > 0)
            far_valid = far[valid_idx]
            frr_valid = frr[valid_idx]

            if len(far_valid) > 0:
                eer = metrics.get('eer', 0)
                plt.plot(far_valid, frr_valid, color=colors[i % len(colors)],
                        label=f'{class_name} (EER={eer:.3f})')

    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Detection Error Tradeoff (DET) Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    # 添加EER等值线
    eer_levels = [0.001, 0.01, 0.1]
    for eer in eer_levels:
        plt.plot([eer, 1], [eer, eer], 'k:', alpha=0.3)
        plt.plot([eer, eer], [eer, 1], 'k:', alpha=0.3)

    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_far_frr_curves(biometric_results, save_path=None, figsize=(12, 5)):
    """
    绘制FAR和FRR随阈值变化的曲线

    Args:
        biometric_results: calculate_biometric_metrics的返回值
        save_path: 保存路径
        figsize: 图表大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))

    for i, (class_name, metrics) in enumerate(biometric_results.items()):
        if class_name in ['macro_avg', 'overall']:
            continue

        if 'far' in metrics and 'frr' in metrics and 'thresholds' in metrics:
            thresholds = np.array(metrics['thresholds'])
            far = np.array(metrics['far'])
            frr = np.array(metrics['frr'])
            eer = metrics.get('eer', 0)

            # FAR vs Threshold
            ax1.plot(thresholds, far, color=colors[i % len(colors)],
                    label=f'{class_name} (EER={eer:.3f})')

            # FRR vs Threshold
            ax2.plot(thresholds, frr, color=colors[i % len(colors)],
                    label=f'{class_name} (EER={eer:.3f})')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('False Acceptance Rate (FAR)')
    ax1.set_title('FAR vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('False Rejection Rate (FRR)')
    ax2.set_title('FRR vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_biometric_results(biometric_results, save_path):
    """
    保存生物识别指标结果到JSON文件

    Args:
        biometric_results: calculate_biometric_metrics的返回值
        save_path: 保存路径
    """
    # 将numpy数组转换为list以便JSON序列化
    serializable_results = {}
    for key, value in biometric_results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_results[key][sub_key] = sub_value.tolist()
                else:
                    serializable_results[key][sub_key] = sub_value
        else:
            serializable_results[key] = value

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str='auto'):
    """获取计算设备"""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str.startswith('cuda'):
        device = torch.device(device_str)
    else:
        device = torch.device('cpu')

    return device


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def save_results_to_json(results, save_path):
    """将结果保存为JSON文件"""
    # 将numpy数组转换为列表，以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def compute_classification_metrics(y_true, y_pred):
    """计算分类指标（precision / recall / f1）。

    供各 Trainer 的 validate_epoch 共用。
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    try:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


class TensorBoardWriter:
    """TensorBoard日志写入器"""

    def __init__(self, log_dir='./logs'):
        self.writer = SummaryWriter(log_dir)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """添加标量数据"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """添加单个标量"""
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        """添加图像"""
        self.writer.add_image(tag, img_tensor, global_step)

    def close(self):
        """关闭写入器"""
        self.writer.close()


def create_data_splits(data_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """创建训练/验证/测试数据分割"""
    import random
    random.seed(seed)

    # 获取所有类别
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    train_data = []
    val_data = []
    test_data = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # 打乱顺序
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_data.extend([(img, class_name) for img in images[:n_train]])
        val_data.extend([(img, class_name) for img in images[n_train:n_train+n_val]])
        test_data.extend([(img, class_name) for img in images[n_train+n_val:]])

    return train_data, val_data, test_data


# 测试代码
if __name__ == "__main__":
    # 测试日志设置
    logger = setup_logger()
    logger.info("工具函数库测试")

    # 测试配置加载
    config_path = '../configs/config.yaml'
    if os.path.exists(config_path):
        config = load_config(config_path)
        logger.info(f"加载配置成功: {config['model']['num_classes']} 类")

    # 测试随机种子
    set_seed(42)
    logger.info("随机种子设置完成")

    # 测试设备获取
    device = get_device()
    logger.info(f"使用设备: {device}")