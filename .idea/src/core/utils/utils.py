# 辅助函数库
import os
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


def setup_logger(log_dir='./logs', log_file='training.log', level=logging.INFO, experiment_name=None):
    """设置日志记录器"""
    # 如果提供了实验名称，在日志目录下创建子目录
    if experiment_name:
        log_dir = os.path.join(log_dir, experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    logger = logging.getLogger('FaceRecognition')
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


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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
        y_true, y_pred, average='macro', zero_division=0
    )

    # 每类精确率、召回率、F1分数
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class.tolist() if num_classes else None,
        'recall_per_class': recall_per_class.tolist() if num_classes else None,
        'f1_per_class': f1_per_class.tolist() if num_classes else None
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


def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
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


class AverageMeter:
    """计算平均值和当前值的实用工具"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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