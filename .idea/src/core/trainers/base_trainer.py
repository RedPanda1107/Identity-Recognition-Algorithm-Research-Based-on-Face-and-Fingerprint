import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve


class AverageMeter:
    """简单的均值计量器，供各 Trainer 子类使用。"""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


class BaseTrainer:
    """所有 Trainer 的基类，提供通用组件和统一接口。

    子类必须实现 train_epoch、validate_epoch、save_checkpoint 方法。
    统一使用 TrainingLogger 进行日志输出。
    """

    MODALITY = 'base'  # Override in subclasses: 'face', 'fingerprint', 'fusion'

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 criterion, device, logger, tb_writer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.tb_writer = tb_writer

    def train_epoch(self, epoch):
        raise NotImplementedError("Subclass must implement train_epoch")

    def validate_epoch(self, epoch):
        raise NotImplementedError("Subclass must implement validate_epoch")

    def save_checkpoint(self, path, **kwargs):
        raise NotImplementedError("Subclass must implement save_checkpoint")

    # ============================================================
    # 统一的日志输出接口
    # ============================================================

    def log_train_epoch(self, epoch, total_epochs, lr, loss, acc,
                         gallery_size=None, query_size=None):
        """统一格式输出训练 epoch 日志。"""
        from ..utils.logger import TrainingLogger
        log_line = TrainingLogger.format_epoch_log(
            phase='train',
            modality=self.MODALITY,
            epoch=epoch,
            total_epochs=total_epochs,
            lr=lr,
            loss=loss,
            acc=acc,
            rank1=None,  # 训练时不输出 rank1
            eer=None,    # 训练时不输出 eer
            gallery_size=gallery_size,
            query_size=query_size
        )
        self.logger.info(log_line)

    def log_val_epoch(self, epoch, total_epochs, lr, loss,
                      rank1=None, eer=None, gallery_size=None, query_size=None):
        """统一格式输出验证 epoch 日志。"""
        from ..utils.logger import TrainingLogger
        log_line = TrainingLogger.format_epoch_log(
            phase='val',
            modality=self.MODALITY,
            epoch=epoch,
            total_epochs=total_epochs,
            lr=lr,
            loss=loss,
            acc=None,  # 验证时不输出分类准确率
            rank1=rank1,
            eer=eer,
            gallery_size=gallery_size,
            query_size=query_size
        )
        self.logger.info(log_line)

    # ============================================================
    # 通用评估工具
    # ============================================================

    def compute_retrieval_metrics(self, query_embeddings, gallery_embeddings,
                                   query_labels, gallery_labels, rank_k=[1, 5, 10, 20]):
        """计算 1:N 检索指标 (Rank-K, EER)。

        Args:
            query_embeddings: (N, D) query 特征
            gallery_embeddings: (M, D) gallery 特征
            query_labels: (N,) query 标签
            gallery_labels: (M,) gallery 标签
            rank_k: list of K values for Rank-K

        Returns:
            dict with keys: rank1, rank5, rank10, rank20, eer
        """
        device = query_embeddings.device
        # 确保为 tensor
        if isinstance(query_labels, np.ndarray):
            query_labels = torch.from_numpy(query_labels)
        if isinstance(gallery_labels, np.ndarray):
            gallery_labels = torch.from_numpy(gallery_labels)
        query_labels = query_labels.to(device)
        gallery_labels = gallery_labels.to(device)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

        # 计算余弦相似度
        similarity = torch.mm(query_embeddings, gallery_embeddings.t())

        # Rank-K 准确率
        results = {}
        for k in rank_k:
            _, top_k_indices = torch.topk(similarity, k=min(k, similarity.size(1)), dim=1)
            top_k_labels = gallery_labels[top_k_indices]
            correct = (top_k_labels == query_labels.unsqueeze(1)).any(dim=1).float()
            results[f'rank{k}'] = correct.mean().item()

        # EER 计算
        eer = self._compute_eer(similarity, query_labels, gallery_labels)
        results['eer'] = eer

        return results

    def _compute_eer(self, similarity_matrix, query_labels, gallery_labels,
                      num_negatives_per_query=3, seed=42):
        """计算 EER (Equal Error Rate)。

        Args:
            similarity_matrix: (N, M) 余弦相似度矩阵
            query_labels: (N,) query 标签
            gallery_labels: (M,) gallery 标签
            num_negatives_per_query: 每 query 采样的负样本数
            seed: 随机种子 (保证 EER 可复现)

        Returns:
            EER value
        """
        np.random.seed(seed)
        device = similarity_matrix.device

        # 转换为 numpy
        similarity_np = similarity_matrix.detach().cpu().numpy()
        query_labels_np = np.asarray(query_labels)
        gallery_labels_np = np.asarray(gallery_labels)

        # 收集正负样本对
        positive_scores = []
        negative_scores = []

        n_queries = similarity_np.shape[0]
        for q_idx in range(n_queries):
            q_label = query_labels_np[q_idx]

            # 正样本：同标签的 gallery
            positive_mask = (gallery_labels_np == q_label)
            positive_indices = np.where(positive_mask)[0]

            if len(positive_indices) > 0:
                pos_scores = similarity_np[q_idx, positive_indices]
                positive_scores.extend(pos_scores.tolist())

            # 负样本：不同标签的 gallery (随机采样)
            negative_mask = (gallery_labels_np != q_label)
            negative_indices = np.where(negative_mask)[0]

            if len(negative_indices) > 0:
                n_sample = min(num_negatives_per_query, len(negative_indices))
                sampled_negatives = np.random.choice(negative_indices, n_sample, replace=False)
                neg_scores = similarity_np[q_idx, sampled_negatives]
                negative_scores.extend(neg_scores.tolist())

        if len(positive_scores) < 50 or len(negative_scores) < 50:
            return float('nan')

        # 构建 ROC 曲线
        y_scores = np.array(positive_scores + negative_scores)
        y_true = np.array([1] * len(positive_scores) + [0] * len(negative_scores))

        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr

        # 找到 FAR = FRR 的点
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        return float(eer)

    def extract_gallery_query_features(self, model, val_loader, device):
        """从验证集提取 Gallery 和 Query 特征。

        子类可以重写此方法以支持自定义的 gallery/query 提取逻辑。

        Returns:
            tuple: (query_embeddings, gallery_embeddings, query_labels, gallery_labels)
        """
        raise NotImplementedError("Subclass must implement extract_gallery_query_features")

