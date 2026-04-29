"""Unified training logger with consistent output format.

All trainers (Face, Fingerprint, Fusion) should use this logger
to ensure consistent, machine-parseable output for analysis.

Format:
    [2026-04-21 10:30:15] phase=val modality=face epoch=10/50 lr=1.0e-4 loss=0.089 acc=N/A rank1=0.942 eer=0.031 gallery=180 query=342

Fields:
    phase: train | val
    modality: face | fingerprint | fusion
    epoch: current/total
    lr: learning rate (scientific notation)
    loss: loss value
    acc: classification accuracy (only for training, N/A for validation)
    rank1: Rank-1 retrieval accuracy (N/A for training)
    eer: Equal Error Rate (N/A for training)
    gallery: gallery set size
    query: query set size
"""

import logging
from datetime import datetime
from typing import Optional

_logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified training log formatter.

    Provides consistent log format across all modality trainers
    for easy analysis and comparison.
    """

    @staticmethod
    def format_epoch_log(
        phase: str,
        modality: str,
        epoch: int,
        total_epochs: int,
        lr: float,
        loss: float,
        acc: Optional[float] = None,
        rank1: Optional[float] = None,
        eer: Optional[float] = None,
        gallery_size: Optional[int] = None,
        query_size: Optional[int] = None
    ) -> str:
        """Format a single epoch log line.

        Args:
            phase: 'train' or 'val'
            modality: 'face', 'fingerprint', or 'fusion'
            epoch: Current epoch number
            total_epochs: Total number of epochs
            lr: Current learning rate
            loss: Loss value
            acc: Classification accuracy (optional, typically only for training)
            rank1: Rank-1 retrieval accuracy
            eer: Equal Error Rate
            gallery_size: Gallery set size
            query_size: Query set size

        Returns:
            Formatted log string
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format fields
        acc_str = f"{acc:.3f}" if acc is not None else "N/A"
        rank1_str = f"{rank1:.3f}" if rank1 is not None else "N/A"
        eer_str = f"{eer:.3f}" if eer is not None else "N/A"
        gallery_str = str(gallery_size) if gallery_size is not None else "N/A"
        query_str = str(query_size) if query_size is not None else "N/A"

        log_parts = [
            f"[{timestamp}]",
            f"phase={phase}",
            f"modality={modality}",
            f"epoch={epoch}/{total_epochs}",
            f"lr={lr:.1e}",
            f"loss={loss:.4f}",
            f"acc={acc_str}",
            f"rank1={rank1_str}",
            f"eer={eer_str}",
            f"gallery={gallery_str}",
            f"query={query_str}"
        ]

        return " ".join(log_parts)

    @staticmethod
    def log_epoch(logger: logging.Logger, **kwargs):
        """Log an epoch to the provided logger.

        Args:
            logger: Python logger instance
            **kwargs: See format_epoch_log parameters
        """
        log_line = TrainingLogger.format_epoch_log(**kwargs)
        logger.info(log_line)

    @staticmethod
    def print_epoch(**kwargs):
        """Print an epoch log to stdout.

        Args:
            **kwargs: See format_epoch_log parameters
        """
        log_line = TrainingLogger.format_epoch_log(**kwargs)
        print(log_line)

    @staticmethod
    def log_comparison(
        logger: logging.Logger,
        modality: str,
        epoch: int,
        total_epochs: int,
        lr: float,
        results: dict
    ):
        """Log a comparison between single-modal and fusion performance.

        Args:
            logger: Python logger instance
            modality: 'face' | 'fingerprint' | 'fusion'
            epoch: Current epoch
            total_epochs: Total epochs
            lr: Current learning rate
            results: Dict with keys like 'face_rank1', 'fingerprint_rank1', 'fusion_rank1', etc.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_parts = [f"[{timestamp}]", f"phase=val", f"modality={modality}"]

        if 'face_rank1' in results:
            log_parts.append(f"face_rank1={results['face_rank1']:.3f}")
        if 'fingerprint_rank1' in results:
            log_parts.append(f"fp_rank1={results['fingerprint_rank1']:.3f}")
        if 'fusion_rank1' in results:
            log_parts.append(f"fusion_rank1={results['fusion_rank1']:.3f}")
        if 'face_eer' in results:
            log_parts.append(f"face_eer={results['face_eer']:.3f}")
        if 'fingerprint_eer' in results:
            log_parts.append(f"fp_eer={results['fingerprint_eer']:.3f}")
        if 'fusion_eer' in results:
            log_parts.append(f"fusion_eer={results['fusion_eer']:.3f}")
        if 'improvement' in results:
            log_parts.append(f"improvement={results['improvement']:+.3f}")

        logger.info(" ".join(log_parts))
