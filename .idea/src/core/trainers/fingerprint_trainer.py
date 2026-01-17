from .base_trainer import BaseTrainer


class FingerprintTrainer(BaseTrainer):
    """Fingerprint-specific trainer. Reuses BaseTrainer behavior with fingerprint-specific optimizations."""

    def __init__(self, *args, **kwargs):
        super(FingerprintTrainer, self).__init__(*args, **kwargs)

    # Future: override methods for fingerprint-specific logging or metrics
    # For now, keep minimal to preserve behavior
    # Potential additions:
    # - Fingerprint-specific validation metrics (ridge/valley detection accuracy)
    # - Enhanced logging for texture feature learning
    # - Specialized learning rate scheduling for fingerprint convergence