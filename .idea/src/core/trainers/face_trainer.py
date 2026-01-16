from .base_trainer import BaseTrainer


class FaceTrainer(BaseTrainer):
    """Face-specific trainer. For now it reuses BaseTrainer behavior."""

    def __init__(self, *args, **kwargs):
        super(FaceTrainer, self).__init__(*args, **kwargs)

    # Future: override methods for face-specific logging or metrics
    # keep minimal for now to preserve behavior

