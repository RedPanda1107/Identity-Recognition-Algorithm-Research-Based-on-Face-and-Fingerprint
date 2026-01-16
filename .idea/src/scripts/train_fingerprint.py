#!/usr/bin/env python
"""
Placeholder training script for fingerprint modality.
This file is intentionally minimal and documents the expected pipeline for
the fingerprint implementation: Dataset -> Model -> Trainer.

Expected implementation structure:
- core/datasets/fingerprint_dataset.py: FingerprintDataset class
- core/models/fingerprint_model.py: Fingerprint feature extractor
- core/trainers/fingerprint_trainer.py: FingerprintTrainer class
"""

import os
import sys
import argparse

# Add project root to path for imports (when implemented)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "configs", "fingerprint.yaml")

    parser = argparse.ArgumentParser(description="Train fingerprint recognition model")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--experiment_name", type=str, default="fingerprint_recognition")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "Fingerprint training script not implemented yet. "
        "Implement the following components first:\n"
        "- core/datasets/fingerprint_dataset.py (FingerprintDataset)\n"
        "- core/models/fingerprint_model.py (create_fingerprint_model)\n"
        "- core/trainers/fingerprint_trainer.py (FingerprintTrainer)\n"
        "- configs/fingerprint.yaml\n"
        "Then populate this script with the training pipeline."
    )


if __name__ == "__main__":
    main()