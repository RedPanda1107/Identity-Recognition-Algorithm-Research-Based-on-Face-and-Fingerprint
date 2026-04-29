"""
推理服务配置文件
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", "1"))

DEVICE = os.getenv("DEVICE", "auto")

MODEL_CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")
MODEL_NUM_CLASSES = 500
MODEL_EMBEDDING_DIM = 512
MODEL_FUSION_DIM = 256

GALLERY_DIR = str(PROJECT_ROOT / "data" / "gallery")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

REQUEST_TIMEOUT = 30

MATCH_TOP_K = 5
MATCH_SCORE_THRESHOLD = 0.5

ALLOWED_ORIGINS = ["*"]

__all__ = [
    "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS",
    "DEVICE", "MODEL_CHECKPOINT_DIR", "MODEL_NUM_CLASSES",
    "MODEL_EMBEDDING_DIM", "MODEL_FUSION_DIM",
    "GALLERY_DIR", "LOG_LEVEL", "LOG_JSON",
    "REQUEST_TIMEOUT", "MATCH_TOP_K", "MATCH_SCORE_THRESHOLD",
    "ALLOWED_ORIGINS",
]
