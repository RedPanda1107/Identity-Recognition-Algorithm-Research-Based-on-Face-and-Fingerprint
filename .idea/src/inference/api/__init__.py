"""Inference API package — FastAPI routes and schemas."""

from inference.api.routes import register, recognize, users
from inference.api.schemas import (
    RegisterRequest,
    RegisterResponse,
    RecognizeRequest,
    RecognizeResponse,
    HealthResponse,
)

__all__ = [
    "register",
    "recognize",
    "users",
    "RegisterRequest",
    "RegisterResponse",
    "RecognizeRequest",
    "RecognizeResponse",
    "HealthResponse",
]
