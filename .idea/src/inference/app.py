"""
FastAPI 主入口 - 门禁系统推理服务
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .config import (
    SERVER_HOST, SERVER_PORT, LOG_LEVEL, LOG_JSON,
    ALLOWED_ORIGINS, MODEL_CHECKPOINT_DIR, DEVICE,
    MODEL_NUM_CLASSES, MODEL_EMBEDDING_DIM, MODEL_FUSION_DIM,
    GALLERY_DIR,
)
from .api.schemas import HealthResponse, ErrorResponse
from .api.dependencies import get_feature_service, get_gallery_manager
from .api.routes import register, recognize, users

LOG_FORMAT = (
    '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
    if LOG_JSON else
    "[%(asctime)s] %(name)s [%(levelname)s] %(message)s"
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AccessLogger")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 50)
    logger.info("门禁系统推理服务启动中...")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Checkpoint: {MODEL_CHECKPOINT_DIR}")
    logger.info(f"  Gallery: {GALLERY_DIR}")
    logger.info("=" * 50)

    try:
        feature_service = get_feature_service()
        gallery = get_gallery_manager()
        logger.info(f"Gallery 加载完成: {gallery.count_users()} 个用户")
    except Exception as e:
        logger.warning(f"预加载失败（服务仍可启动）: {e}")

    yield

    logger.info("门禁系统推理服务关闭")


app = FastAPI(
    title="门禁系统 - 推理服务 API",
    description="基于人脸和指纹的多模态生物特征识别推理服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求"""
    logger.info(f"--> {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"<-- {request.method} {request.url.path} [{response.status_code}]")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
def health_check():
    """健康检查端点"""
    try:
        feature_service = get_feature_service()
        gallery = get_gallery_manager()
        models = feature_service.model_loader.get_loaded_models()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            device=str(feature_service.model_loader.device),
            models_loaded=models,
            gallery_users=gallery.count_users(),
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            device=DEVICE,
            models_loaded={},
            gallery_users=0,
        )


@app.get("/", tags=["根路径"])
def root():
    return {
        "service": "门禁系统推理服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


app.include_router(register.router)
app.include_router(recognize.router)
app.include_router(users.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference.app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )
