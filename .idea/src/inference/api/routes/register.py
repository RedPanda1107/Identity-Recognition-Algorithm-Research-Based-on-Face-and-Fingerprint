"""
用户注册接口
POST /api/v1/users/register
"""

import io
import base64
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image

from ..schemas import RegisterRequest, RegisterResponse
from ..dependencies import get_feature_service, get_gallery_manager

router = APIRouter(prefix="/api/v1/users", tags=["用户管理"])
logger = logging.getLogger("RegisterRoute")


def _decode_image(data: str) -> Image.Image:
    """从 Base64 解码图片"""
    try:
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {e}")


def _load_image(path: str) -> Image.Image:
    """从路径加载图片"""
    try:
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"图片文件不存在: {path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片加载失败: {e}")


@router.post("/register", response_model=RegisterResponse)
def register_user(
    request: RegisterRequest,
    feature_service=Depends(get_feature_service),
    gallery=Depends(get_gallery_manager),
):
    """注册新用户或更新已有用户的人脸/指纹特征"""

    if not request.face_image and not request.fingerprint_image and \
       not request.face_image_path and not request.fingerprint_image_path:
        raise HTTPException(status_code=400, detail="必须提供 face_image 或 fingerprint_image")

    face_feature = None
    fp_feature = None
    face_path = None
    fp_path = None

    try:
        if request.face_image:
            face_img = _decode_image(request.face_image)
            face_feature = feature_service.extract_face(face_img)
        elif request.face_image_path:
            face_img = _load_image(request.face_image_path)
            face_feature = feature_service.extract_face(face_img)
            face_path = request.face_image_path

        if request.fingerprint_image:
            fp_img = _decode_image(request.fingerprint_image)
            fp_feature = feature_service.extract_fingerprint(fp_img)
        elif request.fingerprint_image_path:
            fp_img = _load_image(request.fingerprint_image_path)
            fp_feature = feature_service.extract_fingerprint(fp_img)
            fp_path = request.fingerprint_image_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Register] Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"特征提取失败: {e}")

    result = gallery.register_user(
        user_id=request.user_id,
        face_feature=face_feature,
        fingerprint_feature=fp_feature,
        face_image_path=face_path,
        fingerprint_image_path=fp_path,
    )

    return RegisterResponse(
        success=True,
        user_id=request.user_id,
        message=f"用户 {request.user_id} {'注册' if result['is_new'] else '更新'}成功",
        gallery_updated=True,
    )
