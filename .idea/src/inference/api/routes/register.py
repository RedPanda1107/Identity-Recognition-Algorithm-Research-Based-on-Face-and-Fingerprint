"""
用户注册接口
POST /api/v1/users/register
"""

import io
import base64
import logging
import uuid
from pathlib import Path

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


def _save_image_to_gallery(image: Image.Image, user_id: str, modality: str, project_root: Path) -> str:
    """将图片保存到 Gallery 的 images/{user_id}/ 目录，返回相对路径"""
    images_dir = project_root / "data" / "gallery" / "images" / user_id
    images_dir.mkdir(parents=True, exist_ok=True)

    ext = "jpg"
    filename = f"{modality}.{ext}"
    filepath = images_dir / filename

    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(filepath, "JPEG", quality=95)
    relative_path = str(filepath.relative_to(project_root))
    logger.info(f"[Register] Saved {modality} image for {user_id} -> {filepath}")
    return relative_path


@router.post("/register", response_model=RegisterResponse)
def register_user(
    request: RegisterRequest,
    feature_service=Depends(get_feature_service),
    gallery=Depends(get_gallery_manager),
):
    """注册新用户或更新已有用户的人脸/指纹特征"""

    face_img = None
    fp_img = None

    if request.face_image and request.face_image.strip():
        face_img = _decode_image(request.face_image)
    elif request.face_image_path:
        face_img = _load_image(request.face_image_path)

    if request.fingerprint_image and request.fingerprint_image.strip():
        fp_img = _decode_image(request.fingerprint_image)
    elif request.fingerprint_image_path:
        fp_img = _load_image(request.fingerprint_image_path)

    if face_img is None and fp_img is None:
        raise HTTPException(status_code=400, detail="face_image 和 fingerprint_image 至少需要提供一个")

    face_feature = None
    fp_feature = None
    face_path = None
    fp_path = None

    project_root = Path(__file__).resolve().parent.parent.parent.parent

    try:
        if face_img is not None:
            face_feature = feature_service.extract_face(face_img)
            face_path = _save_image_to_gallery(face_img, request.user_id, "face", project_root)

        if fp_img is not None:
            fp_feature = feature_service.extract_fingerprint(fp_img)
            fp_path = _save_image_to_gallery(fp_img, request.user_id, "fingerprint", project_root)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Register] Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"特征提取失败: {e}")

    result = gallery.register_user(
        user_id=request.user_id,
        name=request.name,
        face_feature=face_feature,
        fingerprint_feature=fp_feature,
        face_image_path=face_path,
        fingerprint_image_path=fp_path,
    )

    return RegisterResponse(
        success=True,
        user_id=request.user_id,
        name=request.name,
        message=f"用户 {request.user_id} {'注册' if result['is_new'] else '更新'}成功",
        gallery_updated=True,
    )
