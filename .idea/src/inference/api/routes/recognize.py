"""
识别接口
POST /api/v1/recognize/{modality}
"""

import io
import base64
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image

from ..schemas import RecognizeRequest, RecognizeResponse, Candidate
from ..dependencies import get_feature_service, get_gallery_manager, get_matching_service

router = APIRouter(prefix="/api/v1/recognize", tags=["身份识别"])
logger = logging.getLogger("RecognizeRoute")


def _decode_image(data: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {e}")


def _load_image(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"图片文件不存在: {path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片加载失败: {e}")


def _get_image(request: RecognizeRequest, key: str) -> Image.Image:
    b64 = getattr(request, key)
    path = getattr(request, key.replace("_image", "_image_path"))
    if b64:
        return _decode_image(b64)
    elif path:
        return _load_image(path)
    raise HTTPException(status_code=400, detail=f"缺少 {key}")


@router.post("/face", response_model=RecognizeResponse)
def recognize_face(request: RecognizeRequest):
    """纯人脸识别"""
    from ..dependencies import get_feature_service, get_gallery_manager, get_matching_service
    feature_service = get_feature_service()
    gallery = get_gallery_manager()
    matching = get_matching_service()

    face_img = _get_image(request, "face_image")
    face_feature = feature_service.extract_face(face_img)

    features, ids = gallery.get_all_features(modality="face")
    if not features:
        raise HTTPException(status_code=400, detail="Gallery 为空，请先注册用户")

    matching.set_gallery(features, ids)
    results = matching.match(face_feature, top_k=request.top_k,
                             modality="face", score_threshold=request.score_threshold)

    if not results:
        return RecognizeResponse(success=True, matched=False, candidates=[], modality="face")

    top = results[0]
    return RecognizeResponse(
        success=True,
        matched=top["confidence"] >= request.score_threshold,
        user_id=top["user_id"],
        confidence=top["confidence"],
        candidates=[Candidate(**r) for r in results],
        modality="face",
    )


@router.post("/fingerprint", response_model=RecognizeResponse)
def recognize_fingerprint(request: RecognizeRequest):
    """纯指纹识别"""
    from ..dependencies import get_feature_service, get_gallery_manager, get_matching_service
    feature_service = get_feature_service()
    gallery = get_gallery_manager()
    matching = get_matching_service()

    fp_img = _get_image(request, "fingerprint_image")
    fp_feature = feature_service.extract_fingerprint(fp_img)

    features, ids = gallery.get_all_features(modality="fingerprint")
    if not features:
        raise HTTPException(status_code=400, detail="Gallery 为空，请先注册用户")

    matching.set_gallery(features, ids)
    results = matching.match(fp_feature, top_k=request.top_k,
                             modality="fingerprint", score_threshold=request.score_threshold)

    if not results:
        return RecognizeResponse(success=True, matched=False, candidates=[], modality="fingerprint")

    top = results[0]
    return RecognizeResponse(
        success=True,
        matched=top["confidence"] >= request.score_threshold,
        user_id=top["user_id"],
        confidence=top["confidence"],
        candidates=[Candidate(**r) for r in results],
        modality="fingerprint",
    )


@router.post("/fusion", response_model=RecognizeResponse)
def recognize_fusion(request: RecognizeRequest):
    """人脸+指纹融合识别，支持单独人脸或单独指纹

    所有推理路径统一经过融合模型：
    - 双模态：完整融合
    - 仅人脸：指纹以零向量注入，融合模型自动降低指纹贡献
    - 仅指纹：人脸以零向量注入，融合模型自动降低人脸贡献

    Gallery 中的单模态特征在识别时会被投影到融合空间，再做匹配。
    """
    from pathlib import Path

    feature_service = get_feature_service()
    gallery = get_gallery_manager()
    matching = get_matching_service()

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

    # 统一走融合模型进行特征提取（支持 None 模态）
    fusion_result = feature_service.fusion.extract_all(
        face_img, fp_img, method=request.fusion_method
    )
    query_fused = fusion_result["fused_embedding"]
    modality_tag = fusion_result["modality"]

    # Gallery 加载
    gallery_face, ids_face = gallery.get_all_features(modality="face")
    gallery_fp, ids_fp = gallery.get_all_features(modality="fingerprint")

    if not gallery_face and not gallery_fp:
        raise HTTPException(status_code=400, detail="Gallery 为空，请先注册用户")

    results = []
    matched_user_id = None
    confidence = 0.0
    face_confidence_val = 0.0
    fp_confidence_val = 0.0

    if modality_tag == "fusion":
        # 双模态识别：Gallery 双模态特征分别投影后加权融合匹配
        if len(gallery_face) != len(gallery_fp):
            raise HTTPException(status_code=400, detail="人脸和指纹 Gallery 大小不一致，请检查数据")

        fusion_weights = (request.fusion_weight_face, request.fusion_weight_fp)
        results = matching.match_multi_modal(
            face_feature=fusion_result["face_embedding"],
            fp_feature=fusion_result["fp_embedding"],
            fused_feature=query_fused,
            gallery_face=gallery_face,
            gallery_fp=gallery_fp,
            gallery_ids=ids_face,
            fusion_weights=fusion_weights,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        if results:
            top = results[0]
            matched_user_id = top["user_id"]
            confidence = top["confidence"]
            face_confidence_val = top.get("face_confidence", 0.0)
            fp_confidence_val = top.get("fingerprint_confidence", 0.0)

    elif modality_tag == "face_only":
        # 仅人脸：Gallery 人脸投影到融合空间后匹配
        if not gallery_face:
            raise HTTPException(status_code=400, detail="Gallery 中没有注册人脸，请先注册带人脸的用户")

        gallery_proj = feature_service.fusion.project_gallery_to_fusion_space(
            gallery_face, modality="face", method=request.fusion_method
        )
        matching.set_gallery(gallery_proj.tolist(), ids_face)
        results = matching.match(
            query_fused, top_k=request.top_k, modality="fusion",
            score_threshold=request.score_threshold
        )
        if results:
            top = results[0]
            matched_user_id = top["user_id"]
            confidence = top["confidence"]
            face_confidence_val = top["confidence"]

    elif modality_tag == "fingerprint_only":
        # 仅指纹：Gallery 指纹投影到融合空间后匹配
        if not gallery_fp:
            raise HTTPException(status_code=400, detail="Gallery 中没有注册指纹，请先注册带指纹的用户")

        gallery_proj = feature_service.fusion.project_gallery_to_fusion_space(
            gallery_fp, modality="fingerprint", method=request.fusion_method
        )
        matching.set_gallery(gallery_proj.tolist(), ids_fp)
        results = matching.match(
            query_fused, top_k=request.top_k, modality="fusion",
            score_threshold=request.score_threshold
        )
        if results:
            top = results[0]
            matched_user_id = top["user_id"]
            confidence = top["confidence"]
            fp_confidence_val = top["confidence"]

    # 填充 name 和 face_image
    name = None
    face_image_b64 = None
    if matched_user_id:
        user_data = gallery.get_user(matched_user_id)
        if user_data:
            name = user_data.get("name")
            face_img_path = user_data.get("face_image_path")
            if face_img_path:
                path = Path(face_img_path)
                if not path.is_absolute():
                    project_root = Path(__file__).resolve().parent.parent.parent.parent
                    path = project_root / face_img_path
                if path.exists():
                    with open(path, "rb") as f:
                        face_image_b64 = base64.b64encode(f.read()).decode("utf-8")

    matched = matched_user_id is not None and confidence >= request.score_threshold

    return RecognizeResponse(
        success=True,
        matched=matched,
        user_id=matched_user_id,
        name=name,
        confidence=confidence,
        face_confidence=face_confidence_val,
        fingerprint_confidence=fp_confidence_val,
        face_image=face_image_b64,
        candidates=[Candidate(**{k: v for k, v in r.items() if k != "modality"}) for r in results],
        modality=modality_tag,
        fusion_method=request.fusion_method,
    )
