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
    """人脸+指纹融合识别"""
    from ..dependencies import get_feature_service, get_gallery_manager, get_matching_service
    feature_service = get_feature_service()
    gallery = get_gallery_manager()
    matching = get_matching_service()

    face_img = _get_image(request, "face_image")
    fp_img = _get_image(request, "fingerprint_image")

    result = feature_service.fusion.extract_all(
        face_img, fp_img, method=request.fusion_method
    )

    face_feat = result["face_embedding"]
    fp_feat = result["fp_embedding"]
    fused_feat = result["fused_embedding"]

    gallery_face, ids = gallery.get_all_features(modality="face")
    gallery_fp, _ = gallery.get_all_features(modality="fingerprint")

    if not gallery_face:
        raise HTTPException(status_code=400, detail="Gallery 为空，请先注册用户")
    if len(gallery_fp) != len(gallery_face):
        raise HTTPException(status_code=400, detail="人脸和指纹 Gallery 大小不一致，请检查数据")

    matching.set_gallery(gallery_face, ids)
    face_results = matching.match(face_feat, top_k=request.top_k, modality="face")

    fusion_weights = (request.fusion_weight_face, request.fusion_weight_fp)

    results = matching.match_multi_modal(
        face_feature=face_feat,
        fp_feature=fp_feat,
        fused_feature=fused_feat,
        gallery_face=gallery_face,
        gallery_fp=gallery_fp,
        gallery_ids=ids,
        fusion_weights=fusion_weights,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    if not results:
        return RecognizeResponse(
            success=True, matched=False, candidates=[],
            modality="fusion", fusion_method=request.fusion_method,
        )

    top = results[0]
    return RecognizeResponse(
        success=True,
        matched=top["confidence"] >= request.score_threshold,
        user_id=top["user_id"],
        confidence=top["confidence"],
        face_confidence=top.get("face_confidence"),
        fingerprint_confidence=top.get("fingerprint_confidence"),
        candidates=[Candidate(**{k: v for k, v in r.items() if k != "modality"}) for r in results],
        modality="fusion",
        fusion_method=request.fusion_method,
    )
