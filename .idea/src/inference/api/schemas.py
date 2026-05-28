"""
Pydantic 请求/响应模型定义
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    user_id: str = Field(..., description="用户 ID", min_length=1, max_length=64)
    name: str = Field(..., description="用户姓名")
    face_image: Optional[str] = Field(None, description="人脸图片 Base64 编码")
    face_image_path: Optional[str] = Field(None, description="人脸图片路径（二选一）")
    fingerprint_image: Optional[str] = Field(None, description="指纹图片 Base64 编码")
    fingerprint_image_path: Optional[str] = Field(None, description="指纹图片路径（二选一）")


class RegisterResponse(BaseModel):
    success: bool
    user_id: str
    name: str
    message: str
    gallery_updated: bool = True


class RecognizeRequest(BaseModel):
    face_image: Optional[str] = Field(None, description="人脸图片 Base64 编码")
    face_image_path: Optional[str] = Field(None, description="人脸图片路径（二选一）")
    fingerprint_image: Optional[str] = Field(None, description="指纹图片 Base64 编码")
    fingerprint_image_path: Optional[str] = Field(None, description="指纹图片路径（二选一）")
    top_k: int = Field(5, ge=1, le=20, description="返回前 k 个候选")
    fusion_method: str = Field("adaptive", description="融合方法: simple / adaptive")
    fusion_weight_face: float = Field(0.5, ge=0.0, le=1.0)
    fusion_weight_fp: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("face_image", "fingerprint_image", "face_image_path", "fingerprint_image_path")
    @classmethod
    def _empty_to_none(cls, v):
        if v == "":
            return None
        return v


class Candidate(BaseModel):
    user_id: str
    rank: int


class RecognizeResponse(BaseModel):
    success: bool
    matched: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    face_image: Optional[str] = None
    candidates: List[Candidate] = []
    modality: str = "fusion"
    fusion_method: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    registered_at: Optional[str] = None
    face_image_path: Optional[str] = None
    fingerprint_image_path: Optional[str] = None


class UserListResponse(BaseModel):
    success: bool
    users: List[str]
    count: int


class UserDetailResponse(BaseModel):
    success: bool
    user: Optional[UserInfo] = None


class DeleteResponse(BaseModel):
    success: bool
    user_id: str
    message: str


class UserCountResponse(BaseModel):
    success: bool
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: dict
    gallery_users: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
