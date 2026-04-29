"""
用户管理接口
GET/DELETE /api/v1/users
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import UserListResponse, UserDetailResponse, DeleteResponse, UserCountResponse, UserInfo
from ..dependencies import get_gallery_manager

router = APIRouter(prefix="/api/v1/users", tags=["用户管理"])
logger = logging.getLogger("UsersRoute")


@router.get("", response_model=UserListResponse)
def list_users(gallery=Depends(get_gallery_manager)):
    """获取所有已注册用户 ID 列表"""
    users = gallery.list_users()
    return UserListResponse(success=True, users=users, count=len(users))


@router.get("/count", response_model=UserCountResponse)
def count_users(gallery=Depends(get_gallery_manager)):
    """获取已注册用户数量"""
    count = gallery.count_users()
    return UserCountResponse(success=True, count=count)


@router.get("/{user_id}", response_model=UserDetailResponse)
def get_user(user_id: str, gallery=Depends(get_gallery_manager)):
    """获取指定用户信息"""
    user = gallery.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    return UserDetailResponse(
        success=True,
        user=UserInfo(
            user_id=user_id,
            registered_at=user.get("registered_at"),
            face_image_path=user.get("face_image_path"),
            fingerprint_image_path=user.get("fingerprint_image_path"),
        ),
    )


@router.delete("/{user_id}", response_model=DeleteResponse)
def delete_user(user_id: str, gallery=Depends(get_gallery_manager)):
    """删除指定用户（从 Gallery 移除）"""
    result = gallery.delete_user(user_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    return DeleteResponse(
        success=True,
        user_id=user_id,
        message=f"用户 {user_id} 已删除",
    )
