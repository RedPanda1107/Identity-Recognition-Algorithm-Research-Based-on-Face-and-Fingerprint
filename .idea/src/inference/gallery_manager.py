"""
Gallery 管理器 - 用户特征向量存储与查询
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

import numpy as np

logger = logging.getLogger("GalleryManager")


class GalleryManager:
    """用户 Gallery 增删改查

    Gallery 结构（JSON 文件存储）：
    {
        "version": "1.0",
        "created_at": "2026-04-29T20:00:00",
        "updated_at": "2026-04-29T20:00:00",
        "users": {
            "001": {
                "registered_at": "2026-04-29T20:00:00",
                "face_feature": [0.123, -0.456, ...],   // 512 维
                "fingerprint_feature": [0.789, 0.234, ...], // 512 维
                "face_image_path": "./gallery/images/001_face.jpg",
                "fingerprint_image_path": "./gallery/images/001_fp.jpg"
            }
        }
    }
    """

    VERSION = "1.0"

    def __init__(
        self,
        gallery_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Args:
            gallery_dir: Gallery 根目录，默认 {project_root}/data/gallery/
            auto_save: 是否在修改后自动保存
        """
        if gallery_dir:
            self.gallery_dir = Path(gallery_dir)
        else:
            project_root = Path(__file__).resolve().parent.parent  # inference/ -> src/
            self.gallery_dir = project_root / "data" / "gallery"

        self.auto_save = auto_save
        self._data: Dict[str, Any] = {
            "version": self.VERSION,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "users": {},
        }
        self._load()

    def _load(self):
        """从文件加载 Gallery 数据，合并人脸和指纹特征"""
        face_file = self.gallery_dir / "features" / "face_features.json"
        fp_file = self.gallery_dir / "features" / "fingerprint_features.json"

        face_data = self._load_file(face_file)
        fp_data = self._load_file(fp_file)

        if face_data:
            self._data = face_data
        elif fp_data:
            self._data = fp_data
        else:
            self._data = {
                "version": self.VERSION,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "users": {},
            }

        if not self._data.get("users"):
            self._data["users"] = {}

        if fp_data:
            for uid, udata in fp_data.get("users", {}).items():
                if uid not in self._data["users"]:
                    self._data["users"][uid] = {}
                fp_feat = udata.get("fingerprint_feature")
                if fp_feat:
                    self._data["users"][uid]["fingerprint_feature"] = fp_feat
                if not self._data["users"][uid].get("name") and udata.get("name"):
                    self._data["users"][uid]["name"] = udata.get("name")
                for k in ("registered_at", "updated_at", "fingerprint_image_path", "face_image_path"):
                    if udata.get(k) and not self._data["users"][uid].get(k):
                        self._data["users"][uid][k] = udata[k]

        # 迁移旧格式路径（001_face.jpg → 001/face.jpg）到新目录结构
        for uid, udata in self._data["users"].items():
            if udata.get("face_image_path"):
                old = udata["face_image_path"]
                new = f"data/gallery/images/{uid}/face.jpg"
                if old and not old.startswith(f"{uid}/") and "gallery/images" in old:
                    udata["face_image_path"] = new
            if udata.get("fingerprint_image_path"):
                old = udata["fingerprint_image_path"]
                new = f"data/gallery/images/{uid}/fingerprint.jpg"
                if old and not old.startswith(f"{uid}/") and "gallery/images" in old:
                    udata["fingerprint_image_path"] = new

        logger.info(f"[Gallery] Loaded {len(self._data['users'])} users")

    def _load_file(self, path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[Gallery] Failed to load {path}: {e}")
            return None

    def _ensure_dir(self):
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        (self.gallery_dir / "features").mkdir(parents=True, exist_ok=True)
        (self.gallery_dir / "images").mkdir(parents=True, exist_ok=True)

    def _save(self):
        """保存 Gallery 数据到文件"""
        self._ensure_dir()
        self._data["updated_at"] = datetime.now().isoformat()

        face_file = self.gallery_dir / "features" / "face_features.json"
        fp_file = self.gallery_dir / "features" / "fingerprint_features.json"

        face_out = dict(self._data)
        face_out["users"] = {}
        for uid, udata in self._data["users"].items():
            face_out["users"][uid] = {
                "registered_at": udata.get("registered_at"),
                "updated_at": udata.get("updated_at"),
                "name": udata.get("name"),
                "face_feature": udata.get("face_feature", []),
                "face_image_path": udata.get("face_image_path"),
            }
        with open(face_file, "w", encoding="utf-8") as f:
            json.dump(face_out, f, ensure_ascii=False, indent=2)

        fp_out = dict(self._data)
        fp_out["users"] = {}
        for uid, udata in self._data["users"].items():
            fp_out["users"][uid] = {
                "registered_at": udata.get("registered_at"),
                "updated_at": udata.get("updated_at"),
                "name": udata.get("name"),
                "fingerprint_feature": udata.get("fingerprint_feature", []),
                "fingerprint_image_path": udata.get("fingerprint_image_path"),
            }
        with open(fp_file, "w", encoding="utf-8") as f:
            json.dump(fp_out, f, ensure_ascii=False, indent=2)

        logger.info(f"[Gallery] Saved to {self.gallery_dir}")

    def register_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        face_feature: Optional[np.ndarray] = None,
        fingerprint_feature: Optional[np.ndarray] = None,
        face_image_path: Optional[str] = None,
        fingerprint_image_path: Optional[str] = None,
    ) -> dict:
        """注册新用户或更新已有用户

        Args:
            user_id: 用户 ID
            name: 用户姓名（前端展示用）
            face_feature: 人脸 512 维特征向量
            fingerprint_feature: 指纹 512 维特征向量
            face_image_path: 人脸图片路径（可选）
            fingerprint_image_path: 指纹图片路径（可选）

        Returns:
            {"success": bool, "is_new": bool, "user_id": str}
        """
        is_new = user_id not in self._data["users"]
        now = datetime.now().isoformat()

        if is_new:
            self._data["users"][user_id] = {
                "registered_at": now,
                "updated_at": now,
            }

        user = self._data["users"][user_id]
        user["updated_at"] = now

        if name is not None:
            user["name"] = name
        if face_feature is not None:
            user["face_feature"] = face_feature.tolist() if hasattr(face_feature, "tolist") else list(face_feature)
        if fingerprint_feature is not None:
            user["fingerprint_feature"] = fingerprint_feature.tolist() if hasattr(fingerprint_feature, "tolist") else list(fingerprint_feature)
        if face_image_path is not None:
            user["face_image_path"] = face_image_path
        if fingerprint_image_path is not None:
            user["fingerprint_image_path"] = fingerprint_image_path

        if self.auto_save:
            self._save()

        action = "registered" if is_new else "updated"
        logger.info(f"[Gallery] User {user_id} {action}")
        return {"success": True, "is_new": is_new, "user_id": user_id}

    def delete_user(self, user_id: str) -> dict:
        """从 Gallery 删除用户"""
        if user_id not in self._data["users"]:
            return {"success": False, "error": f"User {user_id} not found"}

        del self._data["users"][user_id]
        if self.auto_save:
            self._save()

        logger.info(f"[Gallery] User {user_id} deleted")
        return {"success": True, "user_id": user_id}

    def get_user(self, user_id: str) -> Optional[dict]:
        """获取指定用户信息"""
        return self._data["users"].get(user_id)

    def list_users(self) -> List[str]:
        """返回所有已注册用户 ID 列表"""
        return list(self._data["users"].keys())

    def count_users(self) -> int:
        """返回已注册用户数量"""
        return len(self._data["users"])

    def get_all_features(
        self,
        modality: Literal["face", "fingerprint", "both"] = "face",
    ) -> tuple[list[np.ndarray], list[str]]:
        """获取所有用户的特征向量和对应 ID

        Args:
            modality: "face" | "fingerprint" | "both"
                      "both" 时返回拼接的特征向量（需确保两个模态的特征维度相同）

        Returns:
            ([features], [user_ids]) 两个等长列表
        """
        features = []
        ids = []

        for uid, udata in self._data["users"].items():
            if modality == "face":
                feat = udata.get("face_feature")
                if feat is not None and len(feat) > 0:
                    features.append(np.array(feat, dtype=np.float32))
                    ids.append(uid)
            elif modality == "fingerprint":
                feat = udata.get("fingerprint_feature")
                if feat is not None and len(feat) > 0:
                    features.append(np.array(feat, dtype=np.float32))
                    ids.append(uid)
            elif modality == "both":
                face_feat = udata.get("face_feature")
                fp_feat = udata.get("fingerprint_feature")
                if face_feat is not None and len(face_feat) > 0:
                    if fp_feat is not None and len(fp_feat) > 0:
                        combined = np.concatenate([face_feat, fp_feat], axis=0)
                        features.append(combined.astype(np.float32))
                        ids.append(uid)
                    else:
                        logger.warning(f"[Gallery] User {uid} missing fingerprint, skipping in 'both' mode")

        return features, ids

    def clear(self):
        """清空 Gallery（谨慎使用）"""
        self._data["users"].clear()
        if self.auto_save:
            self._save()
        logger.warning("[Gallery] Cleared all users")
