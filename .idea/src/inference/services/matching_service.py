"""
特征匹配服务 - 基于余弦相似度的 1:N 检索
"""

import logging
from typing import List, Dict, Any, Literal

import numpy as np

logger = logging.getLogger("MatchingService")


class MatchingService:
    """余弦相似度特征匹配服务

    支持 1:N 识别模式，计算查询向量与 Gallery 中所有向量的相似度，
    返回 Top-K 排序结果。
    """

    def __init__(self):
        self._gallery_features: List[np.ndarray] = []
        self._gallery_ids: List[str] = []

    def set_gallery(
        self,
        features: List[np.ndarray],
        user_ids: List[str],
    ):
        """设置 Gallery 特征库

        Args:
            features: 特征向量列表 [N, D]
            user_ids: 对应的用户 ID 列表 [N]
        """
        if len(features) != len(user_ids):
            raise ValueError("features 和 user_ids 长度必须一致")

        self._gallery_features = [np.asarray(f, dtype=np.float32) for f in features]
        self._gallery_ids = list(user_ids)
        logger.info(f"[Matching] Gallery set: {len(self._gallery_ids)} entries")

    def match(
        self,
        query_feature: np.ndarray,
        top_k: int = 5,
        modality: Literal["face", "fingerprint", "fusion"] = "fusion",
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """将查询特征与 Gallery 匹配，返回 Top-K 结果

        Args:
            query_feature: 查询特征向量 [D] 或 [1, D]
            top_k: 返回前 k 个候选
            modality: 模态标识（用于日志）
            score_threshold: 最低相似度阈值

        Returns:
            [
                {"user_id": "001", "confidence": 0.95, "rank": 1},
                {"user_id": "042", "confidence": 0.72, "rank": 2},
                ...
            ]
        """
        if not self._gallery_features:
            logger.warning("[Matching] Gallery is empty")
            return []

        query = np.asarray(query_feature, dtype=np.float32).flatten()

        scores = []
        for gallery_feat in self._gallery_features:
            feat = gallery_feat.flatten()
            dot = np.dot(query, feat)
            norm_q = np.linalg.norm(query)
            norm_g = np.linalg.norm(feat)
            if norm_q > 0 and norm_g > 0:
                score = float(dot / (norm_q * norm_g))
            else:
                score = 0.0
            scores.append(score)

        scores = np.array(scores)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            conf = float(scores[idx])
            if conf >= score_threshold:
                results.append({
                    "user_id": self._gallery_ids[idx],
                    "confidence": round(conf, 4),
                    "rank": rank,
                    "modality": modality,
                })

        logger.debug(f"[Matching] Top-1 confidence: {results[0]['confidence'] if results else 'N/A'}")
        return results

    def match_multi_modal(
        self,
        face_feature: np.ndarray,
        fp_feature: np.ndarray,
        fused_feature: np.ndarray,
        gallery_face: List[np.ndarray],
        gallery_fp: List[np.ndarray],
        gallery_ids: List[str],
        fusion_weights: tuple[float, float] = (0.5, 0.5),
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """多模态融合匹配（独立计算人脸和指纹相似度后加权）

        Args:
            face_feature: 查询人脸特征
            fp_feature: 查询指纹特征
            fused_feature: 查询融合特征
            gallery_face: Gallery 人脸特征列表
            gallery_fp: Gallery 指纹特征列表
            gallery_ids: Gallery 用户 ID
            fusion_weights: (人脸权重, 指纹权重)，默认 (0.5, 0.5)
            top_k: 返回前 k 个候选
            score_threshold: 最低相似度阈值

        Returns:
            [
                {
                    "user_id": "001",
                    "confidence": 0.97,
                    "face_confidence": 0.95,
                    "fingerprint_confidence": 0.92,
                    "rank": 1,
                },
                ...
            ]
        """
        if len(gallery_face) != len(gallery_fp) or len(gallery_face) != len(gallery_ids):
            raise ValueError("Gallery 特征与 ID 列表长度不一致")

        face_w, fp_w = fusion_weights
        face_w = float(face_w)
        fp_w = float(fp_w)

        query_face = np.asarray(face_feature, dtype=np.float32).flatten()
        query_fp = np.asarray(fp_feature, dtype=np.float32).flatten()

        face_scores = []
        fp_scores = []

        for g_face, g_fp in zip(gallery_face, gallery_fp):
            g_face = np.asarray(g_face, dtype=np.float32).flatten()
            g_fp = np.asarray(g_fp, dtype=np.float32).flatten()

            def cos(a, b):
                n = np.dot(a, b)
                d = (np.linalg.norm(a) * np.linalg.norm(b))
                return float(n / d) if d > 0 else 0.0

            face_scores.append(cos(query_face, g_face))
            fp_scores.append(cos(query_fp, g_fp))

        face_scores = np.array(face_scores)
        fp_scores = np.array(fp_scores)
        combined_scores = face_w * face_scores + fp_w * fp_scores

        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            conf = float(combined_scores[idx])
            if conf >= score_threshold:
                results.append({
                    "user_id": gallery_ids[idx],
                    "confidence": round(conf, 4),
                    "face_confidence": round(float(face_scores[idx]), 4),
                    "fingerprint_confidence": round(float(fp_scores[idx]), 4),
                    "rank": rank,
                })

        return results

    def clear_gallery(self):
        """清空 Gallery"""
        self._gallery_features.clear()
        self._gallery_ids.clear()
        logger.info("[Matching] Gallery cleared")
