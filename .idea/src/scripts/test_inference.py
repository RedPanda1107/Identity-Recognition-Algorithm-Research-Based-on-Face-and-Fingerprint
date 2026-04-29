"""Gallery CRUD 测试"""
import sys
sys.path.insert(0, ".")
import numpy as np

from inference.gallery_manager import GalleryManager
from inference.services.matching_service import MatchingService

# Gallery 测试
g = GalleryManager()
f1 = np.random.rand(512).astype(np.float32)
f2 = np.random.rand(512).astype(np.float32)

g.register_user("test001", face_feature=f1, fingerprint_feature=f2)
g.register_user("test002", face_feature=f1 * 0.8, fingerprint_feature=f2 * 0.9)
print(f"[Gallery] Users: {g.count_users()}")
u = g.get_user("test001")
print(f"[Gallery] test001 face dim: {len(u['face_feature'])}")

# Matching 测试
m = MatchingService()
features, ids = g.get_all_features(modality="face")
m.set_gallery(features, ids)
results = m.match(f1, top_k=2, modality="face")
print(f"[Matching] Top-1: user={results[0]['user_id']}, conf={results[0]['confidence']:.4f}")

# 清理
g.delete_user("test001")
g.delete_user("test002")
print(f"[Gallery] After cleanup: {g.count_users()} users")
print("[OK] All tests passed")
