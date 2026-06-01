"""
批量注册脚本：将数据集中后 50 人（450~499）注册到 Gallery。
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # -> .idea/src/
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from PIL import Image
from inference.services.feature_service import FeatureService
from inference.gallery_manager import GalleryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("BatchRegister")

DATA_DIR = PROJECT_ROOT / "data"
FACE_DATA_DIR = DATA_DIR / "face" / "face"
FP_DATA_DIR = DATA_DIR / "CASIA-FingerprintV5(BMP)"
GALLERY_DIR = DATA_DIR / "gallery"

SRC_UIDS = list(range(450, 500))
GALLERY_ID_OFFSET = 450


def build_gallery_id(src_uid):
    return f"{src_uid - GALLERY_ID_OFFSET + 1:03d}"


def get_face_path(src_uid):
    return FACE_DATA_DIR / f"{src_uid:03d}" / f"{src_uid:03d}_0.bmp"


def get_fp_path(src_uid):
    return FP_DATA_DIR / f"{src_uid:03d}" / "L" / f"{src_uid:03d}_L0_0.bmp"


def load_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def save_image(image, gallery_id, modality, project_root):
    images_dir = project_root / "data" / "gallery" / "images" / gallery_id
    images_dir.mkdir(parents=True, exist_ok=True)
    filepath = images_dir / f"{modality}.jpg"
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(filepath, "JPEG", quality=95)
    return str(filepath.relative_to(project_root))


def main():
    logger.info("=" * 50)
    logger.info("批量注册开始")
    logger.info(f"目标用户：{SRC_UIDS[0]}~{SRC_UIDS[-1]}（共 {len(SRC_UIDS)} 人）")
    logger.info(f"Gallery ID：001~050")
    logger.info("=" * 50)

    logger.info("加载模型...")
    feature_service = FeatureService(device="auto")
    gallery = GalleryManager(gallery_dir=GALLERY_DIR)
    logger.info("模型加载完成")

    start_time = time.time()
    success = 0
    failed = []

    for idx, src_uid in enumerate(SRC_UIDS, 1):
        gallery_id = build_gallery_id(src_uid)
        face_path = get_face_path(src_uid)
        fp_path = get_fp_path(src_uid)

        if not face_path.exists():
            logger.warning(f"[{idx:03d}/{len(SRC_UIDS)}] {gallery_id} 跳过：人脸文件不存在")
            failed.append((gallery_id, "face file not found"))
            continue
        if not fp_path.exists():
            logger.warning(f"[{idx:03d}/{len(SRC_UIDS)}] {gallery_id} 跳过：指纹文件不存在")
            failed.append((gallery_id, "fingerprint file not found"))
            continue

        try:
            face_img = load_image(face_path)
            fp_img = load_image(fp_path)
            face_feat = feature_service.extract_face(face_img)
            fp_feat = feature_service.extract_fingerprint(fp_img)
            face_img_path = save_image(face_img, gallery_id, "face", PROJECT_ROOT)
            fp_img_path = save_image(fp_img, gallery_id, "fingerprint", PROJECT_ROOT)

            gallery.register_user(
                user_id=gallery_id,
                name=f"User{gallery_id}",
                face_feature=face_feat,
                fingerprint_feature=fp_feat,
                face_image_path=face_img_path,
                fingerprint_image_path=fp_img_path,
            )
            success += 1
            logger.info(f"[{idx:03d}/{len(SRC_UIDS)}] OK  {gallery_id} ({src_uid})")

        except Exception as e:
            logger.error(f"[{idx:03d}/{len(SRC_UIDS)}] FAIL {gallery_id} ({src_uid}): {e}")
            failed.append((gallery_id, str(e)))

    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"完成：{success}/{len(SRC_UIDS)} 成功，{len(failed)} 失败，耗时 {elapsed:.1f}s")
    if failed:
        for gid, reason in failed:
            logger.info(f"  {gid}: {reason}")

    logger.info(f"Gallery 总用户数：{len(gallery.list_users())}")
    logger.info(f"用户列表：{sorted(gallery.list_users())}")


if __name__ == "__main__":
    main()
