"""
统一的生物特征图像预处理工具。

所有预处理参数（normalize mean/std, resize size, CLAHE 参数等）集中管理，
确保训练、验证、推理使用完全一致的预处理流程。

原则：
  - 人脸：ImageNet 归一化（无 CLAHE）
  - 指纹：ImageNet 归一化 + CLAHE 增强（训练和推理均启用）
"""

import cv2
import numpy as np
from PIL import Image


# ── 归一化参数（ImageNet 标准）───────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 指纹归一化：统一使用 ImageNet 参数（与 FusionDataset 保持一致）
FINGERPRINT_MEAN = IMAGENET_MEAN
FINGERPRINT_STD = IMAGENET_STD

# ── CLAHE 参数 ───────────────────────────────────────────────────────────────

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


def get_clahe(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
    """创建 CLAHE 对象（每次调用返回新实例，避免状态问题）。"""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)


def apply_clahe_to_image(img: Image.Image) -> Image.Image:
    """对 PIL Image 应用 CLAHE 增强。

    适用于指纹图像：增强脊线对比度，改善识别效果。
    输入：RGB PIL Image
    输出：RGB PIL Image
    """
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = get_clahe()
    enhanced = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def preprocess_fingerprint(img: Image.Image, apply_clahe: bool = True) -> Image.Image:
    """指纹图像预处理流水线。

    流程：灰度转换 → CLAHE 增强 → RGB 合并 → ImageNet 归一化（在 transforms.Normalize 中完成）
    该函数仅返回处理后的 PIL Image，调用方负责 transforms.ToTensor() 和 transforms.Normalize()。

    Args:
        img: 原始 RGB PIL Image
        apply_clahe: 是否应用 CLAHE（训练=True，推理=True）

    Returns:
        处理后的 RGB PIL Image
    """
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    if apply_clahe:
        clahe = get_clahe()
        gray = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


# ── 预定义 torchvision transforms ─────────────────────────────────────────────

def get_fingerprint_transforms(augment: bool = False,
                              image_size: int = 224,
                              augment_params: dict = None) -> "transforms.Compose":
    """获取指纹图像的 transforms 流水线。

    统一使用 ImageNet 归一化，与 FusionDataset 和推理 pipeline 保持一致。
    CLAHE 在 __getitem__ 中应用（通过 preprocess_fingerprint），不在 transforms 中。
    """
    from torchvision import transforms

    if augment and augment_params:
        t = []
        aug = augment_params

        rot = aug.get("random_rotation", 0)
        if rot:
            t.append(transforms.RandomRotation(rot))

        translate = aug.get("translate", [0.0, 0.0])
        scale = aug.get("scale", [1.0, 1.0])
        if translate != [0.0, 0.0] or scale != [1.0, 1.0]:
            t.append(transforms.RandomAffine(degrees=0, translate=translate, scale=scale))

        t.append(transforms.Resize((image_size, image_size)))

        if aug.get("color_jitter", False):
            t.append(transforms.ColorJitter(
                brightness=aug.get("color_jitter_brightness", 0.2),
                contrast=aug.get("color_jitter_contrast", 0.2),
                saturation=0.0, hue=0.0
            ))

        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=FINGERPRINT_MEAN, std=FINGERPRINT_STD),
        ])

        re_prob = aug.get("random_erasing_prob", 0.0)
        if re_prob > 0:
            t.append(transforms.RandomErasing(p=re_prob))

        return transforms.Compose(t)

    # 验证/推理：无随机增强，统一 ImageNet 归一化
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=FINGERPRINT_MEAN, std=FINGERPRINT_STD),
    ])


def get_face_transforms(augment: bool = False,
                        image_size: int = 224,
                        augment_params: dict = None) -> "transforms.Compose":
    """获取人脸图像的 transforms 流水线。

    统一使用 ImageNet 归一化。
    """
    from torchvision import transforms

    if augment and augment_params:
        t = []
        aug = augment_params

        if aug.get("random_resized_crop", False):
            t.append(transforms.RandomResizedCrop(image_size))
        else:
            t.append(transforms.Resize((image_size, image_size)))

        if aug.get("random_horizontal_flip", True):
            t.append(transforms.RandomHorizontalFlip())

        rot = aug.get("random_rotation", 0)
        if rot:
            t.append(transforms.RandomRotation(rot))

        if aug.get("color_jitter", False):
            t.append(transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ))

        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        re_prob = float(aug.get("random_erasing_prob", 0.0) or 0.0)
        if re_prob > 0:
            t.append(transforms.RandomErasing(p=re_prob))

        return transforms.Compose(t)

    # 验证/推理
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
