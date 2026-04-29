"""
生成人脸-CASIA指纹配对映射文件
自动扫描 face/face 和 CASIA-FingerprintV5(BMP) 目录并生成配对
"""
import os
import json
from pathlib import Path

# 配置路径 (相对于脚本所在目录的父目录)
script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
face_dir = data_dir / "face" / "face"
casia_fp_dir = data_dir / "CASIA-FingerprintV5(BMP)"
output_file = data_dir / "face_casia_mapping.json"

def generate_mapping():
    mapping = {}

    # 获取所有人员ID (人脸目录名)
    if not face_dir.exists():
        print(f"错误: 人脸目录不存在: {face_dir}")
        return

    person_ids = sorted([d.name for d in face_dir.iterdir() if d.is_dir()])
    print(f"找到 {len(person_ids)} 个人员目录")

    for person_id in person_ids:
        face_person_dir = face_dir / person_id
        casia_person_dir = casia_fp_dir / person_id

        # 收集人脸图像
        face_images = []
        if face_person_dir.exists():
            for img in sorted(face_person_dir.glob("*.bmp")):
                # 路径相对于 data 目录
                rel_path = f"face/face/{person_id}/{img.name}"
                face_images.append(rel_path.replace('/', '\\'))

        # 收集CASIA指纹图像 (L=左手, R=右手)
        fingerprint_images = {"left": [], "right": []}

        if casia_person_dir.exists():
            left_dir = casia_person_dir / "L"
            right_dir = casia_person_dir / "R"

            if left_dir.exists():
                for img in sorted(left_dir.glob("*.bmp")):
                    rel_path = f"CASIA-FingerprintV5(BMP)/{person_id}/L/{img.name}"
                    fingerprint_images["left"].append(rel_path.replace('/', '\\'))

            if right_dir.exists():
                for img in sorted(right_dir.glob("*.bmp")):
                    rel_path = f"CASIA-FingerprintV5(BMP)/{person_id}/R/{img.name}"
                    fingerprint_images["right"].append(rel_path.replace('/', '\\'))

        # 只添加有人脸和指纹的记录
        if face_images and (fingerprint_images["left"] or fingerprint_images["right"]):
            mapping[person_id] = {
                "person_id": person_id,
                "face_id": person_id,
                "fingerprint_id": person_id,  # CASIA指纹ID与人脸ID一致
                "face_images": face_images,
                "fingerprint_images": fingerprint_images
            }

    # 保存映射文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"映射文件已生成: {output_file}")
    print(f"共 {len(mapping)} 个有效配对")

    # 统计信息
    total_face = sum(len(m["face_images"]) for m in mapping.values())
    total_fp = sum(
        len(m["fingerprint_images"]["left"]) + len(m["fingerprint_images"]["right"])
        for m in mapping.values()
    )
    print(f"人脸图像总数: {total_face}")
    print(f"指纹图像总数: {total_fp}")

    return mapping

if __name__ == "__main__":
    generate_mapping()
