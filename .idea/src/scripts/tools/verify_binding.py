#!/usr/bin/env python
"""
快速验证数据绑定结果
"""
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("验证数据绑定结果")
    print("=" * 60)

    # 读取映射文件
    mapping_file = Path("data/face_fingerprint_mapping.json")
    stats_file = Path("data/face_fingerprint_stats.json")

    if not mapping_file.exists():
        print("❌ 映射文件不存在，请先运行 bind_fusion_data.py")
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    print(f"\n📊 数据统计:")
    print(f"  总人数: {len(mapping)}")

    # 统计图片数量
    total_face_images = sum(len(v['face_images']) for v in mapping.values())
    total_fp_images = sum(
        len(v['fingerprint_images']['left']) + len(v['fingerprint_images']['right'])
        for v in mapping.values()
    )

    print(f"  人脸图片总数: {total_face_images}")
    print(f"  指纹图片总数: {total_fp_images}")
    print(f"  平均每人脸图片: {total_face_images // len(mapping)}")
    print(f"  平均每人指纹图片: {total_fp_images // len(mapping)}")

    # 检查前3个人
    print(f"\n📋 前3个人样例:")
    for i, (person_id, data) in enumerate(list(mapping.items())[:3]):
        print(f"  [{person_id}]")
        print(f"    人脸: {len(data['face_images'])}张")
        print(f"    指纹: {len(data['fingerprint_images']['left'])}左 + {len(data['fingerprint_images']['right'])}右")

    # 验证ID范围
    face_ids = [k for k in mapping.keys()]
    fp_ids = [v['fingerprint_id'] for v in mapping.values()]

    print(f"\n🔢 ID范围:")
    print(f"  人脸ID: {face_ids[0]} - {face_ids[-1]}")
    print(f"  指纹ID: {fp_ids[0]} - {fp_ids[-1]}")

    # 检查是否有缺失
    face_id_set = set(face_ids)
    expected_face_ids = set(f"{i:03d}" for i in range(300))
    missing = expected_face_ids - face_id_set

    if missing:
        print(f"\n⚠️ 缺少的人脸ID: {sorted(list(missing))[:10]}... (共{len(missing)}个)")
    else:
        print(f"\n✅ 所有300个人的数据都已绑定!")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
