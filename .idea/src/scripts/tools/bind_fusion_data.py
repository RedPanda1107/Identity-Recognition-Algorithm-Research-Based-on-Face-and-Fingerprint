#!/usr/bin/env python
"""
人脸和指纹数据ID绑定脚本 - 限定300人版本
为特征融合实验创建规范化的数据集映射
"""

import os
import json
import csv
from pathlib import Path

# ============= 配置参数 =============
MAX_PERSONS = 300  # 限制300人
FACE_DIR = "data/face/face"
FINGERPRINT_DIR = "data/fingerprint/fingerprint/organized"
OUTPUT_DIR = "data"
# ============= 配置参数 =============


def analyze_data_directories(face_dir, fingerprint_dir):
    """分析人脸和指纹数据目录结构"""
    stats = {
        'face': {'ids': [], 'total_images': 0},
        'fingerprint': {'ids': [], 'total_images': 0}
    }

    # 分析人脸数据
    face_path = Path(face_dir)
    if face_path.exists():
        for person_dir in sorted(face_path.iterdir()):
            if person_dir.is_dir() and person_dir.name.isdigit():
                person_id = person_dir.name
                # 只处理前MAX_PERSONS个人
                if int(person_id) < MAX_PERSONS:
                    stats['face']['ids'].append(person_id)
                    image_count = len([f for f in person_dir.iterdir()
                                     if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
                    stats['face']['total_images'] += image_count

    # 分析指纹数据 (001-600)
    fingerprint_path = Path(fingerprint_dir)
    if fingerprint_path.exists():
        for person_dir in sorted(fingerprint_path.iterdir()):
            if person_dir.is_dir() and person_dir.name.isdigit():
                person_id = person_dir.name
                # 只处理对应的人脸ID (face_id + 1 = fingerprint_id)
                face_num = int(person_id) - 1
                if face_num < MAX_PERSONS:
                    stats['fingerprint']['ids'].append(person_id)
                    image_count = 0
                    for hand_dir in person_dir.iterdir():
                        if hand_dir.is_dir():
                            image_count += len([f for f in hand_dir.iterdir()
                                              if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
                    stats['fingerprint']['total_images'] += image_count

    return stats


def create_unified_index(face_dir, fingerprint_dir, max_persons):
    """
    创建统一的联合数据集索引
    使用直接匹配策略：face_id == fingerprint_id - 1
    """
    unified_index = {}
    face_path = Path(face_dir)
    fingerprint_path = Path(fingerprint_dir)

    for face_id in [f"{i:03d}" for i in range(max_persons)]:
        fingerprint_id = f"{int(face_id) + 1:03d}"

        # 检查数据是否存在
        face_person_dir = face_path / face_id
        fingerprint_person_dir = fingerprint_path / fingerprint_id

        if not face_person_dir.exists() or not fingerprint_person_dir.exists():
            continue

        # 检查是否有图片
        face_images = [f for f in face_person_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']]
        if not face_images:
            continue

        fp_images = {'left': [], 'right': []}
        for hand in ['left', 'right']:
            hand_path = fingerprint_person_dir / hand
            if hand_path.exists():
                for img_file in sorted(hand_path.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in ['.bmp']:
                        fp_images[hand].append(str(img_file))

        if not fp_images['left'] and not fp_images['right']:
            continue

        person_data = {
            'person_id': face_id,
            'face_id': face_id,
            'fingerprint_id': fingerprint_id,
            'face_images': [str(f) for f in sorted(face_images)],
            'fingerprint_images': fp_images
        }

        unified_index[face_id] = person_data

    return unified_index


def generate_statistics(unified_index, stats):
    """生成统计信息"""
    stats_info = {
        'data_summary': stats,
        'mapping_summary': {
            'total_mapped_pairs': len(unified_index),
            'face_ids_mapped': len(unified_index),
            'fingerprint_ids_mapped': len(unified_index),
        },
        'unified_index_summary': {
            'total_persons': len(unified_index),
            'persons_with_face': len(unified_index),
            'persons_with_fingerprint': len(unified_index),
            'persons_with_both_modalities': len(unified_index),
        },
        'config': {
            'max_persons': MAX_PERSONS,
            'mapping_strategy': 'direct',
            'face_id_range': f"000-{MAX_PERSONS-1:03d}",
            'fingerprint_id_range': f"001-{MAX_PERSONS:03d}"
        }
    }
    return stats_info


def main():
    print("=" * 60)
    print("人脸-指纹数据绑定工具 (限定300人)")
    print("=" * 60)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] 分析数据目录...")
    stats = analyze_data_directories(FACE_DIR, FINGERPRINT_DIR)

    print(f"  人脸数据: {len(stats['face']['ids'])}人, {stats['face']['total_images']}张图片")
    print(f"  指纹数据: {len(stats['fingerprint']['ids'])}人, {stats['fingerprint']['total_images']}张图片")

    print(f"\n[2/4] 创建联合索引 (限制前{MAX_PERSONS}人)...")
    unified_index = create_unified_index(FACE_DIR, FINGERPRINT_DIR, MAX_PERSONS)
    print(f"  成功配对: {len(unified_index)}人")

    print(f"\n[3/4] 生成统计信息...")
    stats_info = generate_statistics(unified_index, stats)

    # 保存映射文件
    mapping_file = output_dir / "face_fingerprint_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(unified_index, f, indent=2, ensure_ascii=False)

    # 保存统计信息
    stats_file = output_dir / "face_fingerprint_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_info, f, indent=2, ensure_ascii=False)

    # 生成CSV格式的映射表
    csv_file = output_dir / "face_fingerprint_mapping.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['person_id', 'face_id', 'fingerprint_id'])
        writer.writeheader()
        for person_id, data in unified_index.items():
            writer.writerow({
                'person_id': person_id,
                'face_id': data['face_id'],
                'fingerprint_id': data['fingerprint_id']
            })

    print(f"\n[4/4] 保存输出文件...")
    print(f"  映射表: {mapping_file}")
    print(f"  统计信息: {stats_file}")
    print(f"  CSV映射表: {csv_file}")

    print("\n" + "=" * 60)
    print("绑定完成!")
    print(f"  总人数: {len(unified_index)}")
    print(f"  每人脸图片数: {sum(len(v['face_images']) for v in unified_index.values()) // len(unified_index)}")
    print(f"  指纹图片数: {sum(len(v['fingerprint_images']['left']) + len(v['fingerprint_images']['right']) for v in unified_index.values()) // len(unified_index)}")
    print("=" * 60)

    return unified_index


if __name__ == "__main__":
    main()
