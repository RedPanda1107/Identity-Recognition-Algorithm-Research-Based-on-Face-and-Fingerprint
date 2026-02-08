#!/usr/bin/env python
"""
人脸和指纹数据ID绑定脚本
创建人脸和指纹数据的ID映射关系，支持特征融合
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict

def analyze_data_directories(face_dir, fingerprint_dir):
    """
    分析人脸和指纹数据目录结构

    Returns:
        dict: 包含人脸和指纹数据的统计信息
    """
    stats = {
        'face': {'ids': [], 'total_images': 0},
        'fingerprint': {'ids': [], 'total_images': 0}
    }

    # 分析人脸数据
    face_path = Path(face_dir)
    if face_path.exists():
        for person_dir in sorted(face_path.iterdir()):
            if person_dir.is_dir():
                person_id = person_dir.name
                stats['face']['ids'].append(person_id)

                # 统计该人的图片数量
                image_count = len([f for f in person_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
                stats['face']['total_images'] += image_count

    # 分析指纹数据
    fingerprint_path = Path(fingerprint_dir)
    if fingerprint_path.exists():
        for person_dir in sorted(fingerprint_path.iterdir()):
            if person_dir.is_dir():
                person_id = person_dir.name
                stats['fingerprint']['ids'].append(person_id)

                # 统计该人的图片数量
                image_count = 0
                for hand_dir in person_dir.iterdir():
                    if hand_dir.is_dir():
                        image_count += len([f for f in hand_dir.iterdir()
                                          if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
                stats['fingerprint']['total_images'] += image_count

    return stats

def create_id_mapping(face_ids, fingerprint_ids, mapping_strategy='offset'):
    """
    创建人脸和指纹的ID映射关系

    Args:
        face_ids: 人脸数据ID列表
        fingerprint_ids: 指纹数据ID列表
        mapping_strategy: 映射策略 ('offset' 或 'direct')

    Returns:
        dict: ID映射关系
    """
    mapping = {}

    if mapping_strategy == 'offset':
        # 人脸ID + 1 = 指纹ID (人脸000 -> 指纹001)
        for face_id in face_ids:
            face_num = int(face_id)
            fingerprint_id = f"{face_num + 1:03d}"

            if fingerprint_id in fingerprint_ids:
                mapping[face_id] = fingerprint_id

    elif mapping_strategy == 'direct':
        # 直接匹配相同ID
        face_set = set(face_ids)
        fingerprint_set = set(fingerprint_ids)
        common_ids = face_set.intersection(fingerprint_set)

        for common_id in common_ids:
            mapping[common_id] = common_id

    return mapping

def create_unified_index(face_dir, fingerprint_dir, mapping, output_file=None):
    """
    创建统一的联合数据集索引

    Args:
        face_dir: 人脸数据目录
        fingerprint_dir: 指纹数据目录
        mapping: ID映射关系
        output_file: 输出文件路径

    Returns:
        dict: 联合数据集索引
    """
    unified_index = {}

    face_path = Path(face_dir)
    fingerprint_path = Path(fingerprint_dir)

    for face_id, fingerprint_id in mapping.items():
        person_data = {
            'person_id': face_id,
            'face_id': face_id,
            'fingerprint_id': fingerprint_id,
            'face_images': [],
            'fingerprint_images': {
                'left': [],
                'right': []
            }
        }

        # 收集人脸图片
        face_person_dir = face_path / face_id
        if face_person_dir.exists():
            for img_file in sorted(face_person_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']:
                    person_data['face_images'].append(str(img_file.relative_to(face_path.parent.parent)))

        # 收集指纹图片
        fingerprint_person_dir = fingerprint_path / fingerprint_id
        if fingerprint_person_dir.exists():
            for hand_dir in ['left', 'right']:
                hand_path = fingerprint_person_dir / hand_dir
                if hand_path.exists():
                    for img_file in sorted(hand_path.iterdir()):
                        if img_file.is_file() and img_file.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']:
                            person_data['fingerprint_images'][hand_dir].append(
                                str(img_file.relative_to(fingerprint_path.parent.parent.parent))
                            )

        unified_index[face_id] = person_data

    # 添加只有指纹没有对应人脸的数据
    face_mapped = set(mapping.values())
    for fingerprint_id in set(mapping.values()):
        if fingerprint_id not in face_mapped:
            person_data = {
                'person_id': fingerprint_id,
                'face_id': None,
                'fingerprint_id': fingerprint_id,
                'face_images': [],
                'fingerprint_images': {
                    'left': [],
                    'right': []
                }
            }

            # 收集指纹图片
            fingerprint_person_dir = fingerprint_path / fingerprint_id
            if fingerprint_person_dir.exists():
                for hand_dir in ['left', 'right']:
                    hand_path = fingerprint_person_dir / hand_dir
                    if hand_path.exists():
                        for img_file in sorted(hand_path.iterdir()):
                            if img_file.is_file() and img_file.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']:
                                person_data['fingerprint_images'][hand_dir].append(
                                    str(img_file.relative_to(fingerprint_path.parent.parent.parent))
                                )

            unified_index[fingerprint_id] = person_data

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unified_index, f, indent=2, ensure_ascii=False)

    return unified_index

def generate_statistics(stats, mapping, unified_index):
    """生成统计信息"""
    stats_info = {
        'data_summary': stats,
        'mapping_summary': {
            'total_mapped_pairs': len(mapping),
            'face_ids_mapped': len(set(mapping.keys())),
            'fingerprint_ids_mapped': len(set(mapping.values())),
        },
        'unified_index_summary': {
            'total_persons': len(unified_index),
            'persons_with_face': sum(1 for p in unified_index.values() if p['face_id']),
            'persons_with_fingerprint': sum(1 for p in unified_index.values() if p['fingerprint_id']),
            'persons_with_both_modalities': sum(1 for p in unified_index.values()
                                              if p['face_id'] and p['fingerprint_id']),
        }
    }

    return stats_info

def main():
    import argparse

    parser = argparse.ArgumentParser(description="绑定人脸和指纹数据ID")
    parser.add_argument("--face-dir", "-f",
                       default="data/face/face",
                       help="人脸数据目录")
    parser.add_argument("--fingerprint-dir", "-p",
                       default="data/fingerprint/fingerprint/organized",
                       help="指纹数据目录")
    parser.add_argument("--mapping-strategy", "-m",
                       choices=['offset', 'direct'],
                       default='offset',
                       help="ID映射策略 (offset: 人脸ID+1=指纹ID, direct: 直接匹配)")
    parser.add_argument("--output-dir", "-o",
                       default="data",
                       help="输出目录")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始分析数据目录...")
    stats = analyze_data_directories(args.face_dir, args.fingerprint_dir)

    print("人脸数据统计:")
    print(f"  - ID范围: {stats['face']['ids'][:5]}...{stats['face']['ids'][-5:]} (共{len(stats['face']['ids'])}个)")
    print(f"  - 总图片数: {stats['face']['total_images']}")

    print("指纹数据统计:")
    print(f"  - ID范围: {stats['fingerprint']['ids'][:5]}...{stats['fingerprint']['ids'][-5:]} (共{len(stats['fingerprint']['ids'])}个)")
    print(f"  - 总图片数: {stats['fingerprint']['total_images']}")

    print(f"\n使用映射策略: {args.mapping_strategy}")
    mapping = create_id_mapping(stats['face']['ids'], stats['fingerprint']['ids'], args.mapping_strategy)

    print(f"成功映射 {len(mapping)} 对人脸-指纹数据")

    # 创建联合索引
    mapping_file = output_dir / "face_fingerprint_mapping.json"
    unified_index = create_unified_index(args.face_dir, args.fingerprint_dir, mapping, mapping_file)

    # 生成统计信息
    stats_info = generate_statistics(stats, mapping, unified_index)
    stats_file = output_dir / "face_fingerprint_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_info, f, indent=2, ensure_ascii=False)

    # 生成CSV格式的映射表
    csv_file = output_dir / "face_fingerprint_mapping.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['face_id', 'fingerprint_id'])
        writer.writeheader()
        for face_id, fingerprint_id in mapping.items():
            writer.writerow({'face_id': face_id, 'fingerprint_id': fingerprint_id})

    print(f"\n输出文件:")
    print(f"  - 映射表: {mapping_file}")
    print(f"  - 统计信息: {stats_file}")
    print(f"  - CSV映射表: {csv_file}")

    print("\n联合索引统计:")
    print(f"  - 总人数: {stats_info['unified_index_summary']['total_persons']}")
    print(f"  - 有两人脸数据: {stats_info['unified_index_summary']['persons_with_face']}")
    print(f"  - 有指纹数据: {stats_info['unified_index_summary']['persons_with_fingerprint']}")
    print(f"  - 同时有两种数据: {stats_info['unified_index_summary']['persons_with_both_modalities']}")

    print("\nID绑定完成！")

if __name__ == "__main__":
    main()
