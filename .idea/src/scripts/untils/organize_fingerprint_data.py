#!/usr/bin/env python
"""
指纹数据分类整理脚本
按照人编号和左右手对指纹数据进行分类整理
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict

def parse_fingerprint_filename(filename):
    """
    解析指纹文件名
    格式: [person_id]__[gender]_[hand]_[finger]_finger.BMP
    示例: 1__M_Left_index_finger.BMP
    """
    # 移除.BMP扩展名
    name = filename.replace('.BMP', '').replace('.bmp', '')

    # 使用正则表达式解析
    pattern = r'^(\d+)__([MF])_(Left|Right)_(\w+)_finger$'
    match = re.match(pattern, name)

    if not match:
        return None

    person_id, gender, hand, finger = match.groups()
    return {
        'person_id': person_id.zfill(3),  # 补齐3位
        'gender': gender,
        'hand': hand.lower(),  # 转换为小写
        'finger': finger.lower(),
        'original_name': filename
    }

def organize_fingerprint_data(source_dir, target_dir=None, rename_files=True):
    """
    整理指纹数据

    Args:
        source_dir: 源数据目录
        target_dir: 目标目录，默认为source_dir/organized
        rename_files: 是否重命名文件为简化格式
    """
    if target_dir is None:
        target_dir = os.path.join(source_dir, 'organized')

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = defaultdict(int)

    # 遍历所有BMP文件
    for bmp_file in source_path.glob('*.BMP'):
        parsed = parse_fingerprint_filename(bmp_file.name)

        if not parsed:
            print(f"跳过无法解析的文件: {bmp_file.name}")
            stats['skipped'] += 1
            continue

        # 创建目标目录结构
        person_dir = target_path / parsed['person_id']
        hand_dir = person_dir / parsed['hand']
        hand_dir.mkdir(parents=True, exist_ok=True)

        # 确定目标文件名
        if rename_files:
            # 简化为: finger_type.BMP
            target_filename = f"{parsed['finger']}_finger.BMP"
        else:
            # 保持原名
            target_filename = parsed['original_name']

        target_file = hand_dir / target_filename

        # 复制文件
        shutil.copy2(bmp_file, target_file)

        stats['processed'] += 1
        print(f"处理: {parsed['person_id']} -> {parsed['hand']}/{target_filename}")

    # 打印统计信息
    print("\n整理完成！")
    print(f"处理文件数: {stats['processed']}")
    print(f"跳过文件数: {stats['skipped']}")
    print(f"目标目录: {target_path}")

    return stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description="整理指纹数据")
    parser.add_argument("--source", "-s",
                       default="data/fingerprint/fingerprint",
                       help="源数据目录")
    parser.add_argument("--target", "-t",
                       help="目标目录（默认在源目录下创建organized文件夹）")
    parser.add_argument("--keep-names", action="store_true",
                       help="保持原始文件名而不是简化")

    args = parser.parse_args()

    source_dir = args.source
    target_dir = args.target
    rename_files = not args.keep_names

    if not os.path.exists(source_dir):
        print(f"错误：源目录不存在: {source_dir}")
        return

    print(f"开始整理指纹数据...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir or '自动创建'}")
    print(f"文件重命名: {'是' if rename_files else '否'}")

    try:
        stats = organize_fingerprint_data(source_dir, target_dir, rename_files)
        print("整理成功完成！")
    except Exception as e:
        print(f"整理过程中出错: {e}")

if __name__ == "__main__":
    main()