#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据路径调试脚本
检查数据目录结构和文件是否存在
"""

import os
import sys
sys.path.append('..')

from core.utils import load_config

def check_directory_structure(data_dir):
    """检查目录结构"""
    print(f"检查数据目录: {data_dir}")
    print(f"目录存在: {os.path.exists(data_dir)}")

    if not os.path.exists(data_dir):
        print("❌ 数据目录不存在！")
        return False

    # 列出目录内容
    try:
        items = os.listdir(data_dir)
        print(f"目录内容: {items}")

        # 筛选出文件夹
        subdirs = [item for item in items if os.path.isdir(os.path.join(data_dir, item))]
        print(f"子目录数量: {len(subdirs)}")
        print(f"子目录列表: {subdirs[:10]}...")  # 只显示前10个

        if len(subdirs) == 0:
            print("❌ 没有找到任何子目录（类别文件夹）")
            return False

        # 检查第一个类别文件夹
        first_class_dir = os.path.join(data_dir, subdirs[0])
        print(f"\n检查第一个类别文件夹: {subdirs[0]}")

        if os.path.exists(first_class_dir):
            class_items = os.listdir(first_class_dir)
            print(f"类别文件夹内容: {class_items}")

            # 筛选图像文件
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in class_items if f.lower().endswith(image_extensions)]
            print(f"找到的图像文件: {len(image_files)}")
            print(f"图像文件列表: {image_files[:5]}...")  # 只显示前5个

            if len(image_files) > 0:
                # 检查第一个文件的详细信息
                first_file = os.path.join(first_class_dir, image_files[0])
                print(f"第一个文件路径: {first_file}")
                print(f"文件存在: {os.path.exists(first_file)}")
                print(f"文件大小: {os.path.getsize(first_file) if os.path.exists(first_file) else 'N/A'} bytes")

        return True

    except Exception as e:
        print(f"❌ 检查目录时出错: {e}")
        return False

def main():
    print("=" * 60)
    print("数据路径调试")
    print("=" * 60)

    # 加载配置
    config = load_config('../configs/face_config.yaml')
    face_data_dir = config['paths']['face_data_dir']

    print(f"配置文件中的数据路径: {face_data_dir}")
    print(f"绝对路径: {os.path.abspath(face_data_dir)}")
    print()

    # 检查目录结构
    success = check_directory_structure(face_data_dir)

    if success:
        print("\n✅ 目录结构检查完成")
        print("如果仍有问题，可能是文件格式或权限问题")
    else:
        print("\n❌ 目录结构有问题，请检查数据路径")

    print("=" * 60)

if __name__ == "__main__":
    main()