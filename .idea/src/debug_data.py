#!/usr/bin/env python
import os
import sys
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.utils import load_config

def debug_data_paths():
    """调试数据路径问题"""
    config = load_config("configs/face_config.yaml")

    print("=== 配置信息 ===")
    print(f"config['paths']['data_dir']: {config['paths']['data_dir']}")
    print(f"config['paths']['face_data_dir']: {config['paths'].get('face_data_dir', 'Not set')}")

    # 模拟train_face.py中的路径解析逻辑
    face_data_dir = config["paths"]["data_dir"]
    if not os.path.isabs(face_data_dir):
        # If relative path, make it relative to project root
        # 模拟从scripts/train_face.py计算project_root
        script_dir = os.path.join(os.getcwd(), 'scripts')  # 当前在项目根目录
        project_root = os.path.dirname(script_dir)  # 应该是src目录
        face_data_dir = os.path.join(project_root, face_data_dir.lstrip('./'))

    print(f"\n=== 解析后的路径 ===")
    print(f"script_dir: {script_dir}")
    print(f"project_root: {project_root}")
    print(f"face_data_dir: {face_data_dir}")
    print(f"face_data_dir exists: {os.path.exists(face_data_dir)}")

    if os.path.exists(face_data_dir):
        print(f"face_data_dir contents: {os.listdir(face_data_dir)}")

        # 检查face子目录
        face_subdir = os.path.join(face_data_dir, 'face')
        print(f"\nface subdir exists: {os.path.exists(face_subdir)}")

        if os.path.exists(face_subdir):
            print(f"face subdir contents: {os.listdir(face_subdir)}")

            # 检查face/face子目录
            face_face_dir = os.path.join(face_subdir, 'face')
            print(f"\nface/face dir exists: {os.path.exists(face_face_dir)}")

            if os.path.exists(face_face_dir):
                contents = os.listdir(face_face_dir)
                print(f"face/face dir contents: {contents[:10]}...")  # 只显示前10个

                # 检查第一个类别目录
                if contents:
                    first_class = contents[0]
                    first_class_dir = os.path.join(face_face_dir, first_class)
                    if os.path.isdir(first_class_dir):
                        class_contents = os.listdir(first_class_dir)
                        print(f"\nFirst class '{first_class}' contents: {class_contents[:5]}...")
    else:
        print("数据目录不存在！")
        print("检查以下可能的问题：")
        print("1. 当前工作目录:", os.getcwd())
        print("2. 项目结构是否正确")
        print("3. 数据文件是否在正确位置")

if __name__ == "__main__":
    debug_data_paths()