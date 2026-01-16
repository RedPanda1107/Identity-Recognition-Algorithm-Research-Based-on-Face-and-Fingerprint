#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试YAML配置文件解析
"""

import yaml
import os

def test_yaml_parsing():
    print("测试YAML配置文件解析...")

    config_file = '../configs/face_config.yaml'

    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print("YAML解析成功")
        print(f"配置类型: {type(config)}")

        if 'paths' in config:
            print("paths 部分存在")
            paths = config['paths']
            print(f"paths 内容: {paths}")

            face_data_dir = paths.get('face_data_dir')
            print(f"face_data_dir 值: {repr(face_data_dir)}")
            print(f"face_data_dir 类型: {type(face_data_dir)}")

            if face_data_dir:
                print(f"face_data_dir 存在: '{face_data_dir}'")
            else:
                print("face_data_dir 不存在或为空")
        else:
            print("paths 部分不存在")

    except Exception as e:
        print(f"YAML解析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yaml_parsing()