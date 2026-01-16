#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试FaceDataset
"""

import os
import sys
sys.path.append('..')

def test_face_dataset():
    print("导入FaceDataset...")
    try:
        from core.datasets import FaceDataset
        print("Import successful")
    except ImportError as e:
        print(f"Import failed: {e}")
        return

    data_dir = '../data/face/face'
    print(f"Test data directory: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")

    if not os.path.exists(data_dir):
        print("Data directory does not exist")
        return

    try:
        print("Creating FaceDataset...")
        dataset = FaceDataset(data_dir=data_dir, mode='train', image_size=224)
        print("FaceDataset created successfully")
        print(f"Number of classes: {len(dataset.class_to_idx)}")
        print(f"Number of images: {len(dataset.image_paths)}")

        if len(dataset.image_paths) > 0:
            print("Images found")
            print(f"First image path: {dataset.image_paths[0]}")

            # 测试__getitem__
            try:
                sample = dataset[0]
                print("getitem works")
                print(f"Sample type: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"Sample keys: {list(sample.keys())}")
                else:
                    print(f"Sample: {sample}")
            except Exception as e:
                print(f"getitem failed: {e}")
        else:
            print("No images found")

    except Exception as e:
        print(f"FaceDataset creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_dataset()