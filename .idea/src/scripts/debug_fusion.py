"""
调试脚本：检查融合模型的维度问题
"""
import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models.fusion_model import EnhancedFusionModel
from core.models.face_net import FaceNet
from core.models.fingerprint_net import FingerprintNet


def test_fusion_model():
    print("=" * 60)
    print("维度调试脚本")
    print("=" * 60)
    
    # 测试配置
    batch_size = 4
    face_dim = 512
    fp_dim = 512
    fusion_dim = 512
    num_classes = 300
    
    # 创建设置好分类器的模型
    print("\n[1] 创建模型...")
    fusion_model = EnhancedFusionModel(
        face_embedding_dim=face_dim,
        fingerprint_embedding_dim=fp_dim,
        num_classes=num_classes,
        fusion_dim=fusion_dim,
        dropout_rate=0.5,
        fusion_method='adaptive'
    )
    
    # 设置ArcFace分类器
    from core.losses.arcface import ArcMarginProduct
    fusion_model.arc_classifier = ArcMarginProduct(
        in_features=fusion_dim,
        out_features=num_classes,
        s=40.0, m=0.3
    )
    fusion_model = fusion_model.cuda()
    fusion_model.eval()
    
    # 模拟单模态模型提取的特征
    print(f"\n[2] 模拟输入...")
    face_features = torch.randn(batch_size, face_dim).cuda()
    fp_features = torch.randn(batch_size, fp_dim).cuda()
    targets = torch.randint(0, num_classes, (batch_size,)).cuda()
    
    print(f"    face_features shape: {face_features.shape}")
    print(f"    fp_features shape: {fp_features.shape}")
    print(f"    targets shape: {targets.shape}")
    
    # 测试融合模型
    print(f"\n[3] 测试融合模型前向传播...")
    print(f"    fusion_model.modality_weighting: {type(fusion_model.modality_weighting)}")
    
    # 先测试模态投影
    print(f"\n[4] 测试模态投影...")
    try:
        face_proj, fp_proj = fusion_model._project_features(face_features, fp_features)
        print(f"    face_proj shape: {face_proj.shape}")
        print(f"    fp_proj shape: {fp_proj.shape}")
    except Exception as e:
        print(f"    ❌ 模态投影失败: {e}")
        return
    
    # 测试特征增强
    print(f"\n[5] 测试特征增强...")
    try:
        face_enhanced = fusion_model.feature_enhancer(face_proj)
        fp_enhanced = fusion_model.feature_enhancer(fp_proj)
        print(f"    face_enhanced shape: {face_enhanced.shape}")
        print(f"    fp_enhanced shape: {fp_enhanced.shape}")
    except Exception as e:
        print(f"    ❌ 特征增强失败: {e}")
        return
    
    # 测试融合
    print(f"\n[6] 测试特征融合...")
    try:
        fused = fusion_model._fuse_features(face_enhanced, fp_enhanced)
        print(f"    fused shape: {fused.shape}")
    except Exception as e:
        print(f"    ❌ 特征融合失败: {e}")
        return
    
    # 完整前向传播
    print(f"\n[7] 完整前向传播...")
    try:
        with torch.no_grad():
            outputs = fusion_model(face_features, fp_features, targets)
        print(f"    outputs type: {type(outputs)}")
        if isinstance(outputs, tuple):
            print(f"    outputs[0] shape: {outputs[0].shape}")
        else:
            print(f"    outputs shape: {outputs.shape}")
        print("    ✅ 完整前向传播成功!")
    except Exception as e:
        print(f"    ❌ 完整前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("调试完成！模型维度正确")
    print("=" * 60)


if __name__ == "__main__":
    test_fusion_model()
