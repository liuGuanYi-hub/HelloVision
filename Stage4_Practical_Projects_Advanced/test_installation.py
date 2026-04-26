#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 4 环境验证脚本

检查所有依赖包是否正确安装
"""

import sys
import subprocess

def test_package(package_name, import_name=None):
    """测试包是否安装成功"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} 安装成功")
        return True
    except ImportError as e:
        print(f"✗ {package_name} 安装失败：{str(e)[:50]}")
        return False

def get_version(package_name):
    """获取包的版本号"""
    try:
        module = sys.modules.get(package_name)
        if module and hasattr(module, '__version__'):
            return module.__version__
        return "未知"
    except:
        return "未知"

def main():
    print("=" * 60)
    print("Stage 4: 实战项目与进阶应用 - 环境验证")
    print("=" * 60)
    print()
    
    # Python 版本检查
    print(f"Python 版本：{sys.version}")
    print(f"Python 路径：{sys.executable}")
    print()
    
    # 核心包检查
    print("-" * 60)
    print("核心深度学习框架")
    print("-" * 60)
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(test_package(pkg_name, import_name))
    
    # 图像处理包
    print("\n" + "-" * 60)
    print("图像处理库")
    print("-" * 60)
    image_packages = [
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("imageio", "imageio"),
    ]
    
    for pkg_name, import_name in image_packages:
        results.append(test_package(import_name, pkg_name))
    
    # 数据科学包
    print("\n" + "-" * 60)
    print("数据科学和可视化")
    print("-" * 60)
    data_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
    ]
    
    for pkg_name, import_name in data_packages:
        results.append(test_package(import_name, pkg_name))
    
    # 高级 CV 包
    print("\n" + "-" * 60)
    print("高级计算机视觉库")
    print("-" * 60)
    cv_packages = [
        ("ultralytics", "ultralytics"),
        ("albumentations", "albumentations"),
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ]
    
    for pkg_name, import_name in cv_packages:
        results.append(test_package(import_name, pkg_name))
    
    # 工具包
    print("\n" + "-" * 60)
    print("工具库")
    print("-" * 60)
    tool_packages = [
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard"),
    ]
    
    for pkg_name, import_name in tool_packages:
        results.append(test_package(import_name, pkg_name))
    
    # 总结
    print("\n" + "=" * 60)
    print(f"总计：{sum(results)}/{len(results)} 个包安装成功")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 所有包安装成功！环境配置完成！")
        
        # 检查 GPU 支持
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n🎮 GPU 可用：{torch.cuda.get_device_name(0)}")
                print(f"   CUDA 版本：{torch.version.cuda}")
                print(f"   GPU 内存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("\n💻 未检测到 GPU，将使用 CPU 运行")
                print("   提示：如有 NVIDIA 显卡，可安装 CUDA 版本加速")
        except:
            pass
        
        print("\n" + "=" * 60)
        print("下一步:")
        print("1. cd 4.1_图像分类实战")
        print("2. 查看 学习笔记.md")
        print("3. 准备数据集 (参考 datasets/README.md)")
        print("4. 运行 python Practice.py")
        print("=" * 60)
        
    else:
        print("\n❌ 部分包安装失败，请检查错误信息")
        print("\n建议:")
        print("1. 查看 environment_setup.md 了解详细安装步骤")
        print("2. 使用国内镜像源加速安装:")
        print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)

if __name__ == "__main__":
    main()
