# 🛠️ Stage 4 环境配置指南

> 详细的环境配置步骤，从零开始搭建完整的开发环境！

---

## 📋 目录

1. [系统要求](#-系统要求)
2. [Python 环境配置](#python-环境配置)
3. [CUDA 和 GPU 支持](#cuda-和-gpu-支持可选)
4. [安装依赖包](#安装依赖包)
5. [验证安装](#验证安装)
6. [常见问题解决](#常见问题解决)

---

## 💻 系统要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **内存** | 8GB RAM | 16GB RAM 或更高 |
| **GPU** | 无（仅 CPU） | NVIDIA GTX 1060 6GB+ |
| **存储** | 10GB 可用空间 | 50GB+ SSD |
| **CPU** | 4 核心 | 8 核心+ |

### 软件要求

- **操作系统**: Windows 10/11, macOS 10.15+, 或 Linux
- **Python**: 3.8 或更高版本（推荐 3.10）
- **Git**: 用于代码版本管理

---

## 🐍 Python 环境配置

### 步骤 1: 安装 Python

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 Python 3.10.x（推荐）
3. 安装时勾选 **"Add Python to PATH"**

### 步骤 2: 创建虚拟环境（强烈推荐）

```bash
# 进入 Stage4 目录
cd Stage4_Practical_Projects_Advanced

# 创建虚拟环境
python -m venv venv

# Windows 激活虚拟环境
.\venv\Scripts\activate

# macOS/Linux 激活虚拟环境
source venv/bin/activate
```

### 步骤 3: 验证 Python 环境

```bash
python --version  # 应该显示 Python 3.8+
pip --version     # 应该显示 pip 版本
```

---

## 🎮 CUDA 和 GPU 支持（可选）

### 如果你有 NVIDIA GPU

#### 步骤 1: 检查 GPU 支持

```bash
# Windows: 打开任务管理器 -> 性能 -> GPU
# 或运行
nvidia-smi
```

#### 步骤 2: 安装 CUDA Toolkit

1. 访问 [NVIDIA CUDA 下载](https://developer.nvidia.com/cuda-downloads)
2. 选择你的操作系统和 GPU 型号
3. 下载并安装 CUDA 11.8 或 12.x

#### 步骤 3: 安装 cuDNN

1. 访问 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. 注册并下载对应 CUDA 版本的 cuDNN
3. 解压并复制到 CUDA 安装目录

#### 步骤 4: 安装 PyTorch GPU 版本

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果不确定，使用官方安装命令
pip install torch torchvision torchaudio
```

### 如果你没有 GPU

使用 CPU 版本即可：

```bash
pip install torch torchvision torchaudio
```

---

## 📦 安装依赖包

### 方式 1: 一键安装所有依赖（推荐）

```bash
# 确保在 Stage4 目录下
cd Stage4_Practical_Projects_Advanced

# 安装所有依赖
pip install -r requirements.txt
```

### 方式 2: 分步安装（如果遇到问题）

```bash
# 1. 核心框架
pip install torch torchvision torchaudio

# 2. 图像处理
pip install opencv-python Pillow imageio scikit-image

# 3. 数据增强
pip install albumentations

# 4. 目标检测
pip install ultralytics

# 5. 图像分割
pip install segmentation-models-pytorch

# 6. 数据科学和可视化
pip install numpy pandas matplotlib seaborn scikit-learn

# 7. 训练监控
pip install tensorboard tqdm

# 8. 开发工具
pip install jupyter jupyterlab ipywidgets
```

### 方式 3: 按需安装（最小化安装）

```bash
# 仅安装 4.1 图像分类需要的包
pip install torch torchvision opencv-python matplotlib scikit-learn tqdm

# 仅安装 4.2 目标检测需要的包
pip install torch torchvision ultralytics opencv-python tqdm

# 仅安装 4.3 图像分割需要的包
pip install torch torchvision segmentation-models-pytorch opencv-python matplotlib
```

---

## ✅ 验证安装

### 创建验证脚本

创建文件 `test_installation.py`:

```python
import sys

def test_package(package_name, import_name=None):
    """测试包是否安装成功"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} 安装成功")
        return True
    except ImportError as e:
        print(f"✗ {package_name} 安装失败：{e}")
        return False

def main():
    print("=" * 50)
    print("Stage 4 环境验证")
    print("=" * 50)
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("ultralytics", "ultralytics"),
        ("albumentations", "albumentations"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(test_package(pkg_name, import_name))
    
    print("=" * 50)
    print(f"总计：{sum(results)}/{len(results)} 个包安装成功")
    
    if all(results):
        print("🎉 所有包安装成功！环境配置完成！")
        
        # 检查 GPU 支持
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU 可用：{torch.cuda.get_device_name(0)}")
        else:
            print("💻 未检测到 GPU，将使用 CPU 运行")
    else:
        print("❌ 部分包安装失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 运行验证

```bash
python test_installation.py
```

### 预期输出

```
==================================================
Stage 4 环境验证
==================================================
✓ torch 安装成功
✓ torchvision 安装成功
✓ opencv-python 安装成功
✓ Pillow 安装成功
✓ numpy 安装成功
✓ matplotlib 安装成功
✓ pandas 安装成功
✓ scikit-learn 安装成功
✓ ultralytics 安装成功
✓ albumentations 安装成功
==================================================
总计：10/10 个包安装成功
🎉 所有包安装成功！环境配置完成！
🎮 GPU 可用：NVIDIA GeForce GTX 1060
```

---

## 🔧 常见问题解决

### 问题 1: pip 安装速度慢

**解决方案**: 使用国内镜像

```bash
# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用阿里镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题 2: torch 安装失败

**可能原因**: Python 版本不兼容

**解决方案**:
```bash
# 检查 Python 版本
python --version

# 如果版本过低，升级 Python
# 然后重新创建虚拟环境并安装
```

### 问题 3: CUDA 相关错误

**症状**: `CUDA out of memory` 或 `CUDA not available`

**解决方案**:
```python
import torch

# 检查 CUDA 是否可用
print(f"CUDA 可用：{torch.cuda.is_available()}")

# 如果不可用，检查：
# 1. NVIDIA 驱动是否安装
# 2. CUDA Toolkit 版本是否匹配
# 3. 是否安装了正确的 PyTorch GPU 版本
```

### 问题 4: OpenCV 导入错误

**症状**: `ImportError: DLL load failed`

**解决方案**:
```bash
# 卸载后重新安装
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### 问题 5: 虚拟环境问题

**症状**: 激活虚拟环境后仍然使用系统 Python

**解决方案**:
```bash
# Windows
.\venv\Scripts\activate

# 验证虚拟环境
where python  # Windows
which python  # macOS/Linux

# 应该指向 venv 目录
```

---

## 📊 环境检查清单

在开始学习前，请确保：

- [ ] Python 3.8+ 已安装
- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖包已安装
- [ ] 运行 `test_installation.py` 无错误
- [ ] （可选）GPU 支持已配置
- [ ] 预留足够磁盘空间（建议 50GB+）

---

## 🎯 下一步

环境配置完成后，开始学习：

1. [4.1 图像分类实战](4.1_图像分类实战/README.md)
2. [4.2 目标检测实战](4.2_目标检测实战/README.md)
3. [4.3 图像分割实战](4.3_图像分割实战/README.md)

---

**最后更新**: 2026-04-26
