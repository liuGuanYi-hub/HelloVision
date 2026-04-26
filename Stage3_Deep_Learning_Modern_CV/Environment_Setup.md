# 🔧 环境安装指南

## ⚠️ 重要提示

在开始第三阶段学习之前，需要先安装 PyTorch 深度学习框架。

---

## 📥 安装 PyTorch

### 方法 1：使用 pip（推荐）

**以管理员身份运行命令提示符或 PowerShell**，然后执行：

```bash
# GPU 版本（推荐，如果你有 NVIDIA 显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（如果没有独立显卡）
pip install torch torchvision torchaudio
```

### 方法 2：使用 conda

```bash
# 创建新环境
conda create -n pytorch python=3.10
conda activate pytorch

# 安装 PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

---

## ✅ 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

如果看到输出版本号，说明安装成功！

---

## 🎯 硬件要求

### GPU 加速（推荐）

- **NVIDIA 显卡**：GTX 1060 或更高
- **CUDA**：11.8 版本
- **显存**：至少 4GB

### CPU 版本

- **内存**：至少 8GB RAM
- **处理器**：多核 CPU 推荐

---

## 📦 其他依赖

```bash
# 安装其他必要的库
pip install numpy matplotlib opencv-python pillow jupyterlab
```

---

## 🐛 常见问题

### 问题 1：权限错误

**解决方案**：
- 以管理员身份运行命令提示符
- 或使用 `--user` 参数：`pip install --user torch torchvision`

### 问题 2：下载速度慢

**解决方案**：
- 使用国内镜像源：
```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 3：CUDA 不可用

**解决方案**：
- 确认已安装正确的 CUDA 版本
- 检查 NVIDIA 显卡驱动是否最新
- 使用 CPU 版本继续学习

---

## 🚀 安装完成后

安装成功后，运行练习代码：

```bash
# 运行 3.2 练习
cd 第三阶段_深度学习与现代计算机视觉\3.2_CNN 卷积神经网络
python 练习.py
```

---

## 📚 学习资源

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Anaconda 下载](https://www.anaconda.com/download)

---

> 💡 **提示**：如果遇到任何安装问题，可以查阅官方文档或在社区提问！
