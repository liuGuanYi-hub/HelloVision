# 📘 3.2 CNN 卷积神经网络学习笔记

## 📌 什么是 CNN

卷积神经网络（Convolutional Neural Network, CNN）是专门用于处理网格结构数据（如图像）的深度学习模型。

### CNN 的核心优势

1. **局部连接**：每个神经元只连接输入的一小部分
2. **权值共享**：同一个卷积核在整张图上使用相同的参数
3. **空间层次结构**：从低级特征到高级特征的层次化表示

---

## 🔑 CNN 的核心组件

### 1. 卷积层（Convolutional Layer）

卷积层是 CNN 的核心，通过卷积核（滤波器）提取特征。

```python
import torch.nn as nn

# 2D 卷积层
conv = nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB 图像）
    out_channels=64,    # 输出通道数（卷积核数量）
    kernel_size=3,      # 卷积核大小（3x3）
    stride=1,           # 步长
    padding=1           # 填充
)
```

#### 关键参数

- **kernel_size**：卷积核大小（常用 3x3, 5x5）
- **stride**：滑动步长（常用 1, 2）
- **padding**：填充方式
  - `padding=1`：零填充
  - `padding='same'`：保持尺寸不变
  - `padding='valid'`：不填充

#### 卷积操作可视化

```
输入图像 (5x5)    卷积核 (3x3)     输出特征图 (3x3)
[1 1 1 0 0]      [1 0 1]         [2 2 1]
[0 1 1 1 0]  *   [0 1 0]    =    [2 3 2]
[0 0 1 1 1]      [1 0 1]         [1 2 2]
[0 0 0 1 0]
[0 0 0 0 1]
```

---

### 2. 池化层（Pooling Layer）

池化层用于降维，减少计算量，防止过拟合。

```python
# 最大池化
max_pool = nn.MaxPool2d(
    kernel_size=2,    # 池化窗口大小
    stride=2          # 步长
)

# 平均池化
avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)
```

#### 池化类型

| 类型 | 说明 | 优点 |
|------|------|------|
| **最大池化** | 取窗口内最大值 | 提取最显著特征 |
| **平均池化** | 取窗口内平均值 | 保留背景信息 |
| **全局池化** | 对整个特征图池化 | 大幅降维 |

---

### 3. 激活函数（Activation Function）

激活函数引入非线性，使网络能够学习复杂模式。

```python
# ReLU（最常用）
relu = nn.ReLU()

# Leaky ReLU（解决神经元死亡问题）
lrelu = nn.LeakyReLU(0.01)

# Sigmoid（二分类输出）
sigmoid = nn.Sigmoid()

# Softmax（多分类输出）
softmax = nn.Softmax(dim=1)
```

#### ReLU vs Sigmoid

```python
# ReLU: f(x) = max(0, x)
# 优点：计算快，梯度不消失
# 缺点：负区间梯度为 0

# Sigmoid: f(x) = 1 / (1 + e^(-x))
# 优点：输出范围 (0, 1)
# 缺点：梯度消失，计算慢
```

---

### 4. 全连接层（Fully Connected Layer）

全连接层用于整合特征，进行分类。

```python
fc = nn.Linear(
    in_features=512,   # 输入特征数
    out_features=10    # 输出类别数
)
```

---

## 🏗️ 经典 CNN 架构

### LeNet-5（1998）

最早的 CNN 架构之一，用于手写数字识别。

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)    # 28x28 -> 24x24
        self.pool = nn.MaxPool2d(2, 2)      # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(6, 16, 5)    # 12x12 -> 8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### AlexNet（2012）

ImageNet 竞赛冠军，开启了深度学习热潮。

**架构特点**：
- 5 个卷积层 + 3 个全连接层
- ReLU 激活函数
- Dropout 防止过拟合
- GPU 加速训练

### VGG（2014）

使用小卷积核（3x3）的深层网络。

**架构特点**：
- 全部使用 3x3 卷积核
- 网络深度：16-19 层
- 规则的网络结构

### ResNet（2015）

残差网络，解决了深层网络退化问题。

**核心创新**：残差连接（Skip Connection）

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x  # 保存输入
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out += residual  # 残差连接
        return self.relu(out)
```

---

## 🎯 CNN 工作原理

### 特征提取过程

```
输入图像
    ↓
[卷积层 1] 提取边缘、颜色等低级特征
    ↓
[池化层 1] 降维
    ↓
[卷积层 2] 提取纹理、形状等中级特征
    ↓
[池化层 2] 降维
    ↓
[卷积层 3] 提取部件、对象等高级特征
    ↓
[全连接层] 分类
    ↓
输出预测
```

### 可视化理解

```python
# 特征图可视化
import matplotlib.pyplot as plt

def visualize_feature_maps(conv_layer, input_image):
    """可视化卷积层的特征图"""
    with torch.no_grad():
        features = conv_layer(input_image)
    
    # 显示前 16 个通道的特征图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < features.shape[1]:
            ax.imshow(features[0, i], cmap='viridis')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 🔧 训练技巧

### 1. 数据增强

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomRotation(10),          # 随机旋转
    transforms.ColorJitter(brightness=0.2), # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

### 2. 学习率调度

```python
# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=30,    # 每 30 个 epoch 衰减一次
    gamma=0.1        # 衰减系数
)
```

### 3. 批量归一化（BatchNorm）

```python
# 在卷积层后添加 BatchNorm
nn.Sequential(
    nn.Conv2d(64, 128, 3),
    nn.BatchNorm2d(128),    # 加速收敛，提高稳定性
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

---

## 📊 性能优化

### 1. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # 自动混合精度
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 🎓 实战要点

### 设计 CNN 架构的原则

1. **渐进式降维**：特征图尺寸逐渐减小，通道数逐渐增加
2. **使用小卷积核**：3x3 优于 5x5 和 7x7
3. **添加残差连接**：允许训练更深的网络
4. **批量归一化**：加速训练，提高稳定性
5. **数据增强**：提高泛化能力

### 常见错误

❌ **网络太浅**：无法学习复杂特征  
❌ **缺少归一化**：训练不稳定  
❌ **学习率过大**：无法收敛  
❌ **过拟合**：训练集准确率高，测试集低  
❌ **数据泄露**：测试集信息进入训练  

---

## 📚 推荐资源

### 论文

- AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### 可视化工具

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [TensorFlow Playground](https://playground.tensorflow.org/)

### 代码实现

- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models)

---

## ✅ 学习检查清单

- [ ] 理解卷积操作原理
- [ ] 掌握池化层的作用
- [ ] 能够构建简单的 CNN
- [ ] 理解残差连接的意义
- [ ] 掌握数据增强技术
- [ ] 能够调试训练问题
- [ ] 理解特征可视化方法

---

> 💡 **提示**：CNN 是计算机视觉的基石！理解每个组件的作用，多动手实践，才能真正掌握！
