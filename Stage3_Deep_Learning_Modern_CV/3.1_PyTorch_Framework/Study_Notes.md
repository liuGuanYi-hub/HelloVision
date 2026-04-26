# 🔥 PyTorch 深度学习框架学习笔记

> PyTorch 是目前最流行的深度学习框架之一，以其简洁易用、动态计算图而闻名。

---

## 📖 目录

1. [PyTorch 简介](#1-pytorch-简介)
2. [张量操作](#2-张量操作)
3. [自动求导](#3-自动求导)
4. [神经网络构建](#4-神经网络构建)
5. [数据加载](#5-数据加载)
6. [训练循环](#6-训练循环)
7. [实战练习](#7-实战练习)

---

## 1. PyTorch 简介

### 为什么选择 PyTorch？

- **Pythonic**：代码简洁，符合 Python 编程习惯
- **动态计算图**：调试方便，支持动态网络结构
- **丰富的生态**：torchvision, torchaudio, torchtext 等
- **研究首选**：大多数论文都提供 PyTorch 实现
- **工业应用**：Facebook、Tesla 等公司广泛使用

### 安装

```bash
# GPU 版本（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch torchvision torchaudio
```

### 导入

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")
```

---

## 2. 张量操作

### 2.1 创建张量

```python
import torch

# 从列表创建
tensor1 = torch.tensor([1, 2, 3, 4, 5])

# 创建全 0 张量
zeros = torch.zeros(3, 4)

# 创建全 1 张量
ones = torch.ones(2, 3)

# 创建随机张量
random = torch.rand(3, 3)

# 创建正态分布张量
normal = torch.randn(3, 3)

# 创建等差数列
linspace = torch.linspace(0, 1, 10)

# 创建单位矩阵
eye = torch.eye(3)
```

### 2.2 张量属性

```python
tensor = torch.randn(3, 4, 5)

print(f"形状：{tensor.shape}")
print(f"维度：{tensor.dim()}")
print(f"元素总数：{tensor.numel()}")
print(f"数据类型：{tensor.dtype}")
print(f"设备：{tensor.device}")
```

### 2.3 张量运算

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 加法
c = a + b
c = torch.add(a, b)

# 减法
c = a - b

# 乘法（元素级）
c = a * b

# 矩阵乘法
c = a @ b
c = torch.matmul(a, b)

# 除法
c = a / b

# 幂运算
c = a ** 2

# 转置
c = a.T
c = a.transpose(0, 1)
```

### 2.4 张量索引与切片

```python
tensor = torch.randn(5, 5)

# 基本索引
value = tensor[0, 0]

# 切片
row = tensor[0, :]
col = tensor[:, 0]
sub = tensor[1:3, 1:3]

# 条件索引
mask = tensor > 0
positive = tensor[mask]

# 高级索引
indices = torch.tensor([0, 2, 4])
selected = tensor[indices]
```

### 2.5 张量变形

```python
tensor = torch.randn(3, 4, 5)

# 改变形状
reshaped = tensor.view(12, 5)
reshaped = tensor.reshape(12, 5)

# 展平
flattened = tensor.flatten()
flattened = tensor.view(-1)

# 压缩维度
squeezed = torch.squeeze(tensor)

# 扩展维度
unsqueezed = torch.unsqueeze(tensor, 0)

# 转置
transposed = tensor.permute(2, 0, 1)
```

### 2.6 GPU 加速

```python
# 检查 GPU
if torch.cuda.is_available():
    print(f"GPU 数量：{torch.cuda.device_count()}")
    print(f"当前 GPU：{torch.cuda.get_device_name(0)}")

# 移动到 GPU
tensor = torch.randn(3, 3)
tensor_gpu = tensor.to('cuda')

# 或者
tensor_gpu = tensor.cuda()

# 移回 CPU
tensor_cpu = tensor_gpu.to('cpu')
```

---

## 3. 自动求导

### 3.1 梯度计算

```python
import torch

# 创建需要梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义函数
y = x ** 2 + 2 * x + 1

# 反向传播
y.backward(torch.ones_like(x))

# 获取梯度
print(f"梯度：{x.grad}")
```

### 3.2 计算图

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x ** 2 + y ** 3
z.backward()

print(f"dz/dx = {x.grad}")  # 2x = 4
print(f"dz/dy = {y.grad}")  # 3y^2 = 27
```

### 3.3 停止梯度追踪

```python
x = torch.randn(3, 3, requires_grad=True)

# 方法 1：detach()
y = x.detach()

# 方法 2：no_grad() 上下文
with torch.no_grad():
    y = x ** 2
```

---

## 4. 神经网络构建

### 4.1 定义网络

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 创建网络实例
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
print(model)
```

### 4.2 常用层

```python
# 全连接层
fc = nn.Linear(128, 64)

# 卷积层
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# 池化层
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Dropout
dropout = nn.Dropout(0.5)

# BatchNorm
bn = nn.BatchNorm2d(64)

# LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2)

# Embedding
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
```

### 4.3 激活函数

```python
# ReLU
relu = nn.ReLU()

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()

# Leaky ReLU
leaky_relu = nn.LeakyReLU(0.01)

# Softmax
softmax = nn.Softmax(dim=1)

# 或者使用函数式 API
x = F.relu(x)
x = F.sigmoid(x)
x = F.softmax(x, dim=1)
```

---

## 5. 数据加载

### 5.1 Dataset 和 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集
dataset = CustomDataset(data, labels)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### 5.2 使用内置数据集

```python
from torchvision import datasets, transforms

# MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

---

## 6. 训练循环

### 6.1 完整训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
# ...

# 2. 定义模型
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 验证
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
```

### 6.2 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 6.3 学习率调度

```python
# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 在训练循环中使用
for epoch in range(num_epochs):
    # 训练...
    
    # 更新学习率
    scheduler.step()
```

---

## 7. 实战练习

### 练习 1：张量基础操作

```python
import torch
import numpy as np

print("=" * 60)
print("练习 1：张量基础操作")
print("=" * 60)

# 创建各种张量
tensor1 = torch.tensor([1, 2, 3, 4, 5])
tensor2 = torch.randn(3, 4)
tensor3 = torch.zeros(2, 3, 4)

print(f"1D 张量：{tensor1}")
print(f"2D 张量形状：{tensor2.shape}")
print(f"3D 张量形状：{tensor3.shape}")

# 张量运算
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print(f"\n矩阵加法:\n{a + b}")
print(f"矩阵乘法:\n{a @ b}")

# 张量变形
tensor = torch.randn(3, 4, 5)
flattened = tensor.view(-1)
print(f"\n原始形状：{tensor.shape}")
print(f"展平后形状：{flattened.shape}")
```

### 练习 2：自动求导

```python
import torch

print("\n" + "=" * 60)
print("练习 2：自动求导")
print("=" * 60)

# 简单函数求导
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 - 2 * x ** 2 + 3 * x - 1
y.backward()

print(f"函数：y = x^3 - 2x^2 + 3x - 1")
print(f"当 x = 2 时，dy/dx = {x.grad.item()}")
print(f"理论值：3x^2 - 4x + 3 = {3*2**2 - 4*2 + 3}")

# 多元函数求导
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x ** 2 + y ** 3
z.backward()

print(f"\n函数：z = x^2 + y^3")
print(f"当 x=1, y=2 时：")
print(f"dz/dx = {x.grad.item()} (理论值：2x = 2)")
print(f"dz/dy = {y.grad.item()} (理论值：3y^2 = 12)")
```

### 练习 3：构建简单神经网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("\n" + "=" * 60)
print("练习 3：构建简单神经网络")
print("=" * 60)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建网络
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
print(model)

# 前向传播
x = torch.randn(5, 10)  # batch_size=5
output = model(x)
print(f"\n输入形状：{x.shape}")
print(f"输出形状：{output.shape}")
```

### 练习 4：训练循环实战

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("练习 4：训练循环实战")
print("=" * 60)

# 生成 synthetic 数据
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] ** 2 + X[:, 1] > 0).astype(np.int64)

# 转换为张量
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 创建数据集
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = BinaryClassifier(input_size=10)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in dataloader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)
plt.show()

# 评估
model.eval()
with torch.no_grad():
    output = model(X_tensor)
    _, predicted = torch.max(output, 1)
    accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
    print(f'\n最终准确率：{accuracy * 100:.2f}%')
```

### 练习 5：使用预训练模型

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

print("\n" + "=" * 60)
print("练习 5：使用预训练模型")
print("=" * 60)

# 加载预训练的 ResNet
resnet = models.resnet18(pretrained=True)
resnet.eval()

print("ResNet-18 模型结构：")
print(resnet)

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建测试图像（随机噪声）
test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
pil_image = Image.fromarray(test_image)

# 预处理
input_tensor = transform(pil_image)
input_batch = input_tensor.unsqueeze(0)  # 添加 batch 维度

print(f"\n输入张量形状：{input_batch.shape}")

# 推理
with torch.no_grad():
    output = resnet(input_batch)

print(f"输出张量形状：{output.shape}")
print(f"输出最大值：{output.max().item():.4f}")
print(f"输出最小值：{output.min().item():.4f}")
```

### 练习 6：迁移学习

```python
import torch
import torch.nn as nn
import torchvision.models as models

print("\n" + "=" * 60)
print("练习 6：迁移学习")
print("=" * 60)

# 加载预训练模型
resnet = models.resnet18(pretrained=True)

# 冻结所有参数
for param in resnet.parameters():
    param.requires_grad = False

# 替换最后一层全连接层
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 改为 10 分类

print("修改后的模型结构：")
print(resnet)

# 检查哪些层需要训练
print("\n需要训练的层：")
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# 定义优化器（只优化新层）
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

print(f"\n优化器将优化 {sum(p.numel() for p in resnet.fc.parameters())} 个参数")
print(f"总参数数量：{sum(p.numel() for p in resnet.parameters())}")
```

---

## 📝 练习答案

### 练习 1 答案

```python
# 张量运算
a @ b  # 矩阵乘法
tensor.view(-1)  # 展平
```

### 练习 2 答案

```python
# 自动求导
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 - 2 * x ** 2 + 3 * x - 1
y.backward()
```

### 练习 3 答案

```python
# 构建神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)
```

### 练习 4 答案

```python
# 训练循环
for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

### 练习 5 答案

```python
# 使用预训练模型
model = models.resnet18(pretrained=True)
model.eval()
```

### 练习 6 答案

```python
# 迁移学习
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(num_features, num_classes)
```

---

## 🎯 关键要点总结

1. **张量是基础**：所有数据都是张量，熟练掌握张量操作
2. **自动求导**：PyTorch 的核心特性，理解计算图
3. **nn.Module**：所有神经网络模块的基类
4. **DataLoader**：高效的数据加载工具
5. **训练循环**：标准流程：前向→损失→反向→更新
6. **预训练模型**：torchvision.models 提供丰富选择

> 💡 **下一步**：学习 CNN 卷积神经网络，理解计算机视觉的核心架构！

---

## 📚 扩展学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- 《深度学习之 PyTorch 实战计算机视觉》
- [Fast.ai 课程](https://fast.ai/)

---

> 🌟 **记住**：PyTorch 是深度学习的利器，多动手实践才能真正掌握！
