# -*- coding: utf-8 -*-
"""
PyTorch 基础练习文件 - 张量、自动求导、神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置控制台编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 解决 OpenMP 库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def practice_1_tensor_operations():
    """
    练习 1：张量基础操作
    学习创建和操作张量
    """
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
    print(f"3D 张量元素总数：{tensor3.numel()}")
    
    # 张量运算
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"\n矩阵 a:\n{a}")
    print(f"矩阵 b:\n{b}")
    print(f"矩阵加法:\n{a + b}")
    print(f"矩阵乘法:\n{a @ b}")
    
    # 张量变形
    tensor = torch.randn(3, 4, 5)
    flattened = tensor.view(-1)
    reshaped = tensor.view(12, 5)
    
    print(f"\n原始形状：{tensor.shape}")
    print(f"展平后形状：{flattened.shape}")
    print(f"重塑后形状：{reshaped.shape}")
    
    # GPU 加速（如果有）
    if torch.cuda.is_available():
        print(f"\nGPU 可用：{torch.cuda.get_device_name(0)}")
        tensor_gpu = tensor.cuda()
        print(f"GPU 上的张量形状：{tensor_gpu.shape}")


def practice_2_autograd():
    """
    练习 2：自动求导
    学习 PyTorch 的自动微分功能
    """
    print("\n" + "=" * 60)
    print("练习 2：自动求导")
    print("=" * 60)
    
    # 简单函数求导
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3 - 2 * x ** 2 + 3 * x - 1
    y.backward()
    
    print(f"函数：y = x^3 - 2x^2 + 3x - 1")
    print(f"当 x = 2 时，dy/dx = {x.grad.item():.4f}")
    print(f"理论值：3x^2 - 4x + 3 = {3*2**2 - 4*2 + 3}")
    
    # 多元函数求导
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = x ** 2 + y ** 3
    z.backward()
    
    print(f"\n函数：z = x^2 + y^3")
    print(f"当 x=1, y=2 时：")
    print(f"dz/dx = {x.grad.item():.4f} (理论值：2x = 2)")
    print(f"dz/dy = {y.grad.item():.4f} (理论值：3y^2 = 12)")
    
    # 神经网络梯度
    print("\n神经网络梯度示例：")
    W = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    x = torch.randn(4)
    
    # 前向传播
    output = torch.matmul(W, x) + b
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    print(f"权重梯度形状：{W.grad.shape}")
    print(f"偏置梯度形状：{b.grad.shape}")


def practice_3_build_neural_network():
    """
    练习 3：构建简单神经网络
    学习使用 nn.Module 定义网络
    """
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
    print("神经网络结构：")
    print(model)
    
    # 前向传播
    x = torch.randn(5, 10)  # batch_size=5
    output = model(x)
    
    print(f"\n输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数量：{total_params}")
    print(f"可训练参数：{trainable_params}")


def practice_4_training_loop():
    """
    练习 4：训练循环实战
    学习完整的训练流程
    """
    print("\n" + "=" * 60)
    print("练习 4：训练循环实战")
    print("=" * 60)
    
    # 生成 synthetic 数据
    np.random.seed(42)
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1] > 0).astype(np.int64)
    
    print(f"数据形状：X={X.shape}, y={y.shape}")
    print(f"正样本比例：{y.mean():.2%}")
    
    # 转换为张量
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)
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
    
    print(f"\n开始训练，共 {num_epochs} 个 epoch...")
    
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


def practice_5_pretrained_model():
    """
    练习 5：使用预训练模型
    学习加载和使用预训练模型
    """
    print("\n" + "=" * 60)
    print("练习 5：使用预训练模型")
    print("=" * 60)
    
    import torchvision.models as models
    
    # 加载预训练的 ResNet
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    
    print("ResNet-18 模型结构：")
    print(resnet)
    
    # 创建测试图像（随机噪声）
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # 预处理
    import torchvision.transforms as transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(test_image)
    input_batch = input_tensor.unsqueeze(0)
    
    print(f"\n输入张量形状：{input_batch.shape}")
    
    # 推理
    with torch.no_grad():
        output = resnet(input_batch)
    
    print(f"输出张量形状：{output.shape}")
    print(f"输出最大值：{output.max().item():.4f}")
    print(f"输出最小值：{output.min().item():.4f}")


def practice_6_transfer_learning():
    """
    练习 6：迁移学习
    学习如何微调预训练模型
    """
    print("\n" + "=" * 60)
    print("练习 6：迁移学习")
    print("=" * 60)
    
    import torchvision.models as models
    
    # 加载预训练模型
    resnet = models.resnet18(pretrained=True)
    
    print("原始 ResNet-18 结构：")
    print(resnet)
    
    # 冻结所有参数
    for param in resnet.parameters():
        param.requires_grad = False
    
    # 替换最后一层全连接层
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 10)  # 改为 10 分类
    
    print("\n修改后的模型结构：")
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
    
    # 计算可训练参数比例
    trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in resnet.parameters())
    print(f"可训练参数比例：{trainable / total * 100:.2f}%")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyTorch 基础练习开始！")
    print("=" * 60)
    
    # 运行所有练习
    practice_1_tensor_operations()
    practice_2_autograd()
    practice_3_build_neural_network()
    practice_4_training_loop()
    practice_5_pretrained_model()
    practice_6_transfer_learning()
    
    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)
    
    # 最后显示所有图片
    plt.show()
