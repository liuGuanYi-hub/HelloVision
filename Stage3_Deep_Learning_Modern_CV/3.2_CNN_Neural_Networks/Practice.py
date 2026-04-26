# -*- coding: utf-8 -*-
"""
3.2 CNN 卷积神经网络 - 练习
学习目标：
1. 理解卷积层、池化层的工作原理
2. 掌握 CNN 架构设计
3. 实现完整的 CNN 模型
4. 可视化特征图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import sys

# 设置控制台编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 解决 OpenMP 库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def practice_1_convolution_operation():
    """
    练习 1：卷积操作演示
    理解卷积核如何提取特征
    """
    print("=" * 60)
    print("练习 1：卷积操作演示")
    print("=" * 60)
    
    # 创建简单的输入图像（5x5 灰度图）
    input_image = torch.tensor([
        [1., 1., 1., 0., 0.],
        [0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]
    ]).unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
    
    # 定义卷积核（边缘检测）
    edge_kernel = torch.tensor([
        [1., 0., -1.],
        [1., 0., -1.],
        [1., 0., -1.]
    ]).unsqueeze(0).unsqueeze(0)
    
    # 定义卷积核（垂直边缘检测）
    vertical_kernel = torch.tensor([
        [1., 1., 1.],
        [0., 0., 0.],
        [-1., -1., -1.]
    ]).unsqueeze(0).unsqueeze(0)
    
    print(f"\n输入图像形状：{input_image.shape}")
    print(f"卷积核形状：{edge_kernel.shape}")
    
    # 手动卷积（使用 nn.functional.conv2d）
    edge_output = F.conv2d(input_image, edge_kernel)
    vertical_output = F.conv2d(input_image, vertical_kernel)
    
    print(f"\n边缘检测输出形状：{edge_output.shape}")
    print(f"\n输入图像：")
    print(input_image[0, 0].numpy())
    
    print(f"\n边缘检测卷积核：")
    print(edge_kernel[0, 0].numpy())
    
    print(f"\n边缘检测结果：")
    print(edge_output[0, 0].numpy())
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(input_image[0, 0], cmap='gray')
    axes[0].set_title('输入图像')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(edge_kernel[0, 0], cmap='gray')
    axes[1].set_title('边缘检测卷积核')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(edge_output[0, 0], cmap='hot')
    axes[2].set_title('卷积结果')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('3.2_练习 1_卷积操作演示.png', dpi=150)
    print("\n✓ 图像已保存：3.2_练习 1_卷积操作演示.png")
    plt.show()


def practice_2_cnn_architecture():
    """
    练习 2：构建 CNN 架构
    实现一个完整的 CNN 模型
    """
    print("=" * 60)
    print("练习 2：构建 CNN 架构")
    print("=" * 60)
    
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            
            # 卷积层块 1
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)  # 28x28 -> 14x14
            )
            
            # 卷积层块 2
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)  # 14x14 -> 7x7
            )
            
            # 全连接层
            self.fc_layers = nn.Sequential(
                nn.Flatten(),  # 64*7*7 -> 3136
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.fc_layers(x)
            return x
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    print(f"\n模型结构：")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")
    
    # 测试前向传播
    dummy_input = torch.randn(4, 1, 28, 28)  # batch_size=4, 单通道 28x28 图像
    output = model(dummy_input)
    
    print(f"\n输入形状：{dummy_input.shape}")
    print(f"输出形状：{output.shape}")
    print(f"✓ 前向传播成功！")
    
    # 可视化网络结构
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers_info = [
        '输入\n1x28x28',
        'Conv1\n32 通道',
        'Pool1\n14x14',
        'Conv2\n64 通道',
        'Pool2\n7x7',
        'FC1\n128',
        '输出\n10 类'
    ]
    
    positions = [0, 1, 2, 3, 4, 5, 6]
    colors = ['#FFB6C1', '#87CEFA', '#98FB98', '#DDA0DD', '#F0E68C', '#FFA07A', '#90EE90']
    
    for i, (pos, layer, color) in enumerate(zip(positions, layers_info, colors)):
        ax.barh(0, 1, left=pos, color=color, edgecolor='black', linewidth=2)
        ax.text(pos + 0.5, 0, layer, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('CNN 架构流程图', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3.2_练习 2_CNN 架构.png', dpi=150)
    print("\n✓ 架构图已保存：3.2_练习 2_CNN 架构.png")
    plt.show()


def practice_3_feature_visualization():
    """
    练习 3：特征图可视化
    观察卷积层提取的特征
    """
    print("=" * 60)
    print("练习 3：特征图可视化")
    print("=" * 60)
    
    # 创建简单的 CNN
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            
        def forward(self, x):
            x1 = self.relu(self.conv1(x))
            x2 = self.pool(self.relu(self.conv2(x1)))
            return x1, x2
    
    model = FeatureExtractor()
    
    # 创建随机输入图像
    input_image = torch.randn(1, 1, 28, 28)
    
    # 前向传播
    with torch.no_grad():
        features1, features2 = model(input_image)
    
    print(f"\n输入图像形状：{input_image.shape}")
    print(f"第一层特征图形状：{features1.shape}")
    print(f"第二层特征图形状：{features2.shape}")
    
    # 可视化输入图像
    plt.figure(figsize=(15, 12))
    
    # 输入图像
    plt.subplot(4, 4, 1)
    plt.imshow(input_image[0, 0], cmap='gray')
    plt.title('输入图像', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 第一层特征图（8 个通道）
    for i in range(8):
        plt.subplot(4, 4, i + 2)
        plt.imshow(features1[0, i], cmap='viridis')
        plt.title(f'通道 {i+1}', fontsize=9)
        plt.axis('off')
    
    # 第二层特征图（16 个通道，只显示前 7 个）
    for i in range(7):
        plt.subplot(4, 4, i + 10)
        plt.imshow(features2[0, i], cmap='viridis')
        plt.title(f'通道 {i+1} (池化后)', fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('3.2_练习 3_特征可视化.png', dpi=150)
    print("\n✓ 特征图已保存：3.2_练习 3_特征可视化.png")
    plt.show()


def practice_4_train_cnn():
    """
    练习 4：训练 CNN 模型
    完整的训练流程实战
    """
    print("=" * 60)
    print("练习 4：训练 CNN 模型")
    print("=" * 60)
    
    # 生成 synthetic 数据（模拟图像分类任务）
    np.random.seed(42)
    n_samples = 2000
    
    # 创建简单的图像数据（8x8 灰度图）
    X = np.random.randn(n_samples, 1, 8, 8).astype(np.float32)
    # 创建标签（基于图像亮度）
    y = (X[:, 0, :, :].mean(axis=(1, 2)) > 0).astype(np.int64)
    
    # 转换为张量
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 定义 CNN 模型
    class BinaryImageClassifier(nn.Module):
        def __init__(self):
            super(BinaryImageClassifier, self).__init__()
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),  # 8x8 -> 8x8
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8 -> 4x4
                
                nn.Conv2d(16, 32, 3, padding=1),  # 4x4 -> 4x4
                nn.ReLU(),
                nn.MaxPool2d(2),  # 4x4 -> 2x2
            )
            
            self.fc_layers = nn.Sequential(
                nn.Flatten(),  # 32*2*2 = 128
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 2)
            )
            
        def forward(self, x):
            x = self.conv_layers(x)
            return self.fc_layers(x)
    
    model = BinaryImageClassifier()
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 30
    loss_history = []
    accuracy_history = []
    
    print(f"\n开始训练...")
    print(f"训练集大小：{n_samples}")
    print(f"批次大小：64")
    print(f"训练轮数：{num_epochs}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            # 前向传播
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Accuracy: {accuracy*100:.2f}%')
    
    print("-" * 60)
    print(f"✓ 训练完成！")
    print(f"最终准确率：{accuracy_history[-1]*100:.2f}%")
    
    # 可视化训练过程
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(loss_history, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=min(loss_history), color='r', linestyle='--', 
                label=f'最小损失：{min(loss_history):.4f}')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(accuracy_history, 'g-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('训练准确率曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=max(accuracy_history), color='r', linestyle='--',
                label=f'最高准确率：{max(accuracy_history)*100:.2f}%')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('3.2_练习 4_训练过程.png', dpi=150)
    print("\n✓ 训练曲线已保存：3.2_练习 4_训练过程.png")
    plt.show()


def practice_5_compare_pooling():
    """
    练习 5：比较不同池化方法
    观察最大池化和平均池化的区别
    """
    print("=" * 60)
    print("练习 5：比较不同池化方法")
    print("=" * 60)
    
    # 创建输入特征图
    input_feature = torch.tensor([[
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ]]).unsqueeze(0)
    
    print(f"\n输入特征图形状：{input_feature.shape}")
    print(f"输入特征图：")
    print(input_feature[0, 0].numpy())
    
    # 最大池化
    max_pool = nn.MaxPool2d(2)
    max_pooled = max_pool(input_feature)
    
    # 平均池化
    avg_pool = nn.AvgPool2d(2)
    avg_pooled = avg_pool(input_feature)
    
    print(f"\n最大池化结果 (2x2)：")
    print(max_pooled[0, 0].numpy())
    
    print(f"\n平均池化结果 (2x2)：")
    print(avg_pooled[0, 0].numpy())
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(input_feature[0, 0], cmap='viridis')
    axes[0].set_title('输入特征图', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    for i in range(4):
        for j in range(4):
            text = axes[0].text(j, i, f'{input_feature[0, 0, i, j]:.0f}',
                               ha='center', va='center', color='white', fontsize=14)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_pooled[0, 0], cmap='viridis')
    axes[1].set_title('最大池化', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    for i in range(2):
        for j in range(2):
            text = axes[1].text(j, i, f'{max_pooled[0, 0, i, j]:.0f}',
                               ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(avg_pooled[0, 0], cmap='viridis')
    axes[2].set_title('平均池化', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    for i in range(2):
        for j in range(2):
            text = axes[2].text(j, i, f'{avg_pooled[0, 0, i, j]:.1f}',
                               ha='center', va='center', color='white', fontsize=14)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('3.2_练习 5_池化方法对比.png', dpi=150)
    print("\n✓ 对比图已保存：3.2_练习 5_池化方法对比.png")
    plt.show()
    
    print(f"\n池化方法说明：")
    print(f"  - 最大池化：提取每个窗口的最大值，保留最显著特征")
    print(f"  - 平均池化：计算每个窗口的平均值，保留背景信息")
    print(f"  - 池化作用：降维、减少计算量、防止过拟合")


def main():
    """
    主函数：运行所有练习
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "CNN 卷积神经网络练习")
    print("=" * 60)
    
    while True:
        print("\n请选择要运行的练习：")
        print("1. 卷积操作演示")
        print("2. 构建 CNN 架构")
        print("3. 特征图可视化")
        print("4. 训练 CNN 模型")
        print("5. 池化方法对比")
        print("6. 运行所有练习")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-6): ").strip()
        
        if choice == '1':
            practice_1_convolution_operation()
        elif choice == '2':
            practice_2_cnn_architecture()
        elif choice == '3':
            practice_3_feature_visualization()
        elif choice == '4':
            practice_4_train_cnn()
        elif choice == '5':
            practice_5_compare_pooling()
        elif choice == '6':
            practice_1_convolution_operation()
            input("\n按 Enter 继续下一个练习...")
            practice_2_cnn_architecture()
            input("\n按 Enter 继续下一个练习...")
            practice_3_feature_visualization()
            input("\n按 Enter 继续下一个练习...")
            practice_4_train_cnn()
            input("\n按 Enter 继续下一个练习...")
            practice_5_compare_pooling()
            print("\n✓ 所有练习完成！")
        elif choice == '0':
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择！")


if __name__ == "__main__":
    main()
