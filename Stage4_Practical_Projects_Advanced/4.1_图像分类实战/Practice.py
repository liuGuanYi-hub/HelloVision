#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.1 图像分类实战 - 完整练习代码

项目：猫狗分类 / 手势识别 / 花卉识别
功能：
    - 数据加载与预处理
    - 数据增强
    - 模型构建（ResNet/EfficientNet）
    - 训练与验证
    - 模型评估与可视化

作者：Your Name
日期：2026-04-26
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import time


class ImageClassificationTrainer:
    """图像分类训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置字典，包含：
                - data_dir: 数据集目录
                - model_name: 模型名称
                - num_classes: 分类数量
                - batch_size: 批次大小
                - num_epochs: 训练轮数
                - learning_rate: 学习率
                - device: 运行设备
        """
        self.config = config
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"使用设备：{self.device}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        self.history = None
        
    def prepare_data(self):
        """准备数据加载器"""
        print("\n" + "="*50)
        print("准备数据...")
        print("="*50)
        
        # 数据变换
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ])
        }
        
        # 加载数据集
        data_dir = self.config['data_dir']
        image_datasets = {
            'train': datasets.ImageFolder(
                os.path.join(data_dir, 'train'), 
                data_transforms['train']
            ),
            'val': datasets.ImageFolder(
                os.path.join(data_dir, 'val'), 
                data_transforms['val']
            ),
            'test': datasets.ImageFolder(
                os.path.join(data_dir, 'test'), 
                data_transforms['test']
            )
        }
        
        self.class_names = image_datasets['train'].classes
        print(f"类别数量：{len(self.class_names)}")
        print(f"类别名称：{self.class_names}")
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            image_datasets['train'], 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            image_datasets['val'], 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            image_datasets['test'], 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"训练集大小：{len(image_datasets['train'])}")
        print(f"验证集大小：{len(image_datasets['val'])}")
        print(f"测试集大小：{len(image_datasets['test'])}")
        print("数据准备完成！")
    
    def create_model(self):
        """创建模型"""
        print("\n" + "="*50)
        print("创建模型...")
        print("="*50)
        
        model_name = self.config['model_name']
        num_classes = self.config['num_classes']
        pretrained = self.config.get('pretrained', True)
        
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 
                num_classes
            )
        
        elif model_name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b3(weights=weights)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 
                num_classes
            )
        
        else:
            raise ValueError(f"不支持的模型：{model_name}")
        
        self.model = self.model.to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"模型：{model_name}")
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        print("模型创建完成！")
    
    def set_transfer_learning(self, strategy='feature_extract'):
        """
        设置迁移学习策略
        
        Args:
            strategy: 策略选项
                - 'feature_extract': 仅训练最后分类层
                - 'fine_tune': 微调最后几个卷积层 + 分类层
                - 'full_fine_tune': 全量微调所有层
        """
        print("\n" + "="*50)
        print(f"迁移学习策略：{strategy}")
        print("="*50)
        
        if strategy == 'feature_extract':
            # 冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            # 只训练分类层
            for param in self.model.fc.parameters() if hasattr(self.model, 'fc') \
                         else self.model.classifier.parameters():
                param.requires_grad = True
            print("特征提取模式：仅训练分类层")
        
        elif strategy == 'fine_tune':
            # 冻结所有层
            for param in self.model.parameters():
                param.requires_grad = False
            # 解冻最后几个卷积层
            if hasattr(self.model, 'layer4'):
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
            # 训练分类层
            for param in self.model.fc.parameters() if hasattr(self.model, 'fc') \
                         else self.model.classifier.parameters():
                param.requires_grad = True
            print("微调模式：解冻 layer4 和分类层")
        
        elif strategy == 'full_fine_tune':
            # 解冻所有层
            for param in self.model.parameters():
                param.requires_grad = True
            print("全量微调模式：训练所有层")
        
        else:
            raise ValueError(f"不支持的策略：{strategy}")
    
    def train(self, num_epochs=None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数（可选，覆盖配置）
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "="*50)
        print(f"开始训练 - {num_epochs} 轮")
        print("="*50)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = optim.SGD(
            filter(p => p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc='Training')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc='Validation')
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * correct / total
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\n训练损失：{train_loss:.4f}, 训练准确率：{train_acc:.2f}%")
            print(f"验证损失：{val_loss:.4f}, 验证准确率：{val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models', exist_ok=True)
                torch.save(
                    self.model.state_dict(), 
                    'models/best_model.pth'
                )
                print(f"✨ 保存最佳模型！验证准确率：{val_acc:.2f}%")
            
            # 更新学习率
            scheduler.step(val_loss)
        
        training_time = time.time() - start_time
        print(f"\n🎉 训练完成！总耗时：{training_time/60:.2f} 分钟")
        print(f"最佳验证准确率：{best_val_acc:.2f}%")
        
        return self.history
    
    def evaluate(self):
        """评估模型"""
        print("\n" + "="*50)
        print("模型评估...")
        print("="*50)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 分类报告
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names
        ))
        
        # 计算总体准确率
        accuracy = 100. * sum([a == b for a, b in zip(all_labels, all_preds)]) / len(all_labels)
        print(f"\n测试集准确率：{accuracy:.2f}%")
        
        return cm, accuracy
    
    def visualize_predictions(self, num_images=10):
        """可视化预测结果"""
        print("\n可视化预测结果...")
        
        self.model.eval()
        images, labels = next(iter(self.test_loader))
        images, labels = images[:num_images].to(self.device), labels[:num_images].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            _, preds = outputs.max(1)
        
        # 可视化
        fig, axes = plt.subplots(2, num_images//2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            # 反标准化
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + \
                  np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            color = 'green' if preds[i] == labels[i] else 'red'
            ax.set_title(
                f'True: {self.class_names[labels[i]]}\nPred: {self.class_names[preds[i]]}',
                color=color,
                fontsize=10
            )
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史曲线"""
        print("\n绘制训练历史...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(self.history['val_acc'], label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("="*50)
    print("4.1 图像分类实战")
    print("="*50)
    
    # 配置
    config = {
        'data_dir': 'data',  # 数据集目录
        'model_name': 'resnet50',  # 模型选择：resnet18, resnet50, efficientnet_b0, efficientnet_b3
        'num_classes': 2,  # 分类数量（猫狗=2，花卉=5 等）
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'device': 'cuda',
        'pretrained': True
    }
    
    # 创建训练器
    trainer = ImageClassificationTrainer(config)
    
    # 准备数据
    trainer.prepare_data()
    
    # 创建模型
    trainer.create_model()
    
    # 设置迁移学习策略
    trainer.set_transfer_learning('fine_tune')
    
    # 训练模型
    trainer.train()
    
    # 评估模型
    trainer.evaluate()
    
    # 可视化预测结果
    trainer.visualize_predictions(num_images=10)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == '__main__':
    main()
