#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.4 迁移学习实战 - 完整练习代码

项目：使用迁移学习训练小数据集
功能：
    - 特征提取（冻结 backbone）
    - 微调（解冻部分层）
    - 全量微调
    - 数据增强
    - 学习率调度

作者：Your Name
日期：2026-04-26
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import time


class TransferLearningTrainer:
    """迁移学习训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置字典
                - model_name: 模型名称（resnet18/resnet50/efficientnet_b0）
                - num_classes: 分类数量
                - strategy: 迁移策略（feature_extract/fine_tune/full_fine_tune）
                - unfreeze_layers: 解冻的层（layer4/layer3_4）
                - learning_rate: 学习率
                - device: 设备
        """
        self.config = config
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"使用设备：{self.device}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def create_model(self):
        """创建模型"""
        print("\n" + "="*50)
        print(f"创建模型：{self.config['model_name']}")
        print("="*50)
        
        model_name = self.config['model_name']
        num_classes = self.config['num_classes']
        
        # 加载预训练模型
        if model_name == 'resnet18':
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        elif model_name == 'resnet50':
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"不支持的模型：{model_name}")
        
        # 应用迁移学习策略
        strategy = self.config['strategy']
        
        if strategy == 'feature_extract':
            # 策略 1: 特征提取（冻结所有层）
            print("策略：特征提取（冻结所有卷积层）")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 替换分类层
            if hasattr(self.model, 'fc'):
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif hasattr(self.model, 'classifier'):
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
        elif strategy == 'fine_tune':
            # 策略 2: 微调（解冻部分层）
            print(f"策略：微调（解冻 {self.config['unfreeze_layers']}）")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 解冻指定层
            unfreeze = self.config['unfreeze_layers']
            if unfreeze == 'layer4' and hasattr(self.model, 'layer4'):
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
            elif unfreeze == 'layer3_4':
                if hasattr(self.model, 'layer3'):
                    for param in self.model.layer3.parameters():
                        param.requires_grad = True
                if hasattr(self.model, 'layer4'):
                    for param in self.model.layer4.parameters():
                        param.requires_grad = True
            
            # 替换分类层
            if hasattr(self.model, 'fc'):
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif hasattr(self.model, 'classifier'):
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
        elif strategy == 'full_fine_tune':
            # 策略 3: 全量微调
            print("策略：全量微调（训练所有层）")
            for param in self.model.parameters():
                param.requires_grad = True
            
            # 替换分类层
            if hasattr(self.model, 'fc'):
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif hasattr(self.model, 'classifier'):
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print("模型创建完成！")
    
    def prepare_data(self):
        """准备数据加载器"""
        print("\n" + "="*50)
        print("准备数据...")
        print("="*50)
        
        # 数据变换（强数据增强）
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        train_dataset = datasets.ImageFolder(
            os.path.join(self.config['data_dir'], 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(self.config['data_dir'], 'val'),
            transform=val_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        print(f"训练集：{len(train_dataset)} 张图像")
        print(f"验证集：{len(val_dataset)} 张图像")
        print(f"类别数量：{len(train_dataset.classes)}")
        print(f"类别名称：{train_dataset.classes}")
        print("数据准备完成！")
    
    def train(self, num_epochs=None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "="*50)
        print(f"开始训练 - {num_epochs} 轮")
        print("="*50)
        
        criterion = nn.CrossEntropyLoss()
        
        # 根据策略设置不同的学习率
        strategy = self.config['strategy']
        if strategy == 'feature_extract':
            # 特征提取：只优化分类层，用较大学习率
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['learning_rate']
            )
        elif strategy in ['fine_tune', 'full_fine_tune']:
            # 微调：卷积层用小学习率，分类层用大学习率
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad and 
                                 'fc' not in str(p) and 
                                 'classifier' not in str(p), 
                                 self.model.parameters()),
                 'lr': self.config['learning_rate'] * 0.1},
                {'params': filter(lambda p: p.requires_grad and 
                                 ('fc' in str(p) or 'classifier' in str(p)), 
                                 self.model.parameters()),
                 'lr': self.config['learning_rate']}
            ], momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        
        best_val_acc = 0.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
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
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc='Validation')
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * correct / total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\n训练损失：{train_loss:.4f}, 训练准确率：{train_acc:.2f}%")
            print(f"验证损失：{val_loss:.4f}, 验证准确率：{val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.4_迁移学习实战\models\best_model.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    model_path
                )
                print(f"[*] 保存最佳模型！验证准确率：{val_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时：{training_time/60:.2f} 分钟")
        print(f"最佳验证准确率：{best_val_acc:.2f}%")
        
        return history, best_val_acc
    
    def evaluate(self):
        """评估模型"""
        print("\n" + "="*50)
        print("模型评估...")
        print("="*50)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = 100. * sum([a == b for a, b in zip(all_preds, all_labels)]) / len(all_preds)
        print(f"验证集准确率：{accuracy:.2f}%")
        
        return accuracy


def main():
    """主函数"""
    print("="*50)
    print("4.4 迁移学习实战")
    print("="*50)
    
    # 配置
    config = {
        'data_dir': 'D:/zzd_project/cursor/HelloWorld_Vision/Stage4_Practical_Projects_Advanced/4.4_迁移学习实战/datasets',
        'model_name': 'resnet50',  # resnet18, resnet50, efficientnet_b0
        'num_classes': 5,  # 5种花分类
        'strategy': 'fine_tune',  # feature_extract, fine_tune, full_fine_tune
        'unfreeze_layers': 'layer4',  # layer4, layer3_4
        'batch_size': 16,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'device': 'cuda'
    }
    
    # 创建训练器
    trainer = TransferLearningTrainer(config)
    
    # 创建模型
    trainer.create_model()
    
    # 准备数据
    trainer.prepare_data()
    
    # 训练模型
    trainer.train()
    
    # 评估模型
    trainer.evaluate()
    
    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == '__main__':
    main()
