#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.3 图像分割实战 - 完整练习代码

项目：使用 U-Net 和 DeepLabv3+ 进行图像分割
功能：
    - U-Net 模型实现
    - DeepLabv3+ 使用
    - 数据集加载
    - 模型训练与评估
    - 分割结果可视化

作者：Your Name
日期：2026-04-26
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ==================== U-Net 模型定义 ====================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 模型"""
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ==================== 数据集类 ====================

class SegmentationDataset(Dataset):
    """图像分割数据集"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # 加载图像和掩码
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask


class SimpleSegmentationDataset(Dataset):
    """简化的分割数据集（用于快速测试）"""
    
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.NEAREST),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = self.transform(image)
        mask = self.mask_transform(mask).squeeze(0).long()
        
        return image, mask


# ==================== 训练器类 ====================

class SegmentationTrainer:
    """分割模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"使用设备：{self.device}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def create_model(self, model_type='unet'):
        """创建模型"""
        print("\n" + "="*50)
        print(f"创建模型：{model_type}")
        print("="*50)
        
        if model_type == 'unet':
            self.model = UNet(
                n_channels=3, 
                n_classes=self.config['num_classes'],
                bilinear=True
            )
        elif model_type == 'deeplabv3':
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.model = deeplabv3_resnet50(
                weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1'
            )
            self.model.classifier[4] = nn.Conv2d(
                256, 
                self.config['num_classes'], 
                kernel_size=1
            )
        else:
            raise ValueError(f"不支持的模型：{model_type}")
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        print("模型创建完成！")
    
    def prepare_data(self):
        """准备数据加载器"""
        print("\n" + "="*50)
        print("准备数据...")
        print("="*50)
        
        train_dataset = SimpleSegmentationDataset(
            images_dir=os.path.join(self.config['data_dir'], 'images', 'train'),
            masks_dir=os.path.join(self.config['data_dir'], 'masks', 'train'),
            img_size=self.config['img_size']
        )
        
        val_dataset = SimpleSegmentationDataset(
            images_dir=os.path.join(self.config['data_dir'], 'images', 'val'),
            masks_dir=os.path.join(self.config['data_dir'], 'masks', 'val'),
            img_size=self.config['img_size']
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
        print("数据准备完成！")
    
    def train(self, num_epochs=None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "="*50)
        print(f"开始训练 - {num_epochs} 轮")
        print("="*50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc='Training')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = running_loss / len(self.train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                for images, masks in self.val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # 计算 IoU
                    _, predicted = torch.max(outputs, 1)
                    for i in range(self.config['num_classes']):
                        pred_class = (predicted == i)
                        true_class = (masks == i)
                        intersection = (pred_class & true_class).sum().item()
                        union = pred_class.sum().item() + true_class.sum().item()
                        if union > 0:
                            val_iou += intersection / union
            
            val_loss = val_loss / len(self.val_loader)
            val_iou = val_iou / (len(self.val_loader) * self.config['num_classes'])
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            
            print(f"\n训练损失：{train_loss:.4f}")
            print(f"验证损失：{val_loss:.4f}, 验证 IoU: {val_iou:.4f}")
            
            scheduler.step(val_loss)
            
            # 每个epoch都保存模型
            os.makedirs('models', exist_ok=True)
            torch.save(self.model.state_dict(), 'models/best_model.pth')
            print(f"[S] Model saved. Val IoU: {val_iou:.4f}")
        
        print(f"\n训练完成！最佳验证 IoU: {best_val_loss:.4f}")
        return history
    
    def evaluate(self):
        """评估模型"""
        print("\n" + "="*50)
        print("模型评估...")
        print("="*50)
        
        self.model.eval()
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_masks.extend(masks.numpy())
        
        # 计算平均 IoU
        mean_iou = 0.0
        for pred, mask in zip(all_preds, all_masks):
            for i in range(self.config['num_classes']):
                pred_class = (pred == i)
                true_class = (mask == i)
                intersection = np.logical_and(pred_class, true_class).sum()
                union = pred_class.sum() + true_class.sum()
                if union > 0:
                    mean_iou += intersection / union
        
        mean_iou /= (len(all_preds) * self.config['num_classes'])
        print(f"平均 IoU: {mean_iou:.4f}")
        
        return mean_iou
    
    def visualize_results(self, num_images=5):
        """可视化分割结果"""
        print("\n可视化分割结果...")
        
        self.model.eval()
        
        images, masks = next(iter(self.val_loader))
        images = images[:num_images].to(self.device)
        masks = masks[:num_images]
        
        with torch.no_grad():
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
        
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
        
        for i, ax in enumerate(axes):
            # 原图
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax[0].imshow(img)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            # 真实掩码
            ax[1].imshow(masks[i], cmap='gray')
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')
            
            # 预测掩码
            ax[2].imshow(preds[i], cmap='gray')
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/segmentation_results.png', dpi=300)
        plt.show()


def main():
    """主函数"""
    print("="*50)
    print("4.3 图像分割实战")
    print("="*50)
    
    # 配置
    config = {
        'data_dir': 'D:/zzd_project/cursor/HelloWorld_Vision/Stage4_Practical_Projects_Advanced/4.3_图像分割实战/datasets',
        'model_type': 'unet',  # unet 或 deeplabv3
        'num_classes': 2,  # 背景 + 前景
        'img_size': 256,
        'batch_size': 4,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'device': 'cuda'
    }
    
    # 创建训练器
    trainer = SegmentationTrainer(config)
    
    # 创建模型
    trainer.create_model(config['model_type'])
    
    # 准备数据
    trainer.prepare_data()
    
    # 训练模型
    trainer.train()
    
    # 评估模型
    trainer.evaluate()
    
    # 可视化结果（暂时跳过避免matplotlib问题）
    # trainer.visualize_results(num_images=5)
    
    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == '__main__':
    main()
