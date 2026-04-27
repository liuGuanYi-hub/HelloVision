#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.5 风格迁移实战 - 完整练习代码

项目：神经风格迁移
功能：
    - 基于优化的风格迁移（Gatys et al.）
    - Gram 矩阵计算
    - 内容和风格损失
    - 多风格支持
    - 结果可视化

作者：Your Name
日期：2026-04-26
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


class StyleTransfer:
    """神经风格迁移"""
    
    def __init__(self, content_weight=1.0, style_weight=1e6):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载 VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # 把VGG分成几段用于获取中间层输出
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # conv1_1 (0) -> conv2_1 (5) -> conv3_1 (10) -> conv4_1 (19) -> conv5_1 (28)
        for i in range(0, 6):
            self.slice1.add_module(str(i), vgg[i])
        for i in range(6, 11):
            self.slice2.add_module(str(i), vgg[i])
        for i in range(11, 20):
            self.slice3.add_module(str(i), vgg[i])
        for i in range(20, 29):
            self.slice4.add_module(str(i), vgg[i])
        for i in range(29, len(list(vgg.children()))):
            try:
                self.slice5.add_module(str(i), vgg[i])
            except:
                pass
        
        self.slice1 = self.slice1.to(self.device)
        self.slice2 = self.slice2.to(self.device)
        self.slice3 = self.slice3.to(self.device)
        self.slice4 = self.slice4.to(self.device)
        self.slice5 = self.slice5.to(self.device)
        
        for param in self.slice1.parameters():
            param.requires_grad = False
        for param in self.slice2.parameters():
            param.requires_grad = False
        for param in self.slice3.parameters():
            param.requires_grad = False
        for param in self.slice4.parameters():
            param.requires_grad = False
        for param in self.slice5.parameters():
            param.requires_grad = False
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print(f"VGG19 loaded! Device: {self.device}")
    
    def load_image(self, path, size=512):
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def get_features(self, x):
        """获取各层特征"""
        h1 = self.slice1(x)   # conv1_1
        h2 = self.slice2(h1)  # conv2_1
        h3 = self.slice3(h2)  # conv3_1
        h4 = self.slice4(h3)  # conv4_1
        h5 = self.slice5(h4) # conv5_1
        return {'conv1_1': h1, 'conv2_1': h2, 'conv3_1': h3, 'conv4_1': h4, 'conv5_1': h5}
    
    def content_loss_fn(self, gen_feat, content_feat):
        return torch.mean((gen_feat - content_feat) ** 2)
    
    def style_loss_fn(self, gen_feat, style_feat):
        gen_gram = self.gram_matrix(gen_feat)
        style_gram = self.gram_matrix(style_feat.detach())  # detach style
        return torch.mean((gen_gram - style_gram) ** 2)
    
    def transfer(self, content_path, style_path, num_steps=300,
                 lr=0.02, size=512, save_interval=50):
        print("\n" + "="*50)
        print("Style Transfer Start")
        print("="*50)
        
        content_img = self.load_image(content_path, size)
        style_img = self.load_image(style_path, size)
        
        # 使用内容图像初始化
        gen_img = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([gen_img], lr=lr)
        
        content_feat = self.get_features(content_img)
        style_feat = self.get_features(style_img)
        
        best_loss = float('inf')
        os.makedirs('results', exist_ok=True)
        
        start = time.time()
        for step in range(num_steps):
            optimizer.zero_grad()
            
            gen_feat = self.get_features(gen_img)
            
            # 内容损失 - 使用 conv4_2 近似 (conv4_1)
            c_loss = self.content_loss_fn(gen_feat['conv4_1'], content_feat['conv4_1'])
            
            # 风格损失 - 多层
            s_loss = 0.0
            for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                s_loss += self.style_loss_fn(gen_feat[layer], style_feat[layer])
            
            total_loss = self.content_weight * c_loss + self.style_weight * s_loss
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"Step {step}/{num_steps}: Loss={total_loss.item():.2f}, "
                      f"Content={c_loss.item():.4f}, Style={s_loss.item():.2f}, "
                      f"Time={elapsed:.1f}s")
            
            if step % save_interval == 0 and step > 0:
                self.save_image(gen_img, f'results/result_step_{step}.jpg')
        
        self.save_image(gen_img, 'results/final_result.jpg')
        print(f"\nStyle transfer done! Time: {time.time()-start:.1f}s")
        print(f"Best loss: {best_loss:.2f}")
        print(f"Result saved to results/final_result.jpg")
        
        return gen_img
    
    def save_image(self, tensor, path):
        t = tensor.detach().cpu()
        # Remove batch dim first
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        # Now t is (C, H, W)
        # Denormalize: broadcast std (1,3,1,1) and mean (1,3,1,1) over (C,H,W)
        std_cpu = self.std.cpu()  # (1,3,1,1)
        mean_cpu = self.mean.cpu()  # (1,3,1,1)
        # Reshape to (1,3,1,1) to broadcast with (C,H,W)=(3,384,384)
        t = t.unsqueeze(0) * std_cpu + mean_cpu
        t = t.squeeze(0)  # Back to (C, H, W)
        t = t.clamp(0, 1)
        # Convert to numpy (C, H, W) -> (H, W, C)
        import numpy as np
        np_img = t.numpy()
        np_img = np.transpose(np_img, (1, 2, 0))
        np_img = (np_img * 255).astype(np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(np_img)
        pil_img.save(path)
        print(f"Image saved: {path}")


def main():
    """主函数 - 批量处理images目录中的图"""
    print("="*50)
    print("4.5 Style Transfer")
    print("="*50)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    content_dir = os.path.join(base_dir, 'images', 'content')
    style_dir = os.path.join(base_dir, 'images', 'style')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    content_images = sorted([f for f in os.listdir(content_dir) 
                              if f.lower().endswith(('.jpg','.jpeg','.png'))])
    style_images = sorted([f for f in os.listdir(style_dir) 
                            if f.lower().endswith(('.jpg','.jpeg','.png'))])
    
    print(f"Found {len(content_images)} content images")
    print(f"Found {len(style_images)} style images")
    
    if not content_images or not style_images:
        print("No images found!")
        return
    
    st = StyleTransfer(content_weight=1.0, style_weight=1e5)
    
    # 用前3张内容图 x 前2张风格图 = 6种组合
    n_content = min(3, len(content_images))
    n_style = min(2, len(style_images))
    
    for i in range(n_content):
        for j in range(n_style):
            c_path = os.path.join(content_dir, content_images[i])
            s_path = os.path.join(style_dir, style_images[j])
            print(f"\n[{i+1}/{n_content}] {content_images[i]} + [{j+1}/{n_style}] {style_images[j]}")
            
            try:
                result = st.transfer(
                    content_path=c_path,
                    style_path=s_path,
                    num_steps=300,
                    lr=0.02,
                    size=384,
                    save_interval=100
                )
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print("\n" + "="*50)
    print("All transfers done!")
    print("="*50)


if __name__ == '__main__':
    main()