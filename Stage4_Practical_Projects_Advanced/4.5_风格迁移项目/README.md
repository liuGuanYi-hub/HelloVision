# 🎨 4.5 风格迁移项目

> 把照片变成世界名画！

---

## 📖 学习目标

- [x] 神经风格迁移原理
- [x] Gram 矩阵计算
- [x] 内容损失 vs 风格损失
- [x] 实时风格迁移
- [x] 多风格模型训练

---

## 📁 项目文件

- ✅ `学习笔记.md` - 理论知识和代码示例
- ✅ `Practice.py` - 实战练习代码
- ✅ `images/` - 图像目录
  - `content/` - 内容图像
  - `style/` - 风格图像
- ✅ `results/` - 结果保存目录
- ✅ `models/` - 模型保存目录

---

## 🎯 实战项目

### 项目选项

#### 1. **照片变名画风格** - 梵高、莫奈等 ⭐⭐
**场景**: 将自己的照片转换成艺术画作
- 输入：内容图像 + 风格图像
- 输出：风格化后的图像
- 支持：梵高《星夜》、莫奈《睡莲》等
- 调整：内容和风格权重比例

#### 2. **实时风格迁移** - 视频实时处理 ⭐⭐⭐
**场景**: 摄像头实时风格化
- 使用快速风格迁移网络
- 达到 30+ FPS
- 支持多风格切换
- 可处理视频流

#### 3. **自定义风格** - 创造独特滤镜 ⭐⭐
**场景**: 创建个人专属风格
- 用自己的画作作为风格
- 训练专用风格模型
- 制作独特滤镜效果
- 应用到社交媒体

---

## 📚 核心概念

### 什么是神经风格迁移？

神经风格迁移 = 将一幅图像的风格（纹理、颜色）迁移到另一幅图像的内容上。

**核心思想**:
- 使用预训练 CNN（如 VGG19）
- 内容表示：高层特征图
- 风格表示：Gram 矩阵（特征图相关性）
- 优化生成图像，最小化内容损失 + 风格损失

### 损失函数

```
总损失 = α × 内容损失 + β × 风格损失

内容损失：生成图像与内容图像的特征差异
风格损失：生成图像与风格图像的 Gram 矩阵差异

α 和 β 控制风格化程度：
- α/β 大 → 更像内容图
- α/β 小 → 更像风格图
```

### Gram 矩阵

```
Gram 矩阵 = 特征图之间的相关性

计算步骤:
1. 提取 CNN 某层的特征图 (C 个通道，H×W 空间)
2. 展平空间维度 (C, H×W)
3. 计算自相关矩阵 G = F @ F.T
4. G[i,j] 表示通道 i 和 j 的相关性

物理意义:
- 对角线元素：每个特征的强度
- 非对角线元素：特征间的共现关系
- 捕捉纹理和图案信息
```

---

## 📦 需要的库

```bash
# 核心库
pip install torch torchvision

# 图像处理
pip install opencv-python pillow

# 可视化
pip install matplotlib

# 进度条
pip install tqdm
```

---

## 💡 关键技术

### 1. VGG 特征提取

```python
import torchvision.models as models

# 加载预训练 VGG19
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
vgg.eval()

# 冻结参数
for param in vgg.parameters():
    param.requires_grad = False
```

### 2. 损失计算

```python
def content_loss(content_feat, target_feat):
    """内容损失：MSE"""
    return torch.mean((content_feat - target_feat) ** 2)

def style_loss(style_feat, target_feat):
    """风格损失：Gram 矩阵的 MSE"""
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(c * h * w)
    
    gram_style = gram_matrix(style_feat)
    gram_target = gram_matrix(target_feat)
    return torch.mean((gram_style - gram_target) ** 2)
```

### 3. 优化生成图像

```python
# 初始化生成图像为内容图像
generated = content_image.clone().requires_grad_(True)

# 优化器
optimizer = torch.optim.Adam([generated], lr=0.003)

# 迭代优化
for step in range(num_steps):
    optimizer.zero_grad()
    
    # 提取特征
    gen_features = vgg(generated)
    
    # 计算损失
    loss = alpha * content_loss(...) + beta * style_loss(...)
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

---

## ⏱️ 预计时间

3-4 天

---

## ✅ 完成标准

- [x] 理解风格迁移原理
- [x] 能够实现神经风格迁移
- [x] 计算 Gram 矩阵
- [x] 调整内容和风格权重
- [x] 制作自己的艺术滤镜
- [x] 完成 3+ 种风格示例

---

## 🚀 快速开始

### 1. 准备图像

```bash
# 将内容图片放到 images/content/
# 将风格图片放到 images/style/
```

### 2. 运行风格迁移

```bash
cd 4.5_风格迁移项目
python Practice.py
```

### 3. 调整参数

```python
# 修改 Practice.py 中的权重
config = {
    'content_weight': 1.0,      # 内容权重
    'style_weight': 1e6,        # 风格权重
    'num_steps': 1000,          # 迭代次数
    'learning_rate': 0.003,     # 学习率
    'image_size': 512           # 图像大小
}
```

---

## 📊 实验记录模板

| 实验 | 风格图像 | α:β | 迭代次数 | 效果 | 耗时 | 备注 |
|------|----------|-----|----------|------|------|------|
| 1 | 星夜 | 1:1000 | 1000 | - | 5min | baseline |
| 2 | 睡莲 | 1:500 | 1000 | - | 5min | 降低风格权重 |
| 3 | 自定义 | 1:800 | 1500 | - | 8min | 增加迭代 |

---

## 🎨 风格示例参考

### 经典艺术风格
- 梵高 - 《星夜》
- 莫奈 - 《睡莲》
- 毕加索 - 立体主义
- 葛饰北斋 - 《神奈川冲浪里》

### 现代风格
- 赛博朋克
- 水彩画
- 素描
- 像素艺术

---

**状态**: 🚧 开发中

**最后更新**: 2026-04-26
