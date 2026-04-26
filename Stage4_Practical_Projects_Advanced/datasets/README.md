# 📊 Stage 4 数据集使用指南

> 数据集准备、下载和使用说明！

---

## 📁 数据集目录结构

```
datasets/
├── cats_vs_dogs/          # 猫狗分类数据集
│   ├── train/
│   │   ├── cats/          # 猫训练图像
│   │   └── dogs/          # 狗训练图像
│   ├── val/
│   │   ├── cats/          # 猫验证图像
│   │   └── dogs/          # 狗验证图像
│   └── test/
│       ├── cats/          # 猫测试图像
│       └── dogs/          # 狗测试图像
├── gestures/              # 手势识别数据集
│   ├── train/
│   ├── val/
│   └── test/
└── flowers/               # 花卉识别数据集
    ├── train/
    ├── val/
    └── test/
```

---

## 📥 数据集下载

### 方式 1: Kaggle 下载（推荐）

#### 猫狗分类数据集

```bash
# 安装 Kaggle CLI
pip install kaggle

# 下载猫狗数据集
kaggle datasets download -d tongpython/notebook1404cc3197-0

# 或使用 Kaggle 经典猫狗数据集
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
```

#### 花卉识别数据集

```bash
# 下载花卉数据集
kaggle datasets download -d alxmama/flowers-recognition
```

### 方式 2: 手动下载

1. **Kaggle 猫狗数据集**: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
2. **花卉数据集**: https://www.kaggle.com/datasets/alxmama/flowers-recognition
3. **手势识别数据集**: https://www.kaggle.com/datasets/grassknoted/american-sign-language-letters

### 方式 3: 使用 PyTorch 内置数据集

```python
from torchvision import datasets

# CIFAR-10（10 类物体）
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)

# MNIST（手写数字）
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True
)
```

---

## 🔧 数据集准备

### 猫狗分类数据集

1. **下载数据集**
2. **解压到 `datasets/cats_vs_dogs/` 目录**
3. **重新组织结构**（如需要）：

```python
import os
import shutil
from sklearn.model_selection import train_test_split

def organize_cats_dogs_dataset(source_dir, target_dir):
    """
    重新组织猫狗数据集结构
    """
    # 创建目标目录
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)
    
    # 读取源数据
    cats = [f for f in os.listdir(os.path.join(source_dir, 'train', 'cats')) 
            if f.endswith(('.jpg', '.png'))]
    dogs = [f for f in os.listdir(os.path.join(source_dir, 'train', 'dogs')) 
            if f.endswith(('.jpg', '.png'))]
    
    # 划分数据集
    cats_train, cats_temp = train_test_split(cats, test_size=0.3, random_state=42)
    cats_val, cats_test = train_test_split(cats_temp, test_size=0.5, random_state=42)
    
    dogs_train, dogs_temp = train_test_split(dogs, test_size=0.3, random_state=42)
    dogs_val, dogs_test = train_test_split(dogs_temp, test_size=0.5, random_state=42)
    
    # 复制文件
    for split, cat_list, dog_list in [
        ('train', cats_train, dogs_train),
        ('val', cats_val, dogs_val),
        ('test', cats_test, dogs_test)
    ]:
        for cat in cat_list:
            shutil.copy(
                os.path.join(source_dir, 'train', 'cats', cat),
                os.path.join(target_dir, split, 'cats', cat)
            )
        for dog in dog_list:
            shutil.copy(
                os.path.join(source_dir, 'train', 'dogs', dog),
                os.path.join(target_dir, split, 'dogs', dog)
            )
    
    print(f"数据集组织完成！")
    print(f"训练集：猫 {len(cats_train)}, 狗 {len(dogs_train)}")
    print(f"验证集：猫 {len(cats_val)}, 狗 {len(dogs_val)}")
    print(f"测试集：猫 {len(cats_test)}, 狗 {len(dogs_test)}")

# 使用示例
# organize_cats_dogs_dataset('raw_data', 'datasets/cats_vs_dogs')
```

---

## 📊 推荐数据集配置

### 入门级（快速实验）

| 数据集 | 类别数 | 训练集 | 验证集 | 测试集 | 推荐用途 |
|--------|--------|--------|--------|--------|----------|
| 猫狗分类 | 2 | 1,000 | 200 | 200 | 二分类入门 |
| 手势识别 | 10 | 2,000 | 400 | 400 | 多分类入门 |

### 进阶级（生产级）

| 数据集 | 类别数 | 训练集 | 验证集 | 测试集 | 推荐用途 |
|--------|--------|--------|--------|--------|----------|
| 花卉识别 | 5 | 3,000 | 600 | 600 | 细粒度分类 |
| 食物分类 | 10 | 5,000 | 1,000 | 1,000 | 实用场景 |
| 动物分类 | 20 | 10,000 | 2,000 | 2,000 | 复杂场景 |

---

## 🎯 数据增强建议

### 针对不同数据集的增强策略

#### 猫狗分类
```python
from albumentations import Compose, HorizontalFlip, RandomRotate90

# 猫狗对左右翻转不敏感
train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.3),
    # ... 其他增强
])
```

#### 手势识别
```python
# 手势对方向敏感，谨慎使用翻转
train_transform = Compose([
    # 不使用 HorizontalFlip
    RandomRotate90(p=0.2),
    # ... 其他增强
])
```

#### 花卉识别
```python
# 花卉对旋转、翻转都不敏感
train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    # ... 其他增强
])
```

---

## 📝 数据集检查清单

在开始训练前，请确保：

- [ ] 数据集已下载到正确目录
- [ ] 目录结构符合 ImageFolder 格式
- [ ] 每个类别都有足够的样本（建议每类>100 张）
- [ ] 训练集、验证集、测试集已正确划分
- [ ] 图像格式统一（建议 JPG 或 PNG）
- [ ] 图像大小合理（建议边长>224 像素）

---

## 🔍 常见问题

### Q1: 数据集太小怎么办？

**A**: 使用以下策略：
1. 更强的数据增强
2. 使用预训练模型（迁移学习）
3. 冻结更多层（feature extract 模式）
4. 考虑使用生成对抗网络（GAN）生成合成数据

### Q2: 类别不平衡怎么办？

**A**: 使用以下方法：
1. 过采样少数类
2. 欠采样多数类
3. 使用加权损失函数
4. 数据增强少数类

### Q3: 如何验证数据集质量？

**A**: 运行以下检查：
```python
import os
from PIL import Image

def check_dataset_quality(data_dir):
    """检查数据集质量"""
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ {split} 目录不存在")
            continue
        
        classes = os.listdir(split_dir)
        print(f"\n{split} 集:")
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            
            images = [f for f in os.listdir(cls_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            # 检查图像是否可打开
            valid_count = 0
            for img_name in images[:10]:  # 检查前 10 张
                try:
                    img = Image.open(os.path.join(cls_dir, img_name))
                    img.verify()
                    valid_count += 1
                except:
                    print(f"  ⚠️ {cls}/{img_name} 损坏")
            
            print(f"  {cls}: {len(images)} 张图像，抽样检查有效 {valid_count}/{min(10, len(images))}")

# 使用
# check_dataset_quality('datasets/cats_vs_dogs')
```

---

**最后更新**: 2026-04-26
