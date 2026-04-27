# 🔄 4.4 迁移学习实战

> 用少量数据训练专业模型！

---

## 📖 学习目标

- [x] 为什么需要迁移学习
- [x] 特征提取 vs 微调
- [x] 冻结层策略
- [x] 小样本学习技巧
- [x] 领域自适应

---

## 📁 项目文件

- ✅ `学习笔记.md` - 理论知识和代码示例
- ✅ `Practice.py` - 实战练习代码
- ✅ `datasets/` - 数据集目录
  - `train/` - 训练集图片
  - `val/` - 验证集图片
- ✅ `models/` - 模型保存目录

---

## 🎯 实战项目

### 项目选项

#### 1. **用 100 张图片训练分类器** - 小样本学习 ⭐⭐
**场景**: 你只有少量专业领域的图像数据
- 每类 50-100 张图片
- 使用强预训练模型（EfficientNet）
- 特征提取 + 微调结合
- 目标：准确率>85%

#### 2. **特定领域迁移** - 医疗/工业应用 ⭐⭐⭐
**场景**: 医疗影像或工业质检
- 医学：X 光片异常检测
- 工业：产品缺陷检测
- 使用 ImageNet 预训练权重
- 领域自适应技巧

#### 3. **多模型对比** - ResNet vs EfficientNet ⭐⭐
**场景**: 选择最佳预训练模型
- 对比不同架构
- 分析冻结层数影响
- 学习率策略对比
- 性能 vs 速度权衡

---

## 📚 核心概念

### 什么是迁移学习？

迁移学习 = 将在大数据集（如 ImageNet）上学到的知识，应用到小数据集任务中。

**为什么有效？**
- CNN 底层学习通用特征（边缘、纹理）
- 这些特征在不同任务间可复用
- 只需调整高层特征适配新任务

### 三种迁移策略

```
策略 1: 特征提取（Feature Extraction）
┌─────────────────────┐
│  预训练 CNN (冻结)  │ → 提取特征
└─────────────────────┘
         ↓
┌─────────────────────┐
│  新分类器 (训练)    │ → 预测
└─────────────────────┘

适用场景：数据量小（<1000 张），类别相似


策略 2: 微调（Fine-tuning）
┌─────────────────────┐
│  预训练 CNN (部分)  │ → 微调
└─────────────────────┘
         ↓
┌─────────────────────┐
│  新分类器 (训练)    │ → 预测
└─────────────────────┘

适用场景：数据量中等（1000-10000 张）


策略 3: 全量微调（Full Fine-tuning）
┌─────────────────────┐
│  预训练 CNN (全部)  │ → 微调
└─────────────────────┘
         ↓
┌─────────────────────┐
│  新分类器 (训练)    │ → 预测
└─────────────────────┘

适用场景：数据量大（>10000 张）
```

---

## 📦 需要的库

```bash
# 核心库
pip install torch torchvision

# 图像处理
pip install opencv-python pillow

# 可视化
pip install matplotlib seaborn

# 进度条
pip install tqdm
```

---

## 💡 关键技术

### 1. 冻结层策略

```python
# 冻结 backbone
for param in model.features.parameters():
    param.requires_grad = False

# 只训练分类头
for param in model.classifier.parameters():
    param.requires_grad = True
```

### 2. 分层学习率

```python
# backbone 用小学习率
# 分类头用大学习率
optimizer = torch.optim.SGD([
    {'params': model.features.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-2}
], momentum=0.9)
```

### 3. 渐进式解冻

```python
# 先训练分类头
# 然后解冻最后几层微调
# 逐步提高性能
```

---

## ⏱️ 预计时间

3-4 天

---

## ✅ 完成标准

- [x] 理解迁移学习的优势
- [x] 掌握特征提取方法
- [x] 掌握微调技术
- [x] 用 100 张图片训练出可用模型
- [x] 能够迁移到自己的领域
- [x] 完成实验对比和记录

---

## 🚀 快速开始

### 1. 准备小数据集

```bash
# 每类准备 50-100 张图片
# 放到 datasets/train/ 和 datasets/val/
```

### 2. 运行训练

```bash
cd 4.4_迁移学习实战
python Practice.py
```

### 3. 调整策略

```python
config = {
    'strategy': 'fine_tune',  # feature_extract / fine_tune / full_fine_tune
    'model_name': 'resnet50',  # resnet18 / resnet50 / efficientnet_b0
    'num_epochs': 20,
    'learning_rate': 0.001
}
```

---

## 📊 实验记录模板

| 实验 | 模型 | 策略 | 数据量 | Epochs | 准确率 | 备注 |
|------|------|------|--------|--------|--------|------|
| 1 | ResNet18 | 特征提取 | 100 | 10 | - | baseline |
| 2 | ResNet50 | 微调 | 100 | 10 | - | 解冻 layer4 |
| 3 | EfficientNet | 特征提取 | 100 | 10 | - | 更强 backbone |

---

**状态**: ✅ 已完成基础设置

**最后更新**: 2026-04-26
