# ✂️ 4.3 图像分割实战

> 像素级别的图像理解！

---

## 📖 学习目标

- [x] 语义分割 vs 实例分割
- [x] U-Net 架构
- [x] DeepLab 系列简介
- [x] 分割评估（IoU 指标）
- [x] 数据集标注和准备

---

## 📁 项目文件

- ✅ `学习笔记.md` - 理论知识和代码示例
- ✅ `Practice.py` - 实战练习代码
- ✅ `datasets/` - 数据集目录
  - `images/train/` - 训练集图片
  - `images/val/` - 验证集图片
  - `masks/train/` - 训练集掩码
  - `masks/val/` - 验证集掩码
  - `README.md` - 数据集使用指南

---

## 🎯 实战项目

### 项目选项

#### 1. **人像抠图** - 自动分离人像和背景 ⭐⭐
**场景**: 人像模式、虚拟背景
- 使用 Portrait 数据集
- 二分类分割（人/背景）
- 实时处理

#### 2. **医学图像分割** - 识别病变区域 ⭐⭐⭐
**场景**: 医疗影像分析
- 肺部分割/肿瘤分割
- 高精度要求
- 使用医学影像数据集

#### 3. **道路场景分割** - 自动驾驶应用 ⭐⭐⭐
**场景**: 自动驾驶、辅助驾驶
- 道路、车辆、行人分割
- 多类别分割
- 使用 Cityscapes 数据集

---

## 📦 需要的库

```bash
# 核心库
pip install torch torchvision

# 分割模型
pip install segmentation-models-pytorch

# 图像处理
pip install opencv-python pillow

# 可视化
pip install matplotlib

# 标注工具
pip install labelme
```

---

## 💡 核心概念

### 图像分割 vs 图像分类 vs 目标检测

```
图像分类：
输入：一张图
输出：一个标签（猫）

目标检测：
输入：一张图
输出：多个边界框 + 类别
  - 猫 [x1, y1, x2, y2]
  - 狗 [x1, y1, x2, y2]

图像分割：
输入：一张图
输出：每个像素的类别
  - 像素 (0,0): 猫
  - 像素 (0,1): 猫
  - 像素 (0,2): 背景
  - ...
```

### 语义分割 vs 实例分割

```
语义分割：
- 为每个像素分配类别
- 不区分同一类别的不同实例
- 所有猫都标记为"猫"

实例分割：
- 不仅区分类别，还区分个体
- 每只猫有独立 ID
- 猫 1、猫 2
```

### U-Net 架构

```
编码器（下采样）        解码器（上采样）
     ↓                      ↑
  Conv → Pool          UpConv
     ↓                      ↑
  Conv → Pool    →→→→→→  Conv (跳跃连接)
     ↓                      ↑
  Bottleneck      →→→→→→  UpConv
                        ↓
                      输出
```

### IoU（Intersection over Union）

```
IoU = 预测区域 ∩ 真实区域 / 预测区域 ∪ 真实区域

IoU = 1: 完美重合
IoU > 0.5: 通常认为是成功检测
IoU = 0: 完全不重叠

mIoU（平均 IoU）：所有类别的 IoU 平均值
```

---

## ⏱️ 预计时间

4-6 天

---

## ✅ 完成标准

- [x] 理解语义分割与实例分割的区别
- [x] 能够搭建 U-Net 模型
- [x] 掌握 IoU 指标计算
- [x] 能够标注分割数据集
- [x] 训练并评估分割模型
- [x] 可视化分割结果

---

## 🚀 快速开始

### 1. 使用 U-Net 训练

```bash
cd 4.3_图像分割实战
python Practice.py
```

### 2. 准备数据集

```bash
# 安装标注工具
pip install labelme

# 启动标注
labelme
```

### 3. 训练自定义模型

```python
from Practice import SegmentationTrainer

config = {
    'data_dir': 'datasets',
    'model_type': 'unet',
    'num_classes': 2,
    'img_size': 256,
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda'
}

trainer = SegmentationTrainer(config)
trainer.create_model('unet')
trainer.prepare_data()
trainer.train()
```

### 4. 使用 DeepLabv3+

```python
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(
    weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1'
)

# 修改分类器
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
```

---

## 📊 实验记录

| 实验 | 模型 | 数据集 | Epochs | mIoU | Dice | 备注 |
|------|------|--------|--------|------|------|------|
| 1 | U-Net | 自定义 | 50 | - | - | baseline |
| 2 | DeepLabv3+ | 自定义 | 50 | - | - | 预训练 |
| 3 | U-Net++ | 自定义 | 50 | - | - | 改进架构 |

---

## 🔗 参考资源

- [U-Net 论文](https://arxiv.org/abs/1505.04597)
- [DeepLab 系列](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [LabelMe 标注工具](https://github.com/wkentaro/labelme)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Pascal VOC 数据集](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Cityscapes 数据集](https://www.cityscapes-dataset.com/)

---

**状态**: ✅ 已完成基础设置

**最后更新**: 2026-04-26
