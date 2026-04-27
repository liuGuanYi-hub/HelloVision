# 🔍 4.2 目标检测实战

> 让计算机能够定位和识别图像中的多个物体！

---

## 📖 学习目标

- [x] 目标检测原理（边界框、锚点）
- [x] YOLO 系列简介
- [x] 使用 ultralytics 库（YOLOv8）
- [x] 自定义数据集标注
- [x] 实时检测

---

## 📁 项目文件

- ✅ `学习笔记.md` - 理论知识和代码示例
- ✅ `Practice.py` - 实战练习代码
- ✅ `datasets/` - 数据集目录
  - `images/train/` - 训练集图片
  - `images/val/` - 验证集图片
  - `labels/train/` - 训练集标签
  - `labels/val/` - 验证集标签
  - `data.yaml` - 数据集配置文件
  - `README.md` - 数据集使用指南

---

## 🎯 实战项目

### 项目选项

#### 1. **通用物体检测** - 检测 80 种常见物体 ⭐⭐
**场景**: 使用 COCO 预训练模型
- 加载 YOLOv8n.pt
- 检测日常物体
- 实时摄像头演示

#### 2. **人脸检测** - 定位人脸位置 ⭐⭐⭐
**场景**: 安防、考勤系统
- 收集人脸数据
- 标注数据集
- 训练专用模型
- 评估准确率

#### 3. **交通标志检测** - 自动驾驶应用 ⭐⭐⭐
**场景**: 智能驾驶辅助
- 交通标志数据集
- 多类别检测
- 实时识别

---

## 📦 需要的库

```bash
# 核心库
pip install ultralytics  # YOLOv8

# 图像处理
pip install opencv-python pillow

# 可视化
pip install matplotlib

# 标注工具
pip install labelimg
```

---

## 💡 核心概念

### 目标检测 vs 图像分类

```
图像分类：
输入：一张图
输出：一个标签（猫）

目标检测：
输入：一张图
输出：多个物体 + 位置
  - 猫 (95%, [x1, y1, x2, y2])
  - 狗 (92%, [x1, y1, x2, y2])
  - 沙发 (88%, [x1, y1, x2, y2])
```

### YOLO 检测流程

```
输入图像 → CNN 提取特征 → 特征金字塔 → 
检测头预测 → NMS 去重 → 输出结果
```

### YOLOv8 模型规格

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | 最快 | 中等 | 移动端、实时 |
| YOLOv8s | 11.2M | 快 | 良好 | 平衡性能 |
| YOLOv8m | 25.9M | 中等 | 很好 | 精度优先 |
| YOLOv8l | 43.7M | 慢 | 优秀 | 离线处理 |
| YOLOv8x | 68.2M | 最慢 | 最佳 | 最高精度 |

---

## ⏱️ 预计时间

4-6 天

---

## ✅ 完成标准

- [x] 理解目标检测与图像分类的区别
- [x] 能够使用 YOLOv8 进行推理
- [x] 掌握数据集标注方法
- [x] 能够训练自定义检测模型
- [x] 实现实时摄像头检测
- [ ] 完成一个实际项目（人脸/交通标志等）

---

## 🚀 快速开始

### 1. 使用预训练模型

```bash
cd 4.2_目标检测实战
python Practice.py
```

### 2. 准备自定义数据集

```bash
# 安装标注工具
pip install labelimg

# 启动标注
labelimg
```

### 3. 训练自定义模型

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='datasets/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='0'  # 使用 GPU
)
```

### 4. 实时检测

```python
# 摄像头检测
model.predict(source=0, show=True)

# 视频文件
model.predict(source='video.mp4', save=True)
```

---

## 📊 实验记录

| 实验 | 模型 | 数据集 | Epochs | mAP50 | mAP50-95 | FPS | 备注 |
|------|------|--------|--------|-------|----------|-----|------|
| 1 | YOLOv8n | COCO | 100 | - | - | - | 预训练 baseline |
| 2 | YOLOv8n | 自定义 | 100 | - | - | - | 迁移学习 |
| 3 | YOLOv8s | 自定义 | 100 | - | - | - | 更大模型 |

---

## 🔗 参考资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [LabelImg 标注工具](https://github.com/heartexlabs/labelImg)
- [COCO 数据集](https://cocodataset.org/)

---

**状态**: ✅ 已完成基础设置

**最后更新**: 2026-04-26
