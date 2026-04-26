# 📷 OpenCV 基础学习笔记

> OpenCV 是计算机视觉领域最基础也最重要的工具库。

---

## 📖 目录

1. [OpenCV 简介与安装](#1-opencv-简介与安装)
2. [图像读取与显示](#2-图像读取与显示)
3. [色彩空间转换](#3-色彩空间转换)
4. [图像滤波](#4-图像滤波)
5. [图像变换](#5-图像变换)
6. [实战练习](#6-实战练习)

---

## 1. OpenCV 简介与安装

### 为什么学习 OpenCV？

- **最流行的 CV 库**：被广泛应用于工业界和学术界
- **功能强大**：包含 2500+ 优化算法
- **跨平台**：支持 Windows、Linux、macOS
- **多语言支持**：Python、C++、Java 等
- **CV 基础**：理解 OpenCV 是学习深度学习的前置条件

### 安装

```bash
pip install opencv-python opencv-contrib-python
```

### 导入

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 2. 图像读取与显示

### 2.1 读取图像

```python
import cv2

# 读取图像
image = cv2.imread('image.jpg')

# 读取为灰度图像
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 读取图像并保留 alpha 通道
rgba = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
```

### 2.2 显示图像

```python
# 使用 OpenCV 显示
cv2.imshow('Window Name', image)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()

# 使用 Matplotlib 显示（推荐）
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('图像标题')
plt.axis('off')
plt.show()
```

### 2.3 保存图像

```python
# 保存图像
cv2.imwrite('output.jpg', image)

# 保存为灰度图像
cv2.imwrite('output_gray.jpg', gray)
```

### 2.4 图像属性

```python
image = cv2.imread('image.jpg')

print(f"图像形状：{image.shape}")  # (高度，宽度，通道数)
print(f"图像大小：{image.size}")    # 像素总数
print(f"数据类型：{image.dtype}")   # uint8
```

---

## 3. 色彩空间转换

### 3.1 常见色彩空间

- **BGR**：OpenCV 默认格式（注意：不是 RGB！）
- **RGB**：标准 RGB 格式
- **GRAY**：灰度图像
- **HSV**：色调、饱和度、亮度
- **HLS**：色调、亮度、饱和度
- **LAB**：亮度、a 通道、b 通道

### 3.2 转换方法

```python
# BGR 转 RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# BGR 转灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR 转 HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# BGR 转 LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# 灰度转 BGR
bgr_from_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

### 3.3 HSV 色彩空间应用

```python
import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色范围
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# 创建掩码
mask = cv2.inRange(hsv, lower_red, upper_red)

# 应用掩码
result = cv2.bitwise_and(image, image, mask=mask)
```

---

## 4. 图像滤波

### 4.1 均值滤波

```python
# 3x3 均值滤波
blur = cv2.blur(image, (3, 3))

# 5x5 均值滤波
blur5 = cv2.blur(image, (5, 5))
```

### 4.2 高斯滤波

```python
# 3x3 高斯滤波
gaussian = cv2.GaussianBlur(image, (3, 3), 0)

# 5x5 高斯滤波
gaussian5 = cv2.GaussianBlur(image, (5, 5), 0)

# 指定 sigma 值
gaussian_sigma = cv2.GaussianBlur(image, (0, 0), 2)
```

### 4.3 中值滤波

```python
# 3x3 中值滤波（核大小必须是奇数）
median = cv2.medianBlur(image, 3)

# 5x5 中值滤波
median5 = cv2.medianBlur(image, 5)
```

### 4.4 双边滤波

```python
# 双边滤波（保留边缘）
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
```

### 4.5 滤波对比

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')

# 应用不同滤波
blur = cv2.blur(image, (5, 5))
gaussian = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# 显示对比
plt.figure(figsize=(15, 10))

images = [image, blur, gaussian, median, bilateral]
titles = ['原始图像', '均值滤波', '高斯滤波', '中值滤波', '双边滤波']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. 图像变换

### 5.1 图像缩放

```python
# 按比例缩放
resized = cv2.resize(image, None, fx=0.5, fy=0.5)

# 指定尺寸缩放
resized2 = cv2.resize(image, (400, 300))

# 使用不同插值方法
resized_nearest = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
resized_linear = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
resized_cubic = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
```

### 5.2 图像旋转

```python
# 获取旋转矩阵
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转 45 度

# 应用旋转
rotated = cv2.warpAffine(image, M, (w, h))

# 旋转 90 度
rotated90 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
```

### 5.3 图像平移

```python
# 定义平移矩阵
M = np.float32([[1, 0, 50], [0, 1, 30]])  # 向右 50，向下 30

# 应用平移
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
```

### 5.4 图像翻转

```python
# 水平翻转
h_flip = cv2.flip(image, 1)

# 垂直翻转
v_flip = cv2.flip(image, 0)

# 水平和垂直翻转
hv_flip = cv2.flip(image, -1)
```

---

## 6. 实战练习

### 练习 1：图像基本操作

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = [255, 255, 255]  # 白色背景

# 绘制矩形
cv2.rectangle(image, (50, 50), (250, 250), (0, 0, 255), 3)

# 绘制圆形
cv2.circle(image, (150, 150), 80, (255, 0, 0), -1)

# 绘制直线
cv2.line(image, (0, 0), (300, 300), (0, 255, 0), 2)

# 显示图像
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('绘制的几何图形')
plt.axis('off')
plt.show()
```

### 练习 2：色彩空间转换

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建彩色图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:, :100] = [0, 0, 255]    # 红色
image[:, 100:200] = [0, 255, 0]  # 绿色
image[:, 200:] = [255, 0, 0]     # 蓝色

# 转换到不同色彩空间
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# 显示
plt.figure(figsize=(15, 5))

images = [image, gray, hsv, lab]
titles = ['原始 BGR', '灰度 GRAY', 'HSV', 'LAB']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1, 4, i)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 3：图像滤波效果对比

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建带噪声的图像
np.random.seed(42)
image = np.zeros((200, 200, 3), dtype=np.uint8)
image[50:150, 50:150] = [255, 255, 255]

# 添加高斯噪声
noise = np.random.normal(0, 30, image.shape).astype(np.int16)
noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 应用不同滤波
blur = cv2.blur(noisy, (5, 5))
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
median = cv2.medianBlur(noisy, 5)
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

# 显示
plt.figure(figsize=(15, 10))

images = [noisy, blur, gaussian, median, bilateral]
titles = ['含噪声图像', '均值滤波', '高斯滤波', '中值滤波', '双边滤波']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 4：图像变换操作

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = [255, 255, 255]
cv2.rectangle(image, (50, 50), (250, 250), (0, 0, 255), 3)
cv2.putText(image, 'OpenCV', (80, 160), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

# 缩放
resized = cv2.resize(image, None, fx=0.5, fy=0.5)

# 旋转
(h, w) = image.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

# 翻转
h_flip = cv2.flip(image, 1)

# 显示
plt.figure(figsize=(15, 10))

images = [image, resized, rotated, h_flip]
titles = ['原始图像', '缩小 (0.5x)', '旋转 45°', '水平翻转']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 2, i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 5：HSV 颜色分割

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建包含多种颜色的图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:, :100] = [0, 0, 255]      # 红色
image[:, 100:200] = [0, 255, 0]   # 绿色
image[:, 200:] = [255, 0, 0]      # 蓝色

# 转换到 HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义颜色范围
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 创建掩码
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# 应用掩码
result_red = cv2.bitwise_and(image, image, mask=mask_red)
result_green = cv2.bitwise_and(image, image, mask=mask_green)
result_blue = cv2.bitwise_and(image, image, mask=mask_blue)

# 显示
plt.figure(figsize=(15, 10))

images = [image, mask_red, mask_green, mask_blue, 
          result_red, result_green, result_blue]
titles = ['原始图像', '红色掩码', '绿色掩码', '蓝色掩码',
          '红色提取', '绿色提取', '蓝色提取']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(4, 2, i)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 📝 练习答案

### 练习 1 答案

```python
# 使用 cv2.rectangle, cv2.circle, cv2.line 绘制几何图形
cv2.rectangle(image, (50, 50), (250, 250), (0, 0, 255), 3)
```

### 练习 2 答案

```python
# 使用 cv2.cvtColor 转换色彩空间
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

### 练习 3 答案

```python
# 添加噪声并应用不同滤波
noise = np.random.normal(0, 30, image.shape).astype(np.int16)
noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

### 练习 4 答案

```python
# 使用 cv2.resize, cv2.getRotationMatrix2D, cv2.flip
rotated = cv2.warpAffine(image, M, (w, h))
```

### 练习 5 答案

```python
# 使用 cv2.inRange 创建颜色掩码
mask_red = cv2.inRange(hsv, lower_red, upper_red)
```

---

## 🎯 关键要点总结

1. **OpenCV 使用 BGR 格式**，不是 RGB！
2. **显示图像时**需要转换为 RGB：`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
3. **滤波方法选择**：
   - 均值滤波：简单快速
   - 高斯滤波：平滑效果好
   - 中值滤波：去除椒盐噪声
   - 双边滤波：保留边缘
4. **HSV 色彩空间**更适合颜色分割
5. **图像变换**使用仿射变换矩阵

> 💡 **下一步**：学习 OpenCV 进阶，掌握边缘检测、轮廓提取、形态学操作！

---

## 📚 扩展学习资源

- [OpenCV 官方文档](https://docs.opencv.org/4.x/)
- [OpenCV-Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- 《Python 计算机视觉编程》

---

> 🌟 **记住**：OpenCV 是计算机视觉的基石，熟练掌握图像处理技术将让后续学习事半功倍！
