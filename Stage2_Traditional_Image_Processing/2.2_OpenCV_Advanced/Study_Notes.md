# 📷 OpenCV 进阶学习笔记

> 掌握边缘检测、轮廓提取、形态学操作等高级图像处理技术。

---

## 📖 目录

1. [边缘检测](#1-边缘检测)
2. [轮廓提取](#2-轮廓提取)
3. [形态学操作](#3-形态学操作)
4. [直方图处理](#4-直方图处理)
5. [图像阈值](#5-图像阈值)
6. [实战练习](#6-实战练习)

---

## 1. 边缘检测

### 1.1 Canny 边缘检测

```python
import cv2
import numpy as np

# 读取图像并转换为灰度
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny 边缘检测
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# 参数说明：
# threshold1: 滞后阈值 1（低阈值）
# threshold2: 滞后阈值 2（高阈值）
```

### 1.2 Sobel 边缘检测

```python
# Sobel X 方向
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.uint8(np.absolute(sobel_x))

# Sobel Y 方向
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = np.uint8(np.absolute(sobel_y))

# 合并 X 和 Y 方向
sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
```

### 1.3 Laplacian 边缘检测

```python
# Laplacian 边缘检测
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
```

### 1.4 边缘检测方法对比

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), 255, 3)
cv2.circle(image, (150, 150), 80, 255, 3)

# 应用不同边缘检测
canny = cv2.Canny(image, 50, 150)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.uint8(np.absolute(sobel_x))
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 显示对比
plt.figure(figsize=(15, 5))

images = [image, canny, sobel_x, laplacian]
titles = ['原始图像', 'Canny', 'Sobel X', 'Laplacian']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1, 4, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 2. 轮廓提取

### 2.1 查找轮廓

```python
# 阈值处理
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 参数说明：
# cv2.RETR_EXTERNAL: 只检索最外层轮廓
# cv2.RETR_TREE: 检索所有轮廓并重建完整层次
# cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段
```

### 2.2 绘制轮廓

```python
# 复制图像
image_copy = image.copy()

# 绘制所有轮廓
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

# 绘制单个轮廓
cv2.drawContours(image_copy, contours, 0, (255, 0, 0), 3)
```

### 2.3 轮廓属性

```python
# 遍历所有轮廓
for i, contour in enumerate(contours):
    # 轮廓面积
    area = cv2.contourArea(contour)
    
    # 轮廓周长
    perimeter = cv2.arcLength(contour, True)
    
    # 轮廓近似
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(contour)
    
    print(f"轮廓 {i}: 面积={area}, 周长={perimeter:.2f}")
```

### 2.4 轮廓筛选

```python
# 筛选大轮廓
min_area = 100
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# 按面积排序
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
```

---

## 3. 形态学操作

### 3.1 腐蚀和膨胀

```python
# 定义结构元素
kernel = np.ones((5, 5), np.uint8)

# 腐蚀
eroded = cv2.erode(image, kernel, iterations=1)

# 膨胀
dilated = cv2.dilate(image, kernel, iterations=1)
```

### 3.2 开运算和闭运算

```python
# 开运算：先腐蚀后膨胀（去除噪声）
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 闭运算：先膨胀后腐蚀（填充孔洞）
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

### 3.3 形态学梯度

```python
# 形态学梯度：膨胀减腐蚀（边缘检测）
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
```

### 3.4 顶帽和黑帽

```python
# 顶帽：原图减去开运算（提取亮区域）
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# 黑帽：闭运算减去原图（提取暗区域）
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
```

---

## 4. 直方图处理

### 4.1 计算直方图

```python
# 计算灰度直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 计算彩色图像直方图
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
```

### 4.2 直方图均衡化

```python
# 灰度图像直方图均衡化
equalized = cv2.equalizeHist(gray)

# 彩色图像直方图均衡化（在 YUV 空间）
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
equalized_color = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```

### 4.3 直方图匹配

```python
# 直方图匹配（示例）
def match_histograms(source, reference):
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    source_cdf = source_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    # 归一化
    source_cdf = source_cdf / source_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # 查找映射
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.argmin(np.abs(ref_cdf - source_cdf[i]))
        lookup[i] = idx
    
    return cv2.LUT(source, lookup)
```

---

## 5. 图像阈值

### 5.1 简单阈值

```python
# 二值化
_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 反二值化
_, thresh_binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 截断
_, thresh_trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

# 阈值化为 0
_, thresh_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
```

### 5.2 Otsu 阈值

```python
# Otsu 自动阈值
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_value = thresh_otsu[0]
print(f"Otsu 阈值：{otsu_value}")
```

### 5.3 自适应阈值

```python
# 自适应阈值
adaptive_mean = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

adaptive_gaussian = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
```

---

## 6. 实战练习

### 练习 1：边缘检测对比

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), 255, 3)
cv2.circle(image, (150, 150), 80, 255, 3)
cv2.line(image, (0, 0), (300, 300), 255, 2)

# 应用不同边缘检测
canny = cv2.Canny(image, 50, 150)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.uint8(np.absolute(sobel_x))

sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = np.uint8(np.absolute(sobel_y))

laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 显示对比
plt.figure(figsize=(15, 5))

images = [image, canny, sobel_x, sobel_y, laplacian]
titles = ['原始图像', 'Canny', 'Sobel X', 'Sobel Y', 'Laplacian']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 2：轮廓提取与分析

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
image = np.zeros((400, 400), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
cv2.circle(image, (250, 250), 50, 255, -1)
cv2.ellipse(image, (300, 100), (30, 20), 0, 0, 360, 255, -1)

# 阈值处理
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"找到 {len(contours)} 个轮廓")

# 绘制轮廓
image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    
    print(f"轮廓 {i}: 面积={area}, 周长={perimeter:.2f}, 位置=({x},{y})")
    
    # 绘制轮廓
    cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2)
    
    # 绘制边界框
    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
plt.title(f'找到 {len(contours)} 个轮廓')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 3：形态学操作

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像（带噪声）
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), 255, -1)

# 添加噪声
np.random.seed(42)
noise = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
noisy = cv2.bitwise_or(image, noise)

# 定义结构元素
kernel = np.ones((5, 5), np.uint8)

# 应用形态学操作
eroded = cv2.erode(noisy, kernel, iterations=2)
dilated = cv2.dilate(noisy, kernel, iterations=2)
opening = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel, iterations=2)
gradient = cv2.morphologyEx(noisy, cv2.MORPH_GRADIENT, kernel)

# 显示
plt.figure(figsize=(15, 10))

images = [noisy, eroded, dilated, opening, closing, gradient]
titles = ['含噪声图像', '腐蚀', '膨胀', '开运算', '闭运算', '形态学梯度']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 4：直方图处理

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:, :100] = [50, 50, 50]
image[:, 100:200] = [150, 150, 150]
image[:, 200:] = [250, 250, 250]

# 转换到灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 直方图均衡化
equalized = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# 显示
plt.figure(figsize=(15, 5))

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始灰度图像')
plt.axis('off')

# 原始直方图
plt.subplot(2, 3, 2)
plt.plot(hist, color='blue')
plt.title('原始直方图')
plt.xlabel('像素值')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)

# 均衡化图像
plt.subplot(2, 3, 4)
plt.imshow(equalized, cmap='gray')
plt.title('直方图均衡化后')
plt.axis('off')

# 均衡化直方图
plt.subplot(2, 3, 5)
plt.plot(hist_eq, color='red')
plt.title('均衡化后直方图')
plt.xlabel('像素值')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习 5：阈值处理

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像（渐变）
image = np.zeros((300, 300), dtype=np.uint8)
for i in range(300):
    image[:, i] = int(255 * i / 300)

# 应用不同阈值方法
_, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)

# Otsu 阈值
_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_value = thresh_otsu[0]
print(f"Otsu 阈值：{otsu_value}")

# 显示
plt.figure(figsize=(15, 10))

images = [image, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_otsu[1]]
titles = ['原始图像', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', f'OTSU ({otsu_value:.1f})']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 练习 6：综合实战 - 物体检测

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建复杂测试图像
image = np.zeros((400, 400, 3), dtype=np.uint8)
image[:] = [200, 200, 200]  # 灰色背景

# 绘制多个物体
cv2.rectangle(image, (50, 50), (120, 120), [0, 0, 255], -1)      # 红色方块
cv2.circle(image, (250, 250), 40, [0, 255, 0], -1)               # 绿色圆形
cv2.ellipse(image, (300, 100), (30, 20), 45, 0, 360, [255, 0, 0], -1)  # 蓝色椭圆
cv2.line(image, (0, 300), (400, 300), [0, 0, 0], 3)              # 黑色线条

# 添加噪声
np.random.seed(42)
noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
noisy = cv2.add(image, noise)

# 转换为灰度
gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

# 阈值处理
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# 形态学操作去除噪声
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 查找轮廓
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制结果
result = noisy.copy()
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 500:  # 过滤小轮廓
        # 绘制轮廓
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # 绘制边界框
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 添加标签
        cv2.putText(result, f'Object {i}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

print(f"找到 {len([c for c in contours if cv2.contourArea(c) > 500])} 个物体")

# 显示
plt.figure(figsize=(15, 5))

images = [noisy, gray, thresh, opening, result]
titles = ['含噪声图像', '灰度图像', '阈值处理', '开运算', '检测结果']

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 3, i)
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
# 使用 cv2.Canny, cv2.Sobel, cv2.Laplacian
canny = cv2.Canny(image, 50, 150)
```

### 练习 2 答案

```python
# 使用 cv2.findContours, cv2.drawContours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 练习 3 答案

```python
# 使用 cv2.erode, cv2.dilate, cv2.morphologyEx
opening = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel, iterations=2)
```

### 练习 4 答案

```python
# 使用 cv2.calcHist, cv2.equalizeHist
equalized = cv2.equalizeHist(gray)
```

### 练习 5 答案

```python
# 使用 cv2.threshold, cv2.THRESH_OTSU
_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 练习 6 答案

```python
# 综合使用阈值、形态学、轮廓提取
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

---

## 🎯 关键要点总结

1. **边缘检测**：Canny 最常用，Sobel 和 Laplacian 适用于特定场景
2. **轮廓提取**：findContours 返回轮廓列表和层次结构
3. **形态学操作**：开运算去噪，闭运算填充，梯度提取边缘
4. **直方图均衡化**：增强图像对比度
5. **阈值处理**：Otsu 自动阈值，自适应阈值适用于光照不均
6. **综合应用**：结合多种技术实现物体检测

> 💡 **下一步**：学习深度学习，掌握现代计算机视觉技术！

---

## 📚 扩展学习资源

- [OpenCV 官方文档](https://docs.opencv.org/4.x/)
- [OpenCV-Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- 《Python 计算机视觉编程》

---

> 🌟 **记住**：传统图像处理是深度学习的基础，掌握这些技术将帮助你更好地理解现代 CV 方法！
