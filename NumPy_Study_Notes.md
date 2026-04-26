# 📊 NumPy 基础学习笔记

> NumPy 是 Python 科学计算的基础库，所有图像都可以表示为矩阵，理解 NumPy 是学习计算机视觉的第一步。

---

## 📖 目录

1. [NumPy 简介与安装](#1-numpy-简介与安装)
2. [创建数组](#2-创建数组)
3. [数组基本操作](#3-数组基本操作)
4. [数组索引与切片](#4-数组索引与切片)
5. [矩阵运算](#5-矩阵运算)
6. [实战练习](#6-实战练习)

---

## 1. NumPy 简介与安装

### 为什么学习 NumPy？

- **所有图像都是矩阵**：一张图片可以表示为一个三维数组（高度×宽度×通道）
- **高效计算**：NumPy 的底层使用 C 语言实现，比纯 Python 快 10-100 倍
- **计算机视觉基础**：几乎所有 CV 库（OpenCV、Pillow）都基于 NumPy

### 导入 NumPy

```python
import numpy as np

# 设置打印选项，方便查看大型数组
np.set_printoptions(precision=2, suppress=True)
```

> 💡 **习惯约定**：通常将 numpy 导入为 `np`，这是一个社区约定俗成的做法。

---

## 2. 创建数组

### 2.1 从列表创建

```python
# 一维数组
a = np.array([1, 2, 3, 4, 5])
print("一维数组:", a)
# 输出: [1 2 3 4 5]

# 二维数组（矩阵）
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print("二维数组:\n", b)
# 输出:
# [[1 2 3]
#  [4 5 6]]
```

### 2.2 使用内置函数创建

```python
# 从 0 开始，步长为 1，到但不包括 stop
arr1 = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 在指定范围内生成均匀间隔的数值
arr2 = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# 全 0 数组（常用于初始化）
zeros = np.zeros((3, 4))  # 3行4列的全0矩阵

# 全 1 数组
ones = np.ones((2, 3))  # 2行3列的全1矩阵

# 单位矩阵
eye = np.eye(3)  # 3×3 的单位矩阵

# 随机数组
random = np.random.rand(3, 3)  # 0-1之间的随机数
random_int = np.random.randint(0, 10, (3, 3))  # 0-10之间的随机整数
```

> 💡 **小技巧**：`np.zeros()` 和 `np.ones()` 常用于初始化占位数组。

---

## 3. 数组基本属性

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("数组:", arr)
print("形状:", arr.shape)      # (2, 3) - 2行3列
print("维度:", arr.ndim)       # 2
print("元素总数:", arr.size)   # 6
print("数据类型:", arr.dtype)  # int32 或 int64
```

> 💡 **在图像处理中的应用**：灰度图像是二维数组，彩色图像是三维数组（高度×宽度×3）。

---

## 4. 数组索引与切片

### 4.1 基本索引

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# 访问单个元素
print(arr[0, 0])  # 1（第一行第一列）
print(arr[1, 2])  # 6（第二行第三列）

# 负索引（从后往前数）
print(arr[-1, -1])  # 9（最后一行最后一列）
```

### 4.2 切片操作

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 切片格式：[start:stop:step]
# 省略 start 表示从 0 开始
# 省略 stop 表示到末尾
# 省略 step 表示步长为 1

# 取前两行
print(arr[:2])  # [[1, 2, 3, 4], [5, 6, 7, 8]]

# 取后两列
print(arr[:, 2:])  # [[3, 4], [7, 8], [11, 12]]

# 提取 2×2 子矩阵（取第1-2行，第1-2列）
print(arr[1:3, 1:3])  # [[6, 7], [10, 11]]
```

### 4.3 条件索引

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 找出所有大于 3 的元素
mask = arr > 3
print(arr[mask])  # [4, 5, 6]

# 一步到位
print(arr[arr > 3])  # [4, 5, 6]
```

> 💡 **实战应用**：在图像处理中，可以用条件索引来筛选特定像素值，比如找出所有红色的像素。

---

## 5. 矩阵运算

### 5.1 元素级运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 加法
print(a + b)  # [5, 7, 9]

# 减法
print(a - b)  # [-3, -3, -3]

# 乘法（对应元素相乘）
print(a * b)  # [4, 10, 18]

# 除法
print(a / b)  # [0.25, 0.4, 0.5]

# 幂运算
print(a ** 2)  # [1, 4, 9]

# 标量运算
print(a * 2)  # [2, 4, 6]
```

### 5.2 矩阵乘法（点积）

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 方法1：使用 @ 运算符
C = A @ B

# 方法2：使用 np.dot
C = np.dot(A, B)

print("矩阵乘法结果:\n", C)
# [[1*5+2*7, 1*6+2*8]
#  [3*5+4*7, 3*6+4*8]]
# = [[19, 22],
#    [43, 50]]
```

> ⚠️ **重要**：矩阵乘法不满足交换律，即 `A @ B` 不一定等于 `B @ A`！

### 5.3 常用统计函数

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 求和
print(np.sum(arr))      # 21
print(np.sum(arr, axis=0))  # [5, 7, 9]（按列求和）
print(np.sum(arr, axis=1))  # [6, 15]（按行求和）

# 平均值
print(np.mean(arr))     # 3.5
print(np.mean(arr, axis=0))  # [2.5, 3.5, 4.5]

# 最大值、最小值
print(np.max(arr))      # 6
print(np.min(arr))      # 1
print(np.argmax(arr))   # 5（最大值的位置）
print(np.argmin(arr))   # 0（最小值的位置）

# 标准差、方差
print(np.std(arr))      # 约 1.71
print(np.var(arr))      # 约 2.92
```

### 5.4 形状变换

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

# 展平为一维
flat = arr.flatten()  # [1, 2, 3, 4, 5, 6, 7, 8]

# 改变形状（元素总数必须匹配）
reshaped = arr.reshape(4, 2)
print(reshaped)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# 转置
transposed = arr.T
print(transposed)
# [[1, 5],
#  [2, 6],
#  [3, 7],
#  [4, 8]]
```

> 💡 **图像处理应用**：`reshape` 常用于将图像展平以便输入神经网络，`T`（转置）用于图像旋转。

---

## 6. 实战练习

### 练习 1：图像基本操作模拟

```python
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的 "图像"（8×8 像素的灰度图）
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 255, 255, 255, 0],
    [0, 255, 0, 0, 0, 0, 255, 0],
    [0, 255, 0, 255, 255, 0, 255, 0],
    [0, 255, 0, 255, 255, 0, 255, 0],
    [0, 255, 0, 0, 0, 0, 255, 0],
    [0, 255, 255, 255, 255, 255, 255, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

# 显示图像
plt.imshow(image, cmap='gray')
plt.title('简单的像素图像')
plt.colorbar()
plt.show()

# 提取图像的一部分（比如中间的正方形）
center = image[2:6, 2:6]
print("中心 4×4 区域:\n", center)

# 统计图像信息
print(f"图像形状: {image.shape}")
print(f"像素值范围: {image.min()} - {image.max()}")
print(f"平均亮度: {np.mean(image):.2f}")
```

### 练习 2：图像亮度调整

```python
import numpy as np

# 模拟一张图像
image = np.array([[100, 150, 200],
                  [50, 100, 150]], dtype=np.uint8)

# 增亮（加一个值）
brighter = np.clip(image + 30, 0, 255)  # clip 确保值在 0-255 范围内
print("增亮后:\n", brighter)

# 变暗（减一个值）
darker = np.clip(image - 30, 0, 255)
print("变暗后:\n", darker)

# 对比度调整
contrast = np.clip((image - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
print("增强对比度后:\n", contrast)
```

### 练习 3：图像卷积核模拟

```python
import numpy as np

# 创建一个简单的图像（5×5）
image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=np.float32)

# 定义一个简单的卷积核（边缘检测）
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# 手动实现卷积操作（简化版本）
def simple_convolve(image, kernel):
    """简化版卷积（不处理边界"""
    ki, kj = kernel.shape
    ii, ij = image.shape
    
    # 输出大小
    out_h = ii - ki + 1
    out_w = ij - kj + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+ki, j:j+kj] * kernel)
    
    return output

result = simple_convolve(image, kernel)
print("卷积结果（边缘检测）:\n", result)
```

---

## 📝 练习答案

### 练习 1 答案

```python
# 运行上面的代码即可看到结果
center = image[2:6, 2:6]  # 提取中心区域
```

### 练习 2 答案

```python
# np.clip() 确保像素值不超过 0-255 范围
brighter = np.clip(image + 30, 0, 255)
```

### 练习 3 答案

```python
# 卷积结果会突出显示图像的边缘部分
```

---

## 🎯 关键要点总结

1. **所有图像都是数组**：灰度图是 2D 数组，彩色图是 3D 数组
2. **索引从 0 开始**：Python 的惯例
3. **切片操作**：`arr[start:stop:step]` 格式
4. **矩阵乘法**使用 `@` 运算符或 `np.dot()`
5. **广播机制**：NumPy 自动扩展小数组以匹配大数组的形状
6. **统计函数**：sum、mean、max、min 是最常用的

> 💡 **下一步**：学习 Pandas 进行数据处理，然后学习 Matplotlib 进行可视化！

---

## 📚 扩展学习资源

- [NumPy 官方文档](https://numpy.org/doc/stable/)
- [NumPy 教程 - W3Schools](https://www.w3schools.com/python/numpy/default.asp)
- 《Python数据科学手册》第 2 章

---

> 🌟 **记住**：NumPy 是计算机视觉的基石，熟练掌握数组操作将让后续学习事半功倍！
