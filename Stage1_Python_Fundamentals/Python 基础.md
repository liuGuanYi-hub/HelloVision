# 🐍 Python 基础 - 计算机视觉学习前置知识

> 掌握 Python 编程基础，为学习计算机视觉和深度学习做好准备！

---

## 📖 目录

1. [Python 简介与环境配置](#python-简介与环境配置)
2. [基础语法](#基础语法)
3. [数据结构](#数据结构)
4. [控制流程](#控制流程)
5. [函数](#函数)
6. [模块和包](#模块和包)
7. [文件操作](#文件操作)
8. [异常处理](#异常处理)
9. [面向对象编程](#面向对象编程)
10. [常用库简介](#常用库简介)

---

## 1. Python 简介与环境配置

### 1.1 为什么选择 Python？

- ✅ **简洁易读**：语法清晰，代码如英语
- ✅ **生态丰富**：拥有大量的第三方库
- ✅ **社区活跃**：遇到问题容易找到解决方案
- ✅ **跨平台**：Windows、macOS、Linux 都能运行
- ✅ **AI 首选**：TensorFlow、PyTorch 等主流框架都支持 Python

### 1.2 环境安装

#### 安装 Python

1. 访问官网：https://www.python.org/downloads/
2. 下载 Python 3.8+（推荐 3.10）
3. 安装时勾选 **"Add Python to PATH"**

#### 验证安装

```bash
python --version  # 查看 Python 版本
pip --version     # 查看 pip 版本
```

#### 推荐编辑器

- **VS Code**：轻量级，插件丰富
- **PyCharm**：功能强大的专业 IDE
- **Jupyter Notebook**：交互式编程，适合数据分析

---

## 2. 基础语法

### 2.1 变量和数据类型

```python
# 整数（int）
age = 20
count = -5

# 浮点数（float）
price = 19.99
pi = 3.14159

# 字符串（str）
name = "Alice"
message = 'Hello, World!'

# 布尔值（bool）
is_student = True
has_car = False

# 空值（None）
result = None

# 类型查看
print(type(age))      # <class 'int'>
print(type(price))    # <class 'float'>
print(type(name))     # <class 'str'>
print(type(is_student))  # <class 'bool'>
```

### 2.2 类型转换

```python
# 字符串转数字
num_str = "123"
num_int = int(num_str)      # 123
num_float = float(num_str)  # 123.0

# 数字转字符串
age = 25
age_str = str(age)  # "25"

# 转布尔值
print(bool(0))      # False
print(bool(1))      # True
print(bool(""))     # False
print(bool("Hi"))   # True
```

### 2.3 基本运算

```python
# 算术运算
a = 10
b = 3

print(a + b)   # 加法：13
print(a - b)   # 减法：7
print(a * b)   # 乘法：30
print(a / b)   # 除法：3.333...
print(a // b)  # 整除：3
print(a % b)   # 取余：1
print(a ** b)  # 幂运算：1000

# 比较运算
print(a > b)   # True
print(a == b)  # False
print(a != b)  # True

# 逻辑运算
print(True and False)  # False
print(True or False)   # True
print(not True)        # False
```

### 2.4 字符串操作

```python
text = "Hello, Python!"

# 长度
print(len(text))  # 16

# 索引和切片
print(text[0])     # 'H'
print(text[-1])    # '!'
print(text[0:5])   # 'Hello'
print(text[7:])    # 'Python!'

# 字符串方法
print(text.lower())        # 'hello, python!'
print(text.upper())        # 'HELLO, PYTHON!'
print(text.replace("Python", "World"))  # 'Hello, World!'
print(text.split(", "))    # ['Hello', 'Python!']

# 字符串格式化
name = "Alice"
age = 25

# 方法 1: f-string（推荐）
print(f"My name is {name}, I'm {age} years old.")

# 方法 2: format()
print("My name is {}, I'm {} years old.".format(name, age))

# 方法 3: % 格式化
print("My name is %s, I'm %d years old." % (name, age))
```

---

## 3. 数据结构

### 3.1 列表（List）

```python
# 创建列表
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# 访问元素
print(fruits[0])     # 'apple'
print(fruits[-1])    # 'orange'
print(fruits[1:3])   # ['banana', 'orange']

# 修改列表
fruits.append("grape")      # 添加元素
fruits.insert(1, "pear")    # 插入元素
fruits.remove("banana")     # 删除元素
last = fruits.pop()         # 弹出最后一个元素

# 列表操作
print(len(fruits))          # 长度
print("apple" in fruits)    # 检查是否存在
print(fruits.index("apple")) # 查找索引

# 列表推导式
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### 3.2 字典（Dictionary）

```python
# 创建字典
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "grades": [90, 85, 88]
}

# 访问值
print(student["name"])     # 'Alice'
print(student.get("age"))  # 20

# 修改字典
student["age"] = 21           # 修改值
student["email"] = "a@b.com"  # 添加新键值对
del student["major"]          # 删除键值对

# 遍历字典
for key, value in student.items():
    print(f"{key}: {value}")

# 字典推导式
squares = {x: x**2 for x in range(5)}
```

### 3.3 元组（Tuple）

```python
# 创建元组（不可变列表）
coordinates = (10, 20)
colors = ("red", "green", "blue")

# 访问元素
print(coordinates[0])  # 10

# 元组解包
x, y = coordinates
print(x, y)  # 10 20

# 注意：元组不能修改
# coordinates[0] = 5  # 会报错！
```

### 3.4 集合（Set）

```python
# 创建集合（无序不重复）
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# 集合操作
fruits.add("grape")       # 添加元素
fruits.remove("banana")   # 删除元素

# 集合运算
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(set1 | set2)  # 并集：{1, 2, 3, 4, 5}
print(set1 & set2)  # 交集：{3}
print(set1 - set2)  # 差集：{1, 2}
```

---

## 4. 控制流程

### 4.1 条件语句

```python
age = 18

if age < 13:
    print("儿童")
elif age < 18:
    print("青少年")
elif age < 60:
    print("成年人")
else:
    print("老年人")

# 三元表达式
status = "成年" if age >= 18 else "未成年"
```

### 4.2 for 循环

```python
# 遍历列表
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# 遍历范围
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):   # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

# 遍历字典
student = {"name": "Alice", "age": 20}
for key, value in student.items():
    print(f"{key}: {value}")

# break 和 continue
for i in range(10):
    if i == 3:
        continue  # 跳过本次循环
    if i == 7:
        break     # 退出循环
    print(i)
```

### 4.3 while 循环

```python
count = 0
while count < 5:
    print(count)
    count += 1

# 带 else 的循环
n = 0
while n < 3:
    print(n)
    n += 1
else:
    print("循环结束")  # 正常结束后执行
```

---

## 5. 函数

### 5.1 定义函数

```python
# 基本函数
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Hello, Alice!

# 带默认参数
def introduce(name, age=20):
    print(f"{name}, {age}岁")

introduce("Bob")      # Bob, 20 岁
introduce("Charlie", 25)  # Charlie, 25 岁

# 可变参数
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# 关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=20, city="Beijing")
```

### 5.2 Lambda 函数

```python
# 匿名函数
square = lambda x: x ** 2
print(square(5))  # 25

# 用于排序
numbers = [3, 1, 4, 1, 5, 9, 2]
numbers.sort(key=lambda x: -x)  # 降序排序
print(numbers)  # [9, 5, 4, 3, 2, 1, 1]
```

### 5.3 作用域

```python
global_var = "全局变量"

def my_function():
    local_var = "局部变量"
    print(global_var)  # 可以访问全局变量
    print(local_var)   # 可以访问局部变量

print(global_var)  # 可以访问
# print(local_var)  # 报错！无法访问局部变量
```

---

## 6. 模块和包

### 6.1 导入模块

```python
# 导入整个模块
import math
print(math.sqrt(16))  # 4.0

# 导入特定函数
from math import sqrt, pi
print(sqrt(16))  # 4.0
print(pi)        # 3.14159...

# 导入并重命名
import numpy as np
import pandas as pd

# 导入自定义模块
# import my_module
```

### 6.2 常用标准库

```python
# 数学运算
import math
import random

# 日期时间
import datetime
now = datetime.datetime.now()

# 操作系统
import os
os.getcwd()  # 获取当前目录

# 文件系统
import pathlib
from pathlib import Path

# JSON 处理
import json

# 正则表达式
import re
```

---

## 7. 文件操作

### 7.1 读写文件

```python
# 写入文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("第二行")

# 读取文件
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()      # 读取全部内容
    print(content)

# 逐行读取
with open("example.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())  # 去掉换行符

# 读取所有行
with open("example.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### 7.2 文件模式

- `"r"` - 只读（默认）
- `"w"` - 写入（覆盖）
- `"a"` - 追加
- `"b"` - 二进制模式
- `"+"` - 读写模式

---

## 8. 异常处理

```python
# 基本异常处理
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零！")

# 多个异常
try:
    age = int(input("请输入年龄："))
    if age < 0:
        raise ValueError("年龄不能为负数")
except ValueError as e:
    print(f"输入错误：{e}")
except Exception as e:
    print(f"未知错误：{e}")
finally:
    print("程序结束")

# 抛出异常
def check_age(age):
    if age < 0:
        raise ValueError("年龄不能为负数")
    return age
```

---

## 9. 面向对象编程

### 9.1 类和对象

```python
class Person:
    # 类属性
    species = "Homo sapiens"
    
    # 初始化方法
    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age
    
    # 实例方法
    def introduce(self):
        return f"我是{self.name}，今年{self.age}岁"
    
    # 类方法
    @classmethod
    def get_species(cls):
        return cls.species
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        return age >= 18

# 创建对象
person1 = Person("Alice", 20)
person2 = Person("Bob", 25)

# 调用方法
print(person1.introduce())  # 我是 Alice，今年 20 岁
print(Person.get_species()) # Homo sapiens
print(Person.is_adult(20))  # True
```

### 9.2 继承

```python
class Student(Person):
    def __init__(self, name, age, major):
        super().__init__(name, age)  # 调用父类初始化
        self.major = major
    
    # 重写方法
    def introduce(self):
        return f"我是{self.name}，专业是{self.major}"

student = Student("Charlie", 20, "Computer Science")
print(student.introduce())  # 我是 Charlie，专业是 Computer Science
```

---

## 10. 常用库简介

### 10.1 数据处理

```python
# NumPy - 数值计算
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

# Pandas - 数据分析
import pandas as pd
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [20, 25]})
```

### 10.2 图像处理

```python
# OpenCV - 计算机视觉
import cv2
image = cv2.imread("image.jpg")

# Pillow - 图像处理
from PIL import Image
img = Image.open("image.jpg")
```

### 10.3 深度学习

```python
# PyTorch
import torch
import torch.nn as nn

# TensorFlow
import tensorflow as tf
```

### 10.4 可视化

```python
# Matplotlib - 绘图
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

# Seaborn - 统计绘图
import seaborn as sns
```

---

## 📝 练习建议

### 基础练习
1. 实现一个简单的计算器
2. 编写一个猜数字游戏
3. 实现学生成绩管理系统
4. 编写文件备份脚本

### 进阶练习
1. 爬取网页数据并保存
2. 分析 CSV 数据并生成图表
3. 实现简单的待办事项应用
4. 批量处理图片（调整大小、添加水印）

---

## 🎯 下一步

完成 Python 基础学习后，继续学习：

1. **NumPy 基础** - 数值计算库
2. **Matplotlib** - 数据可视化
3. **OpenCV 入门** - 图像处理基础
4. **Stage 2** - 传统图像处理

---

## 📚 推荐资源

### 在线教程
- [Python 官方文档](https://docs.python.org/zh-cn/3/)
- [菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)
- [廖雪峰 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400)

### 练习平台
- [LeetCode](https://leetcode.com/)
- [HackerRank](https://www.hackerrank.com/)
- [牛客网](https://www.nowcoder.com/)

---

**最后更新**: 2026-04-26
