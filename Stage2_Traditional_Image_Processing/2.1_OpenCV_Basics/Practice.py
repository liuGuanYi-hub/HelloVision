"""
OpenCV 基础练习文件 - 图像处理入门
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def practice_1_basic_operations():
    """
    练习 1：图像基本操作
    学习创建、读取、显示和保存图像
    """
    print("=" * 60)
    print("练习 1：图像基本操作")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:] = [255, 255, 255]  # 白色背景
    
    # 绘制几何图形
    cv2.rectangle(image, (50, 50), (250, 250), (0, 0, 255), 3)
    cv2.circle(image, (150, 150), 80, (255, 0, 0), -1)
    cv2.line(image, (0, 0), (300, 300), (0, 255, 0), 2)
    cv2.putText(image, 'OpenCV', (80, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    print(f"\n图像形状：{image.shape}")
    print(f"图像大小：{image.size}")
    print(f"数据类型：{image.dtype}")
    
    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('几何图形绘制')
    plt.axis('off')
    plt.show()


def practice_2_color_spaces():
    """
    练习 2：色彩空间转换
    学习 BGR、GRAY、HSV、LAB 等色彩空间
    """
    print("=" * 60)
    print("练习 2：色彩空间转换")
    print("=" * 60)
    
    # 创建彩色图像
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, :100] = [0, 0, 255]      # 红色
    image[:, 100:200] = [0, 255, 0]   # 绿色
    image[:, 200:] = [255, 0, 0]      # 蓝色
    
    # 转换到不同色彩空间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    print(f"\n原始图像形状：{image.shape}")
    print(f"灰度图像形状：{gray.shape}")
    print(f"HSV 图像形状：{hsv.shape}")
    print(f"LAB 图像形状：{lab.shape}")
    
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


def practice_3_image_filtering():
    """
    练习 3：图像滤波
    学习均值滤波、高斯滤波、中值滤波、双边滤波
    """
    print("=" * 60)
    print("练习 3：图像滤波")
    print("=" * 60)
    
    # 创建带噪声的图像
    np.random.seed(42)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = [255, 255, 255]
    
    # 添加高斯噪声
    noise = np.random.normal(0, 30, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"\n原始图像形状：{image.shape}")
    print(f"含噪声图像形状：{noisy.shape}")
    
    # 应用不同滤波
    blur = cv2.blur(noisy, (5, 5))
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    
    # 显示
    plt.figure(figsize=(15, 10))
    
    images = [noisy, blur, gaussian, median, bilateral]
    titles = ['含噪声图像', '均值滤波 (5x5)', '高斯滤波 (5x5)', '中值滤波 (5)', '双边滤波']
    
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def practice_4_image_transformations():
    """
    练习 4：图像变换
    学习缩放、旋转、翻转、平移
    """
    print("=" * 60)
    print("练习 4：图像变换")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:] = [255, 255, 255]
    cv2.rectangle(image, (50, 50), (250, 250), (0, 0, 255), 3)
    cv2.putText(image, 'OpenCV', (80, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # 缩放
    resized = cv2.resize(image, None, fx=0.5, fy=0.5)
    print(f"\n原始图像形状：{image.shape}")
    print(f"缩放后形状：{resized.shape}")
    
    # 旋转
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # 翻转
    h_flip = cv2.flip(image, 1)
    v_flip = cv2.flip(image, 0)
    
    # 显示
    plt.figure(figsize=(15, 10))
    
    images = [image, resized, rotated, h_flip, v_flip]
    titles = ['原始图像', '缩小 (0.5x)', '旋转 45°', '水平翻转', '垂直翻转']
    
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def practice_5_hsv_color_segmentation():
    """
    练习 5：HSV 颜色分割
    学习使用 HSV 色彩空间进行颜色提取
    """
    print("=" * 60)
    print("练习 5：HSV 颜色分割")
    print("=" * 60)
    
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


def practice_6_image_properties():
    """
    练习 6：图像属性分析
    学习分析图像的基本属性
    """
    print("=" * 60)
    print("练习 6：图像属性分析")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:] = [128, 128, 128]  # 灰色背景
    cv2.rectangle(image, (50, 50), (150, 150), [255, 255, 255], -1)
    cv2.circle(image, (100, 100), 30, [0, 0, 0], -1)
    
    print(f"\n图像形状：{image.shape}")
    print(f"图像高度：{image.shape[0]}")
    print(f"图像宽度：{image.shape[1]}")
    print(f"通道数：{image.shape[2]}")
    print(f"像素总数：{image.size}")
    print(f"数据类型：{image.dtype}")
    
    # 计算统计信息
    print(f"\n像素值统计：")
    print(f"  最小值：{image.min()}")
    print(f"  最大值：{image.max()}")
    print(f"  平均值：{image.mean():.2f}")
    print(f"  标准差：{image.std():.2f}")
    
    # 计算每个通道的统计信息
    print(f"\n各通道统计：")
    for i, color in enumerate(['B', 'G', 'R']):
        channel = image[:,:,i]
        print(f"  {color} 通道 - 平均值：{channel.mean():.2f}, 标准差：{channel.std():.2f}")
    
    # 显示图像和直方图
    plt.figure(figsize=(15, 5))
    
    # 显示图像
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('测试图像')
    plt.axis('off')
    
    # 显示各通道直方图
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        plt.subplot(1, 4, i+2)
        plt.hist(image[:,:,2-i].flatten(), 256, [0, 256], color=color, alpha=0.7)
        plt.title(f'{color} 通道直方图')
        plt.xlabel('像素值')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenCV 基础练习开始！")
    print("=" * 60)
    
    # 运行所有练习
    practice_1_basic_operations()
    practice_2_color_spaces()
    practice_3_image_filtering()
    practice_4_image_transformations()
    practice_5_hsv_color_segmentation()
    practice_6_image_properties()
    
    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)
    
    # 最后显示所有图片
    plt.show()
