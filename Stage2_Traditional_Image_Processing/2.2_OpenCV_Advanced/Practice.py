"""
OpenCV 进阶练习文件 - 边缘检测、轮廓、形态学、直方图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def practice_1_edge_detection():
    """
    练习 1：边缘检测对比
    学习 Canny、Sobel、Laplacian 边缘检测
    """
    print("=" * 60)
    print("练习 1：边缘检测对比")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 250), 255, 3)
    cv2.circle(image, (150, 150), 80, 255, 3)
    cv2.line(image, (0, 0), (300, 300), 255, 2)
    
    print(f"\n测试图像形状：{image.shape}")
    
    # 应用不同边缘检测
    canny = cv2.Canny(image, 50, 150)
    
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x))
    
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))
    
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    print("边缘检测方法：")
    print("  - Canny: 双阈值边缘检测")
    print("  - Sobel X: X 方向梯度")
    print("  - Sobel Y: Y 方向梯度")
    print("  - Laplacian: 二阶导数")
    
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


def practice_2_contour_extraction():
    """
    练习 2：轮廓提取与分析
    学习查找、绘制和分析轮廓
    """
    print("=" * 60)
    print("练习 2：轮廓提取与分析")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.circle(image, (250, 250), 50, 255, -1)
    cv2.ellipse(image, (300, 100), (30, 20), 0, 0, 360, 255, -1)
    
    print(f"\n测试图像形状：{image.shape}")
    
    # 阈值处理
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n找到 {len(contours)} 个轮廓")
    
    # 分析每个轮廓
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        print(f"轮廓 {i}: 面积={area}, 周长={perimeter:.2f}, 位置=({x},{y}), 尺寸=({w}x{h})")
    
    # 绘制轮廓
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2)
        
        x, y, w, h = cv2.boundingRect(contour)
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


def practice_3_morphological_operations():
    """
    练习 3：形态学操作
    学习腐蚀、膨胀、开运算、闭运算
    """
    print("=" * 60)
    print("练习 3：形态学操作")
    print("=" * 60)
    
    # 创建测试图像（带噪声）
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 250), 255, -1)
    
    # 添加噪声
    np.random.seed(42)
    noise = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
    noisy = cv2.bitwise_or(image, noise)
    
    print(f"\n测试图像形状：{image.shape}")
    print(f"含噪声图像形状：{noisy.shape}")
    
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    
    # 应用形态学操作
    eroded = cv2.erode(noisy, kernel, iterations=2)
    dilated = cv2.dilate(noisy, kernel, iterations=2)
    opening = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel, iterations=2)
    gradient = cv2.morphologyEx(noisy, cv2.MORPH_GRADIENT, kernel)
    
    print("\n形态学操作说明：")
    print("  - 腐蚀：去除小物体")
    print("  - 膨胀：填充孔洞")
    print("  - 开运算：先腐蚀后膨胀（去噪）")
    print("  - 闭运算：先膨胀后腐蚀（填充）")
    print("  - 梯度：膨胀减腐蚀（边缘）")
    
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


def practice_4_histogram_processing():
    """
    练习 4：直方图处理
    学习计算、显示和均衡化直方图
    """
    print("=" * 60)
    print("练习 4：直方图处理")
    print("=" * 60)
    
    # 创建测试图像
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, :100] = [50, 50, 50]
    image[:, 100:200] = [150, 150, 150]
    image[:, 200:] = [250, 250, 250]
    
    # 转换到灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"\n测试图像形状：{image.shape}")
    print(f"灰度图像形状：{gray.shape}")
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # 直方图均衡化
    equalized = cv2.equalizeHist(gray)
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    print("\n直方图统计：")
    print(f"  原始图像 - 最小值：{gray.min()}, 最大值：{gray.max()}, 平均值：{gray.mean():.2f}")
    print(f"  均衡化后 - 最小值：{equalized.min()}, 最大值：{equalized.max()}, 平均值：{equalized.mean():.2f}")
    
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


def practice_5_thresholding():
    """
    练习 5：阈值处理
    学习简单阈值、Otsu 阈值、自适应阈值
    """
    print("=" * 60)
    print("练习 5：阈值处理")
    print("=" * 60)
    
    # 创建测试图像（渐变）
    image = np.zeros((300, 300), dtype=np.uint8)
    for i in range(300):
        image[:, i] = int(255 * i / 300)
    
    print(f"\n测试图像形状：{image.shape}")
    
    # 应用不同阈值方法
    _, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, thresh_binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    _, thresh_tozero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    
    # Otsu 阈值
    otsu_value, thresh_otsu_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"\n阈值方法说明：")
    print(f"  - THRESH_BINARY: 大于阈值设为 255，否则设为 0")
    print(f"  - THRESH_BINARY_INV: 反转 THRESH_BINARY")
    print(f"  - THRESH_TRUNC: 大于阈值设为阈值，否则不变")
    print(f"  - THRESH_TOZERO: 大于阈值不变，否则设为 0")
    print(f"  - OTSU: 自动计算最佳阈值 = {float(otsu_value):.1f}")
    
    # 显示
    plt.figure(figsize=(15, 10))
    
    images = [image, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_otsu_img]
    titles = ['原始图像', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', f'OTSU ({otsu_value:.1f})']
    
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def practice_6_object_detection():
    """
    练习 6：综合实战 - 物体检测
    结合阈值、形态学、轮廓提取实现简单物体检测
    """
    print("=" * 60)
    print("练习 6：综合实战 - 物体检测")
    print("=" * 60)
    
    # 创建复杂测试图像
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:] = [200, 200, 200]  # 灰色背景
    
    # 绘制多个物体
    cv2.rectangle(image, (50, 50), (120, 120), [0, 0, 255], -1)
    cv2.circle(image, (250, 250), 40, [0, 255, 0], -1)
    cv2.ellipse(image, (300, 100), (30, 20), 45, 0, 360, [255, 0, 0], -1)
    cv2.line(image, (0, 300), (400, 300), [0, 0, 0], 3)
    
    # 添加噪声
    np.random.seed(42)
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    noisy = cv2.add(image, noise)
    
    print(f"\n测试图像形状：{image.shape}")
    print(f"含噪声图像形状：{noisy.shape}")
    
    # 转换为灰度
    gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    
    # 阈值处理
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤和绘制结果
    result = noisy.copy()
    valid_contours = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # 过滤小轮廓
            valid_contours.append(contour)
            
            # 绘制轮廓
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # 绘制边界框
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 添加标签
            cv2.putText(result, f'Obj {len(valid_contours)}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    print(f"\n找到 {len(valid_contours)} 个物体")
    
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


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenCV 进阶练习开始！")
    print("=" * 60)
    
    # 运行所有练习
    practice_1_edge_detection()
    practice_2_contour_extraction()
    practice_3_morphological_operations()
    practice_4_histogram_processing()
    practice_5_thresholding()
    practice_6_object_detection()
    
    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)
    
    # 最后显示所有图片
    plt.show()
