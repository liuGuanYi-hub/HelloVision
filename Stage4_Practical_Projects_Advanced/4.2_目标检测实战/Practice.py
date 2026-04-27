#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.2 目标检测实战 - 完整练习代码

项目：使用 YOLOv8 进行目标检测
功能：
    - 使用预训练模型推理
    - 训练自定义数据集
    - 实时摄像头检测
    - 视频文件检测
    - 结果可视化

作者：Your Name
日期：2026-04-26
"""

import os
import cv2
from ultralytics import YOLO
import argparse


class YOLOv8Detector:
    """YOLOv8 目标检测器"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        初始化检测器
        
        Args:
            model_name: 模型名称
                - yolov8n.pt: 最小最快
                - yolov8s.pt: 小型
                - yolov8m.pt: 中型
                - yolov8l.pt: 大型
                - yolov8x.pt: 最大最准确
        """
        print(f"加载模型：{model_name}")
        self.model = YOLO(model_name)
        print("模型加载完成！")
        
    def detect_image(self, image_path, save_result=True, show_result=False):
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果
            show_result: 是否显示结果
        """
        print(f"\n检测图片：{image_path}")
        
        # 执行检测
        results = self.model(image_path, verbose=False)
        result = results[0]
        
        # 解析结果
        boxes = result.boxes
        print(f"检测到 {len(boxes)} 个物体")
        
        for i, box in enumerate(boxes):
            # 边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 类别和置信度
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            print(f"  [{i+1}] {cls_name}: {confidence:.2f} 位置：[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # 保存结果
        if save_result:
            os.makedirs('results', exist_ok=True)
            result_path = os.path.join('results', f'detection_{os.path.basename(image_path)}')
            result.save(result_path)
            print(f"结果已保存至：{result_path}")
        
        # 显示结果
        if show_result:
            result.show()
        
        return result
    
    def detect_batch(self, image_paths, save_results=True):
        """
        批量检测多张图片
        
        Args:
            image_paths: 图片路径列表
            save_results: 是否保存结果
        """
        print(f"\n批量检测 {len(image_paths)} 张图片")
        
        results = self.model(image_paths, verbose=False)
        
        if save_results:
            os.makedirs('results', exist_ok=True)
            for i, result in enumerate(results):
                result_path = os.path.join('results', f'detection_{i}_{os.path.basename(image_paths[i])}')
                result.save(result_path)
        
        print(f"批量检测完成！共检测到 {sum(len(r.boxes) for r in results)} 个物体")
        return results
    
    def train_custom_dataset(self, data_yaml, epochs=100, imgsz=640, batch=16, device='0'):
        """
        训练自定义数据集
        
        Args:
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 输入图像大小
            batch: 批次大小
            device: 设备（'cpu' 或 '0'）
        """
        print("\n" + "="*50)
        print("开始训练自定义数据集")
        print("="*50)
        
        # 训练配置
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'workers': 4,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'patience': 50,
            'save': True,
            'project': 'runs/detect',
            'name': 'custom_exp',
            'verbose': True
        }
        
        # 开始训练
        results = self.model.train(**train_args)
        
        print("\n训练完成！")
        return results
    
    def detect_camera(self, camera_id=0, show=True, save=False):
        """
        实时摄像头检测
        
        Args:
            camera_id: 摄像头 ID（通常 0 是默认摄像头）
            show: 是否显示结果
            save: 是否保存视频
        """
        print("\n" + "="*50)
        print("启动摄像头实时检测")
        print("="*50)
        print("按 'q' 键退出")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        # 获取摄像头参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"摄像头分辨率：{width}x{height}, FPS: {fps}")
        
        # 视频保存
        if save:
            os.makedirs('results', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('results/camera_detection.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            results = self.model(frame, verbose=False)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            # 计算 FPS
            frame_count += 1
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time > 0:
                current_fps = frame_count / elapsed_time
                cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示
            if show:
                cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
            
            # 保存
            if save:
                out.write(annotated_frame)
            
            # 退出检测
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理
        cap.release()
        if save:
            out.release()
            print("视频已保存至：results/camera_detection.mp4")
        
        cv2.destroyAllWindows()
        print(f"平均 FPS: {frame_count/elapsed_time:.1f}")
    
    def detect_video(self, video_path, save_result=True):
        """
        检测视频文件
        
        Args:
            video_path: 视频文件路径
            save_result: 是否保存结果
        """
        print(f"\n检测视频：{video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("错误：无法打开视频文件")
            return
        
        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频分辨率：{width}x{height}, FPS: {fps}, 总帧数：{total_frames}")
        
        # 创建输出视频
        if save_result:
            os.makedirs('results', exist_ok=True)
            output_path = os.path.join('results', f'detection_{os.path.basename(video_path)}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            results = self.model(frame, verbose=False)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            # 保存
            if save_result:
                out.write(annotated_frame)
            
            # 进度
            frame_count += 1
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                print(f"进度：{progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if save_result:
            out.release()
            print(f"结果已保存至：{output_path}")
        
        print("视频检测完成！")


def main():
    """主函数"""
    print("="*50)
    print("4.2 目标检测实战 - YOLOv8")
    print("="*50)
    
    # 创建检测器
    detector = YOLOv8Detector(model_name='yolov8n.pt')
    
    # 示例 1: 使用预训练模型检测图片
    print("\n" + "="*50)
    print("示例 1: 图片检测")
    print("="*50)
    
    # 检查是否有测试图片
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        detector.detect_image(test_image, save_result=True, show_result=False)
    else:
        print(f"未找到测试图片：{test_image}")
        print("提示：将你的图片重命名为 test_image.jpg 放在当前目录")
    
    # 示例 2: 实时摄像头检测（可选）
    print("\n" + "="*50)
    print("示例 2: 摄像头实时检测")
    print("="*50)
    print("是否启动摄像头检测？(y/n): ", end='')
    
    # 注意：实际运行时取消下面的注释
    # response = input().lower()
    # if response == 'y':
    #     detector.detect_camera(camera_id=0, show=True, save=False)
    print("提示：取消代码注释以启用摄像头检测")
    
    # 示例 3: 训练自定义数据集（可选）
    print("\n" + "="*50)
    print("示例 3: 训练自定义数据集")
    print("="*50)
    print("提示：准备数据集后运行训练")
    print("1. 创建数据集目录结构")
    print("2. 标注数据（使用 LabelImg 等工具）")
    print("3. 创建 data.yaml 配置文件")
    print("4. 调用 detector.train_custom_dataset()")
    
    # 训练示例代码（取消注释并修改路径即可运行）
    """
    detector.train_custom_dataset(
        data_yaml='datasets/my_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='0'  # 使用 GPU
    )
    """
    
    print("\n" + "="*50)
    print("所有示例完成！")
    print("="*50)


if __name__ == '__main__':
    main()
