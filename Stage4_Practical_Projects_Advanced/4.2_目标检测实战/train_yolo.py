"""YOLOv8 自定义训练脚本 - 使用4.1猫狗分类数据"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

print('='*50)
print('YOLOv8 自定义目标检测训练')
print('='*50)

# 模型路径
model_path = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\yolov8n.pt'
data_yaml = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\data.yaml'
project_dir = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战'

print(f'加载模型: {model_path}')
model = YOLO(model_path)
print('模型加载完成')

print(f'\n训练配置:')
print(f'  数据: {data_yaml}')
print(f'  Epochs: 20')
print(f'  图片大小: 224')
print(f'  Batch: 16')
print(f'  项目目录: {project_dir}')

print('\n开始训练...')
results = model.train(
    data=data_yaml,
    epochs=20,
    imgsz=224,
    batch=16,
    project=project_dir,
    name='yolo_train',
    verbose=True,
    device=0,
    workers=0
)

print('\n训练完成!')
# 最佳模型路径
best = os.path.join(project_dir, 'yolo_train', 'weights', 'best.pt')
if os.path.exists(best):
    print(f'最佳模型: {best}')
last = os.path.join(project_dir, 'yolo_train', 'weights', 'last.pt')
if os.path.exists(last):
    print(f'最终模型: {last}')

print('\n结果目录:')
results_dir = os.path.join(project_dir, 'yolo_train')
for item in os.listdir(results_dir):
    print(f'  {item}')