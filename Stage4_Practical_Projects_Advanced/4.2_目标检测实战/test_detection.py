"""用训练好的YOLOv8模型测试检测效果"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2

# 加载最佳模型
best_model = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_\yolo_train-2\weights\best.pt'
print(f'加载模型: {best_model}')
model = YOLO(best_model)

# 测试图片目录
test_dir = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\test'
output_dir = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\detect_output'
os.makedirs(output_dir, exist_ok=True)

# 随机选几张图测试
import random
cats = [os.path.join(test_dir, 'cats', f) for f in os.listdir(os.path.join(test_dir, 'cats')) if f.endswith('.jpg')]
dogs = [os.path.join(test_dir, 'dogs', f) for f in os.listdir(os.path.join(test_dir, 'dogs')) if f.endswith('.jpg')]
test_imgs = random.sample(cats, 3) + random.sample(dogs, 3)

print(f'\n测试 {len(test_imgs)} 张图片...\n')
for img_path in test_imgs:
    results = model(img_path, verbose=False)
    r = results[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 绘制检测结果
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f'{model.names[cls_id]} {conf:.2f}'
            
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f'  检测到: {label}')
    else:
        print(f'  无检测结果')
    
    # 保存结果
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f'  保存: {out_path}\n')

print(f'结果已保存到: {output_dir}')