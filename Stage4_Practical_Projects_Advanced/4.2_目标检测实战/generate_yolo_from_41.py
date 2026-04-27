"""从4.1数据集生成YOLO格式的目标检测数据"""
import os, shutil

# 4.1 分类数据目录
src_train_cats = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\train\cats'
src_train_dogs = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\train\dogs'
src_val_cats = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\val\cats'
src_val_dogs = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\val\dogs'
src_test_cats = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\test\cats'
src_test_dogs = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\datasets\cats_vs_dogs\test\dogs'

# 目标目录 - 用 train+val 训练，test做验证
base = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\oxford_yolo'
images_train = os.path.join(base, 'images', 'train')
images_val = os.path.join(base, 'images', 'val')
labels_train = os.path.join(base, 'labels', 'train')
labels_val = os.path.join(base, 'labels', 'val')

for d in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(d, exist_ok=True)

def convert_and_copy(src_dir, class_id, dest_img_dir, dest_lbl_dir):
    if not os.path.exists(src_dir):
        print(f'  WARNING: {src_dir} does not exist, skipping')
        return 0
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = 0
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        base_name = os.path.splitext(fname)[0]
        out_img_name = f'{os.path.basename(src_dir)}_{base_name}.jpg'
        out_lbl_name = out_img_name.replace('.jpg', '.txt')
        out_img_path = os.path.join(dest_img_dir, out_img_name)
        out_lbl_path = os.path.join(dest_lbl_dir, out_lbl_name)
        shutil.copy2(src_path, out_img_path)
        with open(out_lbl_path, 'w') as f:
            f.write(f'{class_id} 0.5 0.5 1.0 1.0\n')
        count += 1
    return count

print('=== 生成YOLO格式数据 ===')
total_train = 0
total_val = 0

for src, cls_id, dest_img, dest_lbl in [
    (src_train_cats, 0, images_train, labels_train),
    (src_train_dogs, 1, images_train, labels_train),
    (src_test_cats, 0, images_val, labels_val),
    (src_test_dogs, 1, images_val, labels_val),
]:
    name = os.path.basename(os.path.dirname(src))
    cls_name = 'cat' if cls_id == 0 else 'dog'
    count = convert_and_copy(src, cls_id, dest_img, dest_lbl)
    print(f'  {name}/{cls_name}: {count} images')
    if 'train' in src:
        total_train += count
    else:
        total_val += count

print(f'\nTrain: {total_train}, Val: {total_val}')

# 验证
def check_dir(img_dir, lbl_dir, name):
    img_count = len(os.listdir(img_dir))
    lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    print(f'{name}: {img_count} images, {lbl_count} labels')

print()
check_dir(images_train, labels_train, 'Train')
check_dir(images_val, labels_val, 'Val')
print('\n完成!')