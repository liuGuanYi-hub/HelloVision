"""Setup Oxford-IIIT-Pet in torchvision structure and download"""
import os

# The structure torchvision expects:
# <root>/
#   images/
#     Abyssinian_1.jpg
#     ...
#   annotations/
#     trimaps/
#       Abyssinian_1.png
#       ...
#   ...

root = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\oxford_pets\oxford-iiit-pet'
images_src = os.path.join(root, 'images')  # Contains 7393 files

# The images are directly in images/ folder (not in a subfolder)
# torchvision expects: <root>/images/<breed>_<num>.jpg
# which is what we have. But it also needs annotations/

# Create proper structure
ann_dir = os.path.join(root, 'annotations')
trimap_dir = os.path.join(ann_dir, 'trimaps')
os.makedirs(trimap_dir, exist_ok=True)

# Move all files from images/ to images/ (already there)
# Just verify
jpg_files = [f for f in os.listdir(images_src) if f.endswith('.jpg')]
print(f'JPG files: {len(jpg_files)}')

# Now let torchvision download the annotations
from torchvision.datasets import OxfordIIITPet

print('Downloading annotations via torchvision...')
try:
    # This will download annotations to the right place
    ds = OxfordIIITPet(
        root=root,
        split='trainval',
        download=True,
        target_types='segmentation'  # This gets the trimap annotations
    )
    print(f'Success! Dataset has {len(ds)} samples')
except Exception as e:
    print(f'Error: {e}')
    
# Verify structure
print('\nFinal structure:')
for item in os.listdir(root):
    path = os.path.join(root, item)
    if os.path.isdir(path):
        sub = os.listdir(path)
        print(f'  [DIR] {item}/ ({len(sub)} items)')
    else:
        size = os.path.getsize(path) / 1024 / 1024
        print(f'  {item} ({size:.1f} MB)')