"""Check Oxford dataset structure and available data"""
import os, urllib.request

# Check what we already have
base = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\oxford_pets\oxford-iiit-pet'
print('=== Available files ===')
for item in os.listdir(base):
    path = os.path.join(base, item)
    if os.path.isdir(path):
        sub = os.listdir(path)
        print(f'DIR: {item}/ ({len(sub)} items)')
        if len(sub) < 5:
            for s in sub: print(f'  {s}')
    else:
        size = os.path.getsize(path)
        print(f'FILE: {item} ({size} bytes)')

# Try downloading annotation list via a different method
# The Oxford dataset has: class_name number bbox_x1 bbox_y1 bbox_x2 bbox_y2 species species_name
print('\n=== Trying Oxford annotation URLs ===')
test_urls = [
    'https://www.robots.ox.ac.uk/~vgg/data/pets/data/boxshots.tar.gz',
    'https://www.robots.ox.ac.uk/~vgg/data/pets/data/list.tar',
]
for url in test_urls:
    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f'OK {resp.status}: {url} ({resp.headers.get("Content-Length")} bytes)')
    except Exception as e:
        print(f'FAIL: {url} -> {e}')

# Try finding a working mirror
print('\n=== Trying mirrors ===')
mirrors = [
    'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/scripts/get_oxford_pet.sh',
    'https://raw.githubusercontent.com/tensorboy/pytorch_apple_hacking/main/datasets/download_oxford_pet.sh',
]
for url in mirrors:
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8')
            print(f'OK: {url}')
            print(content[:500])
            break
    except Exception as e:
        print(f'FAIL: {url} -> {e}')