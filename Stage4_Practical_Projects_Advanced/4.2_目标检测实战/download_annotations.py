"""Download Oxford-IIIT-Pet annotations"""
import urllib.request, os, time

url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
outdir = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\oxford_pets\oxford-iiit-pet'
outpath = os.path.join(outdir, 'annotations_new.tar.gz')

print(f'Downloading annotations from Oxford...')
print(f'Target: {outpath}')

for attempt in range(3):
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get('Content-Length', 0))
            print(f'Size: {total/1024/1024:.1f} MB')
            with open(outpath, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (2*1024*1024) < 8192:
                        print(f'  {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB')
        print(f'Success! Size: {os.path.getsize(outpath)/1024/1024:.1f} MB')
        break
    except Exception as e:
        print(f'Attempt {attempt+1} failed: {e}')
        if os.path.exists(outpath):
            size = os.path.getsize(outpath)
            if size > 1000:
                print(f'Partial file: {size/1024/1024:.1f} MB, trying to continue...')
                # Try resume
                try:
                    req = urllib.request.Request(url)
                    req.add_header('Range', f'bytes={size}-')
                    req.add_header('User-Agent', 'Mozilla/5.0')
                    with urllib.request.urlopen(req, timeout=60) as response:
                        total = int(response.headers.get('Content-Length', 0)) + size
                        with open(outpath, 'ab') as f:
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                downloaded += len(chunk)
                        print(f'Resumed to: {os.path.getsize(outpath)/1024/1024:.1f} MB')
                    break
                except:
                    pass
        time.sleep(5)

# Verify
ann_path = os.path.join(outdir, 'annotations_new.tar.gz')
if os.path.exists(ann_path) and os.path.getsize(ann_path) > 1000000:
    import tarfile
    print('Verifying tar...')
    try:
        with tarfile.open(ann_path, 'r:gz') as tar:
            members = tar.getmembers()[:3]
            print(f'Valid! {len(tar.getmembers())} files')
            for m in members:
                print(f'  {m.name}')
            tar.extractall(outdir)
        print('Extracted!')
    except Exception as e:
        print(f'Invalid tar: {e}')
else:
    print('Download failed or too small')
    print(f'File exists: {os.path.exists(ann_path)}')
    if os.path.exists(ann_path):
        print(f'Size: {os.path.getsize(ann_path)} bytes')