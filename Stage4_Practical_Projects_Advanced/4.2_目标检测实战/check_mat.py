import scipy.io, os

mat_path = r'D:\zzd_project\cursor\HelloWorld_Vision\Stage4_Practical_Projects_Advanced\4.2_目标检测实战\datasets\oxford_pets\oxford-iiit-pet\images\Abyssinian_1.mat'
data = scipy.io.loadmat(mat_path)
print('Keys:', list(data.keys()))
for k in data.keys():
    if not k.startswith('__'):
        v = data[k]
        if hasattr(v, 'shape'):
            print(f'{k}: shape={v.shape}, flatten[:5]={v.flatten()[:5]}')
        else:
            print(f'{k}: type={type(v)}, value={v}')