import os
import pickle
import numpy as np
import cv2
from collections import defaultdict

input_dir = 'CCVID_dataset/output_try'
output_root = 'datasets/CCVID' 

# key: "{pid}/{seq_type}/{view}/{mode}" → value: list of images
data_dict = defaultdict(list)

# 파일 정렬
files = sorted(os.listdir(input_dir))

for file in files:
    if not file.endswith('.png'):
        continue

    parts = file.split('_')
    session = parts[0]       # e.g., session1
    pid = parts[1]           # e.g., 001
    seq = parts[2]           # e.g., 01
    frame_id = parts[3]      # e.g., 00001_mask.png

    view = '000'             # view는 고정
    seq_type = f"{session}-{seq}"

    mode = 'sil' if 'mask' in file else 'ske'
    key = f"{pid}/{seq_type}/{view}/{mode}"

    filepath = os.path.join(input_dir, file)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Failed to load image: {filepath}")
        continue

    data_dict[key].append(img)

for k, frames in data_dict.items():
    data = {
        'input': frames,
        'frame_ids': list(range(len(frames)))
    }
    save_path = os.path.join(output_root, os.path.dirname(k))
    os.makedirs(save_path, exist_ok=True)
    mode = k.split('/')[-1]
    with open(os.path.join(save_path, f'{mode}.pkl'), 'wb') as f:
        pickle.dump(data, f)

print("변환 완료: .pkl 파일 생성됨")
