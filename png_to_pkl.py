import os
import pickle
from PIL import Image
import numpy as np

# 경로 설정
dataset_root = './datasets/CCVID'
output_path = './datasets/CCVID'

# .pkl로 저장할 데이터 저장소
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 모든 사람 ID를 순차적으로 처리
for person_id in os.listdir(dataset_root):
    person_dir = os.path.join(dataset_root, person_id)
    if not os.path.isdir(person_dir):
        continue

    person_data = {}

    # 각 세션을 순차적으로 처리
    for session in os.listdir(person_dir):
        session_dir = os.path.join(person_dir, session)
        if not os.path.isdir(session_dir):
            continue
        
        session_data = {}
        
        # "000" 디렉토리 내의 mask와 skeleton 폴더 확인
        for subdir in os.listdir(session_dir):
            subdir_path = os.path.join(session_dir, subdir)
            if os.path.isdir(subdir_path) and subdir == '000':
                data = {}
                
                # mask와 skeleton 폴더의 이미지 파일을 처리
                for folder in ['mask', 'skeleton']:
                    folder_path = os.path.join(subdir_path, folder)
                    if os.path.isdir(folder_path):
                        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]  # .png로 가정
                        data[folder] = []
                        
                        # 각 이미지를 로드하고 numpy 배열로 변환하여 pkl로 저장
                        for image_path in image_paths:
                            image = Image.open(image_path).convert('RGB')
                            image_data = np.array(image)  # numpy 배열로 변환
                            data[folder].append(image_data)

                session_data[subdir] = data

        person_data[session] = session_data
    
    # 변환된 데이터를 .pkl 파일로 저장
    pkl_file_path = os.path.join(output_path, f"{person_id}_data.pkl")
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(person_data, pkl_file)

print("모든 이미지가 .pkl 형식으로 변환되었습니다.")
