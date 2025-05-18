import os
import tarfile

# 압축 파일들이 위치한 폴더
source_dir = r'C:\Users\PC-3\OpenGait\GaitDatasetB-silh'

# 압축을 해제할 대상 폴더 (원하는 경로로 수정하세요)
target_dir = r'C:\Users\PC-3\OpenGait\datasets\CASIA-B'

# 대상 폴더가 없으면 생성
os.makedirs(target_dir, exist_ok=True)

# source_dir에 있는 모든 .tar.gz 파일 반복
for filename in os.listdir(source_dir):
    if filename.endswith('.tar.gz'):
        file_path = os.path.join(source_dir, filename)
        print(f"Extracting: {filename}")

        with tarfile.open(file_path, 'r:gz') as tar:
            # 예: target_dir/001/ 에 풀기
            extract_path = os.path.join(target_dir, os.path.splitext(os.path.splitext(filename)[0])[0])
            os.makedirs(extract_path, exist_ok=True)
            tar.extractall(path=extract_path)

print("✅ All files extracted successfully.")
