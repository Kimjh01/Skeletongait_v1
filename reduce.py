import os
import shutil
from pathlib import Path
import random

def reduce_grew_dataset(source_base, target_base, reduction_factor=20):
    """
    GREW 데이터셋을 비율에 맞게 축소하여 복사
    """
    
    # 경로 설정
    source_path = Path(source_base)
    target_path = Path(target_base)
    
    # 1. Train 폴더 처리 (1~20000 중 1000개 선택)
    print("Processing train folder...")
    train_source = source_path / "train"
    train_target = target_path / "train"
    train_target.mkdir(parents=True, exist_ok=True)
    
    # 숫자 폴더 목록 가져오기
    train_folders = sorted([f for f in train_source.iterdir() if f.is_dir() and f.name.isdigit()])
    selected_train_count = len(train_folders) // reduction_factor
    selected_train_folders = train_folders[:selected_train_count]  # 앞에서부터 선택
    
    for folder in selected_train_folders:
        shutil.copytree(folder, train_target / folder.name)
    print(f"Copied {len(selected_train_folders)} train folders")
    
    # 2. Test/Gallery 폴더 처리 (1~6000 중 300개 선택)
    print("\nProcessing test/gallery folder...")
    gallery_source = source_path / "test" / "gallery"
    gallery_target = target_path / "test" / "gallery"
    gallery_target.mkdir(parents=True, exist_ok=True)
    
    gallery_folders = sorted([f for f in gallery_source.iterdir() if f.is_dir() and f.name.isdigit()])
    selected_gallery_count = len(gallery_folders) // reduction_factor
    selected_gallery_folders = gallery_folders[:selected_gallery_count]  # 앞에서부터 선택
    
    # Gallery ID 저장 (probe 선택을 위해)
    selected_gallery_ids = set()
    for folder in selected_gallery_folders:
        shutil.copytree(folder, gallery_target / folder.name)
        # Gallery 폴더 내의 실제 ID 수집
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                selected_gallery_ids.add(subfolder.name)
    print(f"Copied {len(selected_gallery_folders)} gallery folders")
    print(f"Total gallery IDs: {len(selected_gallery_ids)}")
    
    # 3. Test/Probe 폴더 처리 (Gallery에 있는 ID만 선택)
    print("\nProcessing test/probe folder...")
    probe_source = source_path / "test" / "probe"
    probe_target = target_path / "test" / "probe"
    probe_target.mkdir(parents=True, exist_ok=True)
    
    probe_copied = 0
    for probe_folder in probe_source.iterdir():
        if probe_folder.is_dir() and probe_folder.name in selected_gallery_ids:
            shutil.copytree(probe_folder, probe_target / probe_folder.name)
            probe_copied += 1
    print(f"Copied {probe_copied} probe folders (matching gallery IDs)")
    
    # 4. Distractor 폴더 처리 (비율에 맞게 랜덤 선택)
    print("\nProcessing distractor folder...")
    distractor_source = source_path / "distractor"
    distractor_target = target_path / "distractor"
    distractor_target.mkdir(parents=True, exist_ok=True)
    
    distractor_folders = list(distractor_source.iterdir())
    distractor_folders = [f for f in distractor_folders if f.is_dir()]
    
    # Distractor는 gallery와 겹치지 않아야 하므로 필터링
    distractor_folders = [f for f in distractor_folders if f.name not in selected_gallery_ids]
    
    selected_distractor_count = len(distractor_folders) // reduction_factor
    # 랜덤하게 선택 (또는 앞에서부터 선택)
    random.seed(42)  # 재현성을 위해 시드 설정
    selected_distractors = random.sample(distractor_folders, min(selected_distractor_count, len(distractor_folders)))
    
    for folder in selected_distractors:
        shutil.copytree(folder, distractor_target / folder.name)
    print(f"Copied {len(selected_distractors)} distractor folders")
    
    # 결과 요약
    print("\n=== Summary ===")
    print(f"Train: {len(selected_train_folders)} folders")
    print(f"Gallery: {len(selected_gallery_folders)} folders")
    print(f"Probe: {probe_copied} folders")
    print(f"Distractor: {len(selected_distractors)} folders")

# 사용 예시
if __name__ == "__main__":
    source_base = "F:/Grew_raw"  # F 드라이브의 원본 경로
    target_base = "C:/Grew_raw_reduced"  # C 드라이브의 대상 경로
    
    reduce_grew_dataset(source_base, target_base, reduction_factor=20)
