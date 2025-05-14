import os
import shutil
from pathlib import Path

src_dir = Path("C:/Users/PC-3/OpenGait/CCVID_dataset/output_try")
dst_dir = Path("C:/Users/PC-3/OpenGait/datasets/CCVID")  # OpenGait 기준 위치

for file in src_dir.glob("*.png"):
    parts = file.stem.split("_")  # e.g., session1_001_01_00001_mask
    session = parts[0]            # e.g., session1
    pid = parts[1]                # e.g., 001
    clip = parts[3]               # e.g., 00001
    category = parts[4]           # 'mask' or 'skeleton'

    # 디렉토리 설정
    target_dir = dst_dir / pid / f"{session}-01" / "000" / category
    target_dir.mkdir(parents=True, exist_ok=True)

    # 새 파일명: 00001.png
    new_name = f"{clip}.png"
    shutil.copy(file, target_dir / new_name)

print("파일 정리가 완료되었습니다.")
