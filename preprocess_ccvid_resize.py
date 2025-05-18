import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_ROOT = os.path.join(BASE_DIR, "CCVID_PROCESS")
RESIZED_ROOT = os.path.join(BASE_DIR, "CCVID_PROCESS_RESIZED")
os.makedirs(RESIZED_ROOT, exist_ok=True)

TARGET_SIZE = (128, 88)

def resize_with_aspect_ratio_and_padding(img, target_size=TARGET_SIZE):
    target_h, target_w = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    if img.mode == "L":
        new_img = Image.new("L", (target_w, target_h))
    else:
        new_img = Image.new(img.mode, (target_w, target_h))

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    new_img.paste(img_resized, (pad_left, pad_top))
    return new_img

def process_and_save_images(input_folder, output_folder, is_skeleton=False):
    frame_paths = sorted(glob(os.path.join(input_folder, "*.png")))
    if not frame_paths:
        return

    os.makedirs(output_folder, exist_ok=True)

    for frame_path in frame_paths:
        img = Image.open(frame_path)
        if is_skeleton:
            img = img.convert("RGB")
        else:
            img = img.convert("L")

        img_resized = resize_with_aspect_ratio_and_padding(img, TARGET_SIZE)

        filename = os.path.basename(frame_path)
        save_path = os.path.join(output_folder, filename)
        img_resized.save(save_path)

def process_split_resize(root, split, output_root):
    sil_root = os.path.join(root, split, "silhouette")
    skel_root = os.path.join(root, split, "skeleton")

    person_ids = sorted(os.listdir(sil_root))
    for pid in tqdm(person_ids, desc=f"{split} persons"):
        pid_sil_path = os.path.join(sil_root, pid)
        pid_skel_path = os.path.join(skel_root, pid)

        sample_folders = sorted(os.listdir(pid_sil_path))
        for sample in sample_folders:
            sil_folder = os.path.join(pid_sil_path, sample)
            skel_folder = os.path.join(pid_skel_path, sample)

            if not (os.path.exists(sil_folder) and os.path.exists(skel_folder)):
                print(f"Missing silhouette or skeleton for {pid}/{sample}, skip")
                continue

            # 경로 설정
            save_sil_folder = os.path.join(output_root, split, "silhouette", pid, sample)
            save_skel_folder = os.path.join(output_root, split, "skeleton", pid, sample)

            # 처리 및 저장
            process_and_save_images(sil_folder, save_sil_folder, is_skeleton=False)
            process_and_save_images(skel_folder, save_skel_folder, is_skeleton=True)

if __name__ == "__main__":
    for split in ["train", "query", "gallery"]:
        process_split_resize(BASE_ROOT, split, RESIZED_ROOT)
