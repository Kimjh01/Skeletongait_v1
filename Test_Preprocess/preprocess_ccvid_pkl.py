import os
import pickle
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
BASE_ROOT = os.path.join(BASE_DIR, "CCVID_PROCESS")  
OUTPUT_ROOT = os.path.join(BASE_DIR, "datasets", "CCVID_PROCESS_PKL")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

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

def load_silhouette_sequence(folder, target_size=TARGET_SIZE):
    frame_paths = sorted(glob(os.path.join(folder, "*.png")))
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("L")
        img_fixed = resize_with_aspect_ratio_and_padding(img, target_size)
        frames.append(np.array(img_fixed))
    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)

def load_skeleton_sequence(folder, target_size=TARGET_SIZE):
    frame_paths = sorted(glob(os.path.join(folder, "*.png")))
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img_fixed = resize_with_aspect_ratio_and_padding(img, target_size)
        img_arr = np.array(img_fixed)
        joint = img_arr[:, :, 0].astype(np.float32) / 255.0
        limb = img_arr[:, :, 1].astype(np.float32) / 255.0
        frame = np.stack([joint, limb], axis=0)
        frames.append(frame)
    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)

def process_split_custom(root, split, output_root):
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

            sil_seq = load_silhouette_sequence(sil_folder)
            skel_seq = load_skeleton_sequence(skel_folder)

            if sil_seq is None or skel_seq is None:
                print(f"No frames in {pid}/{sample}, skip")
                continue

            data = {
                "silhouette": sil_seq,
                "skeleton": skel_seq
            }

            save_dir = os.path.join(output_root, split, pid, sample)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "000.pkl")

            with open(save_path, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved {save_path}")

if __name__ == "__main__":
    for split in ["train", "query", "gallery"]:
        process_split_custom(BASE_ROOT, split, OUTPUT_ROOT)
