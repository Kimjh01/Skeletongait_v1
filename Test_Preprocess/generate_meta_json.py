import os
import json
import pickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_ROOT = os.path.join(BASE_DIR, "CCVID_PROCESS_PKL")  # 상대경로 기준

OUTPUT_META_PATH = os.path.join(PKL_ROOT, "meta.json")
OUTPUT_SPLIT_PATH = os.path.join(PKL_ROOT, "ccvid_split.json")

def collect_all_pkls():
    meta = {}
    splits = ["train", "query", "gallery"]
    all_ids = {"train": set(), "query": set(), "gallery": set()}

    for split in splits:
        split_dir = os.path.join(PKL_ROOT, split)
        if not os.path.exists(split_dir):
            continue

        for sid in tqdm(sorted(os.listdir(split_dir)), desc=f"Processing {split}"):
            sid_path = os.path.join(split_dir, sid)
            if not os.path.isdir(sid_path):
                continue

            all_ids[split].add(sid)
            if sid not in meta:
                meta[sid] = {}

            for session in sorted(os.listdir(sid_path)):
                session_path = os.path.join(sid_path, session)
                pkl_file = os.path.join(session_path, "000.pkl")
                if not os.path.isfile(pkl_file):
                    continue

                try:
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)
                        sil_shape = data["silhouette"].shape  # (T, H, W)
                        skel_shape = data["skeleton"].shape   # (T, 2, H, W)
                        num_frames = sil_shape[0]
                except Exception as e:
                    print(f"Failed to load {pkl_file}: {e}")
                    continue

                rel_path = os.path.relpath(pkl_file, PKL_ROOT).replace("\\", "/")

                meta[sid][session] = {
                    "pkl_path": rel_path,
                    "num_frames": num_frames,
                    "silhouette_shape": list(sil_shape),
                    "skeleton_shape": list(skel_shape)
                }

    return meta, all_ids

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {path}")

def main():
    meta, split_ids = collect_all_pkls()
    save_json(meta, OUTPUT_META_PATH)

    split_json = {
        "TRAIN_SET": sorted(list(split_ids["train"])),
        "TEST_SET": sorted(list(split_ids["gallery"])),
        "QUERY_SET": sorted(list(split_ids["query"])),
        "GALLERY_SET": sorted(list(split_ids["gallery"]))
    }

    save_json(split_json, OUTPUT_SPLIT_PATH)

if __name__ == "__main__":
    main()
