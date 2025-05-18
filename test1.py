import os
import pickle

def check_pkl_files(root_path, max_files=5):
    pkl_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.endswith('.pkl'):
                pkl_files.append(os.path.join(dirpath, f))
                if len(pkl_files) >= max_files:
                    break
        if len(pkl_files) >= max_files:
            break

    print(f"Found {len(pkl_files)} .pkl files. Showing content of first {len(pkl_files)} files:")
    for i, pkl_path in enumerate(pkl_files):
        print(f"\n[{i+1}] {pkl_path}:")
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            print(type(data))
            if isinstance(data, dict):
                print("Keys:", list(data.keys()))
            elif isinstance(data, list):
                print(f"List length: {len(data)}")
            else:
                print(data)
        except Exception as e:
            print("Failed to load pickle:", e)

if __name__ == "__main__":
    dataset_root = 'datasets/CCVID_PROCESS_PKL'  # 여기에 실제 경로 넣기
    check_pkl_files(dataset_root)
