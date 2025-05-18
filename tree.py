import os

def print_dataset_structure_minimal(root_path, max_depth=2, indent=0, max_items=2):
    if indent > max_depth:
        return
    try:
        items = sorted(os.listdir(root_path))
    except Exception as e:
        print(" " * indent * 2 + f"[Error accessing {root_path}]: {e}")
        return

    # 최대 2개 항목만 출력
    items_to_show = items[:max_items]
    for item in items_to_show:
        path = os.path.join(root_path, item)
        print(" " * indent * 2 + item + ("/" if os.path.isdir(path) else ""))
        if os.path.isdir(path):
            print_dataset_structure_minimal(path, max_depth, indent + 1, max_items)

    if len(items) > max_items:
        print(" " * (indent * 2) + f"... and {len(items) - max_items} more items")

if __name__ == "__main__":
    dataset_root =  'datasets/CCVID_PROCESS_PKL'
    print(f"Dataset structure under: {dataset_root}")
    print_dataset_structure_minimal(dataset_root)
