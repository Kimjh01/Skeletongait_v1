import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp

# 모델 및 Mediapipe 초기화
model = YOLO("yolov8m-seg.pt")
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True)

# 사용자 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "CCVID")
OUTPUT_ROOT = os.path.join(BASE_DIR, "CCVID_PROCESS")
TXT_PATHS = {
    "train": os.path.join(DATASET_ROOT, "train.txt"),
    "query": os.path.join(DATASET_ROOT, "query.txt"),
    "gallery": os.path.join(DATASET_ROOT, "gallery.txt"),
}
FRAMES_NUM_FIXED = 6
CANVAS_SIZE = (256, 256)
SIGMA = 4.0

# LIMB 정의
LIMB_PAIRS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 25), (25, 27), (24, 26),
    (26, 28), (23, 24), (11, 23), (12, 24)
]

def generate_gaussian_map(center, shape, sigma):
    x = np.arange(0, shape[1], 1, np.float32)
    y = np.arange(0, shape[0], 1, np.float32)
    xx, yy = np.meshgrid(x, y)
    d2 = (xx - center[0])**2 + (yy - center[1])**2
    return np.exp(-d2 / (2.0 * sigma**2))

def generate_limb_map(p1, p2, shape, sigma):
    canvas = np.zeros(shape, dtype=np.float32)
    if p1[2] < 0.1 or p2[2] < 0.1:
        return canvas
    length = np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))
    num_points = max(1, int(length))
    xs = np.linspace(p1[0], p2[0], num_points)
    ys = np.linspace(p1[1], p2[1], num_points)
    for x, y in zip(xs, ys):
        g = generate_gaussian_map((x, y), shape, sigma)
        canvas = np.maximum(canvas, g)
    return canvas

def generate_maps(joints, shape, sigma):
    joint_map = np.zeros(shape, dtype=np.float32)
    for joint in joints:
        if joint[2] > 0.1:
            g = generate_gaussian_map((joint[0], joint[1]), shape, sigma)
            joint_map = np.maximum(joint_map, g)

    limb_map = np.zeros(shape, dtype=np.float32)
    padded_joints = joints + [(0, 0, 0)] * (33 - len(joints)) if len(joints) < 33 else joints

    for pair in LIMB_PAIRS:
        p1, p2 = padded_joints[pair[0]], padded_joints[pair[1]]
        g = generate_limb_map(p1, p2, shape, sigma)
        limb_map = np.maximum(limb_map, g)

    return joint_map, limb_map

def extract_person_mask(image):
    results = model(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for result in results:
        for box, cls, seg in zip(result.boxes.xyxy, result.boxes.cls, result.masks.xy if result.masks else []):
            if int(cls) == 0 and seg is not None:
                seg_points = np.array([seg], dtype=np.int32)
                cv2.fillPoly(mask, seg_points, 255)
    return mask

def extract_joints(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = POSE.process(img_rgb)
    joints = []
    if result.pose_landmarks:
        for i in range(33):
            lm = result.pose_landmarks.landmark[i]
            x = int(lm.x * CANVAS_SIZE[1])
            y = int(lm.y * CANVAS_SIZE[0])
            c = lm.visibility
            joints.append((x, y, c))
    else:
        joints = [(0, 0, 0)] * 33
    return joints

def sample_frames(frame_paths, n):
    total = len(frame_paths)
    if total <= n:
        return frame_paths
    step = total // n
    return [frame_paths[i] for i in range(0, total, step)][:n]

def save_skeleton_as_png(joint_map, limb_map, save_path):
    h, w = joint_map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (joint_map * 255).astype(np.uint8)  # R 채널: joint map
    img[:, :, 1] = (limb_map * 255).astype(np.uint8)   # G 채널: limb map
    # B 채널은 0 유지
    cv2.imwrite(save_path, img)

def process_image(img_path, sil_path, skel_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"이미지 로드 실패: {img_path}")
        return
    # 실루엣 저장
    mask = extract_person_mask(image)
    cv2.imwrite(sil_path, mask)
    # 스켈레톤 저장 (png)
    joints = extract_joints(image)
    jm, lm = generate_maps(joints, CANVAS_SIZE, SIGMA)
    save_skeleton_as_png(jm, lm, skel_path)

def process_split(split_name, txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()

    # video_path 기준 정렬
    lines = sorted(lines, key=lambda x: x.split()[0])

    for line in tqdm(lines, desc=f"Processing {split_name}"):
        video_path, pid, _ = line.strip().split()
        full_path = os.path.join(DATASET_ROOT, video_path)
        if not os.path.exists(full_path):
            print(f"경로 없음: {full_path}")
            continue
        frame_paths = sorted(glob(os.path.join(full_path, "*.jpg")) + glob(os.path.join(full_path, "*.png")))
        if len(frame_paths) == 0:
            print(f"이미지 없음: {full_path}")
            continue
        frame_paths = sample_frames(frame_paths, FRAMES_NUM_FIXED)

        for i, frame_path in enumerate(frame_paths):
            base = f"{i:05d}"
            sil_out = os.path.join(OUTPUT_ROOT, split_name, "silhouette", pid, video_path.replace("/", "_"))
            skel_out = os.path.join(OUTPUT_ROOT, split_name, "skeleton", pid, video_path.replace("/", "_"))
            os.makedirs(sil_out, exist_ok=True)
            os.makedirs(skel_out, exist_ok=True)
            process_image(frame_path,
                          os.path.join(sil_out, f"{base}.png"),
                          os.path.join(skel_out, f"{base}.png"))

if __name__ == "__main__":
    for split in ["train", "query", "gallery"]:
        process_split(split, TXT_PATHS[split])
