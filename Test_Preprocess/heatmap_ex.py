import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# 1. mediapipe로 keypoint 추출
def extract_mediapipe_keypoints(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    keypoints = np.zeros((33, 3))  # x, y, score
    if results.pose_landmarks:
        h, w, _ = image.shape
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[i, 0] = lm.x * w
            keypoints[i, 1] = lm.y * h
            keypoints[i, 2] = lm.visibility
    return keypoints

# 2. mediapipe → COCO 17개 포맷 매핑 함수
def mediapipe_to_coco17(mediapipe_kps):
    mp_to_coco = [0, 15, 16, 17, 18, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30]
    coco_kps = mediapipe_kps[mp_to_coco]
    return coco_kps

# 3. joint heatmap 생성
def generate_joint_heatmap(joints, heatmap_size, sigma, orig_shape):
    num_joints = joints.shape[0]
    h, w = heatmap_size
    orig_h, orig_w = orig_shape
    heatmaps = np.zeros((num_joints, h, w), dtype=np.float32)
    for i in range(num_joints):
        x, y, v = joints[i]
        if v < 0.1:
            continue
        hm_x = x / orig_w * w
        hm_y = y / orig_h * h
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmap = np.exp(-((xx - hm_x) ** 2 + (yy - hm_y) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = np.clip(heatmap, 0, 1)
    return heatmaps

# 4. limb heatmap 생성 (관절 연결)
# COCO 17관절 기준 limb 정의
COCO_LIMBS = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (5, 6),            # shoulders
    (11, 12),          # hips
    (5, 11), (6, 12),  # torso sides
    (0, 5), (0, 6)     # head to shoulders
]

def generate_limb_heatmap(joints, heatmap_size, sigma, orig_shape):
    h, w = heatmap_size
    orig_h, orig_w = orig_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    for (i, j) in COCO_LIMBS:
        x1, y1, v1 = joints[i]
        x2, y2, v2 = joints[j]
        if v1 < 0.1 or v2 < 0.1:
            continue
        x1 = x1 / orig_w * w
        y1 = y1 / orig_h * h
        x2 = x2 / orig_w * w
        y2 = y2 / orig_h * h
        # 선분을 따라가며 가우시안 그리기
        num_points = int(np.linalg.norm([x2-x1, y2-y1])) * 2
        for t in np.linspace(0, 1, num_points):
            x = x1 * (1-t) + x2 * t
            y = y1 * (1-t) + y2 * t
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            limb_map = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            heatmap = np.maximum(heatmap, limb_map)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

# 5. 합성 skeleton heatmap
def combine_heatmaps(joint_heatmap, limb_heatmap):
    # joint는 채널별 max, limb는 단일채널
    joint_max = np.max(joint_heatmap, axis=0)
    skeleton = np.clip(joint_max + limb_heatmap, 0, 1)
    return skeleton

# 6. 저장 함수
def save_heatmap_img(heatmap, filename):
    # 0~1 float → 0~255 uint8 → colormap
    img = (heatmap * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(filename, img_color)

# ---- 실행 ----
image_path = 'image.png'  # 분석할 원본 이미지 경로

# 1. mediapipe keypoint 추출
mp_keypoints = extract_mediapipe_keypoints(image_path)

# 2. COCO 17개 포맷 변환
coco_kps = mediapipe_to_coco17(mp_keypoints)

# 3. heatmap 해상도 및 원본 크기
heatmap_size = (64, 48)  # (height, width)
orig_h, orig_w = cv2.imread(image_path).shape[:2]

# 4. joint heatmap 생성 및 저장
joint_hm_4 = generate_joint_heatmap(coco_kps, heatmap_size, sigma=4, orig_shape=(orig_h, orig_w))
joint_hm_8 = generate_joint_heatmap(coco_kps, heatmap_size, sigma=8, orig_shape=(orig_h, orig_w))
save_heatmap_img(np.max(joint_hm_4, axis=0), 'joint_sigma4.png')
save_heatmap_img(np.max(joint_hm_8, axis=0), 'joint_sigma8.png')

# 5. limb heatmap 생성 및 저장
limb_hm_4 = generate_limb_heatmap(coco_kps, heatmap_size, sigma=4, orig_shape=(orig_h, orig_w))
limb_hm_8 = generate_limb_heatmap(coco_kps, heatmap_size, sigma=8, orig_shape=(orig_h, orig_w))
save_heatmap_img(limb_hm_4, 'limb_sigma4.png')
save_heatmap_img(limb_hm_8, 'limb_sigma8.png')

# 6. skeleton heatmap(sigma=4) 생성 및 저장 (joint+limb)
skeleton_hm_4 = combine_heatmaps(joint_hm_4, limb_hm_4)
save_heatmap_img(skeleton_hm_4, 'skeleton_sigma4.png')

print("joint_sigma4.png, joint_sigma8.png, limb_sigma4.png, limb_sigma8.png, skeleton_sigma4.png 저장 완료!")
