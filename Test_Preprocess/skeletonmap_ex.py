import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Gaussian 생성 함수
def generate_gaussian_map(center, shape, sigma):
    x = np.arange(0, shape[1], 1, np.float32)
    y = np.arange(0, shape[0], 1, np.float32)
    xx, yy = np.meshgrid(x, y)
    d2 = (xx - center[0])**2 + (yy - center[1])**2
    return np.exp(-d2 / (2.0 * sigma**2))

# Limb 생성 함수
def generate_limb_map(p1, p2, shape, sigma):
    canvas = np.zeros(shape, dtype=np.float32)
    if p1[2] < 0.1 or p2[2] < 0.1:
        return canvas
    length = np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))
    num_points = max(int(length), 1)
    xs = np.linspace(p1[0], p2[0], num_points)
    ys = np.linspace(p1[1], p2[1], num_points)
    for x, y in zip(xs, ys):
        g = generate_gaussian_map((x, y), shape, sigma)
        canvas = np.maximum(canvas, g)
    return canvas

# Joint, Limb, Skeleton map 생성
def generate_maps(joints, canvas_size=(256, 256), sigma=8.0):
    joint_map = np.zeros(canvas_size, dtype=np.float32)
    for joint in joints:
        if joint[2] > 0.1:
            g = generate_gaussian_map((joint[0], joint[1]), canvas_size, sigma)
            joint_map = np.maximum(joint_map, g)
    
    limb_map = np.zeros(canvas_size, dtype=np.float32)
    for pair in LIMB_PAIRS:
        p1, p2 = joints[pair[0]], joints[pair[1]]
        g = generate_limb_map(p1, p2, canvas_size, sigma)
        limb_map = np.maximum(limb_map, g)

    skeleton_map = np.clip(joint_map + limb_map, 0, 1)
    return joint_map, limb_map, skeleton_map

# Mediapipe Pose 세팅
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True)

# Limb 연결 쌍 (Mediapipe 기준)
LIMB_PAIRS = [
    (11, 13), (13, 15),  # 왼팔
    (12, 14), (14, 16),  # 오른팔
    (11, 12),            # 어깨
    (23, 25), (25, 27),  # 왼다리
    (24, 26), (26, 28),  # 오른다리
    (23, 24),            # 골반
    (11, 23), (12, 24)   # 상체 연결
]

# 이미지에서 Joint 좌표 추출
def extract_joints_from_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = POSE.process(img_rgb)

    h, w = image.shape[:2]
    joints = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            c = lm.visibility
            joints.append((x, y, c))
    return joints

# 메인 실행 함수
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다:", image_path)
        return
    
    joints = extract_joints_from_image(image)

    canvas_size = (256, 256)
    h, w = image.shape[:2]
    resized_joints = [(x * canvas_size[1] / w, y * canvas_size[0] / h, c) for x, y, c in joints]

    sigma = 8.0
    jm, lm, sm = generate_maps(resized_joints, canvas_size, sigma)

        # 0~1 float → 0~255 uint8로 변환
    jm_img = (jm * 255).astype(np.uint8)
    lm_img = (lm * 255).astype(np.uint8)
    sm_img = (sm * 255).astype(np.uint8)

    # ---- 컬러 변환 ----
    # joint: 흰색
    jm_rgb = cv2.merge([jm_img, jm_img, jm_img])
    # limb: 보라색(R+B)
    lm_rgb = np.zeros((256,256,3), dtype=np.uint8)
    lm_rgb[:,:,0] = lm_img // 2
    lm_rgb[:,:,2] = lm_img
    # skeleton: 합성
    sk_rgb = np.clip(jm_rgb.astype(np.int16) + lm_rgb.astype(np.int16), 0, 255).astype(np.uint8)

    # ---- 저장 ----
    cv2.imwrite("joint_map_rgb.png", jm_rgb)
    cv2.imwrite("limb_map_rgb.png", lm_rgb)
    cv2.imwrite("skeleton_map_rgb.png", sk_rgb)


if __name__ == "__main__":
    process_image("image.png")
