import cv2
import numpy as np
import mediapipe as mp

def get_silhouette(image_path):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    image = cv2.imread(image_path)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask > 0.1
        silhouette = (mask[..., None] * 255).astype(np.uint8)
        return silhouette

def get_bbox_from_silhouette(silhouette):
    ys, xs = np.where(silhouette[...,0] > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max, y_max

def extract_keypoints_from_image(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = np.zeros((17, 3), dtype=np.float32)
    mediapipe_to_coco = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32, 29, 30]
    if results.pose_landmarks:
        for coco_i, mp_i in enumerate(mediapipe_to_coco):
            lm = results.pose_landmarks.landmark[mp_i]
            keypoints[coco_i, 0] = lm.x * image.shape[1]
            keypoints[coco_i, 1] = lm.y * image.shape[0]
            keypoints[coco_i, 2] = lm.visibility
    return keypoints

def center_and_scale_keypoints(keypoints, bbox, target_shape):
    x_min, y_min, x_max, y_max = bbox
    person_w = x_max - x_min
    person_h = y_max - y_min
    keypoints_xy = keypoints[:, :2] - [x_min, y_min]
    scale = min(target_shape[1] / person_w, target_shape[0] / person_h)
    keypoints_xy = keypoints_xy * scale
    offset_x = (target_shape[1] - person_w * scale) / 2
    offset_y = (target_shape[0] - person_h * scale) / 2
    keypoints_xy = keypoints_xy + [offset_x, offset_y]
    keypoints_scaled = np.concatenate([keypoints_xy, keypoints[:, 2:]], axis=1)
    return keypoints_scaled

def draw_gaussian(arr, center, sigma):
    x, y = int(center[0]), int(center[1])
    tmp_size = int(3 * sigma)
    ul = [max(0, x - tmp_size), max(0, y - tmp_size)]
    br = [min(arr.shape[1], x + tmp_size + 1), min(arr.shape[0], y + tmp_size + 1)]
    for i in range(ul[1], br[1]):
        for j in range(ul[0], br[0]):
            d2 = (j - x) ** 2 + (i - y) ** 2
            arr[i, j] = max(arr[i, j], np.exp(-d2 / (2 * sigma ** 2)))

def draw_limb(arr, pt1, pt2, sigma):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0: return
    for i in range(length + 1):
        x = int(x1 + (x2 - x1) * i / length)
        y = int(y1 + (y2 - y1) * i / length)
        draw_gaussian(arr, (x, y), sigma)

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
    (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
    (6, 12), (12, 14), (14, 16), (11, 12)
]

def generate_joint_heatmap(keypoints, img_shape=(256,192), sigma=4):
    arr = np.zeros(img_shape, dtype=np.float32)
    for x, y, conf in keypoints:
        if conf < 0.1: continue
        draw_gaussian(arr, (x, y), sigma)
    arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
    return arr

def generate_limb_heatmap(keypoints, skeleton=COCO_SKELETON, img_shape=(256,192), sigma=4):
    arr = np.zeros(img_shape, dtype=np.float32)
    for idx1, idx2 in skeleton:
        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]
        if c1 < 0.1 or c2 < 0.1: continue
        draw_limb(arr, (x1, y1), (x2, y2), sigma)
    arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
    return arr

def generate_skeleton_heatmap(joint_map, limb_map):
    arr = np.clip(joint_map.astype(np.float32) + limb_map.astype(np.float32), 0, 255).astype(np.uint8)
    return arr

if __name__ == "__main__":
    image_path = 'image.png'
    silhouette = get_silhouette(image_path)
    bbox = get_bbox_from_silhouette(silhouette)
    keypoints = extract_keypoints_from_image(image_path)
    target_shape = (256, 192)
    keypoints_scaled = center_and_scale_keypoints(keypoints, bbox, target_shape)
    # joint, limb, skeleton heatmap 모두 생성
    joint_heatmap = generate_joint_heatmap(keypoints_scaled, img_shape=target_shape, sigma=4)
    limb_heatmap = generate_limb_heatmap(keypoints_scaled, img_shape=target_shape, sigma=4)
    skeleton_heatmap = generate_skeleton_heatmap(joint_heatmap, limb_heatmap)
    # 저장
    cv2.imwrite('silhouette.png', silhouette)
    cv2.imwrite('joint_heatmap.png', joint_heatmap)
    cv2.imwrite('limb_heatmap.png', limb_heatmap)
    cv2.imwrite('skeleton_heatmap.png', skeleton_heatmap)
    np.savetxt('keypoints_scaled.txt', keypoints_scaled, fmt='%.4f')
