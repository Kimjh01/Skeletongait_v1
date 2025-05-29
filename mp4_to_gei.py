import os
import cv2
import numpy as np
from ultralytics import YOLO

def extract_frames(video_path, output_folder, frames_per_sec=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames_per_sec > fps:
        print(f"입력한 초당 프레임 수({frames_per_sec})가 영상 fps({fps})보다 높아 실제론 {fps}로 저장됩니다.")
        frames_per_sec = fps
    frame_interval = int(fps / frames_per_sec)
    if frame_interval < 1:
        frame_interval = 1
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_path = os.path.join(output_folder, f"{saved:04d}.png")
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 회전 추가
            cv2.imwrite(out_path, rotated)
            saved += 1
        count += 1
    cap.release()
    print(f"[1] 총 {saved}개의 프레임을 저장했습니다. ({output_folder})")


def yolo_silhouette(input_folder, output_folder, model_path='yolov8m-seg.pt'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = YOLO(model_path)
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    for fname in files:
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path)
        results = model(img)
        masks = results[0].masks
        if masks is not None:
            person_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for i, cls in enumerate(results[0].boxes.cls):
                if int(cls) == 0:  # person 클래스
                    mask = masks.data[i].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    person_mask = cv2.bitwise_or(person_mask, mask)
            if len(person_mask.shape) == 3:
                person_mask = cv2.cvtColor(person_mask, cv2.COLOR_BGR2GRAY)
            _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, person_mask)
        else:
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, np.zeros(img.shape[:2], dtype=np.uint8))
    print(f"[2] 실루엣 생성 완료 ({output_folder})")

def make_gei_from_folder(folder_path, output_path='gei.png'):
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if len(png_files) == 0:
        print("폴더에 PNG 파일이 없습니다.")
        return
    first_img = cv2.imread(os.path.join(folder_path, png_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print("이미지 로드 실패")
        return
    h, w = first_img.shape[:2]
    gei_accum = np.zeros((h, w), dtype=np.float32)
    for fname in png_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        gei_accum += img_bin
    gei = gei_accum / len(png_files)
    gei_img = (gei * 255).astype(np.uint8)
    cv2.imwrite(output_path, gei_img)
    print(f"[3] GEI 저장 완료: {output_path}")


if __name__ == '__main__':
    # 1. 프레임 추출
    try:
        fps_input = float(input("1초에 몇 장의 프레임을 추출할까요? (예: 2) : "))
        if fps_input < 1:
            print("최소 1 이상 입력하세요. 기본값 1로 진행합니다.")
            fps_input = 1
    except Exception:
        print("잘못 입력하셨습니다. 기본값 1로 진행합니다.")
        fps_input = 1

    extract_frames('1.mp4', 'frames_original', frames_per_sec=fps_input)

    # 2. y 입력 시 실루엣 생성
    yn = input("실루엣 생성하려면 y를 입력하세요: ")
    if yn.strip().lower() == 'y':
        yolo_silhouette('frames_original', 'silhouettes', model_path='yolov8m-seg.pt')

    # 3. y 입력 시 GEI 생성
    yn = input("GEI 생성하려면 y를 입력하세요: ")
    if yn.strip().lower() == 'y':
        make_gei_from_folder('silhouettes', 'gei_from_yolo.png')

