import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_gei_similarity(gei_path1, gei_path2):
    # 이미지 불러오기 (흑백)
    img1 = cv2.imread(gei_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(gei_path2, cv2.IMREAD_GRAYSCALE)

    # 크기가 다르면 resize (옵션)
    if img1.shape != img2.shape:
        print("이미지 크기가 달라서 resize합니다.")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 1차원 벡터로 변환
    vec1 = img1.flatten().reshape(1, -1)
    vec2 = img2.flatten().reshape(1, -1)

    # 코사인 유사도 계산 (1에 가까울수록 유사)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    print(f"코사인 유사도: {similarity:.4f}")
    return similarity

def make_gei_from_folder(folder_path, output_path='gei.png'):
    # 폴더 내 PNG 파일만 리스트로 불러오기 (정렬)
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # 파일이 하나도 없으면 종료
    if len(png_files) == 0:
        print("폴더에 PNG 파일이 없습니다.")
        return

    # 첫 이미지로 크기 결정
    first_img = cv2.imread(os.path.join(folder_path, png_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first_img.shape

    # 실루엣 누적용 배열
    gei_accum = np.zeros((h, w), dtype=np.float32)

    # 각 이미지 읽어서 누적
    for fname in png_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        # 이미지가 255(흰색) 실루엣이면 1로, 나머지는 0으로 이진화
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        gei_accum += img_bin

    # 평균 내기
    gei = gei_accum / len(png_files)
    # 0~255로 변환 (시각화용)
    gei_img = (gei * 255).astype(np.uint8)

    # 저장
    cv2.imwrite(output_path, gei_img)
    print(f"GEI 저장 완료: {output_path}")

# 사용 예시
make_gei_from_folder(r'F:\Grew_raw\train\00001\4XPn5Z28', 'gei.png')

# 사용 예시
compare_gei_similarity('gei.png', 'test_gei.png')
