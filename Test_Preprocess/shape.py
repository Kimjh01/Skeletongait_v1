import cv2

# 이미지 불러오기 (컬러로)
img = cv2.imread("00000.png")  # 파일 경로를 바꿔주세요
print("Shape:", img.shape)       # (높이, 너비, 채널 수)
