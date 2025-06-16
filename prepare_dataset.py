# prepare_dataset.py
import os
import cv2

input_folder = "data/original"
high_res_folder = "data/high_res"
low_res_folder = "data/low_res"

os.makedirs(high_res_folder, exist_ok=True)
os.makedirs(low_res_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # high_res로 그대로 저장
    cv2.imwrite(os.path.join(high_res_folder, filename), img)

    # low_res: 작은 크기로 줄였다가 다시 원래 크기로 리사이즈 (인위적으로 blur 발생)
    h, w = img.shape[:2]
    downscaled = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(low_res_folder, filename), upscaled)
    print(f"🔄 변환 완료: {filename}")

print("✅ 데이터셋 준비 완료!")