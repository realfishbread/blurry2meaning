# prepare_dataset.py
import os
import cv2

input_folder = "data/original"
high_res_folder = "data/high_res"
low_res_folder = "data/low_res"
# high_res / low_res 폴더가 없으면 자동으로 만들어줌

os.makedirs(high_res_folder, exist_ok=True)
os.makedirs(low_res_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    # data/original 폴더 안에 있는 이미지들을 하나씩 처리
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # 원본 이미지를 high_res 폴더에 그대로 저장 → 정답 이미지 역할
    cv2.imwrite(os.path.join(high_res_folder, filename), img)

    # low_res: 작은 크기로 줄였다가 다시 원래 크기로 리사이즈 (인위적으로 blur 발생)
    # 줄였다가 다시 키움 = 흐릿해짐 (blur 생김) 이게 바로 모델에게 얘좀 어케 복원해보라고 좀 하는거...
    h, w = img.shape[:2]
    downscaled = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
    #INTER_CUBIC은... 이미지를 부드럽고 자연스럽게 확대축소좀....해봐.... 뭐 이런것.
    #왜 저해상도를 만드는ㄷㅔ 인터큐빅을쓰냐.. 조잡하고 드럽게 저해상도가 되기 때문에...그렇습니다. 진짜 저화질처럼 만드려는 노력.

    cv2.imwrite(os.path.join(low_res_folder, filename), upscaled)
    print(f"🔄 변환 완료: {filename}")

print("✅ 데이터셋 준비 완료!")

# prepare_dataset.py는 원본 이미지를 반으로 줄였다가 다시 키워서
# low_res 폴더에 인위적으로 흐릿한 이미지를 만듦.