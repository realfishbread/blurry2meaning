import torch
import os
import cv2
from models.fsrcnn import FSRCNN
from utils.image_utils import image_to_tensor, tensor_to_image

# 경로 설정
model_path = "results/fsrcnn_model.pth"
input_folder = "data/low_res"
output_folder = "results/predicted"
os.makedirs(output_folder, exist_ok=True)

#  장치 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  FSRCNN 모델 로드
model = FSRCNN(scale_factor=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 이미지 복원 실행
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
        continue

    # 이미지 읽기
    img_path = os.path.join(input_folder, filename)
    # 이미지 로드 및 RGB로 변환
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 전처리
    # Tensor 변환 및 모델 입력
    input_tensor = image_to_tensor(img).unsqueeze(0).to(device)

    # 복원
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 결과 저장
    # Tensor → 이미지로 변환
    # img: OpenCV로 불러온 RGB 이미지 (NumPy 배열)

#image_to_tensor(img): 이미지를 PyTorch 모델이 이해할 수 있는 Tensor로 변환함 ([C, H, W] 형식)

#.unsqueeze(0): 차원 추가 → [C, H, W] → [1, C, H, W]
#→ 왜? 모델은 "배치(batch)" 형태로 입력을 받으니까!

# .to(device): GPU나 CPU로 이동시킴
    output_image = tensor_to_image(output_tensor.squeeze(0))
    output_path = os.path.join(output_folder, filename)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_image)
    print(f"✅ 복원 완료: {filename} → {output_path}")

    # "그래디언트"는 모델이 틀린 이유를 알기 위한 '기울기(방향)'야.
# 모델이 예측을 잘못했을 때, 어디를 얼마나 고쳐야 하는지 알려주는 값이지.