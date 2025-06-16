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

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNN(scale_factor=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 이미지 복원 실행
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 이미지 읽기
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 전처리
    input_tensor = image_to_tensor(img).unsqueeze(0).to(device)

    # 복원
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 결과 저장
    output_image = tensor_to_image(output_tensor.squeeze(0))
    output_path = os.path.join(output_folder, filename)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_image)
    print(f"✅ 복원 완료: {filename} → {output_path}")
