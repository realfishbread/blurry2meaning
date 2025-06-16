import torch
import os
import cv2
import json
from PIL import Image
from torchvision import models, transforms
from models.fsrcnn import FSRCNN
from utils.image_utils import image_to_tensor, tensor_to_image

# 경로 설정
model_path = "results/fsrcnn_model.pth"
input_folder = "data/low_res"
output_folder = "results/predicted"
os.makedirs(output_folder, exist_ok=True)

# FSRCNN 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fsrcnn = FSRCNN(scale_factor=2).to(device)
fsrcnn.load_state_dict(torch.load(model_path, map_location=device))
fsrcnn.eval()

# 이미지 분류 모델 로드 (ResNet18)
classifier = models.resnet18(pretrained=True).to(device)
classifier.eval()

# 분류용 전처리
classify_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ImageNet 클래스 이름 로딩
with open("imagenet_classes.json") as f:
    idx_to_label = json.load(f)

# 이미지 복원 + 분류
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 이미지 불러오기 및 복원
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = image_to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_tensor = fsrcnn(input_tensor)
    restored_img = tensor_to_image(restored_tensor.squeeze(0))
    restored_path = os.path.join(output_folder, filename)
    cv2.imwrite(restored_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

    # 분류
    pil_img = Image.fromarray(restored_img)
    input_for_classifier = classify_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(input_for_classifier)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted = torch.argmax(probabilities).item()
        confidence = probabilities[predicted].item()

    label = idx_to_label[str(predicted)]
    print(f"📸 {filename} → 복원 & 분류 결과: {label} ({confidence * 100:.2f}%)")
