import torch # 딥러닝 프레임워크
import os
import cv2 # 이미지 읽기/쓰기 OpenCㅍㅍ
import json
from PIL import Image # 모델 분류 입력용 변환
from torchvision import models, transforms # PyTorch에서 제공하는 ResNet18 모델
from models.fsrcnn import FSRCNN # 그냥 학습된 모델 쓰자 힘들다.
from utils.image_utils import image_to_tensor, tensor_to_image # 헬퍼 함수 np.array <-> tensor 변환용

# 경로 설정
model_path = "results/fsrcnn_model.pth" #학습된 모델
input_folder = "data/low_res" #저해상도 이미지 위치
output_folder = "results/predicted" # 복원된 이미지 저장 위치
os.makedirs(output_folder, exist_ok=True) # 폴더 없으면 만들고요 있으면 뭐...

# FSRCNN 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fsrcnn = FSRCNN(scale_factor=2).to(device) # 해상도 2배로 복원하는 네트워크
#.to(device) -> gpu 사용 가능하면 gpu를 씀. 난 없음
fsrcnn.load_state_dict(torch.load(model_path, map_location=device))
fsrcnn.eval() # 학습이 아닌 추론 모드로 전환: .eval...

# 이미지 분류 모델 로드
classifier = models.resnet18(pretrained=True).to(device) #  ImageNet으로 학습된 사전학습 모델........
classifier.eval()

# 분류용 전처리
classify_transform = transforms.Compose([
    transforms.Resize(256), # 원본 이미지 크기가 일정하질 못함 그래서 리사이즈를 해줄거임
    transforms.CenterCrop(224), # 그 다음 224x224 가운데만 잘라서 사용
    #왜자르냐면... 딥러닝 모델이 판단하기에... 이미지 가운데가 중요할 가능성이 높기 때문
    transforms.ToTensor(), # H x W x C 이미지 → C x H x W 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # 이건 ImageNet 데이터셋 전체의 RGB 평균값과 표준편차
]) # 학습 당시 ResNet이 이 값으로 normalize 된 이미지를 기준으로 weight를 학습했기 때문에, 테스트할 때도 같은 방식으로 normalize 해줘야...

# 분류 결과 이름 불러오기 (예: 208 → golden retriever)
with open("imagenet_classes.json") as f: 
    idx_to_label = json.load(f) # 이미지 파일만 고름

# 이미지 복원이랑 분류....
for filename in os.listdir(input_folder):
    #이미지 파일만 선택적으로 처리
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 이미지 불러오기 및 복원
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path) # BGR로 읽기
    # 왜 BGR로 읽느냐...
    #OpenCV는 C/C++ 기반 라이브러리
    # 내부 최적화를 위해 메모리 구조를 B, G, R 순서로 저장...왜? 메모리 정렬 상 유리한 구조였기 때문
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB 변환
    input_tensor = image_to_tensor(img_rgb).unsqueeze(0).to(device) # np.ndarray → torch.tensor 변환

    # unsqueeze:..배치 차원 추가


    with torch.no_grad():
        restored_tensor = fsrcnn(input_tensor) # 복원 결과는 텐서에서 다시 이미지로 :tensor_to_image, squeeze(0): 배치 차원 제거
    restored_img = tensor_to_image(restored_tensor.squeeze(0))
    restored_path = os.path.join(output_folder, filename)
    cv2.imwrite(restored_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)) # 저장은 BGR로 다시 바꿔서 OpenCV로 저장

    #분류
    pil_img = Image.fromarray(restored_img)
    input_for_classifier = classify_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(input_for_classifier) # 이 output은 1000개의 숫자로 된 벡터
        # softmax([1.0, 2.0, 3.0])  →  [0.09, 0.24, 0.67] 뭐 이런식으로... 비교가능하게.
        # softmax()는: 모델이 이 이미지가 각 클래스일 확률을 구체적인 숫자(0~1)로 알려주는 함수. 그 확률들의 총합은 1
        probabilities = torch.nn.functional.softmax(output[0], dim=0) # 로짓들을 확률 분포로 바꿔줘
        predicted = torch.argmax(probabilities).item() # 가장 확률 높은 클래스 인덱스 .item()으로 숫자화-> 제일 높은 확률의 클래스 번호
        confidence = probabilities[predicted].item() # 그 클래스의 확률값

    label = idx_to_label[str(predicted)]
    print(f"📸 {filename} → 복원 & 분류 결과: {label} ({confidence * 100:.2f}%)")
