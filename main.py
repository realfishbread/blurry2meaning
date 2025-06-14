import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, ToPILImage
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PIL import ImageEnhance

# 1. SRCNN 모델 정의
class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Conv2d(3, 56, kernel_size=5, padding=2)
        self.shrinking = nn.Conv2d(56, 12, kernel_size=1)
        self.mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1)
        )
        self.expanding = nn.Conv2d(12, 56, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(
    56, 3,
    kernel_size=8,   # 조금 작게
    stride=2,
    padding=3,
    output_padding=0  # 가능하면 이걸로 줄무늬 없앰
)

    def forward(self, x):
        x = torch.relu(self.feature_extraction(x))
        x = torch.relu(self.shrinking(x))
        x = self.mapping(x)
        x = torch.relu(self.expanding(x))
        x = self.deconv(x)
        return x

# 2. 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNN().to(device)  # ✅ 이게 맞아!
model.load_state_dict(torch.load("results/fsrcnn_model.pth", map_location=device))

model.eval()

# 3. 사용자 이미지 불러오기
image = Image.open("user_input_lowres.jpg").convert("RGB")
image = image.resize((32, 32))  # CIFAR 스타일 크기
lowres_tensor = ToTensor()(image).unsqueeze(0).to(device)


enhancer = ImageEnhance.Brightness(image)
image = enhancer.enhance(1.5)  # 밝기 1.5배

# 4. 복원
with torch.no_grad():
    restored_tensor = model(lowres_tensor).cpu()

# ✅ 먼저 clamp로 픽셀 값을 [0, 1]로 제한해줘야 함
restored_tensor = torch.clamp(restored_tensor, 0, 1)

# ✅ 그 다음에 PIL 이미지로 변환
restored_image = ToPILImage()(restored_tensor.squeeze(0))

# ✅ 저장
restored_image.save("restored_output.jpg")

# 5. CLIP 모델 로딩 (멀티모달 분류기)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 6. 추론 텍스트 후보
labels = [
    "a sports car", "a dark car", "a toy car", "a blurry car", "a ship", "a motorcycle",
    "a truck", "a spaceship"
]

# 7. 추론 실행
inputs = processor(text=labels, images=restored_image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

top_idx = torch.argmax(probs)
confidence = probs[0][top_idx].item()
label = labels[top_idx]

print(f"💡 AI의 추론: '{label}' ({confidence*100:.2f}%)")