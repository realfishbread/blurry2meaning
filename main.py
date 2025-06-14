import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 결과 저장 폴더
os.makedirs("results", exist_ok=True)

# 1. 데이터 불러오기
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 2. 다운샘플링 + 업샘플링 전처리 함수
def degrade_image(img):
    low_res = transforms.Resize(14)(img)           # 다운샘플링
    upsampled = transforms.Resize(28)(low_res)      # 다시 업샘플
    return upsampled

# 3. 모델 정의 (간단한 CNN)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

# 4. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
for epoch in range(3):  # 빠르게 끝내려면 에폭 3 정도면 충분!
    for images, _ in train_loader:
        images = images.to(device)
        degraded = degrade_image(images).to(device)

        outputs = model(degraded)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")

# 6. 결과 저장 (시각화)
sample, _ = next(iter(train_loader))
sample = sample[:5]
degraded = degrade_image(sample)

model.eval()
with torch.no_grad():
    output = model(degraded.to(device)).cpu()

# 원본 / 손상 / 복원 이미지 비교
for i in range(5):
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    axs[0].imshow(sample[i][0], cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(degraded[i][0], cmap='gray')
    axs[1].set_title("Degraded")
    axs[2].imshow(output[i][0], cmap='gray')
    axs[2].set_title("Restored")
    for ax in axs:
        ax.axis('off')
    plt.savefig(f"results/result_{i}.png")
    plt.close()

print("✅ 결과 이미지가 results 폴더에 저장되었습니다!")
