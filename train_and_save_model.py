# train_fsrcnn_and_save.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

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
            56, 3, kernel_size=8, stride=2, padding=3, output_padding=0
        )  # 지금 네 모델에 맞는 설정

    def forward(self, x):
        x = torch.relu(self.feature_extraction(x))
        x = torch.relu(self.shrinking(x))
        x = self.mapping(x)
        x = torch.relu(self.expanding(x))
        x = self.deconv(x)
        return x

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, _ in train_loader:
        images = images.to(device)
        low_res = transforms.Resize(16)(images)
        degraded = transforms.Resize(32)(low_res).to(device)

        output = model(degraded)
        output = torch.nn.functional.interpolate(output, size=(32, 32), mode="bilinear")  # 크기 강제 정렬
        loss = criterion(output, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch+1}/3] Loss: {loss.item():.4f}")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/fsrcnn_model.pth")
print("✅ FSRCNN 모델 저장 완료!")
