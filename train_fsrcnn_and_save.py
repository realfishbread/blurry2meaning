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
            56, 3, kernel_size=9, stride=2, padding=4
        )  

    def forward(self, x):
        x = torch.relu(self.feature_extraction(x))
        x = torch.relu(self.shrinking(x))
        x = self.mapping(x)
        x = torch.relu(self.expanding(x))
        x = self.deconv(x)
        return x
    
    #fsrcnn.py 참조...

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3): #ㅇㅔ폭 수 늘리면 과적합날듯
    for images, _ in train_loader:
        images = images.to(device)
        low_res = transforms.Resize(16)(images) # 다운그레이드...
        degraded = transforms.Resize(32)(low_res).to(device) #그걸 또 업샘플링했는데 실제 정보는 손실되었을 거임,,그냥 확대한 정도도

        output = model(degraded) # 뿌연 손상된 이미지 넣으면..모델이 그걸 복원해줌
        output = torch.nn.functional.interpolate(output, size=(32, 32), mode="bilinear")  # 크기 강제 정렬... 모델이 출력하는게 규격이 지 멋대로일지도 몰라서..
        loss = criterion(output, images) # 이미지를 비교....

        optimizer.zero_grad() # 이전 step의 gradient를 초기화해줘야 중복 누적 안댐댐
        loss.backward() # loss를 줄이기 위해 각 파라미터가 얼마나 바뀌어야 하는지 계산.. 그래디언트 계싼...
        optimizer.step() # 계산된 gradient를 바탕으로 모델 파라미터를 조금씩 업데이트 ...기울기 업뎃...

    print(f"[Epoch {epoch+1}/3] Loss: {loss.item():.4f}")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/fsrcnn_model.pth")
print("✅ FSRCNN 모델 저장 완료!")  