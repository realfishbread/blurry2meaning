import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Conv2d(3, 56, kernel_size=5, padding=2) # 5x5 필터로 초기 이미지 특징 추출
        self.shrinking = nn.Conv2d(56, 12, kernel_size=1) # 1x1 필터로 채널 수 줄여 연산량 감소
        self.mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1), # 	중간 정보를 학습하는 핵심 부분
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1)
        )
        self.expanding = nn.Conv2d(12, 56, kernel_size=1)# 다시 채널 수 늘려줌
        self.deconv = nn.ConvTranspose2d(56, 3, kernel_size=9, stride=2, padding=4) # 업샘플링 (해상도 복원) 수행

    def forward(self, x):
        x = torch.relu(self.feature_extraction(x))
        x = torch.relu(self.shrinking(x))
        x = self.mapping(x)
        x = torch.relu(self.expanding(x))
        x = self.deconv(x)
        return x

if __name__ == '__main__':
    model = FSRCNN()
    print(model) # 테스트 실행용 코드