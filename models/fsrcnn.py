import torch
import torch.nn as nn

# 모델 정의....학습용....
class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(FSRCNN, self).__init__()

        # 특징 추출 단계 (Feature Extraction)
        # 3채널 이미지, 56채널 특징 맵 , 5x5 필터로 넓게 이미지 특징을 뽑아냄
        self.feature_extraction = nn.Conv2d(3, 56, kernel_size=5, padding=2)

        # 채널 축소 단계 (Shrinking)
        # 56채널 → 12채널로 줄여서 연산량 줄이기, 1x1 필터는 공간 정보는 그대로 두고 채널만 압축함
        self.shrinking = nn.Conv2d(56, 12, kernel_size=1)

        #  매핑 단계 (Mapping)
        # 중간 특징을 계속 변환하면서 의미 있는 패턴 학습
        # 3번 반복된 3x3 conv .. 깊은 표현을 학습
        self.mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1)
        )

        # conv 왜 겹쳐쓰냐면... 한 번 Conv만 쓰면 이미지 한 조각만 보는 거라...
        # 1번 conv: 여기 선이있네요 2번 conv: 여기 선이 전체적인 윤곽인듯 어쩌구...
        #이러면서 고-급 이 됨...
        # 왜 채널이 12개 고정이냐면... 그냥 정보 해석 과정이라 굳이 늘릴필요가 없음....

        #ReVU 끼면 좀 더 복잡한 비선형 계사ㅓㄴㅇ이 가능해서 끼워넣음

        # 채널 복원 단계 (Expanding)
        # 다시 12채널-> 56채널로 확장 (shrinking의 반대)
        self.expanding = nn.Conv2d(12, 56, kernel_size=1)

        # 업샘플링 단계 (Deconvolution)
        # 해상도 업스케일링 수행
        # ConvTranspose2d = 뒤집힌 Convolution -> 이미지 확대하는 효과
        # 9x9 필터로 부드럽게 복원하고, stride=2로 2배 확대
        self.deconv = nn.ConvTranspose2d(56, 3, kernel_size=9, stride=2, padding=4)

    def forward(self, x): # 딥러닝 모델에서 실제로 데이터를 넣었을 때 계산이 진행되는 순서를 정의하는 함수,,,,,,
        # 순전파 정의
        x = torch.relu(self.feature_extraction(x))  # 특징 추출
        # torch.relu()는 음수 다 버리고 양수만 남김 왜냐고...? 왜냐면....중요한것만 남겨버릴거니까,...
        x = torch.relu(self.shrinking(x))           #  채널 축소.. 채널을 확 줄여서 계산 가볍게 함... ReLU로 한 번 더 활성화
        x = self.mapping(x)                          #  매핑... 여러 Conv 레이어로 구성된 부분...ReLU + Conv 여러 번 반복 → 패턴 학습
        x = torch.relu(self.expanding(x))  # 채널 확장... 아까 줄였던 채널 수 다시 늘림 ..왜하냐면.. 정보손실마ㅣㄱ을라고...
        x = self.deconv(x)  # 업샘플링... 해상도 2배로 키우는 단계
        return x

# 단독 실행할 때 구조 출력용
if __name__ == '__main__':
    model = FSRCNN()
    print(model)

    #이ㅏ 하지말걸 왜했지
